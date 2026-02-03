from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from .base import BaseDataSource
import logging
import asyncio

logger = logging.getLogger(__name__)

# Common cookie consent popup selectors
POPUP_SELECTORS = [
    '#onetrust-accept-btn-handler',
    '[class*="cookie"] button[class*="accept"]',
    '[class*="consent"] button',
    'button[name="agree"]',  # Yahoo-specific
    '[id*="cookie"] button[class*="accept"]',
    '[id*="consent"] button[class*="accept"]',
    '.cookie-banner button',
    '#accept-cookies',
    '.accept-cookies',
    'button[data-testid="GDPR-accept"]',
    '.gdpr-accept',
    '#gdpr-consent-accept',
]

# Domains that require browser mode for JavaScript rendering
JS_HEAVY_DOMAINS = [
    'finance.yahoo.com',
    'yahoo.com',
    'bloomberg.com',
    'reuters.com',
    'wsj.com',
    'nytimes.com',
    'twitter.com',
    'x.com',
    'linkedin.com',
    'facebook.com',
    'instagram.com',
    'medium.com',
    'substack.com',
    'notion.so',
    'figma.com',
    'miro.com',
]


class WebSource(BaseDataSource):
    """Extract documents from web pages"""

    def __init__(self):
        super().__init__("WebSource")
        self.max_pages = 100  # Increased from 50
        self.min_content_length = 50  # Configurable threshold
        self._browser = None
        self._playwright = None

    async def _init_browser(self):
        """Initialize Playwright browser if not already initialized"""
        if self._browser is None:
            try:
                from playwright.async_api import async_playwright
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=['--disable-blink-features=AutomationControlled']
                )
                logger.info("Playwright browser initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Playwright browser: {e}")
                raise

    async def _cleanup_browser(self):
        """Close browser resources"""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        logger.info("Playwright browser cleaned up")

    async def _dismiss_popups(self, page) -> bool:
        """Attempt to dismiss cookie consent and other popups"""
        dismissed = False
        for selector in POPUP_SELECTORS:
            try:
                button = page.locator(selector).first
                if await button.is_visible(timeout=500):
                    await button.click()
                    logger.debug(f"Dismissed popup with selector: {selector}")
                    dismissed = True
                    await asyncio.sleep(0.5)  # Wait for popup to close
                    break
            except Exception:
                continue
        return dismissed

    async def _scroll_page(self, page, scroll_count: int = 3):
        """Scroll page to trigger lazy-loaded content"""
        for _ in range(scroll_count):
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await asyncio.sleep(0.3)
        # Scroll back to top
        await page.evaluate("window.scrollTo(0, 0)")

    async def _scrape_with_browser(
        self,
        url: str,
        wait_time: int = 3000,
        auto_dismiss_popups: bool = True,
        wait_for_selector: Optional[str] = None
    ) -> Optional[str]:
        """
        Scrape a URL using Playwright browser for JavaScript-heavy sites

        Args:
            url: URL to scrape
            wait_time: Time to wait for JS content to render (ms)
            auto_dismiss_popups: Whether to auto-dismiss cookie consent popups
            wait_for_selector: CSS selector to wait for before extracting content

        Returns:
            HTML content of the page or None if failed
        """
        await self._init_browser()
        page = None

        try:
            page = await self._browser.new_page(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )

            logger.info(f"Browser navigating to: {url}")
            await page.goto(url, wait_until='domcontentloaded', timeout=30000)

            # Wait for initial JS to execute
            await asyncio.sleep(wait_time / 1000)

            # Auto-dismiss popups if enabled
            if auto_dismiss_popups:
                await self._dismiss_popups(page)

            # Wait for specific selector if provided
            if wait_for_selector:
                try:
                    await page.wait_for_selector(wait_for_selector, timeout=10000)
                    logger.debug(f"Found selector: {wait_for_selector}")
                except Exception:
                    logger.warning(f"Selector not found: {wait_for_selector}")

            # Scroll to trigger lazy-loaded content
            await self._scroll_page(page)

            # Additional wait after scrolling
            await asyncio.sleep(1)

            # Get page content
            content = await page.content()
            logger.info(f"Browser extracted {len(content)} chars from: {url}")

            return content

        except Exception as e:
            logger.error(f"Browser scraping error for {url}: {e}")
            return None
        finally:
            if page:
                await page.close()

    async def extract(
        self,
        url: str = None,
        urls: List[str] = None,
        max_depth: int = 0,
        use_browser: bool = False,
        wait_time: int = 3000,
        auto_dismiss_popups: bool = True,
        wait_for_selector: str = None,
        min_content_length: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract content from web pages

        Args:
            url: Single URL to scrape
            urls: List of URLs to scrape
            max_depth: Crawl depth (0 = single page only)
            use_browser: Force Playwright browser mode for JS-heavy sites
            wait_time: Wait time for JS content to render in ms (browser mode only)
            auto_dismiss_popups: Auto-dismiss cookie consent popups (browser mode only)
            wait_for_selector: CSS selector to wait for before extraction (browser mode only)

        Returns:
            List of documents with content and metadata
        """
        if url:
            urls = [url]

        if not urls:
            raise ValueError("Must provide 'url' or 'urls'")

        documents = []
        visited = set()

        # Auto-detect if browser mode is needed based on domain
        effective_use_browser = use_browser
        if not use_browser:
            for check_url in urls:
                try:
                    parsed = urlparse(check_url)
                    domain = parsed.netloc.lower()
                    # Check if domain or any parent domain is in the JS-heavy list
                    for js_domain in JS_HEAVY_DOMAINS:
                        if domain == js_domain or domain.endswith('.' + js_domain):
                            logger.info(f"Auto-enabling browser mode for JS-heavy domain: {domain}")
                            effective_use_browser = True
                            break
                    if effective_use_browser:
                        break
                except Exception:
                    pass

        # Store browser options for use in _scrape_url
        self._use_browser = effective_use_browser
        self._wait_time = wait_time
        self._auto_dismiss_popups = auto_dismiss_popups
        self._wait_for_selector = wait_for_selector
        self.min_content_length = min_content_length

        try:
            for start_url in urls:
                # Validate URL
                parsed = urlparse(start_url)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError(f"Invalid URL format: {start_url}")

                docs = await self._scrape_url(start_url, visited, max_depth, 0)
                documents.extend(docs)

            logger.info(f"Extracted {len(documents)} documents from {len(visited)} web pages")

            if len(documents) == 0:
                logger.warning(f"No content extracted from {urls}. The pages may be empty or blocked.")
        finally:
            # Cleanup browser if it was used
            if self._use_browser:
                await self._cleanup_browser()

        return documents

    async def _scrape_url(
        self,
        url: str,
        visited: set,
        max_depth: int,
        current_depth: int
    ) -> List[Dict[str, Any]]:
        """Recursively scrape URL and follow links"""

        if url in visited or current_depth > max_depth:
            return []

        # Limit total pages to crawl
        if len(visited) >= self.max_pages:
            logger.warning(f"Reached max page limit ({self.max_pages}), stopping crawl")
            return []

        visited.add(url)
        documents = []

        try:
            logger.info(f"Scraping: {url} (depth: {current_depth}, browser: {self._use_browser})")

            # Choose scraping method based on use_browser flag
            if self._use_browser:
                html_content = await self._scrape_with_browser(
                    url,
                    wait_time=self._wait_time,
                    auto_dismiss_popups=self._auto_dismiss_popups,
                    wait_for_selector=self._wait_for_selector
                )
                if not html_content:
                    return []
                soup = BeautifulSoup(html_content, 'html.parser')
            else:
                # Original requests-based approach
                response = requests.get(url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                }, allow_redirects=True)

                # Check for HTTP errors
                if response.status_code == 403:
                    logger.warning(f"Access forbidden (403): {url}")
                    return []
                elif response.status_code == 404:
                    logger.warning(f"Page not found (404): {url}")
                    return []
                elif response.status_code >= 400:
                    logger.warning(f"HTTP error {response.status_code}: {url}")
                    return []

                response.raise_for_status()

                # Check content type
                content_type = response.headers.get('content-type', '')
                if 'text/html' not in content_type and 'text/plain' not in content_type:
                    logger.debug(f"Skipping non-HTML content: {url} ({content_type})")
                    return []

                # Parse HTML
                soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements - Keeping 'header' and 'aside' as they often contain useful context
            for element in soup(["script", "style", "nav", "footer", "noscript", "iframe"]):
                element.decompose()

            # Extract text
            text = soup.get_text(separator='\n', strip=True)

            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)

            # Only add if we have meaningful content
            if text and len(text) > self.min_content_length:
                # Get page title
                title = soup.title.string.strip() if soup.title and soup.title.string else urlparse(url).path or url

                documents.append({
                    'content': text,
                    'metadata': {
                        'source': self.source_name,
                        'url': url,
                        'title': title[:200],  # Limit title length
                        'type': 'web',
                        'depth': current_depth,
                        'content_length': len(text),
                        'browser_mode': self._use_browser
                    }
                })
                logger.info(f"Extracted {len(text)} chars from: {url}")

            # Follow links if depth allows
            if current_depth < max_depth:
                base_domain = urlparse(url).netloc
                links_to_follow = []

                for link in soup.find_all('a', href=True):
                    href = link['href']

                    # Skip anchors, javascript, mailto, etc.
                    if href.startswith('#') or href.startswith('javascript:') or href.startswith('mailto:'):
                        continue

                    next_url = urljoin(url, href)
                    next_parsed = urlparse(next_url)

                    # Follow links on same domain OR subdomains
                    target_netloc = next_parsed.netloc.lower()
                    is_same_domain = target_netloc == base_domain
                    is_subdomain = target_netloc.endswith('.' + base_domain) or base_domain.endswith('.' + target_netloc)
                    
                    if (is_same_domain or is_subdomain) and next_url not in visited:
                        # Skip common non-content URLs
                        skip_patterns = ['/login', '/signup', '/register', '/logout', '/cart', '/checkout']
                        if not any(pattern in next_url.lower() for pattern in skip_patterns):
                            links_to_follow.append(next_url)

                # Deduplicate and limit links
                links_to_follow = list(set(links_to_follow))[:20]

                for next_url in links_to_follow:
                    if len(visited) >= self.max_pages:
                        break
                    child_docs = await self._scrape_url(next_url, visited, max_depth, current_depth + 1)
                    documents.extend(child_docs)

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout scraping {url}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error scraping {url}: {e}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request error scraping {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")

        return documents
