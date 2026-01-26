from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
import re
from .base import BaseDataSource
import logging

logger = logging.getLogger(__name__)


class ConfluenceSource(BaseDataSource):
    """Simple Confluence (Atlassian Cloud) extractor.

    Minimal implementation: requires `base_url` (e.g. https://your-domain.atlassian.net/wiki)
    and `api_token` (API token for a user) plus `email` (username). You can provide a
    `space_key` or `content_id`. This fetches page storage-format content and strips HTML.
    """

    def __init__(self, base_url: str, email: str, api_token: str):
        super().__init__("ConfluenceSource")
        # Parse and normalize the base URL
        self.base_url = self._normalize_base_url(base_url)
        self.auth = (email, api_token)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({
            'Accept': 'application/json'
        })

    def _normalize_base_url(self, url: str) -> str:
        """Extract base wiki URL from various Confluence URL formats."""
        url = url.strip().rstrip('/')

        # If it's a short link like /x/BoAB, extract content ID
        if '/x/' in url:
            # This is a short link, extract the base
            parts = url.split('/x/')
            return parts[0].rstrip('/')

        # If it contains /wiki/spaces/ or /wiki/pages/, extract base
        if '/wiki/spaces/' in url or '/wiki/pages/' in url:
            idx = url.find('/wiki/')
            return url[:idx + 5]  # Include /wiki

        # If it ends with /wiki, use as-is
        if url.endswith('/wiki'):
            return url

        # If it's just the domain, append /wiki
        parsed = urlparse(url)
        if parsed.path in ['', '/']:
            return f"{parsed.scheme}://{parsed.netloc}/wiki"

        return url

    def _extract_content_id_from_url(self, url: str) -> Optional[str]:
        """Extract content ID from various Confluence URL formats."""
        # Short link format: /x/BoAB (base64-ish encoded)
        if '/x/' in url:
            match = re.search(r'/x/([A-Za-z0-9_-]+)', url)
            if match:
                return match.group(1)

        # Standard page URL: /pages/123456 or /pages/viewpage.action?pageId=123456
        if 'pageId=' in url:
            match = re.search(r'pageId=(\d+)', url)
            if match:
                return match.group(1)

        if '/pages/' in url:
            match = re.search(r'/pages/(\d+)', url)
            if match:
                return match.group(1)

        return None

    def _extract_space_key_from_url(self, url: str) -> Optional[str]:
        """Extract space key from Confluence URL."""
        # Format: /wiki/spaces/SPACEKEY/...
        match = re.search(r'/spaces/([A-Za-z0-9_-]+)', url)
        if match:
            return match.group(1)
        return None

    async def extract(self, space_key: str = None, content_id: str = None, url: str = None, limit: int = 50, **kwargs) -> List[Dict[str, Any]]:
        """Extract content from Confluence.

        Args:
            space_key: Confluence space key to fetch all pages from
            content_id: Specific content/page ID to fetch
            url: Full Confluence URL (will extract space_key or content_id automatically)
            limit: Maximum number of pages to fetch
        """
        documents: List[Dict[str, Any]] = []

        # If URL provided, try to extract space_key or content_id from it
        if url and not space_key and not content_id:
            content_id = self._extract_content_id_from_url(url)
            if not content_id:
                space_key = self._extract_space_key_from_url(url)

        try:
            # First, verify authentication
            test_url = f"{self.base_url}/rest/api/space?limit=1"
            logger.info(f"Testing Confluence connection: {test_url}")
            test_resp = self.session.get(test_url, timeout=30)

            if test_resp.status_code == 401:
                raise ValueError("Confluence authentication failed. Check your email and API token.")
            elif test_resp.status_code == 403:
                raise ValueError("Confluence access denied. Your account may not have permission.")
            elif test_resp.status_code >= 400:
                raise ValueError(f"Confluence API error: {test_resp.status_code} - {test_resp.text[:200]}")

            if content_id:
                # Fetch single content by id
                logger.info(f"Fetching Confluence page with ID: {content_id}")
                url = f"{self.base_url}/rest/api/content/{content_id}?expand=body.storage,version,metadata.labels"
                resp = self.session.get(url, timeout=30)

                if resp.status_code == 404:
                    raise ValueError(f"Confluence page not found with ID: {content_id}")
                resp.raise_for_status()

                obj = resp.json()
                docs = self._content_to_docs(obj)
                documents.extend(docs)

            elif space_key:
                # Query space for pages
                logger.info(f"Fetching pages from Confluence space: {space_key}")
                start = 0
                while True:
                    url = f"{self.base_url}/rest/api/content"
                    params = {
                        'spaceKey': space_key,
                        'limit': min(limit, 50),
                        'start': start,
                        'expand': 'body.storage,version,metadata.labels'
                    }
                    resp = self.session.get(url, params=params, timeout=30)

                    if resp.status_code == 404:
                        raise ValueError(f"Confluence space not found: {space_key}")
                    resp.raise_for_status()

                    data = resp.json()
                    results = data.get('results', [])

                    if not results:
                        if start == 0:
                            logger.warning(f"No pages found in Confluence space: {space_key}")
                        break

                    for item in results:
                        documents.extend(self._content_to_docs(item))

                    if data.get('size', 0) + data.get('start', 0) >= data.get('totalSize', 0):
                        break
                    start += data.get('size', 0) or 0
                    if len(documents) >= limit:
                        break
            else:
                # No content_id or space_key - try to list all spaces and get pages
                logger.info("No space_key or content_id provided, fetching all accessible spaces...")
                spaces_resp = self.session.get(f"{self.base_url}/rest/api/space?limit=10", timeout=30)
                spaces_resp.raise_for_status()
                spaces = spaces_resp.json().get('results', [])

                if not spaces:
                    raise ValueError("No Confluence spaces found. Provide a space_key or content_id.")

                # Get pages from first space
                first_space = spaces[0]['key']
                logger.info(f"Using first available space: {first_space}")
                return await self.extract(space_key=first_space, limit=limit)

            logger.info(f"Extracted {len(documents)} documents from Confluence")
            return documents

        except ValueError:
            raise
        except requests.exceptions.Timeout:
            raise ValueError("Confluence request timed out. Try again later.")
        except requests.exceptions.ConnectionError:
            raise ValueError(f"Could not connect to Confluence at {self.base_url}. Check the URL.")
        except Exception as e:
            logger.error(f"Error extracting Confluence content: {e}")
            raise ValueError(f"Confluence extraction failed: {str(e)[:200]}")

    def _content_to_docs(self, content_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        try:
            storage = content_obj.get('body', {}).get('storage', {}).get('value', '')
            # storage is HTML — convert to plain text
            text = BeautifulSoup(storage, 'html.parser').get_text(separator='\n', strip=True)

            if not text.strip():
                return docs  # Skip empty pages

            title = content_obj.get('title') or content_obj.get('metadata', {}).get('title', '')
            cid = content_obj.get('id')
            page_url = f"{self.base_url}/pages/{cid}"

            docs.append({
                'content': text,
                'metadata': {
                    'source': self.source_name,
                    'content_id': cid,
                    'title': title,
                    'type': 'confluence_page',
                    'url': page_url
                }
            })
        except Exception as e:
            logger.warning(f"Failed to convert Confluence content to doc: {e}")
        return docs
