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
        try:
            from markitdown import MarkItDown
            self._markitdown = MarkItDown()
        except ImportError:
            self._markitdown = None

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

    async def extract(self, **kwargs) -> List[Dict[str, Any]]:
        """Standard batch extraction (returns list)"""
        docs = []
        async for doc in self.extract_stream(**kwargs):
            docs.append(doc)
        return docs

    async def extract_stream(
        self,
        space_key: str = None,
        content_id: str = None,
        url: str = None,
        limit: int = 50,
        **kwargs
    ):
        """Extract content from Confluence as a stream."""
        if url and not space_key and not content_id:
            content_id = self._extract_content_id_from_url(url)
            if not content_id:
                space_key = self._extract_space_key_from_url(url)

        try:
            # Test connection
            test_resp = self.session.get(f"{self.base_url}/rest/api/space?limit=1", timeout=10)
            if test_resp.status_code == 401:
                raise ValueError("Confluence authentication failed.")

            if content_id:
                logger.info(f"Fetching Confluence page ID: {content_id}")
                page_url = f"{self.base_url}/rest/api/content/{content_id}?expand=body.storage,version,metadata.labels"
                resp = self.session.get(page_url, timeout=30)
                if resp.status_code == 200:
                    for doc in self._content_to_docs(resp.json()):
                        yield doc
            elif space_key:
                start = 0
                count = 0
                while count < limit:
                    fetch_url = f"{self.base_url}/rest/api/content"
                    params = {
                        'spaceKey': space_key,
                        'limit': min(limit - count, 50),
                        'start': start,
                        'expand': 'body.storage,version,metadata.labels'
                    }
                    resp = self.session.get(fetch_url, params=params, timeout=30)
                    if resp.status_code != 200: break
                    
                    data = resp.json()
                    results = data.get('results', [])
                    if not results: break

                    for item in results:
                        for doc in self._content_to_docs(item):
                            yield doc
                            count += 1
                        if count >= limit: break
                    
                    if data.get('size', 0) + data.get('start', 0) >= data.get('totalSize', 0):
                        break
                    start += data.get('size', 0)
            else:
                logger.info("No space_key, fetching from first available space...")
                spaces_resp = self.session.get(f"{self.base_url}/rest/api/space?limit=1", timeout=30)
                if spaces_resp.status_code == 200:
                    results = spaces_resp.json().get('results', [])
                    if results:
                        async for doc in self.extract_stream(space_key=results[0]['key'], limit=limit):
                            yield doc

        except Exception as e:
            logger.error(f"Confluence extraction failed: {e}")
            if isinstance(e, ValueError): raise
            raise ValueError(f"Confluence extraction failed: {str(e)[:200]}")

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
            if not storage:
                return docs

            # Use MarkItDown if available for better HTML -> MD conversion
            text = ""
            if self._markitdown:
                try:
                    # MarkItDown typically takes a file or URL, but we can wrap HTML
                    import tempfile
                    import os
                    with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
                        tmp.write(storage.encode('utf-8'))
                        tmp_path = tmp.name
                    try:
                        result = self._markitdown.convert(tmp_path)
                        text = result.text_content
                    finally:
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                except Exception as e:
                    logger.debug(f"MarkItDown conversion failed: {e}")
            
            if not text:
                text = BeautifulSoup(storage, 'html.parser').get_text(separator='\n', strip=True)

            if not text.strip():
                return docs

            title = content_obj.get('title') or content_obj.get('metadata', {}).get('title', '')
            cid = content_obj.get('id')
            page_url = f"{self.base_url}/pages/{cid}"
            version = content_obj.get('version', {}).get('number')
            updated = content_obj.get('version', {}).get('friendlyWhen')

            docs.append({
                'content': text,
                'metadata': {
                    'source': self.source_name,
                    'content_id': cid,
                    'title': title,
                    'type': 'confluence_page',
                    'url': page_url,
                    'version': version,
                    'last_updated': updated
                }
            })
        except Exception as e:
            logger.warning(f"Failed to convert Confluence content to doc: {e}")
        return docs
