from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
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
        self.base_url = base_url.rstrip('/')
        self.auth = (email, api_token)
        self.session = requests.Session()
        self.session.auth = self.auth
        self.session.headers.update({
            'Accept': 'application/json'
        })

    async def extract(self, space_key: str = None, content_id: str = None, limit: int = 50, **kwargs) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []

        try:
            if content_id:
                # Fetch single content by id and expand body.storage
                url = f"{self.base_url}/rest/api/content/{content_id}?expand=body.storage,version,metadata.labels"
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                obj = resp.json()
                docs = self._content_to_docs(obj)
                documents.extend(docs)
            elif space_key:
                # Query space for pages
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
                    resp.raise_for_status()
                    data = resp.json()
                    for item in data.get('results', []):
                        documents.extend(self._content_to_docs(item))

                    if data.get('size', 0) + data.get('start', 0) >= data.get('totalSize', 0):
                        break
                    start += data.get('size', 0) or 0
                    if len(documents) >= limit:
                        break
            else:
                raise ValueError("Must provide either 'space_key' or 'content_id' for ConfluenceSource")

            logger.info(f"Extracted {len(documents)} documents from Confluence")
            return documents

        except Exception as e:
            logger.error(f"Error extracting Confluence content: {e}")
            raise

    def _content_to_docs(self, content_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        try:
            storage = content_obj.get('body', {}).get('storage', {}).get('value', '')
            # storage is HTML — convert to plain text
            text = BeautifulSoup(storage, 'html.parser').get_text(separator='\n', strip=True)

            title = content_obj.get('title') or content_obj.get('metadata', {}).get('title', '')
            cid = content_obj.get('id')
            url = f"{self.base_url}/pages/{cid}"

            docs.append({
                'content': text,
                'metadata': {
                    'source': self.source_name,
                    'content_id': cid,
                    'title': title,
                    'type': 'confluence_page',
                    'url': url
                }
            })
        except Exception as e:
            logger.warning(f"Failed to convert Confluence content to doc: {e}")
        return docs
