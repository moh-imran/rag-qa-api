from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from .base import BaseDataSource
import logging

logger = logging.getLogger(__name__)


class SharePointSource(BaseDataSource):
    """Minimal SharePoint extractor using Microsoft Graph access token.

    This implementation is intentionally simple: it requires an OAuth2 access token
    with permissions to read drive items on the target site. It lists files under
    the site's drive root and downloads text-like files (.txt/.md/.html) to extract.
    For production usage, implement paging, binary parsing for .docx/.pdf, and proper token refresh.
    """

    GRAPH_BASE = "https://graph.microsoft.com/v1.0"

    def __init__(self, access_token: str, site_id: Optional[str] = None):
        super().__init__("SharePointSource")
        self.access_token = access_token
        self.site_id = site_id
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        })

    async def extract(self, **kwargs) -> List[Dict[str, Any]]:
        """Standard batch extraction (returns list)"""
        docs = []
        async for doc in self.extract_stream(**kwargs):
            docs.append(doc)
        return docs

    async def extract_stream(self, path: str = '/', max_items: int = 100, **kwargs):
        """Extract documents from SharePoint as a stream"""
        if not self.site_id:
            raise ValueError("Missing 'site_id' for SharePointSource")

        try:
            # Graph API URL for children of a site's drive
            # If path is '/', we use 'root', otherwise we use folder ID or relative path
            if path == '/':
                url = f"{self.GRAPH_BASE}/sites/{self.site_id}/drive/root/children"
            else:
                # Handle relative path or item ID
                url = f"{self.GRAPH_BASE}/sites/{self.site_id}/drive/root:/{path.strip('/')}:/children"

            count = 0
            while url and count < max_items:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()

                for item in data.get('value', []):
                    if count >= max_items: break
                    
                    # Process file - check if it has a 'file' facet
                    if 'file' not in item: continue 
                    
                    name = item.get('name', '')
                    download_url = item.get('@microsoft.graph.downloadUrl') or f"{self.GRAPH_BASE}/sites/{self.site_id}/drive/items/{item['id']}/content"
                    
                    try:
                        dresp = self.session.get(download_url, timeout=60)
                        if dresp.status_code == 200:
                            doc = self._process_file_content(name, dresp.content, item)
                            if doc:
                                yield doc
                                count += 1
                    except Exception as e:
                        logger.warning(f"Failed to process SharePoint file {name}: {e}")

                url = data.get('@odata.nextLink') # Handle pagination

            logger.info(f"Finished SharePoint extraction for site {self.site_id}")

        except Exception as e:
            logger.error(f"SharePoint extraction failed: {e}")
            raise ValueError(f"SharePoint extraction failed: {str(e)[:200]}")

    def _process_file_content(self, name: str, content: bytes, item: Dict) -> Optional[Dict[str, Any]]:
        """Process binary content into a document"""
        file_ext = (name.split('.')[-1].lower() if '.' in name else '')
        text = ""

        try:
            if file_ext in ('txt', 'md'):
                text = content.decode('utf-8', errors='ignore')
            elif file_ext == 'html':
                text = BeautifulSoup(content, 'html.parser').get_text(separator='\n', strip=True)
            elif file_ext == 'docx':
                import io
                from docx import Document
                doc = Document(io.BytesIO(content))
                text = '\n\n'.join([p.text for p in doc.paragraphs if p.text.strip()])
            elif file_ext == 'pdf':
                import io
                from PyPDF2 import PdfReader
                reader = PdfReader(io.BytesIO(content))
                text = '\n\n'.join([p.extract_text() or '' for p in reader.pages])
            
            if text and text.strip():
                return {
                    'content': text,
                    'metadata': {
                        'source': self.source_name,
                        'item_id': item.get('id'),
                        'filename': name,
                        'type': f'sharepoint_{file_ext}',
                        'size_bytes': item.get('size'),
                        'web_url': item.get('webUrl'),
                        'last_modified': item.get('lastModifiedDateTime'),
                        'created_at': item.get('createdDateTime')
                    }
                }
        except Exception as e:
            logger.debug(f"Failed to parse {name}: {e}")
        
        return None
