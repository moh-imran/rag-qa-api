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

    async def extract(self, path: str = '/', max_items: int = 100, **kwargs) -> List[Dict[str, Any]]:
        documents: List[Dict[str, Any]] = []

        if not self.site_id:
            raise ValueError("Missing 'site_id' for SharePointSource")

        try:
            # List children of root folder
            url = f"{self.GRAPH_BASE}/sites/{self.site_id}/drive/root/children"
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            count = 0
            for item in data.get('value', []):
                if count >= max_items:
                    break
                name = item.get('name', '')
                file_ext = (name.split('.')[-1].lower() if '.' in name else '')
                download_url = f"{self.GRAPH_BASE}/sites/{self.site_id}/drive/items/{item['id']}/content"
                dresp = self.session.get(download_url, timeout=60)
                if dresp.status_code != 200:
                    continue
                # Handle different types
                if file_ext in ('txt', 'md'):
                    text = dresp.text
                elif file_ext == 'html':
                    text = BeautifulSoup(dresp.text, 'html.parser').get_text(separator='\n', strip=True)
                elif file_ext == 'docx':
                    try:
                        from docx import Document as DocxDocument
                        import io
                        doc = DocxDocument(io.BytesIO(dresp.content))
                        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                        text = '\n\n'.join(paragraphs)
                    except Exception as e:
                        logger.warning(f"Failed to parse docx {name}: {e}")
                        continue
                elif file_ext == 'pdf':
                    try:
                        import io
                        from PyPDF2 import PdfReader
                        reader = PdfReader(io.BytesIO(dresp.content))
                        pages = []
                        for p in reader.pages:
                            try:
                                t = p.extract_text() or ''
                                pages.append(t)
                            except Exception:
                                continue
                        text = '\n\n'.join(pages)
                    except Exception as e:
                        logger.warning(f"Failed to parse pdf {name}: {e}")
                        continue
                else:
                    # unsupported type
                    continue

                if text and text.strip():
                    documents.append({
                        'content': text,
                        'metadata': {
                            'source': self.source_name,
                            'item_id': item.get('id'),
                            'filename': name,
                            'type': f'sharepoint_{file_ext}',
                            'size_bytes': item.get('size')
                        }
                    })
                    count += 1

            logger.info(f"Extracted {len(documents)} documents from SharePoint site {self.site_id}")
            return documents
        except Exception as e:
            logger.error(f"Error extracting SharePoint content: {e}")
            raise
