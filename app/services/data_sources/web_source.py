from typing import List, Dict, Any, Optional
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from .base import BaseDataSource
import logging

logger = logging.getLogger(__name__)


class WebSource(BaseDataSource):
    """Extract documents from web pages"""
    
    def __init__(self):
        super().__init__("WebSource")
    
    async def extract(
        self,
        url: str = None,
        urls: List[str] = None,
        max_depth: int = 0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract content from web pages
        
        Args:
            url: Single URL to scrape
            urls: List of URLs to scrape
            max_depth: Crawl depth (0 = single page only)
        
        Returns:
            List of documents with content and metadata
        """
        if url:
            urls = [url]
        
        if not urls:
            raise ValueError("Must provide 'url' or 'urls'")
        
        documents = []
        visited = set()
        
        for start_url in urls:
            docs = await self._scrape_url(start_url, visited, max_depth, 0)
            documents.extend(docs)
        
        logger.info(f"Extracted {len(documents)} documents from {len(visited)} web pages")
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
        
        visited.add(url)
        documents = []
        
        try:
            # Fetch page
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (RAG Bot)'
            })
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            if text:
                # Get page title
                title = soup.title.string if soup.title else urlparse(url).path
                
                documents.append({
                    'content': text,
                    'metadata': {
                        'source': self.source_name,
                        'url': url,
                        'title': title,
                        'type': 'web',
                        'depth': current_depth
                    }
                })
            
            # Follow links if depth allows
            if current_depth < max_depth:
                base_domain = urlparse(url).netloc
                
                for link in soup.find_all('a', href=True):
                    next_url = urljoin(url, link['href'])
                    next_domain = urlparse(next_url).netloc
                    
                    # Only follow links on same domain
                    if next_domain == base_domain and next_url not in visited:
                        child_docs = await self._scrape_url(
                            next_url, visited, max_depth, current_depth + 1
                        )
                        documents.extend(child_docs)
        
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        
        return documents
