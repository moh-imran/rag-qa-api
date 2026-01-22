from typing import List, Dict, Any, Optional
from notion_client import Client
from .base import BaseDataSource
import logging

logger = logging.getLogger(__name__)


class NotionSource(BaseDataSource):
    """Extract documents from Notion workspace"""
    
    def __init__(self, api_key: str):
        super().__init__("NotionSource")
        self.client = Client(auth=api_key)
    
    async def extract(
        self,
        database_id: str = None,
        page_id: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract content from Notion
        
        Args:
            database_id: Notion database ID to extract
            page_id: Specific page ID to extract
        
        Returns:
            List of documents with content and metadata
        """
        documents = []
        
        if database_id:
            documents.extend(await self._extract_database(database_id))
        elif page_id:
            documents.extend(await self._extract_page(page_id))
        else:
            raise ValueError("Must provide either 'database_id' or 'page_id'")
        
        logger.info(f"Extracted {len(documents)} documents from Notion")
        return documents
    
    async def _extract_database(self, database_id: str) -> List[Dict[str, Any]]:
        """Extract all pages from a Notion database"""
        documents = []
        
        try:
            # Query database
            results = self.client.databases.query(database_id=database_id)
            
            for page in results.get('results', []):
                page_docs = await self._extract_page(page['id'])
                documents.extend(page_docs)
        
        except Exception as e:
            logger.error(f"Error extracting Notion database {database_id}: {e}")
            raise
        
        return documents
    
    async def _extract_page(self, page_id: str) -> List[Dict[str, Any]]:
        """Extract content from a single Notion page"""
        try:
            # Get page properties
            page = self.client.pages.retrieve(page_id=page_id)
            
            # Get page blocks (content)
            blocks = self.client.blocks.children.list(block_id=page_id)
            
            # Extract text from blocks
            content_parts = []
            for block in blocks.get('results', []):
                text = self._extract_block_text(block)
                if text:
                    content_parts.append(text)
            
            content = '\n\n'.join(content_parts)
            
            # Get page title
            title = self._get_page_title(page)
            
            return [{
                'content': content,
                'metadata': {
                    'source': self.source_name,
                    'page_id': page_id,
                    'title': title,
                    'type': 'notion_page',
                    'url': page.get('url', '')
                }
            }]
        
        except Exception as e:
            logger.error(f"Error extracting Notion page {page_id}: {e}")
            return []
    
    def _extract_block_text(self, block: Dict) -> str:
        """Extract text from a Notion block"""
        block_type = block.get('type')
        
        if not block_type:
            return ""
        
        block_content = block.get(block_type, {})
        
        # Handle different block types
        if 'rich_text' in block_content:
            texts = [rt.get('plain_text', '') for rt in block_content['rich_text']]
            return ' '.join(texts)
        
        return ""
    
    def _get_page_title(self, page: Dict) -> str:
        """Extract title from page properties"""
        properties = page.get('properties', {})
        
        for prop_name, prop_value in properties.items():
            if prop_value.get('type') == 'title':
                title_array = prop_value.get('title', [])
                if title_array:
                    return title_array[0].get('plain_text', 'Untitled')
        
        return 'Untitled'
