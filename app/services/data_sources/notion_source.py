from typing import List, Dict, Any, Optional
from .base import BaseDataSource
import logging

logger = logging.getLogger(__name__)


class NotionSource(BaseDataSource):
    """Extract documents from Notion workspace"""

    def __init__(self, api_key: str):
        super().__init__("NotionSource")
        if not api_key:
            raise ValueError("Notion API key is required")
        if not api_key.startswith(('secret_', 'ntn_')):
            raise ValueError("Invalid Notion API key format. Key should start with 'secret_' or 'ntn_'")

        try:
            from notion_client import Client
            self.client = Client(auth=api_key)
        except ImportError:
            raise ValueError("notion-client is not installed. Install it with: pip install notion-client")
        
        self.supported_blocks = {
            'paragraph', 'heading_1', 'heading_2', 'heading_3', 
            'bulleted_list_item', 'numbered_list_item', 'to_do', 
            'toggle', 'code', 'quote', 'callout', 'equation'
        }
    
    async def extract(self, **kwargs) -> List[Dict[str, Any]]:
        """Standard batch extraction (returns list)"""
        docs = []
        async for doc in self.extract_stream(**kwargs):
            docs.append(doc)
        return docs

    async def extract_stream(
        self,
        database_id: str = None,
        page_id: str = None,
        **kwargs
    ):
        """
        Extract content from Notion as a stream
        """
        from notion_client import APIResponseError

        try:
            # Test API key
            try:
                self.client.users.me()
            except APIResponseError as e:
                if e.status == 401:
                    raise ValueError("Notion API key is invalid or expired.")
                raise

            if database_id:
                async for doc in self._extract_database_stream(database_id):
                    yield doc
            elif page_id:
                async for doc in self._extract_page_stream(page_id):
                    yield doc
            else:
                logger.info("No database_id or page_id, searching for accessible pages...")
                search_results = self.client.search(filter={"property": "object", "value": "page"}, page_size=10)
                for page in search_results.get('results', []):
                    async for doc in self._extract_page_stream(page['id']):
                        yield doc

        except Exception as e:
            logger.error(f"Notion extraction failed: {e}")
            if isinstance(e, ValueError): raise
            raise ValueError(f"Notion extraction failed: {str(e)[:200]}")
    
    async def _extract_database_stream(self, database_id: str):
        """Extract pages from a database as a stream"""
        has_more = True
        start_cursor = None

        while has_more:
            query_params = {"database_id": database_id}
            if start_cursor:
                query_params["start_cursor"] = start_cursor

            results = self.client.databases.query(**query_params)
            for page in results.get('results', []):
                async for doc in self._extract_page_stream(page['id']):
                    yield doc

            has_more = results.get('has_more', False)
            start_cursor = results.get('next_cursor')

    async def _extract_page_stream(self, page_id: str):
        """Extract content from a single page as a stream"""
        try:
            page = self.client.pages.retrieve(page_id=page_id)
            blocks = self.client.blocks.children.list(block_id=page_id)
            
            content_parts = []
            for block in blocks.get('results', []):
                text = self._extract_block_text(block)
                if text:
                    content_parts.append(text)
            
            content = '\n\n'.join(content_parts)
            if not content.strip():
                return

            yield {
                'content': content,
                'metadata': {
                    'source': self.source_name,
                    'page_id': page_id,
                    'title': self._get_page_title(page),
                    'type': 'notion_page',
                    'url': page.get('url', ''),
                    'created_time': page.get('created_time'),
                    'last_edited_time': page.get('last_edited_time')
                }
            }
        except Exception as e:
            logger.warning(f"Failed to extract Notion page {page_id}: {e}")
    
    def _extract_block_text(self, block: Dict) -> str:
        """Extract text from a Notion block with Markdown formatting"""
        block_type = block.get('type')
        if not block_type: return ""
        
        data = block.get(block_type, {})
        text = ""
        
        if 'rich_text' in data:
            text = "".join(rt.get('plain_text', '') for rt in data['rich_text'])
        
        if not text: return ""

        # Format based on type
        if block_type == 'heading_1': return f"# {text}"
        if block_type == 'heading_2': return f"## {text}"
        if block_type == 'heading_3': return f"### {text}"
        if block_type == 'bulleted_list_item': return f"* {text}"
        if block_type == 'numbered_list_item': return f"1. {text}"
        if block_type == 'to_do':
            check = "[x]" if data.get('checked') else "[ ]"
            return f"{check} {text}"
        if block_type == 'toggle': return f"> {text}"
        if block_type == 'quote': return f"> {text}"
        if block_type == 'code':
            lang = data.get('language', '')
            return f"```{lang}\n{text}\n```"
        if block_type == 'callout':
            icon = data.get('icon', {}).get('emoji', 'ℹ️')
            return f"> {icon} {text}"
            
        return text
    
    def _get_page_title(self, page: Dict) -> str:
        """Extract title from page properties"""
        properties = page.get('properties', {})
        
        for prop_name, prop_value in properties.items():
            if prop_value.get('type') == 'title':
                title_array = prop_value.get('title', [])
                if title_array:
                    return title_array[0].get('plain_text', 'Untitled')
        
        return 'Untitled'
