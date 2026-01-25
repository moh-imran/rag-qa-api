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
        from notion_client import APIResponseError

        documents = []

        try:
            # Test the API key by making a simple request
            try:
                self.client.users.me()
            except APIResponseError as e:
                if e.status == 401:
                    raise ValueError("Notion API key is invalid or expired. Please check your integration token.")
                raise

            if database_id:
                documents.extend(await self._extract_database(database_id))
            elif page_id:
                documents.extend(await self._extract_page(page_id))
            else:
                # If no database_id or page_id, try to list all accessible pages
                logger.info("No database_id or page_id provided, searching for accessible content...")
                try:
                    search_results = self.client.search(filter={"property": "object", "value": "page"}, page_size=50)
                    for page in search_results.get('results', []):
                        page_docs = await self._extract_page(page['id'])
                        documents.extend(page_docs)
                    if not documents:
                        raise ValueError("No accessible pages found. Make sure your integration has access to at least one page or database.")
                except APIResponseError as e:
                    raise ValueError(f"Notion search failed: {str(e)[:200]}")

            logger.info(f"Extracted {len(documents)} documents from Notion")
            return documents

        except ValueError:
            raise
        except APIResponseError as e:
            if e.status == 401:
                raise ValueError("Notion API authentication failed. Check your API key.")
            elif e.status == 403:
                raise ValueError("Notion access denied. Make sure your integration has access to the requested content.")
            elif e.status == 404:
                raise ValueError("Notion content not found. Check the database_id or page_id.")
            raise ValueError(f"Notion API error: {str(e)[:200]}")
        except Exception as e:
            logger.error(f"Error extracting from Notion: {e}")
            raise ValueError(f"Notion extraction failed: {str(e)[:200]}")
    
    async def _extract_database(self, database_id: str) -> List[Dict[str, Any]]:
        """Extract all pages from a Notion database"""
        from notion_client import APIResponseError

        documents = []

        try:
            # Query database with pagination
            has_more = True
            start_cursor = None

            while has_more:
                query_params = {"database_id": database_id}
                if start_cursor:
                    query_params["start_cursor"] = start_cursor

                results = self.client.databases.query(**query_params)

                for page in results.get('results', []):
                    page_docs = await self._extract_page(page['id'])
                    documents.extend(page_docs)

                has_more = results.get('has_more', False)
                start_cursor = results.get('next_cursor')

            if not documents:
                logger.warning(f"No pages found in Notion database {database_id}")

        except APIResponseError as e:
            if e.status == 404:
                raise ValueError(f"Notion database '{database_id}' not found. Check if the ID is correct and your integration has access.")
            elif e.status == 403:
                raise ValueError(f"Access denied to Notion database '{database_id}'. Share the database with your integration.")
            raise ValueError(f"Notion database query failed: {str(e)[:200]}")
        except Exception as e:
            logger.error(f"Error extracting Notion database {database_id}: {e}")
            raise ValueError(f"Failed to extract Notion database: {str(e)[:200]}")

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
