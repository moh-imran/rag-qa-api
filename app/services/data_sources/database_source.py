from typing import List, Dict, Any, Optional
import psycopg2
from psycopg2.extras import RealDictCursor
from .base import BaseDataSource
import logging

logger = logging.getLogger(__name__)


class DatabaseSource(BaseDataSource):
    """Extract documents from SQL databases"""
    
    def __init__(
        self,
        db_type: str = "postgresql",
        host: str = "localhost",
        port: int = 5432,
        database: str = None,
        user: str = None,
        password: str = None
    ):
        super().__init__("DatabaseSource")
        self.db_type = db_type
        self.connection_params = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }
    
    async def extract(
        self,
        query: str = None,
        table: str = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract data from database
        
        Args:
            query: Custom SQL query to execute
            table: Table name to extract (if no custom query)
            columns: Specific columns to extract
            limit: Maximum number of rows
        
        Returns:
            List of documents with content and metadata
        """
        if not query and not table:
            raise ValueError("Must provide either 'query' or 'table'")
        
        # Build query if table provided
        if table and not query:
            cols = ', '.join(columns) if columns else '*'
            query = f"SELECT {cols} FROM {table}"
            if limit:
                query += f" LIMIT {limit}"
        
        documents = []
        
        try:
            # Connect to database
            conn = psycopg2.connect(**self.connection_params)
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Execute query
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Convert each row to a document
            for idx, row in enumerate(rows):
                # Convert row to text representation
                content_parts = []
                for key, value in row.items():
                    if value is not None:
                        content_parts.append(f"{key}: {value}")
                
                content = '\n'.join(content_parts)
                
                documents.append({
                    'content': content,
                    'metadata': {
                        'source': self.source_name,
                        'database': self.connection_params['database'],
                        'table': table or 'custom_query',
                        'row_index': idx,
                        'type': 'database_row',
                        'db_type': self.db_type
                    }
                })
            
            cursor.close()
            conn.close()
            
            logger.info(f"Extracted {len(documents)} rows from database")
        
        except Exception as e:
            logger.error(f"Error extracting from database: {e}")
            raise
        
        return documents
