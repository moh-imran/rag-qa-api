from typing import List, Dict, Any, Optional
from .base import BaseDataSource
import logging

logger = logging.getLogger(__name__)


class DatabaseSource(BaseDataSource):
    """Extract documents from SQL databases (PostgreSQL) or MongoDB"""

    def __init__(
        self,
        db_type: str = "postgresql",
        host: str = "localhost",
        port: int = 5432,
        database: str = None,
        user: str = None,
        password: str = None,
        connection_string: str = None
    ):
        super().__init__("DatabaseSource")
        self.db_type = db_type.lower()
        self.connection_string = connection_string

        # Handle MongoDB connection string format
        if host and host.startswith('mongodb://'):
            self.db_type = 'mongodb'
            self.connection_string = host
            # Extract database name from connection string or use provided
            if database:
                self.database = database
            else:
                # Try to extract from connection string
                parts = host.split('/')
                self.database = parts[-1] if len(parts) > 3 else 'test'
        else:
            self.connection_params = {
                'host': host,
                'port': port,
                'database': database,
                'user': user,
                'password': password
            }
            self.database = database

    async def extract_stream(
        self,
        query: str = None,
        table: str = None,
        collection: str = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        **kwargs
    ):
        """
        Generator-based extraction to avoid memory overhead.
        Yields documents as they are processed.
        """
        if self.db_type == 'mongodb':
            async for doc in self._stream_mongodb(collection=collection or table, query=query, limit=limit):
                yield doc
        else:
            async for doc in self._stream_postgresql(query=query, table=table, columns=columns, limit=limit):
                yield doc

    async def _stream_mongodb(self, collection: str = None, query: str = None, limit: int = None):
        """Stream documents from MongoDB"""
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure
        except ImportError:
            raise ValueError("pymongo is not installed. Install it with: pip install pymongo")

        try:
            client = MongoClient(self.connection_string, serverSelectionTimeoutMS=10000)
            db = client[self.database]

            # Automatic Discovery if no collection provided
            collections_to_process = [collection] if collection else db.list_collection_names()
            if not collections_to_process:
                logger.warning(f"No collections found in database '{self.database}'")
                return

            # Parse query filter if provided
            filter_dict = {}
            if query:
                try:
                    import json
                    filter_dict = json.loads(query)
                except:
                    logger.warning(f"Could not parse query as JSON, using empty filter")

            total_extracted = 0
            for coll_name in collections_to_process:
                logger.info(f"Streaming from MongoDB collection: {coll_name}")
                coll = db[coll_name]
                
                cursor = coll.find(filter_dict)
                if limit:
                    # Adjust limit based on what's already extracted
                    remaining = limit - total_extracted
                    if remaining <= 0:
                        break
                    cursor = cursor.limit(remaining)

                for idx, doc in enumerate(cursor):
                    # Improved formatting for nested structures
                    content = self._format_mongodb_doc(doc)
                    
                    if content.strip():
                        yield {
                            'content': content,
                            'metadata': {
                                'source': self.source_name,
                                'database': self.database,
                                'collection': coll_name,
                                'doc_id': str(doc.get('_id', idx)),
                                'type': 'mongodb_document',
                                'db_type': 'mongodb'
                            }
                        }
                        total_extracted += 1
                        
                        if total_extracted % 1000 == 0:
                            logger.info(f"MongoDB Progress: Extracted {total_extracted} documents...")

                if limit and total_extracted >= limit:
                    break

            client.close()
            logger.info(f"Total extracted from MongoDB: {total_extracted} documents")

        except Exception as e:
            logger.error(f"Error streaming from MongoDB: {e}")
            raise ValueError(f"MongoDB streaming failed: {str(e)[:200]}")

    def _format_mongodb_doc(self, doc: Dict[str, Any], indent: int = 0) -> str:
        """Recursively format MongoDB document with better structure"""
        lines = []
        prefix = "  " * indent
        
        for key, value in doc.items():
            if key == '_id' and indent == 0:
                continue
                
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_mongodb_doc(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                for item in value:
                    if isinstance(item, (dict, list)):
                        lines.append(self._format_mongodb_doc({'item': item}, indent + 1))
                    else:
                        lines.append(f"{prefix}  - {item}")
            else:
                lines.append(f"{prefix}{key}: {value}")
                
        return "\n".join(lines)

    async def _stream_postgresql(self, query: str = None, table: str = None, columns: Optional[List[str]] = None, limit: int = None):
        """Stream rows from PostgreSQL"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
        except ImportError:
            raise ValueError("psycopg2 is not installed. Install it with: pip install psycopg2-binary")

        try:
            conn = psycopg2.connect(**self.connection_params, connect_timeout=10)
            # Use server-side cursor for efficient streaming
            cursor_name = f"cur_{self.source_name.lower()}_{hash(self.connection_string or str(self.connection_params)) % 10000}"
            cursor = conn.cursor(name=cursor_name, cursor_factory=RealDictCursor)

            # Discovery or build query
            if not query and not table:
                # Discovery: find all tables in public schema
                cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_type = 'BASE TABLE'")
                tables = [row['table_name'] for row in cursor.fetchall()]
                if not tables:
                    logger.warning("No tables discovered in 'public' schema")
                else:
                    logger.info(f"Discovered PostgreSQL tables: {tables}")
            else:
                tables = [table] if table else [None]

            total_extracted = 0
            for current_table in tables:
                if limit and total_extracted >= limit:
                    break
                    
                exec_query = query
                if current_table and not exec_query:
                    cols = ', '.join(columns) if columns else '*'
                    exec_query = f"SELECT {cols} FROM {current_table}"
                
                logger.info(f"Streaming from PostgreSQL table/query: {current_table or 'Custom Query'}")
                cursor.execute(exec_query)
                
                while True:
                    # Fetch in batches to be efficient but keep memory low
                    batch_size = 100
                    rows = cursor.fetchmany(batch_size)
                    if not rows:
                        break
                        
                    for idx, row in enumerate(rows):
                        content_parts = []
                        for key, value in row.items():
                            if value is not None:
                                content_parts.append(f"{key}: {value}")

                        content = "\n".join(content_parts)
                        if content.strip():
                            yield {
                                'content': content,
                                'metadata': {
                                    'source': self.source_name,
                                    'database': self.connection_params.get('database'),
                                    'table': current_table or 'custom_query',
                                    'row_index': total_extracted,
                                    'type': 'database_row',
                                    'db_type': 'postgresql'
                                }
                            }
                            total_extracted += 1
                            
                            if total_extracted % 1000 == 0:
                                logger.info(f"PostgreSQL Progress: Extracted {total_extracted} rows...")
                        
                        if limit and total_extracted >= limit:
                            break
                    
                    if limit and total_extracted >= limit:
                        break

            cursor.close()
            conn.close()
            logger.info(f"Total extracted from PostgreSQL: {total_extracted} rows")

        except Exception as e:
            logger.error(f"Error streaming from PostgreSQL: {e}")
            raise ValueError(f"PostgreSQL streaming failed: {str(e)[:200]}")

    async def extract(self, **kwargs) -> List[Dict[str, Any]]:
        """Extract all data into memory using the streaming implementation"""
        documents = []
        async for doc in self.extract_stream(**kwargs):
            documents.append(doc)
        return documents
