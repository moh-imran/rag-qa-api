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

    async def extract(
        self,
        query: str = None,
        table: str = None,
        collection: str = None,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract data from database

        Args:
            query: Custom SQL query (PostgreSQL) or MongoDB filter dict as string
            table: Table name (PostgreSQL)
            collection: Collection name (MongoDB)
            columns: Specific columns/fields to extract
            limit: Maximum number of rows/documents

        Returns:
            List of documents with content and metadata
        """
        if self.db_type == 'mongodb':
            return await self._extract_mongodb(collection=collection or table, query=query, limit=limit)
        else:
            return await self._extract_postgresql(query=query, table=table, columns=columns, limit=limit)

    async def _extract_mongodb(self, collection: str = None, query: str = None, limit: int = None) -> List[Dict[str, Any]]:
        """Extract documents from MongoDB"""
        try:
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure
        except ImportError:
            raise ValueError("pymongo is not installed. Install it with: pip install pymongo")

        if not collection:
            raise ValueError("Must provide 'collection' (or 'table') name for MongoDB")

        documents = []

        try:
            # Connect to MongoDB
            logger.info(f"Connecting to MongoDB: {self.connection_string}")
            client = MongoClient(self.connection_string, serverSelectionTimeoutMS=10000)

            # Test connection
            try:
                client.admin.command('ping')
                logger.info("MongoDB connection successful")
            except ServerSelectionTimeoutError:
                raise ValueError(f"MongoDB connection failed: Could not connect to server at {self.connection_string}. Check if MongoDB is running and the connection string is correct.")
            except ConnectionFailure as e:
                raise ValueError(f"MongoDB connection failed: {str(e)[:200]}")
            except OperationFailure as e:
                if "auth" in str(e).lower():
                    raise ValueError(f"MongoDB authentication failed: Invalid username or password.")
                raise ValueError(f"MongoDB operation failed: {str(e)[:200]}")

            db = client[self.database]

            # Check if collection exists
            try:
                collection_names = db.list_collection_names()
                if collection not in collection_names:
                    if not collection_names:
                        raise ValueError(f"Database '{self.database}' has no collections. Check if the database name is correct.")
                    raise ValueError(f"Collection '{collection}' not found in database '{self.database}'. Available collections: {', '.join(collection_names[:10])}")
            except OperationFailure as e:
                if "auth" in str(e).lower():
                    raise ValueError(f"MongoDB authentication failed: User doesn't have permission to list collections.")
                raise

            coll = db[collection]

            # Parse query filter if provided
            filter_dict = {}
            if query:
                try:
                    import json
                    filter_dict = json.loads(query)
                except:
                    logger.warning(f"Could not parse query as JSON, using empty filter")

            # Fetch documents
            cursor = coll.find(filter_dict)
            if limit:
                cursor = cursor.limit(limit)

            for idx, doc in enumerate(cursor):
                # Convert MongoDB document to text
                content_parts = []
                for key, value in doc.items():
                    if key != '_id':  # Skip MongoDB internal ID
                        if value is not None:
                            content_parts.append(f"{key}: {value}")

                content = '\n'.join(content_parts)

                if content.strip():
                    documents.append({
                        'content': content,
                        'metadata': {
                            'source': self.source_name,
                            'database': self.database,
                            'collection': collection,
                            'doc_id': str(doc.get('_id', idx)),
                            'type': 'mongodb_document',
                            'db_type': 'mongodb'
                        }
                    })

            client.close()
            logger.info(f"Extracted {len(documents)} documents from MongoDB collection '{collection}'")

        except ValueError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str:
                raise ValueError(f"MongoDB connection refused. Check if the server is running at the specified address.")
            elif "auth" in error_str:
                raise ValueError(f"MongoDB authentication failed. Check username and password.")
            elif "timeout" in error_str:
                raise ValueError(f"MongoDB connection timed out. The server may be unavailable or the address is incorrect.")
            logger.error(f"Error extracting from MongoDB: {e}")
            raise ValueError(f"MongoDB extraction failed: {str(e)[:200]}")

        return documents

    async def _extract_postgresql(self, query: str = None, table: str = None, columns: Optional[List[str]] = None, limit: int = None) -> List[Dict[str, Any]]:
        """Extract data from PostgreSQL database"""
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            from psycopg2 import OperationalError, ProgrammingError
        except ImportError:
            raise ValueError("psycopg2 is not installed. Install it with: pip install psycopg2-binary")

        if not query and not table:
            raise ValueError("Must provide either 'query' or 'table' for PostgreSQL")

        # Build query if table provided
        if table and not query:
            cols = ', '.join(columns) if columns else '*'
            query = f"SELECT {cols} FROM {table}"
            if limit:
                query += f" LIMIT {limit}"

        documents = []

        try:
            # Connect to database
            host = self.connection_params.get('host', 'localhost')
            port = self.connection_params.get('port', 5432)
            database = self.connection_params.get('database')
            logger.info(f"Connecting to PostgreSQL: {host}:{port}/{database}")

            try:
                conn = psycopg2.connect(**self.connection_params, connect_timeout=10)
            except OperationalError as e:
                error_str = str(e).lower()
                if "password authentication failed" in error_str:
                    raise ValueError(f"PostgreSQL authentication failed: Invalid username or password for database '{database}'.")
                elif "does not exist" in error_str:
                    raise ValueError(f"PostgreSQL database '{database}' does not exist. Check the database name.")
                elif "could not connect" in error_str or "connection refused" in error_str:
                    raise ValueError(f"PostgreSQL connection failed: Could not connect to server at {host}:{port}. Check if PostgreSQL is running.")
                elif "timeout" in error_str:
                    raise ValueError(f"PostgreSQL connection timed out. The server at {host}:{port} may be unavailable.")
                raise ValueError(f"PostgreSQL connection failed: {str(e)[:200]}")

            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Execute query
            try:
                cursor.execute(query)
                rows = cursor.fetchall()
            except ProgrammingError as e:
                error_str = str(e).lower()
                if "does not exist" in error_str:
                    if table:
                        raise ValueError(f"PostgreSQL table '{table}' does not exist in database '{database}'.")
                    raise ValueError(f"PostgreSQL query error: {str(e)[:200]}")
                elif "permission denied" in error_str:
                    raise ValueError(f"PostgreSQL permission denied: User doesn't have access to the requested table/data.")
                raise ValueError(f"PostgreSQL query failed: {str(e)[:200]}")

            # Convert each row to a document
            for idx, row in enumerate(rows):
                # Convert row to text representation
                content_parts = []
                for key, value in row.items():
                    if value is not None:
                        content_parts.append(f"{key}: {value}")

                content = '\n'.join(content_parts)

                if content.strip():
                    documents.append({
                        'content': content,
                        'metadata': {
                            'source': self.source_name,
                            'database': self.connection_params['database'],
                            'table': table or 'custom_query',
                            'row_index': idx,
                            'type': 'database_row',
                            'db_type': 'postgresql'
                        }
                    })

            cursor.close()
            conn.close()

            logger.info(f"Extracted {len(documents)} rows from PostgreSQL")

        except ValueError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "connection" in error_str or "refused" in error_str:
                raise ValueError(f"PostgreSQL connection refused. Check if the server is running.")
            elif "authentication" in error_str or "password" in error_str:
                raise ValueError(f"PostgreSQL authentication failed. Check username and password.")
            logger.error(f"Error extracting from PostgreSQL: {e}")
            raise ValueError(f"PostgreSQL extraction failed: {str(e)[:200]}")

        return documents
