# app/services/vector_store.py
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
import logging
import uuid

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Handle vector storage and retrieval in Qdrant"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "documents"
    ):
        """
        Initialize Qdrant client
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to use
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        
        # Initialize client
        self.client = QdrantClient(host=host, port=port)
        logger.info(f"Connected to Qdrant at {host}:{port}")
    
    def create_collection(
        self,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ):
        """
        Create a collection in Qdrant
        
        Args:
            vector_size: Dimension of embeddings (e.g., 384 for MiniLM)
            distance: Distance metric (COSINE, EUCLID, DOT)
            recreate: If True, delete existing collection and create new one
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            
            if exists:
                if recreate:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection '{self.collection_name}' already exists")
                    return
            
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            logger.info(f"✅ Created collection '{self.collection_name}' with vector size {vector_size}")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def store_vectors(
        self,
        embedded_chunks: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Store embedded chunks in Qdrant
        
        Args:
            embedded_chunks: List of chunks with embeddings from ETL pipeline
            batch_size: Number of points to upload at once
            
        Returns:
            Dictionary with storage statistics
        """
        if not embedded_chunks:
            logger.warning("No chunks to store")
            return {"stored": 0}
        
        try:
            # Prepare points for Qdrant
            points = []
            for chunk in embedded_chunks:
                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID
                    vector=chunk['embedding'],
                    payload={
                        'content': chunk['content'],
                        'metadata': chunk['metadata']
                    }
                )
                points.append(point)
            
            # Upload in batches
            total_stored = 0
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                total_stored += len(batch)
                logger.info(f"Stored batch {i//batch_size + 1}: {len(batch)} points")
            
            logger.info(f"✅ Successfully stored {total_stored} vectors in Qdrant")
            
            return {
                "status": "success",
                "stored": total_stored,
                "collection": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error storing vectors: {e}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in Qdrant
        
        Args:
            query_vector: Embedding vector to search for
            limit: Number of results to return
            score_threshold: Minimum similarity score (0-1)
            filter_dict: Optional metadata filters
            
        Returns:
            List of matching documents with scores
        """
        try:
            # Build filter if provided
            query_filter = None
            if filter_dict:
                conditions = []
                for key, value in filter_dict.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}",
                            match=MatchValue(value=value)
                        )
                    )
                query_filter = Filter(must=conditions)
            
            # Use query_points (Modern API)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,  # Correct argument for v1.10+
                score_threshold=score_threshold
            )
            
            # Format results
            formatted_results = []
            for result in response.points:
                formatted_results.append({
                    'id': result.id,
                    'score': result.score,
                    'content': result.payload.get('content'),
                    'metadata': result.payload.get('metadata')
                })
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            raise
    
    def delete_by_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Delete vectors by metadata filter
        
        Args:
            filter_dict: Metadata filters for deletion
            
        Example:
            delete_by_filter({'filename': 'doc.pdf'})
        """
        try:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
            
            query_filter = Filter(must=conditions)
            
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=query_filter
            )
            
            logger.info(f"Deleted vectors matching filter: {filter_dict}")
            return {"status": "success", "filter": filter_dict}
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else None,
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise
    
    def delete_collection(self):
        """Delete the entire collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise