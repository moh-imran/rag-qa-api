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
        api_key: Optional[str] = None,
        collection_name: str = "documents"
    ):
        """
        Initialize Qdrant client
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: Qdrant API key for authentication
            collection_name: Name of the collection to use
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection_name = collection_name
        
        # Initialize client
        if api_key:
            # Explicitly set https=False for local/docker hosts to avoid SSL errors,
            # as qdrant-client might default to True when an api_key is present.
            self.client = QdrantClient(host=host, port=port, api_key=api_key, https=False)
            logger.info(f"Initialized Authenticated Qdrant vector store: {host}:{port}/{collection_name}")
        else:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Initialized Qdrant vector store: {host}:{port}/{collection_name}")
    
    def switch_collection(self, collection_name: str):
        """Switch to a different collection"""
        self.collection_name = collection_name
        logger.info(f"Switched to collection: {collection_name}")
    
    def list_collections(self) -> List[str]:
        """List all available collections"""
        try:
            collections = self.client.get_collections()
            return [c.name for c in collections.collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def create_collection(
        self,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        recreate: bool = False
    ):
        """
        Create a collection in Qdrant with Hybrid Search (Dense + Sparse)
        
        Args:
            vector_size: Dimension of dense embeddings (e.g., 384)
            distance: Distance metric for dense vectors
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
            
            # Create collection with named vectors (Hybrid) + Performance Opts
            from qdrant_client.models import (
                SparseVectorParams, Modifier, ScalarQuantization, 
                ScalarType, HnswConfigDiff, OptimizersConfigDiff,
                ScalarQuantizationConfig
            )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text-dense": VectorParams(
                        size=vector_size,
                        distance=distance,
                        on_disk=True  # Support memory-efficient search for huge datasets
                    )
                },
                sparse_vectors_config={
                    "text-sparse": SparseVectorParams(
                        modifier=Modifier.IDF
                    )
                },
                # Enable Scalar Quantization to reduce RAM usage by 4x and speed up search
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        always_ram=True,
                        quantile=0.99
                    )
                ),
                # Optimize HNSW index for fast retrieval
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    on_disk=True  # Keep index on disk to save RAM at scale
                ),
                # Optimization threshold for batch uploads
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000  # Delay indexing until batch significantly loaded
                )
            )
            logger.info(f"✅ Created HYBRID collection '{self.collection_name}' (Dense: {vector_size}, Sparse: configured)")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def store_vectors(
        self,
        embedded_chunks: List[Dict[str, Any]],
        batch_size: int = 100,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store embedded chunks in Qdrant with hybrid vectors
        
        Args:
            embedded_chunks: List of chunks with 'embedding' (dense) and 'sparse_embedding' (sparse)
            batch_size: Number of points to upload at once
            job_id: Optional job ID to associate with the stored vectors
        """
        if not embedded_chunks:
            logger.warning("No chunks to store")
            return {"stored": 0}
        
        try:
            # Prepare points for Qdrant
            points = []
            for chunk in embedded_chunks:
                # Basic validation
                dense_vector = chunk.get('embedding')
                sparse_vector = chunk.get('sparse_embedding')
                
                # Check if we have sparse vectors available
                vector_dict = {}
                if dense_vector:
                    vector_dict["text-dense"] = dense_vector
                
                # Sparse vector format: {'indices': [...], 'values': [...]}
                if sparse_vector:
                    from qdrant_client.models import SparseVector
                    vector_dict["text-sparse"] = SparseVector(
                        indices=sparse_vector['indices'],
                        values=sparse_vector['values']
                    )
                
                # Prepare payload with original metadata and job_id
                payload_metadata = chunk.get('metadata', {})
                if job_id:
                    payload_metadata['job_id'] = job_id

                point = PointStruct(
                    id=str(uuid.uuid4()),  # Generate unique ID
                    vector=vector_dict,
                    payload={
                        'content': chunk['content'],
                        'metadata': payload_metadata
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
            
            logger.info(f"✅ Successfully stored {total_stored} points in Qdrant")
            
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
        sparse_vector: Optional[Dict[str, List]] = None,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Search (Dense + Sparse)
        
        Args:
            query_vector: Dense embedding
            sparse_vector: Sparse indices/values {'indices': [], 'values': []}
            limit: Number of results
            score_threshold: Min score
            filter_dict: Metadata filters
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
            
            # Prepare Query (Hybrid if sparse available)
            from qdrant_client.models import SparseVector
            
            if sparse_vector:
                # Hybrid Query (Prefetch sparse, rescore with dense is common strategy, 
                # OR simple weighted fusion supported in newer Qdrant)
                
                # Using simple fusion via query_points (requires newer Qdrant server)
                # We will prefetch with sparse/dense and fuse.
                # For simplicity in this implementation, we'll rely on Qdrant's automatic Prefetch fusion if passing multiple
                # But actually, query_points expects a single query unless using Batch.
                
                # Let's perform a simple weighted search: pass named vector if we want specific one.
                # But for true hybrid, we want to query BOTH.
                
                # Implementing simple hybrid strategy: Query Dense, but re-order? 
                # No, best is to use Prefetch.
                
                from qdrant_client.models import Prefetch
                
                # Fetch more candidates with sparse, then rescore with dense
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=Prefetch(
                        query=SparseVector(
                            indices=sparse_vector['indices'],
                            values=sparse_vector['values']
                        ),
                        using="text-sparse",
                        limit=limit * 2,
                        filter=query_filter
                    ),
                    query=query_vector,
                    using="text-dense",
                    limit=limit,
                    score_threshold=score_threshold
                )
            else:
                # Standard Dense Search
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    using="text-dense",
                    limit=limit,
                    query_filter=query_filter,
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