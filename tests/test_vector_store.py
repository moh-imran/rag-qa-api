import pytest
import os
from app.models.vector_store import QdrantVectorStore
from qdrant_client.models import Distance


class TestVectorStore:
    """Test Qdrant vector store operations"""
    
    @pytest.fixture
    def vector_store(self):
        """Create vector store instance"""
        host = os.environ.get("QDRANT_HOST", "qdrant")
        return QdrantVectorStore(
            host=host,
            port=6333,
            collection_name="test_collection"
        )
    
    @pytest.mark.integration
    def test_create_collection(self, vector_store):
        """Test creating a collection"""
        try:
            vector_store.delete_collection()
        except:
            pass
        
        vector_store.create_collection(vector_size=384, recreate=True)
        
        info = vector_store.get_collection_info()
        assert info is not None
    
    @pytest.mark.integration
    def test_store_and_search_vectors(self, vector_store):
        """Test storing and searching vectors"""
        # Create collection
        vector_store.create_collection(vector_size=384, recreate=True)
        
        # Store test vectors
        embedded_chunks = [
            {
                'content': 'Machine learning is AI',
                'embedding': [0.1] * 384,
                'metadata': {'source': 'test', 'id': 1}
            },
            {
                'content': 'Deep learning uses neural networks',
                'embedding': [0.2] * 384,
                'metadata': {'source': 'test', 'id': 2}
            }
        ]
        
        result = vector_store.store_vectors(embedded_chunks)
        assert result['stored'] == 2
        
        # Search
        query_vector = [0.15] * 384
        results = vector_store.search(query_vector, limit=2)
        
        assert len(results) <= 2
        assert all('content' in r for r in results)
        assert all('score' in r for r in results)
    
    @pytest.mark.integration
    def test_delete_collection(self, vector_store):
        """Test deleting collection"""
        vector_store.create_collection(vector_size=384, recreate=True)
        vector_store.delete_collection()
        
        with pytest.raises(Exception):
            vector_store.get_collection_info()