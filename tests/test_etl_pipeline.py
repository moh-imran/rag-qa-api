import pytest
from app.services.etl_pipeline import ETLPipeline


class TestETLPipeline:
    """Test complete ETL pipeline"""
    
    @pytest.fixture
    def pipeline(self):
        """Create ETL pipeline instance"""
        # Use mock Qdrant for testing
        return ETLPipeline(
            embedding_model="all-MiniLM-L6-v2",
            qdrant_host="localhost",
            qdrant_port=6333,
            collection_name="test_documents"
        )
    
    def test_chunk_documents(self, pipeline):
        """Test document chunking"""
        documents = [{
            'content': ('A' * 500 + '\n\n') * 4,
            'metadata': {'filename': 'test.txt', 'type': 'txt'}
        }]
        
        chunks = pipeline.chunk_documents(documents, chunk_size=500, chunk_overlap=50)
        
        assert len(chunks) > 1
        assert all('content' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        assert all('chunk_id' in chunk['metadata'] for chunk in chunks)
    
    def test_chunk_small_document(self, pipeline):
        """Test chunking document smaller than chunk_size"""
        documents = [{
            'content': 'Short text',
            'metadata': {'filename': 'test.txt'}
        }]
        
        chunks = pipeline.chunk_documents(documents, chunk_size=1000)
        
        assert len(chunks) == 1
        assert chunks[0]['content'] == 'Short text'
    
    def test_embed_chunks(self, pipeline):
        """Test embedding generation"""
        chunks = [
            {'content': 'Machine learning is great', 'metadata': {'id': 1}},
            {'content': 'Deep learning uses neural networks', 'metadata': {'id': 2}}
        ]
        
        embedded = pipeline.embed_chunks(chunks, batch_size=2)
        
        assert len(embedded) == 2
        assert all('embedding' in chunk for chunk in embedded)
        assert all(len(chunk['embedding']) == 384 for chunk in embedded)  # MiniLM dimension
        assert all(isinstance(chunk['embedding'], list) for chunk in embedded)
    
    def test_embedding_dimension(self, pipeline):
        """Test getting embedding dimension"""
        dim = pipeline.get_embedding_dimension()
        assert dim == 384  # all-MiniLM-L6-v2 dimension