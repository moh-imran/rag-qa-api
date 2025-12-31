import pytest
from unittest.mock import MagicMock, patch
from app.services.rag_pipeline import RAGPipeline

class TestRAGPipeline:
    """Test RAGPipeline service"""

    @pytest.fixture
    def mock_qdrant(self):
        with patch("app.services.rag_pipeline.QdrantVectorStore") as mock:
            yield mock

    @pytest.fixture
    def mock_sentence_transformer(self):
        with patch("app.services.rag_pipeline.SentenceTransformer") as mock:
            # Mock the encode method to return an object with tolist()
            mock_instance = mock.return_value
            mock_embedding = MagicMock()
            mock_embedding.tolist.return_value = [0.1] * 384
            mock_instance.encode.return_value = mock_embedding
            yield mock

    @pytest.fixture
    def rag_pipeline(self, mock_qdrant):
        return RAGPipeline(
            openai_api_key="test-key",
            qdrant_host="localhost",
            qdrant_port=6333
        )

    def test_init(self, rag_pipeline):
        """Test RAGPipeline initialization"""
        assert rag_pipeline.openai_model_name == "gpt-4o-mini"
        assert rag_pipeline.client is not None
        assert rag_pipeline.vector_store is not None

    def test_embed_query(self, rag_pipeline, mock_sentence_transformer):
        """Test query embedding"""
        embedding = rag_pipeline.embed_query("test query")
        assert len(embedding) == 384
        assert embedding[0] == 0.1
        mock_sentence_transformer.return_value.encode.assert_called_once()

    def test_retrieve_context(self, rag_pipeline):
        """Test context retrieval"""
        mock_results = [{"content": "test content", "metadata": {"filename": "test.txt"}, "score": 0.9}]
        rag_pipeline.vector_store.search.return_value = mock_results
        
        results = rag_pipeline.retrieve_context([0.1] * 384)
        assert len(results) == 1
        assert results[0]["content"] == "test content"
        rag_pipeline.vector_store.search.assert_called_once()

    def test_build_prompt(self, rag_pipeline):
        """Test prompt building"""
        query = "What is AI?"
        context_docs = [
            {"content": "AI is artificial intelligence.", "metadata": {"filename": "ai.txt"}, "score": 0.95}
        ]
        
        prompt = rag_pipeline.build_prompt(query, context_docs)
        assert "What is AI?" in prompt
        assert "AI is artificial intelligence." in prompt
        assert "Source: ai.txt" in prompt

    def test_generate_answer_success(self, rag_pipeline):
        """Test successful answer generation"""
        with patch.object(rag_pipeline, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Mocked answer"))]
            mock_client.chat.completions.create.return_value = mock_response
            
            answer = rag_pipeline.generate_answer("test prompt")
            assert answer == "Mocked answer"
            mock_client.chat.completions.create.assert_called_once()

    def test_generate_answer_no_client(self, rag_pipeline):
        """Test answer generation with no OpenAI client"""
        rag_pipeline.client = None
        with pytest.raises(ValueError, match="OpenAI API key not configured"):
            rag_pipeline.generate_answer("test prompt")

    def test_query_success(self, rag_pipeline, mock_sentence_transformer):
        """Test complete query pipeline success"""
        # Mock retrieval
        rag_pipeline.vector_store.search.return_value = [
            {"content": "test content", "metadata": {"filename": "test.txt"}, "score": 0.9}
        ]
        
        # Mock generation
        with patch.object(rag_pipeline, 'client') as mock_client:
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Mocked answer"))]
            mock_client.chat.completions.create.return_value = mock_response
            
            result = rag_pipeline.query("test question")
            
            assert result["answer"] == "Mocked answer"
            assert result["context_used"] is True
            assert len(result["sources"]) == 1
            assert "test content" in result["sources"][0]["content"]

    def test_query_no_results(self, rag_pipeline, mock_sentence_transformer):
        """Test query pipeline with no retrieval results"""
        rag_pipeline.vector_store.search.return_value = []
        
        result = rag_pipeline.query("test question")
        
        assert "couldn't find any relevant information" in result["answer"]
        assert result["context_used"] is False
        assert result["sources"] == []

    def test_chat(self, rag_pipeline, mock_sentence_transformer):
        """Test chat method"""
        with patch.object(rag_pipeline, 'query') as mock_query:
            mock_query.return_value = {"answer": "Chat response", "context_used": True}
            
            messages = [{"role": "user", "content": "Hello"}]
            result = rag_pipeline.chat(messages)
            
            assert result["answer"] == "Chat response"
            mock_query.assert_called_once_with("Hello", top_k=5)
