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
        with patch("app.services.rag_pipeline.SentenceTransformer") as mock_st, \
             patch("app.services.rag_pipeline.CrossEncoder") as mock_ce:
            
            # Mock SentenceTransformer
            mock_instance = mock_st.return_value
            mock_embedding = MagicMock()
            mock_embedding.tolist.return_value = [0.1] * 384
            mock_instance.encode.return_value = mock_embedding
            
            # Mock CrossEncoder
            mock_ce_instance = mock_ce.return_value
            # predict returns list of floats (scores)
            mock_ce_instance.predict.return_value = [0.95, 0.85, 0.75]
            
            yield mock_st, mock_ce

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
        assert rag_pipeline.cross_encoder_model is None

    def test_embed_query(self, rag_pipeline, mock_sentence_transformer):
        """Test query embedding"""
        mock_st, _ = mock_sentence_transformer
        embedding = rag_pipeline.embed_query("test query")
        assert len(embedding) == 384
        assert embedding[0] == 0.1
        mock_st.return_value.encode.assert_called_once()

    def test_retrieve_context(self, rag_pipeline, mock_sentence_transformer):
        """Test context retrieval with reranking"""
        _, mock_ce = mock_sentence_transformer
        
        # Setup mock results (needs to correspond to predict return value length ideally or be handled)
        # In this case, we'll return 3 docs
        mock_results = [
            {"content": "doc1", "metadata": {"filename": "1.txt"}, "score": 0.8},
            {"content": "doc2", "metadata": {"filename": "2.txt"}, "score": 0.7},
            {"content": "doc3", "metadata": {"filename": "3.txt"}, "score": 0.6}
        ]
        rag_pipeline.vector_store.search.return_value = mock_results
        
        # predict returns scores for these 3
        mock_ce.return_value.predict.return_value = [0.1, 0.9, 0.5] # doc2 should be first, then doc3, then doc1
        
        results = rag_pipeline.retrieve_context("query", [0.1] * 384, top_k=2)
        
        assert len(results) == 2
        assert results[0]["content"] == "doc2" # Highest rerank score 0.9
        assert results[1]["content"] == "doc3" # Second highest 0.5
        
        rag_pipeline.vector_store.search.assert_called_once()
        mock_ce.return_value.predict.assert_called_once()

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
        # Mock dependencies
        with patch.object(rag_pipeline, '_rephrase_question') as mock_rephrase, \
             patch.object(rag_pipeline, 'retrieve_context') as mock_retrieve, \
             patch.object(rag_pipeline, 'client') as mock_client:
            
            # Setup mocks
            mock_rephrase.return_value = "Rephrased question"
            
            mock_retrieve.return_value = [
                {"content": "Chat context", "metadata": {"filename": "chat.txt"}, "score": 0.9}
            ]
            
            # Mock OpenAI response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Chat response"))]
            mock_client.chat.completions.create.return_value = mock_response
            
            messages = [{"role": "user", "content": "Hello"}]
            result = rag_pipeline.chat(messages)
            
            assert result["answer"] == "Chat response"
            assert result["context_used"] is True
            assert len(result["sources"]) == 1
            
            mock_rephrase.assert_called_once()
            mock_retrieve.assert_called_once()
            mock_client.chat.completions.create.assert_called_once()
