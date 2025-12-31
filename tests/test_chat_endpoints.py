import pytest
from unittest.mock import MagicMock, patch

class TestChatEndpoints:
    """Test chat API endpoints"""

    @pytest.fixture
    def mock_rag_pipeline(self):
        with patch("app.routers.chat.rag_pipeline") as mock:
            yield mock

    def test_chat_health_not_initialized(self, test_client):
        """Test health check when pipeline not initialized"""
        with patch("app.routers.chat.rag_pipeline", None):
            response = test_client.get("/chat/health")
            assert response.status_code == 503
            assert "not initialized" in response.json()["detail"]

    def test_chat_health_ready(self, test_client, mock_rag_pipeline):
        """Test health check when pipeline is ready"""
        mock_rag_pipeline.openai_model_name = "gpt-4o-mini"
        mock_rag_pipeline.embedding_model_name = "all-MiniLM-L6-v2"
        mock_rag_pipeline.vector_store.collection_name = "test_collection"
        mock_rag_pipeline.client = MagicMock()
        
        response = test_client.get("/chat/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["model"] == "gpt-4o-mini"

    def test_list_models_success(self, test_client, mock_rag_pipeline):
        """Test listing OpenAI models"""
        mock_models = MagicMock()
        mock_models.data = [
            MagicMock(id="gpt-4o", owned_by="openai"),
            MagicMock(id="gpt-4o-mini", owned_by="openai"),
            MagicMock(id="text-embedding-3-small", owned_by="openai")
        ]
        mock_rag_pipeline.client.models.list.return_value = mock_models
        
        response = test_client.get("/chat/models")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        # Should only include GPT models based on the implementation
        assert len(data["available_models"]) == 2
        assert data["available_models"][0]["name"] == "gpt-4o"

    def test_query_documents_success(self, test_client, mock_rag_pipeline):
        """Test successful document query"""
        mock_rag_pipeline.query.return_value = {
            "answer": "The answer is 42",
            "context_used": True,
            "sources": [{"content": "Life, Universe, Everything", "metadata": {}, "score": 1.0}]
        }
        
        payload = {
            "question": "What is the answer?",
            "top_k": 3
        }
        
        response = test_client.post("/chat/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "The answer is 42"
        assert data["context_used"] is True

    def test_chat_success(self, test_client, mock_rag_pipeline):
        """Test successful chat with history"""
        mock_rag_pipeline.chat.return_value = {
            "answer": "I understand you're asking about AI.",
            "context_used": False
        }
        
        payload = {
            "messages": [
                {"role": "user", "content": "Tell me about AI"},
                {"role": "assistant", "content": "AI is..."},
                {"role": "user", "content": "Go on"}
            ],
            "top_k": 5,
            "max_tokens": 500,
            "temperature": 0.5
        }
        
        response = test_client.post("/chat/", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "AI" in data["answer"]
        mock_rag_pipeline.chat.assert_called_once()
        
        # Verify the arguments passed to rag_pipeline.chat
        args, kwargs = mock_rag_pipeline.chat.call_args
        assert len(kwargs["messages"]) == 3
        assert kwargs["top_k"] == 5
        assert kwargs["max_tokens"] == 500
        assert kwargs["temperature"] == 0.5
