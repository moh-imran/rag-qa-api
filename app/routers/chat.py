from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)

rag_pipeline = None


def set_rag_pipeline(pipeline):
    """Set RAG pipeline instance"""
    global rag_pipeline
    rag_pipeline = pipeline


# Request/Response Models
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")


class QueryRequest(BaseModel):
    question: str = Field(..., description="User question")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    score_threshold: Optional[float] = Field(None, description="Minimum similarity score (0-1)")
    system_instruction: Optional[str] = Field(None, description="Custom system instruction")
    max_tokens: int = Field(1000, description="Maximum tokens to generate", ge=100, le=4000)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0, le=1)
    return_sources: bool = Field(True, description="Return source documents")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    max_tokens: int = Field(1000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    system_instruction: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    context_used: bool
    sources: Optional[List[Dict]] = None


@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents and get AI-generated answer
    
    Example:
    ```json
    {
        "question": "What is machine learning?",
        "top_k": 5,
        "temperature": 0.7
    }
    ```
    """
    try:
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            system_instruction=request.system_instruction,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            return_sources=request.return_sources
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def chat(request: ChatRequest):
    """
    Chat with conversation history
    """
    try:
        # Convert to dict format
        messages = [msg.dict() for msg in request.messages]
        
        result = rag_pipeline.chat(
            messages=messages,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_instruction=request.system_instruction
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """List available OpenAI models"""
    try:
        if rag_pipeline.client is None:
            raise HTTPException(status_code=503, detail="OpenAI API not configured")
        
        models = rag_pipeline.client.models.list()
        
        return {
            "available_models": [
                {
                    "name": model.id,
                    "owned_by": model.owned_by
                }
                for model in models.data
                if "gpt" in model.id
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def chat_health():
    """Check if RAG pipeline is ready"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    if rag_pipeline.client is None:
        raise HTTPException(status_code=503, detail="OpenAI API key not configured")
    
    return {
        "status": "ready",
        "model": rag_pipeline.openai_model_name,
        "embedding_model": rag_pipeline.embedding_model_name,
        "collection": rag_pipeline.vector_store.collection_name
    }