from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import logging
import json
import uuid

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
    metadata_filters: Optional[Dict] = Field(None, description="Metadata filters (e.g., {'filename': 'doc.pdf'})")
    use_hyde: bool = Field(False, description="Use HyDE for improved retrieval")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history")
    top_k: int = Field(5, description="Number of documents to retrieve", ge=1, le=20)
    max_tokens: int = Field(1000, description="Maximum tokens to generate")
    temperature: float = Field(0.7, description="Sampling temperature")
    system_instruction: Optional[str] = None
    metadata_filters: Optional[Dict] = Field(None, description="Metadata filters")
    use_hyde: bool = Field(False, description="Use HyDE for improved retrieval")


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
            return_sources=request.return_sources,
            metadata_filters=request.metadata_filters,
            use_hyde=request.use_hyde
        )
        
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """
    Query documents with streaming response

    Returns Server-Sent Events (SSE) stream with:
    - conversation_id: Query ID for feedback tracking
    - retrieval_start: Retrieval begins
    - retrieval_complete: Documents retrieved
    - generation_start: Answer generation begins
    - token: Individual tokens
    - done: Generation complete
    """
    try:
        async def event_generator():
            # Generate a unique query_id for this request
            query_id = str(uuid.uuid4())
            yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': query_id})}\n\n"

            for event in rag_pipeline.query_stream(
                question=request.question,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                system_instruction=request.system_instruction,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                metadata_filters=request.metadata_filters,
                use_hyde=request.use_hyde
            ):
                # Format as Server-Sent Event
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Error processing streaming query: {e}")
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
            system_instruction=request.system_instruction,
            metadata_filters=request.metadata_filters,
            use_hyde=request.use_hyde
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """
    Chat with streaming response

    Returns Server-Sent Events (SSE) stream
    """
    try:
        async def event_generator():
            # Generate a unique query_id for this request
            query_id = str(uuid.uuid4())
            yield f"data: {json.dumps({'type': 'conversation_id', 'conversation_id': query_id})}\n\n"

            # Convert to dict format
            messages = [msg.dict() for msg in request.messages]

            # Get last message as question for streaming
            last_message = messages[-1]['content']

            for event in rag_pipeline.query_stream(
                question=last_message,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                metadata_filters=request.metadata_filters,
                use_hyde=request.use_hyde
            ):
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Error in streaming chat: {e}")
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