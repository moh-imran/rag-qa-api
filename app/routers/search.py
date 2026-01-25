from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

router = APIRouter(prefix="/search", tags=["search"])

etl_pipeline = None

def set_etl_pipeline(pipeline):
    global etl_pipeline
    etl_pipeline = pipeline


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    limit: int = Field(5, description="Number of results", ge=1, le=50)
    score_threshold: Optional[float] = Field(None, description="Minimum similarity score (0-1)")


class SearchResponse(BaseModel):
    query: str
    results: List[Dict]
    total_results: int


@router.post("", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for similar documents using semantic search"""
    try:
        etl_pipeline._load_embedding_model()
        query_embedding = etl_pipeline.embedding_provider.embed([request.query])[0].tolist()

        results = etl_pipeline.vector_store.search(
            query_vector=query_embedding,
            limit=request.limit,
            score_threshold=request.score_threshold
        )

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))