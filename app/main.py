from fastapi import FastAPI
import logging
import os

from app.services.etl_pipeline import ETLPipeline
from app.services.rag_pipeline import RAGPipeline
from app.routers import ingestion, search, collection, chat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Q&A API",
    description="Complete RAG system with document ingestion and AI-powered Q&A",
    version="1.0.0"
)

# Initialize ETL Pipeline
etl_pipeline = ETLPipeline(
    embedding_model="all-MiniLM-L6-v2",
    qdrant_host="qdrant",
    qdrant_port=6333,
    collection_name="documents"
)

# Initialize RAG Pipeline
rag_pipeline = RAGPipeline(
    embedding_model="all-MiniLM-L6-v2",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_model="gpt-4o-mini",
    qdrant_host="qdrant",
    qdrant_port=6333,
    collection_name="documents"
)

# Set pipelines in routers
ingestion.set_etl_pipeline(etl_pipeline)
search.set_etl_pipeline(etl_pipeline)
collection.set_etl_pipeline(etl_pipeline)
chat.set_rag_pipeline(rag_pipeline)

# Include routers
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(collection.router)
app.include_router(chat.router)


@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "RAG Document Q&A API",
        "version": "1.0.0",
        "endpoints": {
            "ingestion": "/ingest/*",
            "search": "/search",
            "chat": "/chat/*",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    try:
        collection_info = etl_pipeline.vector_store.get_collection_info()
        qdrant_status = "connected"
    except:
        collection_info = None
        qdrant_status = "disconnected"
    
    return {
        "status": "healthy",
        "etl": {
            "embedding_model": etl_pipeline.embedding_model_name,
            "embedding_dimension": etl_pipeline.get_embedding_dimension()
        },
        "rag": {
            "llm_model": rag_pipeline.openai_model_name,
            "llm_ready": rag_pipeline.client is not None
        },
        "qdrant": {
            "status": qdrant_status,
            "collection": etl_pipeline.vector_store.collection_name,
            "info": collection_info
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)