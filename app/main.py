from fastapi import FastAPI
import logging
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.services.etl_pipeline import ETLPipeline
from app.services.rag_pipeline import RAGPipeline
from app.routers import ingestion, search, collection, chat, evaluation
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from app.models.job import Job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Q&A API",
    description="Complete RAG system with document ingestion and AI-powered Q&A",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    # Initialize MongoDB / Beanie for job persistence (optional - if MONGODB_URL provided)
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://mongodb:27017/rag_chat")
    if mongodb_url:
        # Process MONGODB_URL to escape username and password if present
        if "://" in mongodb_url and "@" in mongodb_url:
            try:
                scheme, rest = mongodb_url.split("://", 1)
                userinfo, host_rest = rest.rsplit("@", 1)
                if ":" in userinfo:
                    username, password = userinfo.split(":", 1)
                    mongodb_url = f"{scheme}://{quote_plus(username)}:{quote_plus(password)}@{host_rest}"
                else:
                    mongodb_url = f"{scheme}://{quote_plus(userinfo)}@{host_rest}"
            except Exception as e:
                logger.warning(f"Failed to parse MONGODB_URL for escaping: {e}")

        client = AsyncIOMotorClient(mongodb_url)
        await init_beanie(database=client.get_default_database(), document_models=[Job])
        logging.getLogger(__name__).info("Initialized Beanie with MongoDB for job persistence")

# Initialize ETL Pipeline
embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
openai_api_key = os.getenv("OPENAI_API_KEY")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

etl_pipeline = ETLPipeline(
    embedding_provider=embedding_provider,
    embedding_model=embedding_model,
    openai_api_key=openai_api_key,
    qdrant_host="qdrant",
    qdrant_port=6333,
    qdrant_api_key=qdrant_api_key,
    collection_name="documents"
)

# Initialize RAG Pipeline
rag_pipeline = RAGPipeline(
    embedding_provider=embedding_provider,
    embedding_model=embedding_model,
    cross_encoder_model=os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    openai_api_key=openai_api_key,
    openai_model="gpt-4o-mini",
    qdrant_host="qdrant",
    qdrant_port=6333,
    qdrant_api_key=qdrant_api_key,
    collection_name="documents"
)

# Set pipelines in routers
ingestion.set_etl_pipeline(etl_pipeline)
search.set_etl_pipeline(etl_pipeline)
collection.set_etl_pipeline(etl_pipeline)
chat.set_rag_pipeline(rag_pipeline)
evaluation.set_qa_evaluator(rag_pipeline.client)

# Register routers
app.include_router(ingestion.router)
app.include_router(search.router)
app.include_router(collection.router)
app.include_router(chat.router)
app.include_router(evaluation.router)


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