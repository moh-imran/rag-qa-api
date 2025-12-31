from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/collection", tags=["collection"])

etl_pipeline = None

def set_etl_pipeline(pipeline):
    global etl_pipeline
    etl_pipeline = pipeline


@router.get("/info")
async def get_collection_info():
    """Get information about the Qdrant collection"""
    try:
        return etl_pipeline.vector_store.get_collection_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create")
async def create_collection(recreate: bool = False):
    """Create or recreate the Qdrant collection"""
    try:
        vector_size = etl_pipeline.get_embedding_dimension()
        etl_pipeline.vector_store.create_collection(
            vector_size=vector_size,
            recreate=recreate
        )
        return {
            "status": "success",
            "message": "Collection created",
            "vector_size": vector_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("")
async def delete_collection():
    """Delete the entire collection (use with caution!)"""
    try:
        etl_pipeline.vector_store.delete_collection()
        return {"status": "success", "message": "Collection deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))