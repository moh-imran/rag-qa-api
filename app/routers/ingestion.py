from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import tempfile
import os

router = APIRouter(prefix="/ingest", tags=["ingestion"])

etl_pipeline = None

def set_etl_pipeline(pipeline):
    global etl_pipeline
    etl_pipeline = pipeline


class IngestFileRequest(BaseModel):
    file_path: str = Field(..., description="Path to file or directory")
    chunk_size: int = Field(1000, description="Size of text chunks")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    batch_size: int = Field(32, description="Batch size for embedding")
    store_in_qdrant: bool = Field(True, description="Store vectors in Qdrant")


class IngestResponse(BaseModel):
    status: str
    message: str
    total_documents: int
    total_chunks: int
    embedding_dimension: int
    stored_vectors: Optional[int] = None


@router.post("/file", response_model=IngestResponse)
async def ingest_file(request: IngestFileRequest):
    """Ingest documents from local file or directory"""
    try:
        result = await etl_pipeline.run(
            source_type='file',
            path=request.file_path,
            file_path=request.file_path,
            directory_path=request.file_path,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            batch_size=request.batch_size,
            store_in_qdrant=request.store_in_qdrant
        )
        
        return IngestResponse(
            status="success",
            message=f"Successfully processed documents from {request.file_path}",
            total_documents=result['total_documents'],
            total_chunks=result['total_chunks'],
            embedding_dimension=result['embedding_dimension'],
            stored_vectors=result['storage']['stored'] if result.get('storage') else None
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 32,
    store_in_qdrant: bool = True
):
    """Upload and ingest a single file"""
    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in {'.txt', '.pdf', '.docx', '.md'}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            result = await etl_pipeline.run(
                source_type='file',
                file_path=tmp_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                batch_size=batch_size,
                store_in_qdrant=store_in_qdrant
            )
            
            return {
                "status": "success",
                "message": f"Successfully processed {file.filename}",
                "filename": file.filename,
                "total_documents": result['total_documents'],
                "total_chunks": result['total_chunks'],
                "embedding_dimension": result['embedding_dimension'],
                "stored_vectors": result['storage']['stored'] if result.get('storage') else None
            }
        finally:
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))