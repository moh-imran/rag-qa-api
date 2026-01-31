from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from ..services.data_sources.notion_source import NotionSource
from ..services.data_sources.database_source import DatabaseSource
from ..services.data_sources.confluence_source import ConfluenceSource
from ..services.data_sources.sharepoint_source import SharePointSource
from ..services.ingest_jobs import manager as job_manager
from ..services.ingest_jobs import manager as job_manager
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


class IngestRunRequest(BaseModel):
    source_type: str = Field(..., description="Type of data source e.g. file, web, git, notion, database")
    source_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters specific to the source_type")
    chunk_size: int = Field(1000, description="Size of text chunks")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    batch_size: int = Field(32, description="Batch size for embedding")
    store_in_qdrant: bool = Field(True, description="Store vectors in Qdrant")


@router.post('/run', response_model=IngestResponse)
async def ingest_run(request: IngestRunRequest):
    """Generic ingestion endpoint that delegates to the ETL pipeline for any registered source."""
    try:
        # Merge chunking/storage params into source params
        params = dict(request.source_params or {})
        params.update({
            'chunk_size': request.chunk_size,
            'chunk_overlap': request.chunk_overlap,
            'batch_size': request.batch_size,
            'store_in_qdrant': request.store_in_qdrant
        })



        # If the request requires dynamic data source registration (Notion, Database), register it
        if request.source_type == 'notion':
            api_key = params.pop('api_key', None)
            if not api_key:
                raise HTTPException(status_code=400, detail="Missing 'api_key' for Notion source")
            notion_src = NotionSource(api_key=api_key)
            etl_pipeline.add_data_source('notion', notion_src)

        if request.source_type == 'database':
            # Expected params: host, port, database, user, password, db_type
            db_type = params.pop('db_type', 'postgresql')
            host = params.pop('host', None)
            port = params.pop('port', 5432)
            database = params.pop('database', None)
            user = params.pop('user', None)
            password = params.pop('password', None)
            collection = params.get('collection') or params.get('table')  # For MongoDB

            # Auto-detect MongoDB from connection string
            if host and host.startswith('mongodb://'):
                db_type = 'mongodb'
                if not database:
                    database = 'test'
                if not collection:
                    raise HTTPException(status_code=400, detail="Missing 'collection' (or 'table') name for MongoDB")
            elif not database or not host:
                raise HTTPException(status_code=400, detail="Missing database connection parameters (host, database)")

            db_src = DatabaseSource(db_type=db_type, host=host, port=port, database=database, user=user, password=password)
            etl_pipeline.add_data_source('database', db_src)

        result = await etl_pipeline.run(
            source_type=request.source_type,
            **params
        )

        return IngestResponse(
            status="success",
            message=f"Successfully processed source {request.source_type}",
            total_documents=result['total_documents'],
            total_chunks=result['total_chunks'],
            embedding_dimension=result['embedding_dimension'],
            stored_vectors=result['storage']['stored'] if result.get('storage') else None
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



class IngestSubmitRequest(BaseModel):
    name: Optional[str] = Field(None, description="User-friendly job name")
    source_type: str
    source_params: Dict[str, Any] = Field(default_factory=dict)
    chunk_size: int = Field(1000)
    chunk_overlap: int = Field(200)
    batch_size: int = Field(32)
    store_in_qdrant: bool = Field(True)


@router.post('/submit')
async def ingest_submit(request: IngestSubmitRequest):
    """Submit an asynchronous ingest job. Returns `job_id` immediately."""
    try:
        params = dict(request.source_params or {})



        # dynamic registration for credentials
        if request.source_type == 'notion':
            api_key = params.pop('api_key', None)
            if not api_key:
                raise HTTPException(status_code=400, detail="Missing 'api_key' for Notion source")
            etl_pipeline.add_data_source('notion', NotionSource(api_key=api_key))

        if request.source_type == 'database':
            db_type = params.pop('db_type', 'postgresql')
            host = params.pop('host', None)
            port = params.pop('port', 5432)
            database = params.pop('database', None)
            user = params.pop('user', None)
            password = params.pop('password', None)
            collection = params.get('collection') or params.get('table')  # For MongoDB

            # Auto-detect MongoDB from connection string
            if host and host.startswith('mongodb://'):
                db_type = 'mongodb'
                if not database:
                    # Try to extract database from connection string or use default
                    database = 'test'
                if not collection:
                    raise HTTPException(status_code=400, detail="Missing 'collection' (or 'table') name for MongoDB")
            elif not database or not host:
                raise HTTPException(status_code=400, detail="Missing database connection parameters (host, database)")

            etl_pipeline.add_data_source('database', DatabaseSource(db_type=db_type, host=host, port=port, database=database, user=user, password=password))

        if request.source_type == 'confluence':
            base_url = params.pop('base_url', None)
            email = params.pop('email', None)
            api_token = params.pop('api_token', None)
            if not base_url or not email or not api_token:
                raise HTTPException(status_code=400, detail="Missing Confluence credentials (base_url, email, api_token)")
            etl_pipeline.add_data_source('confluence', ConfluenceSource(base_url=base_url, email=email, api_token=api_token))

        if request.source_type == 'sharepoint':
            access_token = params.pop('access_token', None)
            site_id = params.pop('site_id', None)
            if not access_token or not site_id:
                raise HTTPException(status_code=400, detail="Missing SharePoint parameters (access_token, site_id)")
            etl_pipeline.add_data_source('sharepoint', SharePointSource(access_token=access_token, site_id=site_id))

        # wrapper coroutine to run ETL
        async def _run_job(job_id: str):
            return await etl_pipeline.run(
                source_type=request.source_type,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                batch_size=request.batch_size,
                store_in_qdrant=request.store_in_qdrant,
                job_id=job_id,
                **params
            )

        job_id = await job_manager.submit(
            _run_job,
            job_name=request.name,
            created_by=None,  # TODO: Extract from auth context when available
            job_meta={"source_type": request.source_type, "params": request.source_params}
        )
        return {"job_id": job_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/status/{job_id}')
async def ingest_status(job_id: str):
    status = await job_manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@router.get('/jobs')
async def ingest_jobs_list(limit: int = 50, skip: int = 0, search: Optional[str] = None):
    jobs = await job_manager.list_jobs(limit=limit, skip=skip, search=search)
    return {"jobs": jobs}


@router.get('/jobs/{job_id}/logs')
async def ingest_job_logs(job_id: str):
    status = await job_manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Job not found")
    # status contains 'result' and we added 'logs' via Job document
    return {"job_id": job_id, "logs": status.get('result') and status.get('result').get('logs') or status.get('logs') or []}

@router.delete('/jobs/{job_id}')
async def delete_ingest_job(job_id: str):
    """Delete an ingestion job and its associated data."""
    try:
        # 1. Delete vectors from Qdrant
        try:
            etl_pipeline.vector_store.delete_by_filter(filter_dict={'job_id': job_id})
        except Exception as e:
            # We log error but proceed to delete the job record so we don't end up with orphan records
            # that can't be deleted.
            print(f"Warning: Failed to delete vectors for job {job_id}: {e}")

        # 2. Delete job record from MongoDB
        result = await job_manager.delete_job_record(job_id)
        
        if result.get("status") == "not_found":
             raise HTTPException(status_code=404, detail="Job not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def ingest_upload(
    file: UploadFile = File(...),
    name: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 32,
    store_in_qdrant: bool = True
):
    """Upload and ingest a single file - creates a job for tracking"""
    try:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in {'.txt', '.pdf', '.docx', '.md'}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")

        # Save file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        filename = file.filename
        
        # Generate job name if not provided
        if not name:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            name = f"Upload: {filename} - {timestamp}"

        # Create async job for processing
        async def _run_upload_job(job_id: str): # Modified to accept job_id
            try:
                result = await etl_pipeline.run(
                    source_type='file',
                    file_path=tmp_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    batch_size=batch_size,
                    store_in_qdrant=store_in_qdrant,
                    job_id=job_id # Passing job_id
                )
                return {
                    "status": "success",
                    "message": f"Successfully processed {filename}",
                    "filename": filename,
                    "total_documents": result['total_documents'],
                    "total_chunks": result['total_chunks'],
                    "embedding_dimension": result['embedding_dimension'],
                    "stored_vectors": result['storage']['stored'] if result.get('storage') else None
                }
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        job_id = await job_manager.submit(
            _run_upload_job,
            job_name=name,
            created_by=None,  # TODO: Extract from auth context when available
            job_meta={"source_type": "file", "filename": filename}
        )

        return {
            "status": "processing",
            "job_id": job_id,
            "message": f"File {filename} uploaded and processing started",
            "filename": filename
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))