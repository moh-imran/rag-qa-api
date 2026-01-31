import asyncio
import uuid
import logging
import functools
from typing import Dict, Any, Optional
from datetime import datetime

from beanie import PydanticObjectId

from ..models.job import Job

logger = logging.getLogger(__name__)


class IngestJobManager:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def submit(self, coro, *, job_name: str = "Unnamed Job", created_by: Optional[str] = None, job_meta: Optional[Dict[str, Any]] = None) -> str:
        """Submit a coroutine that performs the ingestion work and persist job metadata to MongoDB."""
        job_id = str(uuid.uuid4())

        # Create job document
        job_doc = Job(
            job_id=job_id,
            name=job_name,
            status='running',
            progress=0.0,
            result=None,
            error=None,
            meta=job_meta or {},
            created_by=created_by,
        )
        await job_doc.insert()

        # Wrap the coroutine to pass job_id
        partial_coro = functools.partial(coro, job_id=job_id)

        async def _runner():
            try:
                logger.info(f"Starting job {job_id}")
                res = await partial_coro()
                # Update job doc
                if not job_doc.logs:
                    job_doc.logs = []
                job_doc.logs.append(f"Job {job_id} completed successfully")
                job_doc.status = 'completed'
                job_doc.progress = 100.0
                job_doc.result = res
                job_doc.finished_at = datetime.utcnow()
                await job_doc.save()
                logger.info(f"Job {job_id} completed")
            except Exception as e:
                logger.exception(f"Job {job_id} failed: {e}")
                if not job_doc.logs:
                    job_doc.logs = []
                job_doc.logs.append(f"Job {job_id} failed: {e}")
                job_doc.status = 'failed'
                job_doc.error = str(e)
                job_doc.finished_at = datetime.utcnow()
                await job_doc.save()

        asyncio.create_task(_runner())
        return job_id

    async def delete_job_record(self, job_id: str) -> Dict[str, Any]:
        """Delete a job record from MongoDB. Vector deletion is handled by the caller."""
        job = await Job.find_one(Job.job_id == job_id)
        if not job:
            return {"status": "not_found"}

        # Delete job from MongoDB
        await job.delete()

        return {"status": "success", "message": f"Job record {job_id} deleted."}


    async def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = await Job.find_one(Job.job_id == job_id)
        if not job:
            return None
        return {
            'job_id': job.job_id,
            'name': job.name,
            'status': job.status,
            'progress': job.progress,
            'result': job.result,
            'error': job.error,
            'meta': job.meta,
            'created_by': job.created_by,
            'created_at': job.created_at,
            'finished_at': job.finished_at
        }

    async def list_jobs(self, limit: int = 50, skip: int = 0, search: Optional[str] = None):
        query = Job.find_all()
        if search:
            # Simple regex search on name, filename in meta, or job_id
            # Note: MongoDB regex queries can be slow on large datasets without indexes
            query = Job.find({
                "$or": [
                    {"job_id": {"$regex": search, "$options": "i"}},
                    {"name": {"$regex": search, "$options": "i"}},
                    {"meta.filename": {"$regex": search, "$options": "i"}},
                    {"meta.url": {"$regex": search, "$options": "i"}},
                    {"status": {"$regex": search, "$options": "i"}},
                    {"created_by": {"$regex": search, "$options": "i"}}
                ]
            })
        
        docs = await query.sort(-Job.created_at).skip(skip).limit(limit).to_list()
        return [
            {
                'job_id': d.job_id,
                'name': d.name,
                'status': d.status,
                'progress': d.progress,
                'meta': d.meta,
                'created_by': d.created_by,
                'created_at': d.created_at,
                'finished_at': d.finished_at,
            }
            for d in docs
        ]


# Singleton manager
manager = IngestJobManager()
