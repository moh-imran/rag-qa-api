import asyncio
import uuid
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from beanie import PydanticObjectId

from ..models.job import Job

logger = logging.getLogger(__name__)


class IngestJobManager:
    def __init__(self):
        self._lock = asyncio.Lock()

    async def submit(self, coro, *, job_meta: Optional[Dict[str, Any]] = None) -> str:
        """Submit a coroutine that performs the ingestion work and persist job metadata to MongoDB."""
        job_id = str(uuid.uuid4())

        # Create job document
        job_doc = Job(
            job_id=job_id,
            status='running',
            progress=0.0,
            result=None,
            error=None,
            meta=job_meta or {},
        )
        await job_doc.insert()

        async def _runner():
            try:
                logger.info(f"Starting job {job_id}")
                res = await coro()
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

    async def get_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = await Job.find_one(Job.job_id == job_id)
        if not job:
            return None
        return {
            'job_id': job.job_id,
            'status': job.status,
            'progress': job.progress,
            'result': job.result,
            'error': job.error,
            'meta': job.meta,
            'created_at': job.created_at,
            'finished_at': job.finished_at
        }

    async def list_jobs(self, limit: int = 50):
        docs = await Job.find_all().sort(-Job.created_at).limit(limit).to_list()
        return [
            {
                'job_id': d.job_id,
                'status': d.status,
                'progress': d.progress,
                'meta': d.meta,
                'created_at': d.created_at,
                'finished_at': d.finished_at,
            }
            for d in docs
        ]


# Singleton manager
manager = IngestJobManager()
