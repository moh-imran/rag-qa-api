from beanie import Document
from typing import Optional, Dict, Any
from datetime import datetime


class Job(Document):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: Optional[list] = []
    meta: Optional[Dict[str, Any]] = None
    created_at: datetime = datetime.utcnow()
    finished_at: Optional[datetime] = None

    class Settings:
        name = "ingest_jobs"
