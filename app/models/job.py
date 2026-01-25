from beanie import Document
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import Field


class Job(Document):
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    meta: Optional[Dict[str, Any]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None

    class Settings:
        name = "ingest_jobs"
