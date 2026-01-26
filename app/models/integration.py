from beanie import Document
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import Field


class Integration(Document):
    name: str
    type: str  # e.g., 'confluence', 'sharepoint'
    config: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Settings:
        name = "integrations"
