from abc import ABC, abstractmethod
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseDataSource(ABC):
    """Abstract base class for all data sources"""
    
    def __init__(self, source_name: str):
        self.source_name = source_name
        logger.info(f"Initialized {source_name} data source")
    
    @abstractmethod
    async def extract(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Extract data from source
        
        Returns:
            List of documents with 'content' and 'metadata'
            [{'content': 'text...', 'metadata': {...}}]
        """
        pass
    
    def validate_output(self, documents: List[Dict[str, Any]]) -> bool:
        """Validate extracted documents have required fields"""
        for doc in documents:
            if 'content' not in doc or 'metadata' not in doc:
                raise ValueError(f"Invalid document format from {self.source_name}")
        return True