from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text or list of texts
            **kwargs: Provider-specific parameters
            
        Returns:
            numpy array of embeddings
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass


class HuggingFaceProvider(BaseEmbeddingProvider):
    """HuggingFace SentenceTransformer embedding provider"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded HuggingFace model: {model_name}")
    
    def embed(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate embeddings using SentenceTransformer"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            **kwargs
        )
        
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()


class OpenAIProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider"""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: str = None
    ):
        super().__init__(model_name)
        from openai import OpenAI
        
        if not api_key:
            raise ValueError("OpenAI API key required for OpenAIProvider")
        
        self.client = OpenAI(api_key=api_key)
        self._dimension = None
        
        # Model dimension mapping
        self.dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536
        }
        
        logger.info(f"Initialized OpenAI embeddings: {model_name}")
    
    def embed(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Generate embeddings using OpenAI API"""
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            # Use known dimensions or test with a sample
            if self.model_name in self.dimension_map:
                self._dimension = self.dimension_map[self.model_name]
            else:
                # Test with sample text
                sample_embedding = self.embed("test")
                self._dimension = sample_embedding.shape[1] if len(sample_embedding.shape) > 1 else len(sample_embedding)
        
        return self._dimension


def create_embedding_provider(
    provider_type: str,
    model_name: str,
    api_key: str = None
) -> BaseEmbeddingProvider:
    """
    Factory function to create embedding provider
    
    Args:
        provider_type: "huggingface" or "openai"
        model_name: Model name for the provider
        api_key: API key (required for OpenAI)
    
    Returns:
        Embedding provider instance
    """
    provider_type = provider_type.lower()
    
    if provider_type == "huggingface":
        return HuggingFaceProvider(model_name)
    elif provider_type == "openai":
        return OpenAIProvider(model_name, api_key)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Supported: huggingface, openai")
