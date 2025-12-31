from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import logging
from .data_sources.base import BaseDataSource
from .data_sources.file_source import FileSource
from ..models.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


class ETLPipeline:
    """Orchestrate Extract-Transform-Load pipeline for RAG"""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "documents"
    ):
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.data_sources = {
            'file': FileSource()            
        }
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        
        logger.info("ETL Pipeline initialized")
    
    def _load_embedding_model(self):
        """Lazy load embedding model"""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Model loaded. Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")
    
    async def run(
        self,
        source_type: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 32,
        store_in_qdrant: bool = True,
        **source_params
    ) -> Dict[str, Any]:
        """
        Run complete ETL pipeline
        
        Args:
            source_type: Type of data source ('file', 'api', etc.)
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            batch_size: Batch size for embedding
            store_in_qdrant: If True, store vectors in Qdrant
            **source_params: Parameters specific to the data source
            
        Returns:
            Dictionary with pipeline results and statistics
        """
        logger.info(f"Starting ETL pipeline with source: {source_type}")
        
        # EXTRACT
        documents = await self.extract(source_type, **source_params)
        
        # TRANSFORM
        chunks = self.chunk_documents(documents, chunk_size, chunk_overlap)
        embedded_chunks = self.embed_chunks(chunks, batch_size)
        
        # LOAD (optional)
        storage_result = None
        if store_in_qdrant:
            storage_result = self.store_vectors(embedded_chunks)
        
        result = {
            "status": "success",
            "total_documents": len(documents),
            "total_chunks": len(embedded_chunks),
            "embedding_dimension": self.get_embedding_dimension(),
            "embedded_chunks": embedded_chunks if not store_in_qdrant else None,
            "storage": storage_result
        }
        
        logger.info(f"✅ ETL pipeline complete: {len(embedded_chunks)} embedded chunks")
        return result
    
    async def extract(self, source_type: str, **params) -> List[Dict[str, Any]]:
        """
        EXTRACT phase: Get data from source
        """
        if source_type not in self.data_sources:
            raise ValueError(f"Unknown source type: {source_type}. Available: {list(self.data_sources.keys())}")
        
        source = self.data_sources[source_type]
        documents = await source.extract(**params)
        
        logger.info(f"📥 Extracted {len(documents)} documents from {source_type}")
        return documents
    
    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        TRANSFORM phase: Chunk documents
        """
        all_chunks = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc['metadata'].copy()
            
            # Split into chunks
            text_chunks = self._split_text(content, chunk_size, chunk_overlap)
            
            # Add metadata
            for idx, chunk_text in enumerate(text_chunks):
                chunk = {
                    'content': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_id': idx,
                        'total_chunks': len(text_chunks),
                        'chunk_size': len(chunk_text)
                    }
                }
                all_chunks.append(chunk)
        
        logger.info(f"✂️  Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        TRANSFORM phase: Generate embeddings
        """
        if not chunks:
            return []
        
        self._load_embedding_model()
        
        texts = [chunk['content'] for chunk in chunks]
        
        logger.info(f"🔮 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = {
                **chunk,
                'embedding': embedding.tolist()
            }
            embedded_chunks.append(embedded_chunk)
        
        logger.info(f"✅ Generated {len(embedded_chunks)} embeddings")
        return embedded_chunks
    
    def _split_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separator: str = "\n\n"
    ) -> List[str]:
        """Split text into chunks (simplified version)"""
        if not text or not text.strip():
            return []
        
        splits = text.split(separator)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            split = split.strip()
            if not split:
                continue
            
            split_size = len(split)
            
            if current_size + split_size > chunk_size and current_chunk:
                chunks.append(separator.join(current_chunk))
                
                # Create overlap
                overlap_text = separator.join(current_chunk)[-chunk_overlap:]
                current_chunk = [overlap_text] if overlap_text else []
                current_size = len(overlap_text) if overlap_text else 0
            
            current_chunk.append(split)
            current_size += split_size
        
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def store_vectors(
        self,
        embedded_chunks: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        LOAD phase: Store vectors in Qdrant
        """
        logger.info(f"💾 Storing {len(embedded_chunks)} vectors in Qdrant...")
        
        # Ensure collection exists
        self._ensure_collection_exists()
        
        # Store vectors
        result = self.vector_store.store_vectors(embedded_chunks, batch_size)
        
        logger.info(f"✅ Stored {result['stored']} vectors")
        return result
    
    def _ensure_collection_exists(self):
        """Ensure Qdrant collection exists with correct vector size"""
        try:
            self.vector_store.get_collection_info()
            logger.info(f"Collection '{self.vector_store.collection_name}' exists")
        except:
            # Collection doesn't exist, create it
            logger.info(f"Creating collection '{self.vector_store.collection_name}'")
            vector_size = self.get_embedding_dimension()
            self.vector_store.create_collection(vector_size=vector_size)
    
    def add_data_source(self, name: str, source: BaseDataSource):
        """Register a new data source"""
        self.data_sources[name] = source
        logger.info(f"Registered new data source: {name}")
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        self._load_embedding_model()
        return self.embedding_model.get_sentence_embedding_dimension()