from typing import List, Dict, Any, Optional
from sentence_transformers import util
from fastembed import SparseTextEmbedding
import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from .data_sources.base import BaseDataSource
from .data_sources.file_source import FileSource
from .data_sources.web_source import WebSource
from .data_sources.git_source import GitSource
from .data_sources.notion_source import NotionSource
from .data_sources.database_source import DatabaseSource
from .embedding_providers import BaseEmbeddingProvider, create_embedding_provider
from ..models.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

# Thread pool for CPU-bound operations (embedding, chunking)
_executor = ThreadPoolExecutor(max_workers=2)


class ETLPipeline:
    """Orchestrate Extract-Transform-Load pipeline for RAG"""
    
    def __init__(
        self,
        embedding_provider: str = "huggingface",
        embedding_model: str = "all-MiniLM-L6-v2",
        openai_api_key: str = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "documents"
    ):
        self.embedding_provider_type = embedding_provider
        self.embedding_model_name = embedding_model
        self.embedding_provider = None
        self.openai_api_key = openai_api_key
        self.data_sources = {
            'file': FileSource(),
            'web': WebSource(),
            'git': GitSource()
            # Note: Notion and Database sources require credentials, 
            # so they should be registered dynamically via add_data_source()
        }
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name
        )
        
        logger.info("ETL Pipeline initialized")
    
    async def run_stream(
        self,
        source_type: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        batch_size: int = 32,
        store_in_qdrant: bool = True,
        job_id: Optional[str] = None,
        **source_params
    ):
        """
        Run ETL pipeline in streaming mode. 
        Highly memory efficient for large documents.
        """
        logger.info(f"Starting STREAMING ETL pipeline for: {source_type}")
        loop = asyncio.get_event_loop()
        
        # 1. Start Extraction Stream
        if source_type not in self.data_sources:
             raise ValueError(f"Unknown source type: {source_type}")
        
        data_source = self.data_sources[source_type]
        doc_stream = data_source.extract_stream(**source_params)
        
        chunk_buffer = []
        total_docs = 0
        total_chunks = 0

        async for doc in doc_stream:
            total_docs += 1
            
            # 2. Chunk (CPU bound)
            # Run chunking in thread pool to not block event loop
            content = doc.get('content', '')
            if not content:
                continue
                
            doc_chunks = await loop.run_in_executor(
                _executor,
                self.chunk_documents,
                [doc], chunk_size, chunk_overlap
            )
            
            for chunk in doc_chunks:
                chunk_buffer.append(chunk)
                total_chunks += 1
                
                # 3. If buffer reaches batch_size, Embed & Store
                if len(chunk_buffer) >= batch_size:
                    await self._process_batch(chunk_buffer, batch_size, store_in_qdrant, job_id)
                    chunk_buffer = []

        # 4. Flush remaining chunks
        if chunk_buffer:
            await self._process_batch(chunk_buffer, batch_size, store_in_qdrant, job_id)

        logger.info(f"Streaming ETL complete. Processed {total_docs} docs into {total_chunks} chunks.")
        return {
            "status": "success",
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "embedding_dimension": self.get_embedding_dimension(),
            "storage": {"stored": total_chunks} if store_in_qdrant else None
        }

    async def _process_batch(self, chunks, batch_size, store_in_qdrant, job_id):
        """Helper to embed and store a batch of chunks"""
        loop = asyncio.get_event_loop()
        
        # Embed
        embedded_chunks = await loop.run_in_executor(
            _executor,
            self.embed_chunks,
            chunks, batch_size
        )
        
        # Store
        if store_in_qdrant:
            await loop.run_in_executor(
                _executor,
                self.store_vectors,
                embedded_chunks,
                batch_size,
                job_id
            )

    async def run(self, **kwargs) -> Dict[str, Any]:
        """Backward compatibility: runs streaming ETL"""
        return await self.run_stream(**kwargs)

    def _load_embedding_model(self):
        """Lazy load embedding models (Dense + Sparse)"""
        # Dense
        if self.embedding_provider is None:
            logger.info(f"Loading embedding provider: {self.embedding_provider_type}")
            self.embedding_provider = create_embedding_provider(
                provider_type=self.embedding_provider_type,
                model_name=self.embedding_model_name,
                api_key=self.openai_api_key
            )
            logger.info(f"Embedding provider loaded. Dimension: {self.embedding_provider.get_dimension()}")
            
        # Sparse
        if not hasattr(self, 'sparse_embedding_model') or self.sparse_embedding_model is None:
            logger.info("Loading sparse embedding model: prithivida/Splade_PP_en_v1")
            self.sparse_embedding_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
            logger.info("Sparse model loaded")

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        TRANSFORM phase: Generate embeddings (Dense + Sparse)
        """
        if not chunks:
            return []
        
        self._load_embedding_model()
        
        texts = [chunk['content'] for chunk in chunks]
        
        logger.info(f"🔮 Generating embeddings for {len(texts)} chunks...")
        
        # 1. Generate Dense Embeddings
        dense_embeddings = self.embedding_provider.embed(
            texts,
            batch_size=batch_size,
            show_progress_bar=True
        )
        
        # 2. Generate Sparse Embeddings
        logger.info("Generating sparse embeddings...")
        # yield from returns generator, convert to list
        sparse_embeddings = list(self.sparse_embedding_model.embed(texts, batch_size=batch_size))
        
        # Add embeddings to chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            # Sparse embedding object from fastembed usually has indices/values
            # FastEmbed returns SparseEmbedding(values=..., indices=...) objects or dicts depending on version
            # Assuming it conforms to expected structure or we extract it.
            # sparse_embeddings[i] is typically a SparseEmbedding object.
            
            # Extract usable dict for Qdrant
            # sparse_vector = sparse_embeddings[i]
            # sparse_dict = {"indices": sparse_vector.indices.tolist(), "values": sparse_vector.values.tolist()}
            
            # Actually fastembed yields objects with .indices and .values (numpy arrays)
            sparse_item = sparse_embeddings[i]
            sparse_dict = {
                "indices": sparse_item.indices.tolist(), 
                "values": sparse_item.values.tolist()
            }
            
            embedded_chunk = {
                **chunk,
                'embedding': dense_embeddings[i].tolist(),
                'sparse_embedding': sparse_dict
            }
            embedded_chunks.append(embedded_chunk)
        
        logger.info(f"✅ Generated {len(embedded_chunks)} hybrid embeddings")
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
        batch_size: int = 100,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        LOAD phase: Store vectors in Qdrant
        """
        logger.info(f"💾 Storing {len(embedded_chunks)} vectors in Qdrant...")
        
        # Ensure collection exists
        self._ensure_collection_exists()
        
        # Store vectors using the dedicated service
        result = self.vector_store.store_vectors(embedded_chunks, batch_size, job_id=job_id)
        
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
        return self.embedding_provider.get_dimension()

    async def extract(self, source_type: str, **source_params) -> List[Dict[str, Any]]:
        """
        EXTRACT phase: Get documents from the specified data source
        """
        if source_type not in self.data_sources:
            raise ValueError(f"Unknown source type: {source_type}. Available: {list(self.data_sources.keys())}")

        data_source = self.data_sources[source_type]
        logger.info(f"📥 Extracting from source: {source_type}")

        documents = await data_source.extract(**source_params)
        logger.info(f"✅ Extracted {len(documents)} documents")
        return documents

    def chunk_documents(
        self,
        documents: List[Dict[str, Any]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "fixed"
    ) -> List[Dict[str, Any]]:
        """
        TRANSFORM phase: Chunk documents into smaller pieces
        """
        logger.info(f"📄 Chunking {len(documents)} documents with strategy: {chunking_strategy}")

        all_chunks = []
        for doc in documents:
            content = doc.get('content', '')
            if not content:
                continue

            text_chunks = self._split_text(content, chunk_size, chunk_overlap)

            for i, chunk_text in enumerate(text_chunks):
                chunk = {
                    'content': chunk_text,
                    'metadata': {
                        **doc.get('metadata', {}),
                        'chunk_index': i,
                        'source_doc': doc.get('metadata', {}).get('source', 'unknown')
                    }
                }
                all_chunks.append(chunk)

        logger.info(f"✅ Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks