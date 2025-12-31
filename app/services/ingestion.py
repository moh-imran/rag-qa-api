# app/services/ingestion.py
from typing import List, Dict, Optional
from pathlib import Path
import PyPDF2
from docx import Document
import logging

logger = logging.getLogger(__name__)


class DocumentIngestion:
    """Handle document loading, chunking, embedding, and storage"""
    
    def __init__(self):
        self.supported_formats = {'.txt', '.pdf', '.docx', '.md'}
    
    async def load_docs(
        self, 
        source: str, 
        is_directory: bool = False
    ) -> List[Dict[str, str]]:
        """
        Load documents from file or directory
        
        Args:
            source: File path or directory path
            is_directory: If True, scan directory for all supported files
            
        Returns:
            List of dicts with 'content' and 'metadata' keys
            Example: [{'content': 'text...', 'metadata': {'filename': 'doc.pdf', 'type': 'pdf'}}]
        """
        documents = []
        
        if is_directory:
            documents = await self._load_from_directory(source)
        else:
            doc = await self._load_single_file(source)
            if doc:
                documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    async def _load_from_directory(self, dir_path: str) -> List[Dict[str, str]]:
        """Scan directory and load all supported files"""
        path = Path(dir_path)
        
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")
        
        documents = []
        
        # Find all supported files recursively
        for file_path in path.rglob('*'):
            if file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = await self._load_single_file(str(file_path))
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
                    continue
        
        return documents
    
    async def _load_single_file(self, file_path: str) -> Optional[Dict[str, str]]:
        """Load a single file based on its extension"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported format: {suffix}")
        
        # Route to appropriate loader
        if suffix == '.pdf':
            content = self._load_pdf(path)
        elif suffix == '.docx':
            content = self._load_docx(path)
        elif suffix in {'.txt', '.md'}:
            content = self._load_text(path)
        else:
            return None
        
        return {
            'content': content,
            'metadata': {
                'filename': path.name,
                'filepath': str(path.absolute()),
                'type': suffix[1:],
                'size_bytes': path.stat().st_size
            }
        }
    
    def _load_pdf(self, path: Path) -> str:
        """Extract text from PDF"""
        text = []
        
        with open(path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text.append(page_text)
                except Exception as e:
                    logger.warning(f"Error reading page {page_num} of {path}: {e}")
                    continue
        
        return '\n\n'.join(text)
    
    def _load_docx(self, path: Path) -> str:
        """Extract text from Word document"""
        doc = Document(path)
        
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        
        return '\n\n'.join(paragraphs)
    
    def _load_text(self, path: Path) -> str:
        """Load plain text or markdown file"""
        try:
            # Try UTF-8 first
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to latin-1
            return path.read_text(encoding='latin-1')


    def chunk_docs(
        self,
        documents: List[Dict[str, str]],
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n"
    ) -> List[Dict[str, str]]:
        """
        Split documents into smaller chunks for embedding
        
        Args:
            documents: List of documents from load_docs()
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separator: Primary separator to split on (paragraphs by default)
            
        Returns:
            List of chunks with content and metadata
            Example: [{'content': 'chunk text...', 'metadata': {..., 'chunk_id': 0}}]
        """
        all_chunks = []
        
        for doc in documents:
            content = doc['content']
            metadata = doc['metadata'].copy()
            
            # Split document into chunks
            chunks = self._split_text(
                content, 
                chunk_size, 
                chunk_overlap, 
                separator
            )
            
            # Add metadata to each chunk
            for idx, chunk_text in enumerate(chunks):
                chunk = {
                    'content': chunk_text,
                    'metadata': {
                        **metadata,
                        'chunk_id': idx,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk_text)
                    }
                }
                all_chunks.append(chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
    
    def _split_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separator: str
    ) -> List[str]:
        """
        Split text into chunks with overlap using separators
        
        Strategy:
        1. Try splitting by separator (paragraphs)
        2. If chunks still too large, split by sentences
        3. If still too large, split by character with overlap
        """
        if not text or not text.strip():
            return []
        
        # First, try splitting by the main separator (paragraphs)
        splits = text.split(separator)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            split = split.strip()
            if not split:
                continue
            
            split_size = len(split)
            
            # If single split is larger than chunk_size, need to break it down
            if split_size > chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Break down the large split
                sub_chunks = self._split_large_text(split, chunk_size, chunk_overlap)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this split exceeds chunk_size
            if current_size + split_size + len(separator) > chunk_size and current_chunk:
                # Save current chunk
                chunks.append(separator.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, chunk_overlap)
                current_chunk = [overlap_text] if overlap_text else []
                current_size = len(overlap_text) if overlap_text else 0
                
                # Check again if split fits with overlap
                if current_size + split_size + len(separator) > chunk_size:
                    # Still doesn't fit, save overlap chunk and start fresh
                    if current_chunk:
                        chunks.append(separator.join(current_chunk))
                    current_chunk = [split]
                    current_size = split_size
                    continue
            
            # Add split to current chunk
            current_chunk.append(split)
            current_size += split_size + len(separator)
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(separator.join(current_chunk))
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def _split_large_text(
        self,
        text: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """Split text that's larger than chunk_size using sentence boundaries"""
        # Try splitting by sentences first
        sentence_endings = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            # Calculate end position
            end = start + chunk_size
            
            if end >= text_length:
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            chunk_end = end
            for i in range(end, max(start, end - 200), -1):
                if any(text[i:i+2] == ending for ending in sentence_endings):
                    chunk_end = i + 1
                    break
            
            chunks.append(text[start:chunk_end].strip())
            
            # Move start position with overlap
            start = chunk_end - chunk_overlap
            if start <= 0:
                start = chunk_end
        
        return chunks
    
    def _get_overlap_text(
        self,
        current_chunk: List[str],
        overlap_size: int
    ) -> str:
        """Get the last overlap_size characters from current chunk for overlap"""
        if not current_chunk:
            return ""
        
        combined = ' '.join(current_chunk)
        
        if len(combined) <= overlap_size:
            return combined
        
        return combined[-overlap_size:]


    def _get_overlap_text(
        self,
        current_chunk: List[str],
        overlap_size: int
    ) -> str:
        """Get the last overlap_size characters from current chunk for overlap"""
        if not current_chunk:
            return ""
        
        combined = ' '.join(current_chunk)
        
        if len(combined) <= overlap_size:
            return combined
        
        return combined[-overlap_size:]


    def embed_docs(
        self,
        chunks: List[Dict[str, str]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[Dict[str, any]]:
        """
        Generate embeddings for document chunks
        
        Args:
            chunks: List of chunks from chunk_docs()
            batch_size: Number of chunks to embed at once (for efficiency)
            show_progress: Show progress bar during embedding
            
        Returns:
            List of chunks with embeddings added
            Example: [{'content': '...', 'metadata': {...}, 'embedding': [0.1, 0.2, ...]}]
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return []
        
        # Extract text content from chunks
        texts = [chunk['content'] for chunk in chunks]
        
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        # Generate embeddings in batches for efficiency
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = {
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'embedding': embedding.tolist()  # Convert numpy array to list for JSON serialization
            }
            embedded_chunks.append(embedded_chunk)
        
        logger.info(f"✅ Generated {len(embedded_chunks)} embeddings (dim: {len(embeddings[0])})")
        return embedded_chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        return self.embedding_model.get_sentence_embedding_dimension()