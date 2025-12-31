from pathlib import Path
from typing import List, Dict, Any, Optional
import PyPDF2
from docx import Document
from .base import BaseDataSource
import logging

logger = logging.getLogger(__name__)


class FileSource(BaseDataSource):
    """Extract documents from local files"""
    
    def __init__(self):
        super().__init__("FileSource")
        self.supported_formats = {'.txt', '.pdf', '.docx', '.md'}
    
    async def extract(
        self, 
        path: str = None,
        file_path: str = None,
        directory_path: str = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract documents from files or directory
        
        Args:
            path: Single file or directory path (auto-detect)
            file_path: Explicit single file path
            directory_path: Explicit directory path
        """
        if path:
            # Auto-detect if file or directory
            p = Path(path)
            if p.is_file():
                file_path = path
            elif p.is_dir():
                directory_path = path
        
        if file_path:
            return await self._extract_single_file(file_path)
        elif directory_path:
            return await self._extract_directory(directory_path)
        else:
            raise ValueError("Must provide either 'path', 'file_path', or 'directory_path'")
    
    async def _extract_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Scan directory and extract all supported files"""
        path = Path(dir_path)
        
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Directory not found: {dir_path}")
        
        documents = []
        
        for file_path in path.rglob('*'):
            if file_path.suffix.lower() in self.supported_formats:
                try:
                    docs = await self._extract_single_file(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"Error extracting {file_path}: {e}")
                    continue
        
        logger.info(f"Extracted {len(documents)} documents from directory")
        return documents
    
    async def _extract_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract content from a single file"""
        path = Path(file_path).resolve()
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix not in self.supported_formats:
            raise ValueError(f"Unsupported format: {suffix}")
        
        # Route to appropriate extractor
        if suffix == '.pdf':
            content = self._extract_pdf(path)
        elif suffix == '.docx':
            content = self._extract_docx(path)
        elif suffix in {'.txt', '.md'}:
            content = self._extract_text(path)
        else:
            return []
        
        return [{
            'content': content,
            'metadata': {
                'source': self.source_name,
                'filename': path.name,
                'filepath': str(path.absolute()),
                'type': suffix[1:],
                'size_bytes': path.stat().st_size
            }
        }]
    
    def _extract_pdf(self, path: Path) -> str:
        """Extract text from PDF"""
        text = []
        with open(path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text.append(page_text)
                except Exception as e:
                    logger.warning(f"Error reading page: {e}")
        return '\n\n'.join(text)
    
    def _extract_docx(self, path: Path) -> str:
        """Extract text from Word document"""
        doc = Document(path)
        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
        return '\n\n'.join(paragraphs)
    
    def _extract_text(self, path: Path) -> str:
        """Load plain text or markdown file"""
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            return path.read_text(encoding='latin-1')
