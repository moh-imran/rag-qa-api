from typing import List, Dict, Any, Optional
import tempfile
import shutil
from pathlib import Path
from git import Repo
from .base import BaseDataSource
import logging

logger = logging.getLogger(__name__)


class GitSource(BaseDataSource):
    """Extract documents from Git repositories"""
    
    def __init__(self):
        super().__init__("GitSource")
        self.supported_extensions = {'.md', '.txt', '.py', '.js', '.java', '.go', '.rs', '.cpp', '.c', '.h'}
    
    async def extract(
        self,
        repo_url: str,
        branch: str = "main",
        include_code: bool = True,
        file_extensions: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Extract documents from a Git repository
        
        Args:
            repo_url: Git repository URL (HTTPS or SSH)
            branch: Branch to clone (default: main)
            include_code: Include code files (default: True)
            file_extensions: Custom list of file extensions to include
        
        Returns:
            List of documents with content and metadata
        """
        if file_extensions:
            extensions = set(ext if ext.startswith('.') else f'.{ext}' for ext in file_extensions)
        else:
            extensions = self.supported_extensions
        
        documents = []
        temp_dir = None
        
        try:
            # Clone to temporary directory
            temp_dir = tempfile.mkdtemp()
            logger.info(f"Cloning repository {repo_url} to {temp_dir}")
            
            repo = Repo.clone_from(repo_url, temp_dir, branch=branch, depth=1)
            
            # Walk through repository files
            repo_path = Path(temp_dir)
            
            for file_path in repo_path.rglob('*'):
                if file_path.is_file():
                    # Skip hidden files and directories
                    if any(part.startswith('.') for part in file_path.parts):
                        continue
                    
                    # Check extension
                    if file_path.suffix.lower() not in extensions:
                        continue
                    
                    try:
                        # Read file content
                        content = file_path.read_text(encoding='utf-8')
                        
                        # Get relative path
                        rel_path = file_path.relative_to(repo_path)
                        
                        documents.append({
                            'content': content,
                            'metadata': {
                                'source': self.source_name,
                                'repo_url': repo_url,
                                'branch': branch,
                                'filepath': str(rel_path),
                                'filename': file_path.name,
                                'type': file_path.suffix[1:] if file_path.suffix else 'unknown',
                                'size_bytes': file_path.stat().st_size
                            }
                        })
                    
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
                        continue
            
            logger.info(f"Extracted {len(documents)} documents from Git repository")
        
        except Exception as e:
            logger.error(f"Error cloning repository {repo_url}: {e}")
            raise
        
        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        
        return documents
