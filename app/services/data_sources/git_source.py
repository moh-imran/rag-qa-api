from typing import List, Dict, Any, Optional
import tempfile
import shutil
from pathlib import Path
from git import Repo, GitCommandError
from .base import BaseDataSource
import logging
import os

logger = logging.getLogger(__name__)


class GitSource(BaseDataSource):
    """Extract documents from Git repositories"""

    def __init__(self):
        super().__init__("GitSource")
        self.supported_extensions = {
            '.md', '.txt', '.py', '.js', '.ts', '.jsx', '.tsx',
            '.java', '.go', '.rs', '.cpp', '.c', '.h', '.hpp',
            '.css', '.html', '.json', '.yaml', '.yml', '.xml',
            '.sh', '.bash', '.zsh', '.ps1', '.bat', '.cmd',
            '.sql', '.graphql', '.proto', '.toml', '.ini', '.cfg',
            '.dockerfile', '.makefile', '.cmake', '.gradle'
        }
        self.max_file_size = 1024 * 1024  # 1MB max per file (increased from 512KB)

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
            logger.info(f"Cloning repository {repo_url} (branch: {branch}) to {temp_dir}")

            # Set environment for git to avoid hanging on credentials
            clone_env = os.environ.copy()
            clone_env['GIT_TERMINAL_PROMPT'] = '0'

            try:
                # Only use kill_after_timeout if supported (not on Windows)
                clone_params = {
                    "url": repo_url,
                    "to_path": temp_dir,
                    "branch": branch,
                    "depth": 1,
                    "env": clone_env
                }
                
                if os.name != 'nt':  # 'nt' is Windows
                    clone_params["kill_after_timeout"] = 180
                else:
                    logger.debug("Skipping kill_after_timeout as it is not supported on Windows")

                repo = Repo.clone_from(**clone_params)
                logger.info(f"Successfully cloned repository {repo_url}")
            except GitCommandError as e:
                error_msg = str(e).lower()
                if 'not found' in error_msg or '404' in error_msg or 'repository' in error_msg:
                    raise ValueError(f"Repository not found or inaccessible: {repo_url}. Check if the URL is correct and the repo is public.")
                elif 'authentication' in error_msg or 'permission' in error_msg or '403' in error_msg:
                    raise ValueError(f"Authentication required for repository: {repo_url}. Only public repositories are supported.")
                elif 'branch' in error_msg or 'remote ref' in error_msg:
                    raise ValueError(f"Branch '{branch}' not found in repository: {repo_url}. Try 'main' or 'master'.")
                elif 'timeout' in error_msg or 'timed out' in error_msg:
                    raise ValueError(f"Repository clone timed out. The repository may be too large: {repo_url}")
                else:
                    logger.error(f"Git clone error: {e}")
                    raise ValueError(f"Failed to clone repository: {str(e)[:200]}")

            # Walk through repository files
            repo_path = Path(temp_dir)
            files_processed = 0
            files_skipped = 0

            for file_path in repo_path.rglob('*'):
                if file_path.is_file():
                    # Skip .git directory explicitly
                    if '.git' in file_path.parts:
                        continue

                    # Skip hidden files/dirs EXCEPT .github and other config-heavy dirs
                    important_hidden_dirs = {'.github', '.vscode', '.idea'}
                    if any(part.startswith('.') for part in file_path.parts):
                        # If it's a hidden part, check if it's one we want to keep
                        if not any(important in file_path.parts for important in important_hidden_dirs):
                            # It's a hidden file/dir that isn't in our "important" list
                            continue

                    # Skip node_modules, vendor, and other common large directories
                    skip_dirs = {
                        'node_modules', 'vendor', 'venv', '.venv', '__pycache__', 
                        'dist', 'build', 'target', '.pytest_cache', '.next', 'out',
                        'logs', 'coverage', 'htmlcov'
                    }
                    if any(part in skip_dirs for part in file_path.parts):
                        continue

                    # Check extension
                    suffix = file_path.suffix.lower()
                    # Also check for files without extension or with common config names
                    filename_lower = file_path.name.lower()
                    
                    # Expanded list of "always include" filenames
                    always_include_files = {
                        'dockerfile', 'makefile', 'readme', 'license', 
                        'procfile', 'gemfile', 'package.json', 'composer.json'
                    }
                    
                    if suffix not in extensions and filename_lower not in always_include_files:
                        # For hidden files in important directories, we might want to be more lenient
                        # e.g. .github/workflows/*.yml (which might be covered by .yml but good to be explicit)
                        if not (any(important in file_path.parts for important in important_hidden_dirs) and 
                                suffix in {'.yml', '.yaml', '.json', '.sh', '.js', '.ts'}):
                            continue

                    # Check file size
                    try:
                        file_size = file_path.stat().st_size
                        if file_size > self.max_file_size:
                            logger.debug(f"Skipping large file {file_path}: {file_size} bytes")
                            files_skipped += 1
                            continue
                    except OSError:
                        continue

                    try:
                        # Read file content
                        content = file_path.read_text(encoding='utf-8', errors='ignore')

                        # Skip empty files
                        if not content.strip():
                            continue

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
                                'type': suffix[1:] if suffix else 'unknown',
                                'size_bytes': file_size
                            }
                        })
                        files_processed += 1

                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
                        files_skipped += 1
                        continue

            logger.info(f"Extracted {len(documents)} documents from Git repository (processed: {files_processed}, skipped: {files_skipped})")

            if len(documents) == 0:
                logger.warning(f"No documents extracted from {repo_url}. The repository may be empty or contain only unsupported file types.")

        except ValueError:
            # Re-raise ValueError (our custom errors)
            raise

        except Exception as e:
            logger.error(f"Error processing repository {repo_url}: {e}")
            raise ValueError(f"Failed to process repository: {str(e)[:200]}")

        finally:
            # Clean up temporary directory
            if temp_dir and Path(temp_dir).exists():
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")

        return documents
