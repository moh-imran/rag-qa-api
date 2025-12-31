import pytest
from app.services.data_sources.file_source import FileSource
import tempfile
import os


class TestFileSource:
    """Test FileSource data extraction"""
    
    @pytest.mark.asyncio
    async def test_extract_single_pdf(self, temp_pdf_file):
        """Test extracting a single PDF file"""
        source = FileSource()
        documents = await source.extract(file_path=temp_pdf_file)
        
        assert len(documents) == 1
        assert 'content' in documents[0]
        assert 'metadata' in documents[0]
        assert 'machine learning' in documents[0]['content'].lower()
        assert documents[0]['metadata']['type'] == 'pdf'
    
    @pytest.mark.asyncio
    async def test_extract_single_txt(self, temp_txt_file):
        """Test extracting a single text file"""
        source = FileSource()
        documents = await source.extract(file_path=temp_txt_file)
        
        assert len(documents) == 1
        assert 'artificial intelligence' in documents[0]['content'].lower()
        assert documents[0]['metadata']['type'] == 'txt'
    
    @pytest.mark.asyncio
    async def test_extract_directory(self, temp_directory):
        """Test extracting all files from directory"""
        source = FileSource()
        documents = await source.extract(directory_path=temp_directory)
        
        assert len(documents) == 2
        types = [doc['metadata']['type'] for doc in documents]
        assert 'pdf' in types
        assert 'txt' in types
    
    @pytest.mark.asyncio
    async def test_extract_nonexistent_file(self):
        """Test error handling for non-existent file"""
        source = FileSource()
        
        with pytest.raises(FileNotFoundError):
            await source.extract(file_path='/nonexistent/file.pdf')
    
    @pytest.mark.asyncio
    async def test_extract_unsupported_format(self):
        """Test error handling for unsupported file format"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xyz')
        temp_file.close()
        
        source = FileSource()
        
        with pytest.raises(ValueError):
            await source.extract(file_path=temp_file.name)
        
        os.unlink(temp_file.name)