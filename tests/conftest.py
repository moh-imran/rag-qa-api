import pytest
import tempfile
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to Python path to allow importing 'app'
sys.path.append(str(Path(__file__).parent.parent))


@pytest.fixture
def temp_pdf_file():
    """Create a temporary PDF file for testing"""
    from reportlab.pdfgen import canvas
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    
    # Create simple PDF
    c = canvas.Canvas(temp_file.name)
    c.drawString(100, 750, "Test Document")
    c.drawString(100, 730, "This is a test paragraph about machine learning.")
    c.drawString(100, 710, "Machine learning is a subset of artificial intelligence.")
    c.save()
    
    yield temp_file.name
    
    # Cleanup
    os.unlink(temp_file.name)


@pytest.fixture
def temp_txt_file():
    """Create a temporary text file for testing"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
    temp_file.write("Test document about AI.\n")
    temp_file.write("Artificial intelligence is transforming the world.\n" * 10)
    temp_file.close()
    
    yield temp_file.name
    
    os.unlink(temp_file.name)


@pytest.fixture
def temp_directory(temp_pdf_file, temp_txt_file):
    """Create a temporary directory with test files"""
    temp_dir = tempfile.mkdtemp()
    
    # Copy files to temp directory
    import shutil
    shutil.copy(temp_pdf_file, os.path.join(temp_dir, 'test1.pdf'))
    shutil.copy(temp_txt_file, os.path.join(temp_dir, 'test2.txt'))
    
    yield temp_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_client():
    """Create FastAPI test client"""
    from app.main import app
    return TestClient(app)
