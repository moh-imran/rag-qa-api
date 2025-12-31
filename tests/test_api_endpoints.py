import pytest
from fastapi.testclient import TestClient


class TestIngestionEndpoints:
    """Test ingestion API endpoints"""
    
    def test_health_check(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'embedding_model' in data
        assert 'qdrant' in data
    
    def test_list_sources(self, test_client):
        """Test listing data sources"""
        response = test_client.get("/sources")
        
        assert response.status_code == 200
        data = response.json()
        assert 'available_sources' in data
        assert 'file' in data['available_sources']
    
    def test_ingest_file_missing_path(self, test_client):
        """Test ingestion with missing file path"""
        response = test_client.post(
            "/ingest/file",
            json={"chunk_size": 1000}  # Missing file_path
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_upload_unsupported_file(self, test_client):
        """Test upload with unsupported file type"""
        # Create a fake file
        files = {'file': ('test.xyz', b'fake content', 'application/octet-stream')}
        
        response = test_client.post(
            "/ingest/upload",
            files=files
        )
        
        assert response.status_code == 400
        assert 'Unsupported file type' in response.json()['detail']
    
    @pytest.mark.integration
    def test_ingest_file_success(self, test_client, temp_txt_file):
        """Test successful file ingestion (integration test)"""
        response = test_client.post(
            "/ingest/file",
            json={
                "file_path": temp_txt_file,
                "chunk_size": 500,
                "chunk_overlap": 50,
                "store_in_qdrant": False  # Don't store in Qdrant for test
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert data['total_documents'] > 0
        assert data['total_chunks'] > 0
    
    @pytest.mark.integration
    def test_upload_file_success(self, test_client, temp_txt_file):
        """Test successful file upload (integration test)"""
        with open(temp_txt_file, 'rb') as f:
            files = {'file': ('test.txt', f, 'text/plain')}
            
            response = test_client.post(
                "/ingest/upload",
                files=files,
                data={'store_in_qdrant': 'false'}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert data['total_chunks'] > 0