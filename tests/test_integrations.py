import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock, AsyncMock
import os

# Set test environment variables before importing app
os.environ.setdefault("INTEGRATIONS_KEY", "test-key-for-testing-only-32bytes!")
os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")
os.environ.setdefault("DATABASE_NAME", "test_rag_chat")

from app.main import app
from app.models.integration import Integration


class TestIntegrationsAPI:
    """Test integration CRUD endpoints"""

    @pytest.fixture
    def mock_integration(self):
        """Mock integration document"""
        mock = MagicMock()
        mock.id = "test-integration-id"
        mock.name = "Test Confluence"
        mock.type = "confluence"
        mock.config = {
            "base_url": "https://example.atlassian.net/wiki",
            "email": "test@example.com",
            "api_token": "gAAAAABtest-encrypted-token"
        }
        return mock

    @pytest.mark.asyncio
    async def test_create_integration_confluence(self):
        """Test creating a Confluence integration"""
        with patch.object(Integration, 'insert', new_callable=AsyncMock) as mock_insert:
            mock_insert.return_value = None

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/integrations/", json={
                    "name": "My Confluence",
                    "type": "confluence",
                    "config": {
                        "base_url": "https://mycompany.atlassian.net/wiki",
                        "email": "user@company.com",
                        "api_token": "secret-token-123"
                    }
                })

            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert data["name"] == "My Confluence"
            assert data["type"] == "confluence"

    @pytest.mark.asyncio
    async def test_create_integration_sharepoint(self):
        """Test creating a SharePoint integration"""
        with patch.object(Integration, 'insert', new_callable=AsyncMock) as mock_insert:
            mock_insert.return_value = None

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post("/integrations/", json={
                    "name": "My SharePoint",
                    "type": "sharepoint",
                    "config": {
                        "site_id": "site-123",
                        "access_token": "bearer-token-xyz"
                    }
                })

            assert response.status_code == 200
            data = response.json()
            assert data["type"] == "sharepoint"

    @pytest.mark.asyncio
    async def test_list_integrations(self, mock_integration):
        """Test listing integrations masks sensitive data"""
        with patch.object(Integration, 'find_all') as mock_find:
            mock_cursor = MagicMock()
            mock_cursor.to_list = AsyncMock(return_value=[mock_integration])
            mock_find.return_value = mock_cursor

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/integrations/")

            assert response.status_code == 200
            data = response.json()
            assert "integrations" in data
            assert len(data["integrations"]) == 1
            # Sensitive fields should be masked
            assert data["integrations"][0]["config"]["api_token"] == "***"

    @pytest.mark.asyncio
    async def test_get_integration_by_id(self, mock_integration):
        """Test getting a single integration by ID"""
        with patch.object(Integration, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_integration

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/integrations/test-integration-id")

            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "Test Confluence"
            assert data["config"]["api_token"] == "***"

    @pytest.mark.asyncio
    async def test_get_integration_not_found(self):
        """Test 404 when integration not found"""
        with patch.object(Integration, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.get("/integrations/nonexistent-id")

            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_integration(self, mock_integration):
        """Test deleting an integration"""
        mock_integration.delete = AsyncMock()

        with patch.object(Integration, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_integration

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.delete("/integrations/test-integration-id")

            assert response.status_code == 200
            assert response.json()["status"] == "deleted"

    @pytest.mark.asyncio
    async def test_delete_integration_not_found(self):
        """Test 404 when deleting non-existent integration"""
        with patch.object(Integration, 'get', new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None

            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.delete("/integrations/nonexistent-id")

            assert response.status_code == 404


class TestConfluenceSource:
    """Test Confluence data source"""

    def test_confluence_initialization(self):
        """Test ConfluenceSource initialization"""
        from app.services.data_sources.confluence_source import ConfluenceSource

        source = ConfluenceSource(
            base_url="https://example.atlassian.net/wiki",
            email="user@example.com",
            api_token="test-token"
        )

        assert source.base_url == "https://example.atlassian.net/wiki"
        assert source.email == "user@example.com"

    def test_confluence_content_to_docs(self):
        """Test converting Confluence content to documents"""
        from app.services.data_sources.confluence_source import ConfluenceSource

        source = ConfluenceSource(
            base_url="https://example.atlassian.net/wiki",
            email="user@example.com",
            api_token="test-token"
        )

        content = {
            'id': 'page-123',
            'title': 'Test Page',
            'body': {
                'storage': {
                    'value': '<h1>Hello</h1><p>World</p>'
                }
            }
        }

        docs = source._content_to_docs(content)

        assert len(docs) == 1
        assert docs[0]['metadata']['type'] == 'confluence_page'
        assert docs[0]['metadata']['page_id'] == 'page-123'
        assert docs[0]['metadata']['title'] == 'Test Page'


class TestSharePointSource:
    """Test SharePoint data source"""

    def test_sharepoint_initialization(self):
        """Test SharePointSource initialization"""
        from app.services.data_sources.sharepoint_source import SharePointSource

        source = SharePointSource(access_token="test-token")
        assert source.access_token == "test-token"

    @pytest.mark.asyncio
    async def test_sharepoint_requires_site_id(self):
        """Test SharePoint extract requires site_id"""
        from app.services.data_sources.sharepoint_source import SharePointSource

        source = SharePointSource(access_token="test-token")

        with pytest.raises(ValueError, match="site_id"):
            await source.extract()


class TestNotionSource:
    """Test Notion data source"""

    def test_notion_initialization(self):
        """Test NotionSource initialization"""
        from app.services.data_sources.notion_source import NotionSource

        source = NotionSource(api_key="secret_test_key")
        assert source.api_key == "secret_test_key"

    @pytest.mark.asyncio
    async def test_notion_requires_database_or_page_id(self):
        """Test Notion extract requires database_id or page_id"""
        from app.services.data_sources.notion_source import NotionSource

        source = NotionSource(api_key="secret_test_key")

        with pytest.raises(ValueError):
            await source.extract()


class TestDatabaseSource:
    """Test Database data source"""

    def test_database_initialization(self):
        """Test DatabaseSource initialization"""
        from app.services.data_sources.database_source import DatabaseSource

        source = DatabaseSource(
            db_type="postgresql",
            host="localhost",
            port=5432,
            database="testdb",
            user="testuser",
            password="testpass"
        )

        assert source.db_type == "postgresql"
        assert source.host == "localhost"
        assert source.database == "testdb"

    def test_database_connection_string(self):
        """Test database connection string generation"""
        from app.services.data_sources.database_source import DatabaseSource

        source = DatabaseSource(
            db_type="postgresql",
            host="localhost",
            port=5432,
            database="testdb",
            user="testuser",
            password="testpass"
        )

        # The connection string should be properly formatted
        conn_str = source._get_connection_string()
        assert "postgresql" in conn_str
        assert "localhost" in conn_str
        assert "testdb" in conn_str


class TestWebSource:
    """Test Web data source"""

    def test_web_source_initialization(self):
        """Test WebSource initialization"""
        from app.services.data_sources.web_source import WebSource

        source = WebSource()
        assert source is not None

    @pytest.mark.asyncio
    async def test_web_source_requires_url(self):
        """Test WebSource extract requires URL"""
        from app.services.data_sources.web_source import WebSource

        source = WebSource()

        with pytest.raises((ValueError, TypeError)):
            await source.extract()


class TestGitSource:
    """Test Git data source"""

    def test_git_source_initialization(self):
        """Test GitSource initialization"""
        from app.services.data_sources.git_source import GitSource

        source = GitSource()
        assert source is not None

    @pytest.mark.asyncio
    async def test_git_source_requires_repo_url(self):
        """Test GitSource extract requires repo_url"""
        from app.services.data_sources.git_source import GitSource

        source = GitSource()

        with pytest.raises((ValueError, TypeError)):
            await source.extract()
