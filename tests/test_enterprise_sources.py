import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.data_sources.notion_source import NotionSource
from app.services.data_sources.confluence_source import ConfluenceSource
from app.services.data_sources.sharepoint_source import SharePointSource

class TestNotionSource:
    @pytest.mark.asyncio
    async def test_notion_extract_stream(self):
        with patch('notion_client.Client') as mock_client:
            instance = mock_client.return_value
            instance.users.me.return_value = {}
            instance.pages.retrieve.return_value = {
                'url': 'https://notion.so/test',
                'created_time': '2024-01-01',
                'last_edited_time': '2024-01-02',
                'properties': {'title': {'type': 'title', 'title': [{'plain_text': 'Test Page'}]}}
            }
            instance.blocks.children.list.return_value = {
                'results': [
                    {'type': 'heading_1', 'heading_1': {'rich_text': [{'plain_text': 'Header'}]}},
                    {'type': 'paragraph', 'paragraph': {'rich_text': [{'plain_text': 'Body'}]}}
                ]
            }

            source = NotionSource(api_key="secret_test")
            docs = []
            async for doc in source.extract_stream(page_id="test_id"):
                docs.append(doc)

            assert len(docs) == 1
            assert "# Header" in docs[0]['content']
            assert "Body" in docs[0]['content']
            assert docs[0]['metadata']['title'] == 'Test Page'

class TestConfluenceSource:
    @pytest.mark.asyncio
    async def test_confluence_extract_stream(self):
        source = ConfluenceSource(base_url="https://test.atlassian.net/wiki", email="test@test.com", api_token="token")
        
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            'id': '123',
            'title': 'Test Page',
            'body': {'storage': {'value': '<p>Hello world</p>'}},
            'version': {'number': 1, 'friendlyWhen': 'Today'}
        }

        with patch.object(source.session, 'get', return_value=mock_resp):
            docs = []
            async for doc in source.extract_stream(content_id="123"):
                docs.append(doc)

            assert len(docs) == 1
            assert "Hello world" in docs[0]['content']
            assert docs[0]['metadata']['title'] == 'Test Page'

class TestSharePointSource:
    @pytest.mark.asyncio
    async def test_sharepoint_extract_stream(self):
        source = SharePointSource(access_token="token", site_id="site_id")
        
        mock_list_resp = MagicMock()
        mock_list_resp.status_code = 200
        mock_list_resp.json.return_value = {
            'value': [{
                'id': 'file1',
                'name': 'test.txt',
                'file': {'mimeType': 'text/plain'},
                'webUrl': 'https://sharepoint.com/test.txt'
            }]
        }

        mock_content_resp = MagicMock()
        mock_content_resp.status_code = 200
        mock_content_resp.content = b"SharePoint Content"

        with patch.object(source.session, 'get', side_effect=[mock_list_resp, mock_content_resp]):
            docs = []
            async for doc in source.extract_stream(max_items=1):
                docs.append(doc)

            assert len(docs) == 1
            assert "SharePoint Content" in docs[0]['content']
            assert docs[0]['metadata']['filename'] == 'test.txt'
