import sys
import types
import pytest

# Provide a minimal `bs4` stub so tests run without installing dependencies
if 'bs4' not in sys.modules:
    def _fake_bs4(html, parser):
        class FakeSoup:
            def __init__(self, text):
                self._text = text

            def get_text(self, separator='\n', strip=True):
                # crude extraction used only for tests
                return 'Title\nThis is content.'

        return FakeSoup(html)

    sys.modules['bs4'] = types.SimpleNamespace(BeautifulSoup=_fake_bs4)

from app.services.data_sources.confluence_source import ConfluenceSource
from app.services.data_sources.sharepoint_source import SharePointSource


def test_confluence_content_to_docs_handles_html():
    html = "<h1>Title</h1><p>This is <strong>content</strong>.</p>"
    sample = {
        'id': '123',
        'title': 'My Page',
        'body': {
            'storage': {
                'value': html
            }
        }
    }

    src = ConfluenceSource(base_url='https://example.atlassian.net/wiki', email='u', api_token='t')
    docs = src._content_to_docs(sample)
    assert isinstance(docs, list)
    assert len(docs) == 1
    doc = docs[0]
    assert 'content' in doc
    assert 'This is' in doc['content']
    assert doc['metadata']['type'] == 'confluence_page'


def test_sharepoint_extract_requires_site_id():
    src = SharePointSource(access_token='dummy')
    with pytest.raises(ValueError):
        # missing site_id should raise
        import asyncio
        asyncio.get_event_loop().run_until_complete(src.extract())
