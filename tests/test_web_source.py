import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from app.services.data_sources.web_source import WebSource, POPUP_SELECTORS


class TestWebSourceBasic:
    """Test basic WebSource functionality (requests-based)"""

    @pytest.mark.asyncio
    async def test_extract_requires_url(self):
        """Test that extract raises ValueError when no URL provided"""
        source = WebSource()
        with pytest.raises(ValueError, match="Must provide 'url' or 'urls'"):
            await source.extract()

    @pytest.mark.asyncio
    async def test_extract_invalid_url_format(self):
        """Test that extract raises ValueError for invalid URL"""
        source = WebSource()
        with pytest.raises(ValueError, match="Invalid URL format"):
            await source.extract(url="not-a-valid-url")

    @pytest.mark.asyncio
    async def test_extract_single_url_default_mode(self):
        """Test extraction with default (requests) mode"""
        source = WebSource()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content = b'<html><head><title>Test Page</title></head><body><p>This is test content that is long enough to pass the minimum threshold.</p></body></html>'

        with patch('requests.get', return_value=mock_response):
            documents = await source.extract(url="https://example.com")

        assert len(documents) == 1
        assert 'test content' in documents[0]['content'].lower()
        assert documents[0]['metadata']['browser_mode'] is False

    @pytest.mark.asyncio
    async def test_extract_handles_403_error(self):
        """Test that 403 errors are handled gracefully"""
        source = WebSource()

        mock_response = MagicMock()
        mock_response.status_code = 403

        with patch('requests.get', return_value=mock_response):
            documents = await source.extract(url="https://example.com/forbidden")

        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_extract_handles_404_error(self):
        """Test that 404 errors are handled gracefully"""
        source = WebSource()

        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch('requests.get', return_value=mock_response):
            documents = await source.extract(url="https://example.com/notfound")

        assert len(documents) == 0


class TestWebSourceBrowserMode:
    """Test Playwright browser mode functionality"""

    @pytest.mark.asyncio
    async def test_use_browser_flag_triggers_browser_path(self):
        """Test that use_browser=True triggers Playwright browser scraping"""
        source = WebSource()

        # Mock the browser scraping method
        source._scrape_with_browser = AsyncMock(
            return_value='<html><head><title>JS Page</title></head><body><p>JavaScript rendered content that passes the minimum threshold.</p></body></html>'
        )
        source._cleanup_browser = AsyncMock()

        documents = await source.extract(
            url="https://finance.yahoo.com/quote/AAPL",
            use_browser=True
        )

        # Verify browser method was called
        source._scrape_with_browser.assert_called_once()
        source._cleanup_browser.assert_called_once()

        assert len(documents) == 1
        assert documents[0]['metadata']['browser_mode'] is True

    @pytest.mark.asyncio
    async def test_browser_mode_passes_parameters(self):
        """Test that browser mode parameters are passed correctly"""
        source = WebSource()

        source._scrape_with_browser = AsyncMock(
            return_value='<html><head><title>Test</title></head><body><p>Content that is definitely long enough to pass validation check.</p></body></html>'
        )
        source._cleanup_browser = AsyncMock()

        await source.extract(
            url="https://example.com",
            use_browser=True,
            wait_time=5000,
            auto_dismiss_popups=False,
            wait_for_selector=".custom-selector"
        )

        # Check that parameters were passed to _scrape_with_browser
        call_kwargs = source._scrape_with_browser.call_args[1]
        assert call_kwargs['wait_time'] == 5000
        assert call_kwargs['auto_dismiss_popups'] is False
        assert call_kwargs['wait_for_selector'] == ".custom-selector"

    @pytest.mark.asyncio
    async def test_browser_cleanup_on_error(self):
        """Test that browser is cleaned up even on error"""
        source = WebSource()

        source._scrape_with_browser = AsyncMock(side_effect=Exception("Browser error"))
        source._cleanup_browser = AsyncMock()

        # Should not raise, just return empty
        documents = await source.extract(
            url="https://example.com",
            use_browser=True
        )

        # Cleanup should still be called
        source._cleanup_browser.assert_called_once()
        assert len(documents) == 0

    @pytest.mark.asyncio
    async def test_browser_returns_none_handled(self):
        """Test that None return from browser scraping is handled"""
        source = WebSource()

        source._scrape_with_browser = AsyncMock(return_value=None)
        source._cleanup_browser = AsyncMock()

        documents = await source.extract(
            url="https://example.com",
            use_browser=True
        )

        assert len(documents) == 0


class TestPopupDismissal:
    """Test popup dismissal functionality"""

    def test_popup_selectors_defined(self):
        """Test that popup selectors are properly defined"""
        assert len(POPUP_SELECTORS) > 0
        assert '#onetrust-accept-btn-handler' in POPUP_SELECTORS
        assert 'button[name="agree"]' in POPUP_SELECTORS  # Yahoo-specific

    @pytest.mark.asyncio
    async def test_dismiss_popups_tries_selectors(self):
        """Test that dismiss_popups tries multiple selectors"""
        source = WebSource()

        mock_page = MagicMock()
        mock_locator = MagicMock()
        mock_locator.first = mock_locator
        mock_locator.is_visible = AsyncMock(side_effect=[False, False, True])
        mock_locator.click = AsyncMock()
        mock_page.locator = MagicMock(return_value=mock_locator)

        result = await source._dismiss_popups(mock_page)

        assert result is True
        assert mock_page.locator.call_count >= 3


class TestBrowserInitialization:
    """Test browser initialization and cleanup"""

    @pytest.mark.asyncio
    async def test_init_browser_creates_browser(self):
        """Test that _init_browser initializes Playwright"""
        source = WebSource()

        mock_playwright = MagicMock()
        mock_browser = MagicMock()
        mock_playwright_instance = MagicMock()
        mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)

        with patch('app.services.data_sources.web_source.async_playwright') as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=mock_playwright_instance)

            # Import inside to get the patched version
            from playwright.async_api import async_playwright

            await source._init_browser()

        assert source._browser is not None or mock_playwright_instance.chromium.launch.called

    @pytest.mark.asyncio
    async def test_cleanup_browser_closes_resources(self):
        """Test that _cleanup_browser properly closes resources"""
        source = WebSource()
        source._browser = AsyncMock()
        source._playwright = AsyncMock()

        await source._cleanup_browser()

        assert source._browser is None
        assert source._playwright is None


class TestScrollPage:
    """Test page scrolling functionality"""

    @pytest.mark.asyncio
    async def test_scroll_page_scrolls_multiple_times(self):
        """Test that _scroll_page scrolls the specified number of times"""
        source = WebSource()

        mock_page = MagicMock()
        mock_page.evaluate = AsyncMock()

        await source._scroll_page(mock_page, scroll_count=3)

        # Should scroll down 3 times + scroll back to top
        assert mock_page.evaluate.call_count == 4


class TestMetadata:
    """Test metadata in extracted documents"""

    @pytest.mark.asyncio
    async def test_metadata_includes_browser_mode(self):
        """Test that metadata includes browser_mode flag"""
        source = WebSource()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.content = b'<html><head><title>Test</title></head><body><p>Content that is long enough to pass the validation threshold for documents.</p></body></html>'

        with patch('requests.get', return_value=mock_response):
            documents = await source.extract(url="https://example.com")

        assert 'browser_mode' in documents[0]['metadata']
        assert documents[0]['metadata']['browser_mode'] is False

    @pytest.mark.asyncio
    async def test_extract_parameters_have_defaults(self):
        """Test that new extract parameters have sensible defaults"""
        source = WebSource()

        # Verify default values by checking the signature
        import inspect
        sig = inspect.signature(source.extract)

        assert sig.parameters['use_browser'].default is False
        assert sig.parameters['wait_time'].default == 3000
        assert sig.parameters['auto_dismiss_popups'].default is True
        assert sig.parameters['wait_for_selector'].default is None
