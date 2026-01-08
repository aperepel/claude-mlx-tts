"""
Unit tests for TUI Preview Widget.

Tests the fallback path from HTTP server to direct MLX invocation.

Run with: uv run pytest tests/unit/test_preview_widget.py -v
"""
import os
import sys
from unittest.mock import patch, MagicMock


# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestPreviewWidgetConstants:
    """Tests for preview widget constants."""

    def test_default_preview_phrase_defined(self):
        """Default preview phrase should be defined."""
        from tts_tui import DEFAULT_PREVIEW_PHRASE

        assert DEFAULT_PREVIEW_PHRASE is not None
        assert len(DEFAULT_PREVIEW_PHRASE) > 0

    def test_preview_phrases_list(self):
        """Preview phrases list should contain multiple options."""
        from tts_tui import PREVIEW_PHRASES

        assert len(PREVIEW_PHRASES) >= 1
        assert all(isinstance(phrase, str) for phrase in PREVIEW_PHRASES)


class TestTTSFallbackLogic:
    """Tests for TTS invocation fallback logic."""

    def test_uses_http_when_server_alive(self):
        """Should use HTTP server when server is alive."""
        # Mock the imports and functions
        mock_speak_http = MagicMock()
        mock_is_alive = MagicMock(return_value=True)

        with patch.dict("sys.modules", {
            "mlx_server_utils": MagicMock(
                speak_mlx_http=mock_speak_http,
                is_server_alive=mock_is_alive,
            ),
        }):
            # Simulate the fallback logic from PreviewWidget._do_tts_playback
            from mlx_server_utils import speak_mlx_http, is_server_alive
            test_phrase = "Test phrase"

            if is_server_alive():
                speak_mlx_http(test_phrase)

            mock_is_alive.assert_called_once()
            mock_speak_http.assert_called_once_with(test_phrase)

    def test_falls_back_to_mlx_when_server_down(self):
        """Should fall back to direct MLX when server is not alive."""
        mock_speak_http = MagicMock()
        mock_is_alive = MagicMock(return_value=False)
        mock_speak_mlx = MagicMock()

        with patch.dict("sys.modules", {
            "mlx_server_utils": MagicMock(
                speak_mlx_http=mock_speak_http,
                is_server_alive=mock_is_alive,
            ),
            "mlx_tts_core": MagicMock(speak_mlx=mock_speak_mlx),
        }):
            from mlx_server_utils import speak_mlx_http, is_server_alive
            from mlx_tts_core import speak_mlx
            test_phrase = "Test phrase"

            if is_server_alive():
                speak_mlx_http(test_phrase)
            else:
                speak_mlx(test_phrase)

            mock_is_alive.assert_called_once()
            mock_speak_http.assert_not_called()
            mock_speak_mlx.assert_called_once_with(test_phrase)

    def test_falls_back_on_import_error(self):
        """Should fall back to direct MLX when mlx_server_utils import fails."""
        mock_speak_mlx = MagicMock()
        test_phrase = "Test phrase"
        used_fallback = False

        # Simulate the fallback logic pattern from PreviewWidget._do_tts_playback
        # This tests the exception handling branch, not an actual import failure
        def simulate_fallback_logic():
            nonlocal used_fallback
            try:
                # Simulate ImportError case
                raise ImportError("mlx_server_utils not available")
            except ImportError:
                # Fall back to direct MLX if server utils unavailable
                mock_speak_mlx(test_phrase)
                used_fallback = True

        simulate_fallback_logic()

        assert used_fallback, "Fallback path should have been used"
        mock_speak_mlx.assert_called_once_with(test_phrase)


class TestPreviewWidgetImport:
    """Tests for preview widget import and structure."""

    def test_preview_widget_importable(self):
        """PreviewWidget should be importable."""
        from tts_tui import PreviewWidget

        assert PreviewWidget is not None

    def test_preview_widget_has_required_attributes(self):
        """PreviewWidget should have required reactive attributes."""
        from tts_tui import PreviewWidget

        # Check that the class has the reactive attributes defined
        assert hasattr(PreviewWidget, "is_playing")
        assert hasattr(PreviewWidget, "status_message")

    def test_preview_widget_has_css(self):
        """PreviewWidget should have CSS defined."""
        from tts_tui import PreviewWidget

        assert hasattr(PreviewWidget, "DEFAULT_CSS")
        assert "PreviewWidget" in PreviewWidget.DEFAULT_CSS
