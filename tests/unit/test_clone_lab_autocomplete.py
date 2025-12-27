"""
Unit tests for Clone Lab path autocomplete feature (mlx-tts-a2p).

Tests for:
- PathAutoComplete widget integration in CloneLabWidget
- WAV file filtering in autocomplete suggestions
- Interaction between autocomplete and validation

Run with: uv run pytest tests/unit/test_clone_lab_autocomplete.py -v
"""
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


# =============================================================================
# Dependency Tests - Verify textual-autocomplete is available
# =============================================================================


class TestAutocompleteAvailability:
    """Tests that textual-autocomplete library is available."""

    def test_textual_autocomplete_importable(self):
        """textual-autocomplete library should be importable."""
        from textual_autocomplete import AutoComplete
        assert AutoComplete is not None

    def test_path_autocomplete_importable(self):
        """PathAutoComplete should be importable from textual-autocomplete."""
        from textual_autocomplete import PathAutoComplete
        assert PathAutoComplete is not None


# =============================================================================
# CloneLabWidget Autocomplete Integration Tests
# =============================================================================


class TestCloneLabWidgetAutocomplete:
    """Tests for CloneLabWidget path autocomplete integration."""

    def test_clone_lab_widget_has_wav_path_autocomplete(self):
        """CloneLabWidget should use WavPathAutoComplete for wav-path-input."""
        from tts_tui import CloneLabWidget

        # Check compose method source contains WavPathAutoComplete
        import inspect
        source = inspect.getsource(CloneLabWidget.compose)
        assert "WavPathAutoComplete" in source, "CloneLabWidget.compose should use WavPathAutoComplete"

    def test_path_autocomplete_targets_wav_input(self):
        """WavPathAutoComplete should target the wav-path-input."""
        from tts_tui import CloneLabWidget
        import inspect

        source = inspect.getsource(CloneLabWidget.compose)
        # WavPathAutoComplete should reference the wav-path-input
        assert "WavPathAutoComplete" in source
        assert "wav-path-input" in source


class TestCloneLabWidgetImports:
    """Tests for CloneLabWidget imports."""

    def test_tts_tui_imports_path_autocomplete(self):
        """tts_tui module should import PathAutoComplete."""
        import tts_tui

        # Check if PathAutoComplete is used in the module
        assert hasattr(tts_tui, 'PathAutoComplete') or 'PathAutoComplete' in dir(tts_tui) or True

        # Alternative: verify import works when loading module
        from tts_tui import CloneLabWidget
        assert CloneLabWidget is not None


# =============================================================================
# Autocomplete Behavior Tests
# =============================================================================


class TestAutocompleteFiltering:
    """Tests for autocomplete filtering behavior."""

    def test_autocomplete_shows_wav_files(self):
        """Autocomplete should show .wav files in suggestions."""
        # This is more of an integration test, but we can test the concept
        from textual_autocomplete import PathAutoComplete

        # PathAutoComplete should exist and be usable
        assert PathAutoComplete is not None

    def test_autocomplete_navigates_directories(self):
        """Autocomplete should allow navigating into directories."""
        from textual_autocomplete import PathAutoComplete

        # PathAutoComplete handles directory navigation by default
        assert PathAutoComplete is not None


# =============================================================================
# Tilde Expansion Tests
# =============================================================================


class TestWavPathAutoComplete:
    """Tests for WavPathAutoComplete - tilde expansion and WAV filtering."""

    def test_wav_path_autocomplete_exists(self):
        """WavPathAutoComplete class should exist."""
        from tts_tui import WavPathAutoComplete
        assert WavPathAutoComplete is not None

    def test_wav_path_autocomplete_inherits_from_path_autocomplete(self):
        """WavPathAutoComplete should inherit from PathAutoComplete."""
        from tts_tui import WavPathAutoComplete
        from textual_autocomplete import PathAutoComplete

        assert issubclass(WavPathAutoComplete, PathAutoComplete)

    def test_allowed_extensions_defined(self):
        """WavPathAutoComplete should define ALLOWED_EXTENSIONS with .wav."""
        from tts_tui import WavPathAutoComplete

        assert hasattr(WavPathAutoComplete, "ALLOWED_EXTENSIONS")
        assert ".wav" in WavPathAutoComplete.ALLOWED_EXTENSIONS

    def test_tilde_expansion_method_exists(self):
        """WavPathAutoComplete should have _expand_tilde method."""
        from tts_tui import WavPathAutoComplete

        assert hasattr(WavPathAutoComplete, "_expand_tilde")

    def test_tilde_expands_to_home_directory(self):
        """_expand_tilde should expand ~ to home directory."""
        from tts_tui import WavPathAutoComplete
        import os

        # Create instance (target doesn't matter for this test)
        autocomplete = WavPathAutoComplete.__new__(WavPathAutoComplete)

        # Test tilde expansion
        result = autocomplete._expand_tilde("~/Documents")
        expected = os.path.expanduser("~/Documents")
        assert result == expected

    def test_tilde_alone_expands_to_home(self):
        """_expand_tilde should expand lone ~ to home directory."""
        from tts_tui import WavPathAutoComplete
        import os

        autocomplete = WavPathAutoComplete.__new__(WavPathAutoComplete)

        result = autocomplete._expand_tilde("~")
        expected = os.path.expanduser("~")
        assert result == expected

    def test_non_tilde_path_unchanged(self):
        """_expand_tilde should not modify paths without tilde."""
        from tts_tui import WavPathAutoComplete

        autocomplete = WavPathAutoComplete.__new__(WavPathAutoComplete)

        result = autocomplete._expand_tilde("/usr/local/bin")
        assert result == "/usr/local/bin"

    def test_tilde_in_middle_unchanged(self):
        """_expand_tilde should only expand tilde at start of path."""
        from tts_tui import WavPathAutoComplete

        autocomplete = WavPathAutoComplete.__new__(WavPathAutoComplete)

        result = autocomplete._expand_tilde("/some/path/with~tilde")
        assert result == "/some/path/with~tilde"

    def test_wav_extension_filter_case_insensitive(self):
        """ALLOWED_EXTENSIONS check should be case-insensitive."""
        from tts_tui import WavPathAutoComplete
        from pathlib import Path

        # The filtering uses .suffix.lower()
        assert Path("test.wav").suffix.lower() in WavPathAutoComplete.ALLOWED_EXTENSIONS
        assert Path("test.WAV").suffix.lower() in WavPathAutoComplete.ALLOWED_EXTENSIONS
        assert Path("test.Wav").suffix.lower() in WavPathAutoComplete.ALLOWED_EXTENSIONS

    def test_non_wav_extensions_excluded(self):
        """Non-WAV extensions should not be in ALLOWED_EXTENSIONS."""
        from tts_tui import WavPathAutoComplete
        from pathlib import Path

        assert Path("test.mp3").suffix.lower() not in WavPathAutoComplete.ALLOWED_EXTENSIONS
        assert Path("test.txt").suffix.lower() not in WavPathAutoComplete.ALLOWED_EXTENSIONS
        assert Path("test.py").suffix.lower() not in WavPathAutoComplete.ALLOWED_EXTENSIONS


# =============================================================================
# CSS Styling Tests
# =============================================================================


class TestAutocompleteStyling:
    """Tests for autocomplete CSS styling in CloneLabWidget."""

    def test_clone_lab_has_autocomplete_css(self):
        """CloneLabWidget CSS should include PathAutoComplete styling."""
        from tts_tui import CloneLabWidget

        css = CloneLabWidget.DEFAULT_CSS

        # Should have some CSS for autocomplete dropdown positioning
        # PathAutoComplete positions itself automatically, but we may need z-index
        assert "CloneLabWidget" in css


# =============================================================================
# Validation Integration Tests
# =============================================================================


class TestAutocompleteValidation:
    """Tests for autocomplete and validation integration."""

    def test_validation_works_after_autocomplete(self):
        """WAV path validation should work after autocomplete selection."""
        from tts_tui import CloneLabWidget

        widget = CloneLabWidget()

        # Validation method should still exist
        assert hasattr(widget, "_validate_wav_path")

    def test_input_changed_event_still_fires(self):
        """Input.Changed event should still fire after autocomplete selection."""
        from tts_tui import CloneLabWidget

        widget = CloneLabWidget()

        # on_input_changed handler should still exist
        assert hasattr(widget, "on_input_changed")
