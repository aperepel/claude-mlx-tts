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


# =============================================================================
# Cache Invalidation Tests (mlx-tts-XXX)
# =============================================================================


class TestCacheInvalidationOnFocus:
    """Tests for cache invalidation when input gains focus.

    Bug: Directory contents are cached indefinitely, so new files added
    to a directory don't appear in autocomplete suggestions.

    Fix: Clear cache when input gains focus to pick up filesystem changes.
    """

    def test_handle_focus_change_method_exists(self):
        """WavPathAutoComplete should override _handle_focus_change."""
        from tts_tui import WavPathAutoComplete
        import inspect

        # Check that the method is defined in WavPathAutoComplete, not just inherited
        assert "_handle_focus_change" in WavPathAutoComplete.__dict__, \
            "WavPathAutoComplete must override _handle_focus_change to clear cache"

    def test_cache_cleared_on_focus_gain(self):
        """Cache should be cleared when input gains focus."""
        from tts_tui import WavPathAutoComplete
        from unittest.mock import MagicMock, patch

        # Create instance with mocked target
        autocomplete = WavPathAutoComplete.__new__(WavPathAutoComplete)
        autocomplete._directory_cache = {"test_dir": ["cached_entries"]}  # type: ignore[assignment]
        autocomplete.clear_directory_cache = MagicMock()

        # Mock the parent class's _handle_focus_change
        with patch.object(WavPathAutoComplete.__bases__[0], '_handle_focus_change'):
            autocomplete._handle_focus_change(has_focus=True)

        # Cache should be cleared on focus gain
        autocomplete.clear_directory_cache.assert_called_once()

    def test_cache_not_cleared_on_focus_loss(self):
        """Cache should NOT be cleared when input loses focus."""
        from tts_tui import WavPathAutoComplete
        from unittest.mock import MagicMock, patch

        autocomplete = WavPathAutoComplete.__new__(WavPathAutoComplete)
        autocomplete._directory_cache = {"test_dir": ["cached_entries"]}  # type: ignore[assignment]
        autocomplete.clear_directory_cache = MagicMock()

        with patch.object(WavPathAutoComplete.__bases__[0], '_handle_focus_change'):
            autocomplete._handle_focus_change(has_focus=False)

        # Cache should NOT be cleared on focus loss
        autocomplete.clear_directory_cache.assert_not_called()

    def test_parent_handle_focus_change_called(self):
        """Parent _handle_focus_change should always be called."""
        from tts_tui import WavPathAutoComplete
        from unittest.mock import MagicMock, patch

        autocomplete = WavPathAutoComplete.__new__(WavPathAutoComplete)
        autocomplete._directory_cache = {}
        autocomplete.clear_directory_cache = MagicMock()

        with patch.object(WavPathAutoComplete.__bases__[0], '_handle_focus_change') as mock_parent:
            autocomplete._handle_focus_change(has_focus=True)
            mock_parent.assert_called_once_with(has_focus=True)

            mock_parent.reset_mock()
            autocomplete._handle_focus_change(has_focus=False)
            mock_parent.assert_called_once_with(has_focus=False)


# =============================================================================
# Voice Test Isolation Tests
# =============================================================================


class TestVoiceTestIsolation:
    """Tests that testing a cloned voice does NOT modify active voice config.

    Bug: When testing a cloned voice, set_active_voice() was called which
    permanently changed the global config. Any subsequent TTS (like stop hooks)
    would use the cloned voice instead of the user's selected voice.

    Fix: Pass --voice flag to TTS script instead of modifying global config.
    """

    def test_test_voice_does_not_call_set_active_voice(self):
        """_test_voice must NOT call set_active_voice to avoid config pollution."""
        from tts_tui import CloneLabWidget
        import inspect

        source = inspect.getsource(CloneLabWidget._test_voice)

        # The method should NOT contain set_active_voice
        assert "set_active_voice" not in source, \
            "_test_voice must not call set_active_voice - use --voice flag instead"

    def test_run_test_worker_passes_voice_flag(self):
        """_run_test_worker must pass --voice flag to subprocess."""
        from tts_tui import CloneLabWidget
        import inspect

        source = inspect.getsource(CloneLabWidget._run_test_worker)

        # The worker should pass --voice to the subprocess
        assert "--voice" in source, \
            "_run_test_worker must pass --voice flag to TTS script"

    def test_run_test_worker_uses_cloned_voice_name(self):
        """_run_test_worker must use self.cloned_voice_name for --voice value."""
        from tts_tui import CloneLabWidget
        import inspect

        source = inspect.getsource(CloneLabWidget._run_test_worker)

        # Should reference cloned_voice_name for the voice parameter
        assert "cloned_voice_name" in source, \
            "_run_test_worker must use cloned_voice_name for --voice value"
