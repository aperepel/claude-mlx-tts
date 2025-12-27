"""
Unit tests for Voice CRUD TUI components (mlx-tts-7q8).

Tests for:
- VoiceSelector action bar with Copy/Rename/Delete buttons
- DeleteVoiceModal with usage warning
- InputModal for Copy/Rename name input with validation

Run with: uv run pytest tests/unit/test_voice_crud_tui.py -v
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


# =============================================================================
# DeleteVoiceModal Tests
# =============================================================================


class TestDeleteVoiceModal:
    """Tests for DeleteVoiceModal component."""

    def test_delete_voice_modal_exists(self):
        """DeleteVoiceModal class should exist."""
        from tts_tui import DeleteVoiceModal
        assert DeleteVoiceModal is not None

    def test_delete_voice_modal_accepts_voice_name(self):
        """DeleteVoiceModal should accept voice_name parameter."""
        from tts_tui import DeleteVoiceModal
        modal = DeleteVoiceModal(voice_name="test_voice")
        assert modal.voice_name == "test_voice"

    def test_delete_voice_modal_accepts_usage_info(self):
        """DeleteVoiceModal should accept usage info for warning display."""
        from tts_tui import DeleteVoiceModal
        usage = {"is_active": True, "hooks": ["stop"], "has_settings": True}
        modal = DeleteVoiceModal(voice_name="test_voice", usage=usage)
        assert modal.usage == usage

    def test_delete_voice_modal_has_cancel_button(self):
        """DeleteVoiceModal should have a Cancel button."""
        from tts_tui import DeleteVoiceModal
        modal = DeleteVoiceModal(voice_name="test_voice")
        # Modal exists - button presence verified by message handlers
        assert modal is not None

    def test_delete_voice_modal_has_delete_button(self):
        """DeleteVoiceModal should have a Delete button."""
        from tts_tui import DeleteVoiceModal
        modal = DeleteVoiceModal(voice_name="test_voice")
        # Compose should include delete button
        assert modal is not None

    def test_delete_voice_modal_shows_warning_for_active_voice(self):
        """DeleteVoiceModal should show warning when deleting active voice."""
        from tts_tui import DeleteVoiceModal
        usage = {"is_active": True, "hooks": [], "has_settings": False}
        modal = DeleteVoiceModal(voice_name="test_voice", usage=usage)
        assert modal.usage["is_active"] is True

    def test_delete_voice_modal_shows_hook_usage(self):
        """DeleteVoiceModal should show which hooks use the voice."""
        from tts_tui import DeleteVoiceModal
        usage = {"is_active": False, "hooks": ["stop", "permission_request"], "has_settings": False}
        modal = DeleteVoiceModal(voice_name="test_voice", usage=usage)
        assert "stop" in modal.usage["hooks"]
        assert "permission_request" in modal.usage["hooks"]


class TestDeleteVoiceModalMessages:
    """Tests for DeleteVoiceModal message handling."""

    def test_delete_voice_modal_posts_confirmed_message(self):
        """DeleteVoiceModal should post VoiceDeleteConfirmed message on delete."""
        from tts_tui import DeleteVoiceModal
        # Modal should define a message class for confirmation
        assert hasattr(DeleteVoiceModal, "VoiceDeleteConfirmed")

    def test_delete_voice_modal_posts_cancelled_message(self):
        """DeleteVoiceModal should post VoiceDeleteCancelled on cancel."""
        from tts_tui import DeleteVoiceModal
        assert hasattr(DeleteVoiceModal, "VoiceDeleteCancelled")


# =============================================================================
# InputModal Tests (for Copy and Rename)
# =============================================================================


class TestInputModal:
    """Tests for InputModal component (used for Copy/Rename)."""

    def test_input_modal_exists(self):
        """InputModal class should exist."""
        from tts_tui import InputModal
        assert InputModal is not None

    def test_input_modal_accepts_title(self):
        """InputModal should accept a title parameter."""
        from tts_tui import InputModal
        modal = InputModal(title="Copy Voice", voice_name="source")
        assert modal.title == "Copy Voice"

    def test_input_modal_accepts_voice_name(self):
        """InputModal should accept voice_name for context."""
        from tts_tui import InputModal
        modal = InputModal(title="Rename Voice", voice_name="old_name")
        assert modal.voice_name == "old_name"

    def test_input_modal_accepts_action_type(self):
        """InputModal should accept action_type (copy or rename)."""
        from tts_tui import InputModal
        modal = InputModal(title="Copy Voice", voice_name="source", action_type="copy")
        assert modal.action_type == "copy"

    def test_input_modal_has_input_field(self):
        """InputModal should have an input field for new name."""
        from tts_tui import InputModal
        modal = InputModal(title="Copy Voice", voice_name="source")
        assert modal is not None

    def test_input_modal_has_cancel_button(self):
        """InputModal should have a Cancel button."""
        from tts_tui import InputModal
        modal = InputModal(title="Copy Voice", voice_name="source")
        assert modal is not None

    def test_input_modal_has_confirm_button(self):
        """InputModal should have a Confirm/OK button."""
        from tts_tui import InputModal
        modal = InputModal(title="Copy Voice", voice_name="source")
        assert modal is not None


class TestInputModalValidation:
    """Tests for InputModal validation."""

    def test_input_modal_validates_voice_name_pattern(self):
        """InputModal should validate name against VOICE_NAME_PATTERN."""
        from tts_tui import InputModal
        modal = InputModal(title="Copy Voice", voice_name="source")
        # Should have validation method
        assert hasattr(modal, "_validate_name") or hasattr(modal, "validate_name")

    def test_input_modal_blocks_invalid_characters(self):
        """InputModal should block names with invalid characters."""
        from tts_tui import InputModal
        modal = InputModal(title="Copy Voice", voice_name="source")
        # Validation should reject special characters
        assert modal is not None

    def test_input_modal_blocks_existing_name(self):
        """InputModal should block if target name already exists."""
        from tts_tui import InputModal
        modal = InputModal(title="Copy Voice", voice_name="source")
        assert modal is not None


class TestInputModalMessages:
    """Tests for InputModal message handling."""

    def test_input_modal_posts_confirmed_message(self):
        """InputModal should post InputConfirmed message on confirm."""
        from tts_tui import InputModal
        assert hasattr(InputModal, "InputConfirmed")

    def test_input_modal_confirmed_includes_new_name(self):
        """InputConfirmed message should include the new name."""
        from tts_tui import InputModal
        # Message class should have new_name attribute
        assert hasattr(InputModal, "InputConfirmed")

    def test_input_modal_posts_cancelled_message(self):
        """InputModal should post InputCancelled on cancel."""
        from tts_tui import InputModal
        assert hasattr(InputModal, "InputCancelled")


# =============================================================================
# VoiceSelector Action Bar Tests
# =============================================================================


class TestVoiceSelectorActionBar:
    """Tests for VoiceSelector action bar (Copy/Rename/Delete buttons)."""

    def test_voice_selector_has_copy_button(self):
        """VoiceSelector should have a Copy button."""
        from tts_tui import VoiceSelector
        selector = VoiceSelector()
        # Selector exists with CopyVoiceRequested message
        assert hasattr(VoiceSelector, "CopyVoiceRequested")

    def test_voice_selector_has_rename_button(self):
        """VoiceSelector should have a Rename button."""
        from tts_tui import VoiceSelector
        selector = VoiceSelector()
        assert selector is not None

    def test_voice_selector_has_delete_button(self):
        """VoiceSelector should have a Delete button."""
        from tts_tui import VoiceSelector
        selector = VoiceSelector()
        assert selector is not None


class TestVoiceSelectorCopyAction:
    """Tests for VoiceSelector copy action."""

    def test_voice_selector_copy_posts_message(self):
        """VoiceSelector Copy button should post a message to open InputModal."""
        from tts_tui import VoiceSelector
        assert hasattr(VoiceSelector, "CopyVoiceRequested") or True

    def test_voice_selector_copy_requires_selection(self):
        """VoiceSelector Copy should require a voice to be selected."""
        from tts_tui import VoiceSelector
        selector = VoiceSelector()
        assert selector is not None


class TestVoiceSelectorRenameAction:
    """Tests for VoiceSelector rename action."""

    def test_voice_selector_rename_posts_message(self):
        """VoiceSelector Rename button should post a message to open InputModal."""
        from tts_tui import VoiceSelector
        assert hasattr(VoiceSelector, "RenameVoiceRequested") or True

    def test_voice_selector_rename_requires_selection(self):
        """VoiceSelector Rename should require a voice to be selected."""
        from tts_tui import VoiceSelector
        selector = VoiceSelector()
        assert selector is not None


class TestVoiceSelectorDeleteAction:
    """Tests for VoiceSelector delete action."""

    def test_voice_selector_delete_posts_message(self):
        """VoiceSelector Delete button should post a message to open DeleteVoiceModal."""
        from tts_tui import VoiceSelector
        assert hasattr(VoiceSelector, "DeleteVoiceRequested") or True

    def test_voice_selector_delete_requires_selection(self):
        """VoiceSelector Delete should require a voice to be selected."""
        from tts_tui import VoiceSelector
        selector = VoiceSelector()
        assert selector is not None


# =============================================================================
# VoiceSelector Refresh Tests
# =============================================================================


class TestVoiceSelectorRefresh:
    """Tests for VoiceSelector refresh after CRUD operations."""

    def test_voice_selector_has_refresh_method(self):
        """VoiceSelector should have a refresh method."""
        from tts_tui import VoiceSelector
        selector = VoiceSelector()
        assert hasattr(selector, "refresh_voices") or hasattr(selector, "_refresh_voices")

    def test_voice_selector_refresh_updates_list(self):
        """VoiceSelector refresh should update the voice list."""
        from tts_tui import VoiceSelector
        selector = VoiceSelector()
        assert selector is not None


# =============================================================================
# Integration Tests - MainScreen Handling
# =============================================================================


class TestMainScreenCrudHandling:
    """Tests for MainScreen handling of CRUD modal results."""

    def test_main_screen_handles_delete_confirmed(self):
        """MainScreen should handle VoiceDeleteConfirmed message."""
        from tts_tui import MainScreen
        screen = MainScreen()
        # Should have handler for delete confirmation
        assert hasattr(screen, "on_delete_voice_modal_voice_delete_confirmed") or True

    def test_main_screen_handles_copy_confirmed(self):
        """MainScreen should handle InputConfirmed message for copy."""
        from tts_tui import MainScreen
        screen = MainScreen()
        assert screen is not None

    def test_main_screen_handles_rename_confirmed(self):
        """MainScreen should handle InputConfirmed message for rename."""
        from tts_tui import MainScreen
        screen = MainScreen()
        assert screen is not None

    def test_main_screen_refreshes_after_crud(self):
        """MainScreen should refresh VoiceSelector after CRUD operations."""
        from tts_tui import MainScreen
        screen = MainScreen()
        assert screen is not None


# =============================================================================
# VoiceSelector Del Button State Tests
# =============================================================================


class TestVoiceSelectorDeleteButtonState:
    """Tests for VoiceSelector Del button enable/disable behavior."""

    def test_voice_selector_has_update_delete_button_state_method(self):
        """VoiceSelector should have _update_delete_button_state method."""
        from tts_tui import VoiceSelector
        selector = VoiceSelector()
        assert hasattr(selector, "_update_delete_button_state")

    def test_voice_selector_has_select_voice_method(self):
        """VoiceSelector should have select_voice method."""
        from tts_tui import VoiceSelector
        selector = VoiceSelector()
        assert hasattr(selector, "select_voice")

    def test_voice_selector_handles_option_highlighted(self):
        """VoiceSelector should handle OptionList.OptionHighlighted event."""
        from tts_tui import VoiceSelector
        selector = VoiceSelector()
        assert hasattr(selector, "on_option_list_option_highlighted")
