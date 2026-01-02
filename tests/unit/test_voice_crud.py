"""
Unit tests for Voice CRUD operations (mlx-tts-7q8).

Tests for:
- get_voice_usage(name) - list where voice is referenced
- delete_voice(name) - remove file + clean up config references
- rename_voice(old, new) - rename file + update config references
- copy_voice(source, target) - copy file + settings

Run with: uv run pytest tests/unit/test_voice_crud.py -v
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


# =============================================================================
# get_voice_usage() Tests
# =============================================================================


class TestGetVoiceUsage:
    """Tests for get_voice_usage(name) -> dict."""

    def test_get_voice_usage_returns_dict(self):
        """get_voice_usage should return a dict with usage info."""
        from tts_config import get_voice_usage

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "default",
                "voices": {},
                "hooks": {}
            }
            result = get_voice_usage("default")

        assert isinstance(result, dict)

    def test_get_voice_usage_detects_active_voice(self):
        """get_voice_usage should detect when voice is the active voice."""
        from tts_config import get_voice_usage

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "my_voice",
                "voices": {},
                "hooks": {}
            }
            result = get_voice_usage("my_voice")

        assert result["is_active"] is True

    def test_get_voice_usage_detects_not_active(self):
        """get_voice_usage should detect when voice is not active."""
        from tts_config import get_voice_usage

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "other_voice",
                "voices": {},
                "hooks": {}
            }
            result = get_voice_usage("my_voice")

        assert result["is_active"] is False

    def test_get_voice_usage_detects_hook_usage(self):
        """get_voice_usage should detect when voice is used by hooks."""
        from tts_config import get_voice_usage

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "default",
                "voices": {},
                "hooks": {
                    "stop": {"voice": "my_voice"},
                    "permission_request": {"voice": "other_voice"}
                }
            }
            result = get_voice_usage("my_voice")

        assert "stop" in result["hooks"]
        assert "permission_request" not in result["hooks"]

    def test_get_voice_usage_detects_multiple_hooks(self):
        """get_voice_usage should detect when voice is used by multiple hooks."""
        from tts_config import get_voice_usage

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "default",
                "voices": {},
                "hooks": {
                    "stop": {"voice": "my_voice"},
                    "permission_request": {"voice": "my_voice"}
                }
            }
            result = get_voice_usage("my_voice")

        assert "stop" in result["hooks"]
        assert "permission_request" in result["hooks"]

    def test_get_voice_usage_empty_when_not_used(self):
        """get_voice_usage should return empty hooks list when voice not used."""
        from tts_config import get_voice_usage

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "default",
                "voices": {},
                "hooks": {}
            }
            result = get_voice_usage("unused_voice")

        assert result["is_active"] is False
        assert result["hooks"] == []

    def test_get_voice_usage_has_voice_settings(self):
        """get_voice_usage should detect when voice has custom settings."""
        from tts_config import get_voice_usage

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "default",
                "voices": {
                    "my_voice": {"compressor": {"gain_db": 10}}
                },
                "hooks": {}
            }
            result = get_voice_usage("my_voice")

        assert result["has_settings"] is True

    def test_get_voice_usage_no_settings_when_empty(self):
        """get_voice_usage should detect when voice has no custom settings."""
        from tts_config import get_voice_usage

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "default",
                "voices": {},
                "hooks": {}
            }
            result = get_voice_usage("my_voice")

        assert result["has_settings"] is False


# =============================================================================
# delete_voice() Tests
# =============================================================================


class TestDeleteVoice:
    """Tests for delete_voice(name)."""

    def test_delete_voice_removes_file(self):
        """delete_voice should remove the voice file from assets."""
        from tts_config import delete_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            voice_file = assets_dir / "my_voice.safetensors"
            voice_file.touch()

            # Create another voice so we're not deleting the last one
            other_voice = assets_dir / "other.safetensors"
            other_voice.touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config"):
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {}
                }
                delete_voice("my_voice")

            assert not voice_file.exists()

    def test_delete_voice_removes_wav_file(self):
        """delete_voice should remove .wav files."""
        from tts_config import delete_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            voice_file = assets_dir / "my_voice.wav"
            voice_file.touch()
            other_voice = assets_dir / "other.wav"
            other_voice.touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config"):
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {}
                }
                delete_voice("my_voice")

            assert not voice_file.exists()

    def test_delete_voice_removes_both_formats(self):
        """delete_voice should remove both .wav and .safetensors if both exist."""
        from tts_config import delete_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            wav_file = assets_dir / "my_voice.wav"
            wav_file.touch()
            safetensors_file = assets_dir / "my_voice.safetensors"
            safetensors_file.touch()
            other_voice = assets_dir / "other.safetensors"
            other_voice.touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config"):
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {}
                }
                delete_voice("my_voice")

            assert not wav_file.exists()
            assert not safetensors_file.exists()

    def test_delete_voice_removes_voice_settings(self):
        """delete_voice should remove voice settings from config."""
        from tts_config import delete_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "my_voice.safetensors").touch()
            (assets_dir / "other.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config") as mock_save:
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {
                        "my_voice": {"compressor": {"gain_db": 10}},
                        "other": {"compressor": {"gain_db": 5}}
                    },
                    "hooks": {}
                }
                delete_voice("my_voice")

                saved = mock_save.call_args[0][0]
                assert "my_voice" not in saved["voices"]
                assert "other" in saved["voices"]

    def test_delete_voice_clears_hook_overrides(self):
        """delete_voice should clear hook voice overrides that reference the voice."""
        from tts_config import delete_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "my_voice.safetensors").touch()
            (assets_dir / "other.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config") as mock_save:
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {
                        "stop": {"voice": "my_voice"},
                        "permission_request": {"voice": "other"}
                    }
                }
                delete_voice("my_voice")

                saved = mock_save.call_args[0][0]
                # stop hook should have voice cleared
                assert "voice" not in saved["hooks"].get("stop", {})
                # permission_request should be unchanged
                assert saved["hooks"]["permission_request"]["voice"] == "other"

    def test_delete_voice_updates_active_voice_if_deleted(self):
        """delete_voice should set active_voice to first remaining voice if deleted."""
        from tts_config import delete_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "my_voice.safetensors").touch()
            (assets_dir / "alpha.safetensors").touch()
            (assets_dir / "beta.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config") as mock_save:
                mock_load.return_value = {
                    "active_voice": "my_voice",
                    "voices": {},
                    "hooks": {}
                }
                delete_voice("my_voice")

                saved = mock_save.call_args[0][0]
                # Should be set to first remaining voice (sorted alphabetically)
                assert saved["active_voice"] in ["alpha", "beta"]

    def test_delete_voice_blocks_last_voice(self):
        """delete_voice should raise error when trying to delete last voice."""
        from tts_config import delete_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "only_voice.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load:
                mock_load.return_value = {
                    "active_voice": "only_voice",
                    "voices": {},
                    "hooks": {}
                }
                with pytest.raises(ValueError, match="Cannot delete.*last voice"):
                    delete_voice("only_voice")

    def test_delete_voice_raises_for_nonexistent(self):
        """delete_voice should raise error for nonexistent voice."""
        from tts_config import delete_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "other.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="not found"):
                    delete_voice("nonexistent")

    def test_delete_voice_validates_name(self):
        """delete_voice should validate voice name for security."""
        from tts_config import delete_voice

        with pytest.raises(ValueError, match="Invalid voice name"):
            delete_voice("../../../etc/passwd")


# =============================================================================
# rename_voice() Tests
# =============================================================================


class TestRenameVoice:
    """Tests for rename_voice(old_name, new_name)."""

    def test_rename_voice_renames_file(self):
        """rename_voice should rename the voice file."""
        from tts_config import rename_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            old_file = assets_dir / "old_voice.safetensors"
            old_file.touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config"):
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {}
                }
                rename_voice("old_voice", "new_voice")

            assert not old_file.exists()
            assert (assets_dir / "new_voice.safetensors").exists()

    def test_rename_voice_renames_wav_file(self):
        """rename_voice should rename .wav files."""
        from tts_config import rename_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            old_file = assets_dir / "old_voice.wav"
            old_file.touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config"):
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {}
                }
                rename_voice("old_voice", "new_voice")

            assert not old_file.exists()
            assert (assets_dir / "new_voice.wav").exists()

    def test_rename_voice_renames_both_formats(self):
        """rename_voice should rename both .wav and .safetensors if both exist."""
        from tts_config import rename_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "old.wav").touch()
            (assets_dir / "old.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config"):
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {}
                }
                rename_voice("old", "new")

            assert not (assets_dir / "old.wav").exists()
            assert not (assets_dir / "old.safetensors").exists()
            assert (assets_dir / "new.wav").exists()
            assert (assets_dir / "new.safetensors").exists()

    def test_rename_voice_moves_voice_settings(self):
        """rename_voice should move voice settings to new name."""
        from tts_config import rename_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "old.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config") as mock_save:
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {
                        "old": {"compressor": {"gain_db": 10}}
                    },
                    "hooks": {}
                }
                rename_voice("old", "new")

                saved = mock_save.call_args[0][0]
                assert "old" not in saved["voices"]
                assert saved["voices"]["new"]["compressor"]["gain_db"] == 10

    def test_rename_voice_updates_active_voice(self):
        """rename_voice should update active_voice if renaming active voice."""
        from tts_config import rename_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "old.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config") as mock_save:
                mock_load.return_value = {
                    "active_voice": "old",
                    "voices": {},
                    "hooks": {}
                }
                rename_voice("old", "new")

                saved = mock_save.call_args[0][0]
                assert saved["active_voice"] == "new"

    def test_rename_voice_updates_hook_references(self):
        """rename_voice should update hook voice overrides."""
        from tts_config import rename_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "old.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config") as mock_save:
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {
                        "stop": {"voice": "old"},
                        "permission_request": {"voice": "other"}
                    }
                }
                rename_voice("old", "new")

                saved = mock_save.call_args[0][0]
                assert saved["hooks"]["stop"]["voice"] == "new"
                assert saved["hooks"]["permission_request"]["voice"] == "other"

    def test_rename_voice_blocks_name_collision(self):
        """rename_voice should block if target name already exists."""
        from tts_config import rename_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "old.safetensors").touch()
            (assets_dir / "existing.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="already exists"):
                    rename_voice("old", "existing")

    def test_rename_voice_validates_new_name(self):
        """rename_voice should validate new name for invalid characters."""
        from tts_config import rename_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "old.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="Invalid voice name"):
                    rename_voice("old", "invalid;name")

    def test_rename_voice_raises_for_nonexistent(self):
        """rename_voice should raise error for nonexistent voice."""
        from tts_config import rename_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="not found"):
                    rename_voice("nonexistent", "new_name")


# =============================================================================
# copy_voice() Tests
# =============================================================================


class TestGenerateCopyName:
    """Tests for generate_copy_name helper."""

    def test_generate_copy_name_adds_suffix(self):
        """generate_copy_name should add _copy suffix."""
        from tts_config import generate_copy_name

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "voice.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                name = generate_copy_name("voice")

            assert name == "voice_copy"

    def test_generate_copy_name_increments_on_collision(self):
        """generate_copy_name should add numbers on collision."""
        from tts_config import generate_copy_name

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "voice.safetensors").touch()
            (assets_dir / "voice_copy.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                name = generate_copy_name("voice")

            assert name == "voice_copy2"

    def test_generate_copy_name_finds_next_available(self):
        """generate_copy_name should find next available number."""
        from tts_config import generate_copy_name

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "voice.safetensors").touch()
            (assets_dir / "voice_copy.safetensors").touch()
            (assets_dir / "voice_copy2.safetensors").touch()
            (assets_dir / "voice_copy3.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                name = generate_copy_name("voice")

            assert name == "voice_copy4"


class TestCopyVoice:
    """Tests for copy_voice(source, target)."""

    def test_copy_voice_copies_file(self):
        """copy_voice should copy the voice file."""
        from tts_config import copy_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            source_file = assets_dir / "source.safetensors"
            source_file.write_bytes(b"voice data")

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config"):
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {}
                }
                result = copy_voice("source", "target")

            assert result == "target"
            assert source_file.exists()  # Source still exists
            assert (assets_dir / "target.safetensors").exists()
            assert (assets_dir / "target.safetensors").read_bytes() == b"voice data"

    def test_copy_voice_auto_generates_name(self):
        """copy_voice should auto-generate name when target is None."""
        from tts_config import copy_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "source.safetensors").write_bytes(b"voice data")

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config"):
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {}
                }
                result = copy_voice("source")

            assert result == "source_copy"
            assert (assets_dir / "source_copy.safetensors").exists()

    def test_copy_voice_copies_wav_file(self):
        """copy_voice should copy .wav files."""
        from tts_config import copy_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            source_file = assets_dir / "source.wav"
            source_file.write_bytes(b"wav data")

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config"):
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {}
                }
                copy_voice("source", "target")

            assert source_file.exists()
            assert (assets_dir / "target.wav").exists()

    def test_copy_voice_copies_both_formats(self):
        """copy_voice should copy both formats if both exist."""
        from tts_config import copy_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "source.wav").write_bytes(b"wav")
            (assets_dir / "source.safetensors").write_bytes(b"safetensors")

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config"):
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {}
                }
                copy_voice("source", "target")

            assert (assets_dir / "target.wav").exists()
            assert (assets_dir / "target.safetensors").exists()

    def test_copy_voice_copies_settings(self):
        """copy_voice should copy voice settings."""
        from tts_config import copy_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "source.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config") as mock_save:
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {
                        "source": {
                            "compressor": {"gain_db": 10},
                            "limiter": {"threshold_db": -2.0}
                        }
                    },
                    "hooks": {}
                }
                copy_voice("source", "target")

                saved = mock_save.call_args[0][0]
                # Source still has settings
                assert saved["voices"]["source"]["compressor"]["gain_db"] == 10
                # Target has copied settings
                assert saved["voices"]["target"]["compressor"]["gain_db"] == 10
                assert saved["voices"]["target"]["limiter"]["threshold_db"] == -2.0

    def test_copy_voice_no_settings_if_source_has_none(self):
        """copy_voice should not create settings if source has none."""
        from tts_config import copy_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "source.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config") as mock_save:
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {}
                }
                copy_voice("source", "target")

                saved = mock_save.call_args[0][0]
                # No settings for either source or target
                assert "source" not in saved["voices"]
                assert "target" not in saved["voices"]

    def test_copy_voice_blocks_name_collision(self):
        """copy_voice should block if target name already exists."""
        from tts_config import copy_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "source.safetensors").touch()
            (assets_dir / "existing.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="already exists"):
                    copy_voice("source", "existing")

    def test_copy_voice_validates_target_name(self):
        """copy_voice should validate target name for invalid characters."""
        from tts_config import copy_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "source.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="Invalid voice name"):
                    copy_voice("source", "invalid;name")

    def test_copy_voice_raises_for_nonexistent_source(self):
        """copy_voice should raise error for nonexistent source."""
        from tts_config import copy_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="not found"):
                    copy_voice("nonexistent", "target")

    def test_copy_voice_does_not_copy_hook_references(self):
        """copy_voice should not copy hook voice overrides."""
        from tts_config import copy_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "source.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config") as mock_save:
                mock_load.return_value = {
                    "active_voice": "other",
                    "voices": {},
                    "hooks": {
                        "stop": {"voice": "source"}
                    }
                }
                copy_voice("source", "target")

                saved = mock_save.call_args[0][0]
                # Hooks should not reference target
                assert saved["hooks"]["stop"]["voice"] == "source"


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestVoiceCrudIntegration:
    """Integration tests for voice CRUD operations."""

    def test_delete_then_copy_workflow(self):
        """Test deleting a voice then copying another."""
        from tts_config import delete_voice, copy_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "voice_a.safetensors").touch()
            (assets_dir / "voice_b.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config"):
                mock_load.return_value = {
                    "active_voice": "voice_b",
                    "voices": {},
                    "hooks": {}
                }
                delete_voice("voice_a")
                copy_voice("voice_b", "voice_a")

            assert (assets_dir / "voice_a.safetensors").exists()
            assert (assets_dir / "voice_b.safetensors").exists()

    def test_rename_updates_all_references(self):
        """Test that rename updates all config references."""
        from tts_config import rename_voice

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "old.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)), \
                 patch("tts_config.load_config") as mock_load, \
                 patch("tts_config.save_config") as mock_save:
                mock_load.return_value = {
                    "active_voice": "old",
                    "voices": {
                        "old": {"compressor": {"gain_db": 10}}
                    },
                    "hooks": {
                        "stop": {"voice": "old"},
                        "permission_request": {"voice": "old"}
                    }
                }
                rename_voice("old", "new")

                saved = mock_save.call_args[0][0]
                assert saved["active_voice"] == "new"
                assert "old" not in saved["voices"]
                assert saved["voices"]["new"]["compressor"]["gain_db"] == 10
                assert saved["hooks"]["stop"]["voice"] == "new"
                assert saved["hooks"]["permission_request"]["voice"] == "new"
