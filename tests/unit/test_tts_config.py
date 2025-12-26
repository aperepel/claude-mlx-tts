"""
Unit tests for TTS configuration module.

Run with: uv run pytest tests/unit/test_tts_config.py -v
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestConfigDefaults:
    """Tests for default configuration values."""

    def test_default_speed_is_normal(self):
        """Default playback speed should be 1.3x (normal)."""
        from tts_config import DEFAULT_CONFIG, DEFAULT_SPEED

        assert DEFAULT_SPEED == 1.3
        assert DEFAULT_CONFIG["profiles"]["default"]["speed"] == 1.3

    def test_default_profile_is_default(self):
        """Default active profile should be 'default'."""
        from tts_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["active_profile"] == "default"

    def test_speed_presets_defined(self):
        """Speed presets should be defined with labels."""
        from tts_config import SPEED_PRESETS

        assert 1.0 in SPEED_PRESETS
        assert 1.3 in SPEED_PRESETS
        assert 1.6 in SPEED_PRESETS
        assert 2.0 in SPEED_PRESETS

    def test_speed_preset_labels(self):
        """Speed presets should have descriptive labels."""
        from tts_config import SPEED_PRESETS

        assert SPEED_PRESETS[1.0] == "Slow"
        assert SPEED_PRESETS[1.3] == "Normal"
        assert SPEED_PRESETS[1.6] == "Fast"
        assert SPEED_PRESETS[2.0] == "Turbo"


class TestConfigPath:
    """Tests for configuration file path handling."""

    def test_config_dir_is_plugin_local(self):
        """Config directory should be ${PLUGIN_ROOT}/.config/."""
        from tts_config import get_config_dir, _PLUGIN_ROOT

        config_dir = get_config_dir()
        assert config_dir == _PLUGIN_ROOT / ".config"

    def test_config_file_path(self):
        """Config file should be config.json in .config dir."""
        from tts_config import get_config_path

        config_path = get_config_path()
        assert config_path.name == "config.json"
        assert config_path.parent.name == ".config"


class TestLoadConfig:
    """Tests for loading configuration."""

    def test_load_config_returns_defaults_when_no_file(self):
        """load_config should return defaults when config file doesn't exist."""
        from tts_config import load_config, DEFAULT_CONFIG

        with patch("tts_config.get_config_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/path/config.json")
            config = load_config()

        assert config == DEFAULT_CONFIG

    def test_load_config_reads_existing_file(self):
        """load_config should read from existing config file."""
        from tts_config import load_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text(json.dumps({"playback_speed": 2.0}))

            with patch("tts_config.get_config_path") as mock_path:
                mock_path.return_value = config_file
                config = load_config()

            assert config["playback_speed"] == 2.0

    def test_load_config_merges_with_defaults(self):
        """load_config should merge file config with defaults for missing keys."""
        from tts_config import load_config, DEFAULT_CONFIG

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            # Partial config - missing some keys
            config_file.write_text(json.dumps({"playback_speed": 1.6}))

            with patch("tts_config.get_config_path") as mock_path:
                mock_path.return_value = config_file
                config = load_config()

            # Should have the file value
            assert config["playback_speed"] == 1.6
            # Should have defaults for any other keys that exist in DEFAULT_CONFIG

    def test_load_config_handles_corrupted_json(self):
        """load_config should return defaults for corrupted JSON files."""
        from tts_config import load_config, DEFAULT_CONFIG

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text("not valid json {{{")

            with patch("tts_config.get_config_path") as mock_path:
                mock_path.return_value = config_file
                config = load_config()

            assert config == DEFAULT_CONFIG


class TestSaveConfig:
    """Tests for saving configuration."""

    def test_save_config_creates_directory(self):
        """save_config should create config directory if it doesn't exist."""
        from tts_config import save_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "new_dir"
            config_file = config_dir / "config.json"

            with patch("tts_config.get_config_path") as mock_path:
                mock_path.return_value = config_file
                save_config({"playback_speed": 1.6})

            assert config_dir.exists()

    def test_save_config_writes_json(self):
        """save_config should write valid JSON to file."""
        from tts_config import save_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            with patch("tts_config.get_config_path") as mock_path:
                mock_path.return_value = config_file
                save_config({"playback_speed": 2.0})

            assert config_file.exists()
            saved_data = json.loads(config_file.read_text())
            assert saved_data["playback_speed"] == 2.0

    def test_save_config_overwrites_existing(self):
        """save_config should overwrite existing config file."""
        from tts_config import save_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text(json.dumps({"playback_speed": 1.0}))

            with patch("tts_config.get_config_path") as mock_path:
                mock_path.return_value = config_file
                save_config({"playback_speed": 2.0})

            saved_data = json.loads(config_file.read_text())
            assert saved_data["playback_speed"] == 2.0


class TestGetPlaybackSpeed:
    """Tests for getting current playback speed."""

    def test_get_playback_speed_returns_float(self):
        """get_playback_speed should return a float."""
        from tts_config import get_playback_speed

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "profiles": {"default": {"speed": 1.6}},
                "active_profile": "default"
            }
            speed = get_playback_speed()

        assert isinstance(speed, float)
        assert speed == 1.6

    def test_get_playback_speed_uses_default(self):
        """get_playback_speed should use default when not configured."""
        from tts_config import get_playback_speed, DEFAULT_SPEED

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"profiles": {}, "active_profile": "default"}
            speed = get_playback_speed()

        assert speed == DEFAULT_SPEED


class TestSetPlaybackSpeed:
    """Tests for setting playback speed."""

    def test_set_playback_speed_saves_config(self):
        """set_playback_speed should save the speed to config."""
        from tts_config import set_playback_speed

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "profiles": {"default": {"speed": 1.3}},
                "active_profile": "default"
            }
            set_playback_speed(2.0)

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            assert saved_config["profiles"]["default"]["speed"] == 2.0

    def test_set_playback_speed_validates_preset(self):
        """set_playback_speed should only accept valid preset values."""
        from tts_config import set_playback_speed, SPEED_PRESETS

        with pytest.raises(ValueError):
            set_playback_speed(999.0)  # Invalid speed


class TestFormatCurrentConfig:
    """Tests for displaying current configuration."""

    def test_format_current_config_includes_speed(self):
        """format_current_config should include current speed with label."""
        from tts_config import format_current_config

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "profiles": {"default": {"speed": 1.6}},
                "active_profile": "default"
            }
            output = format_current_config()

        assert "1.6" in output
        assert "Fast" in output

    def test_format_current_config_readable(self):
        """format_current_config should return human-readable string."""
        from tts_config import format_current_config

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "profiles": {"default": {"speed": 1.3}},
                "active_profile": "default"
            }
            output = format_current_config()

        assert isinstance(output, str)
        assert len(output) > 0


class TestTtsConfigCli:
    """Tests for the tts-config CLI commands."""

    def test_cli_show_displays_current_config(self):
        """tts-config show should display current configuration."""
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "tts_config.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text(json.dumps({
                "profiles": {"default": {"speed": 1.6}},
                "active_profile": "default"
            }))

            with patch.dict(os.environ, {"TTS_CONFIG_PATH": str(config_file)}):
                result = subprocess.run(
                    [sys.executable, script_path, "show"],
                    capture_output=True,
                    text=True,
                    env={**os.environ, "TTS_CONFIG_PATH": str(config_file)}
                )

            assert result.returncode == 0
            assert "1.6" in result.stdout
            assert "Fast" in result.stdout

    def test_cli_set_updates_speed(self):
        """tts-config set <speed> should update the playback speed."""
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "tts_config.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            result = subprocess.run(
                [sys.executable, script_path, "set", "2.0"],
                capture_output=True,
                text=True,
                env={**os.environ, "TTS_CONFIG_PATH": str(config_file)}
            )

            assert result.returncode == 0
            assert config_file.exists()
            saved_config = json.loads(config_file.read_text())
            assert saved_config["profiles"]["default"]["speed"] == 2.0

    def test_cli_set_invalid_speed_fails(self):
        """tts-config set with invalid speed should fail."""
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "tts_config.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            result = subprocess.run(
                [sys.executable, script_path, "set", "999"],
                capture_output=True,
                text=True,
                env={**os.environ, "TTS_CONFIG_PATH": str(config_file)}
            )

            assert result.returncode != 0

    def test_cli_no_args_shows_config(self):
        """tts-config with no args should show current config."""
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "tts_config.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text(json.dumps({
                "profiles": {"default": {"speed": 1.3}},
                "active_profile": "default"
            }))

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                env={**os.environ, "TTS_CONFIG_PATH": str(config_file)}
            )

            assert result.returncode == 0
            assert "1.3" in result.stdout
            assert "Normal" in result.stdout

    def test_cli_wizard_outputs_options(self):
        """tts-config wizard should output available speed options."""
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "tts_config.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            result = subprocess.run(
                [sys.executable, script_path, "wizard"],
                capture_output=True,
                text=True,
                env={**os.environ, "TTS_CONFIG_PATH": str(config_file)}
            )

            assert result.returncode == 0
            # Should list all speed options
            assert "Slow" in result.stdout
            assert "Normal" in result.stdout
            assert "Fast" in result.stdout
            assert "Turbo" in result.stdout
