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


# =============================================================================
# NEW TESTS FOR PER-VOICE SCHEMA (TDD - these should FAIL initially)
# =============================================================================


class TestVoiceDiscovery:
    """Tests for discovering voice files from assets directory."""

    def test_discover_voices_finds_wav_files(self):
        """discover_voices should find all .wav files in assets directory."""
        from tts_config import discover_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "voice1.wav").touch()
            (assets_dir / "voice2.wav").touch()
            (assets_dir / "not_a_voice.txt").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                voices = discover_voices()

            assert "voice1" in voices
            assert "voice2" in voices
            assert "not_a_voice" not in voices

    def test_discover_voices_returns_empty_when_no_assets(self):
        """discover_voices should return empty list when assets dir doesn't exist."""
        from tts_config import discover_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                voices = discover_voices()

            assert voices == []

    def test_discover_voices_returns_empty_when_no_wav_files(self):
        """discover_voices should return empty list when no .wav files exist."""
        from tts_config import discover_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "readme.txt").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                voices = discover_voices()

            assert voices == []

    def test_discover_voices_strips_wav_extension(self):
        """discover_voices should return voice names without .wav extension."""
        from tts_config import discover_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "my_voice.wav").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                voices = discover_voices()

            assert "my_voice" in voices
            assert "my_voice.wav" not in voices


class TestActiveVoice:
    """Tests for active voice management."""

    def test_get_active_voice_returns_default_when_not_set(self):
        """get_active_voice should return 'default_voice' when not configured."""
        from tts_config import get_active_voice

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"profiles": {}, "active_profile": "default"}
            voice = get_active_voice()

        assert voice == "default_voice"

    def test_get_active_voice_returns_configured_voice(self):
        """get_active_voice should return the configured active voice."""
        from tts_config import get_active_voice

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"active_voice": "my_custom_voice"}
            voice = get_active_voice()

        assert voice == "my_custom_voice"

    def test_set_active_voice_saves_config(self):
        """set_active_voice should save the voice to config."""
        from tts_config import set_active_voice

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save, \
             patch("tts_config.discover_voices") as mock_discover:
            mock_load.return_value = {"active_voice": "default_voice"}
            mock_discover.return_value = ["default_voice", "my_voice"]
            set_active_voice("my_voice")

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            assert saved_config["active_voice"] == "my_voice"

    def test_set_active_voice_validates_voice_exists(self):
        """set_active_voice should reject voices that don't exist."""
        from tts_config import set_active_voice

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.discover_voices") as mock_discover:
            mock_load.return_value = {"active_voice": "default_voice"}
            mock_discover.return_value = ["default_voice"]

            with pytest.raises(ValueError, match="not found"):
                set_active_voice("nonexistent_voice")


class TestSecureVoiceNameResolution:
    """Tests for secure voice name validation."""

    def test_resolve_voice_path_returns_valid_path(self):
        """resolve_voice_path should return full path for valid voice."""
        from tts_config import resolve_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            voice_file = assets_dir / "my_voice.wav"
            voice_file.touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                path = resolve_voice_path("my_voice")

            assert path == voice_file

    def test_resolve_voice_path_rejects_path_traversal(self):
        """resolve_voice_path should reject path traversal attempts."""
        from tts_config import resolve_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="Invalid voice name"):
                    resolve_voice_path("../../../etc/passwd")

    def test_resolve_voice_path_rejects_absolute_paths(self):
        """resolve_voice_path should reject absolute paths."""
        from tts_config import resolve_voice_path

        with pytest.raises(ValueError, match="Invalid voice name"):
            resolve_voice_path("/etc/passwd")

    def test_resolve_voice_path_rejects_special_characters(self):
        """resolve_voice_path should reject names with special characters."""
        from tts_config import resolve_voice_path

        with pytest.raises(ValueError, match="Invalid voice name"):
            resolve_voice_path("voice;rm -rf /")

    def test_resolve_voice_path_rejects_nonexistent_voice(self):
        """resolve_voice_path should reject voice files that don't exist."""
        from tts_config import resolve_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="not found"):
                    resolve_voice_path("nonexistent_voice")

    def test_resolve_voice_path_rejects_symlink_escape(self):
        """resolve_voice_path should reject symlinks pointing outside assets."""
        from tts_config import resolve_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            # Create a file outside assets
            outside_file = Path(tmpdir) / "secret.wav"
            outside_file.touch()
            # Create a symlink inside assets pointing to the outside file
            symlink = assets_dir / "evil_voice.wav"
            symlink.symlink_to(outside_file)

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="Invalid voice name"):
                    resolve_voice_path("evil_voice")


class TestPerVoiceConfig:
    """Tests for per-voice compressor/limiter configuration."""

    def test_get_voice_config_returns_defaults_for_unconfigured_voice(self):
        """get_voice_config should return default settings for unconfigured voice."""
        from tts_config import get_voice_config, DEFAULT_COMPRESSOR

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"voices": {}}
            config = get_voice_config("my_voice")

        assert config["compressor"] == DEFAULT_COMPRESSOR

    def test_get_voice_config_returns_voice_specific_settings(self):
        """get_voice_config should return voice-specific settings when configured."""
        from tts_config import get_voice_config

        custom_compressor = {"enabled": False, "gain_db": 12}
        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "voices": {
                    "my_voice": {"compressor": custom_compressor}
                }
            }
            config = get_voice_config("my_voice")

        assert config["compressor"]["enabled"] is False
        assert config["compressor"]["gain_db"] == 12

    def test_set_voice_config_saves_voice_settings(self):
        """set_voice_config should save settings for a specific voice."""
        from tts_config import set_voice_config

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {"voices": {}}
            set_voice_config("my_voice", {"compressor": {"gain_db": 10}})

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            assert saved_config["voices"]["my_voice"]["compressor"]["gain_db"] == 10

    def test_set_voice_config_merges_with_existing(self):
        """set_voice_config should merge new settings with existing voice config."""
        from tts_config import set_voice_config

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "voices": {
                    "my_voice": {"compressor": {"enabled": True, "gain_db": 5}}
                }
            }
            set_voice_config("my_voice", {"compressor": {"gain_db": 10}})

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            # gain_db should be updated
            assert saved_config["voices"]["my_voice"]["compressor"]["gain_db"] == 10
            # enabled should be preserved
            assert saved_config["voices"]["my_voice"]["compressor"]["enabled"] is True


class TestCascadingConfigResolution:
    """Tests for cascading config: defaults -> voice-specific overrides."""

    def test_get_effective_compressor_uses_defaults_when_no_voice_config(self):
        """get_effective_compressor should use defaults when voice has no config."""
        from tts_config import get_effective_compressor, DEFAULT_COMPRESSOR

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"voices": {}}
            compressor = get_effective_compressor("unconfigured_voice")

        assert compressor == DEFAULT_COMPRESSOR

    def test_get_effective_compressor_overrides_with_voice_settings(self):
        """get_effective_compressor should override defaults with voice settings."""
        from tts_config import get_effective_compressor, DEFAULT_COMPRESSOR

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "voices": {
                    "my_voice": {"compressor": {"gain_db": 15, "enabled": False}}
                }
            }
            compressor = get_effective_compressor("my_voice")

        # Voice-specific overrides
        assert compressor["gain_db"] == 15
        assert compressor["enabled"] is False
        # Defaults for unspecified keys
        assert compressor["threshold_db"] == DEFAULT_COMPRESSOR["threshold_db"]
        assert compressor["ratio"] == DEFAULT_COMPRESSOR["ratio"]

    def test_get_effective_compressor_for_active_voice(self):
        """get_effective_compressor with no arg should use active voice."""
        from tts_config import get_effective_compressor

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "my_voice",
                "voices": {
                    "my_voice": {"compressor": {"gain_db": 20}}
                }
            }
            compressor = get_effective_compressor()

        assert compressor["gain_db"] == 20


class TestDefaultLimiterConfig:
    """Tests for limiter configuration (separate from compressor)."""

    def test_default_limiter_config_exists(self):
        """DEFAULT_LIMITER should be defined with sensible defaults."""
        from tts_config import DEFAULT_LIMITER

        assert "enabled" in DEFAULT_LIMITER
        assert "threshold_db" in DEFAULT_LIMITER
        assert "release_ms" in DEFAULT_LIMITER

    def test_get_effective_limiter_uses_defaults(self):
        """get_effective_limiter should use defaults for unconfigured voice."""
        from tts_config import get_effective_limiter, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"voices": {}}
            limiter = get_effective_limiter("unconfigured_voice")

        assert limiter == DEFAULT_LIMITER

    def test_get_effective_limiter_overrides_with_voice_settings(self):
        """get_effective_limiter should override defaults with voice settings."""
        from tts_config import get_effective_limiter, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "voices": {
                    "my_voice": {"limiter": {"threshold_db": -1.0}}
                }
            }
            limiter = get_effective_limiter("my_voice")

        assert limiter["threshold_db"] == -1.0
        # Defaults for unspecified keys
        assert limiter["release_ms"] == DEFAULT_LIMITER["release_ms"]


class TestGlobalLimiterConfig:
    """Tests for global limiter configuration (get_limiter_config, set_limiter_setting)."""

    def test_get_limiter_config_returns_defaults_when_not_set(self):
        """get_limiter_config should return defaults when no limiter config exists."""
        from tts_config import get_limiter_config, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {}
            config = get_limiter_config()

        assert config == DEFAULT_LIMITER

    def test_get_limiter_config_merges_with_defaults(self):
        """get_limiter_config should merge file config with defaults."""
        from tts_config import get_limiter_config, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"limiter": {"threshold_db": -5.0}}
            config = get_limiter_config()

        assert config["threshold_db"] == -5.0
        assert config["release_ms"] == DEFAULT_LIMITER["release_ms"]
        assert config["enabled"] == DEFAULT_LIMITER["enabled"]

    def test_set_limiter_setting_saves_config(self):
        """set_limiter_setting should save the setting to config."""
        from tts_config import set_limiter_setting

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {}
            set_limiter_setting("threshold_db", -3.0)

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            assert saved_config["limiter"]["threshold_db"] == -3.0

    def test_set_limiter_setting_validates_key(self):
        """set_limiter_setting should reject invalid keys."""
        from tts_config import set_limiter_setting

        with pytest.raises(ValueError, match="Invalid limiter key"):
            set_limiter_setting("invalid_key", 1.0)

    def test_set_limiter_setting_enabled(self):
        """set_limiter_setting should accept boolean for enabled."""
        from tts_config import set_limiter_setting

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {}
            set_limiter_setting("enabled", False)

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            assert saved_config["limiter"]["enabled"] is False


# =============================================================================
# Input Gain and Master Gain Config Tests (TDD - RED PHASE)
# =============================================================================


class TestInputGainConfig:
    """Tests for input gain configuration."""

    def test_default_compressor_has_input_gain(self):
        """DEFAULT_COMPRESSOR should include input_gain_db."""
        from tts_config import DEFAULT_COMPRESSOR
        assert "input_gain_db" in DEFAULT_COMPRESSOR

    def test_default_input_gain_is_zero(self):
        """Default input gain should be 0 dB (unity)."""
        from tts_config import DEFAULT_COMPRESSOR
        assert DEFAULT_COMPRESSOR["input_gain_db"] == 0.0

    def test_get_compressor_config_returns_input_gain(self):
        """get_compressor_config should return input_gain_db."""
        from tts_config import get_compressor_config

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {}
            config = get_compressor_config()

        assert "input_gain_db" in config

    def test_set_compressor_setting_accepts_input_gain(self):
        """set_compressor_setting should accept input_gain_db."""
        from tts_config import set_compressor_setting

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {}
            set_compressor_setting("input_gain_db", 6.0)

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            assert saved_config["compressor"]["input_gain_db"] == 6.0


class TestMasterGainConfig:
    """Tests for master gain configuration."""

    def test_default_compressor_has_master_gain(self):
        """DEFAULT_COMPRESSOR should include master_gain_db."""
        from tts_config import DEFAULT_COMPRESSOR
        assert "master_gain_db" in DEFAULT_COMPRESSOR

    def test_default_master_gain_is_zero(self):
        """Default master gain should be 0 dB (unity)."""
        from tts_config import DEFAULT_COMPRESSOR
        assert DEFAULT_COMPRESSOR["master_gain_db"] == 0.0

    def test_get_compressor_config_returns_master_gain(self):
        """get_compressor_config should return master_gain_db."""
        from tts_config import get_compressor_config

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {}
            config = get_compressor_config()

        assert "master_gain_db" in config

    def test_set_compressor_setting_accepts_master_gain(self):
        """set_compressor_setting should accept master_gain_db."""
        from tts_config import set_compressor_setting

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {}
            set_compressor_setting("master_gain_db", -3.0)

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            assert saved_config["compressor"]["master_gain_db"] == -3.0


class TestEffectiveGainSettings:
    """Tests for effective gain settings with voice overrides."""

    def test_get_effective_compressor_includes_input_gain(self):
        """get_effective_compressor should include input_gain_db."""
        from tts_config import get_effective_compressor

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"voices": {}}
            compressor = get_effective_compressor("test_voice")

        assert "input_gain_db" in compressor

    def test_get_effective_compressor_includes_master_gain(self):
        """get_effective_compressor should include master_gain_db."""
        from tts_config import get_effective_compressor

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"voices": {}}
            compressor = get_effective_compressor("test_voice")

        assert "master_gain_db" in compressor

    def test_voice_can_override_input_gain(self):
        """Voice-specific input_gain_db should override default."""
        from tts_config import get_effective_compressor

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "voices": {
                    "loud_voice": {"compressor": {"input_gain_db": 6.0}}
                }
            }
            compressor = get_effective_compressor("loud_voice")

        assert compressor["input_gain_db"] == 6.0

    def test_voice_can_override_master_gain(self):
        """Voice-specific master_gain_db should override default."""
        from tts_config import get_effective_compressor

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "voices": {
                    "quiet_voice": {"compressor": {"master_gain_db": -6.0}}
                }
            }
            compressor = get_effective_compressor("quiet_voice")

        assert compressor["master_gain_db"] == -6.0
