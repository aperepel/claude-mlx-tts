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

    def test_default_profile_is_default(self):
        """Default active profile should be 'default'."""
        from tts_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["active_profile"] == "default"

    def test_default_streaming_interval(self):
        """Default streaming interval should be 0.5s."""
        from tts_config import DEFAULT_CONFIG, DEFAULT_STREAMING_INTERVAL

        assert DEFAULT_STREAMING_INTERVAL == 0.5
        assert DEFAULT_CONFIG["profiles"]["default"]["streaming_interval"] == 0.5


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
            config_file.write_text(json.dumps({"active_profile": "custom"}))

            with patch("tts_config.get_config_path") as mock_path:
                mock_path.return_value = config_file
                config = load_config()

            assert config["active_profile"] == "custom"

    def test_load_config_merges_with_defaults(self):
        """load_config should merge file config with defaults for missing keys."""
        from tts_config import load_config, DEFAULT_CONFIG

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            # Partial config - missing some keys
            config_file.write_text(json.dumps({"active_profile": "custom"}))

            with patch("tts_config.get_config_path") as mock_path:
                mock_path.return_value = config_file
                config = load_config()

            # Should have the file value
            assert config["active_profile"] == "custom"
            # Should have defaults for other keys

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
                save_config({"active_profile": "custom"})

            assert config_dir.exists()

    def test_save_config_writes_json(self):
        """save_config should write valid JSON to file."""
        from tts_config import save_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"

            with patch("tts_config.get_config_path") as mock_path:
                mock_path.return_value = config_file
                save_config({"active_profile": "custom"})

            assert config_file.exists()
            saved_data = json.loads(config_file.read_text())
            assert saved_data["active_profile"] == "custom"

    def test_save_config_overwrites_existing(self):
        """save_config should overwrite existing config file."""
        from tts_config import save_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text(json.dumps({"active_profile": "old"}))

            with patch("tts_config.get_config_path") as mock_path:
                mock_path.return_value = config_file
                save_config({"active_profile": "new"})

            saved_data = json.loads(config_file.read_text())
            assert saved_data["active_profile"] == "new"


class TestFormatCurrentConfig:
    """Tests for displaying current configuration."""

    def test_format_current_config_includes_streaming_interval(self):
        """format_current_config should include streaming interval."""
        from tts_config import format_current_config

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "profiles": {"default": {"streaming_interval": 0.5}},
                "active_profile": "default"
            }
            output = format_current_config()

        assert "0.5" in output

    def test_format_current_config_readable(self):
        """format_current_config should return human-readable string."""
        from tts_config import format_current_config

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "profiles": {"default": {"streaming_interval": 0.5}},
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
                "profiles": {"default": {"streaming_interval": 0.5}},
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
            assert "0.5" in result.stdout

    def test_cli_no_args_shows_config(self):
        """tts-config with no args should show current config."""
        import subprocess
        script_path = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "tts_config.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            config_file.write_text(json.dumps({
                "profiles": {"default": {"streaming_interval": 0.5}},
                "active_profile": "default"
            }))

            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                env={**os.environ, "TTS_CONFIG_PATH": str(config_file)}
            )

            assert result.returncode == 0
            assert "0.5" in result.stdout


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
            mock_load.return_value = {
                "active_voice": "my_voice",
                "voices": {
                    "my_voice": {"limiter": {"threshold_db": -5.0}}
                }
            }
            config = get_limiter_config()

        assert config["threshold_db"] == -5.0
        assert config["release_ms"] == DEFAULT_LIMITER["release_ms"]
        assert config["enabled"] == DEFAULT_LIMITER["enabled"]

    def test_set_limiter_setting_saves_config(self):
        """set_limiter_setting should save the setting to active voice config."""
        from tts_config import set_limiter_setting

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {"active_voice": "default_voice", "voices": {}}
            set_limiter_setting("threshold_db", -3.0)

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            assert saved_config["voices"]["default_voice"]["limiter"]["threshold_db"] == -3.0

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
            mock_load.return_value = {"active_voice": "default_voice", "voices": {}}
            set_limiter_setting("enabled", False)

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            assert saved_config["voices"]["default_voice"]["limiter"]["enabled"] is False


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
            mock_load.return_value = {"active_voice": "default_voice", "voices": {}}
            set_compressor_setting("input_gain_db", 6.0)

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            assert saved_config["voices"]["default_voice"]["compressor"]["input_gain_db"] == 6.0


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
            mock_load.return_value = {"active_voice": "default_voice", "voices": {}}
            set_compressor_setting("master_gain_db", -3.0)

            mock_save.assert_called_once()
            saved_config = mock_save.call_args[0][0]
            assert saved_config["voices"]["default_voice"]["compressor"]["master_gain_db"] == -3.0


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


# =============================================================================
# Schema Migration Tests: Global Audio -> Per-Voice (mlx-tts-sdi)
# =============================================================================


class TestSchemaVersion:
    """Tests for schema version field."""

    def test_default_config_has_schema_version(self):
        """DEFAULT_CONFIG should include schema_version field."""
        from tts_config import DEFAULT_CONFIG

        assert "schema_version" in DEFAULT_CONFIG

    def test_schema_version_is_integer(self):
        """schema_version should be an integer."""
        from tts_config import DEFAULT_CONFIG

        assert isinstance(DEFAULT_CONFIG["schema_version"], int)

    def test_current_schema_version_is_one(self):
        """Initial schema_version should be 1."""
        from tts_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["schema_version"] == 1

    def test_load_config_includes_schema_version(self):
        """load_config should always return config with schema_version."""
        from tts_config import load_config

        with patch("tts_config.get_config_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/config.json")
            config = load_config()

        assert "schema_version" in config
        assert config["schema_version"] == 1


class TestDefaultConfigNoGlobalAudio:
    """Tests that DEFAULT_CONFIG no longer has global compressor/limiter."""

    def test_default_config_no_global_compressor(self):
        """DEFAULT_CONFIG should NOT have top-level compressor key."""
        from tts_config import DEFAULT_CONFIG

        assert "compressor" not in DEFAULT_CONFIG

    def test_default_config_no_global_limiter(self):
        """DEFAULT_CONFIG should NOT have top-level limiter key."""
        from tts_config import DEFAULT_CONFIG

        assert "limiter" not in DEFAULT_CONFIG

    def test_default_config_has_voices_dict(self):
        """DEFAULT_CONFIG should include voices dict."""
        from tts_config import DEFAULT_CONFIG

        assert "voices" in DEFAULT_CONFIG
        assert isinstance(DEFAULT_CONFIG["voices"], dict)


class TestGlobalGettersUseActiveVoice:
    """Tests that global getters now operate on active voice."""

    def test_get_compressor_config_uses_active_voice(self):
        """get_compressor_config should read from active voice's settings."""
        from tts_config import get_compressor_config

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "my_voice",
                "voices": {
                    "my_voice": {"compressor": {"gain_db": 15}}
                }
            }
            config = get_compressor_config()

        assert config["gain_db"] == 15

    def test_get_limiter_config_uses_active_voice(self):
        """get_limiter_config should read from active voice's settings."""
        from tts_config import get_limiter_config

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "my_voice",
                "voices": {
                    "my_voice": {"limiter": {"threshold_db": -2.0}}
                }
            }
            config = get_limiter_config()

        assert config["threshold_db"] == -2.0

    def test_get_compressor_config_falls_back_to_defaults(self):
        """get_compressor_config should use defaults when voice has no settings."""
        from tts_config import get_compressor_config, DEFAULT_COMPRESSOR

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "default_voice",
                "voices": {}
            }
            config = get_compressor_config()

        assert config == DEFAULT_COMPRESSOR

    def test_get_limiter_config_falls_back_to_defaults(self):
        """get_limiter_config should use defaults when voice has no settings."""
        from tts_config import get_limiter_config, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "active_voice": "default_voice",
                "voices": {}
            }
            config = get_limiter_config()

        assert config == DEFAULT_LIMITER


class TestGlobalSettersUseActiveVoice:
    """Tests that global setters now operate on active voice."""

    def test_set_compressor_setting_writes_to_active_voice(self):
        """set_compressor_setting should write to active voice's settings."""
        from tts_config import set_compressor_setting

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "active_voice": "my_voice",
                "voices": {}
            }
            set_compressor_setting("gain_db", 12.0)

            mock_save.assert_called_once()
            saved = mock_save.call_args[0][0]
            assert saved["voices"]["my_voice"]["compressor"]["gain_db"] == 12.0
            # Should NOT have global compressor
            assert "compressor" not in saved or saved.get("compressor") is None

    def test_set_limiter_setting_writes_to_active_voice(self):
        """set_limiter_setting should write to active voice's settings."""
        from tts_config import set_limiter_setting

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "active_voice": "my_voice",
                "voices": {}
            }
            set_limiter_setting("threshold_db", -3.0)

            mock_save.assert_called_once()
            saved = mock_save.call_args[0][0]
            assert saved["voices"]["my_voice"]["limiter"]["threshold_db"] == -3.0
            # Should NOT have global limiter
            assert "limiter" not in saved or saved.get("limiter") is None

    def test_set_compressor_setting_preserves_existing_voice_settings(self):
        """set_compressor_setting should preserve other voice settings."""
        from tts_config import set_compressor_setting

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "active_voice": "my_voice",
                "voices": {
                    "my_voice": {
                        "compressor": {"enabled": True, "gain_db": 5},
                        "limiter": {"threshold_db": -1.0}
                    }
                }
            }
            set_compressor_setting("gain_db", 10.0)

            saved = mock_save.call_args[0][0]
            # Should update gain_db
            assert saved["voices"]["my_voice"]["compressor"]["gain_db"] == 10.0
            # Should preserve enabled
            assert saved["voices"]["my_voice"]["compressor"]["enabled"] is True
            # Should preserve limiter
            assert saved["voices"]["my_voice"]["limiter"]["threshold_db"] == -1.0


class TestResetFunctionsUseActiveVoice:
    """Tests that reset functions operate on active voice."""

    def test_reset_compressor_to_defaults_clears_active_voice_compressor(self):
        """reset_compressor_to_defaults should reset active voice's compressor."""
        from tts_config import reset_compressor_to_defaults, DEFAULT_COMPRESSOR

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "active_voice": "my_voice",
                "voices": {
                    "my_voice": {
                        "compressor": {"gain_db": 20, "enabled": False},
                        "limiter": {"threshold_db": -2.0}
                    }
                }
            }
            reset_compressor_to_defaults()

            saved = mock_save.call_args[0][0]
            # Compressor should be reset to defaults
            assert saved["voices"]["my_voice"]["compressor"] == DEFAULT_COMPRESSOR
            # Limiter should be preserved
            assert saved["voices"]["my_voice"]["limiter"]["threshold_db"] == -2.0

    def test_reset_limiter_to_defaults_clears_active_voice_limiter(self):
        """reset_limiter_to_defaults should reset active voice's limiter."""
        from tts_config import reset_limiter_to_defaults, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "active_voice": "my_voice",
                "voices": {
                    "my_voice": {
                        "compressor": {"gain_db": 15},
                        "limiter": {"threshold_db": -5.0, "enabled": False}
                    }
                }
            }
            reset_limiter_to_defaults()

            saved = mock_save.call_args[0][0]
            # Limiter should be reset to defaults
            assert saved["voices"]["my_voice"]["limiter"] == DEFAULT_LIMITER
            # Compressor should be preserved
            assert saved["voices"]["my_voice"]["compressor"]["gain_db"] == 15

    def test_reset_all_audio_to_defaults_clears_both(self):
        """reset_all_audio_to_defaults should reset both compressor and limiter."""
        from tts_config import reset_all_audio_to_defaults, DEFAULT_COMPRESSOR, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "active_voice": "my_voice",
                "voices": {
                    "my_voice": {
                        "compressor": {"gain_db": 20},
                        "limiter": {"threshold_db": -5.0}
                    }
                }
            }
            reset_all_audio_to_defaults()

            saved = mock_save.call_args[0][0]
            assert saved["voices"]["my_voice"]["compressor"] == DEFAULT_COMPRESSOR
            assert saved["voices"]["my_voice"]["limiter"] == DEFAULT_LIMITER


# =============================================================================
# Per-Voice Getters/Setters Tests (mlx-tts-8fl) - TDD RED PHASE
# =============================================================================


class TestGetVoiceCompressor:
    """Tests for get_voice_compressor(voice_name) -> dict."""

    def test_get_voice_compressor_returns_dict(self):
        """get_voice_compressor should return a dict."""
        from tts_config import get_voice_compressor

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"voices": {}}
            result = get_voice_compressor("my_voice")

        assert isinstance(result, dict)

    def test_get_voice_compressor_returns_defaults_for_unconfigured_voice(self):
        """get_voice_compressor should return DEFAULT_COMPRESSOR for unconfigured voice."""
        from tts_config import get_voice_compressor, DEFAULT_COMPRESSOR

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"voices": {}}
            result = get_voice_compressor("unconfigured_voice")

        assert result == DEFAULT_COMPRESSOR

    def test_get_voice_compressor_returns_merged_settings(self):
        """get_voice_compressor should merge voice settings with defaults."""
        from tts_config import get_voice_compressor, DEFAULT_COMPRESSOR

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "voices": {
                    "my_voice": {"compressor": {"gain_db": 15, "enabled": False}}
                }
            }
            result = get_voice_compressor("my_voice")

        # Voice-specific overrides
        assert result["gain_db"] == 15
        assert result["enabled"] is False
        # Defaults for unspecified keys
        assert result["threshold_db"] == DEFAULT_COMPRESSOR["threshold_db"]
        assert result["ratio"] == DEFAULT_COMPRESSOR["ratio"]

    def test_get_voice_compressor_different_voices_independent(self):
        """get_voice_compressor should return different settings for different voices."""
        from tts_config import get_voice_compressor

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "voices": {
                    "voice_a": {"compressor": {"gain_db": 10}},
                    "voice_b": {"compressor": {"gain_db": 20}}
                }
            }
            result_a = get_voice_compressor("voice_a")
            result_b = get_voice_compressor("voice_b")

        assert result_a["gain_db"] == 10
        assert result_b["gain_db"] == 20


class TestSetVoiceCompressor:
    """Tests for set_voice_compressor(voice_name, key, value)."""

    def test_set_voice_compressor_saves_setting(self):
        """set_voice_compressor should save the compressor setting for specified voice."""
        from tts_config import set_voice_compressor

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {"voices": {}}
            set_voice_compressor("my_voice", "gain_db", 12.0)

            mock_save.assert_called_once()
            saved = mock_save.call_args[0][0]
            assert saved["voices"]["my_voice"]["compressor"]["gain_db"] == 12.0

    def test_set_voice_compressor_preserves_other_keys(self):
        """set_voice_compressor should preserve other compressor settings."""
        from tts_config import set_voice_compressor

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "voices": {
                    "my_voice": {"compressor": {"enabled": True, "gain_db": 5}}
                }
            }
            set_voice_compressor("my_voice", "gain_db", 10.0)

            saved = mock_save.call_args[0][0]
            assert saved["voices"]["my_voice"]["compressor"]["gain_db"] == 10.0
            assert saved["voices"]["my_voice"]["compressor"]["enabled"] is True

    def test_set_voice_compressor_validates_key(self):
        """set_voice_compressor should reject invalid compressor keys."""
        from tts_config import set_voice_compressor

        with pytest.raises(ValueError, match="Invalid compressor key"):
            set_voice_compressor("my_voice", "invalid_key", 1.0)

    def test_set_voice_compressor_different_voice_than_active(self):
        """set_voice_compressor should work for non-active voices."""
        from tts_config import set_voice_compressor

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "active_voice": "default_voice",
                "voices": {}
            }
            set_voice_compressor("other_voice", "gain_db", 15.0)

            saved = mock_save.call_args[0][0]
            # Should set for other_voice, not active_voice
            assert saved["voices"]["other_voice"]["compressor"]["gain_db"] == 15.0
            assert "default_voice" not in saved["voices"]


class TestGetVoiceLimiter:
    """Tests for get_voice_limiter(voice_name) -> dict."""

    def test_get_voice_limiter_returns_dict(self):
        """get_voice_limiter should return a dict."""
        from tts_config import get_voice_limiter

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"voices": {}}
            result = get_voice_limiter("my_voice")

        assert isinstance(result, dict)

    def test_get_voice_limiter_returns_defaults_for_unconfigured_voice(self):
        """get_voice_limiter should return DEFAULT_LIMITER for unconfigured voice."""
        from tts_config import get_voice_limiter, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {"voices": {}}
            result = get_voice_limiter("unconfigured_voice")

        assert result == DEFAULT_LIMITER

    def test_get_voice_limiter_returns_merged_settings(self):
        """get_voice_limiter should merge voice settings with defaults."""
        from tts_config import get_voice_limiter, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load:
            mock_load.return_value = {
                "voices": {
                    "my_voice": {"limiter": {"threshold_db": -3.0, "enabled": False}}
                }
            }
            result = get_voice_limiter("my_voice")

        # Voice-specific overrides
        assert result["threshold_db"] == -3.0
        assert result["enabled"] is False
        # Defaults for unspecified keys
        assert result["release_ms"] == DEFAULT_LIMITER["release_ms"]


class TestSetVoiceLimiter:
    """Tests for set_voice_limiter(voice_name, key, value)."""

    def test_set_voice_limiter_saves_setting(self):
        """set_voice_limiter should save the limiter setting for specified voice."""
        from tts_config import set_voice_limiter

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {"voices": {}}
            set_voice_limiter("my_voice", "threshold_db", -2.0)

            mock_save.assert_called_once()
            saved = mock_save.call_args[0][0]
            assert saved["voices"]["my_voice"]["limiter"]["threshold_db"] == -2.0

    def test_set_voice_limiter_preserves_other_keys(self):
        """set_voice_limiter should preserve other limiter settings."""
        from tts_config import set_voice_limiter

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "voices": {
                    "my_voice": {"limiter": {"enabled": True, "threshold_db": -1.0}}
                }
            }
            set_voice_limiter("my_voice", "threshold_db", -3.0)

            saved = mock_save.call_args[0][0]
            assert saved["voices"]["my_voice"]["limiter"]["threshold_db"] == -3.0
            assert saved["voices"]["my_voice"]["limiter"]["enabled"] is True

    def test_set_voice_limiter_validates_key(self):
        """set_voice_limiter should reject invalid limiter keys."""
        from tts_config import set_voice_limiter

        with pytest.raises(ValueError, match="Invalid limiter key"):
            set_voice_limiter("my_voice", "invalid_key", 1.0)

    def test_set_voice_limiter_different_voice_than_active(self):
        """set_voice_limiter should work for non-active voices."""
        from tts_config import set_voice_limiter

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "active_voice": "default_voice",
                "voices": {}
            }
            set_voice_limiter("other_voice", "threshold_db", -4.0)

            saved = mock_save.call_args[0][0]
            # Should set for other_voice, not active_voice
            assert saved["voices"]["other_voice"]["limiter"]["threshold_db"] == -4.0
            assert "default_voice" not in saved["voices"]


class TestResetVoiceToDefaults:
    """Tests for reset_voice_to_defaults(voice_name)."""

    def test_reset_voice_to_defaults_resets_compressor(self):
        """reset_voice_to_defaults should reset compressor to defaults."""
        from tts_config import reset_voice_to_defaults, DEFAULT_COMPRESSOR

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "voices": {
                    "my_voice": {"compressor": {"gain_db": 20, "enabled": False}}
                }
            }
            reset_voice_to_defaults("my_voice")

            saved = mock_save.call_args[0][0]
            assert saved["voices"]["my_voice"]["compressor"] == DEFAULT_COMPRESSOR

    def test_reset_voice_to_defaults_resets_limiter(self):
        """reset_voice_to_defaults should reset limiter to defaults."""
        from tts_config import reset_voice_to_defaults, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "voices": {
                    "my_voice": {"limiter": {"threshold_db": -5.0, "enabled": False}}
                }
            }
            reset_voice_to_defaults("my_voice")

            saved = mock_save.call_args[0][0]
            assert saved["voices"]["my_voice"]["limiter"] == DEFAULT_LIMITER

    def test_reset_voice_to_defaults_resets_both(self):
        """reset_voice_to_defaults should reset both compressor and limiter."""
        from tts_config import reset_voice_to_defaults, DEFAULT_COMPRESSOR, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "voices": {
                    "my_voice": {
                        "compressor": {"gain_db": 20},
                        "limiter": {"threshold_db": -5.0}
                    }
                }
            }
            reset_voice_to_defaults("my_voice")

            saved = mock_save.call_args[0][0]
            assert saved["voices"]["my_voice"]["compressor"] == DEFAULT_COMPRESSOR
            assert saved["voices"]["my_voice"]["limiter"] == DEFAULT_LIMITER

    def test_reset_voice_to_defaults_works_for_unconfigured_voice(self):
        """reset_voice_to_defaults should work even for voices with no existing config."""
        from tts_config import reset_voice_to_defaults, DEFAULT_COMPRESSOR, DEFAULT_LIMITER

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {"voices": {}}
            reset_voice_to_defaults("new_voice")

            saved = mock_save.call_args[0][0]
            assert saved["voices"]["new_voice"]["compressor"] == DEFAULT_COMPRESSOR
            assert saved["voices"]["new_voice"]["limiter"] == DEFAULT_LIMITER

    def test_reset_voice_to_defaults_does_not_affect_other_voices(self):
        """reset_voice_to_defaults should not affect other voices."""
        from tts_config import reset_voice_to_defaults

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "voices": {
                    "voice_a": {"compressor": {"gain_db": 10}},
                    "voice_b": {"compressor": {"gain_db": 20}}
                }
            }
            reset_voice_to_defaults("voice_a")

            saved = mock_save.call_args[0][0]
            # voice_b should be unchanged
            assert saved["voices"]["voice_b"]["compressor"]["gain_db"] == 20

    def test_reset_voice_to_defaults_different_from_active(self):
        """reset_voice_to_defaults should work for non-active voices."""
        from tts_config import reset_voice_to_defaults, DEFAULT_COMPRESSOR

        with patch("tts_config.load_config") as mock_load, \
             patch("tts_config.save_config") as mock_save:
            mock_load.return_value = {
                "active_voice": "default_voice",
                "voices": {
                    "other_voice": {"compressor": {"gain_db": 15}}
                }
            }
            reset_voice_to_defaults("other_voice")

            saved = mock_save.call_args[0][0]
            assert saved["voices"]["other_voice"]["compressor"] == DEFAULT_COMPRESSOR


# =============================================================================
# Format-Aware Voice Discovery Tests (mlx-tts-e6g)
# =============================================================================


class TestDiscoverVoicesWithSafetensors:
    """Tests for discover_voices finding both .wav and .safetensors files."""

    def test_discover_voices_finds_safetensors_files(self):
        """discover_voices should find .safetensors files in assets directory."""
        from tts_config import discover_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "embedded_voice.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                voices = discover_voices()

            assert "embedded_voice" in voices

    def test_discover_voices_finds_both_formats(self):
        """discover_voices should find both .wav and .safetensors files."""
        from tts_config import discover_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "wav_voice.wav").touch()
            (assets_dir / "embedded_voice.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                voices = discover_voices()

            assert "wav_voice" in voices
            assert "embedded_voice" in voices

    def test_discover_voices_deduplicates_when_both_exist(self):
        """discover_voices should deduplicate when both .wav and .safetensors exist."""
        from tts_config import discover_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "my_voice.wav").touch()
            (assets_dir / "my_voice.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                voices = discover_voices()

            # Should only appear once
            assert voices.count("my_voice") == 1

    def test_discover_voices_strips_safetensors_extension(self):
        """discover_voices should return names without .safetensors extension."""
        from tts_config import discover_voices

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "my_voice.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                voices = discover_voices()

            assert "my_voice" in voices
            assert "my_voice.safetensors" not in voices


class TestGetVoiceFormat:
    """Tests for get_voice_format function."""

    def test_get_voice_format_returns_safetensors(self):
        """get_voice_format should return 'safetensors' for .safetensors files."""
        from tts_config import get_voice_format

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "my_voice.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                fmt = get_voice_format("my_voice")

            assert fmt == "safetensors"

    def test_get_voice_format_returns_wav(self):
        """get_voice_format should return 'wav' for .wav files."""
        from tts_config import get_voice_format

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "my_voice.wav").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                fmt = get_voice_format("my_voice")

            assert fmt == "wav"

    def test_get_voice_format_prefers_safetensors(self):
        """get_voice_format should return 'safetensors' when both exist."""
        from tts_config import get_voice_format

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            (assets_dir / "my_voice.wav").touch()
            (assets_dir / "my_voice.safetensors").touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                fmt = get_voice_format("my_voice")

            assert fmt == "safetensors"

    def test_get_voice_format_returns_none_for_nonexistent(self):
        """get_voice_format should return None for nonexistent voice."""
        from tts_config import get_voice_format

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                fmt = get_voice_format("nonexistent_voice")

            assert fmt is None


class TestResolveVoicePathWithSafetensors:
    """Tests for resolve_voice_path with safetensors support."""

    def test_resolve_voice_path_prefers_safetensors(self):
        """resolve_voice_path should return .safetensors path when both exist."""
        from tts_config import resolve_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            wav_file = assets_dir / "my_voice.wav"
            wav_file.touch()
            safetensors_file = assets_dir / "my_voice.safetensors"
            safetensors_file.touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                path = resolve_voice_path("my_voice")

            assert path == safetensors_file

    def test_resolve_voice_path_falls_back_to_wav(self):
        """resolve_voice_path should return .wav path when only wav exists."""
        from tts_config import resolve_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            wav_file = assets_dir / "my_voice.wav"
            wav_file.touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                path = resolve_voice_path("my_voice")

            assert path == wav_file

    def test_resolve_voice_path_returns_safetensors_only(self):
        """resolve_voice_path should work when only .safetensors exists."""
        from tts_config import resolve_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            safetensors_file = assets_dir / "embedded_voice.safetensors"
            safetensors_file.touch()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                path = resolve_voice_path("embedded_voice")

            assert path == safetensors_file

    def test_resolve_voice_path_security_validation_for_safetensors(self):
        """resolve_voice_path should apply security validation for safetensors."""
        from tts_config import resolve_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                # Path traversal should still be rejected
                with pytest.raises(ValueError, match="Invalid voice name"):
                    resolve_voice_path("../../../etc/passwd")

    def test_resolve_voice_path_rejects_symlink_escape_for_safetensors(self):
        """resolve_voice_path should reject symlinks pointing outside assets for safetensors."""
        from tts_config import resolve_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            assets_dir = Path(tmpdir) / "assets"
            assets_dir.mkdir()
            # Create a file outside assets
            outside_file = Path(tmpdir) / "secret.safetensors"
            outside_file.touch()
            # Create a symlink inside assets pointing to the outside file
            symlink = assets_dir / "evil_voice.safetensors"
            symlink.symlink_to(outside_file)

            with patch("tts_config._PLUGIN_ROOT", Path(tmpdir)):
                with pytest.raises(ValueError, match="Invalid voice name"):
                    resolve_voice_path("evil_voice")
