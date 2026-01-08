"""
Integration tests for TTS configuration module.

These tests use real file I/O (in temp directories) to verify
the config module works end-to-end.

Run with: uv run pytest tests/integration/test_tts_config_integration.py -v
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestPerVoiceConfigIntegration:
    """Integration tests for per-voice configuration."""

    def test_full_voice_config_workflow(self):
        """Test complete workflow: discover voices, set config, retrieve effective settings."""
        from tts_config import (
            discover_voices,
            get_active_voice,
            set_active_voice,
            set_voice_config,
            get_effective_compressor,
            get_effective_limiter,
            DEFAULT_COMPRESSOR,
            DEFAULT_LIMITER,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Set up mock plugin root with assets and config
            plugin_root = Path(tmpdir)
            assets_dir = plugin_root / "assets"
            assets_dir.mkdir()
            config_dir = plugin_root / ".config"
            config_path = config_dir / "config.json"

            # Create voice files
            (assets_dir / "voice_alice.wav").touch()
            (assets_dir / "voice_bob.wav").touch()

            with patch("tts_config._PLUGIN_ROOT", plugin_root), \
                 patch("tts_config.get_config_path", return_value=config_path):

                # 1. Discover voices
                voices = discover_voices()
                assert "voice_alice" in voices
                assert "voice_bob" in voices

                # 2. Default active voice
                assert get_active_voice() == "default"

                # 3. Set active voice
                set_active_voice("voice_alice")
                assert get_active_voice() == "voice_alice"

                # 4. Set voice-specific compressor config
                set_voice_config("voice_alice", {
                    "compressor": {"gain_db": 12, "enabled": False}
                })

                # 5. Verify cascading config resolution
                compressor = get_effective_compressor("voice_alice")
                assert compressor["gain_db"] == 12
                assert compressor["enabled"] is False
                # Defaults for unspecified keys
                assert compressor["threshold_db"] == DEFAULT_COMPRESSOR["threshold_db"]

                # 6. Verify unconfigured voice uses defaults
                compressor_bob = get_effective_compressor("voice_bob")
                assert compressor_bob == DEFAULT_COMPRESSOR

                # 7. Verify limiter config
                set_voice_config("voice_alice", {
                    "limiter": {"threshold_db": -2.0}
                })
                limiter = get_effective_limiter("voice_alice")
                assert limiter["threshold_db"] == -2.0
                assert limiter["release_ms"] == DEFAULT_LIMITER["release_ms"]

                # 8. Verify config persisted to file
                saved_config = json.loads(config_path.read_text())
                assert saved_config["active_voice"] == "voice_alice"
                assert "voice_alice" in saved_config["voices"]

    def test_config_merge_preserves_existing_settings(self):
        """Test that setting new config keys doesn't overwrite existing ones."""
        from tts_config import set_voice_config, get_voice_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("tts_config.get_config_path", return_value=config_path):
                # Set initial compressor config
                set_voice_config("test_voice", {
                    "compressor": {"gain_db": 5, "enabled": True, "ratio": 4.0}
                })

                # Update only gain_db
                set_voice_config("test_voice", {
                    "compressor": {"gain_db": 10}
                })

                # Verify other settings preserved
                config = get_voice_config("test_voice")
                assert config["compressor"]["gain_db"] == 10
                assert config["compressor"]["enabled"] is True
                assert config["compressor"]["ratio"] == 4.0


class TestSecureVoicePathIntegration:
    """Integration tests for secure voice path resolution."""

    def test_resolve_real_voice_file(self):
        """Test resolving a real voice file in assets."""
        from tts_config import resolve_voice_path

        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_root = Path(tmpdir)
            assets_dir = plugin_root / "assets"
            assets_dir.mkdir()
            voice_file = assets_dir / "my_voice.wav"
            voice_file.write_bytes(b"RIFF" + b"\x00" * 100)  # Minimal WAV header

            with patch("tts_config._PLUGIN_ROOT", plugin_root):
                resolved = resolve_voice_path("my_voice")
                assert resolved == voice_file
                assert resolved.exists()
