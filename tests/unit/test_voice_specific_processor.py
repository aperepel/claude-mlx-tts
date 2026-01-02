"""
Unit tests for voice-specific audio processor settings.

This tests that hook playback uses the correct voice's compressor/limiter
settings rather than the active voice's settings.

Bug: mlx-tts-1y3 - Hook playback uses wrong voice compressor/limiter settings

Run with: uv run pytest tests/unit/test_voice_specific_processor.py -v
"""
import os
import sys
from unittest.mock import MagicMock, patch
import struct

import numpy as np

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


def create_wav_bytes(
    audio: np.ndarray,
    sample_rate: int = 24000,
    channels: int = 1,
    bits_per_sample: int = 16,
) -> bytes:
    """Create valid WAV file bytes from audio array for testing."""
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio_int = (audio * 32767).astype(np.int16)
    else:
        audio_int = audio.astype(np.int16)

    audio_bytes = audio_int.tobytes()
    data_size = len(audio_bytes)

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * channels * bits_per_sample // 8,
        channels * bits_per_sample // 8,
        bits_per_sample,
        b"data",
        data_size,
    )

    return header + audio_bytes


class TestPlayStreamingHttpUsesVoiceSettings:
    """Tests that play_streaming_http() uses voice-specific processor settings."""

    def test_passes_voice_settings_to_streaming_processor(self):
        """play_streaming_http should use voice-specific settings for audio processor."""
        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        voice_compressor = {
            "enabled": True,
            "input_gain_db": -6.0,
            "threshold_db": -12,
            "ratio": 2.0,
            "attack_ms": 5,
            "release_ms": 100,
            "gain_db": 4.0,
            "master_gain_db": -3.0,
        }
        voice_limiter = {
            "enabled": True,
            "threshold_db": -1.0,
            "release_ms": 50,
        }

        # Need to reimport to pick up the module-level import of tts_config
        import mlx_server_utils

        with patch.object(mlx_server_utils, "requests") as mock_requests:
            mock_requests.post.return_value = mock_response
            with patch.object(mlx_server_utils, "ensure_server_running"):
                with patch.object(mlx_server_utils, "create_processor") as mock_create:
                    mock_processor = MagicMock(return_value=audio)
                    mock_create.return_value = mock_processor
                    with patch.object(mlx_server_utils, "AudioPlayer") as mock_player_cls:
                        mock_player = MagicMock()
                        mock_player.playing = True
                        mock_player.buffered_samples.return_value = 100
                        mock_player_cls.return_value = mock_player
                        with patch.object(mlx_server_utils.tts_config, "get_effective_compressor", return_value=voice_compressor) as mock_comp:
                            with patch.object(mlx_server_utils.tts_config, "get_effective_limiter", return_value=voice_limiter) as mock_lim:
                                mlx_server_utils.play_streaming_http("Test", voice="c3po")

                                # Verify config was looked up for the voice
                                mock_comp.assert_called_with("c3po")
                                mock_lim.assert_called_with("c3po")

                                # Verify create_processor was called with voice-specific settings
                                mock_create.assert_called_once()
                                call_kwargs = mock_create.call_args[1]

                                assert call_kwargs.get("input_gain_db") == -6.0
                                assert call_kwargs.get("threshold_db") == -12
                                assert call_kwargs.get("gain_db") == 4.0

    def test_uses_none_for_voice_config_when_no_voice_specified(self):
        """play_streaming_http should pass None to config lookup when voice is None."""
        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        import mlx_server_utils

        with patch.object(mlx_server_utils, "requests") as mock_requests:
            mock_requests.post.return_value = mock_response
            with patch.object(mlx_server_utils, "ensure_server_running"):
                with patch.object(mlx_server_utils, "create_processor") as mock_create:
                    mock_create.return_value = lambda x: x
                    with patch.object(mlx_server_utils, "AudioPlayer") as mock_player_cls:
                        mock_player = MagicMock()
                        mock_player.playing = True
                        mock_player.buffered_samples.return_value = 100
                        mock_player_cls.return_value = mock_player
                        with patch.object(mlx_server_utils.tts_config, "get_effective_compressor", return_value={"enabled": True}) as mock_comp:
                            with patch.object(mlx_server_utils.tts_config, "get_effective_limiter", return_value={"enabled": True}) as mock_lim:
                                mlx_server_utils.play_streaming_http("Test", voice=None)

                                # Should pass None to config lookup (uses active voice)
                                mock_comp.assert_called_with(None)
                                mock_lim.assert_called_with(None)


class TestGenerateStreamingWithMetricsUsesVoiceSettings:
    """Tests that _generate_streaming_with_metrics() uses voice-specific settings."""

    def test_passes_voice_settings_to_core_processor(self):
        """_generate_streaming_with_metrics should use voice-specific processor settings."""
        voice_compressor = {
            "enabled": True,
            "input_gain_db": -6.0,
            "threshold_db": -12,
            "ratio": 2.0,
            "attack_ms": 5,
            "release_ms": 100,
            "gain_db": 4.0,
            "master_gain_db": -3.0,
        }
        voice_limiter = {
            "enabled": True,
            "threshold_db": -1.0,
            "release_ms": 50,
        }

        # Mock model with generator
        mock_model = MagicMock()
        mock_model.sample_rate = 24000

        mock_result = MagicMock()
        mock_result.audio = np.zeros(2400, dtype=np.float32)
        mock_model.generate.return_value = iter([mock_result])

        import mlx_tts_core

        with patch.object(mlx_tts_core, "AudioPlayer") as mock_player_cls:
            mock_player = MagicMock()
            mock_player.playing = True
            mock_player.buffered_samples.return_value = 100
            mock_player_cls.return_value = mock_player
            with patch("audio_processor.create_processor") as mock_create:
                mock_create.return_value = lambda x: x
                with patch.object(mlx_tts_core.tts_config, "get_effective_compressor", return_value=voice_compressor) as mock_comp:
                    with patch.object(mlx_tts_core.tts_config, "get_effective_limiter", return_value=voice_limiter) as mock_lim:
                        with patch("voice_cache.get_voice_conditionals"):
                            mlx_tts_core._generate_streaming_with_metrics(
                                text="Test",
                                model=mock_model,
                                ref_text=".",
                                play=True,
                                streaming_interval=0.5,
                                voice_name="c3po",
                            )

                            # Verify config was looked up for "c3po"
                            mock_comp.assert_called_with("c3po")
                            mock_lim.assert_called_with("c3po")

                            mock_create.assert_called_once()
                            call_kwargs = mock_create.call_args[1]

                            assert call_kwargs.get("input_gain_db") == -6.0
                            assert call_kwargs.get("threshold_db") == -12
                            assert call_kwargs.get("gain_db") == 4.0

    def test_uses_none_for_config_when_no_voice_name(self):
        """_generate_streaming_with_metrics should pass None when voice_name is None."""
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_result = MagicMock()
        mock_result.audio = np.zeros(2400, dtype=np.float32)
        mock_model.generate.return_value = iter([mock_result])

        import mlx_tts_core

        with patch.object(mlx_tts_core, "AudioPlayer") as mock_player_cls:
            mock_player = MagicMock()
            mock_player.playing = True
            mock_player.buffered_samples.return_value = 100
            mock_player_cls.return_value = mock_player
            with patch("audio_processor.create_processor") as mock_create:
                mock_create.return_value = lambda x: x
                with patch.object(mlx_tts_core.tts_config, "get_effective_compressor", return_value={"enabled": True}) as mock_comp:
                    with patch.object(mlx_tts_core.tts_config, "get_effective_limiter", return_value={"enabled": True}) as mock_lim:
                        with patch("mlx_audio.tts.generate.load_audio"):
                            mlx_tts_core._generate_streaming_with_metrics(
                                text="Test",
                                model=mock_model,
                                ref_text=".",
                                play=True,
                                streaming_interval=0.5,
                                voice_name=None,
                                ref_audio="/path/to/audio.wav",  # Use ref_audio to skip voice loading
                            )

                            # Should pass None (uses active voice)
                            mock_comp.assert_called_with(None)
                            mock_lim.assert_called_with(None)


class TestConfigFunctionsExist:
    """Tests that the config helper functions exist and work correctly."""

    def test_get_effective_compressor_exists(self):
        """get_effective_compressor should exist in tts_config."""
        from tts_config import get_effective_compressor
        assert callable(get_effective_compressor)

    def test_get_effective_limiter_exists(self):
        """get_effective_limiter should exist in tts_config."""
        from tts_config import get_effective_limiter
        assert callable(get_effective_limiter)

    def test_get_effective_compressor_accepts_voice_name(self):
        """get_effective_compressor should accept voice_name parameter."""
        from tts_config import get_effective_compressor
        result = get_effective_compressor("default")
        assert isinstance(result, dict)
        assert "enabled" in result

    def test_get_effective_limiter_accepts_voice_name(self):
        """get_effective_limiter should accept voice_name parameter."""
        from tts_config import get_effective_limiter
        result = get_effective_limiter("default")
        assert isinstance(result, dict)
        assert "enabled" in result

    def test_get_effective_compressor_accepts_none(self):
        """get_effective_compressor should accept None (uses active voice)."""
        from tts_config import get_effective_compressor
        result = get_effective_compressor(None)
        assert isinstance(result, dict)

    def test_get_effective_limiter_accepts_none(self):
        """get_effective_limiter should accept None (uses active voice)."""
        from tts_config import get_effective_limiter
        result = get_effective_limiter(None)
        assert isinstance(result, dict)
