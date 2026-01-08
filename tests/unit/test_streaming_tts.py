"""
Unit tests for MLX TTS streaming functionality.

Run with: uv run pytest tests/unit/test_streaming_tts.py -v
"""
import os
import sys
from unittest.mock import MagicMock, patch, call

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestStreamingConfig:
    """Tests for streaming configuration in tts_config module."""

    def test_default_streaming_interval_exists(self):
        """Config should have a default streaming interval constant."""
        from tts_config import DEFAULT_STREAMING_INTERVAL
        assert DEFAULT_STREAMING_INTERVAL == 0.5

    def test_get_streaming_interval_returns_default(self):
        """get_streaming_interval should return default value when not configured."""
        from tts_config import get_streaming_interval, DEFAULT_STREAMING_INTERVAL
        interval = get_streaming_interval()
        assert interval == DEFAULT_STREAMING_INTERVAL

    def test_set_streaming_interval_valid(self, tmp_path):
        """set_streaming_interval should save valid interval to config."""
        config_path = tmp_path / "config.json"
        with patch.dict(os.environ, {"TTS_CONFIG_PATH": str(config_path)}):
            # Force reimport to pick up new config path
            import importlib
            import tts_config
            importlib.reload(tts_config)

            tts_config.set_streaming_interval(1.0)
            assert tts_config.get_streaming_interval() == 1.0

    def test_set_streaming_interval_rejects_invalid(self):
        """set_streaming_interval should reject values outside valid range."""
        from tts_config import set_streaming_interval

        # Too small
        with pytest.raises(ValueError):
            set_streaming_interval(0.05)

        # Too large
        with pytest.raises(ValueError):
            set_streaming_interval(10.0)

    def test_streaming_interval_range(self):
        """Streaming interval should be between 0.1 and 5.0 seconds."""
        from tts_config import MIN_STREAMING_INTERVAL, MAX_STREAMING_INTERVAL
        assert MIN_STREAMING_INTERVAL == 0.1
        assert MAX_STREAMING_INTERVAL == 5.0


class TestGenerateSpeechStreaming:
    """Tests for streaming support in generate_speech function."""

    def test_generate_speech_has_stream_parameter(self):
        """generate_speech should accept a stream parameter."""
        from mlx_tts_core import generate_speech
        import inspect

        sig = inspect.signature(generate_speech)
        assert "stream" in sig.parameters
        # Default should be True for streaming
        assert sig.parameters["stream"].default is True

    def test_generate_speech_has_streaming_interval_parameter(self):
        """generate_speech should accept streaming_interval parameter."""
        from mlx_tts_core import generate_speech
        import inspect

        sig = inspect.signature(generate_speech)
        assert "streaming_interval" in sig.parameters

    def test_generate_speech_passes_stream_to_model(self):
        """generate_speech should pass stream=True to model.generate()."""
        from mlx_tts_core import generate_speech

        model = MagicMock()
        model.sample_rate = 24000
        model.generate.return_value = iter([])  # Empty generator

        with patch("mlx_tts_core.AudioPlayer"):
            generate_speech("Test", model=model, play=False, stream=True)

        model.generate.assert_called_once()
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs.get("stream") is True

    def test_generate_speech_passes_streaming_interval_to_model(self):
        """generate_speech should pass streaming_interval to model.generate()."""
        from mlx_tts_core import generate_speech

        model = MagicMock()
        model.sample_rate = 24000
        model.generate.return_value = iter([])

        with patch("mlx_tts_core.AudioPlayer"):
            generate_speech("Test", model=model, play=False, stream=True, streaming_interval=0.5)

        model.generate.assert_called_once()
        call_kwargs = model.generate.call_args[1]
        assert call_kwargs.get("streaming_interval") == 0.5

    def test_generate_speech_uses_config_streaming_interval(self):
        """generate_speech should use config value when streaming_interval not specified."""
        with patch("mlx_tts_core._get_configured_streaming_interval", return_value=0.75):
            from mlx_tts_core import generate_speech

            model = MagicMock()
            model.sample_rate = 24000
            model.generate.return_value = iter([])

            with patch("mlx_tts_core.AudioPlayer"):
                generate_speech("Test", model=model, play=False, stream=True)

            call_kwargs = model.generate.call_args[1]
            assert call_kwargs.get("streaming_interval") == 0.75

    def test_generate_speech_stream_false_disables_streaming(self):
        """generate_speech with stream=False should not pass streaming params."""
        from mlx_tts_core import generate_speech

        with patch("mlx_audio.tts.generate.generate_audio") as mock_gen:
            model = MagicMock()
            model.sample_rate = 24000

            generate_speech("Test", model=model, play=False, stream=False)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs.get("stream") is False
            # streaming_interval should not be passed when stream=False
            assert "streaming_interval" not in call_kwargs


class TestSpeakMlxStreaming:
    """Tests for streaming in speak_mlx high-level function."""

    def test_speak_mlx_enables_streaming_by_default(self):
        """speak_mlx should use streaming by default for low latency."""
        from mlx_tts_core import speak_mlx

        with patch("mlx_tts_core.generate_speech") as mock_gen:
            speak_mlx("Test message")

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs.get("stream") is True

    def test_speak_mlx_accepts_stream_parameter(self):
        """speak_mlx should accept and pass stream parameter."""
        from mlx_tts_core import speak_mlx

        with patch("mlx_tts_core.generate_speech") as mock_gen:
            speak_mlx("Test", stream=False)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs.get("stream") is False


class TestStreamingFileSaveConstraint:
    """Tests for streaming mode file save behavior."""

    def test_streaming_mode_warns_about_save_path(self):
        """Streaming mode should warn when save_path is specified."""
        import logging
        from mlx_tts_core import generate_speech

        with patch("mlx_audio.tts.generate.generate_audio"):
            model = MagicMock()
            model.sample_rate = 24000

            with patch("mlx_tts_core.log") as mock_log:
                generate_speech(
                    "Test",
                    model=model,
                    play=False,
                    stream=True,
                    save_path="/tmp/test.wav"
                )

                # Should warn that save_path is ignored in streaming mode
                mock_log.warning.assert_called()
                warning_msg = mock_log.warning.call_args[0][0]
                assert "save_path" in warning_msg.lower() or "stream" in warning_msg.lower()


class TestStreamingPerformance:
    """Performance tests for streaming TTS."""

    def test_streaming_interval_0_5_target(self):
        """Target streaming interval of 0.5s should be used by default."""
        from mlx_tts_core import generate_speech, DEFAULT_STREAMING_INTERVAL
        from tts_config import DEFAULT_STREAMING_INTERVAL as CONFIG_DEFAULT

        # Verify default constants are 0.5s
        assert DEFAULT_STREAMING_INTERVAL == 0.5
        assert CONFIG_DEFAULT == 0.5

        # Mock config to return default value (isolate from stored config)
        with patch("mlx_tts_core._get_configured_streaming_interval", return_value=0.5):
            model = MagicMock()
            model.sample_rate = 24000
            model.generate.return_value = iter([])

            with patch("mlx_tts_core.AudioPlayer"):
                # When not specifying interval, should use config default (0.5s)
                generate_speech("Test", model=model, play=True, stream=True)

            call_kwargs = model.generate.call_args[1]
            assert call_kwargs["streaming_interval"] == 0.5
