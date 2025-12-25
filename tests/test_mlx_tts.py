"""
Tests for MLX TTS core module.

Following TDD approach: these tests are written FIRST before implementation.
Run with: uv run pytest tests/

To run integration tests (require audio hardware):
    uv run pytest tests/ -m integration
"""
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestLoadTtsModel:
    """Tests for model loading functionality."""

    def test_load_tts_model_returns_model_object(self):
        """load_tts_model should return a valid model object."""
        from mlx_tts_core import load_tts_model

        model = load_tts_model()
        assert model is not None

    def test_load_tts_model_has_sample_rate(self):
        """Loaded model should have sample_rate attribute."""
        from mlx_tts_core import load_tts_model

        model = load_tts_model()
        assert hasattr(model, "sample_rate")

    def test_load_tts_model_handles_missing_mlx(self):
        """load_tts_model should raise ImportError when mlx_audio not available."""
        with patch.dict(sys.modules, {"mlx_audio": None, "mlx_audio.tts": None, "mlx_audio.tts.utils": None}):
            # Force reimport
            if "mlx_tts_core" in sys.modules:
                del sys.modules["mlx_tts_core"]

            with pytest.raises(ImportError):
                from mlx_tts_core import load_tts_model
                load_tts_model()


class TestGetModel:
    """Tests for cached model retrieval."""

    def test_get_model_returns_cached_instance(self):
        """get_model should return the same model instance on subsequent calls."""
        from mlx_tts_core import get_model, _clear_model_cache

        _clear_model_cache()  # Reset for clean test
        m1 = get_model()
        m2 = get_model()
        assert m1 is m2

    def test_get_model_loads_on_first_call(self):
        """get_model should load model on first call."""
        from mlx_tts_core import get_model, _clear_model_cache

        _clear_model_cache()
        model = get_model()
        assert model is not None


class TestGenerateSpeech:
    """Tests for speech generation."""

    def test_generate_speech_with_preloaded_model(self):
        """generate_speech should work with a preloaded model."""
        from mlx_tts_core import generate_speech, get_model

        model = get_model()
        # Should not raise - play=False avoids blocking on audio
        generate_speech("Test message", model=model, play=False)

    def test_generate_speech_without_model_uses_cached(self):
        """generate_speech without model arg should use cached model."""
        from mlx_tts_core import generate_speech

        # Should not raise, uses internal cached model
        generate_speech("Hello world", play=False)

    def test_generate_speech_creates_audio_file(self):
        """generate_speech should create an audio output when save_path provided."""
        import tempfile
        from mlx_tts_core import generate_speech

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            generate_speech("Test audio output", save_path=output_path, play=False)
            assert os.path.exists(output_path)
            assert os.path.getsize(output_path) > 0
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


class TestSpeakMlx:
    """Tests for the speak_mlx high-level function."""

    def test_speak_mlx_calls_generate_speech(self):
        """speak_mlx should call generate_speech with play=True."""
        from mlx_tts_core import speak_mlx

        with patch("mlx_tts_core.generate_speech") as mock_gen:
            speak_mlx("Test message")
            mock_gen.assert_called_once()
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["play"] is True
            assert call_kwargs["text"] == "Test message"

    def test_speak_mlx_passes_speed_parameter(self):
        """speak_mlx should pass speed parameter to generate_speech."""
        from mlx_tts_core import speak_mlx

        with patch("mlx_tts_core.generate_speech") as mock_gen:
            speak_mlx("Test", speed=1.2)
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["speed"] == 1.2

    @pytest.mark.integration
    def test_speak_mlx_plays_audio_integration(self):
        """Integration test: speak_mlx should generate and play audio."""
        from mlx_tts_core import speak_mlx

        # This will actually play audio - only run with -m integration
        speak_mlx("Testing speak MLX function")


class TestWarmLatency:
    """Performance tests for warm (cached model) latency."""

    def test_warm_latency_under_threshold(self):
        """With cached model, generation should complete in <3 seconds."""
        from mlx_tts_core import generate_speech, get_model

        # Warm up - load model first
        model = get_model()

        # Time the actual generation
        start = time.time()
        generate_speech("Quick latency test", model=model, play=False)
        elapsed = time.time() - start

        assert elapsed < 3.0, f"Warm generation took {elapsed:.2f}s, expected <3s"


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_generate_speech_empty_text(self):
        """generate_speech should handle empty text gracefully."""
        from mlx_tts_core import generate_speech

        # Should not crash, may log warning
        generate_speech("", play=False)

    def test_generate_speech_short_text(self):
        """generate_speech should handle short text."""
        from mlx_tts_core import generate_speech

        # Should not raise
        generate_speech("Hi", play=False)


class TestIsMlxAvailable:
    """Tests for MLX availability check."""

    def test_is_mlx_available_returns_bool(self):
        """is_mlx_available should return a boolean."""
        from mlx_tts_core import is_mlx_available

        result = is_mlx_available()
        assert isinstance(result, bool)

    def test_is_mlx_available_checks_voice_ref(self):
        """is_mlx_available should check voice reference file exists."""
        from mlx_tts_core import is_mlx_available, MLX_VOICE_REF

        # Should be True if both mlx_audio is installed and voice ref exists
        if os.path.exists(MLX_VOICE_REF):
            assert is_mlx_available() is True
