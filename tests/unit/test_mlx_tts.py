"""
Unit tests for MLX TTS core module.

Following TDD approach: these tests are written FIRST before implementation.
Run with: uv run pytest tests/unit/
"""
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


def _is_mlx_model_available() -> bool:
    """Check if MLX model can be loaded (for skipping integration-like tests)."""
    try:
        from mlx_tts_core import load_tts_model
        model = load_tts_model()
        return model is not None
    except Exception:
        return False


# Cache the result to avoid repeated model loading attempts
_MLX_MODEL_AVAILABLE = None


def is_mlx_model_available() -> bool:
    """Cached check for MLX model availability."""
    global _MLX_MODEL_AVAILABLE
    if _MLX_MODEL_AVAILABLE is None:
        _MLX_MODEL_AVAILABLE = _is_mlx_model_available()
    return _MLX_MODEL_AVAILABLE


requires_mlx_model = pytest.mark.skipif(
    not is_mlx_model_available(),
    reason="MLX model not available or failed to load"
)


class TestLoadTtsModel:
    """Tests for model loading functionality."""

    @requires_mlx_model
    def test_load_tts_model_returns_model_object(self):
        """load_tts_model should return a valid model object."""
        from mlx_tts_core import load_tts_model

        model = load_tts_model()
        assert model is not None

    @requires_mlx_model
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

    @requires_mlx_model
    def test_get_model_returns_cached_instance(self):
        """get_model should return the same model instance on subsequent calls."""
        from mlx_tts_core import get_model, _clear_model_cache

        _clear_model_cache()  # Reset for clean test
        m1 = get_model()
        m2 = get_model()
        assert m1 is m2

    @requires_mlx_model
    def test_get_model_loads_on_first_call(self):
        """get_model should load model on first call."""
        from mlx_tts_core import get_model, _clear_model_cache

        _clear_model_cache()
        model = get_model()
        assert model is not None


class TestGenerateSpeech:
    """Tests for speech generation."""

    @requires_mlx_model
    def test_generate_speech_with_preloaded_model(self):
        """generate_speech should work with a preloaded model."""
        from mlx_tts_core import generate_speech, get_model

        model = get_model()
        # Should not raise - play=False avoids blocking on audio
        generate_speech("Test message", model=model, play=False)

    @requires_mlx_model
    def test_generate_speech_without_model_uses_cached(self):
        """generate_speech without model arg should use cached model."""
        from mlx_tts_core import generate_speech

        # Should not raise, uses internal cached model
        generate_speech("Hello world", play=False)

    @requires_mlx_model
    def test_generate_speech_creates_audio_file(self):
        """generate_speech should create an audio output when save_path provided."""
        import tempfile
        from mlx_tts_core import generate_speech

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            output_path = f.name

        try:
            # Note: stream=False required for file saving
            generate_speech("Test audio output", save_path=output_path, play=False, stream=False)
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


class TestWarmLatency:
    """Performance tests for warm (cached model) latency."""

    @requires_mlx_model
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

        # Empty text returns None early without needing model
        result = generate_speech("", play=False)
        assert result is None

    @requires_mlx_model
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

    def test_is_mlx_available_checks_voices_exist(self):
        """is_mlx_available should check that at least one voice exists."""
        from mlx_tts_core import is_mlx_available
        from tts_config import discover_voices

        # Should be True if both mlx_audio is installed and voices exist
        if len(discover_voices()) > 0:
            assert is_mlx_available() is True


class TestStreamingMetrics:
    """Tests for streaming-aware TTS metrics (TTFT, generation time, etc.)."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model that simulates streaming generation."""
        import numpy as np

        mock = MagicMock()
        mock.sample_rate = 24000

        # Create mock generation results
        def mock_generate(text, ref_audio=None, stream=True, **kwargs):
            """Simulate streaming generation with multiple chunks."""
            chunk_count = 3
            samples_per_chunk = 12000  # 0.5s at 24kHz

            for i in range(chunk_count):
                result = MagicMock()
                result.audio = np.zeros(samples_per_chunk, dtype=np.float32)
                result.real_time_factor = 0.5
                yield result

        mock.generate = mock_generate
        return mock

    @pytest.fixture
    def mock_player(self):
        """Create a mock AudioPlayer."""
        mock = MagicMock()
        mock.queue_audio = MagicMock()
        mock.wait_for_drain = MagicMock()
        return mock

    @pytest.fixture
    def streaming_patches(self, mock_player):
        """Set up patches for streaming mode tests."""
        with patch("mlx_tts_core.AudioPlayer", return_value=mock_player), \
             patch("mlx_audio.tts.generate.load_audio", return_value=MagicMock()):
            yield

    def test_generate_speech_returns_metrics_when_requested(self, mock_model, mock_player, streaming_patches):
        """generate_speech should return metrics dict when return_metrics=True."""
        from mlx_tts_core import generate_speech

        metrics = generate_speech(
            "Test metrics return",
            model=mock_model,
            play=False,
            stream=True,
            return_metrics=True,
        )

        assert metrics is not None
        assert isinstance(metrics, dict)

    def test_metrics_contains_ttft(self, mock_model, mock_player, streaming_patches):
        """Metrics should include time-to-first-audio (ttft)."""
        from mlx_tts_core import generate_speech

        metrics = generate_speech(
            "Test TTFT measurement",
            model=mock_model,
            play=False,
            stream=True,
            return_metrics=True,
        )

        assert metrics is not None
        assert "ttft" in metrics
        assert isinstance(metrics["ttft"], float)
        assert metrics["ttft"] >= 0

    def test_metrics_contains_generation_time(self, mock_model, mock_player, streaming_patches):
        """Metrics should include total generation time."""
        from mlx_tts_core import generate_speech

        metrics = generate_speech(
            "Test generation time",
            model=mock_model,
            play=False,
            stream=True,
            return_metrics=True,
        )

        assert metrics is not None
        assert "gen_time" in metrics
        assert isinstance(metrics["gen_time"], float)
        assert metrics["gen_time"] >= 0
        # Generation time should be >= TTFT
        assert metrics["gen_time"] >= metrics["ttft"]

    def test_metrics_contains_chunk_count(self, mock_model, mock_player, streaming_patches):
        """Metrics should include number of streaming chunks."""
        from mlx_tts_core import generate_speech

        metrics = generate_speech(
            "Test chunk counting",
            model=mock_model,
            play=False,
            stream=True,
            return_metrics=True,
        )

        assert metrics is not None
        assert "chunks" in metrics
        assert isinstance(metrics["chunks"], int)
        assert metrics["chunks"] == 3  # Our mock yields 3 chunks

    def test_metrics_contains_rtf(self, mock_model, mock_player, streaming_patches):
        """Metrics should include real-time factor (RTF)."""
        from mlx_tts_core import generate_speech

        metrics = generate_speech(
            "Test real time factor",
            model=mock_model,
            play=False,
            stream=True,
            return_metrics=True,
        )

        assert metrics is not None
        assert "rtf" in metrics
        assert isinstance(metrics["rtf"], float)

    def test_metrics_contains_audio_duration(self, mock_model, mock_player, streaming_patches):
        """Metrics should include total audio duration."""
        from mlx_tts_core import generate_speech

        metrics = generate_speech(
            "Test audio duration",
            model=mock_model,
            play=False,
            stream=True,
            return_metrics=True,
        )

        assert metrics is not None
        assert "audio_duration" in metrics
        assert isinstance(metrics["audio_duration"], float)
        # 3 chunks * 12000 samples / 24000 Hz = 1.5 seconds
        assert abs(metrics["audio_duration"] - 1.5) < 0.01

    def test_metrics_contains_play_time_when_played(self, mock_model, mock_player, streaming_patches):
        """Metrics should include playback drain time when play=True."""
        from mlx_tts_core import generate_speech

        metrics = generate_speech(
            "Test play time",
            model=mock_model,
            play=True,
            stream=True,
            return_metrics=True,
        )

        assert metrics is not None
        assert "play_time" in metrics
        assert isinstance(metrics["play_time"], float)
        assert metrics["play_time"] >= 0
        # Verify player was used
        assert mock_player.queue_audio.called
        assert mock_player.wait_for_drain.called

    def test_no_play_time_when_not_played(self, mock_model, mock_player, streaming_patches):
        """play_time should be 0 when play=False."""
        from mlx_tts_core import generate_speech

        metrics = generate_speech(
            "Test no play time",
            model=mock_model,
            play=False,
            stream=True,
            return_metrics=True,
        )

        assert metrics is not None
        assert "play_time" in metrics
        assert metrics["play_time"] == 0
        assert not mock_player.wait_for_drain.called

    def test_default_returns_none_for_backwards_compat(self, mock_model, mock_player, streaming_patches):
        """By default generate_speech should return None for backwards compatibility."""
        from mlx_tts_core import generate_speech

        result = generate_speech(
            "Test default return",
            model=mock_model,
            play=False,
            stream=True,
        )

        assert result is None

    def test_metrics_logged_to_info(self, mock_model, mock_player, streaming_patches, caplog):
        """Metrics should be logged at INFO level."""
        import logging
        from mlx_tts_core import generate_speech

        with caplog.at_level(logging.INFO, logger="mlx_tts_core"):
            generate_speech(
                "Test logging",
                model=mock_model,
                play=False,
                stream=True,
                return_metrics=True,
            )

        # Check that metrics were logged
        assert any("TTS:" in record.message for record in caplog.records)
        assert any("ttft=" in record.message for record in caplog.records)
