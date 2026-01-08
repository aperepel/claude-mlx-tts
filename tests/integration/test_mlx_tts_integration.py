"""
Integration tests for MLX TTS core module.

These tests require MLX hardware and audio playback capability.
Run with: uv run pytest tests/integration/
"""
import os
import sys


# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestSpeakMlxIntegration:
    """Integration tests for speak_mlx function."""

    def test_speak_mlx_plays_audio(self):
        """Integration test: speak_mlx should generate and play audio."""
        from mlx_tts_core import speak_mlx

        # This will actually play audio
        speak_mlx("Testing speak MLX function")


class TestStreamingMetricsIntegration:
    """Integration tests for streaming metrics with real model."""

    def test_streaming_metrics_with_real_model(self):
        """Integration test: verify metrics work with real model."""
        from mlx_tts_core import generate_speech, get_model

        model = get_model()
        metrics = generate_speech(
            "Integration test",
            model=model,
            play=False,
            stream=True,
            return_metrics=True,
        )

        assert metrics is not None
        assert metrics["ttft"] > 0
        assert metrics["gen_time"] > 0
        assert metrics["chunks"] >= 1
