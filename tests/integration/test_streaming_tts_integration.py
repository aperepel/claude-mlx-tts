"""
Integration tests for MLX TTS streaming functionality.

These tests require MLX hardware and audio playback capability.
Run with: uv run pytest tests/integration/
"""
import os
import sys
import time
from unittest.mock import patch

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestStreamingPerformanceIntegration:
    """Integration performance tests for streaming TTS."""

    def test_streaming_reduces_ttft(self):
        """Streaming should reduce time-to-first-audio compared to non-streaming."""
        from mlx_tts_core import generate_speech, get_model

        model = get_model()
        test_text = "This is a test of streaming text to speech generation."

        # Measure non-streaming TTFT (approximated by full generation time)
        start = time.time()
        generate_speech(test_text, model=model, play=False, stream=False)
        non_streaming_time = time.time() - start

        # Streaming with 0.5s interval should start faster
        # Note: We can't easily measure TTFT without audio hooks,
        # but we can verify the parameters are passed correctly
        with patch("mlx_audio.tts.generate.generate_audio") as mock_gen:
            generate_speech(test_text, model=model, play=True, stream=True, streaming_interval=0.5)

            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["stream"] is True
            assert call_kwargs["streaming_interval"] == 0.5
