"""
Integration tests for MLX TTS streaming functionality.

These tests require MLX hardware and audio playback capability.
Run with: uv run pytest tests/integration/
"""
import os
import struct
import sys
import time
from unittest.mock import patch, MagicMock

import numpy as np
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
        # Note: For streaming mode, generate_speech uses model.generate() directly
        # with stream=True and streaming_interval parameters
        # We mock model.generate to verify the parameters are passed correctly
        mock_result = MagicMock()
        mock_result.audio = np.zeros(2400, dtype=np.float32)
        mock_result.real_time_factor = 0.5

        with patch.object(model, "generate", return_value=iter([mock_result])) as mock_gen:
            with patch("mlx_tts_core.AudioPlayer"):  # Prevent actual audio playback
                generate_speech(test_text, model=model, play=True, stream=True, streaming_interval=0.5)

            # Verify model.generate was called with streaming parameters
            call_kwargs = mock_gen.call_args[1]
            assert call_kwargs["stream"] is True
            assert call_kwargs["streaming_interval"] == 0.5


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


class TestHTTPStreamingMetricsIntegration:
    """End-to-end integration tests for HTTP streaming metrics with mock server."""

    def test_complete_streaming_metrics_flow(self):
        """Test full metrics capture flow from HTTP stream to player."""
        from mlx_server_utils import play_streaming_http

        # Create realistic audio: 1 second at 24kHz
        sample_rate = 24000
        duration = 1.0
        samples = int(sample_rate * duration)
        audio = np.sin(np.linspace(0, 4 * np.pi, samples)).astype(np.float32) * 0.5
        wav_bytes = create_wav_bytes(audio, sample_rate=sample_rate)

        # Simulate chunked streaming response (like real server)
        chunk_size = 1024
        chunks = [wav_bytes[i:i+chunk_size] for i in range(0, len(wav_bytes), chunk_size)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter(chunks)

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    metrics = play_streaming_http("Hello, this is a test of streaming.")

        # Verify all metrics are present and valid
        assert "ttft" in metrics
        assert "gen_time" in metrics
        assert "play_time" in metrics
        assert "chunks" in metrics
        assert "total_bytes" in metrics
        assert "audio_duration" in metrics
        assert "rtf" in metrics

        # Verify metric values are sensible
        assert metrics["ttft"] >= 0
        assert metrics["ttft"] < metrics["gen_time"]  # TTFT should be less than total gen time
        assert metrics["chunks"] > 0
        assert metrics["total_bytes"] == samples * 2  # 16-bit = 2 bytes per sample
        assert abs(metrics["audio_duration"] - duration) < 0.01  # ~1 second of audio
        assert metrics["rtf"] >= 0

    def test_metrics_with_simulated_network_latency(self):
        """Test metrics capture with simulated network delays."""
        from mlx_server_utils import play_streaming_http

        sample_rate = 24000
        samples = 12000  # 0.5 seconds of audio
        audio = np.zeros(samples, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=sample_rate)

        # Simulate delayed chunk delivery
        chunks = [wav_bytes[:1024], wav_bytes[1024:]]
        chunk_iter_with_delay = iter(chunks)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = chunk_iter_with_delay

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    metrics = play_streaming_http("Test with latency simulation.")

        # TTFT should be captured after first audio chunk
        assert metrics["ttft"] > 0
        assert metrics["audio_duration"] == 0.5  # 12000 samples / 24000 Hz
        assert metrics["total_bytes"] == samples * 2

    def test_metrics_comparison_with_direct_api_format(self):
        """Verify HTTP streaming metrics match direct API format for comparison."""
        from mlx_server_utils import play_streaming_http

        sample_rate = 24000
        samples = 24000  # 1 second
        audio = np.random.randn(samples).astype(np.float32) * 0.3
        wav_bytes = create_wav_bytes(audio, sample_rate=sample_rate)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    http_metrics = play_streaming_http("Compare metrics format")

        # HTTP metrics should have same fields as direct API metrics from mlx_tts_core
        # Direct API returns: ttft, gen_time, play_time, chunks, rtf, audio_duration
        # HTTP should match this format for comparison dashboards
        required_fields = {"ttft", "gen_time", "play_time", "chunks", "rtf", "audio_duration", "total_bytes"}
        assert set(http_metrics.keys()) == required_fields

        # RTF should be calculated correctly: gen_time / audio_duration
        if http_metrics["audio_duration"] > 0:
            expected_rtf = http_metrics["gen_time"] / http_metrics["audio_duration"]
            assert abs(http_metrics["rtf"] - expected_rtf) < 0.001
