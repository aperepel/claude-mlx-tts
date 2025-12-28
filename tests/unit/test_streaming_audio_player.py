"""
Unit tests for streaming HTTP playback with AudioPlayer integration - TDD RED phase.

This module tests the play_streaming_http() function which:
- Consumes stream_tts_http() generator
- Uses AudioPlayer for low-latency streaming playback
- Applies stateful audio compression across chunks
- Captures TTFT and streaming metrics

Run with: uv run pytest tests/unit/test_streaming_audio_player.py -v
"""
import os
import sys
import time
import struct
from unittest.mock import MagicMock, patch, PropertyMock, call

import numpy as np
import pytest

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


class TestPlayStreamingHttpExists:
    """Tests for play_streaming_http function existence."""

    def test_play_streaming_http_function_exists(self):
        """play_streaming_http function should exist in mlx_server_utils."""
        from mlx_server_utils import play_streaming_http
        assert play_streaming_http is not None

    def test_play_streaming_http_returns_metrics(self):
        """play_streaming_http should return metrics dict."""
        from mlx_server_utils import play_streaming_http
        import inspect

        sig = inspect.signature(play_streaming_http)
        # Should have text parameter at minimum
        assert "text" in sig.parameters


class TestAudioPlayerIntegration:
    """Tests for AudioPlayer receiving chunks from HTTP stream."""

    def test_creates_audio_player_with_sample_rate(self):
        """Should create AudioPlayer with correct sample rate from header."""
        from mlx_server_utils import play_streaming_http
        from streaming_wav_parser import WavHeader

        audio = np.sin(np.linspace(0, 2 * np.pi, 2400)).astype(np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)
        chunks = [wav_bytes]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter(chunks)

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player) as mock_ap:
                    play_streaming_http("Test text")

        # AudioPlayer should be created with sample_rate=24000
        mock_ap.assert_called_once()
        call_kwargs = mock_ap.call_args[1]
        assert call_kwargs.get("sample_rate") == 24000

    def test_queues_audio_chunks_to_player(self):
        """Should queue each audio chunk to AudioPlayer."""
        from mlx_server_utils import play_streaming_http

        # Create audio that will produce multiple chunks
        audio = np.sin(np.linspace(0, 2 * np.pi, 4800)).astype(np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)

        # Split into multiple chunks
        chunk_size = 512
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
                    play_streaming_http("Test text")

        # queue_audio should have been called for each audio chunk
        assert mock_player.queue_audio.call_count > 0

    def test_waits_for_drain_after_streaming(self):
        """Should call wait_for_drain after all chunks queued."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    play_streaming_http("Test text")

        mock_player.wait_for_drain.assert_called_once()


class TestAudioCompression:
    """Tests for audio compression stateful across chunks."""

    def test_creates_audio_processor(self):
        """Should create audio processor for compression."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        mock_processor = MagicMock(return_value=np.zeros(100, dtype=np.float32))

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    with patch("mlx_server_utils.create_processor", return_value=mock_processor) as mock_create:
                        play_streaming_http("Test text")

        # create_processor should be called with sample_rate
        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs.get("sample_rate") == 24000

    def test_processes_audio_through_compressor(self):
        """Audio chunks should pass through the processor."""
        from mlx_server_utils import play_streaming_http

        audio = np.sin(np.linspace(0, 2 * np.pi, 2400)).astype(np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        processed_chunks = []

        def track_processor(chunk):
            processed = chunk.copy()
            processed_chunks.append(processed)
            return processed

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    with patch("mlx_server_utils.create_processor", return_value=track_processor):
                        play_streaming_http("Test text")

        # Processor should have been called for audio chunks
        assert len(processed_chunks) > 0

    def test_uses_single_processor_across_chunks(self):
        """Should use the same processor instance for all chunks (stateful)."""
        from mlx_server_utils import play_streaming_http

        audio = np.sin(np.linspace(0, 2 * np.pi, 4800)).astype(np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)

        # Split into multiple chunks
        chunk_size = 256
        chunks = [wav_bytes[i:i+chunk_size] for i in range(0, len(wav_bytes), chunk_size)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter(chunks)

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        mock_processor = MagicMock(side_effect=lambda x: x)

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    with patch("mlx_server_utils.create_processor", return_value=mock_processor) as mock_create:
                        play_streaming_http("Test text")

        # create_processor should only be called once
        assert mock_create.call_count == 1
        # But processor should be called multiple times
        assert mock_processor.call_count > 1


class TestTTFTMetric:
    """Tests for TTFT (time to first token) metric capture."""

    def test_returns_ttft_in_metrics(self):
        """Metrics should include ttft (time to first audio)."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    metrics = play_streaming_http("Test text")

        assert "ttft" in metrics
        assert isinstance(metrics["ttft"], float)
        assert metrics["ttft"] >= 0

    def test_returns_gen_time_in_metrics(self):
        """Metrics should include gen_time (total generation time)."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    metrics = play_streaming_http("Test text")

        assert "gen_time" in metrics
        assert isinstance(metrics["gen_time"], float)

    def test_returns_play_time_in_metrics(self):
        """Metrics should include play_time (drain wait time)."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    metrics = play_streaming_http("Test text")

        assert "play_time" in metrics
        assert isinstance(metrics["play_time"], float)

    def test_returns_chunks_count_in_metrics(self):
        """Metrics should include chunks (number of audio chunks)."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(4800, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        # Split to get multiple chunks
        chunk_size = 256
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
                    metrics = play_streaming_http("Test text")

        assert "chunks" in metrics
        assert isinstance(metrics["chunks"], int)
        assert metrics["chunks"] > 0


class TestPlaybackLifecycle:
    """Tests for playback lifecycle (start, drain, cleanup)."""

    def test_handles_short_audio_clips(self):
        """Should handle short audio that may not meet min buffer threshold."""
        from mlx_server_utils import play_streaming_http

        # Very short audio
        audio = np.zeros(240, dtype=np.float32)  # 10ms at 24kHz
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = False  # Not auto-started
        mock_player.buffered_samples.return_value = 240

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    # Should not raise - handles short audio gracefully
                    metrics = play_streaming_http("Hi")

        # For short audio that didn't auto-start, should call start_stream
        mock_player.start_stream.assert_called()

    def test_skips_drain_when_no_audio(self):
        """Should skip drain wait when no audio was queued."""
        from mlx_server_utils import play_streaming_http

        # Empty generator (text too short, filtered)
        with patch("mlx_server_utils.stream_tts_http") as mock_stream:
            mock_stream.return_value = iter([])

            with patch("mlx_server_utils.ensure_server_running"):
                metrics = play_streaming_http("")

        # Should return early with zero metrics
        assert metrics["chunks"] == 0

    def test_cleanup_on_error(self):
        """Should handle cleanup when iteration error occurs."""
        from mlx_server_utils import play_streaming_http, TTSRequestError

        mock_response = MagicMock()
        mock_response.status_code = 200

        def error_iter():
            yield b"RIFF"
            raise IOError("Connection lost")

        mock_response.iter_content.return_value = error_iter()

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with pytest.raises(TTSRequestError):
                    play_streaming_http("Test")


class TestVoiceParameter:
    """Tests for voice parameter passthrough."""

    def test_passes_voice_to_stream_tts_http(self):
        """Voice parameter should be passed to underlying stream function."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response) as mock_post:
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    play_streaming_http("Test text", voice="jerry")

        # Voice should be in payload
        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert payload.get("voice") == "jerry"


class TestEmptyTextHandling:
    """Tests for empty text input."""

    def test_empty_text_returns_zero_metrics(self):
        """Empty text should return metrics with zeros."""
        from mlx_server_utils import play_streaming_http

        metrics = play_streaming_http("")

        assert metrics["ttft"] == 0.0
        assert metrics["gen_time"] == 0.0
        assert metrics["play_time"] == 0.0
        assert metrics["chunks"] == 0

    def test_whitespace_only_returns_zero_metrics(self):
        """Whitespace-only text should return metrics with zeros."""
        from mlx_server_utils import play_streaming_http

        metrics = play_streaming_http("   \n\t  ")

        assert metrics["chunks"] == 0


class TestTotalBytesMetric:
    """Tests for total_bytes metric (audio data transferred)."""

    def test_returns_total_bytes_in_metrics(self):
        """Metrics should include total_bytes (audio data transferred)."""
        from mlx_server_utils import play_streaming_http

        # Create 2400 samples = 4800 bytes (16-bit PCM)
        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    metrics = play_streaming_http("Test text")

        assert "total_bytes" in metrics
        assert isinstance(metrics["total_bytes"], int)

    def test_total_bytes_matches_audio_data(self):
        """total_bytes should match actual audio data size."""
        from mlx_server_utils import play_streaming_http

        # 2400 samples * 2 bytes/sample = 4800 bytes
        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    metrics = play_streaming_http("Test text")

        # Should be 4800 bytes (2400 samples * 2 bytes per 16-bit sample)
        assert metrics["total_bytes"] == 4800

    def test_total_bytes_accumulates_across_chunks(self):
        """total_bytes should accumulate from all chunks."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(4800, dtype=np.float32)  # 9600 bytes
        wav_bytes = create_wav_bytes(audio)

        # Split into multiple chunks
        chunk_size = 256
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
                    metrics = play_streaming_http("Test text")

        # 4800 samples * 2 bytes = 9600 bytes
        assert metrics["total_bytes"] == 9600


class TestAudioDurationMetric:
    """Tests for audio_duration metric (total audio time)."""

    def test_returns_audio_duration_in_metrics(self):
        """Metrics should include audio_duration (seconds)."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(24000, dtype=np.float32)  # 1 second at 24kHz
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    metrics = play_streaming_http("Test text")

        assert "audio_duration" in metrics
        assert isinstance(metrics["audio_duration"], float)

    def test_audio_duration_calculated_correctly(self):
        """audio_duration should be samples / sample_rate."""
        from mlx_server_utils import play_streaming_http

        # 24000 samples at 24kHz = 1.0 second
        audio = np.zeros(24000, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    metrics = play_streaming_http("Test text")

        assert abs(metrics["audio_duration"] - 1.0) < 0.01


class TestRTFMetric:
    """Tests for RTF (real-time factor) metric."""

    def test_returns_rtf_in_metrics(self):
        """Metrics should include rtf (real-time factor)."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(24000, dtype=np.float32)  # 1 second audio
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    metrics = play_streaming_http("Test text")

        assert "rtf" in metrics
        assert isinstance(metrics["rtf"], float)

    def test_rtf_is_gen_time_over_audio_duration(self):
        """RTF should be gen_time / audio_duration."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(24000, dtype=np.float32)  # 1 second audio
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = True
        mock_player.buffered_samples.return_value = 100

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    metrics = play_streaming_http("Test text")

        # RTF = gen_time / audio_duration
        if metrics["audio_duration"] > 0:
            expected_rtf = metrics["gen_time"] / metrics["audio_duration"]
            assert abs(metrics["rtf"] - expected_rtf) < 0.01

    def test_rtf_zero_when_no_audio_duration(self):
        """RTF should be 0 when audio_duration is 0."""
        from mlx_server_utils import play_streaming_http

        # Empty audio
        metrics = play_streaming_http("")

        assert metrics["rtf"] == 0.0


class TestEmptyTextNewMetrics:
    """Tests for empty text handling with new metrics."""

    def test_empty_text_has_zero_total_bytes(self):
        """Empty text should have zero total_bytes."""
        from mlx_server_utils import play_streaming_http

        metrics = play_streaming_http("")

        assert metrics["total_bytes"] == 0

    def test_empty_text_has_zero_audio_duration(self):
        """Empty text should have zero audio_duration."""
        from mlx_server_utils import play_streaming_http

        metrics = play_streaming_http("")

        assert metrics["audio_duration"] == 0.0
