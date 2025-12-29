"""
Unit tests for streaming HTTP client with iter_content - TDD RED phase first.

The streaming HTTP client uses `stream=True` with requests and `iter_content()`
to process audio chunks as they arrive from the server.

Run with: uv run pytest tests/unit/test_streaming_http_client.py -v
"""
import os
import sys
from unittest.mock import MagicMock, patch, PropertyMock
from io import BytesIO
import struct

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


class TestStreamingHttpClientModuleExists:
    """Tests for streaming HTTP client function existence."""

    def test_stream_tts_http_function_exists(self):
        """stream_tts_http function should exist in mlx_server_utils."""
        from mlx_server_utils import stream_tts_http
        assert stream_tts_http is not None

    def test_stream_tts_http_is_generator(self):
        """stream_tts_http should return a generator."""
        from mlx_server_utils import stream_tts_http
        import inspect
        assert inspect.isgeneratorfunction(stream_tts_http)


class TestStreamResponseYieldsChunks:
    """Tests for streaming response chunk handling."""

    def test_yields_audio_chunks_from_response(self):
        """Should yield audio chunks as they arrive from HTTP response."""
        from mlx_server_utils import stream_tts_http
        from streaming_wav_parser import WavHeader

        # Create mock response with chunked data
        audio = np.sin(np.linspace(0, 2 * np.pi, 2400)).astype(np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)

        # Simulate chunked response
        chunks = [wav_bytes[i:i+512] for i in range(0, len(wav_bytes), 512)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter(chunks)

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                collected = list(stream_tts_http("Test text"))

        # Should yield tuples of (header, audio_chunk)
        assert len(collected) > 0
        first_header, _ = collected[0]
        assert first_header.sample_rate == 24000

    def test_yields_header_with_first_chunk(self):
        """First yielded item should include parsed WAV header."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=44100, channels=2)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                results = list(stream_tts_http("Test"))

        header, _ = results[0]
        assert header.sample_rate == 44100
        assert header.channels == 2

    def test_uses_stream_true_in_request(self):
        """Should use stream=True in requests.post()."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        with patch("mlx_server_utils.requests.post", return_value=mock_response) as mock_post:
            with patch("mlx_server_utils.ensure_server_running"):
                list(stream_tts_http("Test"))

        # Verify stream=True was passed
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs.get("stream") is True

    def test_uses_iter_content_with_chunk_size(self):
        """Should use iter_content() to read chunks."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                list(stream_tts_http("Test"))

        # Verify iter_content was called
        mock_response.iter_content.assert_called_once()
        call_kwargs = mock_response.iter_content.call_args[1]
        assert call_kwargs.get("chunk_size") == 8192


class TestTimeoutHandling:
    """Tests for timeout handling in streaming requests."""

    def test_timeout_on_connection(self):
        """Should raise error on connection timeout."""
        from mlx_server_utils import stream_tts_http, TTSRequestError
        import requests

        with patch("mlx_server_utils.requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
            with patch("mlx_server_utils.ensure_server_running"):
                with pytest.raises(TTSRequestError, match="timeout|Timeout"):
                    list(stream_tts_http("Test"))

    def test_accepts_timeout_parameter(self):
        """Should accept timeout parameter."""
        from mlx_server_utils import stream_tts_http
        import inspect

        sig = inspect.signature(stream_tts_http)
        assert "timeout" in sig.parameters

    def test_passes_timeout_to_request(self):
        """Should pass timeout to requests.post()."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        with patch("mlx_server_utils.requests.post", return_value=mock_response) as mock_post:
            with patch("mlx_server_utils.ensure_server_running"):
                list(stream_tts_http("Test", timeout=30))

        call_kwargs = mock_post.call_args[1]
        assert call_kwargs.get("timeout") == 30


class TestConnectionCleanupOnError:
    """Tests for proper connection cleanup on errors."""

    def test_closes_response_on_wav_parse_error(self):
        """Should close response if WAV parsing fails."""
        from mlx_server_utils import stream_tts_http, TTSRequestError

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([b"NOT_VALID_WAV_DATA" + b"\x00" * 100])

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with pytest.raises(TTSRequestError):
                    list(stream_tts_http("Test"))

        mock_response.close.assert_called()

    def test_closes_response_on_iteration_error(self):
        """Should close response if iteration raises error."""
        from mlx_server_utils import stream_tts_http, TTSRequestError

        mock_response = MagicMock()
        mock_response.status_code = 200

        def error_iter():
            yield b"RIFF"
            raise IOError("Connection lost")

        mock_response.iter_content.return_value = error_iter()

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with pytest.raises(TTSRequestError):
                    list(stream_tts_http("Test"))

        mock_response.close.assert_called()

    def test_raises_error_on_non_200_status(self):
        """Should raise error for non-200 HTTP status."""
        from mlx_server_utils import stream_tts_http, TTSRequestError

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                with pytest.raises(TTSRequestError, match="500"):
                    list(stream_tts_http("Test"))


class TestChunkSizeHandling:
    """Tests for chunk size configuration."""

    def test_default_chunk_size_8192(self):
        """Default chunk size should be 8192 bytes."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                list(stream_tts_http("Test"))

        call_kwargs = mock_response.iter_content.call_args[1]
        assert call_kwargs.get("chunk_size") == 8192

    def test_accepts_chunk_size_parameter(self):
        """Should accept chunk_size parameter."""
        from mlx_server_utils import stream_tts_http
        import inspect

        sig = inspect.signature(stream_tts_http)
        assert "chunk_size" in sig.parameters

    def test_passes_custom_chunk_size(self):
        """Should pass custom chunk_size to iter_content()."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                list(stream_tts_http("Test", chunk_size=4096))

        call_kwargs = mock_response.iter_content.call_args[1]
        assert call_kwargs.get("chunk_size") == 4096


class TestWavParserIntegration:
    """Tests for integration with StreamingWavParser."""

    def test_uses_streaming_wav_parser(self):
        """Should use StreamingWavParser to parse chunked WAV data."""
        from mlx_server_utils import stream_tts_http

        audio = np.sin(np.linspace(0, 2 * np.pi, 2400)).astype(np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)

        # Split into multiple chunks to exercise parser
        chunk_size = 256
        chunks = [wav_bytes[i:i+chunk_size] for i in range(0, len(wav_bytes), chunk_size)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter(chunks)

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                results = list(stream_tts_http("Test"))

        # Collect all audio chunks
        all_audio = [audio for _, audio in results if audio is not None and len(audio) > 0]
        total = np.concatenate(all_audio)

        # Should have all original samples (2400)
        assert len(total) == 2400

    def test_yields_only_when_audio_available(self):
        """Should only yield when parser has audio data available."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        # First chunk is header only (44 bytes) - no audio yet
        chunks = [wav_bytes[:44], wav_bytes[44:]]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter(chunks)

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                results = list(stream_tts_http("Test"))

        # All results should have audio (empty chunks filtered)
        for header, audio_chunk in results:
            assert audio_chunk is not None
            assert len(audio_chunk) > 0

    def test_preserves_audio_quality(self):
        """Streamed audio should match original closely."""
        from mlx_server_utils import stream_tts_http

        original = np.sin(np.linspace(0, 4 * np.pi, 4800)).astype(np.float32) * 0.8
        wav_bytes = create_wav_bytes(original, sample_rate=24000)

        # Realistic chunk sizes
        chunks = [wav_bytes[i:i+1024] for i in range(0, len(wav_bytes), 1024)]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter(chunks)

        with patch("mlx_server_utils.requests.post", return_value=mock_response):
            with patch("mlx_server_utils.ensure_server_running"):
                results = list(stream_tts_http("Test"))

        all_audio = [audio for _, audio in results if audio is not None and len(audio) > 0]
        reconstructed = np.concatenate(all_audio)

        # Should closely match (16-bit quantization introduces tiny error)
        np.testing.assert_allclose(reconstructed, original, atol=0.001)


class TestVoiceParameter:
    """Tests for voice parameter handling."""

    def test_accepts_voice_parameter(self):
        """Should accept voice parameter."""
        from mlx_server_utils import stream_tts_http
        import inspect

        sig = inspect.signature(stream_tts_http)
        assert "voice" in sig.parameters

    def test_passes_voice_in_payload(self):
        """Should include voice in request payload."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        with patch("mlx_server_utils.requests.post", return_value=mock_response) as mock_post:
            with patch("mlx_server_utils.ensure_server_running"):
                list(stream_tts_http("Test", voice="alex"))

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert payload.get("voice") == "alex"


class TestEmptyTextHandling:
    """Tests for empty text input."""

    def test_empty_text_returns_empty_generator(self):
        """Empty text should return empty generator without making request."""
        from mlx_server_utils import stream_tts_http

        with patch("mlx_server_utils.requests.post") as mock_post:
            with patch("mlx_server_utils.ensure_server_running"):
                results = list(stream_tts_http(""))

        assert results == []
        mock_post.assert_not_called()

    def test_whitespace_only_returns_empty_generator(self):
        """Whitespace-only text should return empty generator."""
        from mlx_server_utils import stream_tts_http

        with patch("mlx_server_utils.requests.post") as mock_post:
            with patch("mlx_server_utils.ensure_server_running"):
                results = list(stream_tts_http("   \n\t  "))

        assert results == []
        mock_post.assert_not_called()


class TestTrueStreamingEndpoint:
    """Tests for true streaming endpoint selection."""

    def test_accepts_use_true_streaming_parameter(self):
        """Should accept use_true_streaming parameter."""
        from mlx_server_utils import stream_tts_http
        import inspect

        sig = inspect.signature(stream_tts_http)
        assert "use_true_streaming" in sig.parameters

    def test_accepts_streaming_interval_parameter(self):
        """Should accept streaming_interval parameter."""
        from mlx_server_utils import stream_tts_http
        import inspect

        sig = inspect.signature(stream_tts_http)
        assert "streaming_interval" in sig.parameters

    def test_uses_stream_endpoint_by_default(self):
        """Should use /v1/audio/speech/stream endpoint by default."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        with patch("mlx_server_utils.requests.post", return_value=mock_response) as mock_post:
            with patch("mlx_server_utils.ensure_server_running"):
                list(stream_tts_http("Test"))

        # Verify URL contains /stream
        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "/v1/audio/speech/stream" in url

    def test_uses_legacy_endpoint_when_disabled(self):
        """Should use /v1/audio/speech endpoint when use_true_streaming=False."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        with patch("mlx_server_utils.requests.post", return_value=mock_response) as mock_post:
            with patch("mlx_server_utils.ensure_server_running"):
                list(stream_tts_http("Test", use_true_streaming=False))

        # Verify URL does NOT contain /stream at the end
        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert url.endswith("/v1/audio/speech")

    def test_includes_streaming_interval_in_payload(self):
        """Should include streaming_interval in payload when use_true_streaming=True."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        with patch("mlx_server_utils.requests.post", return_value=mock_response) as mock_post:
            with patch("mlx_server_utils.ensure_server_running"):
                list(stream_tts_http("Test", streaming_interval=0.3))

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert payload.get("streaming_interval") == 0.3

    def test_excludes_streaming_interval_for_legacy_endpoint(self):
        """Should NOT include streaming_interval in payload for legacy endpoint."""
        from mlx_server_utils import stream_tts_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        with patch("mlx_server_utils.requests.post", return_value=mock_response) as mock_post:
            with patch("mlx_server_utils.ensure_server_running"):
                list(stream_tts_http("Test", use_true_streaming=False))

        call_kwargs = mock_post.call_args[1]
        payload = call_kwargs.get("json", {})
        assert "streaming_interval" not in payload

    def test_default_streaming_interval_is_half_second(self):
        """Default streaming_interval should be 0.5 seconds."""
        from mlx_server_utils import stream_tts_http
        import inspect

        sig = inspect.signature(stream_tts_http)
        default = sig.parameters["streaming_interval"].default
        assert default == 0.5


class TestPlayStreamingHttpTrueStreaming:
    """Tests for play_streaming_http with true streaming."""

    def test_play_streaming_accepts_use_true_streaming(self):
        """play_streaming_http should accept use_true_streaming parameter."""
        from mlx_server_utils import play_streaming_http
        import inspect

        sig = inspect.signature(play_streaming_http)
        assert "use_true_streaming" in sig.parameters

    def test_play_streaming_accepts_streaming_interval(self):
        """play_streaming_http should accept streaming_interval parameter."""
        from mlx_server_utils import play_streaming_http
        import inspect

        sig = inspect.signature(play_streaming_http)
        assert "streaming_interval" in sig.parameters

    def test_play_streaming_uses_stream_endpoint_by_default(self):
        """play_streaming_http should use /v1/audio/speech/stream by default."""
        from mlx_server_utils import play_streaming_http

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = iter([wav_bytes])

        mock_player = MagicMock()
        mock_player.playing = False
        mock_player.buffered_samples.return_value = 0

        with patch("mlx_server_utils.requests.post", return_value=mock_response) as mock_post:
            with patch("mlx_server_utils.ensure_server_running"):
                with patch("mlx_server_utils.AudioPlayer", return_value=mock_player):
                    play_streaming_http("Test")

        call_args = mock_post.call_args
        url = call_args[0][0] if call_args[0] else call_args[1].get("url", "")
        assert "/v1/audio/speech/stream" in url
