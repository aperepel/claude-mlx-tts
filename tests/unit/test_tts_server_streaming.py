"""
Unit tests for TTS server streaming endpoint.

Tests the /v1/audio/speech/stream endpoint that uses stream=True
for true streaming with fast TTFT.

Run with: uv run pytest tests/unit/test_tts_server_streaming.py -v
"""
import os
import struct
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestCreateWavHeader:
    """Tests for WAV header creation."""

    def test_create_wav_header_exists(self):
        """create_wav_header function should exist."""
        from tts_server import create_wav_header
        assert create_wav_header is not None

    def test_create_wav_header_returns_bytes(self):
        """Should return bytes."""
        from tts_server import create_wav_header
        header = create_wav_header(sample_rate=24000)
        assert isinstance(header, bytes)

    def test_create_wav_header_length_is_44(self):
        """WAV header should be exactly 44 bytes."""
        from tts_server import create_wav_header
        header = create_wav_header(sample_rate=24000)
        assert len(header) == 44

    def test_create_wav_header_starts_with_riff(self):
        """Header should start with 'RIFF'."""
        from tts_server import create_wav_header
        header = create_wav_header(sample_rate=24000)
        assert header[:4] == b"RIFF"

    def test_create_wav_header_contains_wave(self):
        """Header should contain 'WAVE' marker."""
        from tts_server import create_wav_header
        header = create_wav_header(sample_rate=24000)
        assert header[8:12] == b"WAVE"

    def test_create_wav_header_sample_rate(self):
        """Header should encode correct sample rate."""
        from tts_server import create_wav_header
        header = create_wav_header(sample_rate=24000)
        # Sample rate is at offset 24 (4 bytes, little-endian)
        sample_rate = struct.unpack_from("<I", header, 24)[0]
        assert sample_rate == 24000

    def test_create_wav_header_channels(self):
        """Header should encode correct channel count."""
        from tts_server import create_wav_header
        header = create_wav_header(sample_rate=24000, channels=2)
        # Channels is at offset 22 (2 bytes, little-endian)
        channels = struct.unpack_from("<H", header, 22)[0]
        assert channels == 2

    def test_create_wav_header_bits_per_sample(self):
        """Header should encode correct bits per sample."""
        from tts_server import create_wav_header
        header = create_wav_header(sample_rate=24000, bits_per_sample=16)
        # Bits per sample is at offset 34 (2 bytes, little-endian)
        bits = struct.unpack_from("<H", header, 34)[0]
        assert bits == 16

    def test_create_wav_header_default_data_size(self):
        """Default data_size should be large for streaming."""
        from tts_server import create_wav_header
        header = create_wav_header(sample_rate=24000)
        # Data size is at offset 40 (4 bytes, little-endian)
        data_size = struct.unpack_from("<I", header, 40)[0]
        assert data_size == 0x7FFFFFFF


class TestRegisterStreamingEndpoint:
    """Tests for streaming endpoint registration."""

    def test_register_streaming_endpoint_exists(self):
        """register_streaming_endpoint function should exist."""
        from tts_server import register_streaming_endpoint
        assert register_streaming_endpoint is not None

    def test_register_streaming_endpoint_adds_route(self):
        """Should add /v1/audio/speech/stream route to app."""
        from tts_server import register_streaming_endpoint
        from fastapi import FastAPI

        app = FastAPI()
        mock_model_provider = MagicMock()

        # Get routes before (use getattr since not all routes have path)
        routes_before = [getattr(r, "path", None) for r in app.routes]

        register_streaming_endpoint(app, mock_model_provider, "test-model")

        # Get routes after
        routes_after = [getattr(r, "path", None) for r in app.routes]

        # Should have added the streaming endpoint
        assert "/v1/audio/speech/stream" in routes_after
        assert "/v1/audio/speech/stream" not in routes_before


class TestStreamingEndpointBehavior:
    """Tests for streaming endpoint behavior."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock TTS model."""
        model = MagicMock()
        model.sample_rate = 24000

        # Mock generate to yield chunks
        def mock_generate(text, voice=None, stream=True, streaming_interval=0.5, **kwargs):
            # Simulate 3 chunks of audio
            for i in range(3):
                result = MagicMock()
                result.audio = np.zeros(12000, dtype=np.float32)  # 0.5s at 24kHz
                yield result

        model.generate = mock_generate
        return model

    @pytest.fixture
    def mock_model_provider(self, mock_model):
        """Create a mock model provider."""
        provider = MagicMock()
        provider.load_model.return_value = mock_model
        return provider

    def test_streaming_endpoint_returns_streaming_response(self, mock_model_provider):
        """Endpoint should return a StreamingResponse."""
        from tts_server import register_streaming_endpoint
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        register_streaming_endpoint(app, mock_model_provider, "test-model")

        client = TestClient(app)
        response = client.post(
            "/v1/audio/speech/stream",
            json={"model": "test-model", "input": "Hello"}
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

    def test_streaming_endpoint_returns_wav_header_first(self, mock_model_provider):
        """First bytes should be WAV header (RIFF...)."""
        from tts_server import register_streaming_endpoint
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        register_streaming_endpoint(app, mock_model_provider, "test-model")

        client = TestClient(app)
        response = client.post(
            "/v1/audio/speech/stream",
            json={"model": "test-model", "input": "Hello"}
        )

        # First 4 bytes should be 'RIFF'
        assert response.content[:4] == b"RIFF"
        # Bytes 8-12 should be 'WAVE'
        assert response.content[8:12] == b"WAVE"

    def test_streaming_endpoint_accepts_voice_parameter(self, mock_model_provider, mock_model):
        """Should pass voice parameter to model.generate."""
        from tts_server import register_streaming_endpoint
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        # Track calls to generate
        generate_calls = []
        original_generate = mock_model.generate
        def tracking_generate(*args, **kwargs):
            generate_calls.append(kwargs)
            return original_generate(*args, **kwargs)
        mock_model.generate = tracking_generate

        app = FastAPI()
        register_streaming_endpoint(app, mock_model_provider, "test-model")

        client = TestClient(app)
        response = client.post(
            "/v1/audio/speech/stream",
            json={"model": "test-model", "input": "Hello", "voice": "alex"}
        )

        assert response.status_code == 200
        assert len(generate_calls) == 1
        assert generate_calls[0].get("voice") == "alex"

    def test_streaming_endpoint_accepts_streaming_interval(self, mock_model_provider, mock_model):
        """Should pass streaming_interval to model.generate."""
        from tts_server import register_streaming_endpoint
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        generate_calls = []
        original_generate = mock_model.generate
        def tracking_generate(*args, **kwargs):
            generate_calls.append(kwargs)
            return original_generate(*args, **kwargs)
        mock_model.generate = tracking_generate

        app = FastAPI()
        register_streaming_endpoint(app, mock_model_provider, "test-model")

        client = TestClient(app)
        response = client.post(
            "/v1/audio/speech/stream",
            json={"model": "test-model", "input": "Hello", "streaming_interval": 0.3}
        )

        assert response.status_code == 200
        assert len(generate_calls) == 1
        assert generate_calls[0].get("streaming_interval") == 0.3

    def test_streaming_endpoint_passes_stream_true(self, mock_model_provider, mock_model):
        """Should always pass stream=True to model.generate."""
        from tts_server import register_streaming_endpoint
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        generate_calls = []
        original_generate = mock_model.generate
        def tracking_generate(*args, **kwargs):
            generate_calls.append(kwargs)
            return original_generate(*args, **kwargs)
        mock_model.generate = tracking_generate

        app = FastAPI()
        register_streaming_endpoint(app, mock_model_provider, "test-model")

        client = TestClient(app)
        response = client.post(
            "/v1/audio/speech/stream",
            json={"model": "test-model", "input": "Hello"}
        )

        assert response.status_code == 200
        assert len(generate_calls) == 1
        assert generate_calls[0].get("stream") is True

    def test_streaming_endpoint_returns_audio_data(self, mock_model_provider):
        """Should return audio data after WAV header."""
        from tts_server import register_streaming_endpoint
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()
        register_streaming_endpoint(app, mock_model_provider, "test-model")

        client = TestClient(app)
        response = client.post(
            "/v1/audio/speech/stream",
            json={"model": "test-model", "input": "Hello"}
        )

        # Should have header (44 bytes) + audio data
        # OLA processor may adjust sample count due to crossfade, so check for
        # reasonable audio data rather than exact byte count
        assert len(response.content) >= 44  # At least WAV header
        # 3 chunks * 12000 samples * 2 bytes = 72000 bytes expected (may vary with OLA)
        # Allow some variance due to OLA crossfade processing
        audio_bytes = len(response.content) - 44
        assert audio_bytes > 0, "Should have audio data after header"
        # Audio bytes should be even (int16 samples)
        assert audio_bytes % 2 == 0, "Audio data should be 16-bit aligned"


class TestStreamingEndpointHeaders:
    """Tests for streaming endpoint HTTP headers."""

    @pytest.fixture
    def app_with_endpoint(self):
        """Create app with streaming endpoint."""
        from tts_server import register_streaming_endpoint
        from fastapi import FastAPI

        app = FastAPI()
        mock_provider = MagicMock()
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_model.generate = lambda *args, **kwargs: iter([
            MagicMock(audio=np.zeros(1000, dtype=np.float32))
        ])
        mock_provider.load_model.return_value = mock_model

        register_streaming_endpoint(app, mock_provider, "test-model")
        return app

    def test_content_type_is_audio_wav(self, app_with_endpoint):
        """Content-Type should be audio/wav."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_endpoint)
        response = client.post(
            "/v1/audio/speech/stream",
            json={"model": "test-model", "input": "Hello"}
        )

        assert response.headers["content-type"] == "audio/wav"

    def test_has_streaming_header(self, app_with_endpoint):
        """Should include X-Streaming header."""
        from fastapi.testclient import TestClient

        client = TestClient(app_with_endpoint)
        response = client.post(
            "/v1/audio/speech/stream",
            json={"model": "test-model", "input": "Hello"}
        )

        assert response.headers.get("x-streaming") == "true"
