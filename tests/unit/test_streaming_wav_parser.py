"""
Unit tests for streaming WAV chunk parser - TDD RED phase first.

The streaming WAV parser handles incremental parsing of WAV audio data
arriving in chunks from HTTP streaming response.

Run with: uv run pytest tests/unit/test_streaming_wav_parser.py -v
"""
import os
import sys
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
    # Normalize and convert to 16-bit PCM
    if audio.dtype == np.float32 or audio.dtype == np.float64:
        audio_int = (audio * 32767).astype(np.int16)
    else:
        audio_int = audio.astype(np.int16)

    audio_bytes = audio_int.tobytes()
    data_size = len(audio_bytes)

    # Build WAV header (44 bytes)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,  # File size - 8
        b"WAVE",
        b"fmt ",
        16,  # Format chunk size (PCM)
        1,  # Audio format (1 = PCM)
        channels,
        sample_rate,
        sample_rate * channels * bits_per_sample // 8,  # Byte rate
        channels * bits_per_sample // 8,  # Block align
        bits_per_sample,
        b"data",
        data_size,
    )

    return header + audio_bytes


class TestStreamingWavParserModuleExists:
    """Tests for the streaming_wav_parser module existence and structure."""

    def test_module_exists(self):
        """streaming_wav_parser module should be importable."""
        import streaming_wav_parser
        assert streaming_wav_parser is not None

    def test_streaming_wav_parser_class_exists(self):
        """StreamingWavParser class should exist."""
        from streaming_wav_parser import StreamingWavParser
        assert StreamingWavParser is not None

    def test_wav_header_dataclass_exists(self):
        """WavHeader dataclass should exist for header metadata."""
        from streaming_wav_parser import WavHeader
        assert WavHeader is not None


class TestWavHeaderParsing:
    """Tests for WAV header parsing from first chunk."""

    def test_parse_valid_wav_header(self):
        """Should parse sample rate, channels, bits from valid WAV header."""
        from streaming_wav_parser import StreamingWavParser

        # Create valid WAV with known parameters
        audio = np.sin(np.linspace(0, 2 * np.pi, 2400)).astype(np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=24000, channels=1, bits_per_sample=16)

        parser = StreamingWavParser()
        parser.feed(wav_bytes[:100])  # Feed first chunk containing header

        assert parser.header is not None
        assert parser.header.sample_rate == 24000
        assert parser.header.channels == 1
        assert parser.header.bits_per_sample == 16

    def test_parse_44100_sample_rate(self):
        """Should correctly parse 44100 Hz sample rate."""
        from streaming_wav_parser import StreamingWavParser

        audio = np.zeros(4410, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=44100)

        parser = StreamingWavParser()
        parser.feed(wav_bytes)

        assert parser.header is not None
        assert parser.header.sample_rate == 44100

    def test_parse_stereo_channels(self):
        """Should correctly parse stereo (2 channel) audio."""
        from streaming_wav_parser import StreamingWavParser

        # Create stereo audio
        audio = np.zeros((2400, 2), dtype=np.float32)
        audio_interleaved = audio.flatten()
        wav_bytes = create_wav_bytes(audio_interleaved, channels=2)

        parser = StreamingWavParser()
        parser.feed(wav_bytes)

        assert parser.header is not None
        assert parser.header.channels == 2

    def test_parse_data_size(self):
        """Should extract audio data size from header."""
        from streaming_wav_parser import StreamingWavParser

        audio = np.zeros(2400, dtype=np.float32)  # 2400 samples = 4800 bytes in 16-bit
        wav_bytes = create_wav_bytes(audio)

        parser = StreamingWavParser()
        parser.feed(wav_bytes)

        assert parser.header is not None
        assert parser.header.data_size == 4800  # 2400 samples * 2 bytes

    def test_header_not_available_until_enough_bytes(self):
        """Header should be None until at least 44 bytes received."""
        from streaming_wav_parser import StreamingWavParser

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        parser = StreamingWavParser()

        # Feed only 20 bytes - not enough for header
        parser.feed(wav_bytes[:20])
        assert parser.header is None

        # Feed more to complete header
        parser.feed(wav_bytes[20:44])
        assert parser.header is not None


class TestPartialChunkHandling:
    """Tests for handling chunks split at arbitrary boundaries."""

    def test_header_split_across_chunks(self):
        """Should handle WAV header split across multiple chunks."""
        from streaming_wav_parser import StreamingWavParser

        audio = np.zeros(2400, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio, sample_rate=24000)

        parser = StreamingWavParser()

        # Feed header in small pieces
        parser.feed(wav_bytes[:10])
        assert parser.header is None

        parser.feed(wav_bytes[10:30])
        assert parser.header is None

        parser.feed(wav_bytes[30:50])
        assert parser.header is not None
        assert parser.header.sample_rate == 24000

    def test_data_split_at_sample_boundary(self):
        """Should correctly yield samples when data splits at sample boundary."""
        from streaming_wav_parser import StreamingWavParser

        # 100 samples of a simple pattern
        audio = np.arange(100, dtype=np.float32) / 100
        wav_bytes = create_wav_bytes(audio)

        parser = StreamingWavParser()

        # Feed in chunks, collecting all audio
        collected = []
        chunk_size = 50
        for i in range(0, len(wav_bytes), chunk_size):
            chunk = wav_bytes[i:i + chunk_size]
            parser.feed(chunk)
            audio_chunk = parser.read_audio()
            if audio_chunk is not None and len(audio_chunk) > 0:
                collected.append(audio_chunk)

        # Should have all 100 samples
        total = np.concatenate(collected) if collected else np.array([])
        assert len(total) == 100

    def test_data_split_mid_sample(self):
        """Should handle data split in middle of a 16-bit sample (2 bytes)."""
        from streaming_wav_parser import StreamingWavParser

        audio = np.arange(100, dtype=np.float32) / 100
        wav_bytes = create_wav_bytes(audio)

        parser = StreamingWavParser()

        # Feed header + 3 bytes (1.5 samples)
        parser.feed(wav_bytes[:47])  # 44 header + 3 data bytes
        audio1 = parser.read_audio()

        # Should have 1 complete sample (parser buffers partial)
        assert audio1 is not None
        assert len(audio1) == 1

        # Feed rest
        parser.feed(wav_bytes[47:])
        audio2 = parser.read_audio()

        # Should have remaining 99 samples
        assert audio2 is not None
        assert len(audio2) == 99


class TestAudioDataIteration:
    """Tests for iterating over audio data from chunks."""

    def test_read_audio_returns_float32(self):
        """read_audio should return float32 numpy array."""
        from streaming_wav_parser import StreamingWavParser

        audio = np.sin(np.linspace(0, 2 * np.pi, 2400)).astype(np.float32) * 0.5
        wav_bytes = create_wav_bytes(audio)

        parser = StreamingWavParser()
        parser.feed(wav_bytes)
        result = parser.read_audio()

        assert result is not None
        assert result.dtype == np.float32

    def test_read_audio_normalized_range(self):
        """Audio values should be in [-1.0, 1.0] range."""
        from streaming_wav_parser import StreamingWavParser

        # Create audio that uses full 16-bit range
        audio = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        parser = StreamingWavParser()
        parser.feed(wav_bytes)
        result = parser.read_audio()

        assert result is not None
        assert np.amax(result) <= 1.0
        assert np.amin(result) >= -1.0

    def test_read_audio_preserves_waveform(self):
        """Parsed audio should closely match original."""
        from streaming_wav_parser import StreamingWavParser

        original = np.sin(np.linspace(0, 4 * np.pi, 2400)).astype(np.float32) * 0.8
        wav_bytes = create_wav_bytes(original)

        parser = StreamingWavParser()
        parser.feed(wav_bytes)
        result = parser.read_audio()

        # Should be close (16-bit quantization introduces small error)
        assert result is not None
        np.testing.assert_allclose(result, original, atol=0.001)

    def test_read_audio_empty_before_data(self):
        """read_audio should return empty/None before audio data available."""
        from streaming_wav_parser import StreamingWavParser

        parser = StreamingWavParser()
        result = parser.read_audio()

        assert result is None or len(result) == 0

    def test_multiple_reads_drains_buffer(self):
        """Multiple read_audio calls should drain buffer (no duplicates)."""
        from streaming_wav_parser import StreamingWavParser

        audio = np.arange(100, dtype=np.float32) / 100
        wav_bytes = create_wav_bytes(audio)

        parser = StreamingWavParser()
        parser.feed(wav_bytes)

        result1 = parser.read_audio()
        result2 = parser.read_audio()

        assert result1 is not None
        assert len(result1) == 100
        assert result2 is None or len(result2) == 0


class TestErrorHandling:
    """Tests for error handling with malformed input."""

    def test_invalid_riff_header(self):
        """Should raise error for invalid RIFF header."""
        from streaming_wav_parser import StreamingWavParser, WavParseError

        parser = StreamingWavParser()

        with pytest.raises(WavParseError, match="RIFF"):
            parser.feed(b"NOT_RIFF" + b"\x00" * 36)

    def test_invalid_wave_marker(self):
        """Should raise error if WAVE marker missing."""
        from streaming_wav_parser import StreamingWavParser, WavParseError

        # Valid RIFF but invalid WAVE marker
        bad_wav = b"RIFF" + struct.pack("<I", 100) + b"NOTW" + b"\x00" * 32
        parser = StreamingWavParser()

        with pytest.raises(WavParseError, match="WAVE"):
            parser.feed(bad_wav)

    def test_unsupported_audio_format(self):
        """Should raise error for non-PCM audio format."""
        from streaming_wav_parser import StreamingWavParser, WavParseError

        # Create header with format=3 (float) instead of 1 (PCM)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36, b"WAVE", b"fmt ", 16,
            3,  # Format 3 = IEEE float (we only support PCM=1)
            1, 24000, 48000, 2, 16,
            b"data", 0,
        )

        parser = StreamingWavParser()

        with pytest.raises(WavParseError, match="PCM"):
            parser.feed(header)

    def test_truncated_input_graceful(self):
        """Truncated input should not crash, just wait for more data."""
        from streaming_wav_parser import StreamingWavParser

        parser = StreamingWavParser()

        # Feed truncated data - should not raise
        parser.feed(b"RIFF")
        assert parser.header is None  # Still waiting

        parser.feed(b"\x00" * 10)
        assert parser.header is None  # Still waiting

    def test_empty_feed(self):
        """Empty feed should be no-op."""
        from streaming_wav_parser import StreamingWavParser

        parser = StreamingWavParser()
        parser.feed(b"")  # Should not raise
        assert parser.header is None


class TestStreamingIntegration:
    """Integration tests simulating realistic streaming scenarios."""

    def test_simulate_http_chunked_response(self):
        """Simulate realistic HTTP chunked response with variable chunk sizes."""
        from streaming_wav_parser import StreamingWavParser

        # Create 1 second of audio at 24kHz
        duration = 1.0
        sample_rate = 24000
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.5  # 440 Hz tone
        wav_bytes = create_wav_bytes(audio, sample_rate=sample_rate)

        parser = StreamingWavParser()
        collected_audio = []

        # Simulate variable chunk sizes (like HTTP responses)
        chunk_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        offset = 0
        chunk_idx = 0

        while offset < len(wav_bytes):
            size = chunk_sizes[chunk_idx % len(chunk_sizes)]
            chunk = wav_bytes[offset:offset + size]
            parser.feed(chunk)

            audio_chunk = parser.read_audio()
            if audio_chunk is not None and len(audio_chunk) > 0:
                collected_audio.append(audio_chunk)

            offset += size
            chunk_idx += 1

        # Verify all audio received
        total_audio = np.concatenate(collected_audio)
        assert len(total_audio) == sample_rate  # 1 second = 24000 samples

        # Verify header parsed correctly
        assert parser.header is not None
        assert parser.header.sample_rate == sample_rate

    def test_is_complete_property(self):
        """is_complete should be True when all declared data received."""
        from streaming_wav_parser import StreamingWavParser

        audio = np.zeros(1000, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)

        parser = StreamingWavParser()

        # Feed partial
        parser.feed(wav_bytes[:100])
        assert not parser.is_complete

        # Feed rest
        parser.feed(wav_bytes[100:])
        parser.read_audio()  # Drain buffer
        assert parser.is_complete

    def test_bytes_remaining_property(self):
        """bytes_remaining should track unread audio data."""
        from streaming_wav_parser import StreamingWavParser

        audio = np.zeros(1000, dtype=np.float32)
        wav_bytes = create_wav_bytes(audio)
        data_size = 1000 * 2  # 1000 samples * 2 bytes

        parser = StreamingWavParser()
        parser.feed(wav_bytes)

        assert parser.bytes_remaining == data_size

        parser.read_audio()
        assert parser.bytes_remaining == 0
