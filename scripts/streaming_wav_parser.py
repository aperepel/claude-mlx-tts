"""
Streaming WAV chunk parser for HTTP streaming response.

Parses WAV audio data arriving in chunks, handling:
- Header parsing from first chunk (44 bytes minimum)
- Partial chunk boundaries (data split mid-sample)
- Incremental audio extraction as chunks arrive

Usage:
    from streaming_wav_parser import StreamingWavParser

    parser = StreamingWavParser()
    for chunk in http_response.iter_content(chunk_size=8192):
        parser.feed(chunk)
        audio = parser.read_audio()
        if audio is not None and len(audio) > 0:
            player.queue_audio(audio)

    print(f"Sample rate: {parser.header.sample_rate}")
    print(f"Complete: {parser.is_complete}")
"""
import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np


class WavParseError(Exception):
    """Raised when WAV parsing fails due to invalid format."""
    pass


@dataclass
class WavHeader:
    """WAV file header metadata."""
    sample_rate: int
    channels: int
    bits_per_sample: int
    data_size: int  # Size of audio data section in bytes


# WAV header size (standard RIFF/WAVE PCM format)
WAV_HEADER_SIZE = 44


class StreamingWavParser:
    """
    Incremental WAV parser for streaming audio data.

    Handles WAV data arriving in arbitrary chunks, buffering partial
    samples and providing parsed audio data as it becomes available.
    """

    def __init__(self):
        self._buffer = bytearray()
        self._header: Optional[WavHeader] = None
        self._header_parsed = False
        self._bytes_read = 0  # Audio bytes extracted (not including header)

    @property
    def header(self) -> Optional[WavHeader]:
        """Parsed WAV header, or None if not yet available."""
        return self._header

    @property
    def is_complete(self) -> bool:
        """True if all declared audio data has been received and read."""
        if self._header is None:
            return False
        return self._bytes_read >= self._header.data_size

    @property
    def bytes_remaining(self) -> int:
        """Number of audio data bytes not yet read."""
        if self._header is None:
            return 0
        return self._header.data_size - self._bytes_read

    def feed(self, data: bytes) -> None:
        """
        Feed raw bytes from HTTP response chunk.

        Args:
            data: Raw bytes from response chunk.

        Raises:
            WavParseError: If WAV format is invalid.
        """
        if not data:
            return

        self._buffer.extend(data)

        # Try to parse header if not done yet
        if not self._header_parsed and len(self._buffer) >= WAV_HEADER_SIZE:
            self._parse_header()

    def read_audio(self) -> Optional[np.ndarray]:
        """
        Read available audio data from buffer.

        Returns normalized float32 samples. Buffers partial samples
        (e.g., half of a 16-bit sample) until complete.

        Returns:
            Float32 numpy array of audio samples, or None/empty if no data.
        """
        if self._header is None:
            return None

        # Calculate how many complete samples we have
        bytes_per_sample = self._header.bits_per_sample // 8
        bytes_per_frame = bytes_per_sample * self._header.channels

        available_bytes = len(self._buffer)
        complete_frames = available_bytes // bytes_per_frame
        bytes_to_read = complete_frames * bytes_per_frame

        if bytes_to_read == 0:
            return None

        # Extract complete samples
        audio_bytes = bytes(self._buffer[:bytes_to_read])
        del self._buffer[:bytes_to_read]
        self._bytes_read += bytes_to_read

        # Convert to float32
        if self._header.bits_per_sample == 16:
            # 16-bit signed PCM -> float32
            audio_int = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int.astype(np.float32) / 32767.0
        elif self._header.bits_per_sample == 24:
            # 24-bit PCM -> float32 (less common but handle it)
            # Read as bytes and convert manually
            n_samples = len(audio_bytes) // 3
            audio_float = np.zeros(n_samples, dtype=np.float32)
            for i in range(n_samples):
                b = audio_bytes[i*3:(i+1)*3]
                # Sign-extend 24-bit to 32-bit
                val = int.from_bytes(b, byteorder='little', signed=True)
                audio_float[i] = val / 8388607.0  # 2^23 - 1
        elif self._header.bits_per_sample == 8:
            # 8-bit unsigned PCM -> float32
            audio_uint = np.frombuffer(audio_bytes, dtype=np.uint8)
            audio_float = (audio_uint.astype(np.float32) - 128) / 127.0
        else:
            raise WavParseError(f"Unsupported bits per sample: {self._header.bits_per_sample}")

        # Handle stereo -> mono if needed (or keep as-is for multi-channel)
        if self._header.channels > 1:
            # Reshape to (frames, channels) and keep interleaved
            audio_float = audio_float.reshape(-1, self._header.channels)
            # For now, return interleaved; caller can handle channel separation
            audio_float = audio_float.flatten()

        return audio_float

    def _parse_header(self) -> None:
        """Parse WAV header from buffer."""
        if len(self._buffer) < WAV_HEADER_SIZE:
            return

        header_bytes = bytes(self._buffer[:WAV_HEADER_SIZE])

        # Validate RIFF header
        if header_bytes[:4] != b"RIFF":
            raise WavParseError(f"Invalid WAV: missing RIFF header (got {header_bytes[:4]!r})")

        # Validate WAVE marker
        if header_bytes[8:12] != b"WAVE":
            raise WavParseError(f"Invalid WAV: missing WAVE marker (got {header_bytes[8:12]!r})")

        # Validate fmt chunk
        if header_bytes[12:16] != b"fmt ":
            raise WavParseError(f"Invalid WAV: missing fmt chunk (got {header_bytes[12:16]!r})")

        # Parse format info
        # fmt chunk: <size:4><format:2><channels:2><sample_rate:4><byte_rate:4><block_align:2><bits:2>
        audio_format = struct.unpack_from("<H", header_bytes, 20)[0]
        channels = struct.unpack_from("<H", header_bytes, 22)[0]
        sample_rate = struct.unpack_from("<I", header_bytes, 24)[0]
        bits_per_sample = struct.unpack_from("<H", header_bytes, 34)[0]

        # Validate PCM format (we only support uncompressed PCM)
        if audio_format != 1:
            raise WavParseError(f"Unsupported audio format: {audio_format} (only PCM=1 supported)")

        # Parse data chunk
        if header_bytes[36:40] != b"data":
            raise WavParseError(f"Invalid WAV: missing data chunk (got {header_bytes[36:40]!r})")

        data_size = struct.unpack_from("<I", header_bytes, 40)[0]

        self._header = WavHeader(
            sample_rate=sample_rate,
            channels=channels,
            bits_per_sample=bits_per_sample,
            data_size=data_size,
        )
        self._header_parsed = True

        # Remove header from buffer, leaving only audio data
        del self._buffer[:WAV_HEADER_SIZE]
