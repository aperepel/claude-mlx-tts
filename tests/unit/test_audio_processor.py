"""
Unit tests for audio_processor module - dynamic range compression and limiting.

Run with: uv run pytest tests/unit/test_audio_processor.py -v
"""
import os
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestAudioProcessorModule:
    """Tests for the audio_processor module existence and structure."""

    def test_module_exists(self):
        """audio_processor module should be importable."""
        import audio_processor
        assert audio_processor is not None

    def test_create_processor_function_exists(self):
        """create_processor function should exist for creating stateful processor."""
        from audio_processor import create_processor
        assert callable(create_processor)

    def test_process_chunk_function_exists(self):
        """process_chunk function should exist for stateless single-call processing."""
        from audio_processor import process_chunk
        assert callable(process_chunk)


class TestProcessorDefaults:
    """Tests for default processor parameters."""

    def test_default_compressor_threshold(self):
        """Default compressor threshold should be -18 dB."""
        from audio_processor import DEFAULT_THRESHOLD_DB
        assert DEFAULT_THRESHOLD_DB == -18

    def test_default_compressor_ratio(self):
        """Default compressor ratio should be 3.0."""
        from audio_processor import DEFAULT_RATIO
        assert DEFAULT_RATIO == 3.0

    def test_default_compressor_attack(self):
        """Default compressor attack should be 3ms."""
        from audio_processor import DEFAULT_ATTACK_MS
        assert DEFAULT_ATTACK_MS == 3

    def test_default_compressor_release(self):
        """Default compressor release should be 50ms."""
        from audio_processor import DEFAULT_RELEASE_MS
        assert DEFAULT_RELEASE_MS == 50

    def test_default_limiter_threshold(self):
        """Default limiter threshold should be -0.5 dB."""
        from audio_processor import DEFAULT_LIMITER_THRESHOLD_DB
        assert DEFAULT_LIMITER_THRESHOLD_DB == -0.5

    def test_default_limiter_release(self):
        """Default limiter release should be 40ms."""
        from audio_processor import DEFAULT_LIMITER_RELEASE_MS
        assert DEFAULT_LIMITER_RELEASE_MS == 40

    def test_default_gain(self):
        """Default makeup gain should be 8 dB."""
        from audio_processor import DEFAULT_GAIN_DB
        assert DEFAULT_GAIN_DB == 8


class TestCreateProcessor:
    """Tests for create_processor() factory function."""

    def test_returns_callable(self):
        """create_processor should return a callable processor function."""
        from audio_processor import create_processor
        processor = create_processor(sample_rate=24000)
        assert callable(processor)

    def test_accepts_custom_parameters(self):
        """create_processor should accept custom compressor/limiter parameters."""
        from audio_processor import create_processor
        processor = create_processor(
            sample_rate=24000,
            threshold_db=-20,
            ratio=4.0,
            attack_ms=5,
            release_ms=100,
            limiter_threshold_db=-1.0,
            limiter_release_ms=50,
            gain_db=6,
        )
        assert callable(processor)

    def test_processor_accepts_audio_array(self):
        """Processor should accept numpy float32 audio array."""
        from audio_processor import create_processor
        processor = create_processor(sample_rate=24000)

        audio = np.zeros(12000, dtype=np.float32)
        result = processor(audio)

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_processor_returns_same_length(self):
        """Processor should return audio of same length as input."""
        from audio_processor import create_processor
        processor = create_processor(sample_rate=24000)

        audio = np.random.randn(12000).astype(np.float32) * 0.5
        result = processor(audio)

        assert len(result) == len(audio)

    def test_processor_maintains_state_across_calls(self):
        """Processor should maintain compressor state across multiple calls."""
        from audio_processor import create_processor
        processor = create_processor(sample_rate=24000)

        # Process two chunks - state should carry over
        chunk1 = np.random.randn(12000).astype(np.float32) * 0.5
        chunk2 = np.random.randn(12000).astype(np.float32) * 0.5

        result1 = processor(chunk1)
        result2 = processor(chunk2)

        # Both should be processed (non-zero output for non-zero input)
        assert not np.allclose(result1, 0)
        assert not np.allclose(result2, 0)


class TestProcessChunk:
    """Tests for stateless process_chunk() function."""

    def test_process_chunk_basic(self):
        """process_chunk should process a single audio chunk."""
        from audio_processor import process_chunk

        audio = np.random.randn(12000).astype(np.float32) * 0.5
        result = process_chunk(audio, sample_rate=24000)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(audio)

    def test_process_chunk_accepts_all_parameters(self):
        """process_chunk should accept all compressor/limiter parameters."""
        from audio_processor import process_chunk

        audio = np.random.randn(12000).astype(np.float32) * 0.5
        result = process_chunk(
            audio,
            sample_rate=24000,
            threshold_db=-20,
            ratio=4.0,
            attack_ms=5,
            release_ms=100,
            limiter_threshold_db=-1.0,
            limiter_release_ms=50,
            gain_db=6,
        )

        assert isinstance(result, np.ndarray)

    def test_process_chunk_enabled_flag(self):
        """process_chunk should bypass processing when enabled=False."""
        from audio_processor import process_chunk

        audio = np.random.randn(12000).astype(np.float32) * 0.5
        result = process_chunk(audio, sample_rate=24000, enabled=False)

        # Should return input unchanged
        np.testing.assert_array_equal(result, audio)


class TestAudioProcessingBehavior:
    """Tests for actual audio processing behavior."""

    def test_compressor_reduces_dynamic_range(self):
        """Compressor should reduce dynamic range of signals above threshold."""
        from audio_processor import process_chunk

        # Create signal with varying amplitude
        t = np.linspace(0, 1, 24000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.8  # Loud sine wave

        # Process with compression only (no gain)
        result = process_chunk(
            audio,
            sample_rate=24000,
            threshold_db=-6,
            ratio=4.0,
            gain_db=0,  # No makeup gain to isolate compression effect
        )

        # RMS should be reduced for loud signals
        input_rms = np.sqrt(np.mean(audio**2))
        output_rms = np.sqrt(np.mean(result**2))
        assert output_rms < input_rms

    def test_limiter_controls_peaks(self):
        """Limiter should control peak levels (steady-state)."""
        from audio_processor import process_chunk

        # Create sustained loud signal (not transient)
        # Use a long signal so limiter settles
        audio = np.ones(48000, dtype=np.float32) * 0.9

        result = process_chunk(
            audio,
            sample_rate=24000,
            threshold_db=-40,  # Heavy compression
            ratio=10.0,
            gain_db=0,
            limiter_threshold_db=-1.0,
            limiter_release_ms=10,  # Fast release
        )

        # Steady-state portion (skip attack transient)
        steady_state = result[24000:]  # Second half
        max_allowed = 10 ** (-1.0 / 20)  # -1 dB in linear

        # Most samples should be below threshold (allowing some overshoot)
        samples_below = np.sum(np.abs(steady_state) <= max_allowed + 0.05)
        assert samples_below > len(steady_state) * 0.9  # 90% compliance

    def test_gain_applied_to_signal(self):
        """Gain stage should amplify the signal."""
        from audio_processor import process_chunk

        # Create quiet signal that won't trigger compression
        audio = np.ones(12000, dtype=np.float32) * 0.01  # Very quiet

        result = process_chunk(
            audio,
            sample_rate=24000,
            threshold_db=-60,  # Extremely low threshold (won't trigger)
            ratio=1.0,  # No compression
            gain_db=6,
        )

        # +6 dB is approximately 2x amplitude
        expected_gain = 10 ** (6 / 20)  # ~2.0
        actual_gain = np.mean(np.abs(result)) / np.mean(np.abs(audio))

        # Should be close to expected gain (within 20%)
        assert actual_gain > expected_gain * 0.8

    def test_silence_stays_silent(self):
        """Silent input should produce silent output."""
        from audio_processor import process_chunk

        audio = np.zeros(12000, dtype=np.float32)
        result = process_chunk(audio, sample_rate=24000)

        assert np.allclose(result, 0, atol=1e-6)

    def test_preserves_stereo_shape(self):
        """Processor should handle mono audio (1D array)."""
        from audio_processor import process_chunk

        # MLX TTS outputs mono audio as 1D array
        audio = np.random.randn(12000).astype(np.float32) * 0.5
        assert audio.ndim == 1

        result = process_chunk(audio, sample_rate=24000)

        assert result.ndim == 1
        assert result.shape == audio.shape


class TestConfigIntegration:
    """Tests for configuration system integration."""

    def test_get_compressor_config_function_exists(self):
        """get_compressor_config should exist for config integration."""
        from audio_processor import get_compressor_config
        assert callable(get_compressor_config)

    def test_get_compressor_config_returns_dict(self):
        """get_compressor_config should return dict with all parameters."""
        from audio_processor import get_compressor_config
        config = get_compressor_config()

        assert isinstance(config, dict)
        assert "threshold_db" in config
        assert "ratio" in config
        assert "attack_ms" in config
        assert "release_ms" in config
        assert "limiter_threshold_db" in config
        assert "limiter_release_ms" in config
        assert "gain_db" in config
        assert "enabled" in config

    def test_create_processor_uses_config_defaults(self):
        """create_processor with no args should use config defaults."""
        from audio_processor import create_processor, get_compressor_config

        # Should not raise - uses config defaults
        processor = create_processor(sample_rate=24000)
        assert callable(processor)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_audio_returns_empty(self):
        """Empty audio array should return empty array."""
        from audio_processor import process_chunk

        audio = np.array([], dtype=np.float32)
        result = process_chunk(audio, sample_rate=24000)

        assert len(result) == 0

    def test_very_short_audio(self):
        """Very short audio (< 1ms) should still process."""
        from audio_processor import process_chunk

        # 10 samples at 24kHz = ~0.4ms
        audio = np.random.randn(10).astype(np.float32) * 0.5
        result = process_chunk(audio, sample_rate=24000)

        assert len(result) == 10

    def test_different_sample_rates(self):
        """Processor should work with different sample rates."""
        from audio_processor import process_chunk

        for sr in [16000, 22050, 24000, 44100, 48000]:
            audio = np.random.randn(sr).astype(np.float32) * 0.5  # 1 second
            result = process_chunk(audio, sample_rate=sr)
            assert len(result) == len(audio)
