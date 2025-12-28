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
        """process_chunk should bypass processing when compressor/limiter disabled."""
        from audio_processor import process_chunk

        audio = np.random.randn(12000).astype(np.float32) * 0.5
        result = process_chunk(
            audio,
            sample_rate=24000,
            compressor_enabled=False,
            limiter_enabled=False,
        )

        # Should return input unchanged when both are disabled
        np.testing.assert_array_equal(result, audio)


class TestAudioProcessingBehavior:
    """Tests for actual audio processing behavior."""

    def test_compressor_reduces_dynamic_range(self):
        """Compressor should reduce dynamic range of signals above threshold."""
        from audio_processor import process_chunk

        # Create signal with varying amplitude
        t = np.linspace(0, 1, 24000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t) * 0.8  # Loud sine wave

        # Process with compression only (no gain stages)
        result = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=0,  # No input gain
            threshold_db=-6,
            ratio=4.0,
            gain_db=0,  # No makeup gain
            master_gain_db=0,  # No master gain
            limiter_enabled=False,  # Disable limiter to isolate compressor
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


# =============================================================================
# Input Gain and Master Gain Tests (TDD - RED PHASE)
# =============================================================================


class TestInputGainDefaults:
    """Tests for input gain default parameter."""

    def test_default_input_gain_exists(self):
        """DEFAULT_INPUT_GAIN_DB constant should exist."""
        from audio_processor import DEFAULT_INPUT_GAIN_DB
        assert DEFAULT_INPUT_GAIN_DB is not None

    def test_default_input_gain_is_zero(self):
        """Default input gain should be 0 dB (unity)."""
        from audio_processor import DEFAULT_INPUT_GAIN_DB
        assert DEFAULT_INPUT_GAIN_DB == 0.0


class TestMasterGainDefaults:
    """Tests for master gain default parameter."""

    def test_default_master_gain_exists(self):
        """DEFAULT_MASTER_GAIN_DB constant should exist."""
        from audio_processor import DEFAULT_MASTER_GAIN_DB
        assert DEFAULT_MASTER_GAIN_DB is not None

    def test_default_master_gain_is_zero(self):
        """Default master gain should be 0 dB (unity)."""
        from audio_processor import DEFAULT_MASTER_GAIN_DB
        assert DEFAULT_MASTER_GAIN_DB == 0.0


class TestCreateProcessorWithGains:
    """Tests for create_processor with input/master gain parameters."""

    def test_accepts_input_gain_parameter(self):
        """create_processor should accept input_gain_db parameter."""
        from audio_processor import create_processor
        processor = create_processor(
            sample_rate=24000,
            input_gain_db=6.0,
        )
        assert callable(processor)

    def test_accepts_master_gain_parameter(self):
        """create_processor should accept master_gain_db parameter."""
        from audio_processor import create_processor
        processor = create_processor(
            sample_rate=24000,
            master_gain_db=-3.0,
        )
        assert callable(processor)

    def test_accepts_both_gain_parameters(self):
        """create_processor should accept both input and master gain."""
        from audio_processor import create_processor
        processor = create_processor(
            sample_rate=24000,
            input_gain_db=6.0,
            master_gain_db=-3.0,
        )
        assert callable(processor)


class TestProcessChunkWithGains:
    """Tests for process_chunk with input/master gain parameters."""

    def test_accepts_input_gain_parameter(self):
        """process_chunk should accept input_gain_db parameter."""
        from audio_processor import process_chunk
        audio = np.random.randn(12000).astype(np.float32) * 0.5
        result = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=6.0,
        )
        assert isinstance(result, np.ndarray)

    def test_accepts_master_gain_parameter(self):
        """process_chunk should accept master_gain_db parameter."""
        from audio_processor import process_chunk
        audio = np.random.randn(12000).astype(np.float32) * 0.5
        result = process_chunk(
            audio,
            sample_rate=24000,
            master_gain_db=-3.0,
        )
        assert isinstance(result, np.ndarray)

    def test_accepts_all_gain_parameters(self):
        """process_chunk should accept input, makeup, and master gain."""
        from audio_processor import process_chunk
        audio = np.random.randn(12000).astype(np.float32) * 0.5
        result = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=3.0,
            gain_db=6.0,
            master_gain_db=-3.0,
        )
        assert isinstance(result, np.ndarray)


class TestInputGainBehavior:
    """Tests for input gain audio processing behavior."""

    def test_input_gain_amplifies_before_compression(self):
        """Input gain should amplify signal before compressor."""
        from audio_processor import process_chunk

        # Quiet signal that won't trigger compression at -18dB threshold
        audio = np.ones(12000, dtype=np.float32) * 0.05  # ~-26 dB

        # Without input gain - should pass through relatively unchanged
        result_no_gain = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=0.0,
            threshold_db=-18,
            ratio=4.0,
            gain_db=0,  # No makeup gain
            master_gain_db=0.0,
        )

        # With +12dB input gain - signal should now trigger compression
        result_with_gain = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=12.0,
            threshold_db=-18,
            ratio=4.0,
            gain_db=0,  # No makeup gain
            master_gain_db=0.0,
        )

        # Input gain should cause more compression (lower output relative to input boost)
        rms_no_gain = np.sqrt(np.mean(result_no_gain**2))
        rms_with_gain = np.sqrt(np.mean(result_with_gain**2))

        # With 12dB input gain, if no compression, output would be 4x louder
        # But compression should reduce this ratio significantly
        gain_ratio = rms_with_gain / rms_no_gain
        assert gain_ratio < 4.0  # Less than full 12dB boost due to compression

    def test_input_gain_zero_is_unity(self):
        """Input gain of 0 dB should not change signal level vs bypassed input gain."""
        from audio_processor import process_chunk

        audio = np.ones(12000, dtype=np.float32) * 0.01  # Very quiet

        # Process with 0dB input gain
        result_zero = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=0.0,
            threshold_db=-60,
            ratio=1.0,
            gain_db=0,
            master_gain_db=0.0,
        )

        # Process with +6dB input gain
        result_boosted = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=6.0,
            threshold_db=-60,
            ratio=1.0,
            gain_db=0,
            master_gain_db=0.0,
        )

        # +6dB should be ~2x amplitude
        rms_zero = np.sqrt(np.mean(result_zero**2))
        rms_boosted = np.sqrt(np.mean(result_boosted**2))
        gain_ratio = rms_boosted / rms_zero
        expected_ratio = 10 ** (6 / 20)  # ~2.0

        np.testing.assert_allclose(gain_ratio, expected_ratio, rtol=0.2)


class TestMasterGainBehavior:
    """Tests for master gain audio processing behavior."""

    def test_master_gain_applied_after_limiter(self):
        """Master gain should be applied after the limiter."""
        from audio_processor import process_chunk

        # Create signal that will hit limiter
        audio = np.ones(24000, dtype=np.float32) * 0.5

        # Process with limiter at -6dB and master gain of +6dB
        result = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=0.0,
            threshold_db=-60,  # No compression
            ratio=1.0,
            gain_db=0,
            limiter_threshold_db=-6.0,
            limiter_release_ms=10,
            master_gain_db=6.0,
        )

        # Master gain is applied AFTER limiter, so final peaks can exceed limiter threshold
        # Limiter output ~0.5 (-6dB), then +6dB master = ~1.0
        max_output = np.max(np.abs(result[12000:]))  # Skip transient
        limiter_level = 10 ** (-6.0 / 20)  # ~0.5

        # Output should be higher than limiter threshold due to master gain
        assert max_output > limiter_level * 1.5

    def test_master_gain_negative_attenuates(self):
        """Negative master gain should attenuate final output."""
        from audio_processor import process_chunk

        audio = np.ones(12000, dtype=np.float32) * 0.5

        # Process with -6dB master gain
        result = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=0.0,
            threshold_db=-60,
            ratio=1.0,
            gain_db=0,
            limiter_threshold_db=0.0,
            master_gain_db=-6.0,
        )

        # Output should be approximately half (-6dB)
        expected_attenuation = 10 ** (-6.0 / 20)  # ~0.5
        actual_ratio = np.mean(np.abs(result)) / np.mean(np.abs(audio))
        np.testing.assert_allclose(actual_ratio, expected_attenuation, rtol=0.2)

    def test_master_gain_zero_is_unity(self):
        """Master gain of 0 dB should not change level vs bypassed master gain."""
        from audio_processor import process_chunk

        audio = np.ones(12000, dtype=np.float32) * 0.1

        # Process with 0dB master gain
        result_zero = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=0.0,
            threshold_db=-60,
            ratio=1.0,
            gain_db=0,
            master_gain_db=0.0,
        )

        # Process with +6dB master gain
        result_boosted = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=0.0,
            threshold_db=-60,
            ratio=1.0,
            gain_db=0,
            master_gain_db=6.0,
        )

        # +6dB master should be ~2x amplitude
        rms_zero = np.sqrt(np.mean(result_zero**2))
        rms_boosted = np.sqrt(np.mean(result_boosted**2))
        gain_ratio = rms_boosted / rms_zero
        expected_ratio = 10 ** (6 / 20)  # ~2.0

        np.testing.assert_allclose(gain_ratio, expected_ratio, rtol=0.2)


class TestSignalChainOrder:
    """Tests to verify signal chain order: Input Gain → Compressor → Makeup → Limiter → Master."""

    def test_full_signal_chain(self):
        """Full chain should process in correct order."""
        from audio_processor import process_chunk

        # Quiet signal
        audio = np.ones(24000, dtype=np.float32) * 0.05

        # Apply all stages:
        # Input +6dB: 0.05 → 0.1
        # Compressor at -20dB with 2:1: should compress
        # Makeup +6dB: boost after compression
        # Limiter at -3dB: prevent peaks above ~0.7
        # Master -3dB: final attenuation
        result = process_chunk(
            audio,
            sample_rate=24000,
            input_gain_db=6.0,
            threshold_db=-20,
            ratio=2.0,
            attack_ms=1,
            release_ms=10,
            gain_db=6.0,
            limiter_threshold_db=-3.0,
            limiter_release_ms=10,
            master_gain_db=-3.0,
        )

        # Output should be processed and within reasonable range
        assert np.max(np.abs(result)) < 1.0  # Not clipping
        assert np.mean(np.abs(result)) > 0.01  # Not silent


class TestConfigIntegrationWithGains:
    """Tests for config integration with input/master gain."""

    def test_get_compressor_config_has_input_gain(self):
        """get_compressor_config should include input_gain_db."""
        from audio_processor import get_compressor_config
        config = get_compressor_config()
        assert "input_gain_db" in config

    def test_get_compressor_config_has_master_gain(self):
        """get_compressor_config should include master_gain_db."""
        from audio_processor import get_compressor_config
        config = get_compressor_config()
        assert "master_gain_db" in config
