"""
Audio processor module - dynamic range compression and limiting for streaming TTS.

Uses Spotify's pedalboard library for efficient, stateful audio processing.
Designed for chunk-by-chunk streaming with state preservation across chunks.

Configuration is read from tts_config on each call, allowing runtime changes
without server restart.

Usage:
    # Stateful processing (for streaming - maintains compressor state)
    from audio_processor import create_processor
    processor = create_processor(sample_rate=24000)
    for chunk in audio_chunks:
        processed = processor(chunk)

    # Stateless processing (single chunks)
    from audio_processor import process_chunk
    processed = process_chunk(audio, sample_rate=24000)
"""
import numpy as np
from typing import Callable

# Fallback defaults if tts_config unavailable (notification_punch preset)
DEFAULT_INPUT_GAIN_DB = 0.0
DEFAULT_THRESHOLD_DB = -18
DEFAULT_RATIO = 3.0
DEFAULT_ATTACK_MS = 3
DEFAULT_RELEASE_MS = 50

DEFAULT_LIMITER_THRESHOLD_DB = -0.5
DEFAULT_LIMITER_RELEASE_MS = 40

DEFAULT_GAIN_DB = 8
DEFAULT_MASTER_GAIN_DB = 0.0

DEFAULT_ENABLED = True


def get_compressor_config() -> dict:
    """
    Get compressor configuration. Re-reads from tts_config on each call.

    Returns dict with all compressor/limiter parameters.
    """
    try:
        from tts_config import get_compressor_config as _get_compressor
        from tts_config import get_limiter_config as _get_limiter
        compressor = _get_compressor()
        limiter = _get_limiter()
        # Merge configs, mapping limiter keys to expected names
        return {
            **compressor,
            "limiter_threshold_db": limiter.get("threshold_db", DEFAULT_LIMITER_THRESHOLD_DB),
            "limiter_release_ms": limiter.get("release_ms", DEFAULT_LIMITER_RELEASE_MS),
            "limiter_enabled": limiter.get("enabled", True),
        }
    except ImportError:
        # Fallback to hardcoded defaults if tts_config unavailable
        return {
            "input_gain_db": DEFAULT_INPUT_GAIN_DB,
            "threshold_db": DEFAULT_THRESHOLD_DB,
            "ratio": DEFAULT_RATIO,
            "attack_ms": DEFAULT_ATTACK_MS,
            "release_ms": DEFAULT_RELEASE_MS,
            "limiter_threshold_db": DEFAULT_LIMITER_THRESHOLD_DB,
            "limiter_release_ms": DEFAULT_LIMITER_RELEASE_MS,
            "gain_db": DEFAULT_GAIN_DB,
            "master_gain_db": DEFAULT_MASTER_GAIN_DB,
            "enabled": DEFAULT_ENABLED,
        }


def create_processor(
    sample_rate: int,
    input_gain_db: float | None = None,
    threshold_db: float | None = None,
    ratio: float | None = None,
    attack_ms: float | None = None,
    release_ms: float | None = None,
    limiter_threshold_db: float | None = None,
    limiter_release_ms: float | None = None,
    gain_db: float | None = None,
    master_gain_db: float | None = None,
    compressor_enabled: bool | None = None,
    limiter_enabled: bool | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a stateful audio processor for streaming chunk processing.

    The returned callable maintains compressor/limiter state across calls,
    essential for smooth streaming audio without artifacts at chunk boundaries.

    Signal chain: Input Gain → Compressor → Makeup Gain → Limiter → Master Gain

    Args:
        sample_rate: Audio sample rate in Hz.
        input_gain_db: Input gain in dB before compressor (default 0).
        threshold_db: Compressor threshold in dB (default -18).
        ratio: Compression ratio (default 3.0).
        attack_ms: Compressor attack time in ms (default 3).
        release_ms: Compressor release time in ms (default 50).
        limiter_threshold_db: Limiter threshold in dB (default -0.5).
        limiter_release_ms: Limiter release time in ms (default 40).
        gain_db: Makeup gain in dB (default 8).
        master_gain_db: Master gain in dB after limiter (default 0).
        compressor_enabled: Whether compressor is enabled (reads from config if None).
        limiter_enabled: Whether limiter is enabled (reads from config if None).

    Returns:
        Callable that processes audio chunks while maintaining state.
    """
    from pedalboard import Pedalboard, Compressor, Limiter, Gain

    # Use defaults for any None values
    config = get_compressor_config()
    input_gain_db = input_gain_db if input_gain_db is not None else config.get("input_gain_db", DEFAULT_INPUT_GAIN_DB)
    threshold_db = threshold_db if threshold_db is not None else config["threshold_db"]
    ratio = ratio if ratio is not None else config["ratio"]
    attack_ms = attack_ms if attack_ms is not None else config["attack_ms"]
    release_ms = release_ms if release_ms is not None else config["release_ms"]
    limiter_threshold_db = limiter_threshold_db if limiter_threshold_db is not None else config["limiter_threshold_db"]
    limiter_release_ms = limiter_release_ms if limiter_release_ms is not None else config["limiter_release_ms"]
    gain_db = gain_db if gain_db is not None else config["gain_db"]
    master_gain_db = master_gain_db if master_gain_db is not None else config.get("master_gain_db", DEFAULT_MASTER_GAIN_DB)

    # Read enabled flags from config if not explicitly provided
    comp_enabled = compressor_enabled if compressor_enabled is not None else config.get("enabled", DEFAULT_ENABLED)
    lim_enabled = limiter_enabled if limiter_enabled is not None else config.get("limiter_enabled", True)

    # Build processing chain based on what's enabled
    # Signal chain: Input Gain → [Compressor → Makeup Gain] → [Limiter] → Master Gain
    chain = [Gain(gain_db=input_gain_db)]

    if comp_enabled:
        chain.append(Compressor(
            threshold_db=threshold_db,
            ratio=ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
        ))
        chain.append(Gain(gain_db=gain_db))

    if lim_enabled:
        chain.append(Limiter(
            threshold_db=limiter_threshold_db,
            release_ms=limiter_release_ms,
        ))

    chain.append(Gain(gain_db=master_gain_db))

    board = Pedalboard(chain)

    def processor(audio: np.ndarray) -> np.ndarray:
        """Process an audio chunk, maintaining state across calls."""
        # Skip processing entirely if both are disabled
        if not comp_enabled and not lim_enabled:
            return audio

        if len(audio) == 0:
            return audio

        # Handle MLX arrays - convert to numpy first
        if hasattr(audio, '__module__') and 'mlx' in audio.__module__:
            audio = np.array(audio)

        # Ensure float32 for pedalboard
        audio_f32 = audio.astype(np.float32) if audio.dtype != np.float32 else audio

        # pedalboard expects shape (channels, samples) or (samples,) for mono
        # MLX TTS outputs mono as (samples,), need to reshape for pedalboard
        was_1d = audio_f32.ndim == 1
        if was_1d:
            audio_f32 = audio_f32.reshape(1, -1)

        # Process with state preservation (reset=False)
        processed = board(audio_f32, sample_rate, reset=False)

        # Restore original shape
        if was_1d:
            processed = processed.flatten()

        return processed.astype(np.float32)

    return processor


def process_chunk(
    audio: np.ndarray,
    sample_rate: int,
    input_gain_db: float | None = None,
    threshold_db: float | None = None,
    ratio: float | None = None,
    attack_ms: float | None = None,
    release_ms: float | None = None,
    limiter_threshold_db: float | None = None,
    limiter_release_ms: float | None = None,
    gain_db: float | None = None,
    master_gain_db: float | None = None,
    compressor_enabled: bool | None = None,
    limiter_enabled: bool | None = None,
) -> np.ndarray:
    """
    Process a single audio chunk (stateless).

    For streaming, use create_processor() instead to maintain state.
    This function creates a new processor for each call.

    Signal chain: Input Gain → Compressor → Makeup Gain → Limiter → Master Gain

    Args:
        audio: Audio samples as numpy array (float32).
        sample_rate: Audio sample rate in Hz.
        input_gain_db: Input gain in dB before compressor (default 0).
        threshold_db: Compressor threshold in dB (default -18).
        ratio: Compression ratio (default 3.0).
        attack_ms: Compressor attack time in ms (default 3).
        release_ms: Compressor release time in ms (default 50).
        limiter_threshold_db: Limiter threshold in dB (default -0.5).
        limiter_release_ms: Limiter release time in ms (default 40).
        gain_db: Makeup gain in dB (default 8).
        master_gain_db: Master gain in dB after limiter (default 0).
        compressor_enabled: Whether compressor is enabled (reads from config if None).
        limiter_enabled: Whether limiter is enabled (reads from config if None).

    Returns:
        Processed audio as numpy float32 array.
    """
    if len(audio) == 0:
        return audio

    # Create stateless processor and process
    processor = create_processor(
        sample_rate=sample_rate,
        input_gain_db=input_gain_db,
        threshold_db=threshold_db,
        ratio=ratio,
        attack_ms=attack_ms,
        release_ms=release_ms,
        limiter_threshold_db=limiter_threshold_db,
        limiter_release_ms=limiter_release_ms,
        gain_db=gain_db,
        master_gain_db=master_gain_db,
        compressor_enabled=compressor_enabled,
        limiter_enabled=limiter_enabled,
    )

    return processor(audio)
