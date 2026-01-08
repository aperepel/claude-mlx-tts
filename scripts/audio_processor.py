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
import os
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

# OLA (Overlap-Add) defaults
DEFAULT_CROSSFADE_MS = 20.0  # Crossfade duration in milliseconds


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
    from pedalboard import Pedalboard, Compressor, Limiter, Gain, Plugin  # pyright: ignore[reportPrivateImportUsage]

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
    chain: list[Plugin] = [Gain(gain_db=input_gain_db)]

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

    # State for smooth chunk stitching
    # The model's streaming output has discontinuities at chunk boundaries
    # We fix this by interpolating from the previous chunk's last sample
    # Larger window = gentler correction = less audible artifacts
    # Configurable via TTS_INTERP_MS env var for experimentation
    INTERP_MS = float(os.environ.get("TTS_INTERP_MS", "20.0"))
    interp_samples = max(4, int(sample_rate * INTERP_MS / 1000))
    state = {"prev_last": None}

    # Pre-compute the correction curve (raised cosine for smooth transition)
    interp_curve = np.cos(np.linspace(0, np.pi / 2, interp_samples, dtype=np.float32)) ** 2

    def processor(audio: np.ndarray) -> np.ndarray:
        """Process an audio chunk, maintaining state across calls.

        Fixes discontinuities from the model's streaming output by
        interpolating the first few samples to connect smoothly with
        the previous chunk.
        """
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

        # Fix chunk boundary discontinuities BEFORE compression
        # The model's streaming output has discontinuities that we smooth here
        if state["prev_last"] is not None and len(audio_f32) >= interp_samples:
            disc = audio_f32[0] - state["prev_last"]
            audio_f32 = audio_f32.copy()  # Don't modify input
            audio_f32[:interp_samples] -= disc * interp_curve

        # Store last sample of INPUT for next chunk's interpolation
        if len(audio_f32) > 0:
            state["prev_last"] = float(audio_f32[-1])  # pyright: ignore[reportArgumentType]

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


def create_ola_processor(
    sample_rate: int,
    crossfade_ms: float | None = None,
) -> Callable[[np.ndarray | None], np.ndarray]:
    """
    Create an Overlap-Add (OLA) processor for seamless audio chunk stitching.

    This processor eliminates clicks and discontinuities at chunk boundaries
    by applying Hann-windowed crossfades between consecutive chunks.

    Signal flow:
    1. First chunk: output all but crossfade_samples, save windowed tail
    2. Subsequent chunks: crossfade head with saved tail, output, save new tail
    3. Flush (None input): output remaining windowed tail

    The Hann window ensures that fade_out + fade_in = 1.0 at all points,
    preserving signal amplitude while smoothing discontinuities.

    Args:
        sample_rate: Audio sample rate in Hz.
        crossfade_ms: Crossfade duration in milliseconds (default 20ms).

    Returns:
        Callable that processes audio chunks with OLA. Pass None to flush
        the remaining buffered samples.
    """
    cf_ms = crossfade_ms if crossfade_ms is not None else DEFAULT_CROSSFADE_MS
    crossfade_samples = max(1, int(sample_rate * cf_ms / 1000))

    # Create Hann windows for crossfade (these sum to 1.0 at all points)
    # fade_out: 1.0 -> 0.0 (cosine squared curve)
    # fade_in: 0.0 -> 1.0 (sine squared curve)
    t = np.linspace(0, np.pi / 2, crossfade_samples, dtype=np.float32)
    fade_out = np.cos(t) ** 2
    fade_in = np.sin(t) ** 2

    # State: buffered tail from previous chunk (already faded out)
    state = {"tail": None, "tail_unfaded": None}

    def processor(audio: np.ndarray | None) -> np.ndarray:
        """Process an audio chunk with OLA, or flush if None."""
        # Flush: return remaining buffer with fade-out to silence
        if audio is None:
            if state["tail"] is not None:
                # Return the faded tail - it already has fade-out applied
                # This ensures audio ends smoothly at silence
                result = state["tail"].copy()
                state["tail"] = None
                state["tail_unfaded"] = None
                return result
            return np.array([], dtype=np.float32)

        # Handle empty input
        if len(audio) == 0:
            return np.array([], dtype=np.float32)

        # Handle MLX arrays
        if hasattr(audio, '__module__') and 'mlx' in str(getattr(audio, '__module__', '')):
            audio = np.array(audio)

        # Ensure float32
        audio_f32 = audio.astype(np.float32) if audio.dtype != np.float32 else audio

        # Handle chunks smaller than crossfade region
        if len(audio_f32) < crossfade_samples:
            # Buffer small chunks until we have enough
            if state["tail_unfaded"] is not None:
                # Append to existing buffer
                combined = np.concatenate([state["tail_unfaded"], audio_f32])
                if len(combined) < crossfade_samples:
                    state["tail_unfaded"] = combined  # pyright: ignore[reportArgumentType]
                    state["tail"] = None  # Will recompute when we have enough
                    return np.array([], dtype=np.float32)
                else:
                    # Now have enough - treat combined as the new chunk
                    audio_f32 = combined
                    state["tail"] = None
                    state["tail_unfaded"] = None
            else:
                # First small chunk - just buffer it
                state["tail_unfaded"] = audio_f32.copy()  # pyright: ignore[reportArgumentType]
                return np.array([], dtype=np.float32)

        # Split current chunk into head, middle, and tail
        head = audio_f32[:crossfade_samples]
        if len(audio_f32) > crossfade_samples:
            middle = audio_f32[crossfade_samples:-crossfade_samples] if len(audio_f32) > 2 * crossfade_samples else np.array([], dtype=np.float32)
            tail = audio_f32[-crossfade_samples:]
        else:
            middle = np.array([], dtype=np.float32)
            tail = head  # Chunk is exactly crossfade_samples

        # Apply fade-in to head
        faded_head = head * fade_in

        # Apply fade-out to tail and save for next chunk
        faded_tail = tail * fade_out
        tail_unfaded = tail.copy()

        if state["tail"] is None:
            # First chunk: apply fade-in to head (from silence), output head + middle
            # This prevents clicks at the very start of audio
            if len(audio_f32) > crossfade_samples:
                output = np.concatenate([faded_head, middle])
            else:
                # Chunk is exactly crossfade_samples - just fade in
                output = faded_head
            state["tail"] = faded_tail  # pyright: ignore[reportArgumentType]
            state["tail_unfaded"] = tail_unfaded  # pyright: ignore[reportArgumentType]
            return output.astype(np.float32)

        # Crossfade: previous faded tail + current faded head
        crossfade = state["tail"] + faded_head

        # Output: crossfade region + middle
        if len(middle) > 0:
            output = np.concatenate([crossfade, middle])
        else:
            output = crossfade

        # Save new tail for next chunk
        state["tail"] = faded_tail  # pyright: ignore[reportArgumentType]
        state["tail_unfaded"] = tail_unfaded  # pyright: ignore[reportArgumentType]

        return output.astype(np.float32)

    return processor


def create_processor_with_ola(
    sample_rate: int,
    crossfade_ms: float | None = None,
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
) -> Callable[[np.ndarray | None], np.ndarray]:
    """
    Create a combined OLA + compressor/limiter processor.

    Signal chain: OLA Crossfade → Input Gain → Compressor → Makeup Gain → Limiter → Master Gain

    The OLA stage smooths chunk boundaries before compression, which prevents
    the compressor from reacting to artificial discontinuities.

    Args:
        sample_rate: Audio sample rate in Hz.
        crossfade_ms: OLA crossfade duration in milliseconds (default 20ms).
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
        Callable that processes audio chunks with OLA + dynamics processing.
        Pass None to flush the OLA buffer.
    """
    # Create OLA processor
    ola = create_ola_processor(sample_rate=sample_rate, crossfade_ms=crossfade_ms)

    # Create compressor/limiter processor
    dynamics = create_processor(
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

    def combined_processor(audio: np.ndarray | None) -> np.ndarray:
        """Process audio with OLA then dynamics."""
        # OLA stage (handles None for flush)
        smoothed = ola(audio)

        # Dynamics stage (skip if empty)
        if len(smoothed) == 0:
            return smoothed

        return dynamics(smoothed)

    return combined_processor
