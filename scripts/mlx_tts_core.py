"""
MLX TTS Core - Direct Python API wrapper for mlx_audio TTS.

This module provides a clean interface for:
- Model loading with caching
- Speech generation with configurable options
- Error handling and fallback
- Streaming metrics (TTFT, generation time, playback time, RTF)

Usage:
    from mlx_tts_core import speak_mlx, get_model

    # Simple usage (auto-loads and caches model)
    speak_mlx("Hello world")

    # Advanced usage with model reuse
    model = get_model()
    generate_speech("First message", model=model)
    generate_speech("Second message", model=model)  # No reload

    # Get metrics
    metrics = generate_speech("Test", model=model, return_metrics=True)
    print(f"TTFT: {metrics['ttft']:.2f}s")
"""
import os
import time
import logging
from typing import Any

# Import AudioPlayer for streaming playback
try:
    from mlx_audio.tts.audio_player import AudioPlayer
except ImportError:
    AudioPlayer = None

log = logging.getLogger(__name__)

# Configuration - can be overridden by environment variables
MLX_MODEL = os.environ.get("MLX_TTS_MODEL", "mlx-community/chatterbox-turbo-fp16")

# Default streaming interval fallback
DEFAULT_STREAMING_INTERVAL = 0.5


def _get_configured_streaming_interval() -> float:
    """Get streaming interval from config, with fallback to default."""
    try:
        from tts_config import get_streaming_interval
        return get_streaming_interval()
    except ImportError:
        return DEFAULT_STREAMING_INTERVAL

# Module-level model cache
_cached_model = None


def _clear_model_cache() -> None:
    """Clear the cached model (for testing)."""
    global _cached_model
    _cached_model = None


def load_tts_model(model_path: str | None = None) -> Any:
    """
    Load the TTS model from HuggingFace.

    Args:
        model_path: HuggingFace model ID or local path. Defaults to MLX_MODEL.

    Returns:
        Loaded model object with sample_rate attribute.

    Raises:
        ImportError: If mlx_audio is not installed.
        RuntimeError: If model loading fails.
    """
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError as e:
        log.error("mlx_audio not installed. Install with: uv sync --extra mlx")
        raise ImportError("mlx_audio is required for MLX TTS") from e

    path = model_path or MLX_MODEL
    log.info(f"Loading TTS model: {path}")

    try:
        model = load_model(model_path=path)
        log.info(f"Model loaded successfully (sample_rate={model.sample_rate})")
        return model
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load TTS model: {e}") from e


def get_model(model_path: str | None = None) -> Any:
    """
    Get the cached TTS model, loading it if necessary.

    This function implements lazy loading with caching - the model
    is loaded on first call and reused for subsequent calls.

    Args:
        model_path: HuggingFace model ID. Only used on first load.

    Returns:
        Cached model object.
    """
    global _cached_model
    if _cached_model is None:
        _cached_model = load_tts_model(model_path)
    return _cached_model


def generate_speech(
    text: str,
    model: Any | None = None,
    ref_audio: str | None = None,
    ref_text: str = ".",
    play: bool = True,
    save_path: str | None = None,
    verbose: bool = False,
    stream: bool = True,
    streaming_interval: float | None = None,
    return_metrics: bool = False,
    voice_name: str | None = None,
) -> dict[str, float] | None:
    """
    Generate speech from text using MLX TTS.

    Note: Speed parameter is not supported by Chatterbox model.
    The mlx_audio library explicitly ignores speed for Chatterbox.

    Args:
        text: Text to convert to speech.
        model: Pre-loaded model (uses cached model if None).
        ref_audio: Path to voice reference WAV file (legacy, use voice_name instead).
        ref_text: Transcript of reference audio.
        play: Whether to play audio immediately.
        save_path: Optional path to save audio file.
        verbose: Enable verbose logging from mlx_audio.
        stream: Enable streaming for reduced time-to-first-audio.
        streaming_interval: Chunk interval in seconds (default 0.5s from config).
        return_metrics: If True, return dict with timing metrics.
        voice_name: Voice name (without extension). Uses format-agnostic loading.
                    If None, uses active voice from config.

    Returns:
        If return_metrics=True, returns dict with:
            - ttft: Time to first audio chunk (seconds)
            - gen_time: Total generation time (seconds)
            - play_time: Time waiting for playback drain (seconds)
            - chunks: Number of streaming chunks
            - rtf: Real-time factor (gen_time / audio_duration)
            - audio_duration: Total audio duration (seconds)
        If return_metrics=False (default), returns None.
    """
    if not text or not text.strip():
        log.warning("Empty text provided, skipping generation")
        return None

    # Use provided model or get cached
    if model is None:
        model = get_model()

    # Determine voice source: voice_name takes priority over ref_audio
    actual_voice_name = voice_name
    actual_ref_audio = ref_audio

    if actual_voice_name is None and actual_ref_audio is None:
        # Neither specified - use active voice from config
        from tts_config import get_active_voice
        actual_voice_name = get_active_voice()
        log.debug(f"Using active voice from config: {actual_voice_name}")

    actual_streaming_interval = streaming_interval if streaming_interval is not None else _get_configured_streaming_interval()

    # Warn about save_path incompatibility with streaming
    if stream and save_path:
        log.warning("save_path is ignored in streaming mode - audio plays but cannot be saved")

    log.debug(f"Generating speech: '{text[:50]}...' (play={play}, stream={stream})")

    # Suppress tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        if stream:
            # Use direct model.generate() + AudioPlayer for streaming with metrics
            metrics = _generate_streaming_with_metrics(
                text=text,
                model=model,
                ref_text=ref_text,
                play=play,
                streaming_interval=actual_streaming_interval,
                voice_name=actual_voice_name,
                ref_audio=actual_ref_audio,
            )
        else:
            # Non-streaming mode: use generate_audio for file saving support
            metrics = _generate_non_streaming_with_metrics(
                text=text,
                model=model,
                ref_text=ref_text,
                play=play,
                save_path=save_path,
                verbose=verbose,
                voice_name=actual_voice_name,
                ref_audio=actual_ref_audio,
            )

        # Log metrics
        log.info(
            f"TTS: ttft={metrics['ttft']:.2f}s gen={metrics['gen_time']:.2f}s "
            f"play={metrics['play_time']:.2f}s chunks={metrics['chunks']} "
            f"RTF={metrics['rtf']:.2f}"
        )

        log.debug("Speech generation completed")
        return metrics if return_metrics else None

    except Exception as e:
        log.error(f"Speech generation failed: {e}")
        raise


def _generate_streaming_with_metrics(
    text: str,
    model: Any,
    ref_text: str,
    play: bool,
    streaming_interval: float,
    voice_name: str | None = None,
    ref_audio: str | None = None,
) -> dict[str, float]:
    """
    Generate speech with streaming and capture timing metrics.

    Uses direct model.generate() + AudioPlayer for streaming playback
    while capturing TTFT and other metrics. Audio is processed through
    compressor/limiter for consistent volume levels.

    Voice loading:
    - If voice_name is provided, use get_voice_conditionals() (format-agnostic)
    - If ref_audio is provided (for backwards compat), load from file

    Note: Speed parameter is not passed to model.generate() as
    Chatterbox does not support speed adjustment.
    """
    # Determine voice loading strategy
    ref_audio_data = None

    if voice_name is not None:
        # Format-agnostic voice loading by name
        from voice_cache import get_voice_conditionals
        model._conds = get_voice_conditionals(model, voice_name)
        log.debug(f"Loaded voice conditionals for: {voice_name}")
    elif ref_audio and os.path.exists(ref_audio):
        # Load from ref_audio file path
        from mlx_audio.tts.generate import load_audio
        ref_audio_data = load_audio(ref_audio, sample_rate=model.sample_rate)

    # Initialize player if needed
    player = AudioPlayer(sample_rate=model.sample_rate) if play else None

    # Initialize audio processor for compression/limiting
    # Creates stateful processor that maintains compressor envelope across chunks
    audio_processor = None
    try:
        from audio_processor import create_processor
        audio_processor = create_processor(sample_rate=model.sample_rate)
        log.debug("Audio processor initialized for streaming")
    except ImportError:
        log.debug("audio_processor not available, skipping compression")

    # Metrics tracking
    gen_start = time.perf_counter()
    ttft = None
    chunk_count = 0
    total_samples = 0
    last_rtf = 0.0

    # Stream generation (speed not supported by Chatterbox)
    for result in model.generate(
        text,
        ref_audio=ref_audio_data,
        stream=True,
        streaming_interval=streaming_interval,
    ):
        if ttft is None:
            ttft = time.perf_counter() - gen_start

        chunk_count += 1
        total_samples += len(result.audio)
        # Use real_time_factor if available (varies by model)
        if hasattr(result, 'real_time_factor') and result.real_time_factor:
            last_rtf = result.real_time_factor

        if player:
            # Apply compression/limiting before playback
            audio_chunk = result.audio
            if audio_processor is not None:
                audio_chunk = audio_processor(audio_chunk)
            player.queue_audio(audio_chunk)

    gen_time = time.perf_counter() - gen_start

    # Wait for playback to finish
    play_time = 0.0
    if player:
        # Handle short audio that didn't meet the min buffer threshold:
        # If stream hasn't started but we have audio, force start it
        if not player.playing and player.buffered_samples() > 0:
            log.debug("Force-starting stream for short audio clip")
            player.start_stream()

        # Only wait for drain if stream is actually playing
        if player.playing:
            play_start = time.perf_counter()
            player.wait_for_drain()
            play_time = time.perf_counter() - play_start
        elif player.buffered_samples() == 0:
            log.debug("No audio to play, skipping drain wait")

    # Calculate audio duration
    audio_duration = total_samples / model.sample_rate if model.sample_rate > 0 else 0.0

    # Calculate RTF: compute from gen_time/audio_duration, or use last chunk's value
    rtf = gen_time / audio_duration if audio_duration > 0 else (last_rtf if last_rtf > 0 else 0.0)

    return {
        "ttft": ttft or 0.0,
        "gen_time": gen_time,
        "play_time": play_time,
        "chunks": chunk_count,
        "rtf": rtf,
        "audio_duration": audio_duration,
    }


def _generate_non_streaming_with_metrics(
    text: str,
    model: Any,
    ref_text: str,
    play: bool,
    save_path: str | None,
    verbose: bool,
    voice_name: str | None = None,
    ref_audio: str | None = None,
) -> dict[str, float]:
    """
    Generate speech without streaming and capture timing metrics.

    Uses generate_audio for backwards compatibility with file saving.

    Voice loading:
    - If voice_name is provided, pre-loads conditionals using format-agnostic loader
    - If ref_audio is provided (for backwards compat), passes to generate_audio

    Note: Speed parameter is not passed as Chatterbox does not support it.
    """
    from mlx_audio.tts.generate import generate_audio

    # Pre-load voice conditionals if voice_name is provided
    actual_ref_audio = ref_audio
    if voice_name is not None:
        from voice_cache import get_voice_conditionals
        from tts_config import resolve_voice_path
        model._conds = get_voice_conditionals(model, voice_name)
        # For generate_audio, we still need a ref_audio path, but it won't be used
        # since _conds is already set. Try to get the wav path for consistency.
        try:
            voice_path = resolve_voice_path(voice_name)
            if voice_path.suffix == ".wav":
                actual_ref_audio = str(voice_path)
        except ValueError:
            pass  # Use whatever ref_audio was provided
        log.debug(f"Pre-loaded voice conditionals for: {voice_name}")

    # Determine file_prefix from save_path
    file_prefix = None
    if save_path:
        file_prefix = save_path.rsplit(".", 1)[0] if "." in save_path else save_path

    gen_start = time.perf_counter()

    gen_kwargs = {
        "text": text,
        "model": model,
        "ref_audio": actual_ref_audio,
        "ref_text": ref_text,
        "play": play,
        "verbose": verbose,
        "stream": False,
        # Note: speed not included - Chatterbox doesn't support it
    }
    if file_prefix:
        gen_kwargs["file_prefix"] = file_prefix

    generate_audio(**gen_kwargs)

    gen_time = time.perf_counter() - gen_start

    # Rename output file if needed
    if save_path and file_prefix:
        actual_output = f"{file_prefix}_000.wav"
        if os.path.exists(actual_output) and actual_output != save_path:
            import shutil
            shutil.move(actual_output, save_path)
            log.debug(f"Renamed {actual_output} -> {save_path}")

    # For non-streaming, TTFT equals gen_time (all-or-nothing)
    # We don't have audio duration info without parsing the output file
    return {
        "ttft": gen_time,
        "gen_time": gen_time,
        "play_time": 0.0,  # Playback happens during generate_audio
        "chunks": 1,
        "rtf": 0.0,  # Can't compute without audio duration
        "audio_duration": 0.0,  # Would need to parse wav file
    }


def speak_mlx(
    message: str,
    ref_audio: str | None = None,
    stream: bool = True,
    voice_name: str | None = None,
) -> None:
    """
    High-level function to speak a message using MLX TTS.

    This is the main entry point for simple TTS playback.
    Automatically uses cached model for fast subsequent calls.
    Streaming is enabled by default for lowest latency.

    Note: Speed parameter is not supported by Chatterbox model.

    Args:
        message: Text to speak.
        ref_audio: Optional custom voice reference (legacy, use voice_name instead).
        stream: Enable streaming for reduced time-to-first-audio (default True).
        voice_name: Voice name (without extension). Uses format-agnostic loading.
    """
    generate_speech(
        text=message,
        ref_audio=ref_audio,
        play=True,
        stream=stream,
        voice_name=voice_name,
    )


def is_mlx_available() -> bool:
    """
    Check if MLX TTS is available (mlx_audio installed + at least one voice exists).

    Returns:
        True if MLX TTS can be used, False otherwise.
    """
    try:
        import mlx_audio  # noqa: F401
        from tts_config import discover_voices
        return len(discover_voices()) > 0
    except ImportError:
        return False
