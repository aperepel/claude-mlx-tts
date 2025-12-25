"""
MLX TTS Core - Direct Python API wrapper for mlx_audio TTS.

This module provides a clean interface for:
- Model loading with caching
- Speech generation with configurable options
- Error handling and fallback

Usage:
    from mlx_tts_core import speak_mlx, get_model

    # Simple usage (auto-loads and caches model)
    speak_mlx("Hello world")

    # Advanced usage with model reuse
    model = get_model()
    generate_speech("First message", model=model)
    generate_speech("Second message", model=model)  # No reload
"""
import os
import logging
from typing import Any

log = logging.getLogger(__name__)

# Configuration - can be overridden by environment variables
MLX_MODEL = os.environ.get("MLX_TTS_MODEL", "mlx-community/chatterbox-turbo-fp16")
MLX_VOICE_REF = os.environ.get(
    "MLX_TTS_VOICE_REF",
    os.path.join(os.path.dirname(__file__), "..", "assets", "default_voice.wav")
)
MLX_SPEED = float(os.environ.get("MLX_TTS_SPEED", "1.6"))

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
    speed: float | None = None,
    play: bool = True,
    save_path: str | None = None,
    verbose: bool = False,
) -> None:
    """
    Generate speech from text using MLX TTS.

    Args:
        text: Text to convert to speech.
        model: Pre-loaded model (uses cached model if None).
        ref_audio: Path to voice reference WAV file.
        ref_text: Transcript of reference audio.
        speed: Speech speed multiplier (default MLX_SPEED).
        play: Whether to play audio immediately.
        save_path: Optional path to save audio file.
        verbose: Enable verbose logging from mlx_audio.
    """
    if not text or not text.strip():
        log.warning("Empty text provided, skipping generation")
        return

    try:
        from mlx_audio.tts.generate import generate_audio
    except ImportError as e:
        log.error("mlx_audio not installed")
        raise ImportError("mlx_audio is required") from e

    # Use provided model or get cached
    if model is None:
        model = get_model()

    # Set defaults
    actual_ref_audio = ref_audio or MLX_VOICE_REF
    actual_speed = speed if speed is not None else MLX_SPEED

    # Verify voice reference exists
    if not os.path.exists(actual_ref_audio):
        log.warning(f"Voice reference not found: {actual_ref_audio}")

    log.debug(f"Generating speech: '{text[:50]}...' (speed={actual_speed}, play={play})")

    # Suppress tokenizers parallelism warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    try:
        # Determine file_prefix from save_path
        file_prefix = None
        if save_path:
            # generate_audio expects prefix without extension
            file_prefix = save_path.rsplit(".", 1)[0] if "." in save_path else save_path

        generate_audio(
            text=text,
            model=model,
            ref_audio=actual_ref_audio,
            ref_text=ref_text,
            speed=actual_speed,
            play=play,
            file_prefix=file_prefix,
            verbose=verbose,
        )

        # mlx_audio adds _000 suffix to output files, rename to requested path
        if save_path and file_prefix:
            actual_output = f"{file_prefix}_000.wav"
            if os.path.exists(actual_output) and actual_output != save_path:
                import shutil
                shutil.move(actual_output, save_path)
                log.debug(f"Renamed {actual_output} -> {save_path}")

        log.debug("Speech generation completed")
    except Exception as e:
        log.error(f"Speech generation failed: {e}")
        raise


def speak_mlx(
    message: str,
    speed: float | None = None,
    ref_audio: str | None = None,
) -> None:
    """
    High-level function to speak a message using MLX TTS.

    This is the main entry point for simple TTS playback.
    Automatically uses cached model for fast subsequent calls.

    Args:
        message: Text to speak.
        speed: Speech speed multiplier (default MLX_SPEED).
        ref_audio: Optional custom voice reference.
    """
    generate_speech(
        text=message,
        ref_audio=ref_audio,
        speed=speed,
        play=True,
    )


def is_mlx_available() -> bool:
    """
    Check if MLX TTS is available (mlx_audio installed + voice reference exists).

    Returns:
        True if MLX TTS can be used, False otherwise.
    """
    try:
        import mlx_audio  # noqa: F401
        return os.path.exists(MLX_VOICE_REF)
    except ImportError:
        return False
