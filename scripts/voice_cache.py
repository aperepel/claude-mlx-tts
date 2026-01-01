"""
Voice conditionals caching module.

Provides caching utilities for MLX TTS voice conditionals using safetensors.
This eliminates the need to re-extract voice embeddings on every TTS request.

Usage:
    from voice_cache import get_or_prepare_conditionals

    model = load_model(...)
    conds = get_or_prepare_conditionals(model, "path/to/voice.wav")
"""
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx

if TYPE_CHECKING:
    from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import Conditionals

log = logging.getLogger(__name__)

__all__ = [
    "CACHE_DIR",
    "get_cache_path",
    "is_cache_valid",
    "save_conditionals",
    "load_conditionals",
    "load_conditionals_from_file",
    "get_or_prepare_conditionals",
    "get_voice_conditionals",
]

# Cache directory for voice conditionals (plugin-local)
_PLUGIN_ROOT = Path(__file__).parent.parent
CACHE_DIR = _PLUGIN_ROOT / ".cache" / "voice_conds"


def get_cache_path(voice_ref: str) -> Path:
    """
    Generate cache path based on file content hash (sha256[:16]).

    Args:
        voice_ref: Path to the voice reference file.

    Returns:
        Path to the cache file.

    Raises:
        FileNotFoundError: If the voice reference file doesn't exist.
    """
    voice_path = Path(voice_ref)
    if not voice_path.exists():
        raise FileNotFoundError(f"Voice reference file not found: {voice_ref}")

    # Hash file content for cache key
    content = voice_path.read_bytes()
    content_hash = hashlib.sha256(content).hexdigest()[:16]

    return CACHE_DIR / f"{content_hash}.safetensors"


def is_cache_valid(voice_ref: str) -> bool:
    """
    Check if cached conditionals exist for this voice file.

    Args:
        voice_ref: Path to the voice reference file.

    Returns:
        True if valid cache exists, False otherwise.
    """
    try:
        cache_path = get_cache_path(voice_ref)
        return cache_path.exists()
    except FileNotFoundError:
        return False


def save_conditionals(conds: "Conditionals", voice_ref: str) -> Path:
    """
    Save conditionals to cache using safetensors format.

    Args:
        conds: Conditionals object to save.
        voice_ref: Path to the voice reference file (used for cache key).

    Returns:
        Path to the saved cache file.
    """
    cache_path = get_cache_path(voice_ref)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Flatten conditionals into a dict of arrays
    arrays = {}

    # T3 conditionals
    arrays["t3_speaker_emb"] = conds.t3.speaker_emb
    arrays["t3_cond_prompt_speech_tokens"] = conds.t3.cond_prompt_speech_tokens

    if conds.t3.clap_emb is not None:
        arrays["t3_clap_emb"] = conds.t3.clap_emb
    if conds.t3.cond_prompt_speech_emb is not None:
        arrays["t3_cond_prompt_speech_emb"] = conds.t3.cond_prompt_speech_emb
    if conds.t3.emotion_adv is not None:
        arrays["t3_emotion_adv"] = conds.t3.emotion_adv

    # Gen conditionals
    for key, value in conds.gen.items():
        arrays[f"gen_{key}"] = value

    mx.save_safetensors(str(cache_path), arrays)
    log.debug(f"Saved voice conditionals to cache: {cache_path}")

    return cache_path


def load_conditionals(voice_ref: str) -> "Conditionals | None":
    """
    Load cached conditionals from safetensors file.

    Args:
        voice_ref: Path to the voice reference file (used for cache key).

    Returns:
        Conditionals object if cache exists, None otherwise.
    """
    if not is_cache_valid(voice_ref):
        return None

    try:
        cache_path = get_cache_path(voice_ref)
        loaded = mx.load(str(cache_path))

        # Narrow type: safetensors always returns dict[str, array]
        if not isinstance(loaded, dict):
            log.warning(f"Unexpected load result type: {type(loaded)}")
            return None
        arrays: dict[str, mx.array] = loaded

        # Import the types we need
        from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import (
            Conditionals,
            T3Cond,
        )

        # Reconstruct T3Cond
        t3_cond = T3Cond(
            speaker_emb=arrays["t3_speaker_emb"],
            cond_prompt_speech_tokens=arrays["t3_cond_prompt_speech_tokens"],
            clap_emb=arrays.get("t3_clap_emb"),
            cond_prompt_speech_emb=arrays.get("t3_cond_prompt_speech_emb"),
            emotion_adv=arrays.get("t3_emotion_adv"),
        )

        # Reconstruct gen dict
        gen = {}
        for key, value in arrays.items():
            if key.startswith("gen_"):
                gen_key = key[4:]  # Remove "gen_" prefix
                gen[gen_key] = value

        return Conditionals(t3=t3_cond, gen=gen)

    except Exception as e:
        log.warning(f"Failed to load cached conditionals: {e}")
        return None


def load_conditionals_from_file(safetensors_path: Path) -> "Conditionals":
    """
    Load pre-computed conditionals directly from a safetensors file.

    Unlike load_conditionals(), this function loads directly from the given path
    without requiring a source WAV reference for cache key lookup.

    Args:
        safetensors_path: Path to the safetensors file containing voice conditionals.

    Returns:
        Conditionals object with voice embeddings.

    Raises:
        FileNotFoundError: If the safetensors file doesn't exist.
    """
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Safetensors file not found: {safetensors_path}")

    loaded = mx.load(str(safetensors_path))

    # Narrow type: safetensors always returns dict[str, array]
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected dict from safetensors, got {type(loaded)}")
    arrays: dict[str, mx.array] = loaded

    # Import the types we need
    from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import (
        Conditionals,
        T3Cond,
    )

    # Reconstruct T3Cond
    t3_cond = T3Cond(
        speaker_emb=arrays["t3_speaker_emb"],
        cond_prompt_speech_tokens=arrays["t3_cond_prompt_speech_tokens"],
        clap_emb=arrays.get("t3_clap_emb"),
        cond_prompt_speech_emb=arrays.get("t3_cond_prompt_speech_emb"),
        emotion_adv=arrays.get("t3_emotion_adv"),
    )

    # Reconstruct gen dict
    gen = {}
    for key, value in arrays.items():
        if key.startswith("gen_"):
            gen_key = key[4:]  # Remove "gen_" prefix
            gen[gen_key] = value

    log.debug(f"Loaded conditionals from: {safetensors_path}")
    return Conditionals(t3=t3_cond, gen=gen)


def get_or_prepare_conditionals(model: Any, voice_ref: str) -> "Conditionals":
    """
    Main entry point - load from cache or extract and cache.

    Args:
        model: The TTS model with prepare_conditionals method.
        voice_ref: Path to the voice reference file.

    Returns:
        Conditionals object (from cache or freshly prepared).
    """
    # Try loading from cache first
    cached = load_conditionals(voice_ref)
    if cached is not None:
        log.info(f"Voice conditionals cache hit for: {voice_ref}")
        return cached

    # Cache miss - prepare fresh conditionals
    log.info(f"Voice conditionals cache miss, preparing: {voice_ref}")
    model.prepare_conditionals(voice_ref)
    conds = model._conds

    # Save to cache for next time
    save_conditionals(conds, voice_ref)
    log.info(f"Cached voice conditionals for: {voice_ref}")

    return conds


def get_voice_conditionals(model: Any, voice_name: str) -> "Conditionals":
    """
    Unified entry point - format-agnostic loading by voice name.

    Resolution priority:
    1. assets/{name}.safetensors → load directly (fastest)
    2. assets/{name}.wav → load from cache or extract

    Args:
        model: The TTS model with prepare_conditionals method.
        voice_name: The voice name (without extension).

    Returns:
        Conditionals object (from safetensors or wav extraction).

    Raises:
        ValueError: If no voice files found for the given name.
    """
    assets = _PLUGIN_ROOT / "assets"

    # Priority 1: Pre-computed safetensors (fastest)
    safetensors_path = assets / f"{voice_name}.safetensors"
    if safetensors_path.exists():
        log.info(f"Loading pre-computed conditionals from: {safetensors_path}")
        return load_conditionals_from_file(safetensors_path)

    # Priority 2: WAV with cache fallback
    wav_path = assets / f"{voice_name}.wav"
    if wav_path.exists():
        log.info(f"Loading conditionals from WAV: {wav_path}")
        return get_or_prepare_conditionals(model, str(wav_path))

    raise ValueError(f"No voice files found for: {voice_name}")
