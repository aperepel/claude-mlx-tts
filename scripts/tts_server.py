"""
MLX TTS Server with Voice Caching and Streaming Metrics.

This is a wrapper around mlx_audio.server that pre-warms voice conditionals
from disk cache, eliminating the ~1.5s voice encoding overhead per request.

Usage:
    python scripts/tts_server.py --port 21099

Architecture:
    Startup:
        1. Load TTS model
        2. Load voice conditionals from cache (or prepare and cache)
        3. Pre-set model._conds
        4. Start FastAPI server

    Request handling:
        - Requests can omit ref_audio to use pre-warmed voice
        - If ref_audio is provided and matches default, uses cache
        - If ref_audio differs, prepares fresh conditionals

    Metrics logged per request:
        TTS: ttft=0.26s gen=2.45s chunks=7 RTF=0.58
"""
# Suppress startup warnings BEFORE any imports that trigger them
import os
import warnings

# Suppress transformers "no PyTorch/TensorFlow" advisory (we use MLX)
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Suppress webrtcvad's pkg_resources deprecation warning (upstream issue)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="webrtcvad"
)

import argparse
import logging
import sys
import time
from pathlib import Path

# Setup logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)

# Add scripts directory to path for voice_cache import
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPTS_DIR))

# Configuration
MLX_MODEL = os.environ.get("MLX_TTS_MODEL", "mlx-community/chatterbox-turbo-fp16")
MLX_VOICE_REF = os.environ.get(
    "MLX_TTS_VOICE_REF",
    str(SCRIPTS_DIR.parent / "assets" / "default_voice.wav")
)


def get_default_voice_name() -> str:
    """Get the default voice name from config or environment."""
    try:
        from tts_config import get_active_voice
        return get_active_voice()
    except ImportError:
        # Fallback: extract name from MLX_VOICE_REF path
        return Path(MLX_VOICE_REF).stem


def prewarm_voice_cache(model, voice_name: str | None = None) -> bool:
    """
    Pre-warm the model with cached voice conditionals.

    Uses format-agnostic loading: prefers .safetensors, falls back to .wav.

    Args:
        model: The TTS model to pre-warm.
        voice_name: Voice name (without extension). Uses active voice if None.

    Returns True if using pre-computed embeddings, False if extracted from wav.
    """
    try:
        from voice_cache import get_voice_conditionals
        from tts_config import get_voice_format

        if voice_name is None:
            voice_name = get_default_voice_name()

        fmt = get_voice_format(voice_name)
        log.info(f"Loading voice '{voice_name}' (format: {fmt or 'unknown'})")

        # Load conditionals using format-agnostic loader
        conds = get_voice_conditionals(model, voice_name)

        # Set model's internal conditionals
        model._conds = conds

        log.info(f"Voice conditionals pre-warmed successfully: {voice_name}")
        return fmt == "safetensors"

    except Exception as e:
        log.warning(f"Failed to pre-warm voice cache: {e}")
        return False


def patch_model_generate(model, default_voice_ref: str):
    """
    Patch model.generate to use cached conditionals and log streaming metrics.

    This avoids re-extracting conditionals on every request and logs:
    - TTFT (time to first audio chunk)
    - Total generation time
    - Chunk count
    - RTF (real-time factor)
    """
    original_generate = model.generate
    sample_rate = model.sample_rate

    def cached_generate_with_metrics(
        text,
        ref_audio=None,
        **kwargs
    ):
        # Determine cache status and prepare ref_audio arg
        use_cache = False
        if ref_audio is None:
            use_cache = True
        else:
            ref_path = os.path.abspath(str(ref_audio)) if ref_audio else None
            default_path = os.path.abspath(default_voice_ref)
            if ref_path == default_path:
                use_cache = True

        if use_cache:
            log.debug("[CACHE HIT] Using pre-warmed voice")
            actual_ref_audio = None
        else:
            log.info(f"Different voice requested, extracting fresh: {ref_audio}")
            actual_ref_audio = ref_audio

        # Wrap generator to capture metrics
        gen_start = time.perf_counter()
        ttft = None
        chunk_count = 0
        total_samples = 0
        last_rtf = 0.0

        for result in original_generate(text=text, ref_audio=actual_ref_audio, **kwargs):
            if ttft is None:
                ttft = time.perf_counter() - gen_start

            chunk_count += 1
            total_samples += len(result.audio)
            if hasattr(result, 'real_time_factor') and result.real_time_factor:
                last_rtf = result.real_time_factor

            yield result

        gen_time = time.perf_counter() - gen_start

        # Calculate metrics
        audio_duration = total_samples / sample_rate if sample_rate > 0 else 0.0
        rtf = gen_time / audio_duration if audio_duration > 0 else last_rtf

        # Log metrics
        log.info(
            f"TTS: ttft={ttft:.2f}s gen={gen_time:.2f}s "
            f"chunks={chunk_count} RTF={rtf:.2f}"
        )

    model.generate = cached_generate_with_metrics
    log.info("Patched model.generate to use voice cache with metrics")


def main():
    parser = argparse.ArgumentParser(description="MLX TTS Server with Voice Caching")
    parser.add_argument("--port", type=int, default=21099, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--voice", default=None, help="Voice name (without extension)")
    parser.add_argument("--model", default=MLX_MODEL, help="TTS model to use")
    args = parser.parse_args()

    # Determine voice name
    voice_name = args.voice or get_default_voice_name()

    log.info(f"Starting TTS server with voice caching...")
    log.info(f"Model: {args.model}")
    log.info(f"Voice: {voice_name}")

    # Import server components
    from mlx_audio.server import app, model_provider
    import uvicorn

    # Pre-load model
    log.info("Loading TTS model...")
    model = model_provider.load_model(args.model)
    log.info(f"Model loaded: {type(model).__name__}")

    # Pre-warm voice cache with format-agnostic loading
    prewarm_voice_cache(model, voice_name)

    # Patch model to use cache (uses legacy path for API compatibility)
    # The patched generate will use pre-set _conds when ref_audio matches default
    patch_model_generate(model, MLX_VOICE_REF)

    # Start server (single process to preserve model state)
    log.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
