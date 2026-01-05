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
import argparse
import logging
import os
import struct
import sys
import time
import warnings
from pathlib import Path
from typing import AsyncGenerator

import numpy as np

# Suppress startup warnings BEFORE any imports that trigger them
# Suppress transformers "no PyTorch/TensorFlow" advisory (we use MLX)
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Suppress webrtcvad's pkg_resources deprecation warning (upstream issue)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
    module="webrtcvad"
)

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
DEFAULT_VOICE = "default"


def get_default_voice_name() -> str:
    """Get the default voice name from config."""
    try:
        from tts_config import get_active_voice
        return get_active_voice()
    except ImportError:
        return DEFAULT_VOICE


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


def create_wav_header(
    sample_rate: int,
    channels: int = 1,
    bits_per_sample: int = 16,
    data_size: int = 0x7FFFFFFF,
) -> bytes:
    """
    Create WAV header for streaming.

    Uses a large placeholder data_size since we don't know total length upfront.
    Clients should read until connection closes, not until data_size bytes.
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,  # file size - 8
        b"WAVE",
        b"fmt ",
        16,  # fmt chunk size
        1,   # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size,
    )
    return header


def register_streaming_endpoint(app, model_provider, model_name: str):
    """
    Register a true streaming TTS endpoint that uses stream=True.

    This endpoint yields audio chunks as they're generated, enabling
    playback to start before generation completes (fast TTFT).
    """
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    class StreamingSpeechRequest(BaseModel):
        model: str
        input: str
        voice: str | None = None
        streaming_interval: float = 0.5  # seconds per chunk

    async def generate_streaming_audio(
        model,
        text: str,
        voice: str | None = None,
        streaming_interval: float = 0.5,
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate streaming audio with proper WAV framing.

        Uses OLA (Overlap-Add) processing to eliminate clicks at chunk boundaries,
        plus optional compressor/limiter for consistent volume.

        Yields:
            First: WAV header (44 bytes)
            Then: Raw PCM audio chunks as int16 bytes
        """
        sample_rate = model.sample_rate
        header_sent = False
        gen_start = time.perf_counter()
        ttft = None
        chunk_count = 0
        total_samples = 0

        # Initialize OLA + compressor/limiter processor
        audio_processor = None
        try:
            from audio_processor import create_processor_with_ola, DEFAULT_CROSSFADE_MS
            import tts_config
            compressor = tts_config.get_effective_compressor(voice)
            limiter = tts_config.get_effective_limiter(voice)
            audio_processor = create_processor_with_ola(
                sample_rate=sample_rate,
                crossfade_ms=DEFAULT_CROSSFADE_MS,
                input_gain_db=compressor.get("input_gain_db"),
                threshold_db=compressor.get("threshold_db"),
                ratio=compressor.get("ratio"),
                attack_ms=compressor.get("attack_ms"),
                release_ms=compressor.get("release_ms"),
                gain_db=compressor.get("gain_db"),
                master_gain_db=compressor.get("master_gain_db"),
                compressor_enabled=compressor.get("enabled"),
                limiter_threshold_db=limiter.get("threshold_db"),
                limiter_release_ms=limiter.get("release_ms"),
                limiter_enabled=limiter.get("enabled"),
            )
            log.debug(f"Audio processor with OLA initialized for streaming (voice={voice})")
        except ImportError:
            log.debug("audio_processor not available, skipping OLA and compression")

        for result in model.generate(
            text,
            voice=voice,
            stream=True,
            streaming_interval=streaming_interval,
        ):
            audio = np.asarray(result.audio)

            # Apply OLA + compression/limiting
            if audio_processor is not None:
                audio = audio_processor(audio)

            if not header_sent:
                ttft = time.perf_counter() - gen_start
                yield create_wav_header(sample_rate)
                header_sent = True

            chunk_count += 1
            total_samples += len(audio)

            # Convert float32 [-1, 1] to int16 PCM
            if len(audio) > 0:
                audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                yield audio_int16.tobytes()

        # Flush OLA buffer to get remaining samples
        if audio_processor is not None:
            flush_audio = audio_processor(None)
            if len(flush_audio) > 0:
                total_samples += len(flush_audio)
                flush_int16 = (flush_audio * 32767).clip(-32768, 32767).astype(np.int16)
                yield flush_int16.tobytes()

        gen_time = time.perf_counter() - gen_start
        audio_duration = total_samples / sample_rate if sample_rate > 0 else 0
        rtf = gen_time / audio_duration if audio_duration > 0 else 0

        log.info(
            f"TTS STREAM: ttft={ttft:.2f}s gen={gen_time:.2f}s "
            f"chunks={chunk_count} duration={audio_duration:.1f}s RTF={rtf:.2f}"
        )

    @app.post("/v1/audio/speech/stream")
    async def tts_speech_streaming(payload: StreamingSpeechRequest):
        """
        Generate streaming speech audio with fast TTFT.

        Unlike /v1/audio/speech, this endpoint uses stream=True to yield
        audio chunks as they're generated, enabling playback to start
        before generation completes.

        Returns chunked WAV: header (44 bytes) + raw PCM chunks.
        """
        model = model_provider.load_model(payload.model)
        return StreamingResponse(
            generate_streaming_audio(
                model,
                payload.input,
                payload.voice,
                payload.streaming_interval,
            ),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav",
                "X-Streaming": "true",
            },
        )

    log.info("Registered streaming endpoint: POST /v1/audio/speech/stream")


def patch_model_generate(model, initial_voice: str):
    """
    Patch model.generate for dynamic voice switching and streaming metrics.

    Accepts voice parameter directly from request (preferred), or falls back
    to config. Reloads conditionals when voice changes. Logs streaming metrics.
    """
    original_generate = model.generate
    sample_rate = model.sample_rate

    # Track currently loaded voice
    state = {"current_voice": initial_voice}

    def dynamic_generate_with_metrics(
        text,
        ref_audio=None,
        voice=None,
        **kwargs
    ):
        # Determine requested voice: prefer explicit parameter, fall back to config
        if voice:
            requested_voice = voice
        else:
            try:
                from tts_config import get_active_voice
                requested_voice = get_active_voice()
            except ImportError:
                requested_voice = state["current_voice"]

        # Reload voice if changed
        if requested_voice != state["current_voice"]:
            log.info(f"Voice changed: {state['current_voice']} -> {requested_voice}")
            try:
                from voice_cache import get_voice_conditionals
                model._conds = get_voice_conditionals(model, requested_voice)
                state["current_voice"] = requested_voice
                log.info(f"Loaded voice: {requested_voice}")
            except Exception as e:
                log.warning(f"Failed to load voice {requested_voice}: {e}")

        # Use cached conditionals (ref_audio=None triggers use of model._conds)
        actual_ref_audio = None

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
            f"TTS [{state['current_voice']}]: ttft={ttft:.2f}s gen={gen_time:.2f}s "
            f"chunks={chunk_count} RTF={rtf:.2f}"
        )

    model.generate = dynamic_generate_with_metrics
    log.info(f"Patched model.generate for dynamic voice switching (initial: {initial_voice})")


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

    log.info("Starting TTS server with voice caching...")
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

    # Patch model for dynamic voice switching (re-reads config per request)
    patch_model_generate(model, voice_name)

    # Register our streaming endpoint (uses stream=True for true streaming)
    register_streaming_endpoint(app, model_provider, args.model)

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
