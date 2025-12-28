"""
MLX TTS Server Utilities - Lifecycle management for mlx_audio.server.

This module provides:
- Server health checks
- Auto-start/stop functionality
- HTTP-based TTS requests to the server

Architecture:
    Hook fires → Server alive? ──Yes──▶ Send TTS request
                      │
                      No
                      ↓
                Start server
                Wait for ready (~10s cold)
                Send TTS request

Usage:
    from mlx_server_utils import speak_mlx_http, stop_server, get_server_status

    # TTS (auto-starts server if needed)
    speak_mlx_http("Hello, world!")

    # Check status
    status = get_server_status()

    # Stop server to reclaim memory
    stop_server()
"""
import logging
import os
import subprocess
import sys
import time
from typing import Any, Generator

import requests

from streaming_wav_parser import StreamingWavParser, WavHeader, WavParseError

# Import AudioPlayer for streaming playback
try:
    from mlx_audio.tts.audio_player import AudioPlayer
except ImportError:
    AudioPlayer = None

# Import audio processor for compression
try:
    from audio_processor import create_processor
except ImportError:
    create_processor = None

log = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

TTS_SERVER_PORT = int(os.environ.get("TTS_SERVER_PORT", "21099"))
TTS_SERVER_HOST = os.environ.get("TTS_SERVER_HOST", "localhost")
TTS_START_TIMEOUT = int(os.environ.get("TTS_START_TIMEOUT", "60"))  # 60s for cold start

# Server log file
TTS_SERVER_LOG = os.environ.get(
    "TTS_SERVER_LOG",
    os.path.join(os.path.dirname(__file__), "..", "logs", "mlx-server.log")
)

# MLX model to preload
MLX_MODEL = os.environ.get("MLX_TTS_MODEL", "mlx-community/chatterbox-turbo-fp16")


# =============================================================================
# Exceptions
# =============================================================================

class ServerStartError(Exception):
    """Raised when server fails to start within timeout."""
    pass


class TTSRequestError(Exception):
    """Raised when TTS request to server fails."""
    pass


# =============================================================================
# Server Health Check (Phase 1)
# =============================================================================

def is_server_alive(port: int = TTS_SERVER_PORT, host: str = TTS_SERVER_HOST) -> bool:
    """
    Check if the TTS server is running and responding.

    Args:
        port: Server port to check.
        host: Server host.

    Returns:
        True if server responds to health check, False otherwise.
    """
    try:
        # mlx_audio.server uses root endpoint (no /health)
        response = requests.get(
            f"http://{host}:{port}/",
            timeout=2
        )
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False
    except Exception as e:
        log.debug(f"Health check failed: {e}")
        return False


def wait_for_ready(
    port: int = TTS_SERVER_PORT,
    host: str = TTS_SERVER_HOST,
    timeout: int = 30,
    poll_interval: float = 0.5
) -> bool:
    """
    Wait for server to become ready.

    Args:
        port: Server port.
        host: Server host.
        timeout: Maximum seconds to wait.
        poll_interval: Seconds between health checks.

    Returns:
        True if server becomes ready, False on timeout.
    """
    deadline = time.time() + timeout

    while time.time() < deadline:
        if is_server_alive(port=port, host=host):
            return True
        time.sleep(poll_interval)

    return False


# =============================================================================
# Auto-Start Logic (Phase 2)
# =============================================================================

def ensure_server_running(
    port: int = TTS_SERVER_PORT,
    timeout: int = TTS_START_TIMEOUT
) -> None:
    """
    Ensure the TTS server is running, starting it if necessary.

    Args:
        port: Server port.
        timeout: Maximum seconds to wait for startup.

    Raises:
        ServerStartError: If server fails to start within timeout.
    """
    if is_server_alive(port=port):
        log.debug(f"Server already running on port {port}")
        return

    # Get active voice from config
    try:
        from tts_config import get_active_voice
        voice_name = get_active_voice()
    except ImportError:
        voice_name = None

    log.info(f"Starting TTS server with voice caching on port {port} (voice: {voice_name})")

    # Start our custom server with voice caching pre-warm
    server_script = os.path.join(os.path.dirname(__file__), "tts_server.py")
    cmd = [
        sys.executable, server_script,
        "--port", str(port),
        "--workers", "1",
        "--model", MLX_MODEL,
    ]
    # Pass voice name if available (server will read from config if not provided)
    if voice_name:
        cmd.extend(["--voice", voice_name])

    # Log server output to file for debugging
    log_path = os.path.abspath(TTS_SERVER_LOG)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log.info(f"Server logs: {log_path}")
    log_file = open(log_path, "a")
    log_file.write(f"\n=== Server starting at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    log_file.flush()

    subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Detach from parent
    )

    log.info(f"Waiting for server to become ready (timeout={timeout}s)")

    if not wait_for_ready(port=port, timeout=timeout):
        raise ServerStartError(f"Server failed to start within {timeout}s")

    log.info("Server is ready")


# =============================================================================
# TTS via HTTP (Phase 3)
# =============================================================================

def speak_mlx_http(
    text: str,
    voice: str | None = None,
    timeout: int = 60,
    max_retries: int = 2
) -> None:
    """
    Send TTS request to the server via HTTP.

    Auto-starts server if not running. Retries on transient failures like
    invalid WAV responses (can occur during multi-chunk streaming).

    Note: Speed parameter is not supported by Chatterbox model.
    The mlx_audio library explicitly ignores speed for Chatterbox.

    Args:
        text: Text to convert to speech.
        voice: Voice name to use. Passed directly to server API.
               If None, server uses active voice from config.
        timeout: Request timeout in seconds.
        max_retries: Number of retry attempts for transient failures.

    Raises:
        ServerStartError: If server fails to start.
        TTSRequestError: If TTS request fails after all retries.
    """
    if not text or not text.strip():
        log.warning("Empty text, skipping TTS")
        return

    # Voice is passed directly in the request payload - no config modification needed
    # This avoids race conditions between concurrent hooks

    ensure_server_running()

    url = f"http://{TTS_SERVER_HOST}:{TTS_SERVER_PORT}/v1/audio/speech"

    # Pass voice directly to server API (mlx_audio server supports 'voice' field)
    # Server will use this voice, or fall back to active voice from config if None
    payload = {
        "input": text,
        "model": MLX_MODEL,
        # Note: speed parameter not included - Chatterbox doesn't support it
    }
    if voice:
        payload["voice"] = voice

    last_error = None

    for attempt in range(max_retries + 1):
        try:
            log.debug(f"Sending TTS request (attempt {attempt + 1}): {text[:50]}...")

            gen_start = time.time()
            response = requests.post(url, json=payload, timeout=timeout)

            if response.status_code != 200:
                raise TTSRequestError(
                    f"TTS request failed with status {response.status_code}: {response.text}"
                )

            # Play the audio response
            audio_data = response.content
            gen_time = time.time() - gen_start

            if not audio_data:
                raise TTSRequestError("Empty audio response from server")

            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            import io

            # Validate WAV header before parsing
            if len(audio_data) < 12 or audio_data[:4] != b'RIFF':
                # Log diagnostic info for debugging
                preview = audio_data[:100] if len(audio_data) > 100 else audio_data
                log.warning(
                    f"Invalid WAV response (attempt {attempt + 1}): "
                    f"len={len(audio_data)}, preview={preview!r}"
                )
                raise TTSRequestError(
                    f"Invalid WAV response: missing RIFF header (len={len(audio_data)})"
                )

            # Load audio from response bytes
            audio_buffer = io.BytesIO(audio_data)
            data, samplerate = sf.read(audio_buffer)
            data = data.astype(np.float32)

            # Apply compression/limiting for consistent, punchy playback
            try:
                from audio_processor import create_processor
                processor = create_processor(sample_rate=samplerate)
                data = processor(data)
            except ImportError:
                log.debug("audio_processor not available, skipping compression")

            # Calculate audio duration
            audio_duration = len(data) / samplerate
            log.info(f"Generation: {gen_time:.2f}s ({audio_duration:.1f}s audio)")

            play_start = time.time()
            sd.play(data, samplerate)
            sd.wait()
            play_time = time.time() - play_start
            log.info(f"Playback: {play_time:.2f}s")

            # Success - exit the retry loop
            return

        except requests.exceptions.RequestException as e:
            last_error = TTSRequestError(f"TTS request failed: {e}")
            log.warning(f"Request error (attempt {attempt + 1}): {e}")
        except TTSRequestError as e:
            last_error = e
            if attempt < max_retries:
                log.warning(f"Retrying after error (attempt {attempt + 1}): {e}")
                time.sleep(0.5)  # Brief pause before retry
            else:
                log.warning(f"All retries exhausted: {e}")

    # All retries failed
    raise last_error or TTSRequestError("TTS request failed after all retries")


# =============================================================================
# Streaming TTS via HTTP
# =============================================================================

def stream_tts_http(
    text: str,
    voice: str | None = None,
    timeout: int = 60,
    chunk_size: int = 8192,
) -> Generator[tuple[WavHeader, "np.ndarray"], None, None]:
    """
    Stream TTS audio chunks from the server via HTTP.

    Uses `stream=True` with requests and `iter_content()` to process
    audio data as it arrives, enabling low-latency playback.

    Auto-starts server if not running.

    Args:
        text: Text to convert to speech.
        voice: Voice name to use. If None, server uses active voice from config.
        timeout: Request timeout in seconds (for first byte).
        chunk_size: Size of chunks to read from response (default 8192).

    Yields:
        Tuples of (WavHeader, audio_chunk) where audio_chunk is a float32
        numpy array. Header is the same object for all yields.

    Raises:
        ServerStartError: If server fails to start.
        TTSRequestError: If TTS request fails.
    """
    import numpy as np

    if not text or not text.strip():
        log.warning("Empty text, skipping TTS")
        return

    ensure_server_running()

    url = f"http://{TTS_SERVER_HOST}:{TTS_SERVER_PORT}/v1/audio/speech"

    payload = {
        "input": text,
        "model": MLX_MODEL,
    }
    if voice:
        payload["voice"] = voice

    response = None
    parser = StreamingWavParser()

    try:
        log.debug(f"Sending streaming TTS request: {text[:50]}...")

        response = requests.post(url, json=payload, timeout=timeout, stream=True)

        if response.status_code != 200:
            raise TTSRequestError(
                f"TTS request failed with status {response.status_code}: {response.text}"
            )

        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue

            try:
                parser.feed(chunk)
            except WavParseError as e:
                raise TTSRequestError(f"WAV parsing error: {e}")

            audio = parser.read_audio()
            if audio is not None and len(audio) > 0:
                yield (parser.header, audio)

    except requests.exceptions.Timeout as e:
        raise TTSRequestError(f"TTS request timeout: {e}")
    except requests.exceptions.RequestException as e:
        raise TTSRequestError(f"TTS request failed: {e}")
    except TTSRequestError:
        raise
    except Exception as e:
        raise TTSRequestError(f"Unexpected error during streaming: {e}")
    finally:
        if response is not None:
            response.close()


def play_streaming_http(
    text: str,
    voice: str | None = None,
    timeout: int = 60,
    chunk_size: int = 8192,
) -> dict[str, float]:
    """
    Stream TTS audio from the server and play with AudioPlayer.

    Uses stream_tts_http() to get streaming audio chunks and plays them
    with low-latency AudioPlayer. Applies stateful audio compression
    across chunks and captures timing metrics.

    Auto-starts server if not running.

    Args:
        text: Text to convert to speech.
        voice: Voice name to use. If None, server uses active voice from config.
        timeout: Request timeout in seconds (for first byte).
        chunk_size: Size of chunks to read from response (default 8192).

    Returns:
        Dict with timing metrics:
            - ttft: Time to first audio chunk (seconds)
            - gen_time: Total streaming time (seconds)
            - play_time: Time waiting for playback drain (seconds)
            - chunks: Number of audio chunks received

    Raises:
        ServerStartError: If server fails to start.
        TTSRequestError: If TTS request fails.
    """
    import numpy as np

    # Handle empty text
    if not text or not text.strip():
        log.warning("Empty text, skipping TTS")
        return {
            "ttft": 0.0,
            "gen_time": 0.0,
            "play_time": 0.0,
            "chunks": 0,
        }

    ensure_server_running()

    url = f"http://{TTS_SERVER_HOST}:{TTS_SERVER_PORT}/v1/audio/speech"

    payload = {
        "input": text,
        "model": MLX_MODEL,
    }
    if voice:
        payload["voice"] = voice

    response = None
    parser = StreamingWavParser()
    player = None
    audio_processor = None

    # Metrics tracking
    gen_start = time.perf_counter()
    ttft = None
    chunk_count = 0

    try:
        log.debug(f"Sending streaming TTS request: {text[:50]}...")

        response = requests.post(url, json=payload, timeout=timeout, stream=True)

        if response.status_code != 200:
            raise TTSRequestError(
                f"TTS request failed with status {response.status_code}: {response.text}"
            )

        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue

            try:
                parser.feed(chunk)
            except WavParseError as e:
                raise TTSRequestError(f"WAV parsing error: {e}")

            audio = parser.read_audio()
            if audio is not None and len(audio) > 0:
                # Capture TTFT on first audio chunk
                if ttft is None:
                    ttft = time.perf_counter() - gen_start

                    # Initialize player with sample rate from header
                    if AudioPlayer is not None and parser.header is not None:
                        player = AudioPlayer(sample_rate=parser.header.sample_rate)

                    # Initialize audio processor for compression
                    if create_processor is not None and parser.header is not None:
                        audio_processor = create_processor(sample_rate=parser.header.sample_rate)
                        log.debug("Audio processor initialized for streaming")

                chunk_count += 1

                # Apply audio compression if available
                if audio_processor is not None:
                    audio = audio_processor(audio)

                # Queue audio for playback
                if player is not None:
                    player.queue_audio(audio)

        gen_time = time.perf_counter() - gen_start

        # Wait for playback to finish
        play_time = 0.0
        if player is not None:
            # Handle short audio that didn't meet the min buffer threshold
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

        return {
            "ttft": ttft or 0.0,
            "gen_time": gen_time,
            "play_time": play_time,
            "chunks": chunk_count,
        }

    except requests.exceptions.Timeout as e:
        raise TTSRequestError(f"TTS request timeout: {e}")
    except requests.exceptions.RequestException as e:
        raise TTSRequestError(f"TTS request failed: {e}")
    except TTSRequestError:
        raise
    except Exception as e:
        raise TTSRequestError(f"Unexpected error during streaming: {e}")
    finally:
        if response is not None:
            response.close()


# =============================================================================
# Server Stop (Phase 4)
# =============================================================================

def stop_server(port: int = TTS_SERVER_PORT) -> bool:
    """
    Stop the TTS server if running.

    Args:
        port: Server port.

    Returns:
        True if server was stopped, False if not running.
    """
    if not is_server_alive(port=port):
        log.debug("Server not running, nothing to stop")
        return False

    log.info(f"Stopping server on port {port}")

    # Find and gracefully stop the process using the port
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )

        if result.stdout.strip():
            pids = result.stdout.strip().split("\n")

            # Send SIGINT first - uvicorn handles this more gracefully than SIGTERM
            # This allows proper cleanup of multiprocessing resources (semaphores)
            for pid in pids:
                try:
                    subprocess.run(["kill", "-INT", pid], check=True)
                    log.debug(f"Sent SIGINT to process {pid}")
                except subprocess.CalledProcessError:
                    pass

            # Wait for graceful shutdown (uvicorn needs time to cleanup)
            time.sleep(3)

            # Force kill any remaining processes
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], check=True,
                                   stderr=subprocess.DEVNULL)
                    log.debug(f"Force killed process {pid}")
                except subprocess.CalledProcessError:
                    pass  # Already dead

    except Exception as e:
        log.warning(f"Error stopping server: {e}")
        return False

    # Wait for server to stop
    for _ in range(10):
        if not is_server_alive(port=port):
            log.info("Server stopped")
            return True
        time.sleep(0.5)

    log.warning("Server did not stop cleanly")
    return False


def get_server_status(port: int = TTS_SERVER_PORT) -> dict[str, Any]:
    """
    Get current server status.

    Args:
        port: Server port.

    Returns:
        Dictionary with status information.
    """
    running = is_server_alive(port=port)

    return {
        "running": running,
        "port": port,
        "host": TTS_SERVER_HOST,
        "model": MLX_MODEL,
    }


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python mlx_server_utils.py <text> [--voice <voice_name>]", file=sys.stderr)
        sys.exit(1)

    # Parse arguments
    voice_name = None
    args = sys.argv[1:]
    if "--voice" in args:
        voice_idx = args.index("--voice")
        if voice_idx + 1 < len(args):
            voice_name = args[voice_idx + 1]
            args = args[:voice_idx] + args[voice_idx + 2:]

    text = " ".join(args)

    # Always use speak_mlx for CLI - it correctly reads active voice from config.
    # The HTTP server path (speak_mlx_http) is for hooks where we leverage
    # the pre-warmed server, but it doesn't dynamically switch voices.
    from mlx_tts_core import speak_mlx
    speak_mlx(text, voice_name=voice_name)
