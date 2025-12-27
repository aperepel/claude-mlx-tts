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
from typing import Any

import requests

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
MLX_VOICE_REF = os.environ.get(
    "MLX_TTS_VOICE_REF",
    os.path.join(os.path.dirname(__file__), "..", "assets", "default_voice.wav")
)


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

    log.info(f"Starting TTS server with voice caching on port {port}")

    # Start our custom server with voice caching pre-warm
    server_script = os.path.join(os.path.dirname(__file__), "tts_server.py")
    cmd = [
        sys.executable, server_script,
        "--port", str(port),
        "--workers", "1",
        "--voice", os.path.abspath(MLX_VOICE_REF),
        "--model", MLX_MODEL,
    ]

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
    timeout: int = 60
) -> None:
    """
    Send TTS request to the server via HTTP.

    Auto-starts server if not running.

    Note: Speed parameter is not supported by Chatterbox model.
    The mlx_audio library explicitly ignores speed for Chatterbox.

    Args:
        text: Text to convert to speech.
        voice: Voice/model to use.
        timeout: Request timeout in seconds.

    Raises:
        ServerStartError: If server fails to start.
        TTSRequestError: If TTS request fails.
    """
    if not text or not text.strip():
        log.warning("Empty text, skipping TTS")
        return

    ensure_server_running()

    url = f"http://{TTS_SERVER_HOST}:{TTS_SERVER_PORT}/v1/audio/speech"

    payload = {
        "input": text,
        "model": voice or MLX_MODEL,
        # Note: speed parameter not included - Chatterbox doesn't support it
    }

    log.debug(f"Sending TTS request: {text[:50]}...")

    try:
        gen_start = time.time()
        response = requests.post(url, json=payload, timeout=timeout)

        if response.status_code != 200:
            raise TTSRequestError(
                f"TTS request failed with status {response.status_code}: {response.text}"
            )

        # Play the audio response
        audio_data = response.content
        gen_time = time.time() - gen_start

        if audio_data:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
            import io

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

    except requests.exceptions.RequestException as e:
        raise TTSRequestError(f"TTS request failed: {e}") from e


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
        print("Usage: python mlx_server_utils.py <text>", file=sys.stderr)
        sys.exit(1)

    text = " ".join(sys.argv[1:])

    if is_server_alive():
        speak_mlx_http(text)
    else:
        from mlx_tts_core import speak_mlx
        speak_mlx(text)
