"""
Unit tests for TTS server lifecycle management.

Run with: uv run pytest tests/unit/test_server_lifecycle.py -v
"""
import os
import sys
import time
import socket
from unittest.mock import patch, MagicMock

import pytest

# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


# =============================================================================
# Configuration constants (should match implementation)
# =============================================================================
TTS_SERVER_PORT = 21099
TTS_SERVER_HOST = "localhost"


# =============================================================================
# Phase 1: Server Health Check Tests
# =============================================================================

class TestIsServerAlive:
    """Tests for is_server_alive() function."""

    def test_returns_false_when_no_server(self):
        """is_server_alive should return False when no server is running."""
        from mlx_server_utils import is_server_alive

        # Use a port we know isn't in use
        assert is_server_alive(port=59999) is False

    def test_returns_true_when_server_running(self):
        """is_server_alive should return True when server responds to health check."""
        from mlx_server_utils import is_server_alive

        # Mock a successful HTTP response
        with patch("mlx_server_utils.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            assert is_server_alive(port=TTS_SERVER_PORT) is True
            mock_get.assert_called_once()

    def test_returns_false_on_connection_error(self):
        """is_server_alive should return False on connection errors."""
        from mlx_server_utils import is_server_alive

        with patch("mlx_server_utils.requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.ConnectionError()

            assert is_server_alive(port=TTS_SERVER_PORT) is False

    def test_returns_false_on_timeout(self):
        """is_server_alive should return False on timeout."""
        from mlx_server_utils import is_server_alive

        with patch("mlx_server_utils.requests.get") as mock_get:
            import requests
            mock_get.side_effect = requests.exceptions.Timeout()

            assert is_server_alive(port=TTS_SERVER_PORT) is False


class TestWaitForReady:
    """Tests for wait_for_ready() function."""

    def test_returns_true_when_server_already_up(self):
        """wait_for_ready should return True immediately if server is up."""
        from mlx_server_utils import wait_for_ready

        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            mock_alive.return_value = True

            result = wait_for_ready(port=TTS_SERVER_PORT, timeout=5)
            assert result is True
            # Should only check once if already up
            assert mock_alive.call_count == 1

    def test_returns_true_when_server_comes_up(self):
        """wait_for_ready should poll and return True when server eventually responds."""
        from mlx_server_utils import wait_for_ready

        call_count = [0]

        def mock_alive(port, host=None):
            call_count[0] += 1
            # Simulate server coming up after 2 checks
            return call_count[0] >= 2

        with patch("mlx_server_utils.is_server_alive", side_effect=mock_alive):
            result = wait_for_ready(port=TTS_SERVER_PORT, timeout=10)
            assert result is True
            assert call_count[0] >= 2

    def test_returns_false_on_timeout(self):
        """wait_for_ready should return False if server doesn't respond within timeout."""
        from mlx_server_utils import wait_for_ready

        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            mock_alive.return_value = False

            start = time.time()
            result = wait_for_ready(port=TTS_SERVER_PORT, timeout=1)
            elapsed = time.time() - start

            assert result is False
            # Should timeout after ~1 second
            assert elapsed >= 1.0
            assert elapsed < 2.0

    def test_uses_default_timeout(self):
        """wait_for_ready should use default timeout of 30s."""
        from mlx_server_utils import wait_for_ready

        # Just verify it accepts no timeout arg and runs with defaults
        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            mock_alive.return_value = True
            result = wait_for_ready(port=TTS_SERVER_PORT)
            assert result is True


# =============================================================================
# Phase 2: Auto-Start Logic Tests
# =============================================================================

class TestEnsureServerRunning:
    """Tests for ensure_server_running() function."""

    def test_does_nothing_if_server_already_alive(self):
        """ensure_server_running should not start server if already running."""
        from mlx_server_utils import ensure_server_running

        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            with patch("mlx_server_utils.subprocess.Popen") as mock_popen:
                mock_alive.return_value = True

                ensure_server_running()

                # Should not spawn new process
                mock_popen.assert_not_called()

    def test_starts_server_when_not_running(self):
        """ensure_server_running should start server subprocess if not alive."""
        from mlx_server_utils import ensure_server_running

        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            with patch("mlx_server_utils.wait_for_ready") as mock_wait:
                with patch("mlx_server_utils.subprocess.Popen") as mock_popen:
                    mock_alive.return_value = False
                    mock_wait.return_value = True

                    ensure_server_running()

                    # Should spawn server process
                    mock_popen.assert_called_once()

    def test_waits_for_server_after_start(self):
        """ensure_server_running should wait for server to become ready after starting."""
        from mlx_server_utils import ensure_server_running

        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            with patch("mlx_server_utils.wait_for_ready") as mock_wait:
                with patch("mlx_server_utils.subprocess.Popen"):
                    mock_alive.return_value = False
                    mock_wait.return_value = True

                    ensure_server_running()

                    mock_wait.assert_called_once()

    def test_raises_on_server_start_timeout(self):
        """ensure_server_running should raise if server fails to start."""
        from mlx_server_utils import ensure_server_running, ServerStartError

        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            with patch("mlx_server_utils.wait_for_ready") as mock_wait:
                with patch("mlx_server_utils.subprocess.Popen"):
                    mock_alive.return_value = False
                    mock_wait.return_value = False  # Timeout!

                    with pytest.raises(ServerStartError):
                        ensure_server_running()

    def test_uses_correct_port(self):
        """ensure_server_running should use configured port."""
        from mlx_server_utils import ensure_server_running, TTS_SERVER_PORT

        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            with patch("mlx_server_utils.wait_for_ready") as mock_wait:
                with patch("mlx_server_utils.subprocess.Popen") as mock_popen:
                    mock_alive.return_value = False
                    mock_wait.return_value = True

                    ensure_server_running()

                    # Verify port in command args
                    call_args = mock_popen.call_args[0][0]
                    assert "--port" in call_args
                    port_idx = call_args.index("--port")
                    assert call_args[port_idx + 1] == str(TTS_SERVER_PORT)


# =============================================================================
# Phase 3: TTS via HTTP Tests
# =============================================================================

class TestSpeakMlxHttp:
    """Tests for speak_mlx_http() function."""

    def test_ensures_server_running_before_request(self):
        """speak_mlx_http should ensure server is running before making request."""
        from mlx_server_utils import speak_mlx_http

        with patch("mlx_server_utils.ensure_server_running") as mock_ensure:
            with patch("mlx_server_utils.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b""  # Empty audio data to skip playback
                mock_post.return_value = mock_response

                speak_mlx_http("Hello")

                mock_ensure.assert_called_once()

    def test_sends_correct_payload(self):
        """speak_mlx_http should send correct JSON payload to server."""
        from mlx_server_utils import speak_mlx_http

        with patch("mlx_server_utils.ensure_server_running"):
            with patch("mlx_server_utils.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b""  # Empty audio data to skip playback
                mock_post.return_value = mock_response

                speak_mlx_http("Test message")

                mock_post.assert_called_once()
                call_kwargs = mock_post.call_args[1]
                payload = call_kwargs["json"]
                assert payload["input"] == "Test message"
                # Note: speed not included - Chatterbox doesn't support it
                assert "speed" not in payload

    def test_uses_correct_endpoint(self):
        """speak_mlx_http should POST to /v1/audio/speech endpoint."""
        from mlx_server_utils import speak_mlx_http, TTS_SERVER_PORT

        with patch("mlx_server_utils.ensure_server_running"):
            with patch("mlx_server_utils.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b""  # Empty audio data to skip playback
                mock_post.return_value = mock_response

                speak_mlx_http("Test")

                call_args = mock_post.call_args
                url = call_args[0][0]
                assert f":{TTS_SERVER_PORT}" in url
                assert "/v1/audio/speech" in url

    def test_raises_on_server_error(self):
        """speak_mlx_http should raise on non-200 response."""
        from mlx_server_utils import speak_mlx_http, TTSRequestError

        with patch("mlx_server_utils.ensure_server_running"):
            with patch("mlx_server_utils.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 500
                mock_response.text = "Internal error"
                mock_post.return_value = mock_response

                with pytest.raises(TTSRequestError):
                    speak_mlx_http("Test")

    def test_passes_voice_in_payload_when_specified(self):
        """speak_mlx_http should pass voice parameter directly in request payload."""
        from mlx_server_utils import speak_mlx_http

        with patch("mlx_server_utils.ensure_server_running"):
            with patch("mlx_server_utils.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b""  # Empty audio data to skip playback
                mock_post.return_value = mock_response

                speak_mlx_http("Test message", voice="custom_voice")

                mock_post.assert_called_once()
                call_kwargs = mock_post.call_args[1]
                payload = call_kwargs["json"]
                assert payload["voice"] == "custom_voice"

    def test_omits_voice_from_payload_when_not_specified(self):
        """speak_mlx_http should not include voice in payload when None."""
        from mlx_server_utils import speak_mlx_http

        with patch("mlx_server_utils.ensure_server_running"):
            with patch("mlx_server_utils.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b""  # Empty audio data to skip playback
                mock_post.return_value = mock_response

                speak_mlx_http("Test message")

                mock_post.assert_called_once()
                call_kwargs = mock_post.call_args[1]
                payload = call_kwargs["json"]
                assert "voice" not in payload

    def test_does_not_modify_config_when_voice_specified(self):
        """speak_mlx_http should NOT modify active_voice in config (prevents race conditions)."""
        from mlx_server_utils import speak_mlx_http

        with patch("mlx_server_utils.ensure_server_running"):
            with patch("mlx_server_utils.requests.post") as mock_post:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.content = b""
                mock_post.return_value = mock_response

                # Patch set_active_voice to verify it's NOT called
                with patch("tts_config.set_active_voice") as mock_set_voice:
                    speak_mlx_http("Test", voice="different_voice")

                    # set_active_voice should NOT be called - we pass voice in payload instead
                    mock_set_voice.assert_not_called()


# =============================================================================
# Phase 4: Server Stop Tests
# =============================================================================

class TestStopServer:
    """Tests for stop_server() function."""

    def test_does_nothing_if_server_not_running(self):
        """stop_server should return gracefully if no server running."""
        from mlx_server_utils import stop_server

        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            mock_alive.return_value = False

            # Should not raise
            result = stop_server()
            assert result is False  # No server to stop

    def test_stops_running_server(self):
        """stop_server should terminate a running server process."""
        from mlx_server_utils import stop_server

        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            with patch("mlx_server_utils.subprocess.run") as mock_run:
                mock_alive.side_effect = [True, False]  # Running, then stopped

                result = stop_server()

                assert result is True
                mock_run.assert_called()


class TestGetServerStatus:
    """Tests for get_server_status() function."""

    def test_returns_status_dict(self):
        """get_server_status should return status information dict."""
        from mlx_server_utils import get_server_status

        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            mock_alive.return_value = True

            status = get_server_status()

            assert isinstance(status, dict)
            assert "running" in status
            assert status["running"] is True

    def test_returns_port_info(self):
        """get_server_status should include port information."""
        from mlx_server_utils import get_server_status, TTS_SERVER_PORT

        with patch("mlx_server_utils.is_server_alive") as mock_alive:
            mock_alive.return_value = True

            status = get_server_status()

            assert "port" in status
            assert status["port"] == TTS_SERVER_PORT
