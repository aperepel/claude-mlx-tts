"""
Unit tests for TTS mute module.

Run with: uv run pytest tests/unit/test_tts_mute.py -v
"""
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch


# Add scripts to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))


class TestMuteStatus:
    """Tests for MuteStatus namedtuple."""

    def test_mute_status_fields(self):
        """MuteStatus should have is_muted, expires_at, remaining_seconds fields."""
        from tts_mute import MuteStatus

        status = MuteStatus(is_muted=True, expires_at=1234567890.0, remaining_seconds=300.0)
        assert status.is_muted is True
        assert status.expires_at == 1234567890.0
        assert status.remaining_seconds == 300.0

    def test_mute_status_none_values(self):
        """MuteStatus should accept None for expires_at and remaining_seconds."""
        from tts_mute import MuteStatus

        status = MuteStatus(is_muted=True, expires_at=None, remaining_seconds=None)
        assert status.is_muted is True
        assert status.expires_at is None
        assert status.remaining_seconds is None


class TestUnmuteKeywords:
    """Tests for UNMUTE_KEYWORDS constant."""

    def test_unmute_keywords_exists(self):
        """UNMUTE_KEYWORDS should be defined."""
        from tts_mute import UNMUTE_KEYWORDS

        assert isinstance(UNMUTE_KEYWORDS, list)

    def test_unmute_keywords_contains_resume(self):
        """UNMUTE_KEYWORDS should contain 'resume'."""
        from tts_mute import UNMUTE_KEYWORDS

        assert "resume" in UNMUTE_KEYWORDS

    def test_unmute_keywords_contains_off(self):
        """UNMUTE_KEYWORDS should contain 'off'."""
        from tts_mute import UNMUTE_KEYWORDS

        assert "off" in UNMUTE_KEYWORDS

    def test_unmute_keywords_contains_unmute(self):
        """UNMUTE_KEYWORDS should contain 'unmute'."""
        from tts_mute import UNMUTE_KEYWORDS

        assert "unmute" in UNMUTE_KEYWORDS

    def test_unmute_keywords_contains_cancel(self):
        """UNMUTE_KEYWORDS should contain 'cancel'."""
        from tts_mute import UNMUTE_KEYWORDS

        assert "cancel" in UNMUTE_KEYWORDS


class TestGetMuteStatus:
    """Tests for get_mute_status function."""

    def test_get_mute_status_not_muted_when_no_file(self):
        """get_mute_status should return not muted when file doesn't exist."""
        from tts_mute import get_mute_status

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"

            with patch("tts_mute.MUTE_FILE", mute_file):
                status = get_mute_status()

            assert status.is_muted is False
            assert status.expires_at is None
            assert status.remaining_seconds is None

    def test_get_mute_status_indefinite_mute_when_empty_file(self):
        """get_mute_status should return indefinite mute when file is empty."""
        from tts_mute import get_mute_status

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"
            mute_file.write_text("")

            with patch("tts_mute.MUTE_FILE", mute_file):
                status = get_mute_status()

            assert status.is_muted is True
            assert status.expires_at is None
            assert status.remaining_seconds is None

    def test_get_mute_status_timed_mute_when_future_timestamp(self):
        """get_mute_status should return timed mute when file has future timestamp."""
        from tts_mute import get_mute_status

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"
            future_time = time.time() + 300  # 5 minutes from now
            mute_file.write_text(str(future_time))

            with patch("tts_mute.MUTE_FILE", mute_file):
                status = get_mute_status()

            assert status.is_muted is True
            assert status.expires_at == future_time
            assert status.remaining_seconds is not None
            assert status.remaining_seconds > 0
            assert status.remaining_seconds <= 300

    def test_get_mute_status_not_muted_when_expired(self):
        """get_mute_status should return not muted when timestamp has passed."""
        from tts_mute import get_mute_status

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"
            past_time = time.time() - 100  # 100 seconds ago
            mute_file.write_text(str(past_time))

            with patch("tts_mute.MUTE_FILE", mute_file):
                status = get_mute_status()

            assert status.is_muted is False
            assert status.expires_at is None
            assert status.remaining_seconds is None

    def test_get_mute_status_cleans_up_expired_file(self):
        """get_mute_status should remove expired mute file."""
        from tts_mute import get_mute_status

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"
            past_time = time.time() - 100
            mute_file.write_text(str(past_time))

            with patch("tts_mute.MUTE_FILE", mute_file):
                get_mute_status()

            assert not mute_file.exists()

    def test_get_mute_status_handles_invalid_content(self):
        """get_mute_status should return not muted for invalid file content."""
        from tts_mute import get_mute_status

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"
            mute_file.write_text("not a number")

            with patch("tts_mute.MUTE_FILE", mute_file):
                status = get_mute_status()

            assert status.is_muted is False


class TestIsMuted:
    """Tests for is_muted function."""

    def test_is_muted_returns_false_when_no_file(self):
        """is_muted should return False when no mute file exists."""
        from tts_mute import is_muted

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"

            with patch("tts_mute.MUTE_FILE", mute_file):
                result = is_muted()

            assert result is False

    def test_is_muted_returns_true_when_muted(self):
        """is_muted should return True when mute is active."""
        from tts_mute import is_muted

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"
            mute_file.write_text("")  # Indefinite mute

            with patch("tts_mute.MUTE_FILE", mute_file):
                result = is_muted()

            assert result is True

    def test_is_muted_returns_true_for_future_timestamp(self):
        """is_muted should return True when mute expires in future."""
        from tts_mute import is_muted

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"
            mute_file.write_text(str(time.time() + 300))

            with patch("tts_mute.MUTE_FILE", mute_file):
                result = is_muted()

            assert result is True

    def test_is_muted_returns_false_for_past_timestamp(self):
        """is_muted should return False when mute has expired."""
        from tts_mute import is_muted

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"
            mute_file.write_text(str(time.time() - 100))

            with patch("tts_mute.MUTE_FILE", mute_file):
                result = is_muted()

            assert result is False


class TestClearMute:
    """Tests for clear_mute function."""

    def test_clear_mute_removes_file(self):
        """clear_mute should remove the mute file."""
        from tts_mute import clear_mute

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"
            mute_file.write_text("12345")

            with patch("tts_mute.MUTE_FILE", mute_file):
                clear_mute()

            assert not mute_file.exists()

    def test_clear_mute_does_not_fail_when_no_file(self):
        """clear_mute should not fail when mute file doesn't exist."""
        from tts_mute import clear_mute

        with tempfile.TemporaryDirectory() as tmpdir:
            mute_file = Path(tmpdir) / "mute_until"

            with patch("tts_mute.MUTE_FILE", mute_file):
                # Should not raise
                clear_mute()


class TestSetMute:
    """Tests for set_mute function."""

    def test_set_mute_indefinite_with_none(self):
        """set_mute(None) should create indefinite mute."""
        from tts_mute import set_mute

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            mute_file = config_dir / "mute_until"

            with patch("tts_mute._CONFIG_DIR", config_dir), \
                 patch("tts_mute.MUTE_FILE", mute_file):
                result = set_mute(None)

            assert result is None
            assert mute_file.exists()
            assert mute_file.read_text() == ""

    def test_set_mute_indefinite_with_empty_string(self):
        """set_mute('') should create indefinite mute."""
        from tts_mute import set_mute

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            mute_file = config_dir / "mute_until"

            with patch("tts_mute._CONFIG_DIR", config_dir), \
                 patch("tts_mute.MUTE_FILE", mute_file):
                result = set_mute("")

            assert result is None
            assert mute_file.read_text() == ""

    def test_set_mute_clears_on_resume(self):
        """set_mute('resume') should clear mute."""
        from tts_mute import set_mute

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            mute_file = config_dir / "mute_until"
            mute_file.write_text("")  # Create existing mute

            with patch("tts_mute._CONFIG_DIR", config_dir), \
                 patch("tts_mute.MUTE_FILE", mute_file):
                set_mute("resume")

            assert not mute_file.exists()

    def test_set_mute_clears_on_off(self):
        """set_mute('off') should clear mute."""
        from tts_mute import set_mute

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            mute_file = config_dir / "mute_until"
            mute_file.write_text("")

            with patch("tts_mute._CONFIG_DIR", config_dir), \
                 patch("tts_mute.MUTE_FILE", mute_file):
                set_mute("off")

            assert not mute_file.exists()

    def test_set_mute_clears_on_unmute(self):
        """set_mute('unmute') should clear mute."""
        from tts_mute import set_mute

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            mute_file = config_dir / "mute_until"
            mute_file.write_text("")

            with patch("tts_mute._CONFIG_DIR", config_dir), \
                 patch("tts_mute.MUTE_FILE", mute_file):
                set_mute("unmute")

            assert not mute_file.exists()

    def test_set_mute_clears_on_cancel(self):
        """set_mute('cancel') should clear mute."""
        from tts_mute import set_mute

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            mute_file = config_dir / "mute_until"
            mute_file.write_text("")

            with patch("tts_mute._CONFIG_DIR", config_dir), \
                 patch("tts_mute.MUTE_FILE", mute_file):
                set_mute("cancel")

            assert not mute_file.exists()

    def test_set_mute_creates_config_dir(self):
        """set_mute should create config directory if it doesn't exist."""
        from tts_mute import set_mute

        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "new_dir"
            mute_file = config_dir / "mute_until"

            with patch("tts_mute._CONFIG_DIR", config_dir), \
                 patch("tts_mute.MUTE_FILE", mute_file):
                set_mute(None)

            assert config_dir.exists()
            assert mute_file.exists()


class TestFormatRemainingTime:
    """Tests for format_remaining_time function."""

    def test_format_remaining_time_indefinite(self):
        """format_remaining_time should return 'indefinitely' for None."""
        from tts_mute import format_remaining_time

        result = format_remaining_time(None)
        assert result == "indefinitely"

    def test_format_remaining_time_seconds(self):
        """format_remaining_time should format seconds correctly."""
        from tts_mute import format_remaining_time

        result = format_remaining_time(45)
        assert result == "45 seconds"

    def test_format_remaining_time_one_minute(self):
        """format_remaining_time should format 1 minute correctly."""
        from tts_mute import format_remaining_time

        result = format_remaining_time(60)
        assert result == "1 minute"

    def test_format_remaining_time_minutes(self):
        """format_remaining_time should format minutes correctly."""
        from tts_mute import format_remaining_time

        result = format_remaining_time(300)
        assert result == "5 minutes"

    def test_format_remaining_time_hours(self):
        """format_remaining_time should format hours correctly."""
        from tts_mute import format_remaining_time

        result = format_remaining_time(7200)
        assert "2.0 hours" in result

    def test_format_remaining_time_days(self):
        """format_remaining_time should format days correctly."""
        from tts_mute import format_remaining_time

        result = format_remaining_time(172800)
        assert "2.0 days" in result


class TestMuteFilePath:
    """Tests for mute file path configuration."""

    def test_mute_file_in_config_dir(self):
        """MUTE_FILE should be in .config directory."""
        from tts_mute import MUTE_FILE, _CONFIG_DIR

        assert MUTE_FILE.parent == _CONFIG_DIR

    def test_mute_file_named_correctly(self):
        """MUTE_FILE should be named 'mute_until'."""
        from tts_mute import MUTE_FILE

        assert MUTE_FILE.name == "mute_until"
