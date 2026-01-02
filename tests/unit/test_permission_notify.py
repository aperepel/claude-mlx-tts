"""
Unit tests for permission-notify.py hook.

Run with: uv run pytest tests/unit/test_permission_notify.py

The permission-notify hook triggers TTS when:
1. Time since last user message >= 30s (IDLE_THRESHOLD_SECS)
2. OR >= 3 auto-approved tools in sequence (MIN_AUTO_APPROVED_TOOLS)
3. AND not within 60s cooldown period (NOTIFY_COOLDOWN_SECS)
"""
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

# Import permission_notify.py module dynamically (it's not a package)
HOOKS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "hooks")
PERMISSION_NOTIFY_PATH = os.path.join(HOOKS_DIR, "permission_notify.py")
spec = importlib.util.spec_from_file_location("permission_notify", PERMISSION_NOTIFY_PATH)
assert spec is not None, "Failed to load module spec"
assert spec.loader is not None, "Module spec has no loader"
permission_notify = importlib.util.module_from_spec(spec)
sys.modules['permission_notify'] = permission_notify
spec.loader.exec_module(permission_notify)


# =============================================================================
# FIXTURES - Test Data Helpers
# =============================================================================

@pytest.fixture
def temp_transcript():
    """Create a temporary transcript file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def temp_cooldown_file():
    """Create a temporary cooldown timestamp file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.timestamp', delete=False) as f:
        yield f.name
    # Cleanup
    if os.path.exists(f.name):
        os.unlink(f.name)


def create_user_message(text="Hello", timestamp=None):
    """Create a user message transcript entry."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "type": "user",
        "timestamp": timestamp,
        "message": {
            "role": "user",
            "content": text
        }
    }


def create_tool_result_message(timestamp=None):
    """Create a tool_result user entry (not a real user message)."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "type": "user",
        "timestamp": timestamp,
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "123",
                    "content": "result"
                }
            ]
        }
    }


def create_assistant_message_with_tools(tool_names=None, timestamp=None):
    """Create an assistant message with tool_use blocks."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    if tool_names is None:
        tool_names = ["Bash"]

    content = []
    for i, name in enumerate(tool_names):
        content.append({
            "type": "tool_use",
            "id": f"tool_{i}",
            "name": name,
            "input": {}
        })

    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "role": "assistant",
            "content": content
        }
    }


def create_assistant_text_message(text="Response", timestamp=None):
    """Create an assistant text-only message (no tools)."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "type": "assistant",
        "timestamp": timestamp,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}]
        }
    }


# =============================================================================
# TEST: is_real_user_message()
# =============================================================================

class TestIsRealUserMessage:
    """Tests for distinguishing real user messages from tool_result entries."""

    def test_real_user_message_with_text(self):
        """Should return True for a real user message with text content."""
        from permission_notify import is_real_user_message

        entry = create_user_message("Hello Claude")
        assert is_real_user_message(entry) is True

    def test_tool_result_user_entry_returns_false(self):
        """Should return False for user entries that are only tool_result."""
        from permission_notify import is_real_user_message

        entry = create_tool_result_message()
        assert is_real_user_message(entry) is False

    def test_non_user_type_returns_false(self):
        """Should return False for non-user entry types."""
        from permission_notify import is_real_user_message

        entry = create_assistant_text_message("Hello")
        assert is_real_user_message(entry) is False

    def test_user_message_with_mixed_content(self):
        """Should return True for user message with mixed content (not all tool_result)."""
        from permission_notify import is_real_user_message

        entry = {
            "type": "user",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's the result"},
                    {"type": "tool_result", "tool_use_id": "123", "content": "data"}
                ]
            }
        }
        assert is_real_user_message(entry) is True

    def test_empty_content_user_message(self):
        """Should handle user message with empty content."""
        from permission_notify import is_real_user_message

        entry = {
            "type": "user",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": {
                "role": "user",
                "content": ""
            }
        }
        # Empty string content is still a real user message
        assert is_real_user_message(entry) is True

    def test_missing_message_field(self):
        """Should handle entry with missing message field."""
        from permission_notify import is_real_user_message

        entry = {"type": "user", "timestamp": "2024-01-01T00:00:00Z"}
        # Should not crash, return False (no message content to check)
        result = is_real_user_message(entry)
        assert result is False


# =============================================================================
# TEST: extract_tool_name()
# =============================================================================

class TestExtractToolName:
    """Tests for extracting tool names from permission messages."""

    def test_extract_bash_tool(self):
        """Should extract 'Bash' from permission message."""
        from permission_notify import extract_tool_name

        message = "Claude wants to use Bash"
        assert extract_tool_name(message) == "Bash"

    def test_extract_write_tool(self):
        """Should extract 'Write' from permission message."""
        from permission_notify import extract_tool_name

        message = "Claude wants to use Write"
        assert extract_tool_name(message) == "Write"

    def test_extract_edit_tool(self):
        """Should extract 'Edit' from permission message."""
        from permission_notify import extract_tool_name

        message = "Claude wants to use Edit"
        assert extract_tool_name(message) == "Edit"

    def test_extract_read_tool(self):
        """Should extract 'Read' from permission message."""
        from permission_notify import extract_tool_name

        message = "Claude wants to use Read"
        assert extract_tool_name(message) == "Read"

    def test_extract_webfetch_tool(self):
        """Should extract 'WebFetch' from permission message."""
        from permission_notify import extract_tool_name

        message = "Claude wants to use WebFetch"
        assert extract_tool_name(message) == "WebFetch"

    def test_extract_task_tool(self):
        """Should extract 'Task' from permission message."""
        from permission_notify import extract_tool_name

        message = "Claude wants to use Task"
        assert extract_tool_name(message) == "Task"

    def test_unknown_tool_returns_generic(self):
        """Should return 'a tool' for unknown tool names."""
        from permission_notify import extract_tool_name

        message = "Claude wants to use SomethingUnknown"
        assert extract_tool_name(message) == "a tool"

    def test_empty_message(self):
        """Should return 'a tool' for empty message."""
        from permission_notify import extract_tool_name

        assert extract_tool_name("") == "a tool"


# =============================================================================
# TEST: get_time_since_last_user_message()
# =============================================================================

class TestGetTimeSinceLastUserMessage:
    """Tests for calculating time since last real user message."""

    def test_recent_user_message_returns_small_value(self, temp_transcript):
        """Should return small time delta for recent user message."""
        from permission_notify import get_time_since_last_user_message

        # Create message from 2 seconds ago
        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()
        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello", timestamp)) + '\n')

        result = get_time_since_last_user_message(temp_transcript)
        # Should be around 2 seconds (allow 1s margin for test execution time)
        assert 1.0 <= result <= 3.0

    def test_old_user_message_returns_large_value(self, temp_transcript):
        """Should return large time delta for old user message."""
        from permission_notify import get_time_since_last_user_message

        # Create message from 60 seconds ago
        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()
        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello", timestamp)) + '\n')

        result = get_time_since_last_user_message(temp_transcript)
        # Should be around 60 seconds (allow 1s margin)
        assert 59.0 <= result <= 61.0

    def test_ignores_tool_result_entries(self, temp_transcript):
        """Should skip tool_result entries and find real user message."""
        from permission_notify import get_time_since_last_user_message

        old_timestamp = (datetime.now(timezone.utc) - timedelta(seconds=50)).isoformat()
        recent_timestamp = (datetime.now(timezone.utc) - timedelta(seconds=2)).isoformat()

        with open(temp_transcript, 'w') as f:
            # Real user message 50s ago
            f.write(json.dumps(create_user_message("Hello", old_timestamp)) + '\n')
            # Tool result 2s ago (should be ignored)
            f.write(json.dumps(create_tool_result_message(recent_timestamp)) + '\n')

        result = get_time_since_last_user_message(temp_transcript)
        # Should report ~50s, not 2s
        assert 49.0 <= result <= 51.0

    def test_finds_most_recent_user_message(self, temp_transcript):
        """Should find the most recent real user message."""
        from permission_notify import get_time_since_last_user_message

        old_timestamp = (datetime.now(timezone.utc) - timedelta(seconds=50)).isoformat()
        recent_timestamp = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()

        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("First", old_timestamp)) + '\n')
            f.write(json.dumps(create_user_message("Second", recent_timestamp)) + '\n')

        result = get_time_since_last_user_message(temp_transcript)
        # Should report ~5s, not 50s
        assert 4.0 <= result <= 6.0

    def test_missing_file_returns_infinity(self):
        """Should return infinity for missing transcript file."""
        from permission_notify import get_time_since_last_user_message

        result = get_time_since_last_user_message("/nonexistent/transcript.jsonl")
        assert result == float('inf')

    def test_empty_file_returns_infinity(self, temp_transcript):
        """Should return infinity for empty transcript file."""
        from permission_notify import get_time_since_last_user_message

        # Create empty file
        with open(temp_transcript, 'w'):
            pass

        result = get_time_since_last_user_message(temp_transcript)
        assert result == float('inf')

    def test_malformed_json_lines_are_skipped(self, temp_transcript):
        """Should skip malformed JSON lines and process valid ones."""
        from permission_notify import get_time_since_last_user_message

        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()

        with open(temp_transcript, 'w') as f:
            f.write("invalid json line\n")
            f.write(json.dumps(create_user_message("Valid", timestamp)) + '\n')
            f.write("another bad line\n")

        result = get_time_since_last_user_message(temp_transcript)
        # Should find the valid message
        assert 9.0 <= result <= 11.0

    def test_invalid_timestamp_format_returns_infinity(self, temp_transcript):
        """Should return infinity when timestamp cannot be parsed."""
        from permission_notify import get_time_since_last_user_message

        entry = create_user_message("Hello", "invalid-timestamp")
        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(entry) + '\n')

        result = get_time_since_last_user_message(temp_transcript)
        assert result == float('inf')

    def test_missing_timestamp_returns_infinity(self, temp_transcript):
        """Should return infinity when timestamp field is missing."""
        from permission_notify import get_time_since_last_user_message

        entry = {
            "type": "user",
            "message": {"role": "user", "content": "Hello"}
        }
        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(entry) + '\n')

        result = get_time_since_last_user_message(temp_transcript)
        assert result == float('inf')


# =============================================================================
# TEST: count_auto_approved_tools_since_last_user()
# =============================================================================

class TestCountAutoApprovedToolsSinceLastUser:
    """Tests for counting tool calls since last user message."""

    def test_no_tools_returns_zero(self, temp_transcript):
        """Should return 0 when no tools were called."""
        from permission_notify import count_auto_approved_tools_since_last_user

        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello")) + '\n')
            f.write(json.dumps(create_assistant_text_message("Response")) + '\n')

        result = count_auto_approved_tools_since_last_user(temp_transcript)
        assert result == 0

    def test_counts_single_tool_call(self, temp_transcript):
        """Should count a single tool call."""
        from permission_notify import count_auto_approved_tools_since_last_user

        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello")) + '\n')
            f.write(json.dumps(create_assistant_message_with_tools(["Bash"])) + '\n')

        result = count_auto_approved_tools_since_last_user(temp_transcript)
        assert result == 1

    def test_counts_multiple_tools_in_one_message(self, temp_transcript):
        """Should count multiple tool_use blocks in a single assistant message."""
        from permission_notify import count_auto_approved_tools_since_last_user

        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello")) + '\n')
            f.write(json.dumps(create_assistant_message_with_tools(["Bash", "Read", "Write"])) + '\n')

        result = count_auto_approved_tools_since_last_user(temp_transcript)
        assert result == 3

    def test_counts_tools_across_multiple_assistant_messages(self, temp_transcript):
        """Should count tools across multiple assistant responses."""
        from permission_notify import count_auto_approved_tools_since_last_user

        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello")) + '\n')
            f.write(json.dumps(create_assistant_message_with_tools(["Bash"])) + '\n')
            f.write(json.dumps(create_tool_result_message()) + '\n')
            f.write(json.dumps(create_assistant_message_with_tools(["Read", "Write"])) + '\n')

        result = count_auto_approved_tools_since_last_user(temp_transcript)
        assert result == 3

    def test_resets_count_after_new_user_message(self, temp_transcript):
        """Should only count tools since the LAST user message."""
        from permission_notify import count_auto_approved_tools_since_last_user

        with open(temp_transcript, 'w') as f:
            # First conversation turn
            f.write(json.dumps(create_user_message("First")) + '\n')
            f.write(json.dumps(create_assistant_message_with_tools(["Bash", "Read"])) + '\n')
            # Second conversation turn (should reset count)
            f.write(json.dumps(create_user_message("Second")) + '\n')
            f.write(json.dumps(create_assistant_message_with_tools(["Write"])) + '\n')

        result = count_auto_approved_tools_since_last_user(temp_transcript)
        # Should only count 1 tool (Write), not the 2 before second user message
        assert result == 1

    def test_ignores_tool_result_user_entries(self, temp_transcript):
        """Should not treat tool_result entries as real user messages."""
        from permission_notify import count_auto_approved_tools_since_last_user

        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Real user")) + '\n')
            f.write(json.dumps(create_assistant_message_with_tools(["Bash"])) + '\n')
            f.write(json.dumps(create_tool_result_message()) + '\n')  # Not a real user message
            f.write(json.dumps(create_assistant_message_with_tools(["Read", "Write"])) + '\n')

        result = count_auto_approved_tools_since_last_user(temp_transcript)
        # Should count all 3 tools since the real user message
        assert result == 3

    def test_missing_file_returns_zero(self):
        """Should return 0 for missing transcript file."""
        from permission_notify import count_auto_approved_tools_since_last_user

        result = count_auto_approved_tools_since_last_user("/nonexistent/transcript.jsonl")
        assert result == 0

    def test_empty_file_returns_zero(self, temp_transcript):
        """Should return 0 for empty transcript file."""
        from permission_notify import count_auto_approved_tools_since_last_user

        with open(temp_transcript, 'w'):
            pass

        result = count_auto_approved_tools_since_last_user(temp_transcript)
        assert result == 0

    def test_no_user_message_returns_zero(self, temp_transcript):
        """Should return 0 when transcript has no user messages."""
        from permission_notify import count_auto_approved_tools_since_last_user

        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_assistant_message_with_tools(["Bash", "Read"])) + '\n')

        result = count_auto_approved_tools_since_last_user(temp_transcript)
        assert result == 0

    def test_malformed_json_lines_are_skipped(self, temp_transcript):
        """Should skip malformed JSON lines and count valid tools."""
        from permission_notify import count_auto_approved_tools_since_last_user

        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello")) + '\n')
            f.write("invalid json\n")
            f.write(json.dumps(create_assistant_message_with_tools(["Bash", "Read"])) + '\n')

        result = count_auto_approved_tools_since_last_user(temp_transcript)
        assert result == 2


# =============================================================================
# TEST: is_within_cooldown()
# =============================================================================

class TestIsWithinCooldown:
    """Tests for cooldown period checking."""

    def test_no_cooldown_file_returns_false(self):
        """Should return False when cooldown file doesn't exist."""
        from permission_notify import is_within_cooldown, NOTIFY_TIMESTAMP_FILE

        # Ensure file doesn't exist
        if os.path.exists(NOTIFY_TIMESTAMP_FILE):
            os.unlink(NOTIFY_TIMESTAMP_FILE)

        result = is_within_cooldown()
        assert result is False

    def test_recent_notification_returns_true(self, temp_cooldown_file):
        """Should return True when last notification was recent (within 60s)."""
        from permission_notify import is_within_cooldown

        # Patch the cooldown file path
        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            # Write timestamp from 30 seconds ago (within cooldown)
            last_notify = (datetime.now().timestamp() - 30)
            with open(temp_cooldown_file, 'w') as f:
                f.write(str(last_notify))

            result = is_within_cooldown()
            assert result is True

    def test_old_notification_returns_false(self, temp_cooldown_file):
        """Should return False when last notification was old (> 60s ago)."""
        from permission_notify import is_within_cooldown

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            # Write timestamp from 120 seconds ago (past cooldown)
            last_notify = (datetime.now().timestamp() - 120)
            with open(temp_cooldown_file, 'w') as f:
                f.write(str(last_notify))

            result = is_within_cooldown()
            assert result is False

    def test_exactly_at_cooldown_boundary(self, temp_cooldown_file):
        """Should return False when exactly at 60s boundary."""
        from permission_notify import is_within_cooldown, NOTIFY_COOLDOWN_SECS

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            # Write timestamp exactly 60 seconds ago
            last_notify = (datetime.now().timestamp() - NOTIFY_COOLDOWN_SECS)
            with open(temp_cooldown_file, 'w') as f:
                f.write(str(last_notify))

            result = is_within_cooldown()
            # At exactly 60s, should not be within cooldown (< not <=)
            assert result is False

    def test_invalid_timestamp_returns_false(self, temp_cooldown_file):
        """Should return False when timestamp file contains invalid data."""
        from permission_notify import is_within_cooldown

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            with open(temp_cooldown_file, 'w') as f:
                f.write("not-a-number")

            result = is_within_cooldown()
            assert result is False

    def test_empty_timestamp_file_returns_false(self, temp_cooldown_file):
        """Should return False when timestamp file is empty."""
        from permission_notify import is_within_cooldown

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            with open(temp_cooldown_file, 'w'):
                pass

            result = is_within_cooldown()
            assert result is False


# =============================================================================
# TEST: record_notification()
# =============================================================================

class TestRecordNotification:
    """Tests for recording notification timestamp."""

    def test_creates_timestamp_file(self, temp_cooldown_file):
        """Should create timestamp file when recording notification."""
        from permission_notify import record_notification

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            # Remove file if exists
            if os.path.exists(temp_cooldown_file):
                os.unlink(temp_cooldown_file)

            record_notification()

            assert os.path.exists(temp_cooldown_file)

    def test_writes_current_timestamp(self, temp_cooldown_file):
        """Should write current timestamp to file."""
        from permission_notify import record_notification

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            before = datetime.now().timestamp()
            record_notification()
            after = datetime.now().timestamp()

            with open(temp_cooldown_file) as f:
                recorded = float(f.read().strip())

            # Recorded timestamp should be between before and after
            assert before <= recorded <= after

    def test_overwrites_existing_timestamp(self, temp_cooldown_file):
        """Should overwrite existing timestamp file."""
        from permission_notify import record_notification

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            # Write old timestamp
            with open(temp_cooldown_file, 'w') as f:
                f.write("1000.0")

            # Record new notification
            record_notification()

            with open(temp_cooldown_file) as f:
                recorded = float(f.read().strip())

            # Should be recent, not 1000.0
            assert recorded > 1000.0


# =============================================================================
# TEST: Main Integration Tests
# =============================================================================

class TestMainIntegration:
    """Integration tests for the main() function."""

    def test_exits_1_when_no_transcript_path(self):
        """Should exit with code 1 when no transcript_path provided."""
        from permission_notify import main

        hook_input = json.dumps({})

        with patch('sys.stdin', Mock(read=lambda: hook_input)):
            with patch('permission_notify.get_hook_input', return_value={}):
                with pytest.raises(SystemExit) as exc:
                    main()
                assert exc.value.code == 1

    def test_skips_notification_when_within_cooldown(self, temp_transcript, temp_cooldown_file):
        """Should skip notification when within cooldown period."""
        from permission_notify import main

        # Setup: user message 40s ago (should trigger), but cooldown active
        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=40)).isoformat()
        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello", timestamp)) + '\n')

        # Set recent cooldown timestamp (20s ago)
        recent_cooldown = datetime.now().timestamp() - 20
        with open(temp_cooldown_file, 'w') as f:
            f.write(str(recent_cooldown))

        hook_input = {
            "transcript_path": temp_transcript,
            "message": "Claude wants to use Bash"
        }

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            with patch('permission_notify.get_hook_input', return_value=hook_input):
                with patch('permission_notify.speak_notification') as mock_speak:
                    with pytest.raises(SystemExit) as exc:
                        main()

                    # Should exit 1 but NOT call speak_notification
                    assert exc.value.code == 1
                    mock_speak.assert_not_called()

    def test_notifies_when_user_idle_too_long(self, temp_transcript, temp_cooldown_file):
        """Should notify when user has been idle >= 30 seconds."""
        from permission_notify import main

        # User message 35s ago (past threshold)
        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=35)).isoformat()
        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello", timestamp)) + '\n')

        hook_input = {
            "transcript_path": temp_transcript,
            "message": "Claude wants to use Bash"
        }

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            with patch('permission_notify.get_hook_input', return_value=hook_input):
                with patch('permission_notify.speak_notification') as mock_speak:
                    with pytest.raises(SystemExit) as exc:
                        main()

                    # Should exit 1 AND call speak_notification
                    assert exc.value.code == 1
                    mock_speak.assert_called_once_with("Bash")

    def test_notifies_when_many_tools_auto_approved(self, temp_transcript, temp_cooldown_file):
        """Should notify when >= 3 tools were auto-approved in sequence."""
        from permission_notify import main

        # User message 5s ago (recent, normally wouldn't trigger)
        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello", timestamp)) + '\n')
            # But 3 tools were called
            f.write(json.dumps(create_assistant_message_with_tools(["Bash", "Read", "Write"])) + '\n')

        hook_input = {
            "transcript_path": temp_transcript,
            "message": "Claude wants to use Edit"
        }

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            with patch('permission_notify.get_hook_input', return_value=hook_input):
                with patch('permission_notify.speak_notification') as mock_speak:
                    with pytest.raises(SystemExit) as exc:
                        main()

                    # Should trigger because of tool count threshold
                    assert exc.value.code == 1
                    mock_speak.assert_called_once_with("Edit")

    def test_does_not_notify_when_user_recently_active(self, temp_transcript, temp_cooldown_file):
        """Should NOT notify when user message is recent and few tools."""
        from permission_notify import main

        # User message 5s ago (recent)
        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello", timestamp)) + '\n')
            # Only 1 tool called
            f.write(json.dumps(create_assistant_message_with_tools(["Bash"])) + '\n')

        hook_input = {
            "transcript_path": temp_transcript,
            "message": "Claude wants to use Read"
        }

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            with patch('permission_notify.get_hook_input', return_value=hook_input):
                with patch('permission_notify.speak_notification') as mock_speak:
                    with pytest.raises(SystemExit) as exc:
                        main()

                    # Should exit 1 but NOT notify
                    assert exc.value.code == 1
                    mock_speak.assert_not_called()

    def test_records_timestamp_after_notification(self, temp_transcript, temp_cooldown_file):
        """Should record notification timestamp after speaking."""
        from permission_notify import main

        # Setup to trigger notification
        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=40)).isoformat()
        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello", timestamp)) + '\n')

        hook_input = {
            "transcript_path": temp_transcript,
            "message": "Claude wants to use Bash"
        }

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            with patch('permission_notify.get_hook_input', return_value=hook_input):
                with patch('permission_notify.speak_notification'):
                    before = datetime.now().timestamp()
                    with pytest.raises(SystemExit):
                        main()
                    after = datetime.now().timestamp()

                    # Check cooldown file was written
                    assert os.path.exists(temp_cooldown_file)
                    with open(temp_cooldown_file) as f:
                        recorded = float(f.read().strip())
                    assert before <= recorded <= after

    def test_uses_mlx_when_available(self, temp_transcript, temp_cooldown_file):
        """Should use MLX TTS when available."""
        from permission_notify import main

        # Setup to trigger notification
        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=40)).isoformat()
        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello", timestamp)) + '\n')

        hook_input = {
            "transcript_path": temp_transcript,
            "message": "Claude wants to use Bash"
        }

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            with patch('permission_notify.get_hook_input', return_value=hook_input):
                with patch('permission_notify.is_mlx_available', return_value=True):
                    with patch('permission_notify.speak_mlx') as mock_mlx:
                        with patch('permission_notify.speak_say') as mock_say:
                            with pytest.raises(SystemExit):
                                main()

                            # Should call speak_mlx, not speak_say
                            mock_mlx.assert_called_once()
                            mock_say.assert_not_called()

    def test_uses_say_when_mlx_unavailable(self, temp_transcript, temp_cooldown_file):
        """Should use macOS say when MLX is not available."""
        from permission_notify import main

        # Setup to trigger notification
        timestamp = (datetime.now(timezone.utc) - timedelta(seconds=40)).isoformat()
        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(create_user_message("Hello", timestamp)) + '\n')

        hook_input = {
            "transcript_path": temp_transcript,
            "message": "Claude wants to use Bash"
        }

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            with patch('permission_notify.get_hook_input', return_value=hook_input):
                with patch('permission_notify.is_mlx_available', return_value=False):
                    with patch('permission_notify.speak_mlx') as mock_mlx:
                        with patch('permission_notify.speak_say') as mock_say:
                            with pytest.raises(SystemExit):
                                main()

                            # Should call speak_say, not speak_mlx
                            mock_say.assert_called_once()
                            mock_mlx.assert_not_called()


# =============================================================================
# TEST: Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_malformed_hook_input_json(self):
        """Should handle malformed JSON input gracefully."""
        from permission_notify import get_hook_input

        with patch('sys.stdin', Mock(read=lambda: "invalid json")):
            with patch('json.load', side_effect=json.JSONDecodeError("error", "doc", 0)):
                result = get_hook_input()
                assert result == {}

    def test_handles_eof_on_stdin(self):
        """Should handle EOF when reading stdin."""
        from permission_notify import get_hook_input

        with patch('json.load', side_effect=EOFError()):
            result = get_hook_input()
            assert result == {}

    def test_expanduser_on_transcript_path(self, temp_cooldown_file):
        """Should expand ~ in transcript paths."""
        from permission_notify import main

        # Use a path with ~ that gets expanded
        hook_input = {
            "transcript_path": "~/nonexistent.jsonl",
            "message": "Claude wants to use Bash"
        }

        with patch('permission_notify.NOTIFY_TIMESTAMP_FILE', temp_cooldown_file):
            with patch('permission_notify.get_hook_input', return_value=hook_input):
                with patch('permission_notify.speak_notification') as mock_speak:
                    # Should handle gracefully (return infinity for time, 0 for count)
                    with pytest.raises(SystemExit) as exc:
                        main()

                    # Should exit 1 but not notify (no valid transcript)
                    assert exc.value.code == 1
                    mock_speak.assert_not_called()

    def test_handles_mixed_timezone_timestamps(self, temp_transcript):
        """Should handle timestamps with different timezone formats."""
        from permission_notify import get_time_since_last_user_message

        # UTC timestamp with Z suffix
        timestamp_z = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat().replace('+00:00', 'Z')
        entry = create_user_message("Hello", timestamp_z)

        with open(temp_transcript, 'w') as f:
            f.write(json.dumps(entry) + '\n')

        result = get_time_since_last_user_message(temp_transcript)
        # Should parse correctly
        assert 9.0 <= result <= 11.0
