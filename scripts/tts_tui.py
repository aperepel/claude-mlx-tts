"""
TTS Configuration TUI.

Textual-based terminal UI for configuring MLX TTS voice settings.

Screens:
- Voice Lab: Per-voice compressor/limiter audio shaping
- System: Streaming interval, server status
- About: Plugin info and help

Run with: uv run python scripts/tts_tui.py
"""
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    LoadingIndicator,
    OptionList,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option
from textual.message import Message
from textual.worker import Worker
from textual import work
from textual_autocomplete import PathAutoComplete, DropdownItem, TargetState
from pathlib import Path
import os
import re


class WavPathAutoComplete(PathAutoComplete):
    """PathAutoComplete for WAV files with tilde expansion support."""

    # File extensions to show (lowercase)
    ALLOWED_EXTENSIONS = {".wav"}

    def _expand_tilde(self, path_str: str) -> str:
        """Expand ~ to home directory path."""
        if path_str.startswith("~"):
            return os.path.expanduser(path_str)
        return path_str

    def get_candidates(self, target_state: TargetState) -> list[DropdownItem]:
        """Get candidates with tilde expansion support."""
        current_input = target_state.text[: target_state.cursor_position]

        # Determine directory based on ORIGINAL input structure (not expanded)
        # This avoids confusion from slashes inside expanded home path
        if "/" in current_input:
            last_slash_index = current_input.rindex("/")
            dir_path = current_input[:last_slash_index] if last_slash_index > 0 else "/"
            # Expand tilde only for actual filesystem access
            directory = Path(os.path.expanduser(dir_path))
        elif current_input == "~" or current_input.startswith("~"):
            # Just ~ or ~partial - show home directory
            directory = Path.home()
        else:
            # Relative path without slashes
            directory = self.path

        # Use the directory path as the cache key
        cache_key = str(directory)
        cached_entries = self._directory_cache.get(cache_key)

        if cached_entries is not None:
            entries = cached_entries
        else:
            try:
                entries = list(os.scandir(directory))
                self._directory_cache[cache_key] = entries
            except OSError:
                return []

        results = []
        for entry in entries:
            completion = entry.name
            if not self.show_dotfiles and completion.startswith("."):
                continue
            if entry.is_dir():
                completion += "/"
                results.append((completion, Path(entry.path)))
            elif Path(entry.name).suffix.lower() in self.ALLOWED_EXTENSIONS:
                # Only include files with allowed extensions
                results.append((completion, Path(entry.path)))

        results.sort(key=lambda x: (not x[1].is_dir(), x[0].lower()))

        return [
            DropdownItem(
                item[0],
                prefix=self.folder_prefix if item[1].is_dir() else self.file_prefix,
            )
            for item in results
        ]

    def get_search_string(self, target_state: TargetState) -> str:
        """Return only the current path segment for searching (with tilde support)."""
        current_input = target_state.text[: target_state.cursor_position]

        # Handle ~ as start of path
        if current_input == "~":
            return ""

        if "/" in current_input:
            last_slash_index = current_input.rindex("/")
            return current_input[last_slash_index + 1 :]
        else:
            return current_input

    def apply_completion(self, value: str, state: TargetState) -> None:
        """Apply completion with proper path construction for tilde paths."""
        current_input = state.text

        # Find last slash in entire input to determine path prefix
        if "/" in current_input:
            last_slash_index = current_input.rindex("/")
            path_prefix = current_input[: last_slash_index + 1]
            new_value = path_prefix + value
        elif current_input.startswith("~"):
            # Input is "~" with no slash - prepend "~/" to preserve home dir
            new_value = "~/" + value
        else:
            # No slashes, no tilde - just use value
            new_value = value

        self.target.value = new_value
        self.target.cursor_position = len(new_value)

    def post_completion(self) -> None:
        """After completion, manually trigger Input.Changed for validation.

        The base _complete() method wraps apply_completion in prevent(Input.Changed),
        so the event never fires. We post it manually here, outside the prevent block.
        """
        super().post_completion()
        # Manually post Input.Changed to trigger validation in CloneLabWidget
        self.target.post_message(Input.Changed(self.target, self.target.value))

    def _handle_focus_change(self, has_focus: bool) -> None:
        """Clear directory cache on focus to pick up filesystem changes."""
        if has_focus:
            self.clear_directory_cache()
        super()._handle_focus_change(has_focus=has_focus)


import tts_config
from tts_config import (
    DEFAULT_COMPRESSOR,
    DEFAULT_LIMITER,
    HOOK_DEFAULT_PROMPTS,
    HOOK_TYPES,
    VOICE_NAME_PATTERN,
    copy_voice,
    delete_voice,
    discover_voices,
    get_active_voice,
    get_compressor_config,
    get_default_hook_prompt,
    get_effective_compressor,
    get_effective_hook_prompt,
    get_effective_limiter,
    get_hook_prompt,
    get_hook_voice,
    get_limiter_config,
    get_streaming_interval,
    get_voice_format,
    get_voice_usage,
    has_voice_defaults,
    rename_voice,
    reset_all_audio_to_defaults,
    set_active_voice,
    set_compressor_setting,
    set_hook_prompt,
    set_hook_voice,
    set_limiter_setting,
    set_streaming_interval,
    set_voice_config,
)

# Compressor presets with tuned values
COMPRESSOR_PRESETS = {
    "Punchy": {
        "threshold_db": -18,
        "ratio": 3.0,
        "attack_ms": 3,
        "release_ms": 50,
        "gain_db": 8,
    },
    "Broadcast": {
        "threshold_db": -24,
        "ratio": 4.0,
        "attack_ms": 5,
        "release_ms": 100,
        "gain_db": 12,
    },
    "Gentle": {
        "threshold_db": -12,
        "ratio": 2.0,
        "attack_ms": 10,
        "release_ms": 150,
        "gain_db": 4,
    },
    "Podcast": {
        "threshold_db": -20,
        "ratio": 3.5,
        "attack_ms": 8,
        "release_ms": 80,
        "gain_db": 10,
    },
}

# Limiter presets
LIMITER_PRESETS = {
    "Transparent": {
        "threshold_db": -1.0,
        "release_ms": 100,
    },
    "Brick Wall": {
        "threshold_db": -0.3,
        "release_ms": 20,
    },
    "Soft Clip": {
        "threshold_db": -3.0,
        "release_ms": 50,
    },
}


def get_voice_display(voice_name: str) -> str:
    """Get voice name with format indicator for display.

    Returns:
        Voice name with format tag: "[embedded]" for safetensors, "[wav]" for wav.
    """
    fmt = get_voice_format(voice_name)
    if fmt == "safetensors":
        return f"{voice_name}  [embedded]"
    elif fmt == "wav":
        return f"{voice_name}  [wav]"
    return voice_name


def get_server_status() -> dict:
    """Get TTS server status, handling import errors gracefully."""
    try:
        from mlx_server_utils import get_server_status as _get_status
        return _get_status()
    except ImportError:
        return {"running": False, "port": 21099, "host": "localhost", "model": "unknown"}


def is_server_alive() -> bool:
    """Check if TTS server is running."""
    try:
        from mlx_server_utils import is_server_alive as _is_alive
        return _is_alive()
    except ImportError:
        return False


class ServerStatusWidget(Static):
    """Widget showing TTS server status."""

    running = reactive(False)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.update_status()

    def update_status(self) -> None:
        """Refresh server status."""
        self.running = is_server_alive()

    def watch_running(self, running: bool) -> None:
        """Update display when running state changes."""
        if running:
            self.update("[green]● Server Running[/]")
        else:
            self.update("[dim]○ Server Stopped[/]")

    def on_mount(self) -> None:
        """Set up periodic status refresh."""
        self.set_interval(5.0, self.update_status)


# Human-readable hook type labels
HOOK_LABELS = {
    "stop": "Stop (completion)",
    "permission_request": "Permission Request",
}

# Prompt field labels per hook type
HOOK_PROMPT_LABELS = {
    "stop": "Attention Grabber:",
    "permission_request": "Phrase:",
}

# Help text per hook type
HOOK_HELP_TEXT = {
    "stop": "Plays before the AI-generated summary",
    "permission_request": "Use {tool_name} for the tool name",
}


class DeleteVoiceModal(ModalScreen):
    """Modal screen for confirming voice deletion."""

    class VoiceDeleteConfirmed(Message):
        """Posted when deletion is confirmed."""

        def __init__(self, voice_name: str) -> None:
            self.voice_name = voice_name
            super().__init__()

    class VoiceDeleteCancelled(Message):
        """Posted when deletion is cancelled."""
        pass

    DEFAULT_CSS = """
    DeleteVoiceModal {
        align: center middle;
    }
    DeleteVoiceModal > Vertical {
        width: 50;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    DeleteVoiceModal .modal-title {
        text-style: bold;
        margin-bottom: 1;
    }
    DeleteVoiceModal .warning-text {
        color: $warning;
        margin: 1 0;
    }
    DeleteVoiceModal .usage-list {
        color: $text-muted;
        margin-left: 2;
    }
    DeleteVoiceModal .button-row {
        width: 100%;
        height: 3;
        margin-top: 1;
        align: center middle;
    }
    DeleteVoiceModal .button-row > Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        voice_name: str,
        usage: dict | None = None,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.voice_name = voice_name
        self.usage = usage or {"is_active": False, "hooks": [], "has_settings": False}

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(f"Delete Voice", classes="modal-title")
            yield Static(f'Delete "{self.voice_name}"?')

            # Only warn about hook usage (active voice warning is pointless)
            if self.usage["hooks"]:
                yield Static("This voice is used by:", classes="warning-text")
                for hook in self.usage["hooks"]:
                    label = HOOK_LABELS.get(hook, hook)
                    yield Static(f"  - {label}", classes="usage-list")
                yield Static("These will revert to the default.", classes="warning-text")

            with Horizontal(classes="button-row"):
                yield Button("Delete", id="delete-btn", variant="error")
                yield Button("Cancel", id="cancel-btn", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self.dismiss(None)
        elif event.button.id == "delete-btn":
            self.dismiss(self.voice_name)


class InputModal(ModalScreen):
    """Modal screen for text input (used for Copy/Rename)."""

    class InputConfirmed(Message):
        """Posted when input is confirmed."""

        def __init__(self, voice_name: str, new_name: str, action_type: str) -> None:
            self.voice_name = voice_name
            self.new_name = new_name
            self.action_type = action_type
            super().__init__()

    class InputCancelled(Message):
        """Posted when input is cancelled."""
        pass

    DEFAULT_CSS = """
    InputModal {
        align: center middle;
    }
    InputModal > Vertical {
        width: 50;
        height: auto;
        padding: 1 2;
        border: thick $primary;
        background: $surface;
    }
    InputModal .modal-title {
        text-style: bold;
        margin-bottom: 1;
    }
    InputModal .input-row {
        width: 100%;
        height: 3;
        margin: 1 0;
    }
    InputModal .input-row > Input {
        width: 100%;
    }
    InputModal .validation-msg {
        height: 1;
        color: $error;
    }
    InputModal .button-row {
        width: 100%;
        height: 3;
        margin-top: 1;
        align: center middle;
    }
    InputModal .button-row > Button {
        margin: 0 1;
    }
    """

    def __init__(
        self,
        title: str,
        voice_name: str,
        action_type: str = "copy",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.title = title
        self.voice_name = voice_name
        self.action_type = action_type
        self._is_valid = False

    def compose(self) -> ComposeResult:
        action_label = "Copy" if self.action_type == "copy" else "Rename"
        with Vertical():
            yield Static(self.title, classes="modal-title")
            yield Static(f'{action_label} "{self.voice_name}" as:')
            with Horizontal(classes="input-row"):
                yield Input(
                    placeholder="new_voice_name",
                    id="name-input",
                    restrict=r"^[a-zA-Z0-9_-]*$",
                )
            yield Static("", id="validation-msg", classes="validation-msg")
            with Horizontal(classes="button-row"):
                yield Button("Cancel", id="cancel-btn", variant="default")
                yield Button(action_label, id="confirm-btn", variant="primary", disabled=True)

    def _validate_name(self, name: str) -> tuple[bool, str]:
        """Validate the input name."""
        if not name.strip():
            return False, ""

        cleaned = name.strip()
        if not VOICE_NAME_PATTERN.match(cleaned):
            return False, "Use only letters, numbers, _, -"

        existing = discover_voices()
        if cleaned in existing:
            return False, f"Voice '{cleaned}' already exists"

        return True, ""

    def validate_name(self, name: str) -> tuple[bool, str]:
        """Public validation method."""
        return self._validate_name(name)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Validate input as user types."""
        if event.input.id == "name-input":
            valid, msg = self._validate_name(event.value)
            self._is_valid = valid
            self.query_one("#validation-msg", Static).update(msg)
            self.query_one("#confirm-btn", Button).disabled = not valid

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel-btn":
            self.dismiss(None)
        elif event.button.id == "confirm-btn":
            name_input = self.query_one("#name-input", Input)
            new_name = name_input.value.strip()
            if self._is_valid:
                self.dismiss((self.voice_name, new_name, self.action_type))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input."""
        if event.input.id == "name-input" and self._is_valid:
            new_name = event.value.strip()
            self.dismiss((self.voice_name, new_name, self.action_type))


class VoiceSelector(Container):
    """Voice selection widget with vertical list and action buttons."""

    class VoiceChanged(Message):
        """Posted when a voice is selected."""

        def __init__(self, voice_name: str) -> None:
            self.voice_name = voice_name
            super().__init__()

    class CopyVoiceRequested(Message):
        """Posted when Copy button is pressed."""

        def __init__(self, voice_name: str) -> None:
            self.voice_name = voice_name
            super().__init__()

    class RenameVoiceRequested(Message):
        """Posted when Rename button is pressed."""

        def __init__(self, voice_name: str) -> None:
            self.voice_name = voice_name
            super().__init__()

    class DeleteVoiceRequested(Message):
        """Posted when Delete button is pressed."""

        def __init__(self, voice_name: str) -> None:
            self.voice_name = voice_name
            super().__init__()

    DEFAULT_CSS = """
    VoiceSelector {
        height: 26;
        width: 32;
        padding: 1;
        border: round $primary;
        border-title-color: $text;
        margin: 1 1 1 0;
    }
    VoiceSelector > OptionList {
        height: 1fr;
        width: 100%;
    }
    VoiceSelector > .action-bar {
        height: 3;
        width: 100%;
        dock: bottom;
        align: center middle;
    }
    VoiceSelector > .action-bar > Button {
        width: auto;
        min-width: 6;
        margin: 0 0;
    }
    """

    def compose(self) -> ComposeResult:
        self.border_title = "Voices"
        voices = discover_voices()

        option_list = OptionList(id="voice-list")
        for voice in voices:
            display_name = get_voice_display(voice)
            option_list.add_option(Option(display_name, id=voice))

        yield option_list

        with Horizontal(classes="action-bar"):
            yield Button("Copy", id="copy-btn", variant="default")
            yield Button("Ren", id="rename-btn", variant="default")
            yield Button("Del", id="delete-btn", variant="warning")

    def on_mount(self) -> None:
        """Highlight the active voice on mount."""
        voices = discover_voices()
        active = get_active_voice()
        if active in voices:
            option_list = self.query_one("#voice-list", OptionList)
            idx = voices.index(active)
            option_list.highlighted = idx
        self._update_delete_button_state()

    def _update_delete_button_state(self) -> None:
        """Enable/disable Del button based on selected voice (cannot delete 'default')."""
        voice_name = self._get_selected_voice()
        delete_btn = self.query_one("#delete-btn", Button)
        delete_btn.disabled = voice_name == "default"

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Update Del button state when highlighted voice changes."""
        self._update_delete_button_state()

    def _get_selected_voice(self) -> str | None:
        """Get the currently highlighted voice name."""
        option_list = self.query_one("#voice-list", OptionList)
        if option_list.highlighted is not None:
            option = option_list.get_option_at_index(option_list.highlighted)
            return str(option.id)
        return None

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle voice selection."""
        voice_name = str(event.option.id)

        # Skip if clicking the same voice (no change)
        if voice_name == get_active_voice():
            return

        try:
            set_active_voice(voice_name)
            self.post_message(self.VoiceChanged(voice_name))
        except ValueError as e:
            self.notify(str(e), severity="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle action button presses."""
        voice_name = self._get_selected_voice()
        if not voice_name:
            self.notify("No voice selected", severity="warning")
            return

        if event.button.id == "copy-btn":
            self.post_message(self.CopyVoiceRequested(voice_name))
        elif event.button.id == "rename-btn":
            self.post_message(self.RenameVoiceRequested(voice_name))
        elif event.button.id == "delete-btn":
            self.post_message(self.DeleteVoiceRequested(voice_name))

    def refresh_voices(self) -> None:
        """Refresh the voice list after CRUD operations."""
        option_list = self.query_one("#voice-list", OptionList)
        option_list.clear_options()

        voices = discover_voices()
        for voice in voices:
            display_name = get_voice_display(voice)
            option_list.add_option(Option(display_name, id=voice))

        active = get_active_voice()
        if active in voices:
            idx = voices.index(active)
            option_list.highlighted = idx

        self._update_delete_button_state()

    def select_voice(self, voice_name: str) -> None:
        """Select a voice, update config, and fire VoiceChanged event."""
        voices = discover_voices()
        if voice_name not in voices:
            return

        option_list = self.query_one("#voice-list", OptionList)
        idx = voices.index(voice_name)
        option_list.highlighted = idx

        set_active_voice(voice_name)
        self.post_message(self.VoiceChanged(voice_name))
        self._update_delete_button_state()


class FormField(Horizontal):
    """Form field with label, numeric input, and unit suffix."""

    DEFAULT_CSS = """
    FormField {
        height: 3;
        width: auto;
        padding: 0;
    }
    FormField > .label {
        width: 14;
        padding: 1 1 1 0;
    }
    FormField > .label.highlight {
        color: $accent;
    }
    FormField > Input {
        width: 12;
    }
    FormField > .unit {
        width: auto;
        padding: 1 0 1 1;
        color: $text-muted;
    }
    """

    class Changed(Message):
        """Posted when field value changes."""
        def __init__(self, field: "FormField", value: float) -> None:
            self.field = field
            self.value = value
            super().__init__()

    def __init__(
        self,
        label: str,
        value: float,
        min_val: float,
        max_val: float,
        unit: str = "",
        param_key: str = "",
        highlight: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = value
        self._min = min_val
        self._max = max_val
        self._unit = unit
        self._param_key = param_key
        self._highlight = highlight

    def compose(self) -> ComposeResult:
        label_classes = "label highlight" if self._highlight else "label"
        yield Label(f"{self._label}:", classes=label_classes)
        # Restrict to digits, minus, decimal point (no scientific notation)
        yield Input(
            value=self._format_value(),
            id="field-input",
            restrict=r"^-?[0-9]*\.?[0-9]*$",
        )
        yield Label(self._unit, classes="unit")

    def _format_value(self) -> str:
        """Format the current value for display (number only)."""
        if self._value == int(self._value):
            return str(int(self._value))
        return f"{self._value:.1f}"

    def _update_display(self) -> None:
        """Update the input display."""
        inp = self.query_one("#field-input", Input)
        inp.value = self._format_value()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle value input on Enter."""
        event.stop()
        self._apply_input_value(event.value)

    def on_input_blurred(self, event: Input.Blurred) -> None:
        """Handle value input on focus loss from Input widget."""
        self._apply_input_value(event.value)

    def _apply_input_value(self, raw_value: str) -> None:
        """Parse and apply a new value from input."""
        try:
            new_val = float(raw_value.strip())
            new_val = max(self._min, min(self._max, new_val))
            if new_val != self._value:
                self._value = new_val
                self._update_display()
                self.post_message(self.Changed(self, self._value))
        except ValueError:
            self._update_display()

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, val: float) -> None:
        self._value = max(self._min, min(self._max, val))
        if self.is_mounted:
            self._update_display()

    @property
    def param_key(self) -> str:
        return self._param_key


class CompressorWidget(Container):
    """Compressor settings form."""

    DEFAULT_CSS = """
    CompressorWidget {
        height: 26;
        padding: 1 2;
        border: round $primary;
        border-title-color: $text;
        margin: 1 1 1 0;
        width: 44;
    }
    CompressorWidget > .header-row {
        height: 3;
        width: 100%;
    }
    CompressorWidget > .header-row > Switch {
        width: auto;
    }
    CompressorWidget > .header-row > Select {
        width: 22;
    }
    CompressorWidget > .form-fields {
        height: auto;
        width: 100%;
        padding: 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = get_compressor_config()
        self._initialized = False

    def on_mount(self) -> None:
        """Mark widget as initialized after mount to prevent startup notifications."""
        self._initialized = True

    def compose(self) -> ComposeResult:
        self.border_title = "Compressor"
        config = self._config

        # Header with enable toggle and preset selector
        with Horizontal(classes="header-row"):
            yield Switch(value=config.get("enabled", True), id="compressor-enabled")
            yield Select(
                [("Custom", "custom")] + [(name, name) for name in COMPRESSOR_PRESETS],
                value="custom",
                id="preset-select",
            )

        # Form fields
        with Vertical(classes="form-fields"):
            yield FormField(
                "Input Gain", config.get("input_gain_db", 0.0),
                min_val=-12, max_val=12, unit="dB",
                param_key="input_gain_db", id="input-gain-field",
                highlight=True,
            )
            yield FormField(
                "Threshold", config.get("threshold_db", -18),
                min_val=-40, max_val=0, unit="dB",
                param_key="threshold_db", id="threshold-field",
            )
            yield FormField(
                "Ratio", config.get("ratio", 3.0),
                min_val=1.0, max_val=20.0, unit=":1",
                param_key="ratio", id="ratio-field",
            )
            yield FormField(
                "Attack", config.get("attack_ms", 3),
                min_val=1, max_val=50, unit="ms",
                param_key="attack_ms", id="attack-field",
            )
            yield FormField(
                "Release", config.get("release_ms", 50),
                min_val=10, max_val=500, unit="ms",
                param_key="release_ms", id="release-field",
            )
            yield FormField(
                "Makeup Gain", config.get("gain_db", 8),
                min_val=0, max_val=24, unit="dB",
                param_key="gain_db", id="gain-field",
                highlight=True,
            )

    def on_form_field_changed(self, event: FormField.Changed) -> None:
        """Handle field value changes."""
        param_key = event.field.param_key
        if param_key:
            try:
                set_compressor_setting(param_key, event.value)
                if self._initialized:
                    self.notify("Voice settings saved", severity="information")
            except ValueError as e:
                self.notify(str(e), severity="error")

    def _refresh_fields(self) -> None:
        """Refresh all field values from config."""
        config = get_compressor_config()
        self.query_one("#compressor-enabled", Switch).value = config.get("enabled", True)
        self.query_one("#input-gain-field", FormField).value = config.get("input_gain_db", 0.0)
        self.query_one("#threshold-field", FormField).value = config.get("threshold_db", -18)
        self.query_one("#ratio-field", FormField).value = config.get("ratio", 3.0)
        self.query_one("#attack-field", FormField).value = config.get("attack_ms", 3)
        self.query_one("#release-field", FormField).value = config.get("release_ms", 50)
        self.query_one("#gain-field", FormField).value = config.get("gain_db", 8)
        self.query_one("#preset-select", Select).value = "custom"

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle compressor enable toggle."""
        if event.switch.id == "compressor-enabled":
            try:
                set_compressor_setting("enabled", event.value)
                if self._initialized:
                    self.notify("Voice settings saved", severity="information")
            except ValueError as e:
                self.notify(str(e), severity="error")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle preset selection."""
        if event.select.id == "preset-select" and event.value != "custom":
            preset = COMPRESSOR_PRESETS.get(event.value)
            if preset:
                self._apply_preset(preset)
                if self._initialized:
                    self.notify("Voice settings saved", severity="information")

    def _apply_preset(self, preset: dict) -> None:
        """Apply a compressor preset."""
        for key, value in preset.items():
            try:
                set_compressor_setting(key, value)
            except ValueError:
                pass

        # Update form fields
        self.query_one("#threshold-field", FormField).value = preset["threshold_db"]
        self.query_one("#ratio-field", FormField).value = preset["ratio"]
        self.query_one("#attack-field", FormField).value = preset["attack_ms"]
        self.query_one("#release-field", FormField).value = preset["release_ms"]
        self.query_one("#gain-field", FormField).value = preset["gain_db"]


class LimiterWidget(Container):
    """Limiter settings form."""

    DEFAULT_CSS = """
    LimiterWidget {
        height: 26;
        padding: 1 2;
        border: round $primary;
        border-title-color: $text;
        margin: 1 0 1 1;
        width: 44;
    }
    LimiterWidget > .header-row {
        height: 3;
        width: 100%;
    }
    LimiterWidget > .header-row > Switch {
        width: auto;
    }
    LimiterWidget > .header-row > Select {
        width: 22;
    }
    LimiterWidget > .form-fields {
        height: auto;
        width: 100%;
        padding: 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = get_limiter_config()
        self._initialized = False

    def on_mount(self) -> None:
        """Mark widget as initialized after mount to prevent startup notifications."""
        self._initialized = True

    def compose(self) -> ComposeResult:
        self.border_title = "Limiter"
        config = self._config

        # Header with enable toggle and preset selector
        with Horizontal(classes="header-row"):
            yield Switch(value=config.get("enabled", True), id="limiter-enabled")
            yield Select(
                [("Custom", "custom")] + [(name, name) for name in LIMITER_PRESETS],
                value="custom",
                id="limiter-preset-select",
            )

        # Form fields
        with Vertical(classes="form-fields"):
            yield FormField(
                "Threshold", config.get("threshold_db", -0.5),
                min_val=-20, max_val=0, unit="dB",
                param_key="threshold_db", id="limiter-threshold-field",
            )
            yield FormField(
                "Release", config.get("release_ms", 40),
                min_val=10, max_val=200, unit="ms",
                param_key="release_ms", id="limiter-release-field",
            )
            # Master gain is stored in compressor config but displayed here
            # as it's the final output stage after limiter
            compressor_config = get_compressor_config()
            yield FormField(
                "Master Gain", compressor_config.get("master_gain_db", 0.0),
                min_val=-12, max_val=12, unit="dB",
                param_key="master_gain_db", id="master-gain-field",
                highlight=True,
            )

    def on_form_field_changed(self, event: FormField.Changed) -> None:
        """Handle field value changes."""
        param_key = event.field.param_key
        if param_key:
            try:
                if param_key == "master_gain_db":
                    # Master gain is stored in compressor config
                    set_compressor_setting(param_key, event.value)
                else:
                    set_limiter_setting(param_key, event.value)
                if self._initialized:
                    self.notify("Voice settings saved", severity="information")
            except ValueError as e:
                self.notify(str(e), severity="error")

    def _refresh_fields(self) -> None:
        """Refresh all field values from config."""
        config = get_limiter_config()
        compressor_config = get_compressor_config()
        self.query_one("#limiter-enabled", Switch).value = config.get("enabled", True)
        self.query_one("#limiter-threshold-field", FormField).value = config.get("threshold_db", -0.5)
        self.query_one("#limiter-release-field", FormField).value = config.get("release_ms", 40)
        self.query_one("#master-gain-field", FormField).value = compressor_config.get("master_gain_db", 0.0)
        self.query_one("#limiter-preset-select", Select).value = "custom"

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle limiter enable toggle."""
        if event.switch.id == "limiter-enabled":
            try:
                set_limiter_setting("enabled", event.value)
                if self._initialized:
                    self.notify("Voice settings saved", severity="information")
            except ValueError as e:
                self.notify(str(e), severity="error")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle preset selection."""
        if event.select.id == "limiter-preset-select" and event.value != "custom":
            preset = LIMITER_PRESETS.get(event.value)
            if preset:
                self._apply_preset(preset)
                if self._initialized:
                    self.notify("Voice settings saved", severity="information")

    def _apply_preset(self, preset: dict) -> None:
        """Apply a limiter preset."""
        for key, value in preset.items():
            try:
                set_limiter_setting(key, value)
            except ValueError:
                pass

        # Update form fields
        self.query_one("#limiter-threshold-field", FormField).value = preset["threshold_db"]
        self.query_one("#limiter-release-field", FormField).value = preset["release_ms"]


class HookSettingsWidget(Container):
    """Widget for configuring voice and prompt for a specific hook type."""

    DEFAULT_CSS = """
    HookSettingsWidget {
        height: auto;
        width: 100%;
        border: round $accent;
        border-title-color: $text;
        padding: 1;
        margin-bottom: 1;
    }
    HookSettingsWidget > .hook-row {
        height: 3;
        width: 100%;
    }
    HookSettingsWidget > .hook-row > .hook-label {
        width: 20;
        padding: 1 1 1 0;
    }
    HookSettingsWidget > .hook-row > Select {
        width: 30;
    }
    HookSettingsWidget > .hook-row > Input {
        width: 1fr;
    }
    HookSettingsWidget > .help-row {
        height: auto;
        width: 100%;
        align: left middle;
        padding: 1 0;
    }
    HookSettingsWidget > .help-row > .help-text {
        color: $text-muted;
        width: 1fr;
        padding: 1 0 0 0;
    }
    HookSettingsWidget > .help-row > Button {
        margin-left: 1;
    }
    """

    is_playing = reactive(False)

    # Debounce delay for saving prompts (seconds)
    PROMPT_SAVE_DELAY = 0.5

    def __init__(self, hook_type: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hook_type = hook_type
        self._initialized = False
        self._save_timer = None
        self._refreshing = False  # Flag to prevent saves during refresh

    def on_mount(self) -> None:
        """Schedule initialization flag after initial events are processed."""
        self.call_later(self._mark_initialized)

    def _mark_initialized(self) -> None:
        """Mark widget as initialized - called after initial Changed events."""
        self._initialized = True

    def refresh_voices(self) -> None:
        """Refresh the voice dropdown after voice CRUD operations."""
        # Prevent on_select_changed from saving during refresh
        # Note: Events are queued, so we use call_later to reset after they process
        self._refreshing = True

        voices = discover_voices()
        current_voice = get_hook_voice(self._hook_type)

        # Build voice options: "-- inherit --" (uses active voice) + all available voices
        voice_options = [("-- inherit --", "")]
        voice_options.extend((voice, voice) for voice in voices)

        # Update the Select widget
        voice_select = self.query_one(f"#hook-voice-{self._hook_type}", Select)
        voice_select.set_options(voice_options)

        # Restore selection (empty string = inherit, or specific voice if still exists)
        if current_voice and current_voice in voices:
            voice_select.value = current_voice
        else:
            voice_select.value = ""

        # Reset flag after queued events are processed
        self.call_later(self._clear_refreshing)

    def _clear_refreshing(self) -> None:
        """Clear the refreshing flag after events are processed."""
        self._refreshing = False

    def compose(self) -> ComposeResult:
        self.border_title = HOOK_LABELS.get(self._hook_type, self._hook_type)

        voices = discover_voices()
        current_voice = get_hook_voice(self._hook_type)
        current_prompt = get_hook_prompt(self._hook_type)
        default_prompt = get_default_hook_prompt(self._hook_type)

        # Build voice options: "-- inherit --" (uses active voice) + all available voices
        voice_options = [("-- inherit --", "")]
        voice_options.extend((voice, voice) for voice in voices)

        # Voice row
        with Horizontal(classes="hook-row"):
            yield Label("Voice:", classes="hook-label")
            yield Select(
                voice_options,
                value=current_voice if current_voice else "",
                id=f"hook-voice-{self._hook_type}",
            )

        # Prompt row
        prompt_label = HOOK_PROMPT_LABELS.get(self._hook_type, "Prompt:")
        with Horizontal(classes="hook-row"):
            yield Label(prompt_label, classes="hook-label")
            yield Input(
                value=current_prompt if current_prompt else default_prompt,
                placeholder=default_prompt,
                id=f"hook-prompt-{self._hook_type}",
            )

        # Help text and buttons row
        help_text = HOOK_HELP_TEXT.get(self._hook_type, "")
        with Horizontal(classes="help-row"):
            yield Static(help_text, classes="help-text")
            yield Button("Play", id=f"hook-play-{self._hook_type}", variant="primary")
            yield Button("Reset", id=f"hook-reset-{self._hook_type}", variant="default")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle voice selection change."""
        if event.select.id == f"hook-voice-{self._hook_type}":
            # Skip saves during initialization or refresh
            if not self._initialized or self._refreshing:
                return

            # Skip if value is not a string (e.g., Select.BLANK sentinel)
            if not isinstance(event.value, str):
                return

            # Empty string means inherit from active voice
            new_voice = event.value if event.value else None

            # Skip if value hasn't actually changed (clicking same item)
            current_voice = get_hook_voice(self._hook_type)
            if new_voice == current_voice:
                return

            try:
                set_hook_voice(self._hook_type, new_voice)
            except ValueError as e:
                self.notify(str(e), severity="error")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle prompt input change with debouncing."""
        if event.input.id == f"hook-prompt-{self._hook_type}":
            # Ignore initial population of input fields
            if not self._initialized:
                return

            # Cancel any pending save
            if self._save_timer is not None:
                self._save_timer.stop()

            # Schedule debounced save
            self._save_timer = self.set_timer(
                self.PROMPT_SAVE_DELAY,
                lambda: self._save_prompt(event.value)
            )

    def _save_prompt(self, value: str) -> None:
        """Actually save the prompt after debounce delay."""
        prompt = value.strip()
        default_prompt = get_default_hook_prompt(self._hook_type)

        # Get current value to check if actually changed
        current_prompt = get_hook_prompt(self._hook_type)
        current_effective = current_prompt if current_prompt else default_prompt

        # Skip if value hasn't changed
        if prompt == current_effective:
            return

        # If empty or matches default, clear custom prompt
        if not prompt or prompt == default_prompt:
            set_hook_prompt(self._hook_type, None)
        else:
            set_hook_prompt(self._hook_type, prompt)
        # No notification - auto-save is silent

    def watch_is_playing(self, playing: bool) -> None:
        """Update Play button based on playing state."""
        btn = self.query_one(f"#hook-play-{self._hook_type}", Button)
        if playing:
            btn.variant = "default"
            btn.label = "Playing..."
        else:
            btn.variant = "primary"
            btn.label = "Play"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == f"hook-play-{self._hook_type}":
            self._play_prompt()
        elif event.button.id == f"hook-reset-{self._hook_type}":
            default_prompt = get_default_hook_prompt(self._hook_type)
            set_hook_prompt(self._hook_type, None)

            # Update the input field to show the default
            prompt_input = self.query_one(f"#hook-prompt-{self._hook_type}", Input)
            prompt_input.value = default_prompt

            self.notify("Prompt reset to default", severity="information")

    def _play_prompt(self) -> None:
        """Play the current prompt using TTS."""
        if self.is_playing:
            return

        self.is_playing = True
        self.app.set_focus(None)
        self._run_hook_tts_worker()

    @work(exclusive=True, thread=True)
    def _run_hook_tts_worker(self) -> dict:
        """Background worker for hook prompt TTS playback."""
        import subprocess
        import sys
        import os

        # Get current prompt text from input
        prompt_input = self.query_one(f"#hook-prompt-{self._hook_type}", Input)
        prompt_text = prompt_input.value

        # For permission_request, substitute placeholder with sample tool name
        if self._hook_type == "permission_request":
            prompt_text = prompt_text.format(tool_name="Bash")

        # Get effective voice for this hook
        voice = get_hook_voice(self._hook_type)
        if not voice:
            voice = get_active_voice()

        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        tts_script = os.path.join(scripts_dir, "mlx_server_utils.py")

        try:
            # Pass voice as second argument
            result = subprocess.run(
                [sys.executable, tts_script, prompt_text, "--voice", voice],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {"success": result.returncode == 0, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "TTS timed out after 60s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle TTS worker completion."""
        from textual.worker import WorkerState

        if event.worker.name != "_run_hook_tts_worker":
            return

        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            self.is_playing = False
            if not result.get("success"):
                error = result.get("error") or result.get("stderr", "")[:100]
                self.notify(f"TTS failed: {error}", severity="error")
        elif event.state in (WorkerState.ERROR, WorkerState.CANCELLED):
            self.is_playing = False

        self.app.set_focus(None)


# Default preview phrases
PREVIEW_PHRASES = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello! Testing one, two, three.",
    "Claude is ready to help with your tasks.",
]
DEFAULT_PREVIEW_PHRASE = PREVIEW_PHRASES[0]


class PreviewWidget(Container):
    """Voice preview widget with phrase display and play button."""

    DEFAULT_CSS = """
    PreviewWidget {
        height: auto;
        padding: 1 2;
        border: solid $secondary;
        margin: 1 0;
        width: 100%;
    }
    PreviewWidget > .header-row {
        height: 3;
        width: 100%;
        align: center middle;
    }
    PreviewWidget > .header-row > Button {
        width: auto;
        margin: 0 1;
    }
    PreviewWidget > .phrase-display {
        height: auto;
        padding: 1;
        margin: 1 0;
        background: $surface-darken-1;
        border: round $primary-darken-2;
    }
    PreviewWidget > .status-row {
        height: 3;
        width: 100%;
        align: center middle;
    }
    PreviewWidget > .status-row > LoadingIndicator {
        width: auto;
    }
    PreviewWidget > .status-row > .status-text {
        width: auto;
        padding: 0 1;
        color: $text-muted;
    }
    """

    is_playing = reactive(False)
    status_message = reactive("")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._preview_phrase = DEFAULT_PREVIEW_PHRASE

    def compose(self) -> ComposeResult:
        with Horizontal(classes="header-row"):
            yield Button("Play", id="play-preview-btn", variant="primary")
            yield Button("Reset", id="reset-audio-btn", variant="warning")

        yield Static(f'"{self._preview_phrase}"', id="phrase-display", classes="phrase-display")

        with Horizontal(classes="status-row"):
            yield LoadingIndicator(id="loading-indicator")
            yield Static("", id="status-text", classes="status-text")

    def on_mount(self) -> None:
        """Hide loading indicator initially."""
        self.query_one("#loading-indicator").display = False

    def watch_is_playing(self, playing: bool) -> None:
        """Update UI based on playing state.

        Note: We deliberately don't disable the button because doing so
        causes focus to shift to the next focusable widget (an Input field).
        If the user then presses a number key to switch tabs, the keystroke
        is captured by the Input instead of triggering the tab binding.
        """
        btn = self.query_one("#play-preview-btn", Button)
        loader = self.query_one("#loading-indicator")

        if playing:
            btn.variant = "default"
            btn.label = "Playing..."
            loader.display = True
        else:
            btn.variant = "primary"
            btn.label = "Play"
            loader.display = False

    def watch_status_message(self, message: str) -> None:
        """Update status text."""
        status = self.query_one("#status-text", Static)
        status.update(message)

    class AudioReset(Message):
        """Posted when audio settings are reset."""
        pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "play-preview-btn":
            self._play_preview()
        elif event.button.id == "reset-audio-btn":
            # Check if voice has captured defaults before reset
            voice_name = get_active_voice()
            had_captured = has_voice_defaults(voice_name)

            reset_all_audio_to_defaults()
            self.post_message(self.AudioReset())
            self.app.set_focus(None)

            # Show appropriate message
            if had_captured:
                self.notify(f"Reset to captured defaults", severity="information")
            else:
                self.notify(f"Reset to factory defaults", severity="information")

    def _play_preview(self) -> None:
        """Play the preview phrase using TTS."""
        if self.is_playing:
            return

        self.is_playing = True
        # Blur focus to prevent button state changes from interfering with key events
        self.app.set_focus(None)
        self._run_tts_worker()

    @work(exclusive=True, thread=True)
    def _run_tts_worker(self) -> dict:
        """Background worker for TTS playback.

        Uses Textual's worker system instead of manual threading for better
        integration with the event loop. Returns result dict for on_worker_state_changed.
        """
        import subprocess
        import sys
        import os

        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        tts_script = os.path.join(scripts_dir, "mlx_server_utils.py")

        try:
            result = subprocess.run(
                [sys.executable, tts_script, self._preview_phrase],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {"success": result.returncode == 0, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "TTS timed out after 60s"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle TTS worker completion."""
        from textual.worker import WorkerState

        if event.worker.name != "_run_tts_worker":
            return

        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            self.is_playing = False
            if result.get("success"):
                self.status_message = ""
            else:
                error = result.get("error") or result.get("stderr", "")[:100]
                self.status_message = "Error"
                self.notify(f"TTS failed: {error}", severity="error")
        elif event.state in (WorkerState.ERROR, WorkerState.CANCELLED):
            self.is_playing = False
            self.status_message = "Error"

        # Ensure focus is cleared so key bindings work
        self.app.set_focus(None)

    @property
    def preview_phrase(self) -> str:
        return self._preview_phrase

    @preview_phrase.setter
    def preview_phrase(self, phrase: str) -> None:
        self._preview_phrase = phrase
        if self.is_mounted:
            display = self.query_one("#phrase-display", Static)
            display.update(f'"{phrase}"')


# Voice name validation pattern
VOICE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class CloneLabWidget(Container):
    """Widget for voice cloning workflow."""

    DEFAULT_CSS = """
    CloneLabWidget {
        height: auto;
        padding: 1 2;
        width: 100%;
    }
    CloneLabWidget > .clone-form {
        height: auto;
        width: 100%;
        padding: 1 0;
    }
    CloneLabWidget > .clone-form > .form-row {
        height: 3;
        width: 100%;
        margin: 0 0 1 0;
    }
    CloneLabWidget > .clone-form > .form-row > .field-label {
        width: 14;
        padding: 1 1 1 0;
    }
    CloneLabWidget > .clone-form > .form-row > Input {
        width: 50;
    }
    CloneLabWidget > .clone-form > .form-row > .validation-msg {
        width: auto;
        padding: 1 0 1 2;
    }
    CloneLabWidget > .clone-form > .button-row {
        height: 3;
        width: 100%;
        margin: 1 0;
        align: center middle;
    }
    CloneLabWidget > .clone-form > .button-row > Button {
        width: auto;
        margin: 0 1 0 0;
    }
    CloneLabWidget > .status-section {
        height: auto;
        width: 100%;
        padding: 1;
        margin: 1 0;
        border: round $primary-darken-2;
    }
    CloneLabWidget > .success-section {
        height: auto;
        width: 100%;
        padding: 1 2;
        margin: 1 0;
        border: round green;
        background: $success-darken-3;
    }
    CloneLabWidget > .success-section > .success-header {
        text-style: bold;
        color: $success;
        margin: 0 0 0 0;
    }
    CloneLabWidget > .success-section > .success-stats {
        color: $text-muted;
        margin: 0 0 1 0;
    }
    CloneLabWidget > .success-section > .test-row {
        height: 3;
        width: 100%;
        margin: 1 0;
    }
    CloneLabWidget > .success-section > .test-row > Input {
        width: 1fr;
    }
    CloneLabWidget > .success-section > .test-row > Button {
        width: auto;
        margin: 0 0 0 1;
    }
    CloneLabWidget > .success-section > .button-row {
        height: 3;
        width: 100%;
        margin: 1 0 0 0;
        align: center middle;
    }
    CloneLabWidget > .failure-section {
        height: auto;
        width: 100%;
        padding: 1 2;
        margin: 1 0;
        border: round red;
        background: $error-darken-3;
    }
    CloneLabWidget > .failure-section > .failure-header {
        text-style: bold;
        color: $error;
        margin: 0 0 0 0;
    }
    CloneLabWidget > .failure-section > .failure-message {
        color: $text;
        margin: 0 0 1 0;
    }
    CloneLabWidget > .failure-section > .button-row {
        height: 3;
        width: 100%;
        margin: 1 0 0 0;
        align: center middle;
    }
    """

    class VoiceCloned(Message):
        """Posted when a voice is successfully cloned."""

        def __init__(self, voice_name: str) -> None:
            self.voice_name = voice_name
            super().__init__()

    # State tracking
    is_cloning = reactive(False)
    clone_success = reactive(False)
    cloned_voice_name = reactive("")
    cloned_size_info = reactive("")

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._wav_valid = False
        self._name_valid = False

    def compose(self) -> ComposeResult:
        with Vertical(classes="clone-form"):
            with Horizontal(classes="form-row"):
                yield Label("Source WAV:", classes="field-label")
                wav_input = Input(
                    placeholder="/path/to/voice.wav",
                    id="wav-path-input",
                )
                yield wav_input
                yield WavPathAutoComplete(wav_input, id="wav-path-autocomplete")
                yield Label("", id="wav-validation", classes="validation-msg")

            with Horizontal(classes="form-row"):
                yield Label("Voice Name:", classes="field-label")
                yield Input(
                    placeholder="my_new_voice",
                    id="voice-name-input",
                    restrict=r"^[a-zA-Z0-9_-]*$",
                )
                yield Label("", id="name-validation", classes="validation-msg")

            with Horizontal(classes="button-row"):
                yield Button("Clone Voice", id="clone-btn", variant="primary", disabled=True)
                yield LoadingIndicator(id="clone-loading")

            yield Static("", id="status-msg", classes="status-section")

        with Vertical(classes="success-section", id="success-section"):
            yield Static("", id="success-header", classes="success-header")
            yield Static("", id="success-stats", classes="success-stats")
            with Horizontal(classes="test-row"):
                yield Input(
                    value="Hello, this is my new voice!",
                    id="test-phrase-input",
                )
                yield Button("Speak", id="speak-btn", variant="success")
            with Horizontal(classes="button-row"):
                yield Button("Clone Another", id="clone-another-btn", variant="default")

        with Vertical(classes="failure-section", id="failure-section"):
            yield Static("", id="failure-header", classes="failure-header")
            yield Static("", id="failure-message", classes="failure-message")
            with Horizontal(classes="button-row"):
                yield Button("Retry", id="retry-btn", variant="warning")

    def on_mount(self) -> None:
        """Hide elements initially."""
        self.query_one("#clone-loading").display = False
        self.query_one("#status-msg").display = False
        self.query_one("#success-section").display = False
        self.query_one("#failure-section").display = False

    def _validate_wav_path(self, path: str) -> tuple[bool, str]:
        """Validate WAV file path."""
        if not path.strip():
            return False, ""

        # Expand ~ to home directory before validation
        wav_path = Path(os.path.expanduser(path.strip()))
        if not wav_path.exists():
            return False, "[red]File not found[/]"
        if wav_path.suffix.lower() != ".wav":
            return False, "[red]Not a WAV file[/]"

        # Check file size as a proxy for duration
        size_kb = wav_path.stat().st_size / 1024
        if size_kb < 50:
            return False, "[yellow]File may be too short[/]"

        return True, "[green]✓[/]"

    def _validate_voice_name(self, name: str) -> tuple[bool, str]:
        """Validate voice name."""
        if not name.strip():
            return False, ""

        cleaned = name.strip()
        if not VOICE_NAME_PATTERN.match(cleaned):
            return False, "[red]Use only letters, numbers, _, -[/]"

        # Check if voice already exists
        from tts_config import discover_voices
        existing = discover_voices()
        if cleaned in existing:
            return False, "[yellow]Voice exists (will overwrite)[/]"

        return True, "[green]✓[/]"

    def _update_clone_button(self) -> None:
        """Enable/disable clone button based on validation state."""
        btn = self.query_one("#clone-btn", Button)
        btn.disabled = not (self._wav_valid and self._name_valid) or self.is_cloning

    def on_input_changed(self, event: Input.Changed) -> None:
        """Validate inputs as user types."""
        if event.input.id == "wav-path-input":
            valid, msg = self._validate_wav_path(event.value)
            self._wav_valid = valid
            self.query_one("#wav-validation", Label).update(msg)
            self._update_clone_button()
            # Hide autocomplete when valid file is selected
            if valid:
                self.query_one("#wav-path-autocomplete", WavPathAutoComplete).action_hide()
        elif event.input.id == "voice-name-input":
            valid, msg = self._validate_voice_name(event.value)
            self._name_valid = valid
            self.query_one("#name-validation", Label).update(msg)
            self._update_clone_button()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "clone-btn":
            self._start_cloning()
        elif event.button.id == "clone-another-btn":
            self._reset_form()
        elif event.button.id == "speak-btn":
            self._test_voice()
        elif event.button.id == "retry-btn":
            self._retry_clone()

    def _start_cloning(self) -> None:
        """Start the voice cloning process."""
        if self.is_cloning:
            return

        wav_path = os.path.expanduser(self.query_one("#wav-path-input", Input).value.strip())
        voice_name = self.query_one("#voice-name-input", Input).value.strip()

        self.is_cloning = True
        self.query_one("#clone-btn", Button).display = False
        self.query_one("#clone-loading").display = True
        self.query_one("#status-msg").update("Cloning voice... (loading model)")
        self.query_one("#status-msg").display = True

        self._run_clone_worker(wav_path, voice_name)

    @work(exclusive=True, thread=True)
    def _run_clone_worker(self, wav_path: str, voice_name: str) -> dict:
        """Background worker for voice cloning."""
        import subprocess
        import sys
        import os

        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        clone_script = os.path.join(scripts_dir, "clone_voice.py")

        try:
            # Run clone_voice.py with input "y" to confirm overwrite if needed
            result = subprocess.run(
                [sys.executable, clone_script, wav_path, voice_name],
                capture_output=True,
                text=True,
                input="y\n",
                timeout=300,  # 5 minute timeout
            )

            if result.returncode == 0:
                # Parse output for size info
                output = result.stdout
                return {
                    "success": True,
                    "voice_name": voice_name,
                    "output": output,
                }
            else:
                return {
                    "success": False,
                    "error": result.stderr or result.stdout or "Unknown error",
                }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Cloning timed out after 5 minutes"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle clone worker completion."""
        from textual.worker import WorkerState

        if event.worker.name != "_run_clone_worker":
            return

        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            self.is_cloning = False
            self.query_one("#clone-loading").display = False

            if result.get("success"):
                voice_name = result.get("voice_name", "")
                output = result.get("output", "")

                # Parse size info from output
                size_info = ""
                for line in output.split("\n"):
                    if "Compression:" in line or "Input:" in line or "Output:" in line:
                        size_info += line.strip() + " "

                self.cloned_voice_name = voice_name
                self.cloned_size_info = size_info.strip()
                self.clone_success = True

                # Update UI - show success, hide failure
                self.query_one("#status-msg").display = False
                self.query_one("#failure-section").display = False
                self.query_one("#success-section").display = True
                self.query_one("#success-header", Static).update(
                    f"✓ Voice Created: {voice_name}"
                )
                stats_text = self.cloned_size_info if self.cloned_size_info else "Voice cloned successfully"
                self.query_one("#success-stats", Static).update(f"📊 {stats_text}")

                # Post message for voice selector refresh
                self.post_message(self.VoiceCloned(voice_name))
            else:
                error = result.get("error", "Unknown error")
                self._show_failure(error)

        elif event.state in (WorkerState.ERROR, WorkerState.CANCELLED):
            self.is_cloning = False
            self.query_one("#clone-loading").display = False
            self._show_failure("Cloning failed unexpectedly")

    def _show_failure(self, error: str) -> None:
        """Show failure panel with error message."""
        self.query_one("#status-msg").display = False
        self.query_one("#success-section").display = False
        self.query_one("#failure-section").display = True
        self.query_one("#failure-header", Static).update("✗ Clone Failed")
        self.query_one("#failure-message", Static).update(error)
        self._update_clone_button()

    def _reset_form(self) -> None:
        """Reset form for another clone."""
        self.clone_success = False
        self.cloned_voice_name = ""
        self.cloned_size_info = ""

        self.query_one("#success-section").display = False
        self.query_one("#failure-section").display = False
        self.query_one("#clone-btn", Button).display = True
        self.query_one("#wav-path-input", Input).value = ""
        self.query_one("#voice-name-input", Input).value = ""
        self.query_one("#wav-validation", Label).update("")
        self.query_one("#name-validation", Label).update("")
        self._wav_valid = False
        self._name_valid = False
        self._update_clone_button()

    def _retry_clone(self) -> None:
        """Hide failure panel and let user retry with same inputs."""
        self.query_one("#failure-section").display = False
        self.query_one("#clone-btn", Button).display = True
        self._update_clone_button()

    def _test_voice(self) -> None:
        """Test the cloned voice."""
        if not self.cloned_voice_name:
            return

        phrase = self.query_one("#test-phrase-input", Input).value.strip()
        if not phrase:
            return

        # Pass voice name directly to TTS script via --voice flag
        # This avoids polluting the global config with a temporary test voice
        self._run_test_worker(phrase, self.cloned_voice_name)

    @work(exclusive=True, thread=True)
    def _run_test_worker(self, phrase: str, cloned_voice_name: str) -> dict:
        """Background worker for voice testing."""
        import subprocess
        import sys
        import os

        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        tts_script = os.path.join(scripts_dir, "mlx_server_utils.py")

        try:
            result = subprocess.run(
                [sys.executable, tts_script, phrase, "--voice", cloned_voice_name],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {"success": result.returncode == 0, "stderr": result.stderr}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "TTS timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}


class VoiceLabScreen(Screen):
    """Voice Lab screen for per-voice audio shaping."""

    DEFAULT_CSS = """
    VoiceLabScreen {
        padding: 1 2;
    }
    .section-title {
        text-style: bold;
        margin: 1 0;
        color: $accent;
    }
    .setting-row {
        height: auto;
        margin: 0 0 1 0;
    }
    .setting-label {
        width: 20;
    }
    .setting-value {
        width: 10;
        text-align: right;
    }
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield VoiceSelector()
        yield Label("Compressor", classes="section-title")
        yield Static("Configure compressor settings in the Compressor tab")
        yield Label("Limiter", classes="section-title")
        yield Static("Configure limiter settings in the Limiter tab")
        yield Footer()


class SystemScreen(Screen):
    """System settings screen."""

    DEFAULT_CSS = """
    SystemScreen {
        padding: 1 2;
    }
    .section-title {
        text-style: bold;
        margin: 1 0;
        color: $accent;
    }
    .setting-row {
        height: 3;
        margin: 0 0 1 0;
    }
    .setting-label {
        width: 25;
    }
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Server Status", classes="section-title"),
            ServerStatusWidget(id="server-status"),
            Label("Playback Settings", classes="section-title"),
            Horizontal(
                Label("Streaming Interval:", classes="setting-label"),
                Select(
                    [
                        ("0.1s (Ultra fast)", 0.1),
                        ("0.3s (Fast)", 0.3),
                        ("0.5s (Normal)", 0.5),
                        ("1.0s (Slow)", 1.0),
                        ("2.0s (Very slow)", 2.0),
                    ],
                    value=get_streaming_interval(),
                    id="interval-select",
                ),
                classes="setting-row",
            ),
        )
        yield Footer()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle setting changes."""
        if event.select.id == "interval-select" and event.value is not None:
            try:
                set_streaming_interval(event.value)
            except ValueError as e:
                self.notify(str(e), severity="error")


class AboutScreen(Screen):
    """About screen with plugin info."""

    DEFAULT_CSS = """
    AboutScreen {
        padding: 1 2;
    }
    .title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    .info {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Back"),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("Claude MLX TTS", classes="title"),
            Static(
                "Voice-cloned TTS notifications for Claude Code using MLX.",
                classes="info",
            ),
            Static(""),
            Label("Keyboard Shortcuts", classes="title"),
            Static("1 - Voice Lab (per-voice audio settings)", classes="info"),
            Static("2 - System Settings (speed, streaming)", classes="info"),
            Static("3 - About (this screen)", classes="info"),
            Static("q - Quit", classes="info"),
        )
        yield Footer()


class MainScreen(Screen):
    """Main screen with tabbed navigation."""

    DEFAULT_CSS = """
    MainScreen {
        padding: 0;
    }
    #main-tabs {
        height: 100%;
    }
    .tab-content {
        padding: 1 2;
    }
    .audio-container {
        width: 100%;
        height: auto;
        align: center middle;
    }
    .audio-widgets {
        height: auto;
        width: auto;
    }
    """

    BINDINGS = [
        Binding("1", "switch_tab('voice-lab')", "Voice Lab", show=True, priority=True),
        Binding("2", "switch_tab('clone-lab')", "Clone Lab", show=True, priority=True),
        Binding("3", "switch_tab('hooks')", "Prompts", show=True, priority=True),
        Binding("4", "switch_tab('system')", "System", show=True, priority=True),
        Binding("5", "switch_tab('about')", "About", show=True, priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="main-tabs"):
            with TabPane("Voice Lab", id="voice-lab"):
                with Container(classes="audio-container"):
                    with Horizontal(classes="audio-widgets"):
                        yield VoiceSelector()
                        yield CompressorWidget(id="compressor-widget")
                        yield LimiterWidget(id="limiter-widget")
                yield PreviewWidget(id="preview-widget")
            with TabPane("Clone Lab", id="clone-lab"):
                yield Label("Voice Cloning", classes="section-title")
                yield Static(
                    "Clone a voice from a WAV file (7-20 seconds recommended). "
                    "The voice will be saved as a compact embedding file.",
                    classes="info",
                )
                yield CloneLabWidget(id="clone-lab-widget")
            with TabPane("Prompt Lab", id="hooks"):
                yield Label("Hook Prompt Customization", classes="section-title")
                yield Static(
                    "Customize prompts and voice for each hook type. "
                    "The Stop hook plays before AI summaries; Permission hooks announce tool requests.",
                    classes="info",
                )
                yield Static("")
                for hook_type in HOOK_TYPES:
                    yield HookSettingsWidget(hook_type, id=f"hook-settings-{hook_type}")
            with TabPane("System", id="system"):
                yield Label("Server Status", classes="section-title")
                yield ServerStatusWidget(id="server-status")
                yield Label("Playback Settings", classes="section-title")
                yield Horizontal(
                    Label("Streaming Interval:", classes="setting-label"),
                    Select(
                        [
                            ("0.1s (Ultra fast)", 0.1),
                            ("0.3s (Fast)", 0.3),
                            ("0.5s (Normal)", 0.5),
                            ("1.0s (Slow)", 1.0),
                            ("2.0s (Very slow)", 2.0),
                        ],
                        value=get_streaming_interval(),
                        id="interval-select",
                    ),
                    classes="setting-row",
                )
                yield Static("")
                yield Static(f"Config: {tts_config.get_config_path()}", classes="info")
            with TabPane("About", id="about"):
                yield Label("Claude MLX TTS", classes="title")
                yield Static(
                    "Voice-cloned TTS notifications for Claude Code using MLX.\n\n"
                    "Use keyboard shortcuts 1-5 to navigate tabs:\n"
                    "  1 - Voice Lab (per-voice audio settings)\n"
                    "  2 - Clone Lab (voice cloning)\n"
                    "  3 - Prompt Lab (hook prompts and voice)\n"
                    "  4 - System (server status, streaming)\n"
                    "  5 - About (this screen)"
                )
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to specified tab."""
        tabs = self.query_one("#main-tabs", TabbedContent)
        tabs.active = tab_id

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle setting changes."""
        if event.select.id == "interval-select" and event.value is not None:
            try:
                set_streaming_interval(event.value)
            except ValueError as e:
                self.notify(str(e), severity="error")

    def on_preview_widget_audio_reset(self, event: PreviewWidget.AudioReset) -> None:
        """Handle audio reset - refresh compressor and limiter widgets."""
        self.query_one("#compressor-widget", CompressorWidget)._refresh_fields()
        self.query_one("#limiter-widget", LimiterWidget)._refresh_fields()

    def on_voice_selector_voice_changed(self, event: VoiceSelector.VoiceChanged) -> None:
        """Handle voice change - refresh compressor and limiter widgets with new voice's settings."""
        self.query_one("#compressor-widget", CompressorWidget)._refresh_fields()
        self.query_one("#limiter-widget", LimiterWidget)._refresh_fields()

    def on_clone_lab_widget_voice_cloned(self, event: CloneLabWidget.VoiceCloned) -> None:
        """Handle voice cloned - refresh voice selector with new voice."""
        voice_selector = self.query_one(VoiceSelector)
        option_list = voice_selector.query_one("#voice-list", OptionList)

        # Add the new voice to the list
        display_name = get_voice_display(event.voice_name)
        option_list.add_option(Option(display_name, id=event.voice_name))

        # Refresh hook widget voice dropdowns to include the new voice
        self._refresh_hook_widgets()

        # Notify user
        self.notify(f"Voice '{event.voice_name}' created successfully!", severity="information")

    # =========================================================================
    # Voice CRUD handlers
    # =========================================================================

    def on_voice_selector_copy_voice_requested(self, event: VoiceSelector.CopyVoiceRequested) -> None:
        """Handle copy voice request - copy with auto-generated name."""
        try:
            new_name = copy_voice(event.voice_name)
            self.notify(f"Copied to '{new_name}'", severity="information")

            # Refresh voice selector
            voice_selector = self.query_one(VoiceSelector)
            voice_selector.refresh_voices()

            # Refresh hook widget voice dropdowns
            self._refresh_hook_widgets()
        except ValueError as e:
            self.notify(str(e), severity="error")

    def on_voice_selector_rename_voice_requested(self, event: VoiceSelector.RenameVoiceRequested) -> None:
        """Handle rename voice request - show InputModal."""
        modal = InputModal(
            title="Rename Voice",
            voice_name=event.voice_name,
            action_type="rename",
        )
        self.app.push_screen(modal, callback=self._handle_input_result)

    def on_voice_selector_delete_voice_requested(self, event: VoiceSelector.DeleteVoiceRequested) -> None:
        """Handle delete voice request - show DeleteVoiceModal."""
        usage = get_voice_usage(event.voice_name)
        modal = DeleteVoiceModal(
            voice_name=event.voice_name,
            usage=usage,
        )
        self.app.push_screen(modal, callback=self._handle_delete_result)

    def _refresh_hook_widgets(self) -> None:
        """Refresh all HookSettingsWidget voice dropdowns after voice CRUD."""
        for hook_type in HOOK_TYPES:
            hook_widget = self.query_one(f"#hook-settings-{hook_type}", HookSettingsWidget)
            hook_widget.refresh_voices()

    def _handle_delete_result(self, voice_name: str | None) -> None:
        """Handle delete modal result via callback."""
        if voice_name is None:
            return
        try:
            delete_voice(voice_name)
            self.notify(f"Voice '{voice_name}' deleted", severity="information")

            # Refresh voice selector and select 'default' voice
            voice_selector = self.query_one(VoiceSelector)
            voice_selector.refresh_voices()
            voice_selector.select_voice("default")

            # Refresh compressor/limiter with new active voice
            self.query_one("#compressor-widget", CompressorWidget)._refresh_fields()
            self.query_one("#limiter-widget", LimiterWidget)._refresh_fields()

            # Refresh hook widget voice dropdowns
            self._refresh_hook_widgets()
        except ValueError as e:
            self.notify(str(e), severity="error")

    def _handle_input_result(self, result: tuple[str, str, str] | None) -> None:
        """Handle input modal result via callback (rename/copy)."""
        if result is None:
            return
        voice_name, new_name, action_type = result
        try:
            if action_type == "rename":
                rename_voice(voice_name, new_name)
                self.notify(f"Renamed to '{new_name}'", severity="information")
            elif action_type == "copy":
                copy_voice(voice_name, new_name)
                self.notify(f"Copied to '{new_name}'", severity="information")

            # Refresh voice selector
            voice_selector = self.query_one(VoiceSelector)
            voice_selector.refresh_voices()

            # Refresh compressor/limiter if needed
            self.query_one("#compressor-widget", CompressorWidget)._refresh_fields()
            self.query_one("#limiter-widget", LimiterWidget)._refresh_fields()

            # Refresh hook widget voice dropdowns
            self._refresh_hook_widgets()
        except ValueError as e:
            self.notify(str(e), severity="error")


class TTSConfigApp(App):
    """TTS Configuration TUI Application."""

    TITLE = "TTS Configuration"
    SUB_TITLE = "Claude MLX TTS"

    CSS = """
    Screen {
        background: $surface;
    }
    .section-title {
        text-style: bold;
        margin: 1 0;
        color: $accent;
    }
    .setting-row {
        height: 3;
        margin: 0 0 1 0;
    }
    .setting-label {
        width: 25;
    }
    .title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    SCREENS = {
        "main": MainScreen,
        "voice-lab": VoiceLabScreen,
        "system": SystemScreen,
        "about": AboutScreen,
    }

    def on_mount(self) -> None:
        """Initialize the app."""
        self.push_screen("main")


def main() -> None:
    """Run the TUI application."""
    app = TTSConfigApp()
    app.run()


if __name__ == "__main__":
    main()
