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
from textual.screen import Screen
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
from pathlib import Path
import re

import tts_config
from tts_config import (
    DEFAULT_COMPRESSOR,
    DEFAULT_LIMITER,
    HOOK_TYPES,
    discover_voices,
    get_active_voice,
    get_compressor_config,
    get_effective_compressor,
    get_effective_limiter,
    get_hook_voice,
    get_limiter_config,
    get_streaming_interval,
    get_voice_format,
    reset_all_audio_to_defaults,
    set_active_voice,
    set_compressor_setting,
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


class VoiceSelector(Container):
    """Voice selection widget with vertical list."""

    class VoiceChanged(Message):
        """Posted when a voice is selected."""

        def __init__(self, voice_name: str) -> None:
            self.voice_name = voice_name
            super().__init__()

    DEFAULT_CSS = """
    VoiceSelector {
        height: 22;
        width: 32;
        padding: 1;
        border: round $primary;
        border-title-color: $text;
        margin: 1 1 1 0;
    }
    VoiceSelector > OptionList {
        height: 100%;
        width: 100%;
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

    def on_mount(self) -> None:
        """Highlight the active voice on mount."""
        voices = discover_voices()
        active = get_active_voice()
        if active in voices:
            option_list = self.query_one("#voice-list", OptionList)
            idx = voices.index(active)
            option_list.highlighted = idx

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Handle voice selection."""
        voice_name = str(event.option.id)
        try:
            set_active_voice(voice_name)
            self.post_message(self.VoiceChanged(voice_name))
        except ValueError as e:
            self.notify(str(e), severity="error")


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
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._value = value
        self._min = min_val
        self._max = max_val
        self._unit = unit
        self._param_key = param_key

    def compose(self) -> ComposeResult:
        yield Label(f"{self._label}:", classes="label")
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
        height: 22;
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
            )

    def on_form_field_changed(self, event: FormField.Changed) -> None:
        """Handle field value changes."""
        param_key = event.field.param_key
        if param_key:
            try:
                set_compressor_setting(param_key, event.value)
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
            except ValueError as e:
                self.notify(str(e), severity="error")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle preset selection."""
        if event.select.id == "preset-select" and event.value != "custom":
            preset = COMPRESSOR_PRESETS.get(event.value)
            if preset:
                self._apply_preset(preset)

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
        height: 22;
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
            except ValueError as e:
                self.notify(str(e), severity="error")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle preset selection."""
        if event.select.id == "limiter-preset-select" and event.value != "custom":
            preset = LIMITER_PRESETS.get(event.value)
            if preset:
                self._apply_preset(preset)

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


# Human-readable hook type labels
HOOK_LABELS = {
    "stop": "Stop (completion)",
    "permission_request": "Permission Request",
}


class HookVoiceSelector(Container):
    """Widget for selecting voice override for a specific hook type."""

    DEFAULT_CSS = """
    HookVoiceSelector {
        height: auto;
        width: 100%;
        padding: 1 2;
        margin: 0 0 1 0;
    }
    HookVoiceSelector > .hook-row {
        height: 3;
        width: 100%;
    }
    HookVoiceSelector > .hook-row > .hook-label {
        width: 24;
        padding: 1 1 1 0;
    }
    HookVoiceSelector > .hook-row > Select {
        width: 30;
    }
    HookVoiceSelector > .hook-description {
        color: $text-muted;
        padding: 0 0 0 2;
        height: auto;
    }
    """

    def __init__(self, hook_type: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._hook_type = hook_type

    def compose(self) -> ComposeResult:
        voices = discover_voices()
        current_voice = get_hook_voice(self._hook_type)

        # Build options: "-- inherit --" (uses active voice) + all available voices
        options = [("-- inherit --", "")]
        options.extend((voice, voice) for voice in voices)

        # Determine current value (empty string means inherit)
        current_value = current_voice if current_voice else ""

        with Horizontal(classes="hook-row"):
            yield Label(HOOK_LABELS.get(self._hook_type, self._hook_type), classes="hook-label")
            yield Select(
                options,
                value=current_value,
                id=f"hook-voice-{self._hook_type}",
            )

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle voice selection change."""
        if event.select.id == f"hook-voice-{self._hook_type}":
            # Empty string means inherit from active voice
            voice = event.value if event.value else None
            try:
                set_hook_voice(self._hook_type, voice)
            except ValueError as e:
                self.notify(str(e), severity="error")


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
            reset_all_audio_to_defaults()
            self.post_message(self.AudioReset())
            self.app.set_focus(None)

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
        margin: 0 0 1 0;
    }
    CloneLabWidget > .success-section > .test-row {
        height: 3;
        width: 100%;
        margin: 1 0;
    }
    CloneLabWidget > .success-section > .test-row > Input {
        width: 40;
    }
    CloneLabWidget > .success-section > .test-row > Button {
        width: auto;
        margin: 0 0 0 1;
    }
    CloneLabWidget > .success-section > .button-row {
        height: 3;
        width: 100%;
        margin: 1 0;
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
                yield Input(
                    placeholder="/path/to/voice.wav",
                    id="wav-path-input",
                )
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
            with Horizontal(classes="test-row"):
                yield Input(
                    value="Hello, this is my new voice!",
                    id="test-phrase-input",
                )
                yield Button("Speak", id="speak-btn", variant="success")
            with Horizontal(classes="button-row"):
                yield Button("Clone Another", id="clone-another-btn", variant="default")

    def on_mount(self) -> None:
        """Hide elements initially."""
        self.query_one("#clone-loading").display = False
        self.query_one("#status-msg").display = False
        self.query_one("#success-section").display = False

    def _validate_wav_path(self, path: str) -> tuple[bool, str]:
        """Validate WAV file path."""
        if not path.strip():
            return False, ""

        wav_path = Path(path.strip())
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

    def _start_cloning(self) -> None:
        """Start the voice cloning process."""
        if self.is_cloning:
            return

        wav_path = self.query_one("#wav-path-input", Input).value.strip()
        voice_name = self.query_one("#voice-name-input", Input).value.strip()

        self.is_cloning = True
        self.query_one("#clone-loading").display = True
        self.query_one("#clone-btn", Button).disabled = True
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
                self.cloned_size_info = size_info.strip() or "Voice cloned successfully"
                self.clone_success = True

                # Update UI
                self.query_one("#status-msg").display = False
                self.query_one("#success-section").display = True
                self.query_one("#success-header", Static).update(
                    f"✓ Created: {voice_name}  {self.cloned_size_info}"
                )

                # Post message for voice selector refresh
                self.post_message(self.VoiceCloned(voice_name))
            else:
                error = result.get("error", "Unknown error")
                self.query_one("#status-msg").update(f"[red]Error: {error}[/]")
                self._update_clone_button()

        elif event.state in (WorkerState.ERROR, WorkerState.CANCELLED):
            self.is_cloning = False
            self.query_one("#clone-loading").display = False
            self.query_one("#status-msg").update("[red]Cloning failed unexpectedly[/]")
            self._update_clone_button()

    def _reset_form(self) -> None:
        """Reset form for another clone."""
        self.clone_success = False
        self.cloned_voice_name = ""
        self.cloned_size_info = ""

        self.query_one("#success-section").display = False
        self.query_one("#wav-path-input", Input).value = ""
        self.query_one("#voice-name-input", Input).value = ""
        self.query_one("#wav-validation", Label).update("")
        self.query_one("#name-validation", Label).update("")
        self._wav_valid = False
        self._name_valid = False
        self._update_clone_button()

    def _test_voice(self) -> None:
        """Test the cloned voice."""
        if not self.cloned_voice_name:
            return

        phrase = self.query_one("#test-phrase-input", Input).value.strip()
        if not phrase:
            return

        # Set active voice to the cloned one and play
        from tts_config import set_active_voice
        try:
            set_active_voice(self.cloned_voice_name)
        except ValueError:
            pass

        self._run_test_worker(phrase)

    @work(exclusive=True, thread=True)
    def _run_test_worker(self, phrase: str) -> dict:
        """Background worker for voice testing."""
        import subprocess
        import sys
        import os

        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        tts_script = os.path.join(scripts_dir, "mlx_server_utils.py")

        try:
            result = subprocess.run(
                [sys.executable, tts_script, phrase],
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
        Binding("3", "switch_tab('hooks')", "Hooks", show=True, priority=True),
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
            with TabPane("Hooks", id="hooks"):
                yield Label("Per-Hook Voice Settings", classes="section-title")
                yield Static(
                    "Override the default voice for specific hooks. "
                    "When set, these hooks will use the specified voice instead of the default.",
                    classes="info",
                )
                yield Static("")
                for hook_type in HOOK_TYPES:
                    yield HookVoiceSelector(hook_type, id=f"hook-selector-{hook_type}")
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
                    "  3 - Hooks (per-hook voice overrides)\n"
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

        # Notify user
        self.notify(f"Voice '{event.voice_name}' created successfully!", severity="information")


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
