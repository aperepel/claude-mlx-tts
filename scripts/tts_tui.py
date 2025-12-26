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
    OptionList,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option
from textual.message import Message

import tts_config
from tts_config import (
    DEFAULT_COMPRESSOR,
    DEFAULT_LIMITER,
    SPEED_PRESETS,
    discover_voices,
    get_active_voice,
    get_compressor_config,
    get_effective_compressor,
    get_effective_limiter,
    get_limiter_config,
    get_playback_speed,
    get_streaming_interval,
    set_active_voice,
    set_compressor_setting,
    set_limiter_setting,
    set_playback_speed,
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
    """Voice selection widget with current voice indicator."""

    DEFAULT_CSS = """
    VoiceSelector {
        height: auto;
        padding: 1;
    }
    VoiceSelector > Label {
        margin-bottom: 1;
    }
    VoiceSelector > Select {
        width: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Voice:")
        voices = discover_voices()
        active = get_active_voice()
        options = [(v, v) for v in voices] if voices else [("No voices found", "")]
        yield Select(options, value=active if active in voices else "", id="voice-select")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle voice selection change."""
        if event.value and event.value != "":
            try:
                set_active_voice(event.value)
                self.notify(f"Voice set to: {event.value}")
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
        """Handle value input."""
        event.stop()
        try:
            new_val = float(event.value.strip())
            new_val = max(self._min, min(self._max, new_val))
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
        height: auto;
        padding: 1 2;
        border: solid $primary;
        margin: 1 1 1 0;
        width: 38;
    }
    CompressorWidget > .header-row {
        height: 3;
        width: 100%;
    }
    CompressorWidget > .header-row > Label {
        width: auto;
        text-style: bold;
        padding: 1 1;
    }
    CompressorWidget > .header-row > Switch {
        width: auto;
    }
    CompressorWidget > .header-row > Select {
        width: 16;
    }
    CompressorWidget > .form-fields {
        height: auto;
        width: 100%;
        padding: 1 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = get_compressor_config()

    def compose(self) -> ComposeResult:
        config = self._config

        # Header with enable toggle and preset selector
        with Horizontal(classes="header-row"):
            yield Switch(value=config.get("enabled", True), id="compressor-enabled")
            yield Label("Compressor")
            yield Select(
                [("Custom", "custom")] + [(name, name) for name in COMPRESSOR_PRESETS],
                value="custom",
                id="preset-select",
            )

        # Form fields
        with Vertical(classes="form-fields"):
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

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle compressor enable toggle."""
        if event.switch.id == "compressor-enabled":
            try:
                set_compressor_setting("enabled", event.value)
                state = "enabled" if event.value else "disabled"
                self.notify(f"Compressor {state}")
            except ValueError as e:
                self.notify(str(e), severity="error")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle preset selection."""
        if event.select.id == "preset-select" and event.value != "custom":
            preset = COMPRESSOR_PRESETS.get(event.value)
            if preset:
                self._apply_preset(preset)
                self.notify(f"Preset: {event.value}")

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
        height: auto;
        padding: 1 2;
        border: solid $primary;
        margin: 1 0 1 1;
        width: 38;
    }
    LimiterWidget > .header-row {
        height: 3;
        width: 100%;
    }
    LimiterWidget > .header-row > Label {
        width: auto;
        text-style: bold;
        padding: 1 1;
    }
    LimiterWidget > .header-row > Switch {
        width: auto;
    }
    LimiterWidget > .header-row > Select {
        width: 16;
    }
    LimiterWidget > .form-fields {
        height: auto;
        width: 100%;
        padding: 1 0;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._config = get_limiter_config()

    def compose(self) -> ComposeResult:
        config = self._config

        # Header with enable toggle and preset selector
        with Horizontal(classes="header-row"):
            yield Switch(value=config.get("enabled", True), id="limiter-enabled")
            yield Label("Limiter")
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

    def on_form_field_changed(self, event: FormField.Changed) -> None:
        """Handle field value changes."""
        param_key = event.field.param_key
        if param_key:
            try:
                set_limiter_setting(param_key, event.value)
            except ValueError as e:
                self.notify(str(e), severity="error")

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle limiter enable toggle."""
        if event.switch.id == "limiter-enabled":
            try:
                set_limiter_setting("enabled", event.value)
                state = "enabled" if event.value else "disabled"
                self.notify(f"Limiter {state}")
            except ValueError as e:
                self.notify(str(e), severity="error")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle preset selection."""
        if event.select.id == "limiter-preset-select" and event.value != "custom":
            preset = LIMITER_PRESETS.get(event.value)
            if preset:
                self._apply_preset(preset)
                self.notify(f"Preset: {event.value}")

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
                Label("Speed:", classes="setting-label"),
                Select(
                    [(f"{s}x ({l})", s) for s, l in SPEED_PRESETS.items()],
                    value=get_playback_speed(),
                    id="speed-select",
                ),
                classes="setting-row",
            ),
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
        if event.select.id == "speed-select" and event.value is not None:
            try:
                set_playback_speed(event.value)
                label = SPEED_PRESETS.get(event.value, "Custom")
                self.notify(f"Speed set to {event.value}x ({label})")
            except ValueError as e:
                self.notify(str(e), severity="error")
        elif event.select.id == "interval-select" and event.value is not None:
            try:
                set_streaming_interval(event.value)
                self.notify(f"Streaming interval set to {event.value}s")
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
            Static(""),
            Label("Configuration", classes="title"),
            Static(f"Config path: {tts_config.get_config_path()}", classes="info"),
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
        Binding("1", "switch_tab('voice-lab')", "Voice Lab", show=True),
        Binding("2", "switch_tab('system')", "System", show=True),
        Binding("3", "switch_tab('about')", "About", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent(id="main-tabs"):
            with TabPane("Voice Lab", id="voice-lab"):
                yield VoiceSelector()
                with Container(classes="audio-container"):
                    with Horizontal(classes="audio-widgets"):
                        yield CompressorWidget(id="compressor-widget")
                        yield LimiterWidget(id="limiter-widget")
            with TabPane("System", id="system"):
                yield Label("Server Status", classes="section-title")
                yield ServerStatusWidget(id="server-status")
                yield Label("Playback Settings", classes="section-title")
                yield Horizontal(
                    Label("Speed:", classes="setting-label"),
                    Select(
                        [(f"{s}x ({l})", s) for s, l in SPEED_PRESETS.items()],
                        value=get_playback_speed(),
                        id="speed-select",
                    ),
                    classes="setting-row",
                )
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
            with TabPane("About", id="about"):
                yield Label("Claude MLX TTS", classes="title")
                yield Static(
                    "Voice-cloned TTS notifications for Claude Code using MLX.\n\n"
                    "Use keyboard shortcuts 1/2/3 to navigate tabs."
                )
                yield Static("")
                yield Static(f"Config: {tts_config.get_config_path()}")
        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to specified tab."""
        tabs = self.query_one("#main-tabs", TabbedContent)
        tabs.active = tab_id

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle setting changes."""
        if event.select.id == "speed-select" and event.value is not None:
            try:
                set_playback_speed(event.value)
                label = SPEED_PRESETS.get(event.value, "Custom")
                self.notify(f"Speed set to {event.value}x ({label})")
            except ValueError as e:
                self.notify(str(e), severity="error")
        elif event.select.id == "interval-select" and event.value is not None:
            try:
                set_streaming_interval(event.value)
                self.notify(f"Streaming interval set to {event.value}s")
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
