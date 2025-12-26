"""
TTS Configuration module.

Manages persistent configuration for TTS playback settings.
Config is stored at ${PLUGIN_ROOT}/.config/config.json

Structure:
    {
        "profiles": {
            "default": {"speed": 1.3}
        },
        "active_profile": "default"
    }

Can be overridden with TTS_CONFIG_PATH environment variable for testing.
"""
import json
import os
import sys
from pathlib import Path

# Plugin-local config directory (follows .cache/ pattern)
_PLUGIN_ROOT = Path(__file__).parent.parent
_CONFIG_DIR = _PLUGIN_ROOT / ".config"

# Speed presets: speed multiplier -> human label
SPEED_PRESETS = {
    1.0: "Slow",
    1.3: "Normal",
    1.6: "Fast",
    2.0: "Turbo",
}

# Default speed when no config exists
DEFAULT_SPEED = 1.3

# Streaming configuration
DEFAULT_STREAMING_INTERVAL = 0.5  # Target: 0.5s for ~260ms TTFT
MIN_STREAMING_INTERVAL = 0.1
MAX_STREAMING_INTERVAL = 5.0

# Default configuration values
DEFAULT_CONFIG = {
    "profiles": {
        "default": {"speed": DEFAULT_SPEED, "streaming_interval": DEFAULT_STREAMING_INTERVAL}
    },
    "active_profile": "default",
}


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    return _CONFIG_DIR


def get_config_path() -> Path:
    """Get the configuration file path. Override with TTS_CONFIG_PATH env var."""
    env_path = os.environ.get("TTS_CONFIG_PATH")
    if env_path:
        return Path(env_path)
    return _CONFIG_DIR / "config.json"


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override into base, returning new dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> dict:
    """Load configuration from file, returning defaults if not found or invalid."""
    config_path = get_config_path()
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    try:
        with open(config_path) as f:
            file_config = json.load(f)
        # Deep merge with defaults for any missing keys
        return _deep_merge(DEFAULT_CONFIG, file_config)
    except (json.JSONDecodeError, IOError):
        return DEFAULT_CONFIG.copy()


def save_config(config: dict) -> None:
    """Save configuration to file, creating directory if needed."""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def get_active_profile() -> str:
    """Get the name of the active profile."""
    config = load_config()
    return config.get("active_profile", "default")


def get_profile_speed(profile: str = None) -> float:
    """Get the speed for a profile (default: active profile)."""
    config = load_config()
    if profile is None:
        profile = config.get("active_profile", "default")
    profiles = config.get("profiles", {})
    profile_config = profiles.get(profile, {})
    return profile_config.get("speed", DEFAULT_SPEED)


def get_playback_speed() -> float:
    """Get the current playback speed setting (active profile)."""
    return get_profile_speed()


def set_playback_speed(speed: float, profile: str = None) -> None:
    """Set the playback speed for a profile, must be a valid preset value."""
    if speed not in SPEED_PRESETS:
        valid = ", ".join(f"{s}" for s in SPEED_PRESETS.keys())
        raise ValueError(f"Invalid speed {speed}. Valid speeds: {valid}")

    config = load_config()
    if profile is None:
        profile = config.get("active_profile", "default")

    if "profiles" not in config:
        config["profiles"] = {}
    if profile not in config["profiles"]:
        config["profiles"][profile] = {}

    config["profiles"][profile]["speed"] = speed
    save_config(config)


def get_streaming_interval(profile: str = None) -> float:
    """Get the streaming interval for a profile (default: active profile)."""
    config = load_config()
    if profile is None:
        profile = config.get("active_profile", "default")
    profiles = config.get("profiles", {})
    profile_config = profiles.get(profile, {})
    return profile_config.get("streaming_interval", DEFAULT_STREAMING_INTERVAL)


def set_streaming_interval(interval: float, profile: str = None) -> None:
    """Set the streaming interval for a profile."""
    if interval < MIN_STREAMING_INTERVAL or interval > MAX_STREAMING_INTERVAL:
        raise ValueError(
            f"Invalid streaming interval {interval}. "
            f"Must be between {MIN_STREAMING_INTERVAL} and {MAX_STREAMING_INTERVAL}"
        )

    config = load_config()
    if profile is None:
        profile = config.get("active_profile", "default")

    if "profiles" not in config:
        config["profiles"] = {}
    if profile not in config["profiles"]:
        config["profiles"][profile] = {}

    config["profiles"][profile]["streaming_interval"] = interval
    save_config(config)


def format_current_config() -> str:
    """Format the current configuration for display."""
    config = load_config()
    profile = config.get("active_profile", "default")
    speed = get_profile_speed(profile)
    label = SPEED_PRESETS.get(speed, "Custom")
    return f"Profile: {profile}\nPlayback speed: {speed}x ({label})"


def cmd_show() -> None:
    """Show current configuration."""
    print("Current TTS Configuration:")
    print(format_current_config())


def cmd_set(speed_str: str) -> None:
    """Set playback speed."""
    try:
        speed = float(speed_str)
    except ValueError:
        print(f"Error: Invalid speed value '{speed_str}'", file=sys.stderr)
        sys.exit(1)

    try:
        set_playback_speed(speed)
        label = SPEED_PRESETS.get(speed, "Custom")
        print(f"Playback speed set to {speed}x ({label})")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_wizard() -> None:
    """Output wizard info for Claude to use with AskUserQuestion."""
    config = load_config()
    profile = config.get("active_profile", "default")
    current_speed = get_profile_speed(profile)
    current_label = SPEED_PRESETS.get(current_speed, "Custom")

    print("TTS Configuration Wizard")
    print("=" * 40)
    print(f"\nActive profile: {profile}")
    print(f"Current speed: {current_speed}x ({current_label})")
    print("\nAvailable playback speeds:")
    for speed, label in sorted(SPEED_PRESETS.items()):
        marker = " <-- current" if speed == current_speed else ""
        print(f"  {speed}x - {label}{marker}")
    print("\nUse AskUserQuestion to let the user select a speed,")
    print("then run: tts-config.py set <speed>")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        cmd_show()
        return

    command = sys.argv[1].lower()

    if command == "show":
        cmd_show()
    elif command == "set":
        if len(sys.argv) < 3:
            print("Error: set requires a speed value", file=sys.stderr)
            sys.exit(1)
        cmd_set(sys.argv[2])
    elif command == "wizard":
        cmd_wizard()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Usage: tts-config.py [show|set <speed>|wizard]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
