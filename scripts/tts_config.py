"""
TTS Configuration module.

Manages persistent configuration for TTS playback settings.
Config is stored at ${PLUGIN_ROOT}/.config/config.json

Structure:
    {
        "profiles": {
            "default": {"speed": 1.3, "streaming_interval": 0.5}
        },
        "active_profile": "default",
        "active_voice": "default_voice",
        "voices": {
            "my_voice": {
                "compressor": {"gain_db": 10, "enabled": true},
                "limiter": {"threshold_db": -1.0}
            }
        }
    }

Features:
- Per-voice compressor/limiter settings
- Voice discovery from assets/*.wav
- Cascading config resolution (defaults -> voice-specific)
- Secure voice name validation

Can be overridden with TTS_CONFIG_PATH environment variable for testing.
"""
import json
import os
import re
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

# Compressor configuration (notification_punch preset)
DEFAULT_COMPRESSOR = {
    "enabled": True,
    "threshold_db": -18,
    "ratio": 3.0,
    "attack_ms": 3,
    "release_ms": 50,
    "gain_db": 8,
}

# Limiter configuration (separate from compressor)
DEFAULT_LIMITER = {
    "enabled": True,
    "threshold_db": -0.5,
    "release_ms": 40,
}

# Valid characters for voice names (security)
VOICE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

# Default configuration values
DEFAULT_CONFIG = {
    "profiles": {
        "default": {
            "speed": DEFAULT_SPEED,
            "streaming_interval": DEFAULT_STREAMING_INTERVAL,
        }
    },
    "active_profile": "default",
    "compressor": DEFAULT_COMPRESSOR.copy(),
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


def get_compressor_config() -> dict:
    """Get compressor configuration. Re-reads from disk on each call."""
    config = load_config()
    compressor = config.get("compressor", {})
    # Merge with defaults for any missing keys
    return {**DEFAULT_COMPRESSOR, **compressor}


def set_compressor_setting(key: str, value: float | bool) -> None:
    """Set a single compressor setting."""
    if key not in DEFAULT_COMPRESSOR:
        valid_keys = ", ".join(DEFAULT_COMPRESSOR.keys())
        raise ValueError(f"Invalid compressor key '{key}'. Valid keys: {valid_keys}")

    config = load_config()
    if "compressor" not in config:
        config["compressor"] = DEFAULT_COMPRESSOR.copy()
    config["compressor"][key] = value
    save_config(config)


def set_compressor_gain(gain_db: float) -> None:
    """Set compressor makeup gain in dB."""
    set_compressor_setting("gain_db", gain_db)


def set_compressor_enabled(enabled: bool) -> None:
    """Enable or disable the compressor."""
    set_compressor_setting("enabled", enabled)


def get_limiter_config() -> dict:
    """Get limiter configuration. Re-reads from disk on each call."""
    config = load_config()
    limiter = config.get("limiter", {})
    # Merge with defaults for any missing keys
    return {**DEFAULT_LIMITER, **limiter}


def set_limiter_setting(key: str, value: float | bool) -> None:
    """Set a single limiter setting."""
    if key not in DEFAULT_LIMITER:
        valid_keys = ", ".join(DEFAULT_LIMITER.keys())
        raise ValueError(f"Invalid limiter key '{key}'. Valid keys: {valid_keys}")

    config = load_config()
    if "limiter" not in config:
        config["limiter"] = DEFAULT_LIMITER.copy()
    config["limiter"][key] = value
    save_config(config)


# =============================================================================
# Voice Discovery and Management
# =============================================================================


def discover_voices() -> list[str]:
    """Discover all voice files in the assets directory.

    Returns list of voice names (without .wav extension).
    """
    assets_dir = _PLUGIN_ROOT / "assets"
    if not assets_dir.exists():
        return []

    voices = []
    for wav_file in assets_dir.glob("*.wav"):
        voices.append(wav_file.stem)
    return sorted(voices)


def get_active_voice() -> str:
    """Get the currently active voice name."""
    config = load_config()
    return config.get("active_voice", "default_voice")


def set_active_voice(voice_name: str) -> None:
    """Set the active voice. Validates that voice exists."""
    available_voices = discover_voices()
    if voice_name not in available_voices:
        raise ValueError(f"Voice '{voice_name}' not found. Available: {available_voices}")

    config = load_config()
    config["active_voice"] = voice_name
    save_config(config)


def resolve_voice_path(voice_name: str) -> Path:
    """Securely resolve a voice name to its file path.

    Validates:
    - No path traversal (../)
    - No absolute paths
    - Only alphanumeric, underscore, hyphen
    - File must exist
    - Resolved path stays within assets directory (symlink-safe)

    Returns the full Path to the .wav file.
    Raises ValueError for invalid or non-existent voices.
    """
    # Reject absolute paths
    if voice_name.startswith("/"):
        raise ValueError(f"Invalid voice name: {voice_name}")

    # Reject path traversal and special characters
    if not VOICE_NAME_PATTERN.match(voice_name):
        raise ValueError(f"Invalid voice name: {voice_name}")

    # Construct path and verify it exists
    assets_dir = _PLUGIN_ROOT / "assets"
    voice_path = assets_dir / f"{voice_name}.wav"
    if not voice_path.exists():
        raise ValueError(f"Voice '{voice_name}' not found at {voice_path}")

    # Symlink-safe: ensure resolved path is within assets directory
    resolved = voice_path.resolve()
    resolved_assets = assets_dir.resolve()
    if not str(resolved).startswith(str(resolved_assets) + "/"):
        raise ValueError(f"Invalid voice name: {voice_name}")

    return voice_path


# =============================================================================
# Per-Voice Configuration
# =============================================================================


def get_voice_config(voice_name: str) -> dict:
    """Get configuration for a specific voice.

    Returns voice-specific settings merged with defaults.
    """
    config = load_config()
    voices = config.get("voices", {})
    voice_config = voices.get(voice_name, {})

    # Merge with defaults
    return {
        "compressor": {**DEFAULT_COMPRESSOR, **voice_config.get("compressor", {})},
        "limiter": {**DEFAULT_LIMITER, **voice_config.get("limiter", {})},
    }


def set_voice_config(voice_name: str, settings: dict) -> None:
    """Set configuration for a specific voice.

    Merges new settings with existing voice config.
    """
    config = load_config()

    if "voices" not in config:
        config["voices"] = {}

    if voice_name not in config["voices"]:
        config["voices"][voice_name] = {}

    # Deep merge settings
    for key, value in settings.items():
        if key not in config["voices"][voice_name]:
            config["voices"][voice_name][key] = {}
        if isinstance(value, dict):
            config["voices"][voice_name][key] = {
                **config["voices"][voice_name].get(key, {}),
                **value
            }
        else:
            config["voices"][voice_name][key] = value

    save_config(config)


def get_effective_compressor(voice_name: str = None) -> dict:
    """Get effective compressor settings for a voice.

    Uses cascading resolution: defaults -> voice-specific overrides.
    If voice_name is None, uses the active voice.
    """
    if voice_name is None:
        voice_name = get_active_voice()

    config = load_config()
    voices = config.get("voices", {})
    voice_config = voices.get(voice_name, {})
    voice_compressor = voice_config.get("compressor", {})

    # Cascade: defaults <- voice-specific
    return {**DEFAULT_COMPRESSOR, **voice_compressor}


def get_effective_limiter(voice_name: str = None) -> dict:
    """Get effective limiter settings for a voice.

    Uses cascading resolution: defaults -> voice-specific overrides.
    If voice_name is None, uses the active voice.
    """
    if voice_name is None:
        voice_name = get_active_voice()

    config = load_config()
    voices = config.get("voices", {})
    voice_config = voices.get(voice_name, {})
    voice_limiter = voice_config.get("limiter", {})

    # Cascade: defaults <- voice-specific
    return {**DEFAULT_LIMITER, **voice_limiter}


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


def cmd_status() -> None:
    """Show current configuration and how to change it."""
    config = load_config()
    config_path = get_config_path()
    profile = config.get("active_profile", "default")
    speed = get_profile_speed(profile)
    speed_label = SPEED_PRESETS.get(speed, "Custom")
    interval = get_streaming_interval(profile)
    compressor = get_compressor_config()

    print("TTS Configuration Status")
    print("=" * 40)
    print(f"Config file: {config_path}")
    print(f"Active profile: {profile}")
    print()
    print("Playback Settings:")
    print(f"  Speed: {speed}x ({speed_label})")
    print(f"  Streaming interval: {interval}s")
    print()
    print("Compressor:")
    print(f"  Enabled: {compressor['enabled']}")
    print(f"  Gain: {compressor['gain_db']} dB")
    print()
    print("To configure TTS settings, run:")
    print("  uv run --directory $CLAUDE_PLUGIN_ROOT python scripts/tts_configurator.py")


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
        cmd_status()
        return

    command = sys.argv[1].lower()

    if command == "status":
        cmd_status()
    elif command == "show":
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
        print("Usage: tts-config.py [status|show|set <speed>|wizard]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
