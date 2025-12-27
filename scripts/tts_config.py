"""
TTS Configuration module.

Manages persistent configuration for TTS playback settings.
Config is stored at ${PLUGIN_ROOT}/.config/config.json

Structure:
    {
        "profiles": {
            "default": {"streaming_interval": 0.5}
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

# Streaming configuration
DEFAULT_STREAMING_INTERVAL = 0.5  # Target: 0.5s for ~260ms TTFT
MIN_STREAMING_INTERVAL = 0.1
MAX_STREAMING_INTERVAL = 5.0

# Compressor configuration (notification_punch preset)
DEFAULT_COMPRESSOR = {
    "enabled": True,
    "input_gain_db": 0.0,
    "threshold_db": -18,
    "ratio": 3.0,
    "attack_ms": 3,
    "release_ms": 50,
    "gain_db": 8,
    "master_gain_db": 0.0,
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


# =============================================================================
# Per-Hook Voice Configuration
# =============================================================================

# Available hook types that can have voice overrides
HOOK_TYPES = ["stop", "permission_request"]


def get_hook_voice(hook_type: str) -> str | None:
    """Get the voice override for a specific hook type.

    Returns None if no override is set (uses default voice).
    """
    if hook_type not in HOOK_TYPES:
        raise ValueError(f"Invalid hook type: {hook_type}. Valid: {HOOK_TYPES}")

    config = load_config()
    hooks = config.get("hooks", {})
    return hooks.get(hook_type, {}).get("voice")


def set_hook_voice(hook_type: str, voice_name: str | None) -> None:
    """Set the voice override for a specific hook type.

    Set voice_name to None to clear the override (use default voice).
    """
    if hook_type not in HOOK_TYPES:
        raise ValueError(f"Invalid hook type: {hook_type}. Valid: {HOOK_TYPES}")

    if voice_name is not None:
        available_voices = discover_voices()
        if voice_name not in available_voices:
            raise ValueError(f"Voice '{voice_name}' not found. Available: {available_voices}")

    config = load_config()
    if "hooks" not in config:
        config["hooks"] = {}
    if hook_type not in config["hooks"]:
        config["hooks"][hook_type] = {}

    if voice_name is None:
        # Clear the override
        config["hooks"][hook_type].pop("voice", None)
        # Clean up empty dicts
        if not config["hooks"][hook_type]:
            del config["hooks"][hook_type]
        if not config["hooks"]:
            del config["hooks"]
    else:
        config["hooks"][hook_type]["voice"] = voice_name

    save_config(config)


def get_effective_hook_voice(hook_type: str) -> str:
    """Get the effective voice for a hook (override or default)."""
    override = get_hook_voice(hook_type)
    return override if override else get_active_voice()


def get_all_hook_voices() -> dict[str, str | None]:
    """Get all hook voice overrides."""
    return {hook: get_hook_voice(hook) for hook in HOOK_TYPES}


# =============================================================================
# Reset to Defaults
# =============================================================================


def reset_compressor_to_defaults() -> None:
    """Reset compressor settings to factory defaults."""
    config = load_config()
    config["compressor"] = DEFAULT_COMPRESSOR.copy()
    save_config(config)


def reset_limiter_to_defaults() -> None:
    """Reset limiter settings to factory defaults."""
    config = load_config()
    config["limiter"] = DEFAULT_LIMITER.copy()
    save_config(config)


def reset_all_audio_to_defaults() -> None:
    """Reset all audio settings (compressor + limiter) to factory defaults."""
    config = load_config()
    config["compressor"] = DEFAULT_COMPRESSOR.copy()
    config["limiter"] = DEFAULT_LIMITER.copy()
    save_config(config)


def format_current_config() -> str:
    """Format the current configuration for display."""
    config = load_config()
    profile = config.get("active_profile", "default")
    interval = get_streaming_interval(profile)
    return f"Profile: {profile}\nStreaming interval: {interval}s"


def cmd_show() -> None:
    """Show current configuration."""
    print("Current TTS Configuration:")
    print(format_current_config())


def cmd_status() -> None:
    """Show current configuration and how to change it."""
    config = load_config()
    config_path = get_config_path()
    profile = config.get("active_profile", "default")
    interval = get_streaming_interval(profile)
    compressor = get_compressor_config()

    print("TTS Configuration Status")
    print("=" * 40)
    print(f"Config file: {config_path}")
    print(f"Active profile: {profile}")
    print()
    print("Playback Settings:")
    print(f"  Streaming interval: {interval}s")
    print()
    print("Compressor:")
    print(f"  Enabled: {compressor['enabled']}")
    print(f"  Gain: {compressor['gain_db']} dB")
    print()
    print("To configure TTS settings, run:")
    print("  uv run --directory $CLAUDE_PLUGIN_ROOT python scripts/tts_configurator.py")


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
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Usage: tts-config.py [status|show]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
