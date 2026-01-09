"""
TTS Configuration module.

Manages persistent configuration for TTS playback settings.
Config is stored at ${PLUGIN_ROOT}/.config/config.json

Schema (v1):
    {
        "schema_version": 1,
        "profiles": {
            "default": {"streaming_interval": 0.5}
        },
        "active_profile": "default",
        "active_voice": "default",
        "voices": {
            "my_voice": {
                "compressor": {"gain_db": 10, "enabled": true, ...},
                "limiter": {"threshold_db": -1.0, ...}
            }
        },
        "hooks": {
            "stop": {"voice": "alternate_voice"},
            "permission_request": {"voice": "another_voice"}
        }
    }

Features:
- Per-voice compressor/limiter settings (audio stored in voices dict)
- Voice discovery from assets/*.wav
- Cascading config resolution (defaults -> voice-specific)
- Per-hook voice overrides
- Secure voice name validation

Note: Global getters/setters (get_compressor_config, set_compressor_setting, etc.)
operate on the active voice's settings. Use get_effective_compressor(voice_name) for
explicit voice lookups.

Can be overridden with TTS_CONFIG_PATH environment variable for testing.
"""
import copy
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

# Bundled per-voice defaults (ship with plugin, used as fallback)
# These are tuned settings for bundled voices that override factory defaults
BUNDLED_VOICE_DEFAULTS = {
    "alex": {
        "compressor": {
            "enabled": True,
            "input_gain_db": 0.0,
            "threshold_db": -12,
            "ratio": 2.0,
            "attack_ms": 10,
            "release_ms": 150,
            "gain_db": 4,
            "master_gain_db": -6.0,
        },
        "limiter": {
            "enabled": True,
            "threshold_db": -0.5,
            "release_ms": 40,
        },
    },
    "c3po": {
        "compressor": {
            "enabled": True,
            "input_gain_db": -3.0,
            "threshold_db": -12,
            "ratio": 2.0,
            "attack_ms": 10,
            "release_ms": 150,
            "gain_db": 4,
            "master_gain_db": 0.0,
        },
        "limiter": {
            "enabled": True,
            "threshold_db": -0.3,
            "release_ms": 20,
        },
    },
    "default": {
        "compressor": {
            "enabled": True,
            "input_gain_db": 6.0,
            "threshold_db": -12,
            "ratio": 2.0,
            "attack_ms": 10,
            "release_ms": 150,
            "gain_db": 4,
            "master_gain_db": 6.0,
        },
        "limiter": {
            "enabled": True,
            "threshold_db": -0.5,
            "release_ms": 40,
        },
    },
    "jerry": {
        "compressor": {
            "enabled": True,
            "input_gain_db": 0.0,
            "threshold_db": -12,
            "ratio": 2.0,
            "attack_ms": 10,
            "release_ms": 150,
            "gain_db": 4,
            "master_gain_db": -6.0,
        },
        "limiter": {
            "enabled": True,
            "threshold_db": -0.5,
            "release_ms": 40,
        },
    },
    "scarlett": {
        "compressor": {
            "enabled": True,
            "input_gain_db": 0.0,
            "threshold_db": -18,
            "ratio": 4.0,
            "attack_ms": 3,
            "release_ms": 50,
            "gain_db": 8,
            "master_gain_db": 0.0,
        },
        "limiter": {
            "enabled": True,
            "threshold_db": -0.3,
            "release_ms": 20,
        },
    },
    "snoop": {
        "compressor": {
            "enabled": True,
            "input_gain_db": 3.0,
            "threshold_db": -18,
            "ratio": 3.0,
            "attack_ms": 3,
            "release_ms": 50,
            "gain_db": 0,
            "master_gain_db": 3.0,
        },
        "limiter": {
            "enabled": True,
            "threshold_db": -0.5,
            "release_ms": 40,
        },
    },
}

# Valid characters for voice names (security)
VOICE_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')

# Default configuration values
DEFAULT_CONFIG = {
    "schema_version": 1,
    "profiles": {
        "default": {
            "streaming_interval": DEFAULT_STREAMING_INTERVAL,
        }
    },
    "active_profile": "default",
    "active_voice": "default",
    "voices": {},
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
    """Deep merge override into base, returning new dict without mutating base."""
    result = copy.deepcopy(base)
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
        return copy.deepcopy(DEFAULT_CONFIG)

    try:
        with open(config_path) as f:
            file_config = json.load(f)
        # Deep merge with defaults for any missing keys
        return _deep_merge(DEFAULT_CONFIG, file_config)
    except (json.JSONDecodeError, IOError):
        return copy.deepcopy(DEFAULT_CONFIG)


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


def get_streaming_interval(profile: str | None = None) -> float:
    """Get the streaming interval for a profile (default: active profile).

    Can be overridden via TTS_STREAMING_INTERVAL env var for experimentation.
    """
    # Allow env var override for experimentation
    env_val = os.environ.get("TTS_STREAMING_INTERVAL")
    if env_val:
        return float(env_val)

    config = load_config()
    if profile is None:
        profile = config.get("active_profile", "default")
    profiles = config.get("profiles", {})
    profile_config = profiles.get(profile, {})
    return profile_config.get("streaming_interval", DEFAULT_STREAMING_INTERVAL)


def set_streaming_interval(interval: float, profile: str | None = None) -> None:
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
    """Get compressor configuration for active voice. Re-reads from disk on each call."""
    config = load_config()
    voice_name = config.get("active_voice", "default")
    voices = config.get("voices", {})
    voice_config = voices.get(voice_name, {})
    compressor = voice_config.get("compressor", {})
    # Merge with defaults for any missing keys
    return {**DEFAULT_COMPRESSOR, **compressor}


def set_compressor_setting(key: str, value: float | bool) -> None:
    """Set a single compressor setting for the active voice."""
    if key not in DEFAULT_COMPRESSOR:
        valid_keys = ", ".join(DEFAULT_COMPRESSOR.keys())
        raise ValueError(f"Invalid compressor key '{key}'. Valid keys: {valid_keys}")

    config = load_config()
    voice_name = config.get("active_voice", "default")

    if "voices" not in config:
        config["voices"] = {}
    if voice_name not in config["voices"]:
        config["voices"][voice_name] = {}
    if "compressor" not in config["voices"][voice_name]:
        config["voices"][voice_name]["compressor"] = {}

    config["voices"][voice_name]["compressor"][key] = value
    save_config(config)


def set_compressor_gain(gain_db: float) -> None:
    """Set compressor makeup gain in dB."""
    set_compressor_setting("gain_db", gain_db)


def set_compressor_enabled(enabled: bool) -> None:
    """Enable or disable the compressor."""
    set_compressor_setting("enabled", enabled)


def get_limiter_config() -> dict:
    """Get limiter configuration for active voice. Re-reads from disk on each call."""
    config = load_config()
    voice_name = config.get("active_voice", "default")
    voices = config.get("voices", {})
    voice_config = voices.get(voice_name, {})
    limiter = voice_config.get("limiter", {})
    # Merge with defaults for any missing keys
    return {**DEFAULT_LIMITER, **limiter}


def set_limiter_setting(key: str, value: float | bool) -> None:
    """Set a single limiter setting for the active voice."""
    if key not in DEFAULT_LIMITER:
        valid_keys = ", ".join(DEFAULT_LIMITER.keys())
        raise ValueError(f"Invalid limiter key '{key}'. Valid keys: {valid_keys}")

    config = load_config()
    voice_name = config.get("active_voice", "default")

    if "voices" not in config:
        config["voices"] = {}
    if voice_name not in config["voices"]:
        config["voices"][voice_name] = {}
    if "limiter" not in config["voices"][voice_name]:
        config["voices"][voice_name]["limiter"] = {}

    config["voices"][voice_name]["limiter"][key] = value
    save_config(config)


# =============================================================================
# Voice Discovery and Management
# =============================================================================


def discover_voices() -> list[str]:
    """Discover all voice files in the assets directory.

    Finds both .safetensors (pre-computed embeddings) and .wav files.
    Returns list of unique voice names (without extension).
    When both formats exist for a voice, they are deduplicated.
    """
    assets_dir = _PLUGIN_ROOT / "assets"
    if not assets_dir.exists():
        return []

    voices = set()
    for ext in ("*.safetensors", "*.wav"):
        for f in assets_dir.glob(ext):
            voices.add(f.stem)
    return sorted(voices)


def get_voice_format(voice_name: str) -> str | None:
    """Return the format of a voice file.

    Returns:
        'safetensors' if .safetensors exists (preferred)
        'wav' if only .wav exists
        None if voice doesn't exist
    """
    assets_dir = _PLUGIN_ROOT / "assets"
    if (assets_dir / f"{voice_name}.safetensors").exists():
        return "safetensors"
    if (assets_dir / f"{voice_name}.wav").exists():
        return "wav"
    return None


def get_active_voice() -> str:
    """Get the currently active voice name."""
    config = load_config()
    return config.get("active_voice", "default")


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

    Resolution priority:
    1. .safetensors (pre-computed embeddings, preferred)
    2. .wav (source audio)

    Returns the full Path to the voice file.
    Raises ValueError for invalid or non-existent voices.
    """
    # Reject absolute paths
    if voice_name.startswith("/"):
        raise ValueError(f"Invalid voice name: {voice_name}")

    # Reject path traversal and special characters
    if not VOICE_NAME_PATTERN.match(voice_name):
        raise ValueError(f"Invalid voice name: {voice_name}")

    assets_dir = _PLUGIN_ROOT / "assets"
    resolved_assets = assets_dir.resolve()

    # Priority 1: Check for safetensors file
    safetensors_path = assets_dir / f"{voice_name}.safetensors"
    if safetensors_path.exists():
        resolved = safetensors_path.resolve()
        if not str(resolved).startswith(str(resolved_assets) + "/"):
            raise ValueError(f"Invalid voice name: {voice_name}")
        return safetensors_path

    # Priority 2: Fall back to wav file
    wav_path = assets_dir / f"{voice_name}.wav"
    if wav_path.exists():
        resolved = wav_path.resolve()
        if not str(resolved).startswith(str(resolved_assets) + "/"):
            raise ValueError(f"Invalid voice name: {voice_name}")
        return wav_path

    raise ValueError(f"Voice '{voice_name}' not found")


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


# =============================================================================
# Per-Voice Getters/Setters (explicit voice_name parameter)
# =============================================================================


def get_voice_compressor(voice_name: str) -> dict:
    """Get compressor configuration for a specific voice.

    Priority (later overrides earlier):
        1. Factory defaults (DEFAULT_COMPRESSOR)
        2. Bundled voice defaults (BUNDLED_VOICE_DEFAULTS)
        3. User settings from config (voices[name].compressor)
    """
    config = load_config()
    voices = config.get("voices", {})
    voice_config = voices.get(voice_name, {})
    voice_compressor = voice_config.get("compressor", {})

    # Get bundled defaults for this voice (if any)
    bundled = BUNDLED_VOICE_DEFAULTS.get(voice_name, {})
    bundled_compressor = bundled.get("compressor", {})

    # Merge: factory -> bundled -> user
    return {**DEFAULT_COMPRESSOR, **bundled_compressor, **voice_compressor}


def get_voice_limiter(voice_name: str) -> dict:
    """Get limiter configuration for a specific voice.

    Priority (later overrides earlier):
        1. Factory defaults (DEFAULT_LIMITER)
        2. Bundled voice defaults (BUNDLED_VOICE_DEFAULTS)
        3. User settings from config (voices[name].limiter)
    """
    config = load_config()
    voices = config.get("voices", {})
    voice_config = voices.get(voice_name, {})
    voice_limiter = voice_config.get("limiter", {})

    # Get bundled defaults for this voice (if any)
    bundled = BUNDLED_VOICE_DEFAULTS.get(voice_name, {})
    bundled_limiter = bundled.get("limiter", {})

    # Merge: factory -> bundled -> user
    return {**DEFAULT_LIMITER, **bundled_limiter, **voice_limiter}


def get_effective_compressor(voice_name: str | None = None) -> dict:
    """Get effective compressor settings for a voice.

    Uses cascading resolution: defaults -> voice-specific overrides.
    If voice_name is None, uses the active voice.
    """
    if voice_name is None:
        voice_name = get_active_voice()
    return get_voice_compressor(voice_name)


def get_effective_limiter(voice_name: str | None = None) -> dict:
    """Get effective limiter settings for a voice.

    Uses cascading resolution: defaults -> voice-specific overrides.
    If voice_name is None, uses the active voice.
    """
    if voice_name is None:
        voice_name = get_active_voice()
    return get_voice_limiter(voice_name)


def set_voice_compressor(voice_name: str, key: str, value: float | bool) -> None:
    """Set a single compressor setting for a specific voice.

    Args:
        voice_name: The voice to configure
        key: The compressor setting key (must be valid)
        value: The value to set
    """
    if key not in DEFAULT_COMPRESSOR:
        valid_keys = ", ".join(DEFAULT_COMPRESSOR.keys())
        raise ValueError(f"Invalid compressor key '{key}'. Valid keys: {valid_keys}")

    config = load_config()

    if "voices" not in config:
        config["voices"] = {}
    if voice_name not in config["voices"]:
        config["voices"][voice_name] = {}
    if "compressor" not in config["voices"][voice_name]:
        config["voices"][voice_name]["compressor"] = {}

    config["voices"][voice_name]["compressor"][key] = value
    save_config(config)


def set_voice_limiter(voice_name: str, key: str, value: float | bool) -> None:
    """Set a single limiter setting for a specific voice.

    Args:
        voice_name: The voice to configure
        key: The limiter setting key (must be valid)
        value: The value to set
    """
    if key not in DEFAULT_LIMITER:
        valid_keys = ", ".join(DEFAULT_LIMITER.keys())
        raise ValueError(f"Invalid limiter key '{key}'. Valid keys: {valid_keys}")

    config = load_config()

    if "voices" not in config:
        config["voices"] = {}
    if voice_name not in config["voices"]:
        config["voices"][voice_name] = {}
    if "limiter" not in config["voices"][voice_name]:
        config["voices"][voice_name]["limiter"] = {}

    config["voices"][voice_name]["limiter"][key] = value
    save_config(config)


def reset_voice_to_defaults(voice_name: str) -> None:
    """Reset a specific voice's compressor and limiter settings to factory defaults.

    Args:
        voice_name: The voice to reset
    """
    config = load_config()

    if "voices" not in config:
        config["voices"] = {}
    if voice_name not in config["voices"]:
        config["voices"][voice_name] = {}

    config["voices"][voice_name]["compressor"] = DEFAULT_COMPRESSOR.copy()
    config["voices"][voice_name]["limiter"] = DEFAULT_LIMITER.copy()
    save_config(config)


# =============================================================================
# Per-Hook Voice Configuration
# =============================================================================

# Available hook types that can have voice overrides
HOOK_TYPES = ["stop", "permission_request", "interview_question"]

# Human-friendly labels for hook types
HOOK_LABELS = {
    "stop": "Stop",
    "permission_request": "Permission",
    "interview_question": "Interview",
}

# Default prompts for each hook type
# Note: interview_question prompt is not used (question comes from tool input)
HOOK_DEFAULT_PROMPTS = {
    "stop": "[clear throat] Attention on deck.",
    "permission_request": "Claude needs permission to run {tool_name}.",
    "interview_question": "",  # Prompt comes from AskUserQuestion tool input
}


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
# Per-Hook Prompt Configuration
# =============================================================================


def get_default_hook_prompt(hook_type: str) -> str:
    """Get the default prompt for a hook type.

    Args:
        hook_type: The hook type ('stop' or 'permission_request')

    Returns:
        The default prompt string for the hook type

    Raises:
        ValueError: If hook_type is not valid
    """
    if hook_type not in HOOK_TYPES:
        raise ValueError(f"Invalid hook type: {hook_type}. Valid: {HOOK_TYPES}")
    return HOOK_DEFAULT_PROMPTS[hook_type]


def get_hook_prompt(hook_type: str) -> str | None:
    """Get the custom prompt override for a specific hook type.

    Returns None if no custom prompt is set (uses default).

    Args:
        hook_type: The hook type ('stop' or 'permission_request')

    Returns:
        Custom prompt string if set, None otherwise

    Raises:
        ValueError: If hook_type is not valid
    """
    if hook_type not in HOOK_TYPES:
        raise ValueError(f"Invalid hook type: {hook_type}. Valid: {HOOK_TYPES}")

    config = load_config()
    hooks = config.get("hooks", {})
    return hooks.get(hook_type, {}).get("prompt")


def set_hook_prompt(hook_type: str, prompt: str | None) -> None:
    """Set the custom prompt for a specific hook type.

    Set prompt to None to clear the custom prompt (use default).

    Args:
        hook_type: The hook type ('stop' or 'permission_request')
        prompt: The custom prompt string, or None to clear

    Raises:
        ValueError: If hook_type is not valid
    """
    if hook_type not in HOOK_TYPES:
        raise ValueError(f"Invalid hook type: {hook_type}. Valid: {HOOK_TYPES}")

    config = load_config()
    if "hooks" not in config:
        config["hooks"] = {}
    if hook_type not in config["hooks"]:
        config["hooks"][hook_type] = {}

    if prompt is None:
        # Clear the override
        config["hooks"][hook_type].pop("prompt", None)
        # Clean up empty dicts
        if not config["hooks"][hook_type]:
            del config["hooks"][hook_type]
        if not config["hooks"]:
            del config["hooks"]
    else:
        config["hooks"][hook_type]["prompt"] = prompt

    save_config(config)


def get_effective_hook_prompt(hook_type: str) -> str:
    """Get the effective prompt for a hook (custom override or default).

    Args:
        hook_type: The hook type ('stop' or 'permission_request')

    Returns:
        The effective prompt string to use

    Raises:
        ValueError: If hook_type is not valid
    """
    if hook_type not in HOOK_TYPES:
        raise ValueError(f"Invalid hook type: {hook_type}. Valid: {HOOK_TYPES}")

    custom = get_hook_prompt(hook_type)
    return custom if custom else get_default_hook_prompt(hook_type)


# =============================================================================
# Voice CRUD Operations
# =============================================================================


def get_voice_usage(name: str) -> dict:
    """Get usage information for a voice.

    Returns dict with:
        - is_active: bool - whether this is the active voice
        - hooks: list[str] - hook types using this voice
        - has_settings: bool - whether voice has custom settings

    Args:
        name: The voice name to check

    Returns:
        Dict with usage information
    """
    config = load_config()

    is_active = config.get("active_voice", "default") == name

    hooks_using = []
    for hook_type in HOOK_TYPES:
        hook_config = config.get("hooks", {}).get(hook_type, {})
        if hook_config.get("voice") == name:
            hooks_using.append(hook_type)

    has_settings = name in config.get("voices", {})

    return {
        "is_active": is_active,
        "hooks": hooks_using,
        "has_settings": has_settings,
    }


def delete_voice(name: str) -> None:
    """Delete a voice and clean up all config references.

    Removes:
        - Voice file(s) from assets (.safetensors and/or .wav)
        - Voice settings from config.voices
        - Voice defaults from config.voice_defaults
        - Hook voice overrides that reference this voice

    If deleting the active voice, sets active_voice to first remaining voice.
    Blocks deletion of the last remaining voice.

    Args:
        name: The voice name to delete

    Raises:
        ValueError: If voice name is invalid, not found, or is the last voice
    """

    if not VOICE_NAME_PATTERN.match(name):
        raise ValueError(f"Invalid voice name: {name}")

    available = discover_voices()
    if name not in available:
        raise ValueError(f"Voice '{name}' not found")

    if len(available) <= 1:
        raise ValueError(f"Cannot delete '{name}': it is the last voice")

    assets_dir = _PLUGIN_ROOT / "assets"

    safetensors_file = assets_dir / f"{name}.safetensors"
    if safetensors_file.exists():
        safetensors_file.unlink()

    wav_file = assets_dir / f"{name}.wav"
    if wav_file.exists():
        wav_file.unlink()

    config = load_config()

    if "voices" in config and name in config["voices"]:
        del config["voices"][name]

    # Clean up voice_defaults
    if "voice_defaults" in config and name in config["voice_defaults"]:
        del config["voice_defaults"][name]
        if not config["voice_defaults"]:
            del config["voice_defaults"]

    if "hooks" in config:
        for hook_type in list(config["hooks"].keys()):
            if config["hooks"][hook_type].get("voice") == name:
                del config["hooks"][hook_type]["voice"]
                if not config["hooks"][hook_type]:
                    del config["hooks"][hook_type]
        if not config["hooks"]:
            del config["hooks"]

    if config.get("active_voice") == name:
        remaining = [v for v in available if v != name]
        config["active_voice"] = sorted(remaining)[0]

    save_config(config)


def rename_voice(old_name: str, new_name: str) -> None:
    """Rename a voice and update all config references.

    Updates:
        - Voice file(s) in assets (.safetensors and/or .wav)
        - Voice settings key in config.voices
        - Voice defaults key in config.voice_defaults
        - active_voice if renaming the active voice
        - Hook voice overrides that reference this voice

    Args:
        old_name: The current voice name
        new_name: The new voice name

    Raises:
        ValueError: If old name not found, new name invalid, or new name exists
    """
    if not VOICE_NAME_PATTERN.match(new_name):
        raise ValueError(f"Invalid voice name: {new_name}")

    available = discover_voices()
    if old_name not in available:
        raise ValueError(f"Voice '{old_name}' not found")

    if new_name in available:
        raise ValueError(f"Voice '{new_name}' already exists")

    assets_dir = _PLUGIN_ROOT / "assets"

    old_safetensors = assets_dir / f"{old_name}.safetensors"
    if old_safetensors.exists():
        old_safetensors.rename(assets_dir / f"{new_name}.safetensors")

    old_wav = assets_dir / f"{old_name}.wav"
    if old_wav.exists():
        old_wav.rename(assets_dir / f"{new_name}.wav")

    config = load_config()

    if "voices" in config and old_name in config["voices"]:
        config["voices"][new_name] = config["voices"].pop(old_name)

    # Rename voice_defaults entry
    if "voice_defaults" in config and old_name in config["voice_defaults"]:
        config["voice_defaults"][new_name] = config["voice_defaults"].pop(old_name)

    if config.get("active_voice") == old_name:
        config["active_voice"] = new_name

    if "hooks" in config:
        for hook_type in config["hooks"]:
            if config["hooks"][hook_type].get("voice") == old_name:
                config["hooks"][hook_type]["voice"] = new_name

    save_config(config)


def generate_copy_name(source: str) -> str:
    """Generate a unique copy name for a voice.

    Returns source_copy, source_copy2, source_copy3, etc.
    """
    available = discover_voices()

    # Try _copy first
    candidate = f"{source}_copy"
    if candidate not in available:
        return candidate

    # Try _copy2, _copy3, etc.
    counter = 2
    while True:
        candidate = f"{source}_copy{counter}"
        if candidate not in available:
            return candidate
        counter += 1


def copy_voice(source: str, target: str | None = None) -> str:
    """Copy a voice to a new name.

    Copies:
        - Voice file(s) from assets (.safetensors and/or .wav)
        - Voice settings from config.voices (if any)
        - Voice defaults from config.voice_defaults (if any)

    Does NOT copy:
        - Hook voice overrides

    Args:
        source: The source voice name
        target: The target voice name (auto-generated if None)

    Returns:
        The target voice name (useful when auto-generated)

    Raises:
        ValueError: If source not found, target invalid, or target exists
    """
    import shutil

    available = discover_voices()
    if source not in available:
        raise ValueError(f"Voice '{source}' not found")

    # Auto-generate target name if not provided
    if target is None:
        target = generate_copy_name(source)
    else:
        if not VOICE_NAME_PATTERN.match(target):
            raise ValueError(f"Invalid voice name: {target}")
        if target in available:
            raise ValueError(f"Voice '{target}' already exists")

    assets_dir = _PLUGIN_ROOT / "assets"

    source_safetensors = assets_dir / f"{source}.safetensors"
    if source_safetensors.exists():
        shutil.copy2(source_safetensors, assets_dir / f"{target}.safetensors")

    source_wav = assets_dir / f"{source}.wav"
    if source_wav.exists():
        shutil.copy2(source_wav, assets_dir / f"{target}.wav")

    config = load_config()

    if "voices" in config and source in config["voices"]:
        import copy as copy_module
        config["voices"][target] = copy_module.deepcopy(config["voices"][source])

    # Copy voice_defaults if they exist
    if "voice_defaults" in config and source in config["voice_defaults"]:
        import copy as copy_module
        if "voice_defaults" not in config:
            config["voice_defaults"] = {}
        config["voice_defaults"][target] = copy_module.deepcopy(config["voice_defaults"][source])

    save_config(config)
    return target


# =============================================================================
# Voice Defaults (Captured Baseline Settings)
# =============================================================================


def get_voice_defaults(voice_name: str) -> dict | None:
    """Get the defaults for a specific voice.

    Priority:
        1. User-captured defaults from config (voice_defaults)
        2. Bundled defaults from code (BUNDLED_VOICE_DEFAULTS)
        3. None if no defaults exist

    Returns dict with 'compressor' and 'limiter' keys if defaults exist.
    """
    # Check user-captured defaults first
    config = load_config()
    voice_defaults = config.get("voice_defaults", {})
    if voice_name in voice_defaults:
        return voice_defaults[voice_name]

    # Fall back to bundled defaults
    return BUNDLED_VOICE_DEFAULTS.get(voice_name)


def capture_voice_defaults(voice_name: str) -> dict:
    """Capture current effective settings as the default for a voice.

    Takes the current effective compressor and limiter settings
    (merged with factory defaults) and stores them as the captured
    defaults for this voice.

    Args:
        voice_name: The voice to capture defaults for

    Returns:
        The captured defaults dict
    """
    # Get current effective settings (merged with factory defaults)
    effective_compressor = get_voice_compressor(voice_name)
    effective_limiter = get_voice_limiter(voice_name)

    captured = {
        "compressor": effective_compressor.copy(),
        "limiter": effective_limiter.copy(),
    }

    config = load_config()

    if "voice_defaults" not in config:
        config["voice_defaults"] = {}

    config["voice_defaults"][voice_name] = captured
    save_config(config)

    return captured


def capture_all_voice_defaults() -> dict[str, dict]:
    """Capture current settings as defaults for all voices with config.

    Returns dict mapping voice_name -> captured defaults.
    """
    config = load_config()
    voices = config.get("voices", {})

    captured_all = {}
    for voice_name in voices:
        captured_all[voice_name] = capture_voice_defaults(voice_name)

    return captured_all


def has_voice_defaults(voice_name: str) -> bool:
    """Check if a voice has captured defaults."""
    return get_voice_defaults(voice_name) is not None


def delete_voice_defaults(voice_name: str) -> None:
    """Delete captured defaults for a voice.

    Used when deleting a voice to clean up.
    """
    config = load_config()
    voice_defaults = config.get("voice_defaults", {})

    if voice_name in voice_defaults:
        del voice_defaults[voice_name]
        if not voice_defaults:
            del config["voice_defaults"]
        save_config(config)


def reset_voice_to_captured_defaults(voice_name: str) -> bool:
    """Reset a voice's settings to its captured defaults.

    If no captured defaults exist for this voice, returns False
    and does nothing. Use reset_voice_to_factory_defaults() instead.

    Args:
        voice_name: The voice to reset

    Returns:
        True if reset was successful, False if no captured defaults exist
    """
    captured = get_voice_defaults(voice_name)
    if captured is None:
        return False

    config = load_config()

    if "voices" not in config:
        config["voices"] = {}
    if voice_name not in config["voices"]:
        config["voices"][voice_name] = {}

    config["voices"][voice_name]["compressor"] = captured["compressor"].copy()
    config["voices"][voice_name]["limiter"] = captured["limiter"].copy()
    save_config(config)

    return True


# =============================================================================
# Reset to Defaults
# =============================================================================


def reset_voice_to_factory_defaults(voice_name: str) -> None:
    """Reset a specific voice's settings to factory defaults.

    This ignores any captured defaults and uses DEFAULT_COMPRESSOR/DEFAULT_LIMITER.
    """
    config = load_config()

    if "voices" not in config:
        config["voices"] = {}
    if voice_name not in config["voices"]:
        config["voices"][voice_name] = {}

    config["voices"][voice_name]["compressor"] = DEFAULT_COMPRESSOR.copy()
    config["voices"][voice_name]["limiter"] = DEFAULT_LIMITER.copy()
    save_config(config)


def reset_compressor_to_defaults() -> None:
    """Reset compressor settings for active voice to factory defaults."""
    config = load_config()
    voice_name = config.get("active_voice", "default")

    if "voices" not in config:
        config["voices"] = {}
    if voice_name not in config["voices"]:
        config["voices"][voice_name] = {}

    config["voices"][voice_name]["compressor"] = DEFAULT_COMPRESSOR.copy()
    save_config(config)


def reset_limiter_to_defaults() -> None:
    """Reset limiter settings for active voice to factory defaults."""
    config = load_config()
    voice_name = config.get("active_voice", "default")

    if "voices" not in config:
        config["voices"] = {}
    if voice_name not in config["voices"]:
        config["voices"][voice_name] = {}

    config["voices"][voice_name]["limiter"] = DEFAULT_LIMITER.copy()
    save_config(config)


def reset_all_audio_to_defaults() -> None:
    """Reset audio settings for active voice.

    If captured defaults exist for this voice, resets to those.
    Otherwise, resets to factory defaults.
    """
    config = load_config()
    voice_name = config.get("active_voice", "default")

    # Try captured defaults first
    if reset_voice_to_captured_defaults(voice_name):
        return

    # Fall back to factory defaults
    if "voices" not in config:
        config["voices"] = {}
    if voice_name not in config["voices"]:
        config["voices"][voice_name] = {}

    config["voices"][voice_name]["compressor"] = DEFAULT_COMPRESSOR.copy()
    config["voices"][voice_name]["limiter"] = DEFAULT_LIMITER.copy()
    save_config(config)


def format_current_config() -> str:
    """Format the current configuration for display."""
    config = load_config()
    profile = config.get("active_profile", "default")
    interval = get_streaming_interval(profile)
    return f"Profile: {profile}\nStreaming interval: {interval}s"


def cmd_full_status() -> None:
    """Display comprehensive TTS configuration status.

    Shows all TTS settings in a formatted view including:
    - Active voice with format indicator
    - Available voices
    - Hook overrides (voice and prompt per hook type)
    - Server status (running indicator, port, model)
    - Audio processing summary for active voice
    - Config file path
    - TUI launch command
    """
    # Import server utils for status check (optional dependency)
    is_server_alive = None
    try:
        from mlx_server_utils import (
            is_server_alive as _is_server_alive,
            TTS_SERVER_PORT,
            MLX_MODEL,
        )
        is_server_alive = _is_server_alive
    except ImportError:
        TTS_SERVER_PORT = 21099
        MLX_MODEL = "mlx-community/chatterbox-turbo-fp16"

    # Get all configuration data
    active_voice = get_active_voice()
    available_voices = discover_voices()
    voice_format = get_voice_format(active_voice)
    config_path = get_config_path().resolve()
    compressor = get_effective_compressor(active_voice)
    limiter = get_effective_limiter(active_voice)

    # Format voice format indicator
    format_indicator = ""
    if voice_format == "safetensors":
        format_indicator = " [embedded]"
    elif voice_format == "wav":
        format_indicator = " [wav]"

    # Check server status
    if is_server_alive is not None:
        server_running = is_server_alive()
    else:
        server_running = False

    # Build output
    print("TTS Configuration")
    print("═" * 67)
    print()

    # Voice Settings
    print("Voice Settings")
    print(f"  Active Voice: {active_voice}{format_indicator}")
    print(f"  Available: {', '.join(available_voices)}")
    print()

    # Mute Status
    try:
        from tts_mute import get_mute_status, format_remaining_time
        mute_status = get_mute_status()
        print("Mute Status")
        if mute_status.is_muted:
            remaining = format_remaining_time(mute_status.remaining_seconds)
            if mute_status.expires_at is not None:
                from datetime import datetime
                expires_dt = datetime.fromtimestamp(mute_status.expires_at)
                expires_str = expires_dt.strftime("%I:%M %p")
                print(f"  Status:  ● Muted for {remaining} (until {expires_str})")
            else:
                print("  Status:  ● Muted indefinitely")
            print("  Resume:  /tts-mute resume")
        else:
            print("  Status:  ○ Not muted")
        print()
    except ImportError:
        pass  # tts_mute not available

    # Hook Overrides
    print("Hook Overrides")
    for hook_type in HOOK_TYPES:
        label = HOOK_LABELS.get(hook_type, hook_type)
        hook_voice = get_hook_voice(hook_type)
        effective_prompt = get_effective_hook_prompt(hook_type)

        # Show voice (or 'default' if inheriting)
        voice_display = hook_voice if hook_voice else "default"

        # Truncate prompt if too long
        max_prompt_len = 45
        if len(effective_prompt) > max_prompt_len:
            prompt_display = effective_prompt[:max_prompt_len - 3] + "..."
        else:
            prompt_display = effective_prompt

        print(f"  {label + ':':<12} {voice_display:<10} → \"{prompt_display}\"")
    print()

    # Server Status
    print("Server Status")
    if server_running:
        status_indicator = "● Running"
    else:
        status_indicator = "○ Stopped"
    print(f"  Status: {status_indicator}")
    print(f"  Port:   {TTS_SERVER_PORT}")
    print(f"  Model:  {MLX_MODEL}")
    print()

    # Audio Processing (for active voice)
    print(f"Audio Processing ({active_voice})")

    # Compressor summary
    if compressor.get("enabled", True):
        comp_summary = (
            f"enabled (input: {compressor.get('input_gain_db', 0)}dB, "
            f"threshold: {compressor.get('threshold_db', -18)}dB, "
            f"ratio: {compressor.get('ratio', 3.0)}:1, "
            f"makeup: {compressor.get('gain_db', 8)}dB)"
        )
    else:
        comp_summary = "disabled"
    print(f"  Compressor: {comp_summary}")

    # Limiter summary
    if limiter.get("enabled", True):
        lim_summary = f"enabled (threshold: {limiter.get('threshold_db', -0.5)}dB)"
    else:
        lim_summary = "disabled"
    print(f"  Limiter:    {lim_summary}")

    # Master gain
    master_gain = compressor.get("master_gain_db", 0.0)
    print(f"  Master:     {master_gain}dB")
    print()

    # Config path
    print(f"Config: {config_path}")
    print()

    # TUI command
    print("─" * 67)
    print("Open TUI in a new terminal:")
    plugin_root = _PLUGIN_ROOT.resolve()
    print(f"  cd {plugin_root} && uv run --extra mlx python scripts/tts_tui.py")


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


def cmd_capture_defaults(voice_name: str | None = None) -> None:
    """Capture current settings as defaults for voice(s).

    If voice_name is provided, captures for that voice.
    Otherwise, captures for all voices with custom settings.
    """
    if voice_name:
        # Capture for specific voice
        captured = capture_voice_defaults(voice_name)
        print(f"Captured defaults for '{voice_name}':")
        print(f"  Compressor: {len(captured['compressor'])} settings")
        print(f"  Limiter: {len(captured['limiter'])} settings")
    else:
        # Capture for all voices
        config = load_config()
        voices = config.get("voices", {})

        if not voices:
            print("No voices with custom settings found.")
            return

        print(f"Capturing defaults for {len(voices)} voice(s)...")
        for name in sorted(voices.keys()):
            captured = capture_voice_defaults(name)
            print(f"  ✓ {name}")

        print("\nCaptured defaults saved to config.")


def cmd_list_defaults() -> None:
    """List voices with captured defaults."""
    config = load_config()
    voice_defaults = config.get("voice_defaults", {})

    if not voice_defaults:
        print("No captured defaults found.")
        print("Run 'capture-defaults' to capture current settings as defaults.")
        return

    print(f"Voices with captured defaults ({len(voice_defaults)}):")
    for name in sorted(voice_defaults.keys()):
        defaults = voice_defaults[name]
        comp_enabled = defaults.get("compressor", {}).get("enabled", True)
        lim_enabled = defaults.get("limiter", {}).get("enabled", True)
        status = []
        if comp_enabled:
            status.append("comp")
        if lim_enabled:
            status.append("lim")
        print(f"  {name} [{', '.join(status)}]")


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) < 2:
        cmd_full_status()
        return

    command = sys.argv[1].lower()

    if command == "full-status":
        cmd_full_status()
    elif command == "status":
        cmd_status()
    elif command == "show":
        cmd_show()
    elif command == "capture-defaults":
        # Optional voice name argument
        voice_name = sys.argv[2] if len(sys.argv) > 2 else None
        cmd_capture_defaults(voice_name)
    elif command == "list-defaults":
        cmd_list_defaults()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        print("Usage: tts-config.py [full-status|status|show|capture-defaults|list-defaults]", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
