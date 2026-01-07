"""
Centralized logging configuration for Claude TTS plugin.

Provides unified logging to logs/plugin.log for all plugin entry points.
Library modules continue using logging.getLogger(__name__) and inherit
configuration from whichever entry point imports this module first.

Usage:
    from plugin_logging import setup_plugin_logging

    log = setup_plugin_logging()
    log.info("Message")

Environment variables:
    TTS_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR). Default: INFO
"""
import inspect
import logging
import os
from pathlib import Path

# Plugin-local log directory
_PLUGIN_ROOT = Path(__file__).parent.parent
LOG_DIR = _PLUGIN_ROOT / "logs"
LOG_FILE = LOG_DIR / "plugin.log"

# Track initialization to avoid duplicate basicConfig calls
_logging_configured = False


def setup_plugin_logging() -> logging.Logger:
    """
    Configure unified plugin logging to logs/plugin.log.

    Configures root logger on first call (subsequent calls are no-ops).
    Reads TTS_LOG_LEVEL env var (default: INFO).

    Returns:
        Logger instance using caller's module name.
    """
    global _logging_configured

    if not _logging_configured:
        # Ensure log directory exists
        LOG_DIR.mkdir(exist_ok=True)

        # Determine log level from env
        level_name = os.environ.get("TTS_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

        # Configure root logger
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            handlers=[logging.FileHandler(LOG_FILE)],
            force=True,  # Override any existing config
        )

        _logging_configured = True

    # Return logger for caller's module
    frame = inspect.currentframe()
    if frame is not None and frame.f_back is not None:
        module_name = frame.f_back.f_globals.get("__name__", "__main__")
    else:
        module_name = "__main__"

    return logging.getLogger(module_name)
