"""
Neon Python Logging Module

Provides a unified logging interface for Neon Python code that integrates
with the C++ logging system.

Usage:
    from neon.logging import logger
    
    logger.info("Grid initialized")
    logger.debug("Debug details...")
    logger.warning("Something might be wrong")
    logger.error("An error occurred")
    
Configuration:
    import neon
    
    # Set log level
    neon.set_log_level("DEBUG")  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Enable/disable Python logging
    neon.set_python_logging(True)
    
    # Enable/disable C++ INFO logging
    neon.set_info_logging(True)
"""

import logging
import sys
from typing import Optional

# Create Neon logger
logger = logging.getLogger("neon")

# Default handler with colored output (if terminal supports it)
_handler: Optional[logging.Handler] = None
_is_enabled = True


class NeonFormatter(logging.Formatter):
    """Custom formatter with optional color support."""
    
    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and _supports_color()
    
    def format(self, record: logging.LogRecord) -> str:
        # Format: [TIME] [neon-py] [LEVEL] message
        timestamp = self.formatTime(record, "%H:%M:%S")
        level = record.levelname.ljust(5)
        
        if self.use_colors:
            color = self.COLORS.get(record.levelno, "")
            return f"{color}[{timestamp}] [neon-py] [{level}] {record.getMessage()}{self.RESET}"
        else:
            return f"[{timestamp}] [neon-py] [{level}] {record.getMessage()}"


def _supports_color() -> bool:
    """Check if the terminal supports color output."""
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    try:
        import curses
        curses.setupterm()
        return curses.tigetnum("colors") > 0
    except Exception:
        # Assume color support on common terminals
        return sys.platform != "win32" or "ANSICON" in __import__("os").environ


def _setup_default_handler():
    """Set up the default console handler."""
    global _handler
    if _handler is not None:
        return
    
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(NeonFormatter(use_colors=True))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def set_level(level: str) -> None:
    """
    Set the logging level for Neon Python logging.
    
    Args:
        level: One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    
    Example:
        >>> import neon
        >>> neon.set_log_level("DEBUG")  # Show all messages
        >>> neon.set_log_level("WARNING")  # Only warnings and above
    """
    _setup_default_handler()
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level_upper = level.upper()
    if level_upper not in level_map:
        raise ValueError(f"Invalid log level: {level}. Must be one of {list(level_map.keys())}")
    logger.setLevel(level_map[level_upper])


def set_enabled(enabled: bool) -> None:
    """
    Enable or disable Neon Python logging.
    
    Args:
        enabled: If True, logging is enabled. If False, all log messages are suppressed.
    
    Example:
        >>> import neon
        >>> neon.set_python_logging(False)  # Disable all Python log output
    """
    global _is_enabled
    _is_enabled = enabled
    if enabled:
        logger.disabled = False
    else:
        logger.disabled = True


def is_enabled() -> bool:
    """Check if Neon Python logging is enabled."""
    return _is_enabled and not logger.disabled


# Initialize default handler on module import
_setup_default_handler()


# Convenience functions for direct logging
def debug(msg: str, *args, **kwargs) -> None:
    """Log a debug message."""
    logger.debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs) -> None:
    """Log an info message."""
    logger.info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs) -> None:
    """Log a warning message."""
    logger.warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs) -> None:
    """Log an error message."""
    logger.error(msg, *args, **kwargs)


def critical(msg: str, *args, **kwargs) -> None:
    """Log a critical message."""
    logger.critical(msg, *args, **kwargs)
