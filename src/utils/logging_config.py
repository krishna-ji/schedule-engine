"""Centralized logging configuration for the Schedule Engine."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import ClassVar


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    # ANSI color codes
    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color based on level."""
        if record.levelno >= logging.ERROR:
            color = self.COLORS["ERROR"]
        elif record.levelno >= logging.WARNING:
            color = self.COLORS["WARNING"]
        elif record.levelno >= logging.INFO:
            color = self.COLORS["INFO"]
        else:
            color = self.COLORS["DEBUG"]

        # Format: [LEVEL] module:line - message
        formatted = f"{color}[{record.levelname}]{self.COLORS['RESET']} "
        formatted += f"{record.name}:{record.lineno} - {record.getMessage()}"

        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


def setup_logging(
    level: str = "DEBUG", log_file: Path | None = None, verbose: bool = False
) -> logging.Logger:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        verbose: If True, set DEBUG level and show more detail

    Returns:
        Configured root logger

    Example:
        >>> logger = setup_logging(level="DEBUG", verbose=True)
        >>> logger.info("Starting GA run")
        >>> logger.debug("Population size: 100")
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if verbose:
        numeric_level = logging.DEBUG

    # Get root logger
    logger = logging.getLogger("src")
    logger.setLevel(numeric_level)

    # Remove existing handlers
    logger.handlers.clear()

    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = ColoredFormatter()
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if log_file specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = "src") -> logging.Logger:
    """
    Get logger instance for a module.

    Args:
        name: Logger name (typically __name__ of the module)

    Returns:
        Logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
    """
    return logging.getLogger(name)


# Module-level convenience functions
def debug(msg: str) -> None:
    """Log debug message."""
    get_logger().debug(msg)


def info(msg: str) -> None:
    """Log info message."""
    get_logger().info(msg)


def warning(msg: str) -> None:
    """Log warning message."""
    get_logger().warning(msg)


def error(msg: str, exc_info: bool = False) -> None:
    """
    Log error message.

    Args:
        msg: Error message
        exc_info: If True, include exception traceback
    """
    get_logger().error(msg, exc_info=exc_info)


def critical(msg: str, exc_info: bool = False) -> None:
    """
    Log critical error message.

    Args:
        msg: Critical error message
        exc_info: If True, include exception traceback
    """
    get_logger().critical(msg, exc_info=exc_info)


# ---------------------------------------------------------------------------
# Event tracking (merged from event_tracker.py)
# ---------------------------------------------------------------------------


class EventTracker:
    """Helper class to track events during a GA generation.

    Events tracked:
    - crossover_repair_applied, mutation_repair_applied
    - stagnation_detected, hypermutation_start, hypermutation_ended
    - population_restart, perfect_solution
    """

    def __init__(self) -> None:
        self.events: list[str] = []

    def add(self, event: str) -> None:
        """Add an event to the tracker."""
        self.events.append(event)

    def has_events(self) -> bool:
        """Check if any events were recorded."""
        return bool(self.events)

    def get_events(self) -> list[str]:
        """Get list of events."""
        return list(self.events)

    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()
