"""Utility functions and services.

Usage::

    from src.utils import get_console, setup_unified_logging, get_logger
    from src.utils import LogContext, log_duration, log_call, quick_setup
"""

from __future__ import annotations

from src.utils.console_service import get_console
from src.utils.logging_config import (
    LogContext,
    LogStats,
    get_log_stats,
    get_logger,
    log_call,
    log_duration,
    quick_setup,
    setup_unified_logging,
)

__all__ = [
    "LogContext",
    "LogStats",
    "get_console",
    "get_log_stats",
    "get_logger",
    "log_call",
    "log_duration",
    "quick_setup",
    "setup_unified_logging",
]
