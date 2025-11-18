"""
Centralized console service for consistent Rich console output.

Provides singleton console instance to avoid multiple console objects
scattered across the codebase, ensuring consistent formatting and
enabling easier output redirection for testing.
"""

from rich.console import Console
from typing import Optional
import sys


_console: Optional[Console] = None


def get_console() -> Console:
    """
    Get singleton console instance.

    Returns:
        Console: Rich console for formatted output

    Example:
        >>> from src.utils.console_service import get_console
        >>> console = get_console()
        >>> console.print("[green]Success![/green]")
    """
    global _console
    if _console is None:
        _console = Console()
    return _console


def reset_console() -> None:
    """
    Reset console instance (primarily for testing).

    Allows tests to configure console with custom settings
    (e.g., record=True for capturing output).
    """
    global _console
    _console = None


def configure_console(
    *,
    file=None,
    width: Optional[int] = None,
    force_terminal: Optional[bool] = None,
    force_jupyter: Optional[bool] = None,
    no_color: bool = False,
    record: bool = False,
) -> Console:
    """
    Configure console with specific settings.

    Args:
        file: File-like object for output (default: sys.stdout)
        width: Console width in characters
        force_terminal: Force terminal mode even if not a TTY
        force_jupyter: Force Jupyter mode
        no_color: Disable color output
        record: Enable output recording for testing

    Returns:
        Console: Configured console instance

    Example:
        >>> # Capture output for testing
        >>> console = configure_console(record=True)
        >>> console.print("Test output")
        >>> output = console.export_text()
    """
    global _console
    _console = Console(
        file=file or sys.stdout,
        width=width,
        force_terminal=force_terminal,
        force_jupyter=force_jupyter,
        no_color=no_color,
        record=record,
    )
    return _console


# Convenience re-exports for common usage
__all__ = [
    "get_console",
    "reset_console",
    "configure_console",
]
