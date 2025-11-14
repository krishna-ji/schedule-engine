"""
Console output helpers for standardized messaging.

Provides consistent formatting for success, warning, error, and info messages
throughout the application using Rich console formatting.
"""

from rich.console import Console

console = Console()


def print_success(message: str, detail: str = None) -> None:
    """
    Print success message in green.

    Args:
        message: Main success message
        detail: Optional additional detail to print on next line

    Example:
        >>> print_success("validation passed")
        >>> print_success("schedule generated", "output/evaluation_123/")
    """
    console.print(f"[green][!ok] {message}[/green]")
    if detail:
        console.print(f"  [dim]{detail}[/dim]")


def print_warning(message: str, detail: str = None) -> None:
    """
    Print warning message in yellow.

    Args:
        message: Main warning message
        detail: Optional additional detail to print on next line

    Example:
        >>> print_warning("hard violations found", "count: 12")
    """
    console.print(f"[yellow][!warn] {message}[/yellow]")
    if detail:
        console.print(f"  [dim]{detail}[/dim]")


def print_error(message: str, detail: str = None) -> None:
    """
    Print error message in red.

    Args:
        message: Main error message
        detail: Optional additional detail to print on next line

    Example:
        >>> print_error("failed to load config", str(exception))
    """
    console.print(f"[bold red][!err] {message}[/bold red]")
    if detail:
        console.print(f"  [dim]{detail}[/dim]")


def print_info(message: str, detail: str = None) -> None:
    """
    Print informational message in cyan.

    Args:
        message: Main info message
        detail: Optional additional detail to print on next line

    Example:
        >>> print_info("parallel mode", "8 workers")
    """
    console.print(f"[cyan][!info] {message}[/cyan]")
    if detail:
        console.print(f"  [dim]{detail}[/dim]")


def print_section(title: str) -> None:
    """
    Print section header in bold cyan.

    Args:
        title: Section title

    Example:
        >>> print_section("genetic algorithm")
    """
    console.print()
    console.print(f"[bold cyan]{title}[/bold cyan]")
