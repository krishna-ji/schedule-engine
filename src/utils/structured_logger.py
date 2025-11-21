"""
Structured logging service for Schedule Engine.

Provides clean, readable logs with:
- Compact one-line format for key events
- File output (everything) + console output (filtered by level)
- Reduced redundancy (context shown once per group)
- Visual hierarchy with Rich formatting
- Thread-safe logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text


console = Console()


class CompactFormatter(logging.Formatter):
    """
    Compact formatter for file output (no colors, structured).

    Format: [timestamp] [LEVEL] [module] message
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with compact structure."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]

        # Extract context from record (set by StructuredLogger)
        context_parts = []
        if hasattr(record, "env_rank") and record.env_rank is not None:
            context_parts.append(f"env={record.env_rank}")
        if hasattr(record, "generation") and record.generation is not None:
            context_parts.append(f"gen={record.generation}")
        if hasattr(record, "step") and record.step is not None:
            context_parts.append(f"step={record.step}")

        context_str = f"[{' '.join(context_parts)}]" if context_parts else ""

        formatted = (
            f"[{timestamp}] [{record.levelname:8}] {context_str} {record.getMessage()}"
        )

        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"

        return formatted


class StructuredLogger:
    """
    Unified logging service with structured output.

    Features:
    - Console: Rich-formatted, configurable verbosity
    - File: Detailed structured logs, always DEBUG level
    - Context-aware: Environment, generation, step tracking
    - Thread-safe

    Example:
        >>> logger = StructuredLogger.get_logger("rl_training")
        >>> logger.info("Starting training", total_steps=10000)
        >>> logger.set_context(env_rank=0, generation=5)
        >>> logger.debug("Step completed", action="heuristic_1", reward=0.5)
    """

    _loggers: Dict[str, logging.Logger] = {}
    _context_stack: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def setup(
        cls,
        log_file: Optional[Path] = None,
        console_level: str = "DEBUG",
        file_level: str = "DEBUG",
        show_time: bool = True,
        show_path: bool = False,
    ) -> None:
        """
        Configure global logging settings.

        Args:
            log_file: Path to log file (auto-created with timestamp if None)
            console_level: Console verbosity (DEBUG, INFO, WARNING, ERROR)
            file_level: File verbosity (typically DEBUG for full detail)
            show_time: Show timestamp in console (default False for cleaner output)
            show_path: Show file path in console (default False)
        """
        # Create default log file if not specified
        if log_file is None:
            log_dir = Path("logs") / "training"
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"training_{timestamp}.log"
        else:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger("schedule_engine")
        root_logger.setLevel(
            logging.DEBUG
        )  # Capture everything, filter at handler level
        root_logger.handlers.clear()

        # Console handler (Rich formatting, minimal)
        console_handler = RichHandler(
            console=console,
            show_time=show_time,
            show_path=show_path,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )
        console_handler.setLevel(getattr(logging, console_level.upper(), logging.INFO))
        root_logger.addHandler(console_handler)

        # File handler (detailed, structured)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        file_handler.setFormatter(CompactFormatter())
        root_logger.addHandler(file_handler)

        root_logger.propagate = False

        console.print(f"[dim]Logs writing to: {log_file}[/dim]")

    @classmethod
    def get_logger(cls, name: str = "schedule_engine") -> "StructuredLogger":
        """
        Get or create logger instance for a module.

        Args:
            name: Logger name (typically __name__ or module path)

        Returns:
            StructuredLogger instance
        """
        # Ensure logger name is under schedule_engine hierarchy
        if not name.startswith("schedule_engine"):
            if name == "__main__" or name == "test":
                logger_name = f"schedule_engine.{name}"
            else:
                logger_name = name
        else:
            logger_name = name

        if logger_name not in cls._loggers:
            logger = logging.getLogger(logger_name)
            cls._loggers[logger_name] = logger
            cls._context_stack[logger_name] = {}

        return cls(logger_name)

    def __init__(self, name: str):
        """Initialize structured logger (use get_logger() instead)."""
        self.name = name
        self._logger = logging.getLogger(name)
        self._context: Dict[str, Any] = {}

    def set_context(
        self,
        env_rank: Optional[int] = None,
        generation: Optional[int] = None,
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Set context for subsequent log messages.

        Context persists until explicitly changed or cleared.

        Args:
            env_rank: Environment rank (for parallel training)
            generation: GA generation number
            step: RL step number
            **kwargs: Additional context key-value pairs
        """
        if env_rank is not None:
            self._context["env_rank"] = env_rank
        if generation is not None:
            self._context["generation"] = generation
        if step is not None:
            self._context["step"] = step
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Clear all context."""
        self._context.clear()

    def _add_context_to_record(self, record: logging.LogRecord) -> None:
        """Inject context into log record."""
        for key, value in self._context.items():
            setattr(record, key, value)

    def _log(self, level: int, msg: str, *args, **kwargs) -> None:
        """Internal log with context injection."""
        # Extract exc_info if present
        exc_info = kwargs.pop("exc_info", False)

        # Create log record
        if args:
            msg = msg % args

        # Add context kwargs to message
        if kwargs:
            context_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
            msg = f"{msg} {context_str}"

        # Log with context
        extra = self._context.copy()
        self._logger.log(level, msg, extra=extra, exc_info=exc_info)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log info message."""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def success(self, msg: str, *args, **kwargs) -> None:
        """Log success message (INFO level with green formatting)."""
        self.info(f"[green]✓[/green] {msg}", *args, **kwargs)

    def action(self, action_name: str, success: bool, **kwargs) -> None:
        """
        Log action execution result.

        Args:
            action_name: Name of action/heuristic
            success: Whether action succeeded
            **kwargs: Additional context (reward, duration, etc.)
        """
        status = "[green]✓[/green]" if success else "[red]✗[/red]"
        self.info(f"Action {action_name} {status}", **kwargs)

    def step_summary(
        self,
        action: str,
        reward: float,
        success: bool,
        best_fitness: float,
        diversity: float,
        stagnation: int,
        duration_ms: float,
        improvement: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Log compact step summary (one line).

        Args:
            action: Action name
            reward: Step reward
            success: Action success flag
            best_fitness: Best fitness in population
            diversity: Population diversity
            stagnation: Generations without improvement
            duration_ms: Step duration in milliseconds
            improvement: Fitness improvement (if any)
            **kwargs: Additional metrics
        """
        status = "[green]✓[/green]" if success else "[red]✗[/red]"
        improvement_str = f"[cyan]+{improvement:.1f}[/cyan]" if improvement else ""

        msg = (
            f"{status} {action} "
            f"r={reward:.3f} "
            f"best={best_fitness:.1f} "
            f"div={diversity:.1f} "
            f"stag={stagnation} "
            f"{improvement_str} "
            f"[dim]{duration_ms:.1f}ms[/dim]"
        )

        self.info(msg, **kwargs)


# Module-level convenience functions
_default_logger: Optional[StructuredLogger] = None


def get_logger(name: str = "schedule_engine") -> StructuredLogger:
    """Get structured logger instance."""
    return StructuredLogger.get_logger(name)


def setup_logging(
    log_file: Optional[Path] = None,
    console_level: str = "DEBUG",
    file_level: str = "DEBUG",
    **kwargs,
) -> None:
    """Setup global logging configuration."""
    StructuredLogger.setup(
        log_file=log_file,
        console_level=console_level,
        file_level=file_level,
        **kwargs,
    )
