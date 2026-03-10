"""Centralized logging configuration for the Schedule Engine.

The **single source of truth** for all logging setup.  Every module uses::

    import logging
    logger = logging.getLogger(__name__)

and never touches handlers directly.  ``setup_unified_logging()`` is called
once per experiment run (from ``BaseExperiment.run()``) after the output
directory is known.

Features
--------
- **Rich console output** — colour-coded, with beautiful tracebacks via
  ``rich.logging.RichHandler``.
- **Rotating plain-text log file** — human-readable, auto-rotated at 10 MB
  with up to 5 back-ups.
- **Structured JSONL log file** — machine-readable companion log for
  downstream analytics and dashboards.
- **Context injection** — ``LogContext`` context-manager injects experiment /
  generation / phase info into every log record.
- **Performance helpers** — ``@log_duration`` and ``@log_call`` decorators
  for automatic function timing.
- **Noise suppression** — chatty third-party loggers (pymoo, matplotlib,
  PIL, urllib3…) are silenced to WARNING.
- **Run statistics** — ``LogStats`` counter tracks messages per level so
  the experiment can include a log-health summary in results.
- **Idempotent** — safe to call multiple times; handlers are replaced, never
  stacked.
"""

from __future__ import annotations

import functools
import json as _json
import logging
import logging.handlers
import sys
import threading
import time as _time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

# ── Public API ────────────────────────────────────────────────────────
__all__ = [
    "EventTracker",
    "LogContext",
    "LogStats",
    "get_logger",
    "is_logging_configured",
    "log_call",
    "log_duration",
    "setup_logging",
    "setup_unified_logging",
]

# ── Constants ─────────────────────────────────────────────────────────
_ROOT_LOGGER_NAME = "src"

# Human-readable format (file handler)
_LOG_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Noisy third-party loggers to silence
_NOISY_LOGGERS: tuple[str, ...] = (
    "matplotlib",
    "PIL",
    "urllib3",
    "pymoo",
    "numba",
    "torch",
    "tensorflow",
    "absl",
)

# Sentinel
_LOGGING_CONFIGURED = False

# Global stats collector (singleton)
_log_stats: LogStats | None = None


# =====================================================================
#  Rich console handler (beautiful terminal output)
# =====================================================================


def _make_rich_handler(level: int) -> logging.Handler:
    """Create a ``RichHandler`` for gorgeous console output.

    Falls back to a plain ``StreamHandler`` with ANSI colours if Rich is
    not installed (shouldn't happen — it's a pinned dependency).
    """
    try:
        from rich.console import Console
        from rich.logging import RichHandler
        from rich.traceback import install as install_rich_traceback

        # Force a sane minimum width so output never degrades to
        # char-per-line when the terminal reports a tiny width (e.g. CI,
        # piped output, or embedded tool terminals).
        console = Console(stderr=False, width=max(120, (Console().width or 120)))

        # Install rich tracebacks globally — beautiful crash reports
        install_rich_traceback(
            show_locals=True,
            width=120,
            word_wrap=True,
            extra_lines=3,
        )

        handler = RichHandler(
            level=level,
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
            tracebacks_word_wrap=True,
            log_time_format="[%Y-%m-%d %H:%M:%S]",
        )
        return handler

    except ImportError:
        # Graceful fallback — should not happen in this project
        return _make_ansi_handler(level)


class _AnsiFormatter(logging.Formatter):
    """Fallback console formatter with ANSI colour codes per log level."""

    COLORS: ClassVar[dict[str, str]] = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[1;35m",  # Bold Magenta
        "RESET": "\033[0m",
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
        reset = self.COLORS["RESET"]
        saved = record.levelname
        record.levelname = f"{color}{saved}{reset}"
        result = super().format(record)
        record.levelname = saved
        return result


def _make_ansi_handler(level: int) -> logging.Handler:
    """Plain ANSI-coloured ``StreamHandler`` — fallback when Rich is missing."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(_AnsiFormatter(_LOG_FMT, datefmt=_LOG_DATEFMT))
    return handler


# Keep old name available for imports
ColoredFormatter = _AnsiFormatter


# =====================================================================
#  Structured JSON log file handler
# =====================================================================


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object (JSONL).

    The output is designed for easy ingestion by ``jq``, Pandas, or any
    log-aggregation tool.
    """

    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno,
        }

        # Inject any context vars set via LogContext
        for attr in ("experiment", "generation", "phase", "extra_ctx"):
            val = getattr(record, attr, None)
            if val is not None:
                doc[attr] = val

        if record.exc_info and record.exc_info[0] is not None:
            doc["exception"] = self.formatException(record.exc_info)

        return _json.dumps(doc, default=str, ensure_ascii=False)


# =====================================================================
#  Context injection — LogContext
# =====================================================================


class _ContextFilter(logging.Filter):
    """Injects thread-local context variables into every ``LogRecord``.

    Used by ``LogContext`` so that all log lines emitted inside the
    context manager carry structured metadata.
    """

    _ctx: threading.local = threading.local()

    def filter(self, record: logging.LogRecord) -> bool:
        ctx: dict[str, Any] = getattr(self._ctx, "data", {})
        for key, value in ctx.items():
            setattr(record, key, value)
        return True

    @classmethod
    def push(cls, **kwargs: Any) -> None:
        if not hasattr(cls._ctx, "stack"):
            cls._ctx.stack = []
            cls._ctx.data = {}
        cls._ctx.stack.append(dict(cls._ctx.data))  # snapshot
        cls._ctx.data.update(kwargs)

    @classmethod
    def pop(cls) -> None:
        if hasattr(cls._ctx, "stack") and cls._ctx.stack:
            cls._ctx.data = cls._ctx.stack.pop()
        else:
            cls._ctx.data = {}


class LogContext:
    """Context manager that injects structured metadata into all log records.

    Usage::

        with LogContext(experiment="ga_01", phase="crossover"):
            logger.info("Operating on population")
            # → {"experiment": "ga_01", "phase": "crossover", "msg": "..."}

        # Or for generation tracking inside a loop:
        with LogContext(generation=42):
            logger.info("Best fitness improved")
    """

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def __enter__(self) -> LogContext:
        _ContextFilter.push(**self._kwargs)
        return self

    def __exit__(self, *exc: object) -> None:
        _ContextFilter.pop()


# =====================================================================
#  Log statistics collector
# =====================================================================


class LogStats(logging.Handler):
    """Lightweight handler that counts log messages per level.

    Attach to a logger and later call ``.summary()`` to get a dict
    like ``{"DEBUG": 142, "INFO": 89, "WARNING": 3, "ERROR": 0, ...}``.
    Useful for including log-health metadata in experiment results.
    """

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self._counts: dict[str, int] = {
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0,
        }
        self._lock = threading.Lock()
        self._first_ts: float | None = None
        self._last_ts: float | None = None

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            self._counts[record.levelname] = self._counts.get(record.levelname, 0) + 1
            if self._first_ts is None:
                self._first_ts = record.created
            self._last_ts = record.created

    def summary(self) -> dict[str, Any]:
        """Return a snapshot of log statistics."""
        with self._lock:
            total = sum(self._counts.values())
            elapsed = (
                round(self._last_ts - self._first_ts, 2)
                if self._first_ts and self._last_ts
                else 0.0
            )
            return {
                "counts": dict(self._counts),
                "total": total,
                "elapsed_s": elapsed,
                "rate_per_s": round(total / elapsed, 1) if elapsed > 0 else 0.0,
            }

    def reset(self) -> None:
        """Zero all counters."""
        with self._lock:
            for k in self._counts:
                self._counts[k] = 0
            self._first_ts = None
            self._last_ts = None


# =====================================================================
#  Performance decorators
# =====================================================================


def log_duration(
    logger_name: str | None = None,
    level: int = logging.DEBUG,
) -> Any:
    """Decorator that logs elapsed wall-clock time of a function call.

    Usage::

        @log_duration()
        def train_population(pop):
            ...

        @log_duration("src.ga.core", level=logging.INFO)
        def expensive_repair(individual):
            ...
    """

    def decorator(fn: Any) -> Any:
        _logger = logging.getLogger(logger_name or fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = _time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = _time.perf_counter() - t0
                _logger.log(
                    level,
                    "[bold] %s[/bold] completed in %.4fs",
                    fn.__qualname__,
                    elapsed,
                )

        return wrapper

    return decorator


def log_call(
    logger_name: str | None = None,
    level: int = logging.DEBUG,
    show_args: bool = False,
) -> Any:
    """Decorator that logs function entry, exit, and duration.

    Usage::

        @log_call(show_args=True)
        def mutate(individual, rate):
            ...
    """

    def decorator(fn: Any) -> Any:
        _logger = logging.getLogger(logger_name or fn.__module__)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if show_args:
                arg_str = ", ".join(
                    [repr(a) for a in args[:3]]
                    + [f"{k}={v!r}" for k, v in list(kwargs.items())[:3]]
                )
                if len(args) > 3 or len(kwargs) > 3:
                    arg_str += ", ..."
                _logger.log(level, "→ %s(%s)", fn.__qualname__, arg_str)
            else:
                _logger.log(level, "→ %s()", fn.__qualname__)

            t0 = _time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception:
                elapsed = _time.perf_counter() - t0
                _logger.log(
                    logging.ERROR,
                    "✗ %s FAILED after %.4fs",
                    fn.__qualname__,
                    elapsed,
                )
                raise
            elapsed = _time.perf_counter() - t0
            _logger.log(level, "← %s returned in %.4fs", fn.__qualname__, elapsed)
            return result

        return wrapper

    return decorator


# =====================================================================
#  Main setup function
# =====================================================================


def setup_unified_logging(
    log_file: Path | str | None = None,
    verbose: bool = False,
    *,
    console_level: int | None = None,
    json_log: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    suppress_noisy: bool = True,
    use_rich: bool = True,
    enable_stats: bool = True,
) -> logging.Logger:
    """Configure project-wide logging — the **one call to rule them all**.

    **Idempotent**: safe to call multiple times.  Existing handlers are
    removed and replaced so repeated calls never stack duplicate handlers.

    Parameters
    ----------
    log_file : Path | str | None
        Path to a human-readable log file.  A companion ``.jsonl`` file is
        created alongside it when *json_log* is True.  Parent directories
        are created automatically.  If *None*, only console output is used.
    verbose : bool
        When *True*, console shows DEBUG messages; otherwise INFO.
    console_level : int | None
        Explicit console log level.  Overrides *verbose* when set.
    json_log : bool
        Write a structured JSONL log file alongside the plain-text log.
    max_bytes : int
        Maximum size per log file before rotation (default 10 MB).
    backup_count : int
        Number of rotated log files to keep (default 5).
    suppress_noisy : bool
        Silence chatty third-party loggers (matplotlib, pymoo, etc.).
    use_rich : bool
        Use ``rich.logging.RichHandler`` for console output.  Falls back to
        plain ANSI colours if Rich is unavailable.
    enable_stats : bool
        Attach a ``LogStats`` handler that counts messages per level.

    Returns
    -------
    logging.Logger
        The ``"src"`` root logger.  All child loggers propagate to it.
    """
    global _LOGGING_CONFIGURED, _log_stats

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(logging.DEBUG)

    # ── Determine console level ───────────────────────────────────
    c_level = (
        console_level
        if console_level is not None
        else (logging.DEBUG if verbose else logging.INFO)
    )

    # ── Remove ALL existing handlers (idempotent) ─────────────────
    root.handlers.clear()

    # ── Shared context filter (injected into every handler) ────────
    ctx_filter = _ContextFilter()

    # ── Console handler ───────────────────────────────────────────
    if use_rich:
        console_handler = _make_rich_handler(c_level)
    else:
        console_handler = _make_ansi_handler(c_level)
    console_handler.addFilter(ctx_filter)
    root.addHandler(console_handler)

    # ── File handlers ─────────────────────────────────────────────
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Rotating plain-text log
        rfh = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        rfh.setLevel(logging.DEBUG)
        rfh.setFormatter(logging.Formatter(_LOG_FMT, datefmt=_LOG_DATEFMT))
        rfh.addFilter(ctx_filter)
        root.addHandler(rfh)

        # 2. Structured JSONL log (companion)
        if json_log:
            jsonl_path = log_path.with_suffix(".jsonl")
            jfh = logging.handlers.RotatingFileHandler(
                jsonl_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding="utf-8",
            )
            jfh.setLevel(logging.DEBUG)
            jfh.setFormatter(_JsonFormatter())
            jfh.addFilter(ctx_filter)
            root.addHandler(jfh)

    # ── Log stats collector ───────────────────────────────────────
    if enable_stats:
        _log_stats = LogStats()
        root.addHandler(_log_stats)

    # ── Suppress noisy third-party loggers ────────────────────────
    if suppress_noisy:
        for name in _NOISY_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)

    # ── Prevent double-output via the Python root logger ──────────
    root.propagate = False

    _LOGGING_CONFIGURED = True
    return root


# Backwards-compatible alias
setup_logging = setup_unified_logging


# =====================================================================
#  Public helpers
# =====================================================================


def get_logger(name: str = "src") -> logging.Logger:
    """Return a logger.  Typically called as ``get_logger(__name__)``."""
    return logging.getLogger(name)


def is_logging_configured() -> bool:
    """Return *True* if ``setup_unified_logging`` has been called."""
    return _LOGGING_CONFIGURED


def get_log_stats() -> LogStats | None:
    """Return the global ``LogStats`` instance (or *None* if stats are disabled)."""
    return _log_stats


def quick_setup(verbose: bool = False) -> logging.Logger:
    """Console-only logging setup for scripts and one-off tools.

    Equivalent to ``setup_unified_logging()`` with no file output and
    stats disabled — perfect for ``scripts/*.py`` entry points::

        from src.utils.logging_config import quick_setup
        logger = quick_setup()
    """
    return setup_unified_logging(
        verbose=verbose,
        json_log=False,
        enable_stats=False,
    )


# =====================================================================
#  Event tracking (merged from event_tracker.py)
# =====================================================================


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
