"""Comprehensive tests for the unified logging system.

Covers:
- Logger setup idempotence (calling setup twice does not duplicate handlers)
- Per-run log file creation (plain text + JSONL)
- Rotating file handler behaviour
- Log context injection (LogContext)
- Performance decorators (log_duration, log_call)
- LogStats message counting
- Child logger propagation
- No duplicate log lines
- No print() calls in src/experiments/*.py
- Verbose flag controls console level
- quick_setup() for scripts
- Noise suppression for third-party loggers
- JSON log structure validation
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
from pathlib import Path

import pytest

# ── Helpers ───────────────────────────────────────────────────────────


def _flush_root() -> None:
    """Flush all handlers on the 'src' root logger."""
    for handler in logging.getLogger("src").handlers:
        handler.flush()


def _reset_logging() -> None:
    """Reset logging state between tests."""
    import src.utils.logging_config as lc

    root = logging.getLogger("src")
    root.handlers.clear()
    lc._LOGGING_CONFIGURED = False
    lc._log_stats = None


@pytest.fixture(autouse=True)
def _clean_logging():
    """Reset logging before and after each test."""
    _reset_logging()
    yield
    _reset_logging()


# =====================================================================
#  1. Core setup tests
# =====================================================================


def test_setup_unified_logging_idempotent(tmp_path: Path) -> None:
    """Calling setup_unified_logging twice must NOT stack duplicate handlers."""
    from src.utils.logging_config import setup_unified_logging

    log_file = tmp_path / "test_idem.log"

    root1 = setup_unified_logging(log_file=log_file, verbose=False)
    n_handlers_1 = len(root1.handlers)

    root2 = setup_unified_logging(log_file=log_file, verbose=False)
    n_handlers_2 = len(root2.handlers)

    assert root1 is root2, "Should return the same logger instance"
    assert n_handlers_1 == n_handlers_2, (
        f"Handler count changed: {n_handlers_1} → {n_handlers_2} — "
        "setup_unified_logging is not idempotent"
    )


def test_setup_creates_log_file(tmp_path: Path) -> None:
    """setup_unified_logging must create the log file on disk."""
    from src.utils.logging_config import setup_unified_logging

    log_file = tmp_path / "subdir" / "run.log"
    setup_unified_logging(log_file=log_file, verbose=False)

    logger = logging.getLogger("src.test_logging_probe")
    logger.info("hello from test")
    _flush_root()

    assert log_file.exists(), f"Log file not created: {log_file}"
    content = log_file.read_text(encoding="utf-8")
    assert "hello from test" in content


def test_child_logger_propagates_to_file(tmp_path: Path) -> None:
    """A child logger (e.g. src.experiments.callback_core) should write to
    the unified log file via propagation."""
    from src.utils.logging_config import setup_unified_logging

    log_file = tmp_path / "propagation.log"
    setup_unified_logging(log_file=log_file, verbose=True)

    child_logger = logging.getLogger("src.experiments.callback_core")
    child_logger.info("callback says hello")
    _flush_root()

    content = log_file.read_text(encoding="utf-8")
    assert "callback says hello" in content
    assert "src.experiments.callback_core" in content


def test_no_duplicate_lines(tmp_path: Path) -> None:
    """A single logger.info call must produce exactly ONE line in the file."""
    from src.utils.logging_config import setup_unified_logging

    log_file = tmp_path / "dedup.log"
    setup_unified_logging(log_file=log_file, verbose=False)

    logger = logging.getLogger("src.dedup_test")
    unique_msg = "UNIQUE_DEDUP_TOKEN_12345"
    logger.info(unique_msg)
    _flush_root()

    content = log_file.read_text(encoding="utf-8")
    count = content.count(unique_msg)
    assert count == 1, f"Expected 1 occurrence, got {count} — duplicate handlers?"


def test_verbose_controls_console_level(tmp_path: Path) -> None:
    """verbose=True → console handler at DEBUG, verbose=False → INFO."""
    from src.utils.logging_config import setup_unified_logging

    setup_unified_logging(log_file=tmp_path / "v.log", verbose=True)
    root = logging.getLogger("src")
    # Find non-file, non-stats handlers (i.e. the console handler)
    console_handlers = [
        h
        for h in root.handlers
        if hasattr(h, "level")
        and not isinstance(h, logging.FileHandler)
        and not isinstance(h, logging.handlers.RotatingFileHandler)
        and not hasattr(h, "summary")  # exclude LogStats
    ]
    assert console_handlers, "No console handler found"
    assert console_handlers[0].level == logging.DEBUG

    setup_unified_logging(log_file=tmp_path / "v2.log", verbose=False)
    root = logging.getLogger("src")
    console_handlers = [
        h
        for h in root.handlers
        if hasattr(h, "level")
        and not isinstance(h, logging.FileHandler)
        and not isinstance(h, logging.handlers.RotatingFileHandler)
        and not hasattr(h, "summary")
    ]
    assert console_handlers[0].level == logging.INFO


# =====================================================================
#  2. JSONL structured logging
# =====================================================================


def test_jsonl_log_created(tmp_path: Path) -> None:
    """A .jsonl companion file should be created alongside the plain log."""
    from src.utils.logging_config import setup_unified_logging

    log_file = tmp_path / "run.log"
    setup_unified_logging(log_file=log_file, json_log=True)

    logger = logging.getLogger("src.jsonl_test")
    logger.info("structured message")
    _flush_root()

    jsonl_path = log_file.with_suffix(".jsonl")
    assert jsonl_path.exists(), f"JSONL log not created: {jsonl_path}"

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1

    # Validate JSON structure
    doc = json.loads(lines[-1])
    assert doc["level"] == "INFO"
    assert doc["msg"] == "structured message"
    assert "ts" in doc
    assert doc["logger"] == "src.jsonl_test"


def test_jsonl_disabled(tmp_path: Path) -> None:
    """json_log=False should NOT create a .jsonl file."""
    from src.utils.logging_config import setup_unified_logging

    log_file = tmp_path / "nojson.log"
    setup_unified_logging(log_file=log_file, json_log=False)

    logger = logging.getLogger("src.nojson")
    logger.info("plain only")
    _flush_root()

    jsonl_path = log_file.with_suffix(".jsonl")
    assert not jsonl_path.exists()


# =====================================================================
#  3. LogContext injection
# =====================================================================


def test_log_context_injects_metadata(tmp_path: Path) -> None:
    """LogContext should inject experiment/phase metadata into JSONL records."""
    from src.utils.logging_config import LogContext, setup_unified_logging

    log_file = tmp_path / "ctx.log"
    setup_unified_logging(log_file=log_file, json_log=True)

    logger = logging.getLogger("src.ctx_test")
    with LogContext(experiment="ga_01", phase="crossover"):
        logger.info("inside context")
    _flush_root()

    jsonl_path = log_file.with_suffix(".jsonl")
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    doc = json.loads(lines[-1])

    assert doc["experiment"] == "ga_01"
    assert doc["phase"] == "crossover"
    assert doc["msg"] == "inside context"


def test_log_context_nesting(tmp_path: Path) -> None:
    """Nested LogContext should stack and restore correctly."""
    from src.utils.logging_config import LogContext, setup_unified_logging

    log_file = tmp_path / "nest.log"
    setup_unified_logging(log_file=log_file, json_log=True)

    logger = logging.getLogger("src.nest_test")

    with LogContext(experiment="ga_02"):
        logger.info("outer")
        with LogContext(phase="mutation", generation=10):
            logger.info("inner")
        logger.info("back to outer")
    _flush_root()

    jsonl_path = log_file.with_suffix(".jsonl")
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    docs = [json.loads(line) for line in lines]

    # Filter to only our test messages
    our_docs = [d for d in docs if d["logger"] == "src.nest_test"]
    assert len(our_docs) == 3

    # Outer context
    assert our_docs[0]["experiment"] == "ga_02"
    assert "phase" not in our_docs[0]

    # Inner (merged) context
    assert our_docs[1]["experiment"] == "ga_02"
    assert our_docs[1]["phase"] == "mutation"
    assert our_docs[1]["generation"] == 10

    # After inner exits — restored to outer
    assert our_docs[2]["experiment"] == "ga_02"
    assert "phase" not in our_docs[2]


# =====================================================================
#  4. LogStats counter
# =====================================================================


def test_log_stats_counting(tmp_path: Path) -> None:
    """LogStats should accurately count messages per level."""
    from src.utils.logging_config import get_log_stats, setup_unified_logging

    setup_unified_logging(log_file=tmp_path / "stats.log", enable_stats=True)
    stats = get_log_stats()
    assert stats is not None

    logger = logging.getLogger("src.stats_test")
    logger.debug("d1")
    logger.debug("d2")
    logger.info("i1")
    logger.warning("w1")
    logger.error("e1")

    summary = stats.summary()
    assert summary["counts"]["DEBUG"] >= 2
    assert summary["counts"]["INFO"] >= 1
    assert summary["counts"]["WARNING"] >= 1
    assert summary["counts"]["ERROR"] >= 1
    assert summary["total"] >= 5


def test_log_stats_disabled(tmp_path: Path) -> None:
    """enable_stats=False should not attach a LogStats handler."""
    from src.utils.logging_config import get_log_stats, setup_unified_logging

    setup_unified_logging(
        log_file=tmp_path / "nostats.log",
        enable_stats=False,
    )
    stats = get_log_stats()
    assert stats is None


def test_log_stats_reset(tmp_path: Path) -> None:
    """LogStats.reset() should zero all counters."""
    from src.utils.logging_config import get_log_stats, setup_unified_logging

    setup_unified_logging(log_file=tmp_path / "sr.log", enable_stats=True)
    stats = get_log_stats()
    assert stats is not None

    logger = logging.getLogger("src.reset_test")
    logger.info("before reset")
    assert stats.summary()["total"] >= 1

    stats.reset()
    assert stats.summary()["total"] == 0


# =====================================================================
#  5. Performance decorators
# =====================================================================


def test_log_duration_decorator(tmp_path: Path) -> None:
    """@log_duration should log the elapsed time of a function."""
    from src.utils.logging_config import log_duration, setup_unified_logging

    log_file = tmp_path / "duration.log"
    setup_unified_logging(log_file=log_file)

    @log_duration("src.test_perf", level=logging.INFO)
    def fast_func():
        return 42

    result = fast_func()
    assert result == 42
    _flush_root()

    content = log_file.read_text(encoding="utf-8")
    assert "fast_func" in content
    assert "completed in" in content


def test_log_call_decorator(tmp_path: Path) -> None:
    """@log_call should log entry and exit of a function."""
    from src.utils.logging_config import log_call, setup_unified_logging

    log_file = tmp_path / "call.log"
    setup_unified_logging(log_file=log_file)

    @log_call("src.test_call", level=logging.INFO, show_args=True)
    def add(a, b):
        return a + b

    result = add(3, 4)
    assert result == 7
    _flush_root()

    content = log_file.read_text(encoding="utf-8")
    assert "add" in content
    assert "returned in" in content


def test_log_call_captures_exceptions(tmp_path: Path) -> None:
    """@log_call should log FAILED when the function raises."""
    from src.utils.logging_config import log_call, setup_unified_logging

    log_file = tmp_path / "call_fail.log"
    setup_unified_logging(log_file=log_file)

    @log_call("src.test_fail", level=logging.INFO)
    def bad_func():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        bad_func()
    _flush_root()

    content = log_file.read_text(encoding="utf-8")
    assert "FAILED" in content


# =====================================================================
#  6. quick_setup for scripts
# =====================================================================


def test_quick_setup_console_only() -> None:
    """quick_setup() should configure logging with console only, no files."""
    from src.utils.logging_config import is_logging_configured, quick_setup

    root = quick_setup()
    assert is_logging_configured()
    assert root.name == "src"

    # Should have exactly 0 file handlers
    file_handlers = [
        h
        for h in root.handlers
        if isinstance(h, (logging.FileHandler, logging.handlers.RotatingFileHandler))
    ]
    assert len(file_handlers) == 0


# =====================================================================
#  7. Noise suppression
# =====================================================================


def test_noisy_loggers_suppressed(tmp_path: Path) -> None:
    """Chatty third-party loggers should be set to WARNING."""
    from src.utils.logging_config import setup_unified_logging

    setup_unified_logging(log_file=tmp_path / "noise.log", suppress_noisy=True)

    for name in ("matplotlib", "PIL", "urllib3", "pymoo"):
        assert (
            logging.getLogger(name).level >= logging.WARNING
        ), f"Logger '{name}' not suppressed"


def test_noise_suppression_disabled(tmp_path: Path) -> None:
    """suppress_noisy=False should leave third-party loggers untouched."""
    from src.utils.logging_config import setup_unified_logging

    # Set them to DEBUG first
    for name in ("matplotlib", "PIL"):
        logging.getLogger(name).setLevel(logging.DEBUG)

    setup_unified_logging(
        log_file=tmp_path / "no_noise.log",
        suppress_noisy=False,
    )

    for name in ("matplotlib", "PIL"):
        assert logging.getLogger(name).level == logging.DEBUG


# =====================================================================
#  8. Rotating file handler
# =====================================================================


def test_rotating_file_handler_used(tmp_path: Path) -> None:
    """The file handler should be a RotatingFileHandler, not plain FileHandler."""
    from src.utils.logging_config import setup_unified_logging

    setup_unified_logging(log_file=tmp_path / "rotate.log")
    root = logging.getLogger("src")

    rotating_handlers = [
        h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert len(rotating_handlers) >= 1, "No RotatingFileHandler found"


# =====================================================================
#  9. Code quality — no print() in src/experiments/
# =====================================================================


def test_no_print_in_experiments() -> None:
    """src/experiments/*.py must not contain bare print() calls."""
    experiments_dir = Path(__file__).resolve().parent.parent / "src" / "experiments"
    violations: list[str] = []

    for py_file in sorted(experiments_dir.glob("*.py")):
        for i, line in enumerate(py_file.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.lstrip()
            if (
                stripped.startswith("#")
                or stripped.startswith('"""')
                or stripped.startswith(">>>")
            ):
                continue
            if re.search(r"\bprint\s*\(", stripped) and "console.print" not in line:
                violations.append(f"{py_file.name}:{i}: {line.rstrip()}")

    assert not violations, (
        "Found print() calls in src/experiments/ "
        "(should use logger instead):\n" + "\n".join(violations)
    )


# =====================================================================
#  10. is_logging_configured sentinel
# =====================================================================


def test_is_logging_configured_sentinel() -> None:
    """is_logging_configured() should return False before setup, True after."""
    from src.utils.logging_config import is_logging_configured, setup_unified_logging

    assert not is_logging_configured()
    setup_unified_logging()
    assert is_logging_configured()


# =====================================================================
#  11. Context filter is always attached
# =====================================================================


def test_context_filter_attached_to_handlers(tmp_path: Path) -> None:
    """The _ContextFilter should be attached to every handler."""
    from src.utils.logging_config import setup_unified_logging

    setup_unified_logging(log_file=tmp_path / "f.log")
    root = logging.getLogger("src")
    for handler in root.handlers:
        if hasattr(handler, "summary"):  # skip LogStats
            continue
        assert (
            len(handler.filters) >= 1
        ), f"Handler {handler.__class__.__name__} has no context filter"


# =====================================================================
#  12. Backwards compatibility
# =====================================================================


def test_setup_logging_alias() -> None:
    """setup_logging should be an alias for setup_unified_logging."""
    from src.utils.logging_config import setup_logging, setup_unified_logging

    assert setup_logging is setup_unified_logging


def test_colored_formatter_alias() -> None:
    """ColoredFormatter should still be importable (backwards compat)."""
    from src.utils.logging_config import ColoredFormatter

    assert ColoredFormatter is not None
