"""Base experiment class — shared infrastructure for all experiment types.

Provides:
- Timestamped output directories
- Dual logging (file + console + JSONL) via unified logger
- Log context injection (experiment name, tag)
- Log statistics included in result metadata
- Timing & metadata
- JSON result export
- Reproducible seeding
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.utils.logging_config import LogContext, get_log_stats, setup_unified_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Abstract experiment runner.

    Subclasses must implement ``_execute()`` which returns a results dict.
    The ``run()`` method wraps it with logging, timing, and result saving.

    Parameters
    ----------
    name : str
        Human-readable experiment name (used for log headers).
    tag : str
        Short tag for output directory naming (e.g. ``"ga_01_baseline"``).
    seed : int
        Random seed for reproducibility.
    data_dir : Path | str | None
        Path to data directory.  Defaults to ``<project>/data``.
    output_dir : Path | str | None
        Explicit output directory.  If *None*, auto-generated as
        ``output/<tag>/<timestamp>``.
    verbose : bool
        Print detailed progress to console.
    """

    def __init__(
        self,
        *,
        name: str,
        tag: str,
        seed: int = 42,
        data_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
        verbose: bool = True,
    ) -> None:
        self.name = name
        self.tag = tag
        self.seed = seed
        self.verbose = verbose

        self.data_dir = Path(data_dir) if data_dir else PROJECT_ROOT / "data"

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = PROJECT_ROOT / "output" / tag / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._logger: logging.Logger | None = None
        self._log_file: Path | None = None

    # ── Logging ───────────────────────────────────────────────────

    @property
    def logger(self) -> logging.Logger:
        """Return the project-wide logger, lazily initialised on first access."""
        if self._logger is None:
            self._logger = self._setup_logging()
        return self._logger

    def _setup_logging(self) -> logging.Logger:
        """Initialise unified logging (console + per-run log file + JSONL).

        Uses the centralised ``setup_unified_logging`` so every ``src.*``
        module logger writes to the same file + console handlers.
        Also enables structured JSONL logging and log-health statistics.
        """
        self._log_file = self.output_dir / "run.log"
        setup_unified_logging(
            log_file=self._log_file,
            verbose=self.verbose,
        )
        return logging.getLogger(__name__)

    # ── Public API ────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Execute the experiment with timing, logging, and result saving.

        Returns the results dict.
        """
        self.logger.info("[START] %s  seed=%d  output=%s", self.name, self.seed, self.output_dir)

        t0 = time.time()
        try:
            with LogContext(experiment=self.tag, phase="execute"):
                results = self._execute()
        except Exception:
            self.logger.exception("Experiment failed")
            raise
        elapsed = time.time() - t0

        # ── Gather log statistics ─────────────────────────────────
        stats = get_log_stats()
        log_summary = stats.summary() if stats else None

        results["_meta"] = {
            "experiment": self.tag,
            "name": self.name,
            "timestamp": datetime.now(UTC).isoformat(),
            "seed": self.seed,
            "elapsed_s": round(elapsed, 2),
            "log_file": str(self._log_file) if self._log_file else None,
            "log_stats": log_summary,
        }

        # Save results
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to: {results_path}")
        self.logger.info(f"Elapsed: {elapsed:.1f}s")
        if log_summary:
            self.logger.info(
                "Log stats: %d total (%d warn, %d err)",
                log_summary["total"],
                log_summary["counts"].get("WARNING", 0),
                log_summary["counts"].get("ERROR", 0),
            )
        self.logger.info("[DONE] %s  elapsed=%.1fs", self.name, elapsed)

        return results

    # ── Subclass hook ─────────────────────────────────────────────

    @abstractmethod
    def _execute(self) -> dict[str, Any]:
        """Run the actual experiment logic.

        Returns a dict that will be merged with ``_meta`` and saved as JSON.
        """
        ...

    # ── Helpers ───────────────────────────────────────────────────

    def _config_dict(self) -> dict[str, Any]:
        """Gather all public non-callable attrs as a config summary."""
        skip = {"name", "tag", "output_dir", "data_dir", "timestamp", "verbose"}
        out: dict[str, Any] = {}
        for k, v in vars(self).items():
            if k.startswith("_") or k in skip or callable(v):
                continue
            if isinstance(v, Path):
                out[k] = str(v)
            else:
                out[k] = v
        return out
