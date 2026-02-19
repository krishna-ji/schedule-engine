"""Base experiment class — shared infrastructure for all experiment types.

Provides:
- Timestamped output directories
- Dual logging (file + console)
- Timing & metadata
- JSON result export
- Reproducible seeding
"""

from __future__ import annotations

import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


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

    # ── Logging ───────────────────────────────────────────────────

    @property
    def logger(self) -> logging.Logger:
        """Lazily create a logger on first access."""
        if self._logger is None:
            self._logger = self._setup_logging()
        return self._logger

    def _setup_logging(self) -> logging.Logger:
        """Create file + console logger."""
        log_file = self.output_dir / f"{self.tag}.log"

        fmt = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO if self.verbose else logging.WARNING)
        ch.setFormatter(fmt)

        logger = logging.getLogger(self.tag)
        logger.setLevel(logging.DEBUG)
        # Avoid duplicate handlers on repeated calls
        if not logger.handlers:
            logger.addHandler(fh)
            logger.addHandler(ch)

        return logger

    # ── Public API ────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Execute the experiment with timing, logging, and result saving.

        Returns the results dict.
        """
        self.logger.info("=" * 60)
        self.logger.info(self.name.upper())
        self.logger.info("=" * 60)
        self.logger.info(f"Seed:   {self.seed}")
        self.logger.info(f"Output: {self.output_dir}")

        t0 = time.time()
        try:
            results = self._execute()
        except Exception:
            self.logger.exception("Experiment failed")
            raise
        elapsed = time.time() - t0

        results["_meta"] = {
            "experiment": self.tag,
            "name": self.name,
            "timestamp": datetime.now(UTC).isoformat(),
            "seed": self.seed,
            "elapsed_s": round(elapsed, 2),
        }

        # Save results
        results_path = self.output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to: {results_path}")
        self.logger.info(f"Elapsed: {elapsed:.1f}s")
        self.logger.info("=" * 60)
        self.logger.info(f"{self.name.upper()} COMPLETE")
        self.logger.info("=" * 60)

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
