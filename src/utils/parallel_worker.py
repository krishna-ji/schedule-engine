"""Helper utilities for multiprocessing worker initialization.

Loads scheduling context from disk into each worker process so the large
``SchedulingContext`` object does not need to be pickled across the process
boundary.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from io import StringIO
from typing import Any

# Global worker context (set once per worker process)
_WORKER_CONTEXT: dict[str, Any] | None = None


def init_worker(
    data_dir: str, seed: int, config_dict: dict[str, Any] | None = None
) -> None:
    """Initialize worker process by loading data via :class:`DataStore`.

    Called once when each worker process starts.  Loads scheduling context
    from disk so we don't have to pickle it.
    """
    global _WORKER_CONTEXT

    os.environ["_GA_WORKER_PROCESS"] = "1"

    # Suppress print output from data loading (workers should be silent)
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        from src.config import Config, init_config
        from src.io.data_store import DataStore

        # Initialize config in worker process
        if config_dict is not None:
            config_obj = (
                Config.from_dict(config_dict)
                if isinstance(config_dict, dict)
                else config_dict
            )
            init_config(config_obj=config_obj)

        # Load everything via DataStore (single source of truth)
        extra_pairs: list[tuple[str, str]] = []
        if config_dict is not None:
            try:
                raw = config_dict.get("time", {}).get("cohort_pairs", [])
                extra_pairs = [(str(a).strip(), str(b).strip()) for a, b in raw]
            except (AttributeError, KeyError, TypeError, ValueError):
                extra_pairs = []

        store = DataStore.from_json(
            data_dir, extra_cohort_pairs=extra_pairs, run_preflight=False
        )
        context = store.to_context()

    except Exception as e:
        sys.stdout = old_stdout
        logging.getLogger(__name__).error("Worker initialization failed: %s", e)
        raise
    finally:
        sys.stdout = old_stdout

    _WORKER_CONTEXT = {
        "courses": store.courses,
        "instructors": store.instructors,
        "groups": store.groups,
        "rooms": store.rooms,
        "qts": store.qts,
        "context": context,
    }

    # Propagate random seed
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def get_worker_context() -> dict[str, Any]:
    """Get the global worker context.

    Returns
    -------
    dict
        Contains ``'courses'``, ``'instructors'``, ``'groups'``, ``'rooms'``,
        ``'qts'``, ``'context'``.

    Raises
    ------
    RuntimeError
        If context is not initialised (not in a worker process).
    """
    global _WORKER_CONTEXT  # noqa: PLW0602
    if _WORKER_CONTEXT is None:
        raise RuntimeError(
            "Worker context not initialized. Are you running in a worker process?"
        )
    return _WORKER_CONTEXT
