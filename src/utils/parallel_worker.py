"""Helper utilities for multiprocessing worker initialization."""

from __future__ import annotations

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
    """
    Initialize worker process by loading data from JSON files.

    This function is called once when each worker process starts.
    It sets up DEAP creator types and loads scheduling context from disk.

    Args:
        data_dir: Directory containing input JSON files
        seed: Random seed for reproducibility
        config_dict: Serialized config dict to reinitialize in worker
    """
    global _WORKER_CONTEXT

    # Set environment variable to indicate we're in a worker process
    os.environ["_GA_WORKER_PROCESS"] = "1"

    # Suppress all print output from data loading (workers should be silent)
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        from deap import base, creator

        from src.config import init_config
        from src.config.models import Config
        from src.core.types import SchedulingContext
        from src.encoder.input_encoder import (
            link_courses_and_groups,
            link_courses_and_instructors,
            load_courses,
            load_groups,
            load_instructors,
            load_rooms,
        )
        from src.encoder.quantum_time_system import QuantumTimeSystem

        # Initialize config in worker process (required for constraint evaluation)
        if config_dict is not None:
            config_obj = Config.model_validate(config_dict)
            init_config(config_obj=config_obj)

        # Set up DEAP creator types (required for Windows spawn)
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        # Load data from JSON files
        qts = QuantumTimeSystem()
        groups = load_groups(os.path.join(data_dir, "Groups.json"), qts)

        # Get enrolled course codes
        enrolled_course_codes = set()
        for group in groups.values():
            enrolled_course_codes.update(group.enrolled_courses)

        # Load and filter courses
        all_courses = load_courses(os.path.join(data_dir, "Course.json"))
        courses = {
            key: course
            for key, course in all_courses.items()
            if key[0] in enrolled_course_codes
        }

        instructors = load_instructors(os.path.join(data_dir, "Instructors.json"), qts)
        rooms = load_rooms(os.path.join(data_dir, "Rooms.json"), qts)

        # Link relationships
        link_courses_and_groups(courses, groups)
        link_courses_and_instructors(courses, instructors)

        # Get cohort_pairs from config if available
        cohort_pairs_list: list[tuple[str, str]] | None = None
        if config_dict is not None:
            try:
                cohort_pairs_list = config_dict.get("time", {}).get("cohort_pairs")
            except (AttributeError, KeyError):
                cohort_pairs_list = None

        # Create context object (optional, but useful if code expects it)
        # We don't have config here, but that's usually fine for workers
        context = SchedulingContext(
            courses=courses,
            groups=groups,
            instructors=instructors,
            rooms=rooms,
            available_quanta=list(qts.get_all_operating_quanta()),
            config=None,
            cohort_pairs=cohort_pairs_list,
        )

    except Exception as e:
        # Restore stdout to print error
        sys.stdout = old_stdout
        print(f"Worker initialization failed: {e}")
        raise
    finally:
        # Restore stdout
        sys.stdout = old_stdout

    # Store scheduling context in module-level variable
    _WORKER_CONTEXT = {
        "courses": courses,
        "instructors": instructors,
        "groups": groups,
        "rooms": rooms,
        "qts": qts,
        "context": context,
    }

    # Propagate random seed to worker
    random.seed(seed)

    # Also seed numpy if available
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def get_worker_context() -> dict[str, Any]:
    """
    Get the global worker context.

    Returns:
        Dict containing 'courses', 'instructors', 'groups', 'rooms', 'qts', 'context'

    Raises:
        RuntimeError: If context is not initialized (not in a worker process)
    """
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is None:
        # Fallback for sequential execution or if init failed
        # But strictly speaking, this should only be called in workers
        raise RuntimeError(
            "Worker context not initialized. Are you running in a worker process?"
        )
    return _WORKER_CONTEXT
