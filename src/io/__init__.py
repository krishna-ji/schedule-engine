"""I/O layer: Data loading, export, validation, and time system.

This module consolidates all input/output operations.

Usage:
    from src.io import load_courses, load_groups, QuantumTimeSystem
    from src.io import decode_individual, validate_input
    from src.io.export import export_everything
"""

from __future__ import annotations

from src.io.data_loader import (
    derive_cohort_pairs_from_groups,
    encode_availability,
    link_courses_and_groups,
    link_courses_and_instructors,
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
)
from src.io.data_store import DataStore
from src.io.decoder import decode_individual
from src.io.feasibility import (
    check_feasibility,
    generate_feasibility_report_file,
)
from src.io.time_system import QuantumTimeSystem
from src.io.validator import validate_input

__all__ = [
    # DataStore (preferred entry point)
    "DataStore",
    # Time system
    "QuantumTimeSystem",
    "check_feasibility",
    # Decoding
    "decode_individual",
    "derive_cohort_pairs_from_groups",
    "encode_availability",
    "generate_feasibility_report_file",
    "link_courses_and_groups",
    "link_courses_and_instructors",
    # Data loading
    "load_courses",
    "load_groups",
    "load_instructors",
    "load_rooms",
    # Validation
    "validate_input",
]
