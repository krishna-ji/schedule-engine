"""I/O layer: Data loading, export, validation, and time system.

This module consolidates all input/output operations.

Usage:
    from schedule_engine.io import load_courses, load_groups, QuantumTimeSystem
    from schedule_engine.io import decode_individual, validate_input
    from schedule_engine.io.export import export_everything
"""

from __future__ import annotations

from schedule_engine.io.data_loader import (
    derive_cohort_pairs_from_groups,
    encode_availability,
    link_courses_and_groups,
    link_courses_and_instructors,
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
)
from schedule_engine.io.data_store import DataStore
from schedule_engine.io.decoder import decode_individual
from schedule_engine.io.feasibility import (
    check_feasibility,
    generate_feasibility_report_file,
)
from schedule_engine.io.time_system import QuantumTimeSystem
from schedule_engine.io.validator import validate_input

__all__ = [
    # DataStore (preferred entry point)
    "DataStore",
    # Data loading
    "load_courses",
    "load_groups",
    "load_instructors",
    "load_rooms",
    "link_courses_and_groups",
    "link_courses_and_instructors",
    "encode_availability",
    "derive_cohort_pairs_from_groups",
    # Time system
    "QuantumTimeSystem",
    # Decoding
    "decode_individual",
    # Validation
    "validate_input",
    "check_feasibility",
    "generate_feasibility_report_file",
]
