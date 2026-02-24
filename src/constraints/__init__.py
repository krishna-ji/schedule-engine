"""Clean OOP constraint system.

Self-contained constraint classes with configurable weights and parameters.

Public API:
- ``Constraint`` protocol
- ``Evaluator`` unified fitness evaluator
- ``ALL_CONSTRAINTS`` registry (15 default instances)
- ``HARD_CONSTRAINT_CLASSES`` registry (9 default instances)
- ``SOFT_CONSTRAINT_CLASSES`` registry (6 default instances)
- ``build_constraints()`` factory for custom configs

Usage:
    from src.constraints import Evaluator, ALL_CONSTRAINTS

    evaluator = Evaluator(constraints=ALL_CONSTRAINTS)
    hard, soft = evaluator.fitness(genes, context, qts)

    # Custom weights and params
    from src.constraints import build_constraints

    constraints = build_constraints(
        hard_weight=10.0,              # Scale all hard constraints
        soft_weight=0.5,               # Scale all soft constraints
        isolated_slot_penalty=50.0,    # Custom magic value
        break_min_quanta=4,            # Custom magic value
    )
    evaluator = Evaluator(constraints=constraints)
"""

from __future__ import annotations

from src.constraints.constraints import (  # Individual constraint classes
    ALL_CONSTRAINTS,
    HARD_CONSTRAINT_CLASSES,
    SOFT_CONSTRAINT_CLASSES,
    BreakPlacementCompliance,
    Constraint,
    CourseCompleteness,
    InstructorExclusivity,
    InstructorQualifications,
    InstructorScheduleCompactness,
    InstructorTimeAvailability,
    PairedCohortPracticalAlignment,
    RoomExclusivity,
    RoomSuitability,
    SessionContinuity,
    SiblingSameDay,
    StudentGroupExclusivity,
    StudentLunchBreak,
    StudentScheduleCompactness,
    build_constraints,
)
from src.constraints.evaluator import Evaluator

# Backward-compatible name exports
HARD_CONSTRAINTS = HARD_CONSTRAINT_CLASSES
SOFT_CONSTRAINTS = SOFT_CONSTRAINT_CLASSES
HARD_CONSTRAINT_NAMES = [c.name for c in HARD_CONSTRAINT_CLASSES]
SOFT_CONSTRAINT_NAMES = [c.name for c in SOFT_CONSTRAINT_CLASSES]

__all__ = [
    "ALL_CONSTRAINTS",
    "HARD_CONSTRAINTS",
    "HARD_CONSTRAINT_CLASSES",
    "HARD_CONSTRAINT_NAMES",
    "SOFT_CONSTRAINTS",
    "SOFT_CONSTRAINT_CLASSES",
    "SOFT_CONSTRAINT_NAMES",
    "BreakPlacementCompliance",
    "Constraint",
    "CourseCompleteness",
    "Evaluator",
    "InstructorExclusivity",
    "InstructorQualifications",
    "InstructorScheduleCompactness",
    "InstructorTimeAvailability",
    "PairedCohortPracticalAlignment",
    "RoomExclusivity",
    "RoomSuitability",
    "SessionContinuity",
    "SiblingSameDay",
    # Individual classes (for custom configs)
    "StudentGroupExclusivity",
    "StudentLunchBreak",
    "StudentScheduleCompactness",
    "build_constraints",
]
