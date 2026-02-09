"""Clean OOP constraint system.

Self-contained constraint classes with configurable weights and parameters.
Zero backward compatibility with old function-based system.

Public API:
- ``Constraint`` protocol
- ``ALL_CONSTRAINTS`` registry (14 default instances)
- ``HARD_CONSTRAINT_CLASSES`` registry (8 default instances)
- ``SOFT_CONSTRAINT_CLASSES`` registry (6 default instances)
- ``build_constraints()`` factory for custom configs

Usage:
    # Use defaults (all weights = 1.0)
    from schedule_engine.constraints import ALL_CONSTRAINTS
    from schedule_engine.evaluation import Evaluator

    evaluator = Evaluator(constraints=ALL_CONSTRAINTS)
    hard, soft = evaluator.fitness(genes, context, qts)

    # Custom weights and params
    from schedule_engine.constraints import build_constraints

    constraints = build_constraints(
        hard_weight=10.0,              # Scale all hard constraints
        soft_weight=0.5,               # Scale all soft constraints
        isolated_slot_penalty=50.0,    # Custom magic value
        break_min_quanta=4,            # Custom magic value
    )
    evaluator = Evaluator(constraints=constraints)
"""

from __future__ import annotations

from schedule_engine.constraints.constraints import (  # Individual constraint classes
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
    RoomTimeAvailability,
    SessionContinuity,
    StudentGroupExclusivity,
    StudentLunchBreak,
    StudentScheduleCompactness,
    build_constraints,
)

# Backward-compatible name exports
HARD_CONSTRAINTS = HARD_CONSTRAINT_CLASSES
SOFT_CONSTRAINTS = SOFT_CONSTRAINT_CLASSES
HARD_CONSTRAINT_NAMES = [c.name for c in HARD_CONSTRAINT_CLASSES]
SOFT_CONSTRAINT_NAMES = [c.name for c in SOFT_CONSTRAINT_CLASSES]

__all__ = [
    "Constraint",
    "ALL_CONSTRAINTS",
    "HARD_CONSTRAINT_CLASSES",
    "SOFT_CONSTRAINT_CLASSES",
    "HARD_CONSTRAINTS",
    "SOFT_CONSTRAINTS",
    "HARD_CONSTRAINT_NAMES",
    "SOFT_CONSTRAINT_NAMES",
    "build_constraints",
    # Individual classes (for custom configs)
    "StudentGroupExclusivity",
    "InstructorExclusivity",
    "RoomExclusivity",
    "InstructorQualifications",
    "RoomSuitability",
    "InstructorTimeAvailability",
    "RoomTimeAvailability",
    "CourseCompleteness",
    "StudentScheduleCompactness",
    "InstructorScheduleCompactness",
    "StudentLunchBreak",
    "SessionContinuity",
    "PairedCohortPracticalAlignment",
    "BreakPlacementCompliance",
]
