"""Custom exception hierarchy for Schedule Engine."""

from __future__ import annotations

__all__ = [
    "ConfigurationError",
    "ConstraintViolationError",
    "DataValidationError",
    "ExportError",
    "FeasibilityError",
    "GAExecutionError",
    "RLTrainingError",
    "ScheduleEngineError",
]


class ScheduleEngineError(Exception):
    """Base exception for all schedule engine errors."""


class ConfigurationError(ScheduleEngineError):
    """Configuration loading or validation failed.

    Raised when:
    - Config file is missing or malformed
    - Config values fail validation
    - Runtime mode config violates mode constraints
    """


class DataValidationError(ScheduleEngineError):
    """Input data validation failed.

    Raised when:
    - Required JSON files are missing
    - Data format is invalid
    - Referential integrity violations
    - Duplicate enrollments detected
    """


class FeasibilityError(ScheduleEngineError):
    """Problem is provably infeasible.

    Raised when:
    - Insufficient instructor capacity
    - Room capacity bottleneck
    - Qualification bottleneck
    - Pigeonhole principle violation
    """


class ConstraintViolationError(ScheduleEngineError):
    """Hard constraint cannot be satisfied.

    Raised when:
    - Repair mechanisms cannot fix violations
    - Individual is structurally invalid
    - Constraint is mathematically impossible
    """


class GAExecutionError(ScheduleEngineError):
    """Genetic algorithm execution failed.

    Raised when:
    - Population initialization fails
    - Evolution encounters unexpected error
    - Multiprocessing worker crashes
    """


class ExportError(ScheduleEngineError):
    """Result export or reporting failed.

    Raised when:
    - Cannot create output directory
    - PDF generation fails
    - Plot rendering fails
    """


class RLTrainingError(ScheduleEngineError):
    """RL training or deployment failed.

    Raised when:
    - Agent initialization fails
    - Training encounters error
    - Model loading/saving fails
    - Checkpoint is corrupted
    """
