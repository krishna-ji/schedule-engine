"""
Custom exception hierarchy for Schedule Engine.

Provides specific exception types for different failure modes,
enabling better error handling and debugging.
"""


class ScheduleEngineError(Exception):
    """Base exception for all schedule engine errors."""

    pass


class ConfigurationError(ScheduleEngineError):
    """Configuration loading or validation failed.

    Raised when:
    - Config file is missing or malformed
    - Config values fail validation
    - Runtime mode config violates mode constraints
    """

    pass


class DataValidationError(ScheduleEngineError):
    """Input data validation failed.

    Raised when:
    - Required JSON files are missing
    - Data format is invalid
    - Referential integrity violations
    - Duplicate enrollments detected
    """

    pass


class FeasibilityError(ScheduleEngineError):
    """Problem is provably infeasible.

    Raised when:
    - Insufficient instructor capacity
    - Room capacity bottleneck
    - Qualification bottleneck
    - Pigeonhole principle violation
    """

    pass


class ConstraintViolationError(ScheduleEngineError):
    """Hard constraint cannot be satisfied.

    Raised when:
    - Repair mechanisms cannot fix violations
    - Individual is structurally invalid
    - Constraint is mathematically impossible
    """

    pass


class GAExecutionError(ScheduleEngineError):
    """Genetic algorithm execution failed.

    Raised when:
    - Population initialization fails
    - Evolution encounters unexpected error
    - Multiprocessing worker crashes
    """

    pass


class ExportError(ScheduleEngineError):
    """Result export or reporting failed.

    Raised when:
    - Cannot create output directory
    - PDF generation fails
    - Plot rendering fails
    """

    pass


class RLTrainingError(ScheduleEngineError):
    """RL training or deployment failed.

    Raised when:
    - Agent initialization fails
    - Training encounters error
    - Model loading/saving fails
    - Checkpoint is corrupted
    """

    pass
