"""
Decorator-based constraint registry - single source of truth for all constraints.

This module provides decorators that auto-register constraints with their metadata
when functions are defined. Eliminates duplication between function definitions,
config models, and evaluators.

Usage:
    from src.constraints.registry import hard_constraint, soft_constraint

    @hard_constraint(
        name="student_group_exclusivity",
        description="Ensures each student group can only be in one session at a time",
        default_weight=3.0,
        needs_courses=False
    )
    def student_group_exclusivity(sessions):
        # implementation
        return violations

Benefits:
    - Single source of truth for constraint metadata
    - Auto-registration eliminates manual registry updates
    - Type-safe constraint definitions
    - Easy to add new constraints or metadata fields
    - Supports dynamic config generation
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ConstraintMetadata:
    """
    Metadata for a registered constraint.

    Attributes:
        name: Unique constraint identifier (used in config)
        function: The constraint evaluation function
        description: Human-readable explanation of what constraint enforces
        default_weight: Default weight value for config
        needs_courses: Whether constraint needs courses parameter
        constraint_type: "hard" or "soft"
        enabled_by_default: Whether constraint is enabled by default in config
    """

    name: str
    function: Callable
    description: str
    default_weight: float
    needs_courses: bool = False
    constraint_type: str = "hard"  # "hard" or "soft"
    enabled_by_default: bool = True


# ================
# GLOBAL REGISTRIES
# ================

_HARD_CONSTRAINTS: dict[str, ConstraintMetadata] = {}
_SOFT_CONSTRAINTS: dict[str, ConstraintMetadata] = {}


# ================
# DECORATOR FUNCTIONS
# ================


def hard_constraint(
    name: str,
    description: str,
    default_weight: float = 1.0,
    needs_courses: bool = False,
    enabled_by_default: bool = True,
) -> Callable[[Callable[..., int]], Callable[..., int]]:
    """
    Decorator to register a hard constraint function.

    Hard constraints ensure schedule feasibility - they must be satisfied
    for a valid timetable. Violations are heavily penalized.

    Args:
        name: Constraint identifier (must match config field name)
        description: Human-readable explanation of what is enforced
        default_weight: Default penalty weight (higher = more important)
        needs_courses: Whether constraint function needs courses parameter
        enabled_by_default: Whether constraint is enabled by default

    Example:
        @hard_constraint(
            name="student_group_exclusivity",
            description="Ensures each student group can only be in one session at a time",
            default_weight=3.0,
            needs_courses=False
        )
        def student_group_exclusivity(sessions: List[CourseSession]) -> int:
            violations = 0
            # ... count violations ...
            return violations

    Function Signature Requirements:
        - Must accept 'sessions' as first parameter
        - If needs_courses=True, must accept 'courses' as second parameter
        - Must return int (violation count or penalty score)
    """

    def decorator(func: Callable) -> Callable:
        metadata = ConstraintMetadata(
            name=name,
            function=func,
            description=description,
            default_weight=default_weight,
            needs_courses=needs_courses,
            constraint_type="hard",
            enabled_by_default=enabled_by_default,
        )
        _HARD_CONSTRAINTS[name] = metadata
        # Store metadata on function for introspection
        func._constraint_metadata = metadata  # type: ignore[attr-defined]
        return func

    return decorator


def soft_constraint(
    name: str,
    description: str,
    default_weight: float = 1.0,
    needs_courses: bool = False,
    enabled_by_default: bool = True,
) -> Callable[[Callable[..., float]], Callable[..., float]]:
    """
    Decorator to register a soft constraint function.

    Soft constraints optimize schedule quality - they should be minimized
    but don't prevent a schedule from being valid. Penalties are scaled.

    Args:
        name: Constraint identifier (must match config field name)
        description: Human-readable explanation of what is optimized
        default_weight: Default penalty weight (higher = more important)
        needs_courses: Whether constraint function needs courses parameter
        enabled_by_default: Whether constraint is enabled by default

    Example:
        @soft_constraint(
            name="student_schedule_compactness",
            description="Minimizes gaps in student schedules",
            default_weight=1.5,
            needs_courses=False
        )
        def student_schedule_compactness(sessions: List[CourseSession]) -> int:
            penalty = 0
            # ... calculate penalty ...
            return penalty

    Function Signature Requirements:
        - Must accept 'sessions' as first parameter
        - If needs_courses=True, must accept 'courses' as second parameter
        - Must return int or float (penalty score)
    """

    def decorator(func: Callable) -> Callable:
        metadata = ConstraintMetadata(
            name=name,
            function=func,
            description=description,
            default_weight=default_weight,
            needs_courses=needs_courses,
            constraint_type="soft",
            enabled_by_default=enabled_by_default,
        )
        _SOFT_CONSTRAINTS[name] = metadata
        # Store metadata on function for introspection
        func._constraint_metadata = metadata  # type: ignore[attr-defined]
        return func

    return decorator


# ================
# REGISTRY ACCESS FUNCTIONS
# ================


def get_all_hard_constraints() -> dict[str, ConstraintMetadata]:
    """
    Get all registered hard constraints with their metadata.

    Returns:
        Dict mapping constraint names to ConstraintMetadata objects
    """
    return _HARD_CONSTRAINTS.copy()


def get_all_soft_constraints() -> dict[str, ConstraintMetadata]:
    """
    Get all registered soft constraints with their metadata.

    Returns:
        Dict mapping constraint names to ConstraintMetadata objects
    """
    return _SOFT_CONSTRAINTS.copy()


def get_constraint_metadata(name: str) -> ConstraintMetadata | None:
    """
    Get constraint metadata by name (searches both hard and soft).

    Args:
        name: Constraint identifier

    Returns:
        ConstraintMetadata if found, None otherwise
    """
    return _HARD_CONSTRAINTS.get(name) or _SOFT_CONSTRAINTS.get(name)


def get_hard_constraint_function(name: str) -> Callable | None:
    """
    Get hard constraint evaluation function by name.

    Args:
        name: Constraint identifier

    Returns:
        Constraint function if found, None otherwise
    """
    metadata = _HARD_CONSTRAINTS.get(name)
    return metadata.function if metadata else None


def get_soft_constraint_function(name: str) -> Callable | None:
    """
    Get soft constraint evaluation function by name.

    Args:
        name: Constraint identifier

    Returns:
        Constraint function if found, None otherwise
    """
    metadata = _SOFT_CONSTRAINTS.get(name)
    return metadata.function if metadata else None


def constraint_needs_courses(name: str) -> bool:
    """
    Check if a constraint needs the 'courses' parameter.

    Args:
        name: Constraint identifier

    Returns:
        True if constraint needs courses parameter, False otherwise
    """
    metadata = get_constraint_metadata(name)
    return metadata.needs_courses if metadata else False


def get_constraint_description(name: str) -> str:
    """
    Get human-readable description of a constraint.

    Args:
        name: Constraint identifier

    Returns:
        Description string, or "Unknown constraint" if not found
    """
    metadata = get_constraint_metadata(name)
    return metadata.description if metadata else "Unknown constraint"


def get_all_constraint_names() -> dict[str, list[str]]:
    """
    Get all constraint names organized by type.

    Returns:
        Dict with 'hard' and 'soft' keys containing lists of constraint names
    """
    return {
        "hard": list(_HARD_CONSTRAINTS.keys()),
        "soft": list(_SOFT_CONSTRAINTS.keys()),
    }


# ================
# CONFIGURATION GENERATION HELPERS
# ================


def get_constraints_needing_courses() -> list[str]:
    """
    Get list of all constraint names that need courses parameter.

    Returns:
        List of constraint names requiring courses parameter
    """
    constraints = []
    for name, metadata in _HARD_CONSTRAINTS.items():
        if metadata.needs_courses:
            constraints.append(name)
    for name, metadata in _SOFT_CONSTRAINTS.items():
        if metadata.needs_courses:
            constraints.append(name)
    return constraints


def generate_constraint_config_template() -> dict[str, Any]:
    """
    Generate a configuration template from registered constraints.

    Useful for:
    - Validating config files have all required constraints
    - Generating default configs
    - Documentation generation

    Returns:
        Dict with 'hard_constraints' and 'soft_constraints' sections
    """
    config: dict[str, dict[str, dict[str, bool | float | str]]] = {
        "hard_constraints": {},
        "soft_constraints": {},
    }

    for name, metadata in _HARD_CONSTRAINTS.items():
        config["hard_constraints"][name] = {
            "enabled": metadata.enabled_by_default,
            "weight": metadata.default_weight,
            "description": metadata.description,
        }

    for name, metadata in _SOFT_CONSTRAINTS.items():
        config["soft_constraints"][name] = {
            "enabled": metadata.enabled_by_default,
            "weight": metadata.default_weight,
            "description": metadata.description,
        }

    return config


# ================
# VALIDATION FUNCTIONS
# ================


def validate_constraint_exists(name: str) -> bool:
    """
    Check if a constraint with given name is registered.

    Args:
        name: Constraint identifier to check

    Returns:
        True if constraint exists, False otherwise
    """
    return name in _HARD_CONSTRAINTS or name in _SOFT_CONSTRAINTS


def get_enabled_hard_constraints() -> dict[str, dict[str, Any]]:
    """
    Returns only the enabled hard constraints based on config.

    Uses decorator-based registry for constraint metadata and config for enable/weight.

    Returns:
        Dict[str, dict]: Mapping of enabled constraint names to their config (function, weight).
    """
    from src.config import get_config

    enabled = {}
    cfg = get_config().hard_constraints

    for name, metadata in _HARD_CONSTRAINTS.items():
        constraint_cfg = getattr(cfg, name, None)
        if constraint_cfg and constraint_cfg.enabled:
            enabled[name] = {
                "function": metadata.function,
                "weight": constraint_cfg.weight,
            }

    return enabled


def get_enabled_soft_constraints() -> dict[str, dict[str, Any]]:
    """
    Returns only the enabled soft constraints based on config.

    Uses decorator-based registry for constraint metadata and config for enable/weight.

    Returns:
        Dict[str, dict]: Mapping of enabled constraint names to their config (function, weight).
    """
    from src.config import get_config

    enabled = {}
    cfg = get_config().soft_constraints

    for name, metadata in _SOFT_CONSTRAINTS.items():
        constraint_cfg = getattr(cfg, name, None)
        if constraint_cfg and constraint_cfg.enabled:
            enabled[name] = {
                "function": metadata.function,
                "weight": constraint_cfg.weight,
            }

    return enabled


def get_registry_stats() -> dict[str, Any]:
    """
    Get statistics about registered constraints.

    Returns:
        Dict with counts and other registry information
    """
    hard_needing_courses = sum(1 for m in _HARD_CONSTRAINTS.values() if m.needs_courses)
    soft_needing_courses = sum(1 for m in _SOFT_CONSTRAINTS.values() if m.needs_courses)

    return {
        "total_hard_constraints": len(_HARD_CONSTRAINTS),
        "total_soft_constraints": len(_SOFT_CONSTRAINTS),
        "total_constraints": len(_HARD_CONSTRAINTS) + len(_SOFT_CONSTRAINTS),
        "hard_needing_courses": hard_needing_courses,
        "soft_needing_courses": soft_needing_courses,
        "hard_constraint_names": list(_HARD_CONSTRAINTS.keys()),
        "soft_constraint_names": list(_SOFT_CONSTRAINTS.keys()),
    }


# Import constraint modules to trigger decorator registration
# This must happen AFTER decorator definitions so decorators can execute
# and populate _HARD_CONSTRAINTS and _SOFT_CONSTRAINTS when modules are imported
from src.constraints import hard, soft  # noqa: E402, F401
