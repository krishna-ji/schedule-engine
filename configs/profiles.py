"""
Profile configurations: TEST and PROD scaling.

Profiles inherit from BaseConfig and override scaling parameters:
- TEST: Quick smoke tests (30 gens, 10 pop, ~2-5 min)
- PROD: Full production runs (2000 gens, 200 pop, ~1-3 hours)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .base import BaseConfig


class Profile(str, Enum):
    """Profile enum for backward compatibility."""

    TEST = "test"
    PROD = "prod"

    @classmethod
    def from_string(cls, value: str | None) -> Profile:
        if value is None:
            return cls.TEST
        normalized = value.lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Unknown profile '{value}'. Valid options: {', '.join(p.value for p in cls)}"
            ) from exc


@dataclass
class TestConfig(BaseConfig):
    """
    Test profile: Quick smoke tests.

    Optimized for rapid iteration during development.
    Typical runtime: 2-5 minutes.
    """

    ngen: int = 30
    pop_size: int = 10
    environment: str = "test"

    # Reduce metric frequency for faster tests
    advanced_metrics_frequency: int = 5

    # Lighter profiling for tests
    performance_profiling_enabled: bool = True


@dataclass
class ProdConfig(BaseConfig):
    """
    Production profile: Full evaluation runs.

    Optimized for thesis experiments and production deployments.
    Typical runtime: 1-3 hours.
    """

    ngen: int = 2000
    pop_size: int = 200
    environment: str = "prod"

    # Full metrics for production
    advanced_metrics_frequency: int = 10
    hypervolume_enabled: bool = True
    igd_enabled: bool = True
    gd_enabled: bool = True

    # Detailed profiling for production
    performance_profiling_enabled: bool = True
