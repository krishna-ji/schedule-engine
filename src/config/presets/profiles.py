from __future__ import annotations

from enum import Enum
from typing import Any

from .data import PROFILE_OVERRIDES
from .utils import deep_merge


class Profile(str, Enum):
    TEST = "test"
    PROD = "prod"

    @classmethod
    def from_string(cls, value: str | None) -> Profile:
        if value is None:
            return cls.TEST
        normalized = value.lower()
        try:
            return cls(normalized)
        except ValueError as exc:  # pragma: no cover - defensive parsing
            raise ValueError(
                f"Unknown profile '{value}'. Valid options: {', '.join(p.value for p in cls)}"
            ) from exc


DEFAULT_PROFILE: Profile = Profile.TEST


def apply_profile_overrides(data: dict[str, Any], profile: Profile) -> dict[str, Any]:
    overrides = PROFILE_OVERRIDES.get(profile.value)
    if overrides:
        deep_merge(data, overrides)
    data["environment"] = profile.value
    return data
