from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from src.config.models import Config
from src.config.runtime_mode import RuntimeMode

from .data import BASE_DEFAULTS, MODE_OVERRIDES
from .profiles import Profile, apply_profile_overrides
from .utils import apply_dynamic_overrides, deep_merge


class ConfigBlueprint:
    """Abstract base class for Python-native configuration presets."""

    # Optional key into MODE_OVERRIDES for data-driven presets
    override_key: str | None = None
    name: str | None = None
    description: str | None = None

    def __init__(self) -> None:
        self._runtime_mode: RuntimeMode | None = None

    def bind_runtime_mode(self, mode: RuntimeMode) -> None:
        """Attach runtime mode metadata for downstream consumers."""

        self._runtime_mode = mode

    def build(self, profile: Profile) -> Config:
        """Compose Config object from defaults, overrides, and profile."""

        data = deepcopy(BASE_DEFAULTS)
        deep_merge(data, self.base_overrides(profile))

        if self.override_key:
            deep_merge(data, MODE_OVERRIDES.get(self.override_key, {}))

        deep_merge(data, self.additional_overrides(profile))
        deep_merge(data, self.metadata_overrides(profile))
        apply_profile_overrides(data, profile)
        data = apply_dynamic_overrides(data)

        if self.name and not data.get("name"):
            data["name"] = self.name
        if "environment" not in data:
            data["environment"] = profile.value

        return Config(**data)

    def base_overrides(self, profile: Profile) -> Mapping[str, Any]:
        """Override shared defaults before mode-specific data is applied."""

        return {}

    def additional_overrides(self, profile: Profile) -> Mapping[str, Any]:
        """Hook for subclasses to customize beyond MODE_OVERRIDES."""

        return {}

    def metadata_overrides(self, profile: Profile) -> Mapping[str, Any]:
        """Inject runtime metadata for downstream reporting."""

        metadata: dict[str, Any] = {
            "profile": profile.value,
            "blueprint": self.__class__.__name__,
        }

        if self.override_key:
            metadata["override_key"] = self.override_key

        if self._runtime_mode is not None:
            metadata["runtime_mode"] = self._runtime_mode.value
            metadata["display_name"] = self._runtime_mode.display_name

        if self.description and "description" not in metadata:
            metadata["description"] = self.description

        return {"metadata": metadata}
