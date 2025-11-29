"""Python-native configuration loader backed by blueprint classes."""

from __future__ import annotations

import os

from src.config.models import Config
from src.config.presets.base import ConfigBlueprint
from src.config.presets.profiles import DEFAULT_PROFILE, Profile
from src.config.runtime_mode import RuntimeMode

DEFAULT_RUNTIME_MODE = RuntimeMode.NSGA_FULL


def load_config(
    runtime_mode: RuntimeMode | None = None,
    profile: Profile | str | None = None,
    blueprint: ConfigBlueprint | None = None,
) -> Config:
    """Build a :class:`Config` using Python presets instead of YAML files."""

    resolved_profile = _resolve_profile(profile)

    if blueprint is None:
        mode = runtime_mode or DEFAULT_RUNTIME_MODE
        blueprint = mode.instantiate_blueprint()
    else:
        if runtime_mode is not None:
            blueprint.bind_runtime_mode(runtime_mode)

    config = blueprint.build(resolved_profile)

    # Validate runtime-mode specific expectations when possible
    if runtime_mode is not None:
        runtime_mode.validate_config(config.model_dump())

    return config


def _resolve_profile(profile: Profile | str | None) -> Profile:
    if profile is None:
        env_profile = os.environ.get("ENVIRONMENT")
        if env_profile:
            try:
                return Profile.from_string(env_profile)
            except ValueError:
                pass
        return DEFAULT_PROFILE
    if isinstance(profile, Profile):
        return profile
    return Profile.from_string(profile)
