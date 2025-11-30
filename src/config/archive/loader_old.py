"""Python-native configuration loader backed by blueprint classes."""

from __future__ import annotations

import os

from src.config.models import Config
from src.config.presets.base import ConfigBlueprint
from src.config.presets.profiles import DEFAULT_PROFILE, Profile


def load_config(
    blueprint: ConfigBlueprint,
    profile: Profile | str | None = None,
) -> Config:
    """
    Build a Config using Python blueprint presets.

    Args:
        blueprint: Blueprint instance to build config from
        profile: Profile to use (TEST/PROD/DEBUG), defaults to TEST

    Returns:
        Validated Config instance

    Example:
        from configs import experiment_a
        from src.config.presets.profiles import Profile

        config = load_config(experiment_a, Profile.TEST)
    """
    resolved_profile = _resolve_profile(profile)
    config = blueprint.build(resolved_profile)
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
