from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


def deep_merge(
    target: dict[str, Any], overrides: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Recursively merge *overrides* into *target* in-place."""

    if not overrides:
        return target

    for key, value in overrides.items():
        base_value = target.get(key)
        if isinstance(base_value, dict) and isinstance(value, Mapping):
            deep_merge(base_value, value)
        else:
            target[key] = deepcopy(value)
    return target


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_REGISTRY_PATH = _PROJECT_ROOT / "models" / "rl_agents" / "registry.json"


def apply_dynamic_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Inject dynamic settings (e.g., active RL agent) at build time."""

    try:
        if not _REGISTRY_PATH.exists():
            return data
        with open(_REGISTRY_PATH) as fh:
            registry = json.load(fh)
    except Exception:
        return data

    deployments = registry.get("deployments", [])
    active = next(
        (dep for dep in reversed(deployments) if dep.get("status") == "active"),
        None,
    )
    if not active:
        return data

    rl_cfg = data.setdefault("rl", {})
    agent_cfg = rl_cfg.setdefault("agent", {})
    if active.get("model_path"):
        agent_cfg["model_path"] = active["model_path"]
    if active.get("agent_type"):
        agent_cfg["type"] = active["agent_type"]

    return data
