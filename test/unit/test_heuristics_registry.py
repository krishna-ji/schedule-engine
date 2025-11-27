import types

import src.heuristics  # noqa: F401 - ensures decorators register heuristics
from src.heuristics.registry import (
    HeuristicCategory,
    get_enabled_heuristics,
    get_heuristics_by_category,
)


def test_each_category_has_registered_heuristics():
    """All heuristic categories should register at least one operator."""
    for category in HeuristicCategory:
        heuristics = get_heuristics_by_category(category)
        assert heuristics, f"Expected heuristics for category '{category.value}'"


def test_get_enabled_heuristics_respects_priority_overrides(monkeypatch):
    """Config overrides must reorder heuristics based on priority."""

    def _fake_config():
        return types.SimpleNamespace(
            heuristics=types.SimpleNamespace(
                improvement={
                    "kempe_chain": {"enabled": True, "priority": 50},
                    "ejection_chain": {"enabled": True, "priority": 1},
                }
            )
        )

    monkeypatch.setattr("src.heuristics.registry.get_config", _fake_config)

    ordered = list(
        get_enabled_heuristics(category=HeuristicCategory.IMPROVEMENT).keys()
    )
    assert ordered[:2] == ["ejection_chain", "kempe_chain"]


def test_disabled_heuristics_removed_by_config(monkeypatch):
    """Disabling a heuristic via config removes it from the enabled set."""

    def _fake_config():
        return types.SimpleNamespace(
            heuristics=types.SimpleNamespace(
                perturbation={"random_swap": {"enabled": False}}
            )
        )

    monkeypatch.setattr("src.heuristics.registry.get_config", _fake_config)

    enabled = get_enabled_heuristics(category=HeuristicCategory.PERTURBATION)
    assert "random_swap" not in enabled
