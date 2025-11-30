"""
Registry mapping RuntimeMode enum values to ConfigBlueprint classes.

Provides centralized mode-to-blueprint resolution for the configuration system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config.runtime_mode import RuntimeMode

from .blueprints import (
    Mode1PureNsgaBlueprint,
    Mode2NsgaRepairsBlueprint,
    Mode3NsgaHeuristicsBlueprint,
    Mode4NsgaFullBlueprint,
    Mode5RlGuidedBlueprint,
    Mode6RoundRobinBlueprint,
    Mode7RlSpecialistsBlueprint,
    Mode8ArchiveDiversityBlueprint,
    Mode9RlHierarchicalBlueprint,
    Mode10RlMultiAgentBlueprint,
    ModeAPureNsgaBlueprint,
    ModeBNsgaMemeticBlueprint,
    ModeCRoundRobinBlueprint,
    ModeDAdaptiveBlueprint,
    ModeERlGuidedBlueprint,
)

if TYPE_CHECKING:
    from .base import ConfigBlueprint


# ==============================================================================
# MODE → BLUEPRINT REGISTRY
# ==============================================================================

RUNTIME_MODE_BLUEPRINTS: dict[RuntimeMode, type[ConfigBlueprint]] = {
    # Thesis progressive modes (A-E)
    RuntimeMode.BASELINE_PURE_NSGA: ModeAPureNsgaBlueprint,
    RuntimeMode.NSGA_MEMETIC: ModeBNsgaMemeticBlueprint,
    RuntimeMode.ROUNDROBIN: ModeCRoundRobinBlueprint,
    RuntimeMode.ADAPTIVE: ModeDAdaptiveBlueprint,
    RuntimeMode.RL_GUIDED: ModeERlGuidedBlueprint,
    # Numbered feature set (1-10)
    RuntimeMode.MODE_1_PURE_NSGA: Mode1PureNsgaBlueprint,
    RuntimeMode.MODE_2_NSGA_REPAIRS: Mode2NsgaRepairsBlueprint,
    RuntimeMode.MODE_3_NSGA_HEURISTICS: Mode3NsgaHeuristicsBlueprint,
    RuntimeMode.MODE_4_NSGA_FULL: Mode4NsgaFullBlueprint,
    RuntimeMode.MODE_5_RL_GUIDED: Mode5RlGuidedBlueprint,
    RuntimeMode.MODE_6_ROUNDROBIN: Mode6RoundRobinBlueprint,
    RuntimeMode.MODE_7_RL_SPECIALISTS: Mode7RlSpecialistsBlueprint,
    RuntimeMode.MODE_8_ARCHIVE_DIVERSITY: Mode8ArchiveDiversityBlueprint,
    RuntimeMode.MODE_9_RL_HIERARCHICAL: Mode9RlHierarchicalBlueprint,
    RuntimeMode.MODE_10_RL_MULTIAGENT: Mode10RlMultiAgentBlueprint,
}
