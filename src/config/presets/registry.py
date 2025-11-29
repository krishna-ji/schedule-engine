from __future__ import annotations

from src.config.runtime_mode import RuntimeMode

from .base import ConfigBlueprint
from .modes import (
    ArchiveDiversityBlueprint,
    BasePureNsgaBlueprint,
    ModeAPureNsgaBlueprint,
    ModeBNsgaMemeticBlueprint,
    ModeCRoundRobinBlueprint,
    ModeDAdaptiveBlueprint,
    ModeERlGuidedBlueprint,
    NsgaFullBlueprint,
    NsgaHeuristicsBlueprint,
    NsgaRepairsBlueprint,
    RlGuidedBlueprint,
    RlHierarchicalBlueprint,
    RlMultiAgentBlueprint,
    RlSpecialistsBlueprint,
    RoundRobinBlueprint,
)

RUNTIME_MODE_BLUEPRINTS: dict[RuntimeMode, type[ConfigBlueprint]] = {
    RuntimeMode.BASELINE: BasePureNsgaBlueprint,
    RuntimeMode.NSGA_REPAIRS: NsgaRepairsBlueprint,
    RuntimeMode.NSGA_HEURISTICS: NsgaHeuristicsBlueprint,
    RuntimeMode.NSGA_FULL: NsgaFullBlueprint,
    RuntimeMode.RL_GUIDED: RlGuidedBlueprint,
    RuntimeMode.ROUND_ROBIN: RoundRobinBlueprint,
    RuntimeMode.RL_SPECIALISTS: RlSpecialistsBlueprint,
    RuntimeMode.ARCHIVE_DIVERSITY: ArchiveDiversityBlueprint,
    RuntimeMode.RL_HIERARCHICAL: RlHierarchicalBlueprint,
    RuntimeMode.RL_MULTIAGENT: RlMultiAgentBlueprint,
    RuntimeMode.MODE_A: ModeAPureNsgaBlueprint,
    RuntimeMode.MODE_B: ModeBNsgaMemeticBlueprint,
    RuntimeMode.MODE_C: ModeCRoundRobinBlueprint,
    RuntimeMode.MODE_D: ModeDAdaptiveBlueprint,
    RuntimeMode.MODE_E: ModeERlGuidedBlueprint,
}
