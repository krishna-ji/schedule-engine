from __future__ import annotations

from .base import ConfigBlueprint


class BasePureNsgaBlueprint(ConfigBlueprint):
    override_key = "1-pure-nsga"
    name = "Pure NSGA-II (Baseline)"


class ModeAPureNsgaBlueprint(ConfigBlueprint):
    override_key = "a-pure-nsga"
    name = "Mode A · Pure NSGA-II"


class NsgaRepairsBlueprint(ConfigBlueprint):
    override_key = "2-nsga-repairs"
    name = "NSGA-II + Repairs"


class NsgaHeuristicsBlueprint(ConfigBlueprint):
    override_key = "3-nsga-heuristics"
    name = "NSGA-II + Repairs + Heuristics"


class NsgaFullBlueprint(ConfigBlueprint):
    override_key = "4-nsga-full"
    name = "NSGA-II Full Stack"


class RlGuidedBlueprint(ConfigBlueprint):
    override_key = "5-rl-guided"
    name = "RL-Guided NSGA-II"


class RoundRobinBlueprint(ConfigBlueprint):
    override_key = "6-roundrobin"
    name = "Round-Robin Heuristics"


class ModeBNsgaMemeticBlueprint(ConfigBlueprint):
    override_key = "b-nsga-memetic"
    name = "Mode B · NSGA-II + Memetic"


class ModeCRoundRobinBlueprint(ConfigBlueprint):
    override_key = "c-roundrobin"
    name = "Mode C · Round-Robin"


class ModeDAdaptiveBlueprint(ConfigBlueprint):
    override_key = "d-adaptive"
    name = "Mode D · Adaptive"


class ModeERlGuidedBlueprint(ConfigBlueprint):
    override_key = "e-rl-guided"
    name = "Mode E · RL-Guided"


class RlSpecialistsBlueprint(ConfigBlueprint):
    override_key = "7-rl-specialists"
    name = "RL Specialist Agents"


class ArchiveDiversityBlueprint(ConfigBlueprint):
    override_key = "8-archive-diversity"
    name = "Archive Diversity"


class RlHierarchicalBlueprint(ConfigBlueprint):
    override_key = "9-rl-hierarchical"
    name = "Hierarchical RL"


class RlMultiAgentBlueprint(ConfigBlueprint):
    override_key = "10-rl-multiagent"
    name = "Multi-Agent RL"
