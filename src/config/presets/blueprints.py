"""
Blueprint classes for different GA configurations.

Each blueprint defines a specific algorithmic configuration through method overrides.
Experiments instantiate these blueprints with custom killswitches.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import ConfigBlueprint
from .profiles import Profile

# ==============================================================================
# BASE ALGORITHMIC BLUEPRINTS (Reusable Components)
# ==============================================================================


class PureNsgaBlueprint(ConfigBlueprint):
    """Pure NSGA-II with no enhancements."""

    name = "Pure NSGA-II"
    description = "Minimal NSGA-II baseline (no repairs, no heuristics)"

    def base_overrides(self, profile: Profile) -> Mapping[str, Any]:
        return {
            "ga": {
                "population_strategy": "random",
                "use_adaptive_probabilities": False,
                "use_constraint_guided_mutation": False,
            },
            "repair": {"enabled": False},
            "lns": {"enabled": False},
            "rl": {"enabled": False, "mode": "disabled"},
            "enhancements": {"master_enabled": False},
            "heuristics": {
                "adaptive_priority": {"enabled": False},
                "construction": {},
                "perturbation": {},
                "improvement": {},
                "meta": {},
                "diversity": {},
            },
        }


class MemeticNsgaBlueprint(ConfigBlueprint):
    """NSGA-II with memetic local search (repairs)."""

    name = "Memetic NSGA-II"
    description = "NSGA-II with memetic local search repairs"

    def base_overrides(self, profile: Profile) -> Mapping[str, Any]:
        return {
            "ga": {
                "population_strategy": "random",
                "use_constraint_guided_mutation": True,
            },
            "repair": {
                "enabled": True,
                "memetic_mode": True,
                "selective_repair": {"enabled": True},
            },
            "lns": {"enabled": False},
            "rl": {"enabled": False, "mode": "disabled"},
            "enhancements": {"master_enabled": False},
            "heuristics": {
                "adaptive_priority": {"enabled": False},
            },
        }


class RoundRobinHeuristicBlueprint(ConfigBlueprint):
    """NSGA-II with round-robin heuristic selection."""

    name = "Round-Robin Heuristics"
    description = "NSGA-II with fixed rotation through heuristics"

    def base_overrides(self, profile: Profile) -> Mapping[str, Any]:
        return {
            "ga": {
                "population_strategy": "hybrid",
                "use_constraint_guided_mutation": True,
            },
            "repair": {
                "enabled": True,
                "memetic_mode": True,
                "selective_repair": {"enabled": True},
            },
            "heuristics": {
                "adaptive_priority": {"enabled": False},
                "construction": {
                    "largest_degree_first": {"enabled": True, "priority": 1},
                    "most_constrained_first": {"enabled": True, "priority": 2},
                    "earliest_deadline_first": {"enabled": True, "priority": 3},
                },
                "perturbation": {
                    "random_swap": {"enabled": True, "priority": 1},
                    "temporal_shift": {"enabled": True, "priority": 2},
                    "room_shuffle": {"enabled": True, "priority": 3},
                    "instructor_reassign": {"enabled": True, "priority": 4},
                },
            },
            "lns": {"enabled": False},
            "rl": {"enabled": False, "mode": "disabled"},
            "enhancements": {"master_enabled": True},
        }


class AdaptiveHeuristicBlueprint(ConfigBlueprint):
    """NSGA-II with adaptive heuristic priority."""

    name = "Adaptive Heuristics"
    description = "NSGA-II with performance-based heuristic selection"

    def base_overrides(self, profile: Profile) -> Mapping[str, Any]:
        return {
            "ga": {
                "population_strategy": "hybrid",
                "use_adaptive_probabilities": True,
                "use_constraint_guided_mutation": True,
            },
            "repair": {
                "enabled": True,
                "memetic_mode": True,
                "selective_repair": {"enabled": True},
            },
            "heuristics": {
                "adaptive_priority": {
                    "enabled": True,
                    "evaluation_window": 10,
                    "reorder_interval": 10,
                },
                "construction": {
                    "largest_degree_first": {"enabled": True, "priority": 1},
                    "most_constrained_first": {"enabled": True, "priority": 2},
                    "earliest_deadline_first": {"enabled": True, "priority": 3},
                },
                "perturbation": {
                    "random_swap": {"enabled": True, "priority": 1},
                    "temporal_shift": {"enabled": True, "priority": 2},
                    "room_shuffle": {"enabled": True, "priority": 3},
                    "instructor_reassign": {"enabled": True, "priority": 4},
                },
                "improvement": {
                    "kempe_chain": {"enabled": True, "priority": 1},
                    "ejection_chain": {"enabled": True, "priority": 2},
                    "variable_depth_search": {"enabled": True, "priority": 3},
                },
            },
            "lns": {"enabled": False},
            "rl": {"enabled": False, "mode": "disabled"},
            "enhancements": {
                "master_enabled": True,
                "hypermutation": {"enabled": True, "trigger_on_stagnation": True},
            },
        }


class RlGuidedBlueprint(ConfigBlueprint):
    """RL-guided heuristic selection with full NSGA-II stack."""

    name = "RL-Guided"
    description = "RL-guided heuristic selection with full NSGA-II features"

    def base_overrides(self, profile: Profile) -> Mapping[str, Any]:
        return {
            "ga": {
                "population_strategy": "hybrid",
                "use_adaptive_probabilities": True,
                "use_constraint_guided_mutation": True,
            },
            "repair": {
                "enabled": True,
                "memetic_mode": True,
                "selective_repair": {"enabled": True},
            },
            "heuristics": {
                "adaptive_priority": {"enabled": False},
                "construction": {
                    "largest_degree_first": {"enabled": True},
                    "most_constrained_first": {"enabled": True},
                    "earliest_deadline_first": {"enabled": True},
                },
                "perturbation": {
                    "random_swap": {"enabled": True},
                    "temporal_shift": {"enabled": True},
                    "room_shuffle": {"enabled": True},
                    "instructor_reassign": {"enabled": True},
                },
                "improvement": {
                    "kempe_chain": {"enabled": True},
                    "ejection_chain": {"enabled": True},
                    "variable_depth_search": {"enabled": True},
                },
                "meta": {
                    "variable_neighborhood_descent": {"enabled": True},
                    "iterated_local_search": {"enabled": True},
                    "adaptive_large_neighborhood": {"enabled": True},
                    "guided_local_search": {"enabled": True},
                },
            },
            "lns": {"enabled": True},
            "rl": {
                "enabled": True,
                "mode": "rl_primary",
                "hybrid": {"mode": "rl_primary", "rl_probability": 0.8},
            },
            "enhancements": {
                "master_enabled": True,
                "memetic_mode": True,
                "hypermutation": {"enabled": True, "trigger_on_stagnation": True},
                "population_restart": {"enabled": True},
            },
        }


class FullStackNsgaBlueprint(ConfigBlueprint):
    """Complete NSGA-II with all features (no RL)."""

    name = "Full Stack NSGA-II"
    description = "Complete NSGA-II with all enhancements (no RL)"

    def base_overrides(self, profile: Profile) -> Mapping[str, Any]:
        return {
            "ga": {
                "population_strategy": "hybrid",
                "use_adaptive_probabilities": True,
                "use_constraint_guided_mutation": True,
            },
            "repair": {
                "enabled": True,
                "memetic_mode": True,
                "selective_repair": {"enabled": True},
                "stagnation_repair": {"enabled": True},
            },
            "heuristics": {
                "construction": {
                    "largest_degree_first": {"enabled": True},
                    "most_constrained_first": {"enabled": True},
                    "earliest_deadline_first": {"enabled": True},
                },
                "perturbation": {
                    "random_swap": {"enabled": True},
                    "temporal_shift": {"enabled": True},
                    "room_shuffle": {"enabled": True},
                    "instructor_reassign": {"enabled": True},
                    "multi_perturbation": {"enabled": True},
                },
                "improvement": {
                    "kempe_chain": {"enabled": True},
                    "ejection_chain": {"enabled": True},
                    "variable_depth_search": {"enabled": True},
                },
                "meta": {
                    "variable_neighborhood_descent": {"enabled": True},
                    "iterated_local_search": {"enabled": True},
                    "adaptive_large_neighborhood": {"enabled": True},
                    "guided_local_search": {"enabled": True},
                },
                "diversity": {
                    "distance_preserving_crossover": {"enabled": True},
                    "crowding_mutation": {"enabled": True},
                    "niching_selection": {"enabled": True},
                    "adaptive_diversity_maintenance": {"enabled": True},
                },
            },
            "lns": {"enabled": True},
            "rl": {"enabled": False, "mode": "disabled"},
            "enhancements": {
                "master_enabled": True,
                "memetic_mode": True,
                "hypermutation": {"enabled": True, "trigger_on_stagnation": True},
                "constraint_priorities": {"enabled": True},
                "multi_neighborhood": {"enabled": True},
                "population_restart": {"enabled": True},
                "violation_heatmap": {"enabled": True},
            },
        }


class RlSpecialistBlueprint(ConfigBlueprint):
    """RL with specialist agents for constraint-specific repair."""

    name = "RL Specialists"
    description = "RL specialist agents for constraint-specific operations"

    def base_overrides(self, profile: Profile) -> Mapping[str, Any]:
        return {
            "ga": {
                "population_strategy": "hybrid",
                "use_adaptive_probabilities": True,
                "use_constraint_guided_mutation": True,
            },
            "repair": {
                "enabled": True,
                "memetic_mode": True,
                "selective_repair": {"enabled": True},
            },
            "heuristics": {
                "construction": {"largest_degree_first": {"enabled": True}},
                "perturbation": {"random_swap": {"enabled": True}},
                "improvement": {"kempe_chain": {"enabled": True}},
            },
            "lns": {"enabled": True},
            "rl": {
                "enabled": True,
                "mode": "specialists",
                "specialists": {
                    "enabled": True,
                    "num_specialists": 4,
                    "constraint_mapping": {
                        "specialist_1": ["instructor_conflicts"],
                        "specialist_2": ["room_conflicts"],
                        "specialist_3": ["group_overlaps"],
                        "specialist_4": ["time_availability"],
                    },
                },
            },
            "enhancements": {"master_enabled": True, "memetic_mode": True},
        }


class ArchiveDiversityBlueprint(ConfigBlueprint):
    """Archive-based diversity maintenance via novelty search."""

    name = "Archive Diversity"
    description = "NSGA-II with archive-based diversity preservation"

    def base_overrides(self, profile: Profile) -> Mapping[str, Any]:
        return {
            "ga": {
                "population_strategy": "hybrid",
                "use_adaptive_probabilities": True,
            },
            "repair": {"enabled": True, "memetic_mode": True},
            "heuristics": {
                "diversity": {
                    "adaptive_diversity_maintenance": {
                        "enabled": True,
                        "diversity_threshold": 0.15,
                        "archive_size": 100,
                    },
                    "distance_preserving_crossover": {"enabled": True},
                    "crowding_mutation": {"enabled": True, "intensity": 0.4},
                },
            },
            "enhancements": {
                "master_enabled": True,
                "memetic_mode": True,
                "population_restart": {"enabled": True},
            },
            "rl": {"enabled": False, "mode": "disabled"},
        }


class HierarchicalRlBlueprint(ConfigBlueprint):
    """Hierarchical RL with strategic and tactical policies."""

    name = "Hierarchical RL"
    description = "Hierarchical RL with two-level policy decomposition"

    def base_overrides(self, profile: Profile) -> Mapping[str, Any]:
        return {
            "ga": {
                "population_strategy": "hybrid",
                "use_adaptive_probabilities": True,
            },
            "repair": {"enabled": True, "memetic_mode": True},
            "heuristics": {
                "construction": {"largest_degree_first": {"enabled": True}},
                "perturbation": {"random_swap": {"enabled": True}},
                "improvement": {"kempe_chain": {"enabled": True}},
                "meta": {"guided_local_search": {"enabled": True}},
            },
            "lns": {"enabled": True},
            "rl": {
                "enabled": True,
                "mode": "hierarchical",
                "hierarchical": {
                    "enabled": True,
                    "strategic_model": "models/rl_agents/strategic_policy.zip",
                    "tactical_model": "models/rl_agents/tactical_policy.zip",
                    "decision_interval": 5,
                },
            },
            "enhancements": {"master_enabled": True, "memetic_mode": True},
        }


class MultiAgentRlBlueprint(ConfigBlueprint):
    """Multi-agent RL with rank-based specialization."""

    name = "Multi-Agent RL"
    description = "Multi-agent RL with rank-based agent specialization"

    def base_overrides(self, profile: Profile) -> Mapping[str, Any]:
        return {
            "ga": {
                "population_strategy": "hybrid",
                "use_adaptive_probabilities": True,
            },
            "repair": {"enabled": True, "memetic_mode": True},
            "heuristics": {
                "construction": {"largest_degree_first": {"enabled": True}},
                "perturbation": {"random_swap": {"enabled": True}},
                "improvement": {"kempe_chain": {"enabled": True}},
            },
            "lns": {"enabled": True},
            "rl": {
                "enabled": True,
                "mode": "multiagent",
                "multiagent": {
                    "enabled": True,
                    "num_rank_agents": 4,
                    "rank_configs": {
                        "rank_1": {"exploration_rate": 0.05, "learning_rate": 0.0001},
                        "rank_2": {"exploration_rate": 0.1, "learning_rate": 0.0003},
                        "rank_3": {"exploration_rate": 0.2, "learning_rate": 0.0005},
                        "rank_4": {"exploration_rate": 0.3, "learning_rate": 0.001},
                    },
                },
                "reward": {
                    "type": "decomposed",
                    "decomposed": {"decomposition_method": "tchebycheff"},
                },
            },
            "enhancements": {"master_enabled": True, "memetic_mode": True},
        }
