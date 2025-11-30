"""
Runtime mode enumeration for experiment tracking.

This module provides the RuntimeMode enum used by the old blueprint system.
New dataclass system doesn't require this - it's here for backward compatibility.
"""

from __future__ import annotations

from enum import Enum


class RuntimeMode(str, Enum):
    """Runtime mode identifiers for different experimental configurations."""

    # Thesis progressive modes (A-E)
    BASELINE_PURE_NSGA = "a-baseline"
    NSGA_MEMETIC = "b-memetic"
    ROUNDROBIN = "c-roundrobin"
    ADAPTIVE = "d-adaptive"
    RL_GUIDED = "e-rl-guided"

    # Numbered feature sets (1-10) - legacy
    MODE_1_PURE_NSGA = "1-pure-nsga"
    MODE_2_NSGA_REPAIRS = "2-nsga-repairs"
    MODE_3_NSGA_HEURISTICS = "3-nsga-heuristics"
    MODE_4_NSGA_FULL = "4-nsga-full"
    MODE_5_RL_GUIDED = "5-rl-guided"
    MODE_6_ROUNDROBIN = "6-roundrobin"
    MODE_7_RL_SPECIALISTS = "7-rl-specialists"
    MODE_8_ARCHIVE_DIVERSITY = "8-archive-diversity"
    MODE_9_RL_HIERARCHICAL = "9-rl-hierarchical"
    MODE_10_RL_MULTIAGENT = "10-rl-multiagent"

    @property
    def display_name(self) -> str:
        """Human-readable display name for this mode."""
        names = {
            self.BASELINE_PURE_NSGA: "Pure NSGA-II Baseline",
            self.NSGA_MEMETIC: "Memetic NSGA-II",
            self.ROUNDROBIN: "Round-Robin Heuristics",
            self.ADAPTIVE: "Adaptive Heuristic Selection",
            self.RL_GUIDED: "RL-Guided Hyper-Heuristic",
            self.MODE_1_PURE_NSGA: "Mode 1: Pure NSGA-II",
            self.MODE_2_NSGA_REPAIRS: "Mode 2: NSGA-II + Repairs",
            self.MODE_3_NSGA_HEURISTICS: "Mode 3: NSGA-II + Heuristics",
            self.MODE_4_NSGA_FULL: "Mode 4: Full NSGA-II Stack",
            self.MODE_5_RL_GUIDED: "Mode 5: RL-Guided",
            self.MODE_6_ROUNDROBIN: "Mode 6: Round-Robin",
            self.MODE_7_RL_SPECIALISTS: "Mode 7: RL Specialists",
            self.MODE_8_ARCHIVE_DIVERSITY: "Mode 8: Archive Diversity",
            self.MODE_9_RL_HIERARCHICAL: "Mode 9: Hierarchical RL",
            self.MODE_10_RL_MULTIAGENT: "Mode 10: Multi-Agent RL",
        }
        return names.get(self, self.value)
