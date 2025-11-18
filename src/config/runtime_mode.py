"""
Runtime mode selector for experiment management.

Provides enum-based runtime mode selection with killswitch validation
and modular config loading for research experiments.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


class RuntimeMode(str, Enum):
    """
    Enum for supported runtime modes.

    Each mode represents a different GA configuration for benchmarking:
    - BASELINE: Pure NSGA-II (no repairs, no heuristics)
    - NSGA_REPAIRS: NSGA-II + IGLS repairs only
    - NSGA_HEURISTICS: NSGA-II + repairs + Phase 1.5 heuristics
    - NSGA_FULL: NSGA-II + repairs + heuristics + local search
    - RL_GUIDED: NSGA-II + RL-guided heuristic selection
    - ROUND_ROBIN: NSGA-II + round-robin heuristic selection
    - RL_SPECIALISTS: RL with specialist agents (Enhancement #4)
    - ARCHIVE_DIVERSITY: Archive-based diversity maintenance (Enhancement #5)
    - RL_HIERARCHICAL: Hierarchical RL with two-level policies (Enhancement #7)
    - RL_MULTIAGENT: Rank-based multi-agent RL (Enhancement #8)
    """

    BASELINE = "1-pure-nsga"
    NSGA_REPAIRS = "2-nsga-repairs"
    NSGA_HEURISTICS = "3-nsga-heuristics"
    NSGA_FULL = "4-nsga-full"
    RL_GUIDED = "5-rl-guided"
    ROUND_ROBIN = "6-roundrobin"
    RL_SPECIALISTS = "7-rl-specialists"
    ARCHIVE_DIVERSITY = "8-archive-diversity"
    RL_HIERARCHICAL = "9-rl-hierarchical"
    RL_MULTIAGENT = "10-rl-multiagent"

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            RuntimeMode.BASELINE: "Pure NSGA-II (Baseline)",
            RuntimeMode.NSGA_REPAIRS: "NSGA-II + Repairs",
            RuntimeMode.NSGA_HEURISTICS: "NSGA-II + Repairs + Heuristics",
            RuntimeMode.NSGA_FULL: "NSGA-II + Full (Repairs + Heuristics + LS)",
            RuntimeMode.RL_GUIDED: "RL-Guided Heuristic Selection",
            RuntimeMode.ROUND_ROBIN: "Round-Robin Heuristic Selection",
            RuntimeMode.RL_SPECIALISTS: "RL with Specialist Agents",
            RuntimeMode.ARCHIVE_DIVERSITY: "Archive-Based Diversity",
            RuntimeMode.RL_HIERARCHICAL: "Hierarchical RL (Two-Level)",
            RuntimeMode.RL_MULTIAGENT: "Rank-Based Multi-Agent RL",
        }
        return names[self]

    @property
    def config_path(self) -> Path:
        """
        Get config file path for this runtime mode.

        Returns:
            Path to config YAML file (e.g., configs/baseline/1-pure-nsga.yaml)
        """
        category_map = {
            RuntimeMode.BASELINE: "baseline",
            RuntimeMode.NSGA_REPAIRS: "nsga",
            RuntimeMode.NSGA_HEURISTICS: "nsga",
            RuntimeMode.NSGA_FULL: "nsga",
            RuntimeMode.RL_GUIDED: "rl",
            RuntimeMode.ROUND_ROBIN: "hybrid",
            RuntimeMode.RL_SPECIALISTS: "rl",
            RuntimeMode.ARCHIVE_DIVERSITY: "rl",
            RuntimeMode.RL_HIERARCHICAL: "rl",
            RuntimeMode.RL_MULTIAGENT: "rl",
        }
        category = category_map[self]
        return Path(f"configs/{category}/{self.value}.yaml")

    @property
    def description(self) -> str:
        """Detailed description of this runtime mode."""
        descriptions = {
            RuntimeMode.BASELINE: (
                "Minimal NSGA-II with no repairs, no heuristics, no enhancements. "
                "Use as baseline for comparing all other modes."
            ),
            RuntimeMode.NSGA_REPAIRS: (
                "NSGA-II with IGLS repair system but no advanced heuristics. "
                "Tests effectiveness of repair system alone."
            ),
            RuntimeMode.NSGA_HEURISTICS: (
                "NSGA-II with repairs + Phase 1.5 heuristic toolbox (19 operators). "
                "Tests effectiveness of heuristic operators."
            ),
            RuntimeMode.NSGA_FULL: (
                "Full NSGA-II with repairs, heuristics, and LNS-IGLS local search. "
                "Best GA configuration without RL."
            ),
            RuntimeMode.RL_GUIDED: (
                "Full NSGA-II with RL agent controlling heuristic selection. "
                "RL guides both repair strategies and local search budget."
            ),
            RuntimeMode.ROUND_ROBIN: (
                "Full NSGA-II with round-robin heuristic selection (no RL). "
                "Cycles through all enabled heuristics in fixed order."
            ),
            RuntimeMode.RL_SPECIALISTS: (
                "RL with 4 specialist agents (Repair, Optimizer, Explorer, Intensifier). "
                "Context-aware agent selection based on solution state."
            ),
            RuntimeMode.ARCHIVE_DIVERSITY: (
                "Archive-based diversity via novelty search and MAP-Elites. "
                "Maintains behavioral diversity independent of fitness."
            ),
            RuntimeMode.RL_HIERARCHICAL: (
                "Hierarchical RL with two-level policy (category → heuristic). "
                "Faster learning via reduced action space."
            ),
            RuntimeMode.RL_MULTIAGENT: (
                "Rank-based multi-agent RL (4 agents for Pareto ranks 1-4). "
                "Specialists for elite, good, moderate, and poor solutions."
            ),
        }
        return descriptions[self]

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate that config matches expected killswitches for this mode.

        Args:
            config: Loaded config dictionary

        Returns:
            True if config is valid for this mode

        Raises:
            ValueError: If config violates mode constraints
        """
        if self == RuntimeMode.BASELINE:
            # Baseline: everything disabled
            if config.get("repair", {}).get("enabled", False):
                raise ValueError("Baseline mode must have repair.enabled=false")
            if config.get("rl", {}).get("enabled", False):
                raise ValueError("Baseline mode must have rl.enabled=false")
            if config.get("enhancements", {}).get("master_enabled", False):
                raise ValueError(
                    "Baseline mode must have enhancements.master_enabled=false"
                )

        elif self == RuntimeMode.NSGA_REPAIRS:
            # Repairs only: RL and heuristics disabled
            if not config.get("repair", {}).get("enabled", False):
                raise ValueError("NSGA_REPAIRS mode must have repair.enabled=true")
            if config.get("rl", {}).get("enabled", False):
                raise ValueError("NSGA_REPAIRS mode must have rl.enabled=false")

        elif self == RuntimeMode.RL_GUIDED:
            # RL guided: RL must be enabled
            if not config.get("rl", {}).get("enabled", False):
                raise ValueError("RL_GUIDED mode must have rl.enabled=true")
            if config.get("rl", {}).get("mode") not in ["inference", "hybrid"]:
                raise ValueError(
                    "RL_GUIDED mode must have rl.mode='inference' or 'hybrid'"
                )

        elif self == RuntimeMode.ROUND_ROBIN:
            # Round-robin: RL disabled, heuristics enabled
            if config.get("rl", {}).get("enabled", False):
                raise ValueError("ROUND_ROBIN mode must have rl.enabled=false")
            if config.get("ga", {}).get("use_adaptive_probabilities", False):
                raise ValueError(
                    "ROUND_ROBIN mode should have use_adaptive_probabilities=false"
                )

        return True

    @classmethod
    def from_string(cls, mode_str: str) -> "RuntimeMode":
        """
        Parse runtime mode from string.

        Args:
            mode_str: String like "baseline", "nsga-full", "rl-guided"

        Returns:
            RuntimeMode enum value

        Raises:
            ValueError: If mode string is invalid
        """
        # Normalize input
        normalized = mode_str.lower().replace("_", "-")

        # Map common aliases
        aliases = {
            "baseline": RuntimeMode.BASELINE,
            "pure-nsga": RuntimeMode.BASELINE,
            "pure": RuntimeMode.BASELINE,
            "nsga-repairs": RuntimeMode.NSGA_REPAIRS,
            "repairs": RuntimeMode.NSGA_REPAIRS,
            "nsga-heuristics": RuntimeMode.NSGA_HEURISTICS,
            "heuristics": RuntimeMode.NSGA_HEURISTICS,
            "nsga-full": RuntimeMode.NSGA_FULL,
            "full": RuntimeMode.NSGA_FULL,
            "rl-guided": RuntimeMode.RL_GUIDED,
            "rl": RuntimeMode.RL_GUIDED,
            "round-robin": RuntimeMode.ROUND_ROBIN,
            "roundrobin": RuntimeMode.ROUND_ROBIN,
            "rr": RuntimeMode.ROUND_ROBIN,
            "rl-specialists": RuntimeMode.RL_SPECIALISTS,
            "specialists": RuntimeMode.RL_SPECIALISTS,
            "archive-diversity": RuntimeMode.ARCHIVE_DIVERSITY,
            "archive": RuntimeMode.ARCHIVE_DIVERSITY,
            "rl-hierarchical": RuntimeMode.RL_HIERARCHICAL,
            "hierarchical": RuntimeMode.RL_HIERARCHICAL,
            "rl-multiagent": RuntimeMode.RL_MULTIAGENT,
            "multiagent": RuntimeMode.RL_MULTIAGENT,
        }

        if normalized in aliases:
            return aliases[normalized]

        # Try direct enum value match
        for mode in RuntimeMode:
            if mode.value == normalized:
                return mode

        raise ValueError(
            f"Invalid runtime mode: '{mode_str}'. "
            f"Valid options: {', '.join([m.value for m in RuntimeMode])}"
        )

    @classmethod
    def list_modes(cls) -> str:
        """
        Get formatted list of all runtime modes.

        Returns:
            Multi-line string with all modes and descriptions
        """
        lines = ["Available Runtime Modes:", ""]
        for mode in RuntimeMode:
            lines.append(f"  {mode.value}")
            lines.append(f"    {mode.display_name}")
            lines.append(f"    {mode.description}")
            lines.append("")
        return "\n".join(lines)


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    Combines runtime mode with config parameters and metadata.
    """

    mode: RuntimeMode
    config_path: Path
    experiment_name: Optional[str] = None
    output_dir: Optional[str] = None
    seed: int = 69
    notes: Optional[str] = None

    def __post_init__(self):
        """Validate config path exists."""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path}\n"
                f"Expected path for mode '{self.mode.value}': {self.mode.config_path}"
            )

    @property
    def mode_name(self) -> str:
        """Short mode name for file/folder naming."""
        return self.mode.value.split("-", 1)[1]  # e.g., "pure-nsga" -> "pure-nsga"

    def summary(self) -> str:
        """Get human-readable summary of experiment config."""
        lines = [
            f"Experiment Configuration:",
            f"  Mode: {self.mode.display_name}",
            f"  Config: {self.config_path}",
        ]
        if self.experiment_name:
            lines.append(f"  Name: {self.experiment_name}")
        if self.output_dir:
            lines.append(f"  Output: {self.output_dir}")
        lines.append(f"  Seed: {self.seed}")
        if self.notes:
            lines.append(f"  Notes: {self.notes}")
        return "\n".join(lines)
