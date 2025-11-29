"""Runtime mode selector for experiment management."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class RuntimeMode(str, Enum):
    """
    Enum for supported runtime modes.

    Each mode represents a different GA configuration for benchmarking:

    Numbered modes (1-10): Comprehensive experiments
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

    Lettered modes (A-E): Progressive thesis experiments
    - MODE_A: Pure NSGA-II baseline
    - MODE_B: + Memetic local search
    - MODE_C: + Round-robin heuristics
    - MODE_D: + Adaptive selection
    - MODE_E: + RL-guided (full system)
    """

    # Numbered modes (1-10): Comprehensive experiments
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

    # Lettered modes (A-E): Progressive thesis experiments
    MODE_A = "a-pure-nsga"
    MODE_B = "b-nsga-memetic"
    MODE_C = "c-roundrobin"
    MODE_D = "d-adaptive"
    MODE_E = "e-rl-guided"

    @property
    def display_name(self) -> str:
        """Human-readable display name."""
        names = {
            # Numbered modes (1-10)
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
            # Lettered modes (A-E): Progressive thesis
            RuntimeMode.MODE_A: "Mode A: Pure NSGA-II",
            RuntimeMode.MODE_B: "Mode B: + Memetic Local Search",
            RuntimeMode.MODE_C: "Mode C: + Round-Robin Heuristics",
            RuntimeMode.MODE_D: "Mode D: + Adaptive Selection",
            RuntimeMode.MODE_E: "Mode E: + RL-Guided (Full System)",
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
            # Numbered modes (1-10)
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
            # Lettered modes (A-E)
            RuntimeMode.MODE_A: "baseline",
            RuntimeMode.MODE_B: "nsga",
            RuntimeMode.MODE_C: "hybrid",
            RuntimeMode.MODE_D: "hybrid",
            RuntimeMode.MODE_E: "rl",
        }
        overrides = {
            # Progressive A–E configs already live under their canonical paths
            RuntimeMode.BASELINE: Path("configs/baseline/a-pure-nsga.yaml"),
            RuntimeMode.NSGA_FULL: Path("configs/nsga/5-nsga-full.yaml"),
            RuntimeMode.RL_GUIDED: Path("configs/rl/e-rl-guided.yaml"),
            RuntimeMode.ROUND_ROBIN: Path("configs/hybrid/c-roundrobin.yaml"),
            RuntimeMode.RL_SPECIALISTS: Path("configs/archive/7-rl-specialists.yaml"),
            RuntimeMode.ARCHIVE_DIVERSITY: Path(
                "configs/archive/8-archive-diversity.yaml"
            ),
            RuntimeMode.RL_HIERARCHICAL: Path("configs/archive/9-rl-hierarchical.yaml"),
            RuntimeMode.RL_MULTIAGENT: Path("configs/archive/10-rl-multiagent.yaml"),
        }

        if self in overrides:
            return overrides[self]

        category = category_map[self]
        return Path(f"configs/{category}/{self.value}.yaml")

    @property
    def description(self) -> str:
        """Detailed description of this runtime mode."""
        descriptions = {
            # Numbered modes (1-10)
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
            # Lettered modes (A-E): Progressive thesis
            RuntimeMode.MODE_A: (
                "Pure NSGA-II baseline - no repairs, no heuristics, no enhancements. "
                "Starting point for progressive thesis experiments."
            ),
            RuntimeMode.MODE_B: (
                "NSGA-II + memetic local search (repairs enabled). "
                "Adds iterative greedy local search repairs to baseline."
            ),
            RuntimeMode.MODE_C: (
                "NSGA-II + round-robin heuristics (fixed rotation). "
                "Adds 19 heuristics with round-robin selection."
            ),
            RuntimeMode.MODE_D: (
                "NSGA-II + adaptive heuristic selection (performance-based). "
                "Heuristics selected adaptively based on past performance."
            ),
            RuntimeMode.MODE_E: (
                "NSGA-II + RL-guided hyper-heuristic (full system). "
                "Deploys all techniques with RL controlling heuristic selection."
            ),
        }
        return descriptions[self]

    def validate_config(self, config: Mapping[str, Any]) -> bool:
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
    def from_string(cls, mode_str: str) -> RuntimeMode:
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
            # Numbered modes (1-10)
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
            # Lettered modes (A-E): Progressive thesis
            "mode-a": RuntimeMode.MODE_A,
            "mode-b": RuntimeMode.MODE_B,
            "mode-c": RuntimeMode.MODE_C,
            "mode-d": RuntimeMode.MODE_D,
            "mode-e": RuntimeMode.MODE_E,
            "a": RuntimeMode.MODE_A,
            "b": RuntimeMode.MODE_B,
            "c": RuntimeMode.MODE_C,
            "d": RuntimeMode.MODE_D,
            "e": RuntimeMode.MODE_E,
            "memetic": RuntimeMode.MODE_B,
            "adaptive": RuntimeMode.MODE_D,
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
    def from_config(cls, config: Mapping[str, Any]) -> RuntimeMode:
        """
        Infer runtime mode from loaded config dictionary.

        Args:
            config: Loaded configuration dictionary or Pydantic Config model

        Returns:
            RuntimeMode enum value

        Raises:
            ValueError: If mode cannot be determined from config
        """
        # Convert Pydantic model to dict if needed
        if hasattr(config, "model_dump"):
            config = config.model_dump()
        elif not isinstance(config, Mapping):
            raise ValueError(
                f"Config must be dict or Pydantic model, got {type(config)}"
            )

        # Try to extract mode from metadata first
        if "metadata" in config and "runtime_mode" in config["metadata"]:
            mode_str = config["metadata"]["runtime_mode"]
            return cls.from_string(mode_str)

        # Otherwise infer from config features
        rl_enabled = config.get("rl", {}).get("enabled", False)
        repair_enabled = (
            config.get("repair", {}).get("enabled") is not False
        )  # True or None = enabled
        adaptive_probs = config.get("ga", {}).get("use_adaptive_probabilities", False)
        enhancements = config.get("enhancements", {}).get("master_enabled", False)

        # Check if heuristics are enabled (any category with enabled heuristics)
        heuristics_config = config.get("heuristics", {})
        heuristics_enabled = False
        if heuristics_config:
            for category in [
                "construction",
                "perturbation",
                "improvement",
                "diversity",
                "meta",
            ]:
                if category in heuristics_config:
                    for _heuristic_name, heuristic_cfg in heuristics_config[
                        category
                    ].items():
                        if isinstance(heuristic_cfg, dict) and heuristic_cfg.get(
                            "enabled", False
                        ):
                            heuristics_enabled = True
                            break
                if heuristics_enabled:
                    break

        # Decision tree for mode inference
        if rl_enabled:
            # RL modes (5, 7, 9, 10)
            rl_mode = config.get("rl", {}).get("mode", "inference")
            if rl_mode == "hierarchical":
                return RuntimeMode.RL_HIERARCHICAL
            elif rl_mode == "multiagent":
                return RuntimeMode.RL_MULTIAGENT
            elif config.get("rl", {}).get("use_specialists", False):
                return RuntimeMode.RL_SPECIALISTS
            else:
                return RuntimeMode.RL_GUIDED
        elif enhancements and repair_enabled and heuristics_enabled and not rl_enabled:
            # Full GA with heuristics enabled (modes 3, 4, 6, 8)
            if (
                config.get("enhancements", {})
                .get("archive_diversity", {})
                .get("enabled", False)
            ):
                return RuntimeMode.ARCHIVE_DIVERSITY  # Mode 8
            elif not adaptive_probs:
                return RuntimeMode.ROUND_ROBIN  # Mode 6: Fixed round-robin
            else:
                return RuntimeMode.NSGA_FULL  # Mode 4: Full with local search
        elif repair_enabled and heuristics_enabled and not enhancements:
            # Heuristics without enhancements
            return RuntimeMode.NSGA_HEURISTICS  # Mode 3
        elif repair_enabled and not heuristics_enabled and not enhancements:
            # Repair only, no heuristics
            return RuntimeMode.NSGA_REPAIRS  # Mode 2
        elif not repair_enabled and not heuristics_enabled and not enhancements:
            # Pure baseline
            return RuntimeMode.BASELINE  # Mode 1
        else:
            # Fallback: if uncertain, default to baseline
            return RuntimeMode.BASELINE

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
    experiment_name: str | None = None
    output_dir: str | None = None
    seed: int = 69
    notes: str | None = None

    def __post_init__(self) -> None:
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
            "Experiment Configuration:",
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
