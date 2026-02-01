"""
Config loader utilities for new experiment system.

Provides helper functions to convert between dataclass dicts and Pydantic models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from schedule_engine.config.models import Config as PydanticConfig


def dict_to_pydantic(config_dict: dict[str, Any]) -> PydanticConfig:
    """
    Convert configuration dictionary to Pydantic Config model.

    This function handles the mapping from flat dictionary structure
    (from ExperimentConfig.with_profile()) to the nested Pydantic structure.

    Args:
        config_dict: Flat configuration dictionary

    Returns:
        Validated Pydantic Config instance
    """
    from schedule_engine.config.models import Config

    # Build nested structure expected by Pydantic model
    pydantic_dict = _build_pydantic_structure(config_dict)

    return Config(**pydantic_dict)


def _build_pydantic_structure(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Convert flat config dict to nested Pydantic structure.

    Maps flat keys to nested structure:
    - ngen, pop_size, etc. → ga.*
    - repair_enabled, etc. → repair.*
    - heuristics_mode, etc. → heuristics.*
    - rl_enabled, etc. → rl.*

    Args:
        flat_dict: Flat dictionary from ExperimentConfig

    Returns:
        Nested dictionary matching Pydantic Config schema
    """
    # Extract metadata
    name = flat_dict.get("experiment_name", "Unknown Experiment")
    environment = flat_dict.get("environment", "test")  # Default to test, not custom

    # Extract I/O paths
    data_dir = flat_dict.get("data_dir", "data")
    output_dir = flat_dict.get("output_dir", "output")

    # Build nested structure
    return {
        "name": name,
        "environment": environment,
        "io": {
            "data_dir": data_dir,
            "output_dir": output_dir,
        },
        "time": _extract_time_config(flat_dict),
        "ga": _extract_ga_config(flat_dict),
        "constraints": _extract_constraints_config(flat_dict),
        "fitness": _extract_fitness_config(flat_dict),
        "population": _extract_population_config(flat_dict),
        "repair": _extract_repair_config(flat_dict),
        "heuristics": _extract_heuristics_config(flat_dict),
        "lns": _extract_lns_config(flat_dict),
        "rl": _extract_rl_config(flat_dict),
        "parallel": _extract_parallel_config(flat_dict),
        "metrics": _extract_metrics_config(flat_dict),
        "export": _extract_export_config(flat_dict),
        "enhancements": _extract_enhancements_config(flat_dict),
    }


def _extract_time_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract time-related configuration."""
    return {
        "quantum_minutes": flat_dict.get("quantum_minutes", 60),
        "opening_time": flat_dict.get("opening_time", "10:00"),
        "closing_time": flat_dict.get("closing_time", "17:00"),
        "closed_days": flat_dict.get("closed_days", ["Saturday"]),
        "cohort_pairs": flat_dict.get("cohort_pairs", []),
        "midday_break_start": flat_dict.get("midday_break_start", "12:00"),
        "midday_break_end": flat_dict.get("midday_break_end", "13:00"),
        "enforce_break_placement": flat_dict.get("enforce_break_placement", False),
        "break_window_start": flat_dict.get("break_window_start", "12:00"),
        "break_window_end": flat_dict.get("break_window_end", "14:00"),
        "break_min_quanta": flat_dict.get("break_min_quanta", 1),
        "break_violation_penalty": flat_dict.get("break_violation_penalty", 50),
        "max_session_coalescence": flat_dict.get("max_session_coalescence", 3),
        "max_sessions_per_day": flat_dict.get("max_sessions_per_day", 3),
        "preferred_block_size_min": flat_dict.get("preferred_block_size_min", 1),
        "preferred_block_size_max": flat_dict.get("preferred_block_size_max", 3),
        "theory_isolated_penalty": flat_dict.get("theory_isolated_penalty", 10),
        "theory_oversized_penalty_per_quantum": flat_dict.get(
            "theory_oversized_penalty_per_quantum", 5
        ),
        "theory_max_excused_isolated": flat_dict.get("theory_max_excused_isolated", 1),
        "lab_isolated_penalty": flat_dict.get("lab_isolated_penalty", 15),
        "lab_oversized_penalty_per_quantum": flat_dict.get(
            "lab_oversized_penalty_per_quantum", 8
        ),
        "practical_fragmentation_penalty": flat_dict.get(
            "practical_fragmentation_penalty", 20
        ),
        "isolated_session_penalty": flat_dict.get("isolated_session_penalty", 5),
        "oversized_block_penalty_per_quantum": flat_dict.get(
            "oversized_block_penalty_per_quantum", 1
        ),
        "day_overrides": flat_dict.get("day_overrides", {}),
    }


def _extract_ga_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract GA algorithm configuration."""
    return {
        "ngen": flat_dict.get("ngen", 100),
        "pop_size": flat_dict.get("pop_size", 50),
        "cxpb": flat_dict.get("cxpb", 0.70),
        "mutpb": flat_dict.get("mutpb", 0.20),
        "elite_preservation": flat_dict.get("elite_preservation", True),
        "elite_size": flat_dict.get("elite_size", 0.05),
        "tournament_size": flat_dict.get("tournament_size", 2),
        "use_constraint_guided_mutation": flat_dict.get(
            "use_constraint_guided_mutation", False
        ),
    }


def _extract_constraints_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract constraints configuration."""
    return {
        "hard": {
            "instructor_conflicts": True,
            "group_conflicts": True,
            "room_conflicts": True,
            "instructor_availability": True,
            "group_availability": True,
            "room_availability": True,
            "instructor_qualification": True,
            "room_type_mismatch": True,
            "room_capacity": True,
            "paired_cohort_practicals": True,
        },
        "soft": {
            "minimize_gaps": True,
            "cluster_sessions": True,
            "preferred_times": True,
            "avoid_long_days": True,
            "balanced_schedule": True,
            "room_preferences": True,
        },
    }


def _extract_fitness_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract fitness function configuration."""
    return {
        "weights": (-1.0, -0.01),  # (hard, soft)
        "penalty_instructor_conflict": 100,
        "penalty_group_conflict": 100,
        "penalty_room_conflict": 100,
        "penalty_instructor_availability": 50,
        "penalty_group_availability": 50,
        "penalty_room_availability": 50,
        "penalty_instructor_qualification": 75,
        "penalty_room_type_mismatch": 75,
        "penalty_room_capacity": 50,
        "penalty_paired_cohort_practicals": 80,
        "penalty_schedule_gap": 5,
        "penalty_session_clustering": 3,
        "penalty_preferred_time": 2,
        "penalty_long_day": 4,
        "penalty_unbalanced_schedule": 3,
        "penalty_room_preference": 1,
    }


def _extract_population_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract population initialization configuration."""
    return {
        "strategy": flat_dict.get("population_strategy", "random"),
        "greedy_percentage": flat_dict.get("greedy_percentage", 0.00),
        "smart_percentage": flat_dict.get("smart_percentage", 0.00),
        "random_percentage": flat_dict.get("random_percentage", 1.00),
    }


def _extract_repair_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract repair/IGLS configuration."""
    return {
        "enabled": flat_dict.get("repair_enabled", False),
        "max_iterations": flat_dict.get("repair_max_iterations", 100),
        "apply_after_mutation": flat_dict.get("repair_apply_after_mutation", False),
        "apply_after_crossover": flat_dict.get("repair_apply_after_crossover", False),
        "memetic_mode": flat_dict.get("repair_memetic_mode", False),
        "elite_percentage": flat_dict.get("repair_elite_percentage", 0.20),
        "budget_ms_per_generation": flat_dict.get(
            "repair_budget_ms_per_generation", 50
        ),
        "max_steps_per_individual": flat_dict.get(
            "repair_max_steps_per_individual", 5
        ),
        "max_candidates_per_operator": flat_dict.get(
            "repair_max_candidates_per_operator", 20
        ),
        "policy": flat_dict.get("repair_policy", "round_robin"),
        "epsilon": flat_dict.get("repair_epsilon", 0.1),
        "heuristics_overrides": flat_dict.get("repair_heuristics_overrides", {}),
    }


def _extract_heuristics_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract heuristics configuration."""
    return {
        "master_enabled": flat_dict.get("heuristics_master_enabled", False),
        "mode": flat_dict.get("heuristics_mode", "off"),
        "round_robin_interval": flat_dict.get("heuristics_round_robin_interval", 50),
        "adaptive_selection_enabled": flat_dict.get(
            "heuristics_adaptive_selection_enabled", False
        ),
        "adaptive_priority_enabled": flat_dict.get(
            "heuristics_adaptive_priority_enabled", False
        ),
        "adaptive_update_interval": flat_dict.get(
            "heuristics_adaptive_update_interval", 10
        ),
        "rl_enabled": flat_dict.get("heuristics_rl_enabled", False),
    }


def _extract_lns_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract LNS configuration."""
    return {
        "enabled": flat_dict.get("lns_enabled", False),
        "destroy_percentage": flat_dict.get("lns_destroy_percentage", 0.30),
        "max_iterations": flat_dict.get("lns_max_iterations", 50),
    }


def _extract_rl_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract RL configuration."""
    # Map rl_mode to valid Literal values
    rl_mode = flat_dict.get("rl_mode", "disabled")
    if rl_mode == "off":
        rl_mode = "disabled"

    return {
        "enabled": flat_dict.get("rl_enabled", False),
        "mode": rl_mode,
        "agent": {
            "type": flat_dict.get("rl_agent_type", "ppo"),
            "model_path": flat_dict.get(
                "rl_agent_path", "models/rl_agents/best_model.zip"
            ),
        },
        "environment": {},
        "reward": {},
        "training": {},
        "inference": {},
        "hybrid": {},
        "evaluation": {},
        "logging": {},
    }


def _extract_parallel_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract parallel processing configuration."""
    return {
        "use_multiprocessing": flat_dict.get("use_multiprocessing", True),
        "num_workers": flat_dict.get("num_workers"),
    }


def _extract_metrics_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract metrics configuration."""
    return {
        "advanced_metrics_frequency": flat_dict.get(
            "metrics_advanced_metrics_frequency", 50
        ),
        "enable_hypervolume": True,
        "enable_igd": True,
        "enable_gd": True,
        "enable_diversity": True,
    }


def _extract_export_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract export configuration."""
    return {
        "pdf_enabled": True,
        "json_enabled": True,
        "plots_enabled": True,
        "statistics_enabled": True,
    }


def _extract_enhancements_config(flat_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract advanced enhancements configuration."""
    return {
        "master_enabled": flat_dict.get("enhancements_master_enabled", False),
        "diversity_maintenance_enabled": flat_dict.get(
            "diversity_maintenance_enabled", False
        ),
        "archive_diversity_enabled": flat_dict.get("archive_diversity_enabled", False),
    }


__all__ = [
    "dict_to_pydantic",
]
