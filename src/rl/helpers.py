"""
Notebook utilities for RL experiments.

These helpers wrap the production ScheduleEnv and RL agents so notebooks stay
thin and aligned with the main codebase.
"""

from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.types import SchedulingContext

from src.config import Config, init_config
from src.ga.core.population import generate_course_group_aware_population
from src.io.data_store import load_input_data
from src.rl.agents import RandomAgent, create_dqn_agent, create_ppo_agent
from src.rl.gym_env.schedule_env import ScheduleEnv, create_schedule_env


def set_global_seed(seed: int) -> None:
    """Set global RNG seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def build_notebook_config(
    seed: int = 42,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Create and initialize a config for RL notebook experiments.

    Args:
        seed: Random seed for the experiment.
        overrides: Optional flat config overrides (GA keys extracted automatically).

    Returns:
        Initialized Config instance.
    """
    ga_kwargs: dict[str, Any] = {
        "ngen": 50,
        "pop_size": 20,
        "cxpb": 0.7,
        "mutpb": 0.2,
        "elite_preservation": True,
        "elite_size": 0.05,
        "use_adaptive_probabilities": True,
        "use_constraint_guided_mutation": True,
        "population_strategy": "hybrid",
        "validate_population_integrity": False,
    }
    if overrides:
        # Apply GA-level overrides
        for k in list(overrides):
            if k in ("ngen", "pop_size", "cxpb", "mutpb"):
                ga_kwargs[k] = overrides.pop(k)

    config = Config(
        name="notebook-experiment",
        environment="test",
        ga=ga_kwargs,
        repair={
            "enabled": True,
            "detection_strategy": "hybrid",
            "heuristics": {},
            "memetic_mode": False,
            "max_iterations": 3,
            "apply_after_mutation": True,
            "apply_after_crossover": True,
            "selective_mode": True,
            "recheck_after_repair": True,
            "budget_ms_per_generation": 50,
            "max_steps_per_individual": 5,
            "max_candidates_per_operator": 20,
            "policy": "round_robin",
            "epsilon": 0.1,
            "exhaustive_search": {"enabled": True, "generations": [3, 25]},
            "stagnation_repair": {"enabled": True, "patience": 5, "min_generation": 8},
            "selective_repair": {
                "enabled": True,
                "apply_probability": 0.3,
                "detection_strategy": "hybrid",
            },
        },
        enhancements={
            "master_enabled": True,
            "memetic_mode": True,
            "increased_population": True,
            "frequent_repair": True,
            "greedy_initialization_percent": 0.4,
            "hypermutation": {"enabled": True, "trigger_on_stagnation": True},
            "violation_heatmap": {"enabled": True, "target_hot_genes": True},
            "multi_neighborhood": {"enabled": True, "max_combinations": 50},
        },
        heuristics={"master_enabled": True},
        soft_constraints={
            "student_schedule_compactness": {
                "enabled": True,
                "weight": 1.0,
                "gap_penalty_per_quantum": 1,
            },
            "instructor_schedule_compactness": {
                "enabled": True,
                "weight": 1.0,
                "gap_penalty_per_quantum": 1,
            },
            "student_lunch_break": {
                "enabled": True,
                "weight": 1.0,
                "distance_penalty_per_quantum": 1,
            },
            "session_continuity": {"enabled": True, "weight": 1.0},
            "paired_cohort_practical_alignment": {"enabled": True, "weight": 1.0},
            "soft_weight_factor": 1.0,
        },
        rl={
            "enabled": False,
            "mode": "disabled",
            "environment": {
                "max_steps_per_episode": 100,
                "observation_history_size": 10,
                "diversity_update_interval": 1,
                "diversity_sample_size": None,
                "action_id_map": {},
                "render_mode": None,
            },
            "reward": {
                "fitness_weight": 10.0,
                "diversity_weight": 1.0,
                "time_weight": 0.01,
                "normalize": False,
            },
            "agent": {
                "type": "ppo",
                "model_path": "models/rl_agents/best_model.zip",
                "device": "cpu",
                "ppo": {
                    "learning_rate": 0.0003,
                    "n_steps": 512,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                },
                "dqn": {
                    "learning_rate": 0.0001,
                    "buffer_size": 100000,
                    "batch_size": 32,
                    "gamma": 0.99,
                    "exploration_fraction": 0.1,
                    "exploration_final_eps": 0.05,
                },
            },
            "training": {
                "total_timesteps": 100000,
                "checkpoint_interval": 10000,
                "evaluation_interval": 5000,
                "tensorboard_log": "logs/tensorboard",
                "checkpoint_dir": "models/rl_agents/checkpoints",
                "save_dir": "models/rl_agents",
                "verbose": 1,
                "curriculum": [],
            },
            "inference": {
                "batch_prediction": False,
                "timeout_ms": 10,
                "fallback_on_timeout": True,
                "cache_predictions": False,
            },
            "hybrid": {
                "mode": "rl_primary",
                "fallback_strategy": "random",
                "rl_probability": 0.8,
                "enable_action_masking": True,
            },
            "evaluation": {
                "baseline_strategies": ["random", "round_robin", "greedy"],
                "num_evaluation_episodes": 10,
                "save_metrics": True,
                "metrics_dir": "output/rl_metrics",
            },
            "logging": {
                "log_heuristic_usage": True,
                "log_rewards": True,
                "log_state_transitions": False,
                "log_inference_time": True,
            },
        },
        io={"data_dir": "data", "output_dir": "output"},
        parallel={"use_multiprocessing": True, "num_workers": None},
        cohort_pairs=[],
    )
    init_config(config)
    return config


def load_context(
    data_dir: str | Path,
    config: Config,
) -> tuple[Any, SchedulingContext]:
    """Load scheduling context using the production loader.

    Args:
        data_dir: Directory containing JSON data files.
        config: Active config instance.

    Returns:
        Tuple of (QuantumTimeSystem, SchedulingContext).
    """
    return load_input_data(str(data_dir), config)


def create_env(
    context: SchedulingContext,
    pop_size: int,
    max_generations: int,
    max_steps: int,
    initial_population: list[Any] | None = None,
    debug_logging: bool = False,
) -> ScheduleEnv:
    """Create a ScheduleEnv backed by a fresh population.

    Args:
        context: SchedulingContext for the environment.
        pop_size: Population size used for the environment.
        max_generations: Maximum GA generations per episode.
        max_steps: Maximum RL steps per episode.
        initial_population: Optional pre-generated population to reuse.
        debug_logging: Enable verbose environment logging.

    Returns:
        Configured ScheduleEnv instance.
    """
    if initial_population is None:
        population = generate_course_group_aware_population(
            n=pop_size,
            context=context,
            parallel=False,
        )
    else:
        population = initial_population
    return create_schedule_env(
        initial_population=population,
        context=context,
        max_generations=max_generations,
        max_steps_per_episode=max_steps,
        fast_evaluation=True,
        debug_logging=debug_logging,
    )


def train_agent(
    agent_type: str,
    env: ScheduleEnv,
    timesteps: int,
    seed: int,
    **agent_kwargs: Any,
) -> tuple[Any, float]:
    """Train an RL agent on the provided environment.

    Args:
        agent_type: One of "ppo", "dqn", or "random".
        env: ScheduleEnv instance.
        timesteps: Number of training steps.
        seed: Random seed for the agent.

    Returns:
        Tuple of (agent, training_time_seconds).
    """
    agent_type = agent_type.lower()
    agent: Any
    if agent_type == "ppo":
        agent = create_ppo_agent(env=env, seed=seed, verbose=0, **agent_kwargs)
    elif agent_type == "dqn":
        agent = create_dqn_agent(env=env, seed=seed, verbose=0, **agent_kwargs)
    elif agent_type == "random":
        agent = RandomAgent(env, seed=seed)
    else:
        raise ValueError(f"Unsupported agent_type: {agent_type}")

    start = time.time()
    if agent_type != "random":
        agent.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.time() - start
    return agent, elapsed


@dataclass
class EvaluationResult:
    """Results from a single evaluation run."""

    best_fitness: float
    convergence_gen: int


def evaluate_agent(
    agent: Any,
    env: ScheduleEnv,
    max_generations: int,
) -> EvaluationResult:
    """Evaluate a trained agent on the environment.

    Args:
        agent: Trained agent with a predict method.
        env: ScheduleEnv instance.
        max_generations: Max generations to simulate.

    Returns:
        EvaluationResult with best fitness and convergence generation.
    """
    obs, info = env.reset()
    convergence = max_generations

    for gen in range(max_generations):
        action, _ = agent.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        best_fitness = float(info["best_fitness"])

        if best_fitness == 0 and convergence == max_generations:
            convergence = gen

        if terminated or truncated:
            break

    return EvaluationResult(
        best_fitness=float(info["best_fitness"]),
        convergence_gen=convergence,
    )


def run_ablation(
    methods: dict[str, dict[str, Any]],
    data_dir: str | Path,
    trials: int,
    timesteps: int,
    pop_size: int,
    max_generations: int,
    max_steps: int,
    seed: int,
) -> dict[str, list[EvaluationResult]]:
    """Run a systematic ablation study across multiple methods.

    Args:
        methods: Mapping of method key to config dict with agent_type entries.
        data_dir: Directory containing JSON data files.
        trials: Number of trials per method.
        timesteps: Training timesteps per trial (ignored for random).
        pop_size: Population size.
        max_generations: Max generations per evaluation.
        max_steps: Max RL steps per episode.
        seed: Base seed for reproducibility.

    Returns:
        Dictionary mapping method key to list of EvaluationResult per trial.
    """
    results: dict[str, list[EvaluationResult]] = {k: [] for k in methods}

    for trial in range(trials):
        trial_seed = seed + trial * 101
        set_global_seed(trial_seed)
        config = build_notebook_config(
            seed=trial_seed, overrides={"pop_size": pop_size}
        )
        _, context = load_context(data_dir, config)
        base_population = generate_course_group_aware_population(
            n=pop_size,
            context=context,
            parallel=False,
        )

        for method_key, method_cfg in methods.items():
            set_global_seed(trial_seed)
            agent_type = method_cfg["agent_type"]
            env = create_env(
                context=context,
                pop_size=pop_size,
                max_generations=max_generations,
                max_steps=max_steps,
                initial_population=copy.deepcopy(base_population),
                debug_logging=False,
            )
            agent, _ = train_agent(
                agent_type=agent_type,
                env=env,
                timesteps=timesteps,
                seed=trial_seed,
            )
            result = evaluate_agent(agent, env, max_generations=max_generations)
            results[method_key].append(result)

    return results
