"""
Notebook utilities for RL experiments.

These helpers wrap the production ScheduleEnv and RL agents so notebooks stay
thin and aligned with the main codebase.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import init_config
from src.config.loader import dict_to_pydantic
from src.config.models import Config
from src.domain.types import SchedulingContext
from src.ga.population import generate_course_group_aware_population
from src.rl.agents import RandomAgent, create_dqn_agent, create_ppo_agent
from src.rl.gym_env.schedule_env import ScheduleEnv, create_schedule_env
from src.workflows.standard_run import load_input_data


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
    """Create and initialize a Pydantic config for notebooks.

    Args:
        seed: Random seed for the experiment.
        overrides: Optional flat config overrides.

    Returns:
        Initialized Config instance.
    """
    config_dict: dict[str, Any] = {
        "experiment_name": "notebook-experiment",
        "environment": "notebook",
        "ngen": 50,
        "pop_size": 20,
        "cxpb": 0.7,
        "mutpb": 0.2,
        "seed": seed,
    }
    if overrides:
        config_dict.update(overrides)

    config = dict_to_pydantic(config_dict)
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
    debug_logging: bool = False,
) -> ScheduleEnv:
    """Create a ScheduleEnv backed by a fresh population.

    Args:
        context: SchedulingContext for the environment.
        pop_size: Population size used for the environment.
        max_generations: Maximum GA generations per episode.
        max_steps: Maximum RL steps per episode.
        debug_logging: Enable verbose environment logging.

    Returns:
        Configured ScheduleEnv instance.
    """
    population = generate_course_group_aware_population(
        n=pop_size,
        context=context,
        parallel=False,
    )
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

        for method_key, method_cfg in methods.items():
            agent_type = method_cfg["agent_type"]
            env = create_env(
                context=context,
                pop_size=pop_size,
                max_generations=max_generations,
                max_steps=max_steps,
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
