"""Python-native presets for RL training profiles."""

from __future__ import annotations

from typing import Any, Final

TRAINING_BASE_DEFAULTS: Final[dict[str, Any]] = {
    "profile": "base",
    "agent_type": "ppo",
    "data_dir": "data",
    "save_dir": "models/rl_agents",
    "tensorboard_log": "logs/tensorboard/train",
    "timesteps": 50_000,
    "max_generations": 120,
    "max_steps": 60,
    "population_size": 50,
    "eval_episodes": 5,
    "enable_eval": True,
    "save_prefix": "rl_agent",
    "seed": 42,
    "parallel": {
        "n_envs": None,
        "use_subproc": True,
    },
    "device": "auto",
}

TRAINING_PROFILE_OVERRIDES: Final[dict[str, dict[str, Any]]] = {
    "test": {
        "profile": "test",
        "timesteps": 500,
        "max_generations": 30,
        "max_steps": 20,
        "population_size": 10,
        "eval_episodes": 1,
        "save_prefix": "rl_agent_test",
        "parallel": {
            "n_envs": 1,
            "use_subproc": False,
        },
        "debug_logging": True,
        "debug_log_interval": 10,
        "device": "auto",
    },
    "med": {
        "profile": "med",
        "timesteps": 100_000,
        "max_generations": 120,
        "max_steps": 60,
        "population_size": 50,
        "eval_episodes": 5,
        "save_prefix": "rl_agent_med",
        "parallel": {
            "n_envs": None,
            "use_subproc": True,
        },
    },
    "prod": {
        "profile": "prod",
        "timesteps": 300_000,
        "max_generations": 200,
        "max_steps": 80,
        "population_size": 80,
        "eval_episodes": 10,
        "save_prefix": "rl_agent_prod",
        "parallel": {
            "n_envs": None,
            "use_subproc": True,
        },
        "debug_logging": True,
        "debug_log_interval": 25,
    },
    "debug": {
        "inherits": "test",
        "profile": "debug",
        "timesteps": 200,
        "max_generations": 10,
        "max_steps": 5,
        "population_size": 8,
        "seed": 123,
        "parallel": {
            "n_envs": 1,
            "use_subproc": False,
        },
        "device": "cpu",
        "save_prefix": "rl_agent_debug",
    },
}
