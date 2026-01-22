"""
PPO agent wrapper for Stable-Baselines3.

Provides pre-configured PPO agent for heuristic selection.
"""

from typing import Any

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv

from schedule_engine.config import get_config


def create_ppo_agent(
    env: gym.Env | VecEnv,
    learning_rate: float | None = None,
    n_steps: int | None = None,
    batch_size: int | None = None,
    n_epochs: int | None = None,
    gamma: float | None = None,
    tensorboard_log: str | None = None,
    verbose: int = 1,
    **kwargs: Any,
) -> PPO:
    """
    Create and configure PPO agent.

    Args:
        env: Gymnasium environment
        learning_rate: Learning rate (overrides config)
        n_steps: Steps per update (overrides config)
        batch_size: Minibatch size (overrides config)
        n_epochs: Optimization epochs per update (overrides config)
        gamma: Discount factor (overrides config)
        tensorboard_log: TensorBoard log directory
        verbose: Verbosity level (0=none, 1=info, 2=debug)
        **kwargs: Additional PPO arguments

    Returns:
        Configured PPO agent
    """
    config = get_config()
    ppo_config = config.rl.agent.ppo

    # Use config values if not provided
    learning_rate = learning_rate or ppo_config.learning_rate
    n_steps = n_steps or ppo_config.n_steps
    batch_size = batch_size or ppo_config.batch_size
    n_epochs = n_epochs or ppo_config.n_epochs
    gamma = gamma or ppo_config.gamma

    # Wrap environment in DummyVecEnv if not already vectorized (required by SB3)
    if isinstance(env, VecEnv):
        vec_env = env
        n_envs = vec_env.num_envs
    else:
        vec_env = DummyVecEnv([lambda: env])
        n_envs = 1

    # VALIDATE: batch_size must evenly divide rollout buffer
    rollout_buffer_size = n_steps * n_envs
    if rollout_buffer_size % batch_size != 0:
        valid_batch_sizes = [
            d
            for d in range(64, min(rollout_buffer_size + 1, 1025), 64)
            if rollout_buffer_size % d == 0
        ]
        raise ValueError(
            f"batch_size ({batch_size}) must evenly divide n_steps * n_envs ({n_steps} * {n_envs} = {rollout_buffer_size}). "
            f"Current remainder: {rollout_buffer_size % batch_size}. "
            f"Valid batch_sizes (multiples of 64): {valid_batch_sizes[:10]}"
        )

    # Extract device from kwargs or use config (prevents duplicate parameter error)
    device = kwargs.pop("device", config.rl.agent.device)

    # Create PPO agent
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=ppo_config.gae_lambda,
        clip_range=ppo_config.clip_range,
        ent_coef=ppo_config.ent_coef,
        vf_coef=ppo_config.vf_coef,
        max_grad_norm=ppo_config.max_grad_norm,
        tensorboard_log=tensorboard_log or config.rl.training.tensorboard_log,
        verbose=verbose,
        device=device,
        **kwargs,
    )

    return model


def load_ppo_agent(
    model_path: str,
    env: gym.Env | VecEnv | None = None,
    device: str = "cpu",
) -> PPO:
    """
    Load trained PPO agent from checkpoint.

    Args:
        model_path: Path to saved model (.zip)
        env: Environment to use (optional)
        device: Device to load model on (CPU-only)

    Returns:
        Loaded PPO agent
    """
    if env is not None:
        # Wrap environment if not already vectorized
        vec_env = env if isinstance(env, VecEnv) else DummyVecEnv([lambda: env])
        model = PPO.load(model_path, env=vec_env, device=device)
    else:
        model = PPO.load(model_path, device=device)

    return model


def get_ppo_config() -> dict[str, Any]:
    """Get current PPO configuration from config."""
    config = get_config()
    return {
        "learning_rate": config.rl.agent.ppo.learning_rate,
        "n_steps": config.rl.agent.ppo.n_steps,
        "batch_size": config.rl.agent.ppo.batch_size,
        "n_epochs": config.rl.agent.ppo.n_epochs,
        "gamma": config.rl.agent.ppo.gamma,
        "gae_lambda": config.rl.agent.ppo.gae_lambda,
        "clip_range": config.rl.agent.ppo.clip_range,
        "ent_coef": config.rl.agent.ppo.ent_coef,
        "vf_coef": config.rl.agent.ppo.vf_coef,
        "max_grad_norm": config.rl.agent.ppo.max_grad_norm,
        "device": config.rl.agent.device,
    }
