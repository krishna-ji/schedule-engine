"""
DQN agent wrapper for Stable-Baselines3.

Provides pre-configured DQN agent for heuristic selection.
"""

from typing import Optional, Dict, Any
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
import gymnasium as gym

from src.config import get_config


def create_dqn_agent(
    env: gym.Env,
    learning_rate: Optional[float] = None,
    buffer_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    gamma: Optional[float] = None,
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
    **kwargs,
) -> DQN:
    """
    Create and configure DQN agent.

    Args:
        env: Gymnasium environment
        learning_rate: Learning rate (overrides config)
        buffer_size: Replay buffer size (overrides config)
        batch_size: Minibatch size (overrides config)
        gamma: Discount factor (overrides config)
        tensorboard_log: TensorBoard log directory
        verbose: Verbosity level (0=none, 1=info, 2=debug)
        **kwargs: Additional DQN arguments

    Returns:
        Configured DQN agent
    """
    config = get_config()
    dqn_config = config.rl.agent.dqn

    # Use config values if not provided
    learning_rate = learning_rate or dqn_config.learning_rate
    buffer_size = buffer_size or dqn_config.buffer_size
    batch_size = batch_size or dqn_config.batch_size
    gamma = gamma or dqn_config.gamma

    # Wrap environment in DummyVecEnv if not already vectorized (required by SB3)
    if isinstance(env, VecEnv):
        vec_env = env
    else:
        vec_env = DummyVecEnv([lambda: env])

    # Extract device from kwargs or use config (prevents duplicate parameter error)
    device = kwargs.pop("device", config.rl.agent.device)

    # Create DQN agent
    model = DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=dqn_config.learning_starts,
        batch_size=batch_size,
        tau=dqn_config.tau,
        gamma=gamma,
        exploration_fraction=dqn_config.exploration_fraction,
        exploration_initial_eps=dqn_config.exploration_initial_eps,
        exploration_final_eps=dqn_config.exploration_final_eps,
        tensorboard_log=tensorboard_log or config.rl.training.tensorboard_log,
        verbose=verbose,
        device=device,
        **kwargs,
    )

    return model


def load_dqn_agent(
    model_path: str,
    env: Optional[gym.Env] = None,
    device: str = "auto",
) -> DQN:
    """
    Load trained DQN agent from checkpoint.

    Args:
        model_path: Path to saved model (.zip)
        env: Environment to use (optional)
        device: Device to load model on

    Returns:
        Loaded DQN agent
    """
    if env is not None:
        # Wrap environment if not already vectorized
        if isinstance(env, VecEnv):
            vec_env = env
        else:
            vec_env = DummyVecEnv([lambda: env])
        model = DQN.load(model_path, env=vec_env, device=device)
    else:
        model = DQN.load(model_path, device=device)

    return model


def get_dqn_config() -> Dict[str, Any]:
    """Get current DQN configuration from config."""
    config = get_config()
    return {
        "learning_rate": config.rl.agent.dqn.learning_rate,
        "buffer_size": config.rl.agent.dqn.buffer_size,
        "learning_starts": config.rl.agent.dqn.learning_starts,
        "batch_size": config.rl.agent.dqn.batch_size,
        "tau": config.rl.agent.dqn.tau,
        "gamma": config.rl.agent.dqn.gamma,
        "exploration_fraction": config.rl.agent.dqn.exploration_fraction,
        "exploration_initial_eps": config.rl.agent.dqn.exploration_initial_eps,
        "exploration_final_eps": config.rl.agent.dqn.exploration_final_eps,
        "device": config.rl.agent.device,
    }
