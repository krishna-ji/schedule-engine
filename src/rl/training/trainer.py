"""
RL Trainer for heuristic selection agent.

Trains RL agents (PPO/DQN) to learn optimal heuristic selection strategies
for the GA scheduler. Supports:
- Basic training with progress tracking
- TensorBoard logging
- Model checkpointing
- Curriculum learning integration
- Validation set evaluation
"""

from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import time
import json
from datetime import datetime

import gymnasium as gym
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CallbackList, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import torch

from src.rl.agents import create_ppo_agent, create_dqn_agent
from src.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RolloutProgressCallback(BaseCallback):
    """Logs collected timesteps during training to avoid silent rollouts."""

    def __init__(self, log_interval_steps: int, total_timesteps: int) -> None:
        super().__init__()
        self.log_interval_steps = max(1, log_interval_steps)
        self.total_timesteps = total_timesteps
        self._last_logged = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_logged >= self.log_interval_steps:
            pct = (
                (self.num_timesteps / self.total_timesteps) * 100
                if self.total_timesteps
                else 0.0
            )
            logger.info(
                "Rollout progress: %s/%s env steps (%.1f%%)",
                f"{self.num_timesteps:,}",
                f"{self.total_timesteps:,}" if self.total_timesteps else "?",
                pct,
            )
            self._last_logged = self.num_timesteps
        return True


class RLTrainer:
    """
    Trains RL agent to select heuristics for GA scheduler.

    Features:
    - Support for PPO and DQN agents
    - TensorBoard logging
    - Model checkpointing with metadata
    - Curriculum learning support
    - Validation set evaluation
    - Training statistics tracking
    """

    def __init__(
        self,
        env: gym.Env,
        agent_type: str = "ppo",
        save_dir: Optional[str] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 1,
        n_envs: int = 1,
        use_subproc: bool = False,
        device: str = "auto",
        debug_logging: bool = False,
        **agent_kwargs,
    ):
        """
        Initialize trainer.

        Args:
            env: Gymnasium environment (ScheduleEnv) or env factory function
            agent_type: Agent type ("ppo" or "dqn")
            save_dir: Directory to save models
            tensorboard_log: TensorBoard log directory
            verbose: Verbosity level (0=none, 1=info, 2=debug)
            n_envs: Number of parallel environments (1=no parallelization)
            use_subproc: Use SubprocVecEnv for true parallelism (recommended for CPU-heavy)
            device: PyTorch device ("auto", "cuda", "cpu")
            debug_logging: Enable detailed progress logging
            **agent_kwargs: Additional agent-specific arguments
        """
        self.env = env
        self.agent_type = agent_type.lower()
        self.verbose = verbose
        self.n_envs = n_envs
        self.use_subproc = use_subproc
        self.debug_logging = debug_logging

        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Training device: {self.device}")
        if self.device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

        # Load config
        config = get_config()

        # Set up directories
        self.save_dir = Path(save_dir or config.rl.training.checkpoint_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_log = tensorboard_log or config.rl.training.tensorboard_log
        Path(self.tensorboard_log).mkdir(parents=True, exist_ok=True)

        # Initialize agent
        self.agent: Optional[BaseAlgorithm] = None
        self.agent_kwargs = agent_kwargs

        # Training statistics
        self.training_start_time: Optional[float] = None
        self.total_timesteps_trained: int = 0
        self.training_history: List[Dict[str, Any]] = []

        logger.info(f"Initialized RLTrainer with {agent_type.upper()} agent")
        logger.info(f"Save directory: {self.save_dir}")
        logger.info(f"TensorBoard logs: {self.tensorboard_log}")

    def create_agent(self, **override_kwargs) -> BaseAlgorithm:
        """
        Create RL agent with configured parameters.

        Args:
            **override_kwargs: Override agent parameters

        Returns:
            Configured RL agent
        """
        # Merge kwargs
        agent_kwargs = {**self.agent_kwargs, **override_kwargs}

        # Add device to agent kwargs
        agent_kwargs["device"] = self.device

        # Create agent based on type
        if self.agent_type == "ppo":
            agent = create_ppo_agent(
                env=self.env,
                tensorboard_log=self.tensorboard_log,
                verbose=self.verbose,
                **agent_kwargs,
            )
        elif self.agent_type == "dqn":
            agent = create_dqn_agent(
                env=self.env,
                tensorboard_log=self.tensorboard_log,
                verbose=self.verbose,
                **agent_kwargs,
            )
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        if self.debug_logging and self.agent_type == "ppo":
            rollout_steps = getattr(agent, "n_steps", 0)
            total_steps = rollout_steps * max(1, self.n_envs)
            logger.info(
                "PPO rollout size: %s steps/env (total %s per update)",
                rollout_steps,
                total_steps,
            )

        logger.info(f"Created {self.agent_type.upper()} agent")
        return agent

    def train(
        self,
        total_timesteps: int,
        callbacks: Optional[List[BaseCallback]] = None,
        tb_log_name: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = True,
    ) -> BaseAlgorithm:
        """
        Train agent for specified timesteps.

        Args:
            total_timesteps: Number of timesteps to train
            callbacks: List of training callbacks
            tb_log_name: TensorBoard run name
            reset_num_timesteps: Reset timestep counter
            progress_bar: Show progress bar

        Returns:
            Trained agent
        """
        # Create agent if not exists
        if self.agent is None:
            self.agent = self.create_agent()

        # Inject rollout progress callback if debug logging requested
        if self.debug_logging:
            if callbacks is None:
                callbacks = []
            callbacks = list(callbacks)
            log_interval = getattr(self.agent, "n_steps", 2048)
            callbacks.append(
                RolloutProgressCallback(
                    log_interval_steps=max(1, log_interval // 4),
                    total_timesteps=total_timesteps,
                )
            )

        # Set up callbacks
        callback_list = CallbackList(callbacks) if callbacks else None

        # Generate log name
        if tb_log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tb_log_name = f"{self.agent_type}_training_{timestamp}"

        # Start training
        logger.info(f"Starting training for {total_timesteps:,} timesteps...")
        logger.info(f"TensorBoard run name: {tb_log_name}")

        # DEBUG: Log rollout buffer info for PPO
        if self.agent_type == "ppo":
            n_steps = getattr(self.agent, "n_steps", 2048)
            batch_size = getattr(self.agent, "batch_size", 64)
            n_epochs = getattr(self.agent, "n_epochs", 10)
            logger.info(f"")
            logger.info(f"═" * 60)
            logger.info(f"🔍 PPO TRAINING DIAGNOSTICS")
            logger.info(f"═" * 60)
            logger.info(
                f"Rollout buffer: {n_steps} steps/env × {self.n_envs} envs = {n_steps * self.n_envs} total steps"
            )
            logger.info(
                f"PPO will collect {n_steps} steps from EACH of {self.n_envs} environments"
            )
            logger.info(
                f"Then train for {n_epochs} epochs with batch_size={batch_size}"
            )
            logger.info(f"")
            logger.info(f"⏳ EXPECTED BEHAVIOR:")
            logger.info(
                f"   1. Environments reset (you should see [ENV 0-{self.n_envs-1}] Reset logs)"
            )
            logger.info(
                f"   2. Collect {n_steps} steps from each env (watch for step logs)"
            )
            logger.info(f"   3. Policy update (progress bar increments)")
            logger.info(f"   4. Repeat until {total_timesteps:,} total steps")
            logger.info(f"")
            logger.info(
                f"If no environment logs appear within 1 minute, training is likely frozen."
            )
            logger.info(f"═" * 60)
            logger.info(f"")

        logger.info("🚀 Starting rollout collection now...")
        import sys

        sys.stdout.flush()  # Force output to appear immediately

        self.training_start_time = time.time()

        try:
            self.agent.learn(
                total_timesteps=total_timesteps,
                callback=callback_list,
                tb_log_name=tb_log_name,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=progress_bar,
            )

            # Update statistics
            self.total_timesteps_trained += total_timesteps
            elapsed = time.time() - self.training_start_time

            logger.info(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
            logger.info(f"Total timesteps trained: {self.total_timesteps_trained:,}")

            # Record training history
            self.training_history.append(
                {
                    "timesteps": total_timesteps,
                    "elapsed_seconds": elapsed,
                    "tb_log_name": tb_log_name,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

        return self.agent

    def save_model(
        self,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save trained model with metadata.

        Args:
            filename: Model filename (without extension)
            metadata: Additional metadata to save

        Returns:
            Path to saved model
        """
        if self.agent is None:
            raise RuntimeError("No agent to save. Train first.")

        # Ensure .zip extension
        if not filename.endswith(".zip"):
            filename = f"{filename}.zip"

        model_path = self.save_dir / filename

        # Save model
        self.agent.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Save metadata
        if metadata is None:
            metadata = {}

        metadata.update(
            {
                "agent_type": self.agent_type,
                "total_timesteps_trained": self.total_timesteps_trained,
                "save_time": datetime.now().isoformat(),
                "training_history": self.training_history,
            }
        )

        metadata_path = model_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

        return model_path

    def load_model(self, model_path: str) -> BaseAlgorithm:
        """
        Load trained model from checkpoint.

        Args:
            model_path: Path to model checkpoint (.zip)

        Returns:
            Loaded agent
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model from {model_path}...")

        if self.agent_type == "ppo":
            from src.rl.agents.ppo_agent import load_ppo_agent

            self.agent = load_ppo_agent(str(model_path), env=self.env)
        elif self.agent_type == "dqn":
            from src.rl.agents.dqn_agent import load_dqn_agent

            self.agent = load_dqn_agent(str(model_path), env=self.env)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        # Load metadata if exists
        metadata_path = model_path.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata: {metadata.get('save_time', 'unknown')}")

        logger.info("Model loaded successfully")
        return self.agent

    def evaluate(
        self,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate trained agent.

        Args:
            n_eval_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy

        Returns:
            Evaluation metrics
        """
        if self.agent is None:
            raise RuntimeError("No agent to evaluate. Train or load first.")

        logger.info(f"Evaluating agent over {n_eval_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []

        for episode in range(n_eval_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if self.verbose > 0:
                logger.debug(
                    f"Episode {episode+1}: reward={episode_reward:.2f}, length={episode_length}"
                )

        # Compute statistics
        import numpy as np

        metrics = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "std_length": float(np.std(episode_lengths)),
        }

        logger.info(
            f"Evaluation results: mean_reward={metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
        )

        return metrics

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        total_time = sum(h["elapsed_seconds"] for h in self.training_history)

        return {
            "total_timesteps": self.total_timesteps_trained,
            "total_training_time": total_time,
            "num_training_runs": len(self.training_history),
            "training_history": self.training_history,
        }


def create_trainer(
    env: gym.Env,
    agent_type: str = "ppo",
    **kwargs,
) -> RLTrainer:
    """
    Convenience function to create trainer.

    Args:
        env: Gymnasium environment
        agent_type: Agent type ("ppo" or "dqn")
        **kwargs: Additional trainer arguments

    Returns:
        Configured trainer
    """
    return RLTrainer(env=env, agent_type=agent_type, **kwargs)
