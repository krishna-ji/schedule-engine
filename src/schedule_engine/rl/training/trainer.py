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

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
from numpy.typing import NDArray
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import VecEnv

from schedule_engine.config import get_config
from schedule_engine.rl.agents import create_dqn_agent, create_ppo_agent
from schedule_engine.utils.logging_config import get_logger

logger = get_logger(__name__)

ObservationType = NDArray[Any]


class RolloutProgressCallback(BaseCallback):
    """Logs collected timesteps during training with timing info."""

    def __init__(self, log_interval_steps: int, total_timesteps: int) -> None:
        super().__init__()
        self.log_interval_steps = max(1, log_interval_steps)
        self.total_timesteps = total_timesteps
        self._last_logged = 0
        self._start_time: float | None = None
        self._last_time: float | None = None

    def _on_step(self) -> bool:
        import time

        current_time = time.time()

        # Initialize timing on first step
        if self._start_time is None:
            self._start_time = current_time
            self._last_time = current_time

        if self.num_timesteps - self._last_logged >= self.log_interval_steps:
            # Calculate timing metrics
            assert self._start_time is not None
            assert self._last_time is not None
            elapsed = current_time - self._start_time
            interval_time = current_time - self._last_time
            steps_in_interval = self.num_timesteps - self._last_logged

            # Calculate speed (steps/sec)
            speed = steps_in_interval / interval_time if interval_time > 0 else 0

            # Calculate ETA
            remaining_steps = self.total_timesteps - self.num_timesteps
            eta_seconds = remaining_steps / speed if speed > 0 else 0

            # Format time values as hh:mm:ss
            def format_time(seconds: float) -> str:
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"

            elapsed_str = format_time(elapsed)
            eta_str = format_time(eta_seconds)
            speed_str = f"{speed:.1f} steps/s"

            pct = (
                (self.num_timesteps / self.total_timesteps) * 100
                if self.total_timesteps
                else 0.0
            )

            # Calculate width for step formatting (based on total_timesteps)
            total_width = len(f"{self.total_timesteps:,}")

            logger.info(
                "[!ok] step %s/%s (%.1f%%), t=%s, eta=%s, %s",
                f"{self.num_timesteps:>{total_width},}",
                f"{self.total_timesteps:,}" if self.total_timesteps else "?",
                pct,
                elapsed_str,
                eta_str,
                speed_str,
            )

            self._last_logged = self.num_timesteps
            self._last_time = current_time

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
        env: gym.Env[Any, Any] | VecEnv,
        agent_type: str = "ppo",
        save_dir: str | None = None,
        tensorboard_log: str | None = None,
        verbose: int = 1,
        n_envs: int = 1,
        use_subproc: bool = False,
        device: str = "cpu",
        debug_logging: bool = False,
        **agent_kwargs: Any,
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
            device: PyTorch device (CPU-only)
            debug_logging: Enable detailed progress logging
            **agent_kwargs: Additional agent-specific arguments
        """
        self.env = env
        self.agent_type = agent_type.lower()
        self.verbose = verbose
        self.n_envs = n_envs
        self.use_subproc = use_subproc
        self.debug_logging = debug_logging

        # Device is forced to CPU for compatibility with CPU-only environments
        self.device = self._resolve_device(device)

        logger.info(f"Training device: {self.device}")

        # Load config
        config = get_config()
        self.config = config

        # Set up directories
        self.save_dir = Path(save_dir or config.rl.training.checkpoint_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.tensorboard_log = tensorboard_log or config.rl.training.tensorboard_log
        Path(self.tensorboard_log).mkdir(parents=True, exist_ok=True)

        # Initialize agent
        self.agent: BaseAlgorithm | None = None
        self.agent_kwargs = agent_kwargs

        # Training statistics
        self.training_start_time: float | None = None
        self.total_timesteps_trained: int = 0
        self.training_history: list[dict[str, Any]] = []

        logger.info(f"Initialized RLTrainer with {agent_type.upper()} agent")
        logger.info(f"Save directory: {self.save_dir}")
        logger.info(f"TensorBoard logs: {self.tensorboard_log}")

    def create_agent(self, **override_kwargs: Any) -> BaseAlgorithm:
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
            self.agent = create_ppo_agent(
                env=self.env,
                tensorboard_log=self.tensorboard_log,
                verbose=self.verbose,
                **agent_kwargs,
            )
        elif self.agent_type == "dqn":
            self.agent = create_dqn_agent(
                env=self.env,
                tensorboard_log=self.tensorboard_log,
                verbose=self.verbose,
                **agent_kwargs,
            )
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        if self.debug_logging and self.agent_type == "ppo":
            rollout_steps = getattr(self.agent, "n_steps", 0)
            total_steps = rollout_steps * max(1, self.n_envs)
            logger.info(
                "PPO rollout size: %s steps/env (total %s per update)",
                rollout_steps,
                total_steps,
            )

        logger.info(f"Created {self.agent_type.upper()} agent")
        return self.agent

    def _resolve_device(self, device: str) -> str:
        """Resolve training device (CUDA for RL neural networks if available).

        RL neural networks can leverage GPU acceleration for policy/value network
        training. The GA fitness evaluation remains CPU-only.
        """
        import torch

        device_lower = device.lower()

        # Explicit device selection
        if device_lower in ("cpu", "cuda"):
            if device_lower == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
            return device_lower

        # Auto-detect best device
        if device_lower == "auto":
            if torch.cuda.is_available():
                logger.info(f"Auto-detected CUDA GPU: {torch.cuda.get_device_name(0)}")
                return "cuda"
            logger.info("No CUDA GPU detected, using CPU")
            return "cpu"

        # Invalid device specification
        logger.warning(f"Unknown device '{device}', defaulting to CPU")
        return "cpu"

    def train_with_curriculum(
        self,
        curriculum_manager: Any,
        callbacks: list[BaseCallback] | None = None,
        progress_bar: bool = False,
    ) -> dict[str, Any]:
        """Train agent using curriculum learning stages.

        Args:
            curriculum_manager: CurriculumManager instance
            callbacks: Optional training callbacks
            progress_bar: Show progress bar

        Returns:
            Training statistics including curriculum progress
        """
        total_timesteps_trained = 0
        stage_results = []

        for stage_idx in range(len(curriculum_manager.stages)):
            current_stage = curriculum_manager.get_current_stage()
            if current_stage is None:
                break

            logger.info(
                f"Starting curriculum stage {stage_idx + 1}/{len(curriculum_manager.stages)}: {current_stage.name}"
            )
            logger.info(f"  Episodes: {current_stage.num_episodes}")
            logger.info(f"  Max generations: {current_stage.max_generations}")

            # Calculate timesteps for this stage (episodes × max_steps)
            # Default to 100 steps per episode if max_steps not configured
            stage_timesteps = current_stage.num_episodes * 100

            # Train for this stage
            self.train(
                total_timesteps=stage_timesteps,
                callbacks=callbacks,
                tb_log_name=f"curriculum_stage_{current_stage.name}",
                reset_num_timesteps=False,  # Accumulate across stages
                progress_bar=progress_bar,
            )

            total_timesteps_trained += stage_timesteps

            # Validation
            logger.info(f"Validating stage '{current_stage.name}'...")
            val_metrics = self.evaluate(
                n_eval_episodes=current_stage.validation_episodes
            )
            val_score = val_metrics["mean_reward"]

            logger.info(
                f"Validation: mean={val_score:.4f}, threshold={current_stage.threshold:.4f}"
            )

            # Check advancement
            should_advance = curriculum_manager.should_advance(val_score)

            stage_results.append(
                {
                    "stage_name": current_stage.name,
                    "stage_idx": stage_idx,
                    "timesteps": stage_timesteps,
                    "validation_score": val_score,
                    "advanced": should_advance,
                }
            )

            # Advance or continue
            if should_advance and stage_idx < len(curriculum_manager.stages) - 1:
                logger.info(f"[OK] Advancing from stage '{current_stage.name}'")
                curriculum_manager.advance_stage()
            elif stage_idx < len(curriculum_manager.stages) - 1:
                logger.warning(
                    f"Stage '{current_stage.name}' validation below threshold, advancing anyway"
                )
                curriculum_manager.advance_stage()

        # Return statistics
        stats = {
            "total_timesteps": total_timesteps_trained,
            "num_stages_completed": len(stage_results),
            "stage_results": stage_results,
            "curriculum_stats": curriculum_manager.get_statistics(),
        }

        logger.info(
            f"Curriculum training completed: {total_timesteps_trained:,} timesteps"
        )
        return stats

    def train(
        self,
        total_timesteps: int,
        callbacks: list[BaseCallback] | None = None,
        tb_log_name: str | None = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
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
            override_kwargs = self._prepare_agent_overrides(total_timesteps)
            self.agent = self.create_agent(**override_kwargs)

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
            logger.info("")
            logger.info("=" * 60)
            logger.info("PPO TRAINING DIAGNOSTICS")
            logger.info("=" * 60)
            logger.info(
                f"Rollout buffer: {n_steps} steps/env x {self.n_envs} envs = {n_steps * self.n_envs} total steps"
            )
            logger.info(
                f"PPO will collect {n_steps} steps from EACH of {self.n_envs} environments"
            )
            logger.info(
                f"Then train for {n_epochs} epochs with batch_size={batch_size}"
            )
            logger.info("")
            logger.info("EXPECTED BEHAVIOR:")
            logger.info(
                f"   1. Environments reset (you should see [ENV 0-{self.n_envs - 1}] Reset logs)"
            )
            logger.info(
                f"   2. Collect {n_steps} steps from each env (watch for step logs)"
            )
            logger.info("   3. Policy update (progress bar increments)")
            logger.info(f"   4. Repeat until {total_timesteps:,} total steps")
            logger.info("")
            logger.info(
                "If no environment logs appear within 1 minute, training is likely frozen."
            )
            logger.info("=" * 60)
            logger.info("")

        logger.info("Starting rollout collection now...")
        logger.info(
            "NOTE: First rollout collection may take 2-5 minutes (environments are working)"
        )
        logger.info(
            "      Progress bar will update after first policy update completes"
        )
        import sys

        sys.stdout.flush()  # Force output to appear immediately

        self.training_start_time = time.time()

        try:
            logger.info("Calling agent.learn() - entering SB3 training loop...")
            sys.stdout.flush()
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

            logger.info(
                f"Training completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)"
            )
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
        metadata: dict[str, Any] | None = None,
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
        model_path_obj = Path(model_path)

        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info(f"Loading model from {model_path}...")

        if self.agent_type == "ppo":
            from schedule_engine.rl.agents.ppo_agent import load_ppo_agent

            self.agent = load_ppo_agent(str(model_path_obj), env=self.env)
        elif self.agent_type == "dqn":
            from schedule_engine.rl.agents.dqn_agent import load_dqn_agent

            self.agent = load_dqn_agent(str(model_path_obj), env=self.env)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

        # Load metadata if exists
        metadata_path = model_path_obj.with_suffix(".json")
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata: {metadata.get('save_time', 'unknown')}")

        logger.info("Model loaded successfully")
        return self.agent

    def _get_eval_env(self) -> gym.Env[Any, Any] | VecEnv:
        """Return a gym.Env for evaluation, unwrapping VecEnv if needed.

        Returns VecEnv directly if underlying environments cannot be extracted.
        """
        if isinstance(self.env, VecEnv):
            # Try to get underlying environment from VecEnv
            # DummyVecEnv stores envs in .envs attribute
            envs = getattr(self.env, "envs", None)
            if envs and len(envs) > 0:
                # DummyVecEnv wraps environments directly
                base_env = envs[0]
                # Check if it's already a gym.Env
                if isinstance(base_env, gym.Env):
                    return base_env
                # Try to unwrap if it has an .env attribute
                underlying = getattr(base_env, "env", None)
                if underlying and isinstance(underlying, gym.Env):
                    return cast(gym.Env[Any, Any], underlying)

            # SubprocVecEnv doesn't expose environments directly
            # Return VecEnv directly (caller will use evaluate_policy instead)
            logger.warning(
                "VecEnv does not expose gym environments. Using VecEnv for evaluation."
            )
            return self.env  # type: ignore[return-value]

        if isinstance(self.env, gym.Env):
            return self.env

        raise RuntimeError("Trainer environment must be gym-compatible for evaluation")

    def evaluate(
        self,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
    ) -> dict[str, float]:
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

        # Check if we have a VecEnv
        if isinstance(self.env, VecEnv):
            # Use stable-baselines3's evaluate_policy for VecEnv
            from stable_baselines3.common.evaluation import evaluate_policy

            mean_reward, std_reward = evaluate_policy(
                self.agent,
                self.env,
                n_eval_episodes=n_eval_episodes,
                deterministic=deterministic,
                return_episode_rewards=False,
            )

            # Extract scalar values (evaluate_policy may return arrays)
            mean_rew_val = (
                float(mean_reward)
                if not isinstance(mean_reward, list)
                else float(mean_reward[0])
            )
            std_rew_val = (
                float(std_reward)
                if not isinstance(std_reward, list)
                else float(std_reward[0])
            )

            metrics = {
                "mean_reward": mean_rew_val,
                "std_reward": std_rew_val,
                "min_reward": mean_rew_val - std_rew_val,  # Approximation
                "max_reward": mean_rew_val + std_rew_val,  # Approximation
                "mean_length": 0.0,  # Not available from evaluate_policy
                "std_length": 0.0,
            }

            logger.info(
                f"Evaluation results: mean_reward={metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
            )
            return metrics

        # Single environment evaluation
        episode_rewards = []
        episode_lengths: list[int | float] = []

        eval_env = self._get_eval_env()

        for episode in range(n_eval_episodes):
            obs_raw, _ = eval_env.reset()
            obs: ObservationType = cast(ObservationType, obs_raw)
            done = False
            episode_reward: float = 0.0
            episode_length = 0  # Will be cast to float for calculations

            while not done:
                action, _ = self.agent.predict(obs, deterministic=deterministic)
                step_result: tuple[Any, Any, Any, Any, Any] = eval_env.step(action)  # type: ignore[assignment]
                obs_raw, reward, terminated, truncated, info = step_result
                obs = cast(ObservationType, obs_raw)
                done = terminated or truncated

                episode_reward += float(reward)
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(float(episode_length))  # Convert to float for mypy

            if self.verbose > 0:
                logger.debug(
                    f"Episode {episode + 1}: reward={episode_reward:.2f}, length={episode_length}"
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

    def get_training_statistics(self) -> dict[str, Any]:
        """Get training statistics."""
        total_time = sum(h["elapsed_seconds"] for h in self.training_history)

        return {
            "total_timesteps": self.total_timesteps_trained,
            "total_training_time": total_time,
            "num_training_runs": len(self.training_history),
            "training_history": self.training_history,
        }

    def _prepare_agent_overrides(self, total_timesteps: int | None) -> dict[str, Any]:
        """Auto-tune PPO rollout parameters for tiny smoke tests."""

        if total_timesteps is None or total_timesteps <= 0 or self.agent_type != "ppo":
            return {}

        n_envs = max(1, self.n_envs)
        base_n_steps = (
            self.agent_kwargs.get("n_steps") or self.config.rl.agent.ppo.n_steps
        )
        base_batch_size = (
            self.agent_kwargs.get("batch_size") or self.config.rl.agent.ppo.batch_size
        )
        rollout_target = base_n_steps * n_envs

        if total_timesteps >= rollout_target:
            return {}

        per_env_budget = max(1, total_timesteps // n_envs)
        adjusted_n_steps = max(1, min(base_n_steps, per_env_budget))
        adjusted_rollout = adjusted_n_steps * n_envs

        if adjusted_rollout == 0:
            adjusted_rollout = max(1, total_timesteps)

        batch_upper_bound = min(base_batch_size, adjusted_n_steps)
        adjusted_batch_size = self._select_batch_size(
            adjusted_rollout, batch_upper_bound
        )

        logger.warning(
            "Requested %s timesteps but PPO rollout requires %s (n_steps=%s, envs=%s); "
            "auto-adjusting to n_steps=%s, batch_size=%s so rollouts fit the smoke-test budget.",
            f"{total_timesteps:,}",
            f"{rollout_target:,}",
            base_n_steps,
            n_envs,
            adjusted_n_steps,
            adjusted_batch_size,
        )

        return {"n_steps": adjusted_n_steps, "batch_size": adjusted_batch_size}

    @staticmethod
    def _select_batch_size(rollout_size: int, upper_bound: int) -> int:
        """Pick the largest divisor (>=32 when possible) that fits PPO constraints."""

        if rollout_size <= 0:
            return max(1, upper_bound)

        min_candidate = 1 if rollout_size < 32 else 32
        max_candidate = max(1, min(upper_bound, rollout_size))

        for candidate in range(max_candidate, min_candidate - 1, -1):
            if rollout_size % candidate == 0:
                return candidate

        return min_candidate if rollout_size % min_candidate == 0 else 1


def create_trainer(
    env: gym.Env[Any, Any],
    agent_type: str = "ppo",
    **kwargs: Any,
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
