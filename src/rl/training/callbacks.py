"""
Custom training callbacks for RL agent training.

Provides Stable-Baselines3 compatible callbacks:
- PeriodicEvaluationCallback: Evaluate agent periodically
- EarlyStoppingCallback: Stop training if no improvement
- CheckpointCallback: Save model checkpoints
- ManifestCallback: Track checkpoints in manifest.json
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PeriodicEvaluationCallback(BaseCallback):
    """
    Evaluates agent periodically during training.

    Runs evaluation episodes at specified intervals and logs results
    to TensorBoard and console.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        deterministic: bool = True,
        log_path: str | None = None,
        verbose: int = 1,
    ):
        """
        Initialize callback.

        Args:
            eval_env: Evaluation environment (can be same as training env)
            eval_freq: Evaluate every N timesteps
            n_eval_episodes: Number of episodes per evaluation
            deterministic: Use deterministic policy during evaluation
            log_path: Path to save evaluation logs (JSON)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.log_path = Path(log_path) if log_path else None

        self.eval_history: list[dict[str, float]] = []
        self.best_mean_reward = -np.inf

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        """Called at each training step."""

        if self.n_calls % self.eval_freq == 0:
            # Run evaluation
            episode_rewards = []
            episode_lengths = []

            for _ in range(self.n_eval_episodes):
                reset_result = self.eval_env.reset()
                # Handle both old gym (obs) and new gymnasium (obs, info) returns
                obs = (
                    reset_result[0] if isinstance(reset_result, tuple) else reset_result
                )
                done = False
                episode_reward = 0
                episode_length = 0

                while not done:
                    action, _ = self.model.predict(
                        obs, deterministic=self.deterministic
                    )  # type: ignore[arg-type]
                    step_result = self.eval_env.step(action)

                    # Handle both old gym and new gymnasium APIs
                    if (
                        len(step_result) == 5
                    ):  # gymnasium: obs, reward, terminated, truncated, info
                        obs_new, reward, terminated, truncated, _ = step_result
                        obs = obs_new  # type: ignore[assignment]
                        done = terminated or truncated
                    else:  # old gym: obs, reward, done, info
                        obs_new, reward, done_result, _ = step_result
                        obs = obs_new  # type: ignore[assignment]
                        done = (
                            done_result
                            if isinstance(done_result, bool)
                            else bool(done_result[0])
                        )

                    episode_reward += (
                        reward[0] if isinstance(reward, np.ndarray) else reward
                    )
                    episode_length += 1

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

            # Compute statistics
            mean_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))
            mean_length = float(np.mean(episode_lengths))

            # Log to TensorBoard
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/std_reward", std_reward)
            self.logger.record("eval/mean_length", mean_length)
            self.logger.record(
                "eval/best_mean_reward", max(self.best_mean_reward, mean_reward)
            )

            # Update best
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward

            # Store history
            eval_result: dict[str, float | str] = {
                "timestep": float(self.n_calls),
                "mean_reward": mean_reward,
                "std_reward": std_reward,
                "mean_length": mean_length,
                "min_reward": float(np.min(episode_rewards)),
                "max_reward": float(np.max(episode_rewards)),
                "timestamp": datetime.now().isoformat(),
            }
            self.eval_history.append(eval_result)  # type: ignore[arg-type]

            # Log to console
            if self.verbose > 0:
                logger.info(
                    f"Eval @ {self.n_calls:,} steps: "
                    f"reward={mean_reward:.2f}±{std_reward:.2f}, "
                    f"length={mean_length:.1f}"
                )

            # Save to file
            if self.log_path:
                with open(self.log_path, "w") as f:
                    json.dump(self.eval_history, f, indent=2)

        return True


class EarlyStoppingCallback(BaseCallback):
    """
    Stops training if no improvement for N evaluations.

    Monitors evaluation metric and stops training if no improvement
    is observed for specified patience period.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.01,
        metric_name: str = "eval/mean_reward",
        verbose: int = 1,
    ):
        """
        Initialize callback.

        Args:
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum change to count as improvement
            metric_name: Name of metric to monitor in logger
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name

        self.best_metric = -np.inf
        self.wait = 0
        self.stopped_timestep = 0

    def _on_step(self) -> bool:
        """Check if training should stop."""

        # Get current metric from logger
        if self.metric_name in self.logger.name_to_value:
            current_metric = self.logger.name_to_value[self.metric_name]

            # Check for improvement
            if current_metric > self.best_metric + self.min_delta:
                self.best_metric = current_metric
                self.wait = 0

                if self.verbose > 0:
                    logger.debug(f"New best {self.metric_name}: {current_metric:.4f}")
            else:
                self.wait += 1

                if self.verbose > 1:
                    logger.debug(
                        f"No improvement for {self.wait}/{self.patience} checks "
                        f"(current={current_metric:.4f}, best={self.best_metric:.4f})"
                    )

                if self.wait >= self.patience:
                    self.stopped_timestep = self.n_calls

                    if self.verbose > 0:
                        logger.warning(
                            f"Early stopping at {self.n_calls:,} steps: "
                            f"no improvement for {self.patience} evaluations "
                            f"(best={self.best_metric:.4f})"
                        )

                    return False  # Stop training

        return True  # Continue training


class CheckpointCallback(BaseCallback):
    """
    Saves model checkpoints during training.

    Saves periodic checkpoints and tracks best model based on
    evaluation metric.
    """

    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "models/checkpoints",
        name_prefix: str = "rl_model",
        save_best: bool = True,
        metric_name: str = "eval/mean_reward",
        verbose: int = 1,
    ):
        """
        Initialize callback.

        Args:
            save_freq: Save checkpoint every N timesteps
            save_path: Directory to save checkpoints
            name_prefix: Prefix for checkpoint filenames
            save_best: Also save best model separately
            metric_name: Metric to use for determining best model
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_best = save_best
        self.metric_name = metric_name

        self.save_path.mkdir(parents=True, exist_ok=True)

        self.best_metric = -np.inf
        self.checkpoint_count = 0

    def _on_step(self) -> bool:
        """Save checkpoint if needed."""

        if self.n_calls % self.save_freq == 0:
            # Save periodic checkpoint
            checkpoint_path = self.save_path / f"{self.name_prefix}_step_{self.n_calls}"
            self.model.save(checkpoint_path)
            self.checkpoint_count += 1

            if self.verbose > 0:
                logger.info(
                    f"Saved checkpoint #{self.checkpoint_count} at {self.n_calls:,} steps: {checkpoint_path}.zip"
                )

            # Check if best model
            if self.save_best and self.metric_name in self.logger.name_to_value:
                current_metric = self.logger.name_to_value[self.metric_name]

                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    best_path = self.save_path / f"{self.name_prefix}_best"
                    self.model.save(best_path)

                    if self.verbose > 0:
                        logger.info(
                            f"New best model saved: {self.metric_name}={current_metric:.4f}"
                        )

        return True


class ManifestCallback(BaseCallback):
    """
    Tracks checkpoints in manifest.json file.

    Records metadata for each checkpoint including:
    - Timestep, timestamp
    - Validation metrics
    - Hyperparameters
    - Stage info (for curriculum learning)
    """

    def __init__(
        self,
        manifest_path: str = "models/rl_agents/manifest.json",
        checkpoint_freq: int = 10000,
        stage_name: str | None = None,
        seed: int | None = None,
        verbose: int = 1,
    ):
        """
        Initialize callback.

        Args:
            manifest_path: Path to manifest JSON file
            checkpoint_freq: Checkpoint frequency (must match CheckpointCallback)
            stage_name: Current training stage (for curriculum learning)
            seed: Random seed
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.manifest_path = Path(manifest_path)
        self.checkpoint_freq = checkpoint_freq
        self.stage_name = stage_name
        self.seed = seed

        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing manifest
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> list[Any]:
        """Load existing manifest or create empty."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)  # type: ignore[no-any-return]
        return []

    def _save_manifest(self):
        """Save manifest to file."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _on_step(self) -> bool:
        """Record checkpoint in manifest."""

        if self.n_calls % self.checkpoint_freq == 0:
            # Collect metrics from logger
            metrics = {}
            for key in ["eval/mean_reward", "eval/std_reward", "eval/mean_length"]:
                if key in self.logger.name_to_value:
                    metrics[key.replace("eval/", "")] = float(
                        self.logger.name_to_value[key]
                    )

            # Create checkpoint entry
            entry = {
                "timestep": self.n_calls,
                "timestamp": datetime.now().isoformat(),
                "stage": self.stage_name,
                "seed": self.seed,
                "metrics": metrics,
                "status": "checkpoint",
            }

            self.manifest.append(entry)
            self._save_manifest()

            if self.verbose > 1:
                logger.debug(
                    f"Updated manifest with checkpoint at {self.n_calls:,} steps"
                )

        return True


def create_training_callbacks(
    eval_env: VecEnv,
    save_dir: str = "models/checkpoints",
    manifest_path: str = "models/rl_agents/manifest.json",
    eval_freq: int = 5000,
    save_freq: int = 10000,
    n_eval_episodes: int = 5,
    patience: int = 5,
    stage_name: str | None = None,
    seed: int | None = None,
    verbose: int = 1,
) -> list:
    """
    Create standard set of training callbacks.

    Args:
        eval_env: Evaluation environment
        save_dir: Directory for checkpoints
        manifest_path: Path to manifest file
        eval_freq: Evaluation frequency
        save_freq: Checkpoint save frequency
        n_eval_episodes: Episodes per evaluation
        patience: Early stopping patience
        stage_name: Current training stage
        seed: Random seed
        verbose: Verbosity level

    Returns:
        List of configured callbacks
    """
    callbacks = [
        PeriodicEvaluationCallback(
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            verbose=verbose,
        ),
        EarlyStoppingCallback(
            patience=patience,
            verbose=verbose,
        ),
        CheckpointCallback(
            save_freq=save_freq,
            save_path=save_dir,
            verbose=verbose,
        ),
        ManifestCallback(
            manifest_path=manifest_path,
            checkpoint_freq=save_freq,
            stage_name=stage_name,
            seed=seed,
            verbose=verbose,
        ),
    ]

    return callbacks
