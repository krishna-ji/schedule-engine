"""Thesis-grade CSV logging callback for Stable-Baselines3.

Tracks per-step and per-episode metrics during PPO training and dumps
them to ``training_curve.csv`` at the end of training.  No TensorBoard
— raw CSV for direct LaTeX/pgfplots ingestion.

Columns in ``training_curve.csv``::

    timestep, episode, episode_reward, episode_length,
    action_0_count, action_1_count, ..., action_5_count

Usage::

    from src.rl.training.thesis_callback import ThesisLoggingCallback

    cb = ThesisLoggingCallback(run_dir="output/rl_vectorized/20260225_120000")
    model.learn(total_timesteps=2000, callback=cb)
"""

from __future__ import annotations

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.rl.actions.vectorized_ops import NUM_ACTIONS

logger = logging.getLogger(__name__)


class ThesisLoggingCallback(BaseCallback):
    """SB3 callback that logs episode-level metrics to CSV.

    Parameters
    ----------
    run_dir : str | Path
        Directory where ``training_curve.csv`` will be written.
    verbose : int
        0 = silent, 1 = episode summaries, 2 = every step.
    """

    def __init__(self, run_dir: str | Path, verbose: int = 1):
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Per-episode accumulators
        self._ep_reward: float = 0.0
        self._ep_length: int = 0
        self._ep_actions: dict[int, int] = defaultdict(int)

        # Collected rows
        self._episodes: list[dict[str, Any]] = []
        self._episode_count: int = 0

        # Per-step log (for fine-grained analysis)
        self._step_log: list[dict[str, Any]] = []

    def _on_training_start(self) -> None:
        """Reset accumulators at training start."""
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_actions = defaultdict(int)
        self._episodes = []
        self._step_log = []
        self._episode_count = 0

    def _on_step(self) -> bool:
        """Called at every environment step."""
        # Extract action and reward from the rollout buffer
        # SB3 stores them in self.locals
        action = self.locals.get("actions")
        reward = self.locals.get("rewards")
        done = self.locals.get("dones")
        info = self.locals.get("infos")

        # Handle vectorised envs (always arrays in SB3)
        if action is not None:
            act_val = int(action[0]) if hasattr(action, "__len__") else int(action)
        else:
            act_val = -1

        if reward is not None:
            rew_val = float(reward[0]) if hasattr(reward, "__len__") else float(reward)
        else:
            rew_val = 0.0

        if done is not None:
            done_val = bool(done[0]) if hasattr(done, "__len__") else bool(done)
        else:
            done_val = False

        # Extract info dict values
        info_dict = {}
        if info is not None:
            info_item = info[0] if isinstance(info, (list, np.ndarray)) else info
            if isinstance(info_item, dict):
                info_dict = info_item

        # Accumulate
        self._ep_reward += rew_val
        self._ep_length += 1
        self._ep_actions[act_val] += 1

        # Step-level log
        self._step_log.append(
            {
                "timestep": self.num_timesteps,
                "action": act_val,
                "reward": rew_val,
                "best_hard": info_dict.get("best_hard", np.nan),
                "best_soft": info_dict.get("best_soft", np.nan),
                "feasible_frac": info_dict.get("feasible_frac", np.nan),
                "rejected": info_dict.get("rejected", False),
            }
        )

        if self.verbose >= 2:
            logger.info(
                "  ts=%d act=%d rew=%.4f hard=%.1f",
                self.num_timesteps,
                act_val,
                rew_val,
                info_dict.get("best_hard", -1),
            )

        # Episode boundary
        if done_val:
            self._episode_count += 1
            row: dict[str, Any] = {
                "timestep": self.num_timesteps,
                "episode": self._episode_count,
                "episode_reward": round(self._ep_reward, 6),
                "episode_length": self._ep_length,
            }
            # Per-action counts
            for a in range(NUM_ACTIONS):
                row[f"action_{a}_count"] = self._ep_actions.get(a, 0)

            self._episodes.append(row)

            if self.verbose >= 1:
                logger.info(
                    "Episode %d | reward=%.4f | length=%d | ts=%d",
                    self._episode_count,
                    self._ep_reward,
                    self._ep_length,
                    self.num_timesteps,
                )

            # Reset accumulators
            self._ep_reward = 0.0
            self._ep_length = 0
            self._ep_actions = defaultdict(int)

        return True  # continue training

    def _on_training_end(self) -> None:
        """Dump collected data to CSV."""
        # If there's an unfinished episode, flush it
        if self._ep_length > 0:
            self._episode_count += 1
            row: dict[str, Any] = {
                "timestep": self.num_timesteps,
                "episode": self._episode_count,
                "episode_reward": round(self._ep_reward, 6),
                "episode_length": self._ep_length,
            }
            for a in range(NUM_ACTIONS):
                row[f"action_{a}_count"] = self._ep_actions.get(a, 0)
            self._episodes.append(row)

        # -- Write training_curve.csv -----------------------------------
        csv_path = self.run_dir / "training_curve.csv"
        if self._episodes:
            fieldnames = list(self._episodes[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self._episodes)
            logger.info(
                "Saved training curve: %s (%d episodes)", csv_path, len(self._episodes)
            )
        else:
            logger.warning("No episodes completed during training.")

        # -- Write step_log.csv (fine-grained) --------------------------
        step_csv = self.run_dir / "step_log.csv"
        if self._step_log:
            fieldnames_s = list(self._step_log[0].keys())
            with open(step_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames_s)
                writer.writeheader()
                writer.writerows(self._step_log)
            logger.info("Saved step log: %s (%d steps)", step_csv, len(self._step_log))
