"""Thesis-grade CSV logging callback for Stable-Baselines3.

Tracks per-step and per-episode metrics during PPO training and dumps
them to ``training_curve.csv`` at the end of training.  No TensorBoard
— raw CSV for direct LaTeX/pgfplots ingestion.

Also tracks per-action DeltaHard and DeltaSoft for the **Heuristic Efficacy
Matrix** — printed at the end of training as empirical proof that each
of the Elite 8 heuristics contributes meaningfully during the MDP.

Columns in ``training_curve.csv``::

    timestep, episode, episode_reward, episode_length,
    action_0_count, action_1_count, ..., action_7_count

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

        # -- Per-action efficacy tracking (Heuristic Efficacy Matrix) ----
        self._action_count: dict[int, int] = defaultdict(int)
        self._action_delta_hard_sum: dict[int, float] = defaultdict(float)
        self._action_delta_soft_sum: dict[int, float] = defaultdict(float)

        # -- SB3 internal training metrics (loss, entropy, etc.) --------
        self._sb3_metrics: list[dict[str, Any]] = []
        self._last_sb3_scrape_ts: int = -1

    def _on_training_start(self) -> None:
        """Reset accumulators at training start."""
        self._ep_reward = 0.0
        self._ep_length = 0
        self._ep_actions = defaultdict(int)
        self._episodes = []
        self._step_log = []
        self._episode_count = 0
        # Reset efficacy trackers
        self._action_count = defaultdict(int)
        self._action_delta_hard_sum = defaultdict(float)
        self._action_delta_soft_sum = defaultdict(float)
        # Reset SB3 metric trackers
        self._sb3_metrics = []
        self._last_sb3_scrape_ts = -1

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

        # -- Per-action efficacy accumulation ---------------------------
        if act_val >= 0:
            self._action_count[act_val] += 1
            delta_hard = info_dict.get("delta_hard", 0.0)
            delta_soft = info_dict.get("delta_soft", 0.0)
            if delta_hard is not None and not np.isnan(delta_hard):
                self._action_delta_hard_sum[act_val] += float(delta_hard)
            if delta_soft is not None and not np.isnan(delta_soft):
                self._action_delta_soft_sum[act_val] += float(delta_soft)

        # -- Scrape SB3 internal training metrics (once per rollout) ----
        if (
            self.model is not None
            and hasattr(self.model, "logger")
            and self.num_timesteps != self._last_sb3_scrape_ts
        ):
            sb3_log = getattr(self.model.logger, "name_to_value", {})
            if sb3_log:
                metrics_row: dict[str, Any] = {"timestep": self.num_timesteps}
                _KEYS = [
                    "train/policy_gradient_loss",
                    "train/value_loss",
                    "train/entropy_loss",
                    "train/approx_kl",
                    "train/clip_fraction",
                    "train/explained_variance",
                    "train/loss",
                    "train/learning_rate",
                ]
                for k in _KEYS:
                    if k in sb3_log:
                        metrics_row[k.replace("train/", "")] = float(sb3_log[k])
                if len(metrics_row) > 1:  # has at least one real metric
                    self._sb3_metrics.append(metrics_row)
                    self._last_sb3_scrape_ts = self.num_timesteps

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
                "delta_hard": info_dict.get("delta_hard", 0.0),
                "delta_soft": info_dict.get("delta_soft", 0.0),
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

        # -- Write sb3_training_metrics.csv (loss, entropy, etc.) -------
        metrics_csv = self.run_dir / "sb3_training_metrics.csv"
        if self._sb3_metrics:
            all_keys: list[str] = []
            seen: set[str] = set()
            for row in self._sb3_metrics:
                for k in row:
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)
            with open(metrics_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self._sb3_metrics)
            logger.info(
                "Saved SB3 training metrics: %s (%d rows)",
                metrics_csv,
                len(self._sb3_metrics),
            )
        else:
            logger.warning("No SB3 training metrics captured.")

        # -- Print Heuristic Efficacy Matrix ----------------------------
        self._print_efficacy_matrix()

    # ------------------------------------------------------------------
    # Heuristic Efficacy Matrix
    # ------------------------------------------------------------------

    def _print_efficacy_matrix(self) -> None:
        """Print a formatted table of per-action statistics to stdout.

        Columns:
            Action ID | Action Name | Count | Avg DeltaHard | Avg DeltaSoft
        """
        from src.rl.actions.vectorized_ops import ACTION_NAMES

        total_steps = sum(self._action_count.values())
        if total_steps == 0:
            print("\n[Efficacy Matrix] No actions recorded.\n")
            return

        header = (
            f"{'ID':>3}  {'Action Name':<32}  {'Count':>7}  "
            f"{'%Share':>7}  {'Avg DHard':>10}  {'Avg DSoft':>10}"
        )
        sep = "-" * len(header)

        lines: list[str] = [
            "",
            "=" * len(header),
            "  HEURISTIC EFFICACY MATRIX  (Training Phase)",
            "=" * len(header),
            header,
            sep,
        ]

        for aid in range(NUM_ACTIONS):
            cnt = self._action_count.get(aid, 0)
            name = ACTION_NAMES.get(aid, f"action_{aid}")
            pct = 100.0 * cnt / total_steps if total_steps > 0 else 0.0
            avg_dh = self._action_delta_hard_sum.get(aid, 0.0) / cnt if cnt > 0 else 0.0
            avg_ds = self._action_delta_soft_sum.get(aid, 0.0) / cnt if cnt > 0 else 0.0
            lines.append(
                f"{aid:>3}  {name:<32}  {cnt:>7}  {pct:>6.1f}%  {avg_dh:>+10.2f}  {avg_ds:>+10.2f}"
            )

        lines.append(sep)
        lines.append(f"{'':>3}  {'TOTAL':<32}  {total_steps:>7}  {'100.0':>6}%")
        lines.append("=" * len(header))
        lines.append("")

        matrix_text = "\n".join(lines)
        print(matrix_text)
        logger.info(matrix_text)

        # Also save to file for archival
        matrix_path = self.run_dir / "heuristic_efficacy_matrix.txt"
        with open(matrix_path, "w") as f:
            f.write(matrix_text)
        logger.info("Saved efficacy matrix: %s", matrix_path)
