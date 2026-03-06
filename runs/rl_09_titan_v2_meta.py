#!/usr/bin/env python3
r"""RL 09 — The "Titan V2" SOTA Run: MaskablePPO + Meta-Heuristic Action Space.

Phase 52 definitive training run combining:
  - **Single-environment** MaskablePPO (sb3-contrib)
  - **State-conditioned action masking** (soft optimizers blocked when
    hard constraints are violated)
  - **Meta-Heuristic Overhaul** (Phase 51):
      * Action 2: LNS Ruin & Recreate — destroys top 5% worst events,
        greedy best-fit reinsertion by domain restrictiveness
      * Action 5: Kempe Chain Interchange — bipartite time-slot sub-graph
        swaps via conflict-density-weighted cascade tracing

Full Discrete(8) Action Space:
  0: SpatialResourceProjection     — conflict-directed k=5 room sniper
  1: FacultyTemporalProjection     — instructor clash repair
  2: LargeNeighborhoodSearch       — LNS Ruin & Recreate (META-HEURISTIC)
  3: SymmetricSubcohortSync        — SSCP paired-practical sync (soft)
  4: UniversalFeasibilityProjection — bounded depth-3 ejection chains
  5: KempeChainInterchange         — Kempe Chain (META-HEURISTIC)
  6: StochasticSpatialPerturbation — room exploration perturbation
  7: MeridianCompactionHeuristic   — feasibility-gated soft optimizer

Configuration:
  pop_size = 120, max_generations = 50, timesteps = 100,000

Usage::

    python runs/rl_09_titan_v2_meta.py

Outputs::

    output/models/maskable_ppo_titan_v2_meta.zip — final trained model
    output/titan_v2/<timestamp>/titan_v2_training_log.csv
    output/titan_v2/<timestamp>/titan_v2_step_log.csv
    output/titan_v2/<timestamp>/titan_v2_report.txt
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rl_09_titan_v2")

# ======================================================================
# TITAN V2 CONFIGURATION
# ======================================================================

SEED = 42
POP_SIZE = 120
MAX_GENERATIONS = 50
TOTAL_TIMESTEPS = 100_000
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
NET_ARCH = [64, 64]
N_STEPS = 2048  # PPO rollout buffer
BATCH_SIZE = 64
N_EPOCHS = 10
GAE_LAMBDA = 0.95
GAMMA = 0.99
ENT_COEF = 0.01  # Encourage exploration
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
PKL_PATH = ".cache/events_with_domains.pkl"
ACCEPTANCE_TOLERANCE = 5.0  # Allow some degradation for exploration


# ======================================================================
# Titan V2 Training
# ======================================================================


def train_titan_v2():
    """Execute the Titan V2 training run: MaskablePPO + Meta-Heuristic Elite 8."""

    # -- Import MaskablePPO ------------------------------------------------
    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.utils import set_random_seed
    except ImportError:
        logger.error("sb3-contrib not found!  Install: pip install sb3-contrib")
        sys.exit(1)

    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    # Set global seed
    set_random_seed(SEED)

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / "titan_v2" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  RL 09 — THE TITAN V2 SOTA RUN")
    logger.info("  MaskablePPO + Meta-Heuristic Action Space (LNS + Kempe)")
    logger.info("=" * 70)
    logger.info("  run_dir           : %s", run_dir)
    logger.info("  pop_size          : %d", POP_SIZE)
    logger.info("  max_generations   : %d", MAX_GENERATIONS)
    logger.info("  total_timesteps   : %d", TOTAL_TIMESTEPS)
    logger.info("  n_steps (rollout) : %d", N_STEPS)
    logger.info("  batch_size        : %d", BATCH_SIZE)
    logger.info("  net_arch          : %s", NET_ARCH)
    logger.info("  learning_rate     : %.1e", LEARNING_RATE)
    logger.info("  acceptance_tol    : %.1f", ACCEPTANCE_TOLERANCE)
    logger.info("=" * 70)

    # -- Create SINGLE Environment -----------------------------------------
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
        acceptance_tolerance=ACCEPTANCE_TOLERANCE,
    )

    assert hasattr(env, "action_masks"), "Environment must support action_masks()"
    logger.info("Environment created with action masking support")

    # Verify Meta-Heuristic action space is active
    from src.rl.actions.vectorized_ops import ACTION_NAMES, VECTORIZED_ACTION_SPACE

    logger.info("Action space (%d actions):", len(VECTORIZED_ACTION_SPACE))
    for aid, name in ACTION_NAMES.items():
        logger.info("  %d: %s", aid, name)

    # Verify LNS and Kempe are present
    assert (
        ACTION_NAMES[2] == "large_neighborhood_search"
    ), f"Action 2 must be LNS, got: {ACTION_NAMES[2]}"
    assert (
        ACTION_NAMES[5] == "kempe_chain_interchange"
    ), f"Action 5 must be Kempe Chain, got: {ACTION_NAMES[5]}"
    logger.info("Meta-Heuristic overhaul verified: LNS (2) + Kempe (5)")

    # -- Fresh MaskablePPO Agent -------------------------------------------
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        policy_kwargs=dict(net_arch=NET_ARCH),
        seed=SEED,
        verbose=1,
        device="cpu",
    )

    logger.info("MaskablePPO initialized (FRESH — no V1 weights loaded)")

    # -- Titan V2 Callback -------------------------------------------------
    class TitanV2Callback(BaseCallback):
        """Comprehensive logging callback for Titan V2."""

        def __init__(self, run_dir: Path):
            super().__init__(verbose=0)
            self.run_dir = run_dir
            self.episode_count = 0
            self.step_count = 0
            self.best_hard_ever = np.inf
            self.best_soft_ever = np.inf
            self.mask_blocked_total = 0
            self.mask_total = 0
            self.episode_rewards = []
            self.action_counts = np.zeros(len(ACTION_NAMES), dtype=np.int64)
            self.t_start = time.perf_counter()

            # Episode CSV
            self.ep_csv = run_dir / "titan_v2_training_log.csv"
            with open(self.ep_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "episode",
                        "timestep",
                        "ep_reward",
                        "ep_length",
                        "best_hard",
                        "best_soft",
                        "feasible_frac",
                        "mask_blocked_pct",
                        "wall_time_s",
                        "best_hard_ever",
                        "best_soft_ever",
                    ]
                )

            # Step CSV
            self.step_csv = run_dir / "titan_v2_step_log.csv"
            with open(self.step_csv, "w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "timestep",
                        "action",
                        "action_name",
                        "reward",
                        "best_hard",
                        "best_soft",
                        "rejected",
                        "delta_hard",
                        "mask_3_blocked",
                        "mask_7_blocked",
                    ]
                )

        def _on_step(self) -> bool:
            self.step_count += 1

            # Track action usage
            actions = self.locals.get("actions")
            if actions is not None:
                a = int(actions[0]) if hasattr(actions, "__len__") else int(actions)
                if 0 <= a < len(self.action_counts):
                    self.action_counts[a] += 1

            # Track mask usage
            try:
                env_inner = self.training_env.envs[0]
                masks = env_inner.action_masks()
                self.mask_total += 1
                if not masks[3] or not masks[7]:
                    self.mask_blocked_total += 1
            except Exception:
                pass

            # Detect episode end
            infos = self.locals.get("infos", [{}])
            if infos and "episode" in infos[0]:
                info = infos[0]
                ep = info["episode"]
                self.episode_count += 1
                self.episode_rewards.append(ep["r"])

                best_h = info.get("best_hard", np.inf)
                best_s = info.get("best_soft", np.inf)
                self.best_hard_ever = min(self.best_hard_ever, best_h)
                self.best_soft_ever = min(self.best_soft_ever, best_s)

                mask_pct = (self.mask_blocked_total / max(self.mask_total, 1)) * 100
                wall_t = time.perf_counter() - self.t_start

                with open(self.ep_csv, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            self.episode_count,
                            self.num_timesteps,
                            f"{ep['r']:.6f}",
                            ep["l"],
                            best_h,
                            best_s,
                            info.get("feasible_frac", 0.0),
                            f"{mask_pct:.1f}",
                            f"{wall_t:.1f}",
                            self.best_hard_ever,
                            self.best_soft_ever,
                        ]
                    )

                if self.episode_count % 25 == 0:
                    fps = self.num_timesteps / wall_t if wall_t > 0 else 0
                    logger.info(
                        "EP %4d | ts=%6d | R=%+.3f | hard=%7.1f | "
                        "best_ever=%7.1f | soft=%7.1f | mask%%=%.0f | "
                        "FPS=%.1f | wall=%.0fs",
                        self.episode_count,
                        self.num_timesteps,
                        ep["r"],
                        best_h,
                        self.best_hard_ever,
                        best_s,
                        mask_pct,
                        fps,
                        wall_t,
                    )

            return True

    callback = TitanV2Callback(run_dir)

    # -- FPS Estimate ------------------------------------------------------
    logger.info("Running FPS benchmark (5 steps)...")
    obs, info = env.reset()
    t_bench = time.perf_counter()
    bench_steps = 5
    for _ in range(bench_steps):
        masks = env.action_masks()
        valid = np.where(masks)[0]
        obs, reward, term, trunc, info = env.step(np.random.choice(valid))
        if term or trunc:
            obs, info = env.reset()
    fps_estimate = bench_steps / (time.perf_counter() - t_bench)
    env.close()

    eta_seconds = TOTAL_TIMESTEPS / fps_estimate
    eta_minutes = eta_seconds / 60
    eta_hours = eta_minutes / 60
    logger.info(
        "FPS estimate: %.1f steps/s | ETA: %.0f seconds (%.1f min / %.2f hrs)",
        fps_estimate,
        eta_seconds,
        eta_minutes,
        eta_hours,
    )

    # Re-create environment (reset after benchmark)
    env = PymooHyperHeuristicEnv(
        pkl_path=PKL_PATH,
        max_generations=MAX_GENERATIONS,
        pop_size=POP_SIZE,
        algorithm_name="nsga2",
        seed=SEED,
        acceptance_tolerance=ACCEPTANCE_TOLERANCE,
    )
    model.set_env(env)

    # -- TRAIN -------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("  STARTING TITAN V2 TRAINING: %d timesteps", TOTAL_TIMESTEPS)
    logger.info("  Estimated completion: %.1f min (%.2f hrs)", eta_minutes, eta_hours)
    logger.info("  Meta-Heuristics: LNS (action 2) + Kempe Chain (action 5)")
    logger.info("=" * 70)

    t0 = time.perf_counter()

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callback,
        log_interval=10,
        progress_bar=True,
    )

    train_time = time.perf_counter() - t0
    actual_fps = TOTAL_TIMESTEPS / train_time

    logger.info("=" * 70)
    logger.info("  TITAN V2 TRAINING COMPLETE")
    logger.info("  Wall time          : %.1fs (%.1f min)", train_time, train_time / 60)
    logger.info("  Actual FPS         : %.1f", actual_fps)
    logger.info("  Episodes           : %d", callback.episode_count)
    logger.info("  Best Hard (ever)   : %.1f", callback.best_hard_ever)
    logger.info("  Best Soft (ever)   : %.1f", callback.best_soft_ever)
    logger.info("=" * 70)

    # -- Save Model --------------------------------------------------------
    # Save to run directory
    model_run = run_dir / "maskable_ppo_titan_v2_meta.zip"
    model.save(str(model_run))
    logger.info("Model saved (run): %s", model_run)

    # Save to canonical path
    canonical_dir = PROJECT_ROOT / "output" / "models"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = canonical_dir / "maskable_ppo_titan_v2_meta.zip"
    model.save(str(canonical_path))
    logger.info("Model saved (canonical): %s", canonical_path)

    # -- Action Usage Report -----------------------------------------------
    total_actions = callback.action_counts.sum()
    action_usage_lines = []
    for aid, name in ACTION_NAMES.items():
        cnt = callback.action_counts[aid]
        pct = (cnt / max(total_actions, 1)) * 100
        meta = " (META)" if aid in (2, 5) else ""
        line = f"  {aid}: {name:<35s}  {cnt:>6d} ({pct:>5.1f}%){meta}"
        action_usage_lines.append(line)
        logger.info(line.strip())

    # -- Generate Report ---------------------------------------------------
    ep_rewards = callback.episode_rewards
    report_path = run_dir / "titan_v2_report.txt"
    with open(report_path, "w") as f:
        f.write("TITAN V2 SOTA RUN — TRAINING REPORT\n")
        f.write("=" * 60 + "\n")
        f.write("Meta-Heuristic Action Space (Phase 51 Overhaul)\n\n")
        f.write("CONFIGURATION\n")
        f.write(f"  Population Size    : {POP_SIZE}\n")
        f.write(f"  Max Generations    : {MAX_GENERATIONS}\n")
        f.write(f"  Total Timesteps    : {TOTAL_TIMESTEPS:,}\n")
        f.write(f"  Learning Rate      : {LEARNING_RATE}\n")
        f.write(f"  Network Arch       : {NET_ARCH}\n")
        f.write(f"  N_Steps (rollout)  : {N_STEPS}\n")
        f.write(f"  Batch Size         : {BATCH_SIZE}\n")
        f.write(f"  Acceptance Tol     : {ACCEPTANCE_TOLERANCE}\n")
        f.write(f"  Seed               : {SEED}\n\n")

        f.write("RESULTS\n")
        f.write(f"  Training Time      : {train_time:.1f}s ({train_time/60:.1f} min)\n")
        f.write(f"  Actual FPS         : {actual_fps:.1f}\n")
        f.write(f"  Episodes Completed : {callback.episode_count}\n")
        f.write(f"  Steps Completed    : {callback.step_count}\n")
        f.write(f"  Best Hard (ever)   : {callback.best_hard_ever:.1f}\n")
        f.write(f"  Best Soft (ever)   : {callback.best_soft_ever:.1f}\n\n")

        f.write("REWARD STATISTICS\n")
        if ep_rewards:
            f.write(f"  Mean Episode R     : {np.mean(ep_rewards):.4f}\n")
            f.write(f"  Std Episode R      : {np.std(ep_rewards):.4f}\n")
            f.write(f"  Min Episode R      : {np.min(ep_rewards):.4f}\n")
            f.write(f"  Max Episode R      : {np.max(ep_rewards):.4f}\n\n")

        f.write("ACTION MASKING\n")
        mask_pct = (callback.mask_blocked_total / max(callback.mask_total, 1)) * 100
        f.write(f"  Total Steps Tracked: {callback.mask_total}\n")
        f.write(f"  Steps w/ Masking   : {callback.mask_blocked_total}\n")
        f.write(f"  Mask Rate          : {mask_pct:.1f}%\n\n")

        f.write("ACTION USAGE (learned policy preferences)\n")
        for line in action_usage_lines:
            f.write(line + "\n")
        f.write("\n")

        f.write("META-HEURISTIC ACTION SPACE (Phase 51)\n")
        for aid, name in ACTION_NAMES.items():
            meta = " ★ META-HEURISTIC" if aid in (2, 5) else ""
            f.write(f"  {aid}: {name}{meta}\n")
        f.write("\n")
        f.write("KEY META-HEURISTIC OPERATORS\n")
        f.write(
            "  2: LargeNeighborhoodSearch  — Ruin top 5%, greedy best-fit recreate\n"
        )
        f.write("  5: KempeChainInterchange    — Bipartite time-slot sub-graph swaps\n")

    logger.info("Report saved: %s", report_path)

    env.close()
    return model, run_dir, callback


# ======================================================================
# Main
# ======================================================================


def main():
    try:
        model, run_dir, callback = train_titan_v2()

        print("\n" + "=" * 70)
        print("  THE TITAN V2 RUN IS COMPLETE")
        print("  Model:      output/models/maskable_ppo_titan_v2_meta.zip")
        print(f"  Logs:       {run_dir}")
        print(f"  Best Hard:  {callback.best_hard_ever:.1f}")
        print(f"  Best Soft:  {callback.best_soft_ever:.1f}")
        print(f"  Episodes:   {callback.episode_count}")
        print("=" * 70)

    except KeyboardInterrupt:
        logger.warning("Titan V2 run interrupted by user")
    except Exception as e:
        logger.error("Titan V2 run failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
