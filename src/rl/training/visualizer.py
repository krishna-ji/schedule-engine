"""
RL training visualization and plotting.

Automatically generates publication-quality figures after each training run:
- Training curves (rewards, losses, episode lengths)
- Learning metrics (learning rate, entropy, explained variance)
- Statistical summaries and distributions

Output: Organized in respective experiment output folders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

from src.utils.logging_config import get_logger

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)

# Publication-quality settings
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 13

COLORS = sns.color_palette("husl", 8)


def load_tensorboard_data(logdir: Path) -> dict[str, dict[str, list]]:
    """Load TensorBoard event files and extract scalar data."""

    event_files = list(logdir.rglob("events.out.tfevents.*"))
    if not event_files:
        logger.warning(f"No TensorBoard event files found in {logdir}")
        return {}

    logger.info(f"Loading {len(event_files)} TensorBoard event file(s)...")

    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()

    data: dict[str, dict[str, list]] = {}

    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
            "wall_time": [e.wall_time for e in events],
        }

    logger.info(f"Loaded {len(data)} metric(s)")
    return data


def plot_training_curves(data: dict[str, dict], output_dir: Path) -> None:
    """Generate training curves plot (4-panel grid)."""

    if not data:
        logger.warning("No data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("RL Training Progress", fontweight="bold")

    # Panel 1: Episode reward
    if "rollout/ep_rew_mean" in data:
        ax = axes[0, 0]
        steps = data["rollout/ep_rew_mean"]["steps"]
        values = data["rollout/ep_rew_mean"]["values"]
        ax.plot(steps, values, color=COLORS[0], linewidth=1.5, label="Episode Reward")

        # Trend line
        if len(steps) > 3:
            z = np.polyfit(steps, values, min(3, len(steps) - 1))
            p = np.polyval(z, steps)
            ax.plot(
                steps, p, "--", color=COLORS[1], linewidth=1, alpha=0.7, label="Trend"
            )

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title("Episode Reward Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Panel 2: Episode length
    if "rollout/ep_len_mean" in data:
        ax = axes[0, 1]
        steps = data["rollout/ep_len_mean"]["steps"]
        values = data["rollout/ep_len_mean"]["values"]
        ax.plot(steps, values, color=COLORS[2], linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Episode Length")
        ax.set_title("Episode Length Over Time")
        ax.grid(True, alpha=0.3)

    # Panel 3: Policy loss
    if "train/policy_gradient_loss" in data:
        ax = axes[1, 0]
        steps = data["train/policy_gradient_loss"]["steps"]
        values = data["train/policy_gradient_loss"]["values"]
        ax.plot(steps, values, color=COLORS[3], linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Policy Loss")
        ax.set_title("Policy Network Loss")
        ax.grid(True, alpha=0.3)

    # Panel 4: Value loss
    if "train/value_loss" in data:
        ax = axes[1, 1]
        steps = data["train/value_loss"]["steps"]
        values = data["train/value_loss"]["values"]
        ax.plot(steps, values, color=COLORS[4], linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Value Loss")
        ax.set_title("Value Network Loss")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    pdf_path = output_dir / "rl_training_curves.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {pdf_path.name}")

    logger.info(f"Saved: {pdf_path.name}")


def plot_learning_metrics(data: dict[str, dict], output_dir: Path) -> None:
    """Generate learning metrics plot (4-panel grid)."""

    if not data:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("RL Learning Metrics", fontweight="bold")

    # Learning rate
    if "train/learning_rate" in data:
        ax = axes[0, 0]
        steps = data["train/learning_rate"]["steps"]
        values = data["train/learning_rate"]["values"]
        ax.plot(steps, values, color=COLORS[5], linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # Entropy
    if "train/entropy_loss" in data:
        ax = axes[0, 1]
        steps = data["train/entropy_loss"]["steps"]
        values = data["train/entropy_loss"]["values"]
        ax.plot(steps, values, color=COLORS[6], linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Entropy Loss")
        ax.set_title("Policy Entropy Over Time")
        ax.grid(True, alpha=0.3)

    # Explained variance
    if "train/explained_variance" in data:
        ax = axes[1, 0]
        steps = data["train/explained_variance"]["steps"]
        values = data["train/explained_variance"]["values"]
        ax.plot(steps, values, color=COLORS[7], linewidth=1.5)
        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Explained Variance")
        ax.set_title("Value Function Quality")
        ax.grid(True, alpha=0.3)

    # Clip fraction
    if "train/clip_fraction" in data:
        ax = axes[1, 1]
        steps = data["train/clip_fraction"]["steps"]
        values = data["train/clip_fraction"]["values"]
        ax.plot(steps, values, color=COLORS[0], linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Clip Fraction")
        ax.set_title("PPO Clipping Rate")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    pdf_path = output_dir / "rl_learning_metrics.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {pdf_path.name}")


def plot_summary_dashboard(data: dict[str, dict], output_dir: Path) -> None:
    """Generate comprehensive summary dashboard."""

    if not data:
        return

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main reward plot
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    if "rollout/ep_rew_mean" in data:
        steps = data["rollout/ep_rew_mean"]["steps"]
        values = data["rollout/ep_rew_mean"]["values"]
        ax_main.plot(
            steps,
            values,
            color=COLORS[0],
            linewidth=2,
            label="Episode Reward",
            alpha=0.6,
        )

        # Smoothed line
        if len(values) > 10:
            window = max(1, len(values) // 20)
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
            smooth_steps = steps[: len(smoothed)]
            ax_main.plot(
                smooth_steps, smoothed, color=COLORS[1], linewidth=2.5, label="Smoothed"
            )

        ax_main.set_xlabel("Training Step", fontsize=12)
        ax_main.set_ylabel("Mean Episode Reward", fontsize=12)
        ax_main.set_title(
            "RL Training Progress (Main View)", fontsize=13, fontweight="bold"
        )
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

    # Statistics table
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.axis("off")

    if "rollout/ep_rew_mean" in data:
        values = data["rollout/ep_rew_mean"]["values"]
        stats_text = f"""Training Statistics

Total Steps: {len(values):,}

Reward:
  Mean: {np.mean(values):.3f}
  Std: {np.std(values):.3f}
  Min: {np.min(values):.3f}
  Max: {np.max(values):.3f}

Improvement:
  Start: {values[0]:.3f}
  End: {values[-1]:.3f}
  Delta: {values[-1] - values[0]:.3f}
"""
        ax_stats.text(
            0.1,
            0.9,
            stats_text,
            transform=ax_stats.transAxes,
            fontsize=9,
            verticalalignment="top",
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.3},
        )

    # Bottom row: Loss plots
    ax_policy = fig.add_subplot(gs[2, 0])
    if "train/policy_gradient_loss" in data:
        steps = data["train/policy_gradient_loss"]["steps"]
        values = data["train/policy_gradient_loss"]["values"]
        ax_policy.plot(steps, values, color=COLORS[3], linewidth=1.5)
        ax_policy.set_xlabel("Step")
        ax_policy.set_ylabel("Policy Loss")
        ax_policy.set_title("Policy Loss")
        ax_policy.grid(True, alpha=0.3)

    ax_value = fig.add_subplot(gs[2, 1])
    if "train/value_loss" in data:
        steps = data["train/value_loss"]["steps"]
        values = data["train/value_loss"]["values"]
        ax_value.plot(steps, values, color=COLORS[4], linewidth=1.5)
        ax_value.set_xlabel("Step")
        ax_value.set_ylabel("Value Loss")
        ax_value.set_title("Value Loss")
        ax_value.grid(True, alpha=0.3)

    ax_entropy = fig.add_subplot(gs[2, 2])
    if "train/entropy_loss" in data:
        steps = data["train/entropy_loss"]["steps"]
        values = data["train/entropy_loss"]["values"]
        ax_entropy.plot(steps, values, color=COLORS[6], linewidth=1.5)
        ax_entropy.set_xlabel("Step")
        ax_entropy.set_ylabel("Entropy")
        ax_entropy.set_title("Policy Entropy")
        ax_entropy.grid(True, alpha=0.3)

    # Episode length mini plot
    ax_eplen = fig.add_subplot(gs[1, 2])
    if "rollout/ep_len_mean" in data:
        steps = data["rollout/ep_len_mean"]["steps"]
        values = data["rollout/ep_len_mean"]["values"]
        ax_eplen.plot(steps, values, color=COLORS[2], linewidth=1.5)
        ax_eplen.set_xlabel("Step")
        ax_eplen.set_ylabel("Episode Length")
        ax_eplen.set_title("Episode Length")
        ax_eplen.grid(True, alpha=0.3)

    plt.suptitle(
        "RL Training Summary Dashboard", fontsize=15, fontweight="bold", y=0.995
    )

    pdf_path = output_dir / "rl_summary_dashboard.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved: {pdf_path.name}")


def export_individual_charts(data: dict[str, dict], output_dir: Path) -> None:
    """Export individual charts as separate PDF/PNG files for thesis."""

    individual_dir = output_dir / "individual_charts"
    individual_dir.mkdir(exist_ok=True)

    chart_count = 0

    # Episode reward
    if "rollout/ep_rew_mean" in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = data["rollout/ep_rew_mean"]["steps"]
        values = data["rollout/ep_rew_mean"]["values"]
        ax.plot(steps, values, linewidth=2, color=COLORS[0])
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Mean Episode Reward", fontsize=11)
        ax.set_title("Episode Reward Progression", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(individual_dir / "episode_reward.pdf", bbox_inches="tight")
        plt.savefig(individual_dir / "episode_reward.png", dpi=300, bbox_inches="tight")
        plt.close()
        chart_count += 1

    # Episode length
    if "rollout/ep_len_mean" in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = data["rollout/ep_len_mean"]["steps"]
        values = data["rollout/ep_len_mean"]["values"]
        ax.plot(steps, values, linewidth=2, color=COLORS[2])
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Mean Episode Length", fontsize=11)
        ax.set_title("Episode Length Over Time", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(individual_dir / "episode_length.pdf", bbox_inches="tight")
        plt.savefig(individual_dir / "episode_length.png", dpi=300, bbox_inches="tight")
        plt.close()
        chart_count += 1

    # Policy loss
    if "train/policy_gradient_loss" in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = data["train/policy_gradient_loss"]["steps"]
        values = data["train/policy_gradient_loss"]["values"]
        ax.plot(steps, values, linewidth=2, color=COLORS[3])
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Policy Gradient Loss", fontsize=11)
        ax.set_title("Policy Network Loss", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(individual_dir / "policy_loss.pdf", bbox_inches="tight")
        plt.savefig(individual_dir / "policy_loss.png", dpi=300, bbox_inches="tight")
        plt.close()
        chart_count += 1

    # Value loss
    if "train/value_loss" in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = data["train/value_loss"]["steps"]
        values = data["train/value_loss"]["values"]
        ax.plot(steps, values, linewidth=2, color=COLORS[4])
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Value Function Loss", fontsize=11)
        ax.set_title("Value Network Loss", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(individual_dir / "value_loss.pdf", bbox_inches="tight")
        plt.savefig(individual_dir / "value_loss.png", dpi=300, bbox_inches="tight")
        plt.close()
        chart_count += 1

    # Learning rate
    if "train/learning_rate" in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = data["train/learning_rate"]["steps"]
        values = data["train/learning_rate"]["values"]
        ax.plot(steps, values, linewidth=2, color=COLORS[5])
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Learning Rate", fontsize=11)
        ax.set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(individual_dir / "learning_rate.pdf", bbox_inches="tight")
        plt.savefig(individual_dir / "learning_rate.png", dpi=300, bbox_inches="tight")
        plt.close()
        chart_count += 1

    # Entropy
    if "train/entropy_loss" in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = data["train/entropy_loss"]["steps"]
        values = data["train/entropy_loss"]["values"]
        ax.plot(steps, values, linewidth=2, color=COLORS[6])
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Policy Entropy", fontsize=11)
        ax.set_title("Policy Entropy Over Time", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(individual_dir / "entropy.pdf", bbox_inches="tight")
        plt.savefig(individual_dir / "entropy.png", dpi=300, bbox_inches="tight")
        plt.close()
        chart_count += 1

    # Explained variance
    if "train/explained_variance" in data:
        fig, ax = plt.subplots(figsize=(8, 5))
        steps = data["train/explained_variance"]["steps"]
        values = data["train/explained_variance"]["values"]
        ax.plot(steps, values, linewidth=2, color=COLORS[7])
        ax.axhline(y=0, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel("Explained Variance", fontsize=11)
        ax.set_title("Value Function Quality", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(individual_dir / "explained_variance.pdf", bbox_inches="tight")
        plt.savefig(
            individual_dir / "explained_variance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        chart_count += 1

    logger.info(f"Exported {chart_count} individual chart(s) to {individual_dir.name}/")


def export_csv_data(data: dict[str, dict], output_dir: Path) -> None:
    """Export raw metrics as CSV files."""

    csv_dir = output_dir / "rl_metrics_csv"
    csv_dir.mkdir(exist_ok=True)

    for tag, values in data.items():
        if not isinstance(values, dict) or "steps" not in values:
            continue

        filename = tag.replace("/", "_").replace("\\", "_") + ".csv"
        csv_path = csv_dir / filename

        with csv_path.open("w") as f:
            f.write("step,value,wall_time\n")
            for step, value, wall_time in zip(
                values["steps"], values["values"], values["wall_time"], strict=False
            ):
                f.write(f"{step},{value},{wall_time}\n")

    logger.info(f"Exported {len(data)} CSV file(s) to {csv_dir.name}/")


def generate_visualizations(
    tensorboard_logdir: Path,
    output_dir: Path,
    experiment_name: str | None = None,
) -> None:
    """
    Generate all RL training visualizations.

    Args:
        tensorboard_logdir: Path to TensorBoard logs directory
        output_dir: Output directory for plots (will create rl_analysis/ subfolder)
        experiment_name: Optional experiment name for plot titles
    """

    logger.info("[START] RL training visualization")

    # Create visualization output directory
    viz_dir = output_dir / "rl_analysis"
    viz_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        data = load_tensorboard_data(tensorboard_logdir)

        if not data:
            logger.warning("No training data found - skipping visualization")
            return

        # Generate plots
        logger.info("Generating plots...")
        plot_training_curves(data, viz_dir)
        plot_learning_metrics(data, viz_dir)
        plot_summary_dashboard(data, viz_dir)
        export_individual_charts(data, viz_dir)  # NEW: Individual charts for thesis
        export_csv_data(data, viz_dir)

        logger.info("[OK] Visualizations saved to: %s", viz_dir)

    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
