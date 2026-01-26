"""
Generate publication-quality RL training visualizations.

Exports all necessary figures for thesis/academic publication:
- Training curves (rewards, losses, episode lengths)
- Heuristic selection patterns
- Learning progress metrics
- Comparative analysis plots

Output: output/rl_training_analysis/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

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

# Color palette
COLORS = sns.color_palette("husl", 8)


def load_tensorboard_data(logdir: Path) -> dict[str, any]:
    """Load TensorBoard event files."""

    print(f"Loading TensorBoard logs from: {logdir}")

    # Find event files
    event_files = list(logdir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {logdir}")

    print(f"Found {len(event_files)} event file(s)")

    # Load data from first event file
    ea = event_accumulator.EventAccumulator(str(event_files[0]))
    ea.Reload()

    data = {}

    # Get available tags
    tags = ea.Tags()
    print(f"\nAvailable scalar tags: {tags.get('scalars', [])}")

    # Load scalar data
    for tag in tags.get("scalars", []):
        events = ea.Scalars(tag)
        data[tag] = {
            "steps": [e.step for e in events],
            "values": [e.value for e in events],
            "wall_time": [e.wall_time for e in events],
        }

    return data


def plot_training_curves(data: dict, output_dir: Path) -> None:
    """Plot training reward curves."""

    print("\n[1/6] Plotting training curves...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("RL Training Progress", fontweight="bold")

    # Reward curves
    if "rollout/ep_rew_mean" in data:
        ax = axes[0, 0]
        steps = data["rollout/ep_rew_mean"]["steps"]
        values = data["rollout/ep_rew_mean"]["values"]
        ax.plot(steps, values, color=COLORS[0], linewidth=1.5, label="Episode Reward")

        # Add trend line
        z = np.polyfit(steps, values, 3)
        p = np.polyval(z, steps)
        ax.plot(steps, p, "--", color=COLORS[1], linewidth=1, alpha=0.7, label="Trend")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Episode Reward")
        ax.set_title("Episode Reward Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Episode length
    if "rollout/ep_len_mean" in data:
        ax = axes[0, 1]
        steps = data["rollout/ep_len_mean"]["steps"]
        values = data["rollout/ep_len_mean"]["values"]
        ax.plot(steps, values, color=COLORS[2], linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Mean Episode Length")
        ax.set_title("Episode Length Over Time")
        ax.grid(True, alpha=0.3)

    # Policy loss
    if "train/policy_loss" in data:
        ax = axes[1, 0]
        steps = data["train/policy_loss"]["steps"]
        values = data["train/policy_loss"]["values"]
        ax.plot(steps, values, color=COLORS[3], linewidth=1.5)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Policy Loss")
        ax.set_title("Policy Network Loss")
        ax.grid(True, alpha=0.3)

    # Value loss
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

    # Save as PDF and PNG
    pdf_path = output_dir / "01_training_curves.pdf"
    png_path = output_dir / "01_training_curves.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {pdf_path.name}")


def plot_learning_metrics(data: dict, output_dir: Path) -> None:
    """Plot learning rate, entropy, and other metrics."""

    print("[2/6] Plotting learning metrics...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("RL Training Metrics", fontweight="bold")

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

    pdf_path = output_dir / "02_learning_metrics.pdf"
    png_path = output_dir / "02_learning_metrics.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {pdf_path.name}")


def plot_reward_distribution(data: dict, output_dir: Path) -> None:
    """Plot reward distribution histogram."""

    print("[3/6] Plotting reward distribution...")

    if "rollout/ep_rew_mean" not in data:
        print("   Skipped (no reward data)")
        return

    values = data["rollout/ep_rew_mean"]["values"]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(values, bins=30, color=COLORS[0], alpha=0.7, edgecolor="black")
    ax.axvline(
        np.mean(values),
        color=COLORS[1],
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(values):.2f}",
    )
    ax.axvline(
        np.median(values),
        color=COLORS[2],
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(values):.2f}",
    )
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    pdf_path = output_dir / "03_reward_distribution.pdf"
    png_path = output_dir / "03_reward_distribution.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {pdf_path.name}")


def plot_training_summary(data: dict, output_dir: Path) -> None:
    """Plot comprehensive training summary."""

    print("[4/6] Plotting training summary...")

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Main reward plot (larger)
    ax_main = fig.add_subplot(gs[0:2, 0:2])
    if "rollout/ep_rew_mean" in data:
        steps = data["rollout/ep_rew_mean"]["steps"]
        values = data["rollout/ep_rew_mean"]["values"]
        ax_main.plot(
            steps, values, color=COLORS[0], linewidth=2, label="Episode Reward"
        )

        # Smoothed line
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
  Δ: {values[-1] - values[0]:.3f}
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

    # Loss plots (bottom row)
    ax_policy = fig.add_subplot(gs[2, 0])
    if "train/policy_loss" in data:
        steps = data["train/policy_loss"]["steps"]
        values = data["train/policy_loss"]["values"]
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

    pdf_path = output_dir / "04_training_summary_dashboard.pdf"
    png_path = output_dir / "04_training_summary_dashboard.png"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {pdf_path.name}")


def export_data_tables(data: dict, output_dir: Path) -> None:
    """Export raw data as CSV for further analysis."""

    print("[5/6] Exporting data tables...")

    for tag, values in data.items():
        if not isinstance(values, dict) or "steps" not in values:
            continue

        # Sanitize filename
        filename = tag.replace("/", "_").replace("\\", "_") + ".csv"
        csv_path = output_dir / "csv" / filename
        csv_path.parent.mkdir(exist_ok=True)

        # Write CSV
        with open(csv_path, "w") as f:
            f.write("step,value,wall_time\n")
            for step, value, wall_time in zip(
                values["steps"], values["values"], values["wall_time"], strict=False
            ):
                f.write(f"{step},{value},{wall_time}\n")

    print(f"  ✓ Exported {len(data)} CSV files to csv/")


def generate_readme(output_dir: Path, data: dict) -> None:
    """Generate README with figure descriptions."""

    print("[6/6] Generating README...")

    readme_content = f"""# RL Training Analysis

Generated: {Path(__file__).name}
Date: {np.datetime64('today')}

## Publication-Quality Figures

### 01_training_curves.pdf
Main training curves showing episode rewards, episode length, policy loss, and value loss over time.
- **Use for**: Overall training progress visualization
- **Format**: 2x2 grid, 300 DPI

### 02_learning_metrics.pdf
Learning rate schedule, policy entropy, explained variance, and PPO clipping rate.
- **Use for**: Detailed learning dynamics analysis
- **Format**: 2x2 grid, 300 DPI

### 03_reward_distribution.pdf
Histogram showing distribution of episode rewards with mean/median markers.
- **Use for**: Statistical analysis of training outcomes
- **Format**: Single plot, 300 DPI

### 04_training_summary_dashboard.pdf
Comprehensive dashboard with main reward plot, statistics, and auxiliary metrics.
- **Use for**: Thesis figure, presentation slide
- **Format**: Complex layout, 300 DPI

## Data Files

### csv/
Raw CSV exports of all TensorBoard metrics for custom analysis.
- Import into Excel, MATLAB, R, or Python for further processing
- Columns: step, value, wall_time

## Training Statistics

Total training steps: {len(data.get('rollout/ep_rew_mean', {}).get('steps', [])):,}
Available metrics: {len(data)} scalars

### Key Metrics
{chr(10).join(f"- {tag}" for tag in sorted(data.keys())[:10])}

## Usage in Thesis

All PDF files are publication-ready:
- Vector graphics (crisp at any zoom level)
- Consistent fonts and styling
- 300 DPI resolution
- Suitable for grayscale printing

## Citation

If using these figures, cite the RL training configuration:
- Agent: PPO
- Environment: ScheduleEnv (Gymnasium)
- Observation space: 39-dimensional
- Action space: 20 discrete heuristics
"""

    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    print(f"  ✓ Saved: {readme_path.name}")


def main() -> None:
    """Main visualization pipeline."""

    print("RL TRAINING VISUALIZATION GENERATOR")
    print("Publication-Quality Figures for Thesis/Academic Use")

    # Paths
    project_root = Path(__file__).resolve().parents[2]
    tensorboard_dir = project_root / "logs" / "tensorboard" / "train"
    output_dir = project_root / "output" / "rl_training_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        print("\n Loading TensorBoard data...")
        data = load_tensorboard_data(tensorboard_dir)

        if not data:
            print("\n No training data found!")
            print(f"Expected TensorBoard logs in: {tensorboard_dir}")
            return

        print(f"\n Loaded {len(data)} metric(s)")

        # Generate all visualizations
        print("\n Generating publication-quality figures...")
        plot_training_curves(data, output_dir)
        plot_learning_metrics(data, output_dir)
        plot_reward_distribution(data, output_dir)
        plot_training_summary(data, output_dir)
        export_data_tables(data, output_dir)
        generate_readme(output_dir, data)

        print("\n" + "=" * 60)
        print(" VISUALIZATION COMPLETE!")

        print(f"\nOutput directory: {output_dir}")
        print("\nGenerated files:")
        print("   01_training_curves.pdf")
        print("   02_learning_metrics.pdf")
        print("   03_reward_distribution.pdf")
        print("   04_training_summary_dashboard.pdf")
        print("   csv/ (raw data exports)")
        print("   README.md (figure descriptions)")
        print("\n All figures are publication-ready (300 DPI, vector graphics)")

    except Exception as e:
        print(f"\n Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
