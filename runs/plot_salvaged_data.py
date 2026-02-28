#!/usr/bin/env python3
"""
Plot Salvaged 150k Training Data
Reads the training logs from the crashed capstone run and plots convergence.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def plot_salvaged_training_data():
    """Plot the salvaged training data from the 150k timestep run."""

    # Path to the salvaged data
    salvaged_dir = PROJECT_ROOT / "output" / "rl_capstone" / "20260227_220706"
    step_log_path = salvaged_dir / "step_log.csv"
    training_curve_path = salvaged_dir / "training_curve.csv"

    print(f"Loading salvaged data from: {salvaged_dir}")

    if not step_log_path.exists():
        print(f"❌ Step log not found: {step_log_path}")
        return

    if not training_curve_path.exists():
        print(f"❌ Training curve not found: {training_curve_path}")
        return

    # Load the data
    print("📊 Loading step log (151,552 timesteps)...")
    step_df = pd.read_csv(step_log_path)

    print("📈 Loading training curve (3,093 episodes)...")
    episode_df = pd.read_csv(training_curve_path)

    print(f"Step log shape: {step_df.shape}")
    print(f"Episode data shape: {episode_df.shape}")
    print("\nStep log columns:", list(step_df.columns))
    print("Episode data columns:", list(episode_df.columns))

    # Create the convergence plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "Salvaged 150k Training Data - Capstone Thesis Run",
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Step-level convergence (best_hard vs timestep)
    if "best_hard" in step_df.columns and "timestep" in step_df.columns:
        ax1 = axes[0, 0]
        ax1.plot(
            step_df["timestep"], step_df["best_hard"], "b-", alpha=0.7, linewidth=0.8
        )
        ax1.set_xlabel("Timestep")
        ax1.set_ylabel("Best Hard Violations")
        ax1.set_title("Hard Constraint Convergence (150k Steps)")
        ax1.grid(True, alpha=0.3)

        # Find and highlight the minimum
        min_hard_idx = step_df["best_hard"].idxmin()
        min_hard_value = step_df.loc[min_hard_idx, "best_hard"]
        min_hard_timestep = step_df.loc[min_hard_idx, "timestep"]
        ax1.plot(
            min_hard_timestep,
            min_hard_value,
            "ro",
            markersize=8,
            label=f"Min: {min_hard_value:.2f} @ t={min_hard_timestep}",
        )
        ax1.legend()

        print(
            f"🏆 LOWEST BEST_HARD SCORE: {min_hard_value:.2f} at timestep {min_hard_timestep}"
        )

    # Plot 2: Step-level convergence (best_soft vs timestep)
    if "best_soft" in step_df.columns and "timestep" in step_df.columns:
        ax2 = axes[0, 1]
        ax2.plot(
            step_df["timestep"], step_df["best_soft"], "g-", alpha=0.7, linewidth=0.8
        )
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Best Soft Violations")
        ax2.set_title("Soft Constraint Convergence (150k Steps)")
        ax2.grid(True, alpha=0.3)

        # Find and highlight the minimum
        min_soft_idx = step_df["best_soft"].idxmin()
        min_soft_value = step_df.loc[min_soft_idx, "best_soft"]
        min_soft_timestep = step_df.loc[min_soft_idx, "timestep"]
        ax2.plot(
            min_soft_timestep,
            min_soft_value,
            "ro",
            markersize=8,
            label=f"Min: {min_soft_value:.2f} @ t={min_soft_timestep}",
        )
        ax2.legend()

        print(
            f"🏆 LOWEST BEST_SOFT SCORE: {min_soft_value:.2f} at timestep {min_soft_timestep}"
        )

    # Plot 3: Episode rewards over time
    if "episode" in episode_df.columns and "reward" in episode_df.columns:
        ax3 = axes[1, 0]
        ax3.plot(
            episode_df["episode"],
            episode_df["reward"],
            "purple",
            alpha=0.7,
            linewidth=1.0,
        )
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Episode Reward")
        ax3.set_title("Reward Learning Curve (3,093 Episodes)")
        ax3.grid(True, alpha=0.3)

        # Add moving average
        if len(episode_df) > 100:
            window = min(100, len(episode_df) // 10)
            moving_avg = episode_df["reward"].rolling(window=window, center=True).mean()
            ax3.plot(
                episode_df["episode"],
                moving_avg,
                "red",
                linewidth=2,
                label=f"{window}-episode moving avg",
            )
            ax3.legend()

    # Plot 4: Episode length over time
    if "episode" in episode_df.columns and "length" in episode_df.columns:
        ax4 = axes[1, 1]
        ax4.plot(
            episode_df["episode"],
            episode_df["length"],
            "orange",
            alpha=0.7,
            linewidth=1.0,
        )
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Episode Length")
        ax4.set_title("Episode Length Stability")
        ax4.grid(True, alpha=0.3)

        # Show mean line
        mean_length = episode_df["length"].mean()
        ax4.axhline(
            y=mean_length, color="red", linestyle="--", label=f"Mean: {mean_length:.1f}"
        )
        ax4.legend()

    plt.tight_layout()

    # Save the plot
    output_path = salvaged_dir / "salvaged_training_convergence.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved plot: {output_path}")

    # Also save as PNG for quick viewing
    png_path = salvaged_dir / "salvaged_training_convergence.png"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved plot: {png_path}")

    # Don't show interactive plot in terminal
    # plt.show()

    # Summary statistics
    print("\n" + "=" * 60)
    print("📋 SALVAGED DATA SUMMARY")
    print("=" * 60)
    if "best_hard" in step_df.columns:
        print("Best Hard Violations:")
        print(f"  Minimum: {step_df['best_hard'].min():.2f}")
        print(f"  Maximum: {step_df['best_hard'].max():.2f}")
        print(f"  Final:   {step_df['best_hard'].iloc[-1]:.2f}")

    if "best_soft" in step_df.columns:
        print("Best Soft Violations:")
        print(f"  Minimum: {step_df['best_soft'].min():.2f}")
        print(f"  Maximum: {step_df['best_soft'].max():.2f}")
        print(f"  Final:   {step_df['best_soft'].iloc[-1]:.2f}")

    if "reward" in episode_df.columns:
        print("Episode Rewards:")
        print(f"  Mean:     {episode_df['reward'].mean():.4f}")
        print(f"  Best:     {episode_df['reward'].max():.4f}")
        print(f"  Final:    {episode_df['reward'].iloc[-1]:.4f}")

    print("Total Training:")
    print(f"  Timesteps: {len(step_df):,}")
    print(f"  Episodes:  {len(episode_df):,}")
    print("=" * 60)


if __name__ == "__main__":
    plot_salvaged_training_data()
