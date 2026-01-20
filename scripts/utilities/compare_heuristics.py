"""
Heuristic Comparison Analysis Script

Collects results from all individual heuristic tests and generates:
1. Comparison CSV with key metrics
2. Comparison plots (improvement, success rate, efficiency)
3. Summary statistics report

Usage:
    uv run python scripts/utilities/compare_heuristics.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


def collect_results() -> pd.DataFrame:
    """Collect results from all heuristic tests."""
    base_dir = Path("output/f-heuristic-testing")

    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist. Run tests first!")
        return pd.DataFrame()

    results: list[dict[str, Any]] = []

    for test_dir in sorted(base_dir.iterdir()):
        if not test_dir.is_dir() or not test_dir.name.startswith("test-"):
            continue

        # Find latest evaluation folder
        eval_dirs = sorted(test_dir.glob("evaluation_*"))
        if not eval_dirs:
            print(f"Warning: No evaluation folder in {test_dir.name}")
            continue

        latest = eval_dirs[-1]
        tracking_file = latest / "heuristic_tracking.json"
        results_file = latest / "results.json"

        if not tracking_file.exists():
            print(f"Warning: No heuristic_tracking.json in {latest.name}")
            continue
        if not results_file.exists():
            print(f"Warning: No results.json in {latest.name}")
            continue

        # Load data
        with open(tracking_file) as f:
            tracking = json.load(f)
        with open(results_file) as f:
            results_data = json.load(f)

        # Extract key metrics
        heuristic_name = test_dir.name.replace("test-", "").replace("-", "_")
        summary = tracking.get("heuristic_summary", {}).get(heuristic_name, {})

        # Get fitness data
        best_fitness = results_data.get("best_fitness", [999, 999])
        if len(best_fitness) < 2:
            best_fitness = [999, 999]

        results.append(
            {
                "heuristic": heuristic_name,
                "total_improvement": summary.get("total_improvement", 0.0),
                "success_rate": summary.get("success_rate", 0.0),
                "avg_improvement": summary.get("average_improvement", 0.0),
                "total_applications": summary.get("total_applications", 0),
                "successful_applications": summary.get("successful_applications", 0),
                "avg_time_ms": summary.get("average_time", 0.0) * 1000,  # Convert to ms
                "total_time_s": summary.get("total_time", 0.0),
                "best_hard_violations": best_fitness[0],
                "best_soft_penalty": best_fitness[1],
                "generations": results_data.get("generations", 0),
            }
        )

    if not results:
        print("Error: No results collected. Check output directory!")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    print(f"Collected {len(df)} heuristic test results")
    return df


def generate_comparison_report(df: pd.DataFrame) -> None:
    """Generate comparison plots and CSV."""
    if df.empty:
        print("No data to generate report!")
        return

    output_dir = Path("output/f-heuristic-testing")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate efficiency (improvement per millisecond)
    df["efficiency"] = df["total_improvement"].abs() / (df["avg_time_ms"] + 0.001)

    # Sort by total improvement (descending = more negative = better)
    df_sorted = df.sort_values("total_improvement", ascending=True)

    # Save CSV
    csv_path = output_dir / "heuristic_comparison.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"Saved comparison CSV: {csv_path}")

    # ========================================
    # PLOT 1: Total Improvement Comparison
    # ========================================
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["green" if x < 0 else "red" for x in df_sorted["total_improvement"]]
    ax.barh(df_sorted["heuristic"], df_sorted["total_improvement"], color=colors)
    ax.set_xlabel("Total Improvement (Negative = Better)")
    ax.set_title("Heuristic Comparison: Total Constraint Reduction")
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1)
    plt.tight_layout()
    plot1_path = output_dir / "comparison_total_improvement.png"
    plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot1_path}")

    # ========================================
    # PLOT 2: Success Rate Comparison
    # ========================================
    df_sorted_success = df.sort_values("success_rate", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(df_sorted_success["heuristic"], df_sorted_success["success_rate"])
    ax.set_xlabel("Success Rate (0-1)")
    ax.set_title("Heuristic Comparison: Success Rate")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plot2_path = output_dir / "comparison_success_rate.png"
    plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot2_path}")

    # ========================================
    # PLOT 3: Efficiency Comparison
    # ========================================
    df_sorted_eff = df.sort_values("efficiency", ascending=False)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(df_sorted_eff["heuristic"], df_sorted_eff["efficiency"])
    ax.set_xlabel("Efficiency (Improvement per ms)")
    ax.set_title("Heuristic Comparison: Time Efficiency")
    plt.tight_layout()
    plot3_path = output_dir / "comparison_efficiency.png"
    plt.savefig(plot3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot3_path}")

    # ========================================
    # PLOT 4: Scatter - Success Rate vs Improvement
    # ========================================
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df["success_rate"],
        df["total_improvement"],
        s=100,
        c=df["efficiency"],
        cmap="viridis",
        alpha=0.7,
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["heuristic"],
            (row["success_rate"], row["total_improvement"]),
            fontsize=8,
            alpha=0.7,
        )
    ax.set_xlabel("Success Rate")
    ax.set_ylabel("Total Improvement (Negative = Better)")
    ax.set_title("Heuristic Performance: Success Rate vs Improvement")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)
    plt.colorbar(scatter, label="Efficiency (Improvement/ms)")
    plt.tight_layout()
    plot4_path = output_dir / "comparison_scatter.png"
    plt.savefig(plot4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot4_path}")

    # ========================================
    # TEXT REPORT
    # ========================================
    print("\n" + "=" * 80)
    print("HEURISTIC COMPARISON REPORT")
    print("=" * 80)

    print("\n--- TOP 5 BY TOTAL IMPROVEMENT ---")
    top5_improvement = df_sorted.head(5)[
        ["heuristic", "total_improvement", "success_rate", "avg_time_ms"]
    ]
    print(top5_improvement.to_string(index=False))

    print("\n--- TOP 5 BY SUCCESS RATE ---")
    top5_success = df.nlargest(5, "success_rate")[
        ["heuristic", "success_rate", "total_improvement", "avg_time_ms"]
    ]
    print(top5_success.to_string(index=False))

    print("\n--- TOP 5 BY EFFICIENCY ---")
    top5_efficiency = df.nlargest(5, "efficiency")[
        ["heuristic", "efficiency", "total_improvement", "avg_time_ms"]
    ]
    print(top5_efficiency.to_string(index=False))

    print("\n--- SUMMARY STATISTICS ---")
    print(f"Total heuristics tested: {len(df)}")
    print(f"Best total improvement: {df['total_improvement'].min():.2f}")
    print(f"Best heuristic: {df.loc[df['total_improvement'].idxmin(), 'heuristic']}")
    print(
        f"Highest success rate: {df['success_rate'].max():.2%} ({df.loc[df['success_rate'].idxmax(), 'heuristic']})"
    )
    print(
        f"Most efficient: {df.loc[df['efficiency'].idxmax(), 'heuristic']} ({df['efficiency'].max():.4f} imp/ms)"
    )

    print("\n--- BEST FINAL SOLUTIONS ---")
    best_hard = df.nsmallest(3, "best_hard_violations")[
        ["heuristic", "best_hard_violations", "best_soft_penalty"]
    ]
    print(best_hard.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"All results saved to: {output_dir}")
    print("=" * 80 + "\n")


def main() -> None:
    """Main execution."""
    print("\n=== HEURISTIC COMPARISON ANALYSIS ===\n")

    # Collect results
    df = collect_results()

    if df.empty:
        return

    # Generate report
    generate_comparison_report(df)


if __name__ == "__main__":
    main()
