"""
Analysis Scripts for Thesis Experiments

Provides statistical comparison, visualization, and result analysis tools
for the comprehensive experimental framework.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def load_experiment_results() -> list[dict[str, Any]]:
    """
    Load all experiment results from consolidated output/experiments/ structure.

    Returns:
        List of experiment dictionaries with metrics
    """
    experiments_dir = Path("output/experiments")
    experiments = []

    if not experiments_dir.exists():
        console.print(
            "[red]No experiments directory found at output/experiments/[/red]"
        )
        console.print(
            "[yellow]Run 'uv run migrate' to organize existing experiments[/yellow]"
        )
        return []

    # Find all experiment directories in organized structure
    for exp_dir in experiments_dir.rglob("evaluation_*"):
        if not exp_dir.is_dir():
            continue

        # Look for metrics file
        metrics_file = exp_dir / "evolution_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    data = json.load(f)
                    data["experiment_dir"] = str(exp_dir)
                    data["experiment_name"] = exp_dir.name
                    # Extract method from path (e.g., output/experiments/baseline/pure-nsga/evaluation_...)
                    path_parts = exp_dir.parts
                    if len(path_parts) >= 4:
                        data["method_category"] = path_parts[
                            -3
                        ]  # baseline, repairs, etc.
                        data["method_type"] = path_parts[
                            -2
                        ]  # pure-nsga, nsga-repairs, etc.
                    experiments.append(data)
            except Exception as e:
                console.print(
                    f"[yellow]Warning: Could not load {metrics_file}: {e}[/yellow]"
                )

    console.print(
        f"[green]Loaded {len(experiments)} experiments from output/experiments/[/green]"
    )
    return experiments


def generate_comparison_table(experiments: list[dict[str, Any]]) -> Table:
    """
    Generate comparison table for all experiments.

    Args:
        experiments: List of experiment results

    Returns:
        Rich table with comparison metrics
    """
    table = Table(title="Thesis Experiments - Performance Comparison")
    table.add_column("Method", style="cyan")
    table.add_column("Group", style="dim")
    table.add_column("Hard Violations", justify="right")
    table.add_column("Soft Penalty", justify="right")
    table.add_column("Runtime (h)", justify="right")
    table.add_column("Hypervolume", justify="right")
    table.add_column("IGD", justify="right")

    # Sort experiments by group and method
    sorted_experiments = sorted(experiments, key=lambda x: x["experiment_name"])

    for exp in sorted_experiments:
        # Use new organized structure for method identification
        method_category = exp.get("method_category", "unknown")
        method_type = exp.get("method_type", "unknown")

        # Map to thesis experimental groups
        if method_category == "baseline":
            if "pure-nsga" in method_type:
                group, method = "A", "Pure NSGA-II"
            else:
                group, method = "A", "NSGA + Repairs"
        elif method_category == "repairs":
            group, method = "A", "NSGA + Repairs"
        elif method_category == "heuristics":
            group, method = "B", "NSGA + Heuristics"
        elif method_category == "full":
            group, method = "B", "Full GA"
        elif method_category == "roundrobin":
            group, method = "C", "Round-Robin"
        elif method_category == "rl":
            group, method = "C", "RL-Guided"
        else:
            group, method = "?", f"{method_category}/{method_type}"

        # Extract metrics (use final values)
        hard_violations = exp.get("final_hard_violations", "N/A")
        soft_penalty = exp.get("final_soft_penalty", "N/A")
        runtime_hours = exp.get("runtime_seconds", 0) / 3600
        hypervolume = exp.get("final_hypervolume", "N/A")
        igd = exp.get("final_igd", "N/A")

        table.add_row(
            method,
            group,
            (
                f"{hard_violations:.0f}"
                if isinstance(hard_violations, int | float)
                else str(hard_violations)
            ),
            (
                f"{soft_penalty:.2f}"
                if isinstance(soft_penalty, int | float)
                else str(soft_penalty)
            ),
            f"{runtime_hours:.1f}",
            (
                f"{hypervolume:.3f}"
                if isinstance(hypervolume, int | float)
                else str(hypervolume)
            ),
            f"{igd:.2f}" if isinstance(igd, int | float) else str(igd),
        )

    return table


def plot_convergence_comparison(experiments: list[dict[str, Any]]) -> None:
    """
    Generate convergence plots comparing all methods.

    Args:
        experiments: List of experiment results
    """
    plt.figure(figsize=(15, 10))

    # Create subplots for hard and soft violations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]
    color_idx = 0

    for exp in experiments:
        name = exp["experiment_name"]

        # Extract evolution data
        hard_violations = exp.get("hard_violations_per_generation", [])
        soft_penalties = exp.get("soft_penalties_per_generation", [])

        if hard_violations and soft_penalties:
            generations = range(len(hard_violations))

            # Plot hard violations
            ax1.plot(
                generations,
                hard_violations,
                label=name[:20],
                color=colors[color_idx % len(colors)],
                linewidth=2,
                alpha=0.8,
            )

            # Plot soft penalties
            ax2.plot(
                generations,
                soft_penalties,
                label=name[:20],
                color=colors[color_idx % len(colors)],
                linewidth=2,
                alpha=0.8,
            )

            color_idx += 1

    ax1.set_title("Hard Constraint Violations Over Generations")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Hard Violations")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Soft Constraint Penalties Over Generations")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Soft Penalty")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save to analysis directory
    output_path = Path("output/analysis")
    output_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path / "convergence_comparison.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    console.print(
        f"[green]Convergence comparison plot saved to {output_path}/convergence_comparison.png[/green]"
    )


def plot_runtime_vs_quality(experiments: list[dict[str, Any]]) -> None:
    """
    Generate runtime vs quality trade-off plots.

    Args:
        experiments: List of experiment results
    """
    # Extract data for plotting
    methods = []
    runtimes = []
    hard_violations = []
    soft_penalties = []

    for exp in experiments:
        name = exp["experiment_name"]
        methods.append(name[:15])  # Truncate for readability

        runtime_hours = exp.get("runtime_seconds", 0) / 3600
        runtimes.append(runtime_hours)

        hard_viol = exp.get("final_hard_violations", 0)
        hard_violations.append(hard_viol)

        soft_pen = exp.get("final_soft_penalty", 0)
        soft_penalties.append(soft_pen)

    # Create scatter plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Runtime vs Hard Violations
    ax1.scatter(runtimes, hard_violations, s=100, alpha=0.7, c=range(len(methods)))
    ax1.set_xlabel("Runtime (hours)")
    ax1.set_ylabel("Hard Constraint Violations")
    ax1.set_title("Runtime vs Hard Violations Trade-off")
    ax1.grid(True, alpha=0.3)

    # Add method labels
    for i, method in enumerate(methods):
        ax1.annotate(
            method,
            (runtimes[i], hard_violations[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Runtime vs Soft Penalties
    ax2.scatter(runtimes, soft_penalties, s=100, alpha=0.7, c=range(len(methods)))
    ax2.set_xlabel("Runtime (hours)")
    ax2.set_ylabel("Soft Constraint Penalties")
    ax2.set_title("Runtime vs Soft Penalties Trade-off")
    ax2.grid(True, alpha=0.3)

    # Add method labels
    for i, method in enumerate(methods):
        ax2.annotate(
            method,
            (runtimes[i], soft_penalties[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.tight_layout()
    output_path = Path("output/analysis")
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / "runtime_vs_quality.png", dpi=300, bbox_inches="tight")
    plt.close()

    console.print(
        f"[green]Runtime vs quality plot saved to {output_path}/runtime_vs_quality.png[/green]"
    )


def generate_statistical_analysis(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Perform statistical analysis on experiment results.

    Args:
        experiments: List of experiment results

    Returns:
        Dictionary with statistical analysis results
    """
    # Convert to DataFrame for easier analysis
    data = []
    for exp in experiments:
        row = {
            "method": exp["experiment_name"][:15],
            "hard_violations": exp.get("final_hard_violations", 0),
            "soft_penalty": exp.get("final_soft_penalty", 0),
            "runtime_hours": exp.get("runtime_seconds", 0) / 3600,
            "hypervolume": exp.get("final_hypervolume", 0),
            "igd": exp.get("final_igd", 0),
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Calculate summary statistics
    stats = {
        "summary": df.describe(),
        "best_hard": df.loc[df["hard_violations"].idxmin()]["method"],
        "best_soft": df.loc[df["soft_penalty"].idxmin()]["method"],
        "fastest": df.loc[df["runtime_hours"].idxmin()]["method"],
        "best_hypervolume": (
            df.loc[df["hypervolume"].idxmax()]["method"]
            if df["hypervolume"].max() > 0
            else "N/A"
        ),
        "correlation": df.corr(),
    }

    return stats


def main():
    """Main analysis function."""
    console.print("[cyan]Loading experiment results...[/cyan]")
    experiments = load_experiment_results()

    if not experiments:
        console.print("[red]No experiment results found in output/ directory[/red]")
        console.print(
            "[dim]Run experiments first: scripts/run_thesis_experiments.ps1[/dim]"
        )
        return

    console.print("[cyan]Generating comparison table...[/cyan]")
    table = generate_comparison_table(experiments)
    console.print(table)

    console.print("\n[cyan]Generating convergence plots...[/cyan]")
    plot_convergence_comparison(experiments)

    console.print("[cyan]Generating runtime vs quality plots...[/cyan]")
    plot_runtime_vs_quality(experiments)

    console.print("[cyan]Performing statistical analysis...[/cyan]")
    stats = generate_statistical_analysis(experiments)

    console.print("\n[green]Analysis Summary:[/green]")
    console.print(f"  Best Hard Violations: {stats['best_hard']}")
    console.print(f"  Best Soft Penalty: {stats['best_soft']}")
    console.print(f"  Fastest Runtime: {stats['fastest']}")
    console.print(f"  Best Hypervolume: {stats['best_hypervolume']}")

    # Save statistical results
    analysis_dir = Path("output/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    stats_path = analysis_dir / "statistical_analysis.json"

    with open(stats_path, "w") as f:
        # Convert numpy types to JSON serializable
        serializable_stats = {
            "summary": stats["summary"].to_dict(),
            "best_hard": stats["best_hard"],
            "best_soft": stats["best_soft"],
            "fastest": stats["fastest"],
            "best_hypervolume": stats["best_hypervolume"],
            "correlation": stats["correlation"].to_dict(),
        }
        json.dump(serializable_stats, f, indent=2)

    console.print(f"\n[green]Statistical analysis saved to {stats_path}[/green]")
    console.print("[green]All analysis complete![/green]")


if __name__ == "__main__":
    main()
