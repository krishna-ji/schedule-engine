import matplotlib.pyplot as plt
import numpy as np

from schedule_engine.utils.output_paths import get_nsga_plot_dir

from .thesis_style import (
    apply_thesis_style,
    create_thesis_figure,
    format_axis,
    get_color,
    save_figure,
)

# Apply thesis styling
apply_thesis_style()


def _extract_repair_series(repair_history: list[dict]) -> dict[str, list[float]]:
    if not repair_history:
        return {}

    ordered = sorted(repair_history, key=lambda item: int(item.get("generation", 0)))
    generations = [int(item.get("generation", idx)) for idx, item in enumerate(ordered)]
    repairs_applied = [float(item.get("repairs_applied", 0.0)) for item in ordered]
    delta_hard = [float(item.get("delta_hard", 0.0)) for item in ordered]
    delta_soft = [float(item.get("delta_soft", 0.0)) for item in ordered]
    repair_time_ms = [
        (
            float(item.get("repair_time_ms"))
            if item.get("repair_time_ms") is not None
            else np.nan
        )
        for item in ordered
    ]

    return {
        "generations": generations,
        "repairs_applied": repairs_applied,
        "delta_hard": delta_hard,
        "delta_soft": delta_soft,
        "repair_time_ms": repair_time_ms,
    }


def plot_repair_efficacy_over_generations(
    repair_history: list[dict], output_dir: str
) -> None:
    """Plot repairs applied, delta hard, and delta soft over generations."""
    series = _extract_repair_series(repair_history)
    if not series:
        return

    plot_dir = get_nsga_plot_dir(output_dir)
    generations = series["generations"]

    # Repairs applied
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))
    ax.plot(
        generations,
        series["repairs_applied"],
        color=get_color("blue"),
        linewidth=2.5,
        marker="o",
        markersize=4,
        markevery=max(1, len(generations) // 20),
    )
    format_axis(
        ax,
        xlabel="Generation",
        ylabel="Repairs Applied",
        title="Repair Steps Applied Over Generations",
        legend=False,
    )
    plt.tight_layout()
    save_figure(fig, plot_dir / "repair_steps_applied_over_generations.pdf")

    # Delta hard
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))
    ax.plot(
        generations,
        series["delta_hard"],
        color=get_color("red"),
        linewidth=2.5,
        marker="s",
        markersize=4,
        markevery=max(1, len(generations) // 20),
    )
    format_axis(
        ax,
        xlabel="Generation",
        ylabel="Hard Violation Improvement",
        title="Repair Improvement in Hard Violations Over Generations",
        legend=False,
    )
    plt.tight_layout()
    save_figure(fig, plot_dir / "repair_delta_hard_over_generations.pdf")

    # Delta soft
    fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))
    ax.plot(
        generations,
        series["delta_soft"],
        color=get_color("green"),
        linewidth=2.5,
        marker="^",
        markersize=4,
        markevery=max(1, len(generations) // 20),
    )
    format_axis(
        ax,
        xlabel="Generation",
        ylabel="Soft Penalty Improvement",
        title="Repair Improvement in Soft Penalties Over Generations",
        legend=False,
    )
    plt.tight_layout()
    save_figure(fig, plot_dir / "repair_delta_soft_over_generations.pdf")


def plot_repair_time_and_share(
    repair_history: list[dict],
    generation_times: list[float] | None,
    output_dir: str,
) -> None:
    """Plot repair time per generation and repair time share if available."""
    series = _extract_repair_series(repair_history)
    if not series:
        return

    plot_dir = get_nsga_plot_dir(output_dir)
    generations = series["generations"]

    repair_time_ms = np.array(series["repair_time_ms"], dtype=float)
    if not np.all(np.isnan(repair_time_ms)):
        fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))
        ax.plot(
            generations,
            repair_time_ms,
            color=get_color("purple"),
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(generations) // 20),
        )
        format_axis(
            ax,
            xlabel="Generation",
            ylabel="Repair Time (ms)",
            title="Repair Time Per Generation",
            legend=False,
        )
        plt.tight_layout()
        save_figure(fig, plot_dir / "repair_time_ms_over_generations.pdf")

    if generation_times:
        gen_times = np.array(generation_times, dtype=float)
        length = min(len(gen_times), len(generations))
        if length > 0:
            fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))
            ax.plot(
                generations[:length],
                gen_times[:length],
                color=get_color("orange"),
                linewidth=2.5,
                marker="s",
                markersize=4,
                markevery=max(1, length // 20),
            )
            format_axis(
                ax,
                xlabel="Generation",
                ylabel="Runtime (s)",
                title="Runtime Per Generation",
                legend=False,
            )
            plt.tight_layout()
            save_figure(fig, plot_dir / "runtime_per_generation_seconds.pdf")

            if not np.all(np.isnan(repair_time_ms)):
                share = (repair_time_ms[:length] / (gen_times[:length] * 1000.0)) * 100
                fig, ax = create_thesis_figure(1, 1, figsize=(10, 6))
                ax.plot(
                    generations[:length],
                    share,
                    color=get_color("brown"),
                    linewidth=2.5,
                    marker="^",
                    markersize=4,
                    markevery=max(1, length // 20),
                )
                format_axis(
                    ax,
                    xlabel="Generation",
                    ylabel="Repair Time Share (%)",
                    title="Repair Time Share Over Generations",
                    legend=False,
                )
                plt.tight_layout()
                save_figure(
                    fig, plot_dir / "repair_time_share_percent_over_generations.pdf"
                )


def plot_operator_performance(
    operator_stats: dict[str, dict[str, float]], output_dir: str
) -> None:
    """Plot operator success rate and average improvement per operator."""
    if not operator_stats:
        return

    names = sorted(operator_stats.keys())
    steps = np.array([operator_stats[n].get("steps", 0.0) for n in names], dtype=float)
    applied = np.array(
        [operator_stats[n].get("applied", 0.0) for n in names], dtype=float
    )
    delta_hard = np.array(
        [operator_stats[n].get("delta_hard", 0.0) for n in names], dtype=float
    )
    delta_soft = np.array(
        [operator_stats[n].get("delta_soft", 0.0) for n in names], dtype=float
    )

    success_rate = np.where(steps > 0, (applied / steps) * 100.0, 0.0)
    avg_improvement = np.where(
        applied > 0, (delta_hard * 1000.0 + delta_soft) / applied, 0.0
    )

    plot_dir = get_nsga_plot_dir(output_dir)

    fig, ax = create_thesis_figure(1, 1, figsize=(11, 6))
    ax.bar(
        names,
        success_rate,
        color=get_color("green"),
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )
    format_axis(
        ax,
        xlabel="Operator",
        ylabel="Success Rate (%)",
        title="Operator Success Rate",
        legend=False,
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    plt.tight_layout()
    save_figure(fig, plot_dir / "operator_success_rate_percent.pdf")

    fig, ax = create_thesis_figure(1, 1, figsize=(11, 6))
    ax.bar(
        names,
        avg_improvement,
        color=get_color("blue"),
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )
    format_axis(
        ax,
        xlabel="Operator",
        ylabel="Average Improvement Score",
        title="Operator Average Improvement",
        legend=False,
    )
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    plt.tight_layout()
    save_figure(fig, plot_dir / "operator_average_improvement_score.pdf")
