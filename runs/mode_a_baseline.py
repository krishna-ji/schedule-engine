#!/usr/bin/env python3
"""
Mode A: Baseline Pure NSGA-II

Pure NSGA-II baseline - No enhancements, no repair heuristics, no RL guidance.
This script is the foundation for comparing all other modes (B, C, D, E).

Usage:
    python runs/mode_a_baseline.py
"""
from __future__ import annotations

import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schedule_engine.metrics import (
    average_pairwise_diversity,
    calculate_hypervolume,
    calculate_spacing,
)
from schedule_engine.notebooks.core import (
    EvolutionConfig,
    EvolutionStats,
    course_aware_crossover,
    create_evaluator,
    create_random_individual,
    get_best_individual,
    get_constraint_breakdown,
    load_data,
    run_nsga2,
    smart_mutation,
)
from schedule_engine.notebooks.export import export_full_results
from schedule_engine.notebooks.viz import (
    plot_constraint_breakdown,
    plot_convergence,
    plot_diversity_metrics,
    plot_feasibility_progress,
    plot_pareto_front,
    print_summary,
)


def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging to file and console."""
    log_file = output_dir / "mode_a_baseline.log"

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger("mode_a_baseline")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def _plot_metrics_summary(
    stats: "EvolutionStats",
    spacing: float,
    hypervolume: float,
    diversity: float,
    save_path: Path,
) -> None:
    """Plot a summary of NSGA-II quality metrics."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Convergence (Hard)
    ax1 = axes[0, 0]
    ax1.plot(stats.generations, stats.min_hard, "b-", linewidth=2, label="Min")
    ax1.plot(stats.generations, stats.avg_hard, "g--", linewidth=1, label="Avg")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Hard Violations")
    ax1.set_title("Hard Constraint Convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Convergence (Soft)
    ax2 = axes[0, 1]
    ax2.plot(stats.generations, stats.min_soft, "r-", linewidth=2, label="Min")
    ax2.plot(
        stats.generations,
        stats.avg_soft,
        "orange",
        linestyle="--",
        linewidth=1,
        label="Avg",
    )
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Soft Penalty")
    ax2.set_title("Soft Constraint Convergence")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Feasibility Progress
    ax3 = axes[1, 0]
    ax3.plot(stats.generations, stats.feasible_count, "g-", linewidth=2)
    ax3.fill_between(
        stats.generations, 0, stats.feasible_count, alpha=0.3, color="green"
    )
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Feasible Count")
    ax3.set_title("Feasibility Progress")
    ax3.grid(True, alpha=0.3)

    # 4. Quality Metrics Summary (Text)
    ax4 = axes[1, 1]
    ax4.axis("off")

    metrics_text = f"""
    NSGA-II Quality Metrics
    ══════════════════════════════

    Spacing:           {spacing:.4f}
    (Lower = more uniform Pareto front)

    Hypervolume:       {hypervolume:.2f}
    (Higher = better Pareto front quality)

    Population Diversity: {diversity:.4f}
    (Higher = more diverse solutions)

    ──────────────────────────────
    Final Results:
    • Min Hard: {stats.min_hard[-1]:.0f}
    • Min Soft: {stats.min_soft[-1]:.1f}
    • Feasible: {stats.feasible_count[-1]}/{len(stats.generations)}
    • Time: {stats.elapsed_time:.1f}s
    """

    ax4.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle(
        "Mode A: NSGA-II Baseline - Complete Metrics Summary",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")


def main() -> None:
    """Run Mode A: Baseline Pure NSGA-II experiment."""
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # GA Parameters - SAME AS MODE B1 for fair comparison
    POP_SIZE = 50
    NGEN = 200
    CXPB = 0.9
    MUTPB = 0.2

    # Fitness weights: -1.0 = minimize both (equal weight)
    FITNESS_WEIGHTS = (-1.0, -1.0)

    # Paths
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output" / "mode_a_baseline" / TIMESTAMP
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(OUTPUT_DIR)
    logger.info("=" * 60)
    logger.info("MODE A: BASELINE PURE NSGA-II")
    logger.info("=" * 60)
    logger.info(f"Config: pop={POP_SIZE}, ngen={NGEN}, weights={FITNESS_WEIGHTS}")
    logger.info(f"Output: {OUTPUT_DIR}")

    # Evolution config
    config = EvolutionConfig(
        pop_size=POP_SIZE,
        ngen=NGEN,
        cxpb=CXPB,
        mutpb=MUTPB,
        fitness_weights=FITNESS_WEIGHTS,
        verbose=True,
        log_interval=20,
    )

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    logger.info("Loading data...")
    data = load_data(
        data_dir=DATA_DIR,
        opening_time="10:00",
        closing_time="17:00",
        closed_days=["Saturday"],
    )
    logger.info(f"Data loaded: {data.summary()}")

    # ==========================================================================
    # TEST COMPONENTS
    # ==========================================================================
    logger.info("Testing components...")
    test_ind = create_random_individual(data)
    logger.info(f"Individual has {len(test_ind)} genes")

    evaluate = create_evaluator(data)
    test_fitness = evaluate(test_ind)
    logger.info(f"Test fitness: hard={test_fitness[0]}, soft={test_fitness[1]}")

    # ==========================================================================
    # RUN NSGA-II EVOLUTION
    # ==========================================================================
    logger.info("Starting NSGA-II evolution...")
    final_pop, stats = run_nsga2(
        data=data,
        config=config,
        create_individual_fn=create_random_individual,
        evaluate_fn=evaluate,
        crossover_fn=course_aware_crossover,
        mutate_fn=lambda ind: smart_mutation(ind, data),
        seed=SEED,
    )
    logger.info(f"Evolution completed in {stats.elapsed_time:.1f}s")

    # ==========================================================================
    # RESULTS & VISUALIZATION
    # ==========================================================================
    logger.info("Generating results and visualizations...")

    # Get best solution
    best = get_best_individual(final_pop)
    breakdown = get_constraint_breakdown(best, data)

    # Print summary
    print_summary(final_pop, stats, breakdown)

    # ==========================================================================
    # NSGA-II SPECIFIC VISUALIZATIONS
    # ==========================================================================
    logger.info("Generating NSGA-II visualizations...")

    # 1. Basic Convergence Plot
    plot_convergence(
        stats,
        OUTPUT_DIR / "mode_a_convergence.png",
        title_prefix="Mode A: ",
        show=False,
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_a_convergence.png'}")

    # 2. Constraint Breakdown
    plot_constraint_breakdown(
        breakdown,
        OUTPUT_DIR / "mode_a_breakdown.png",
        title="Mode A: Constraint Violations",
        show=False,
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_a_breakdown.png'}")

    # 3. Pareto Front Plot (Objective Space)
    plot_pareto_front(
        final_pop,
        OUTPUT_DIR / "mode_a_pareto_front.png",
        title="Mode A: Pareto Front (Hard vs Soft)",
        show=False,
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_a_pareto_front.png'}")

    # 4. Feasibility Progress
    plot_feasibility_progress(
        stats,
        OUTPUT_DIR / "mode_a_feasibility.png",
        title_prefix="Mode A: ",
        show=False,
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_a_feasibility.png'}")

    # 5. Calculate and log NSGA-II metrics
    logger.info("Calculating NSGA-II quality metrics...")

    # Spacing (Pareto front uniformity)
    spacing = calculate_spacing(final_pop)
    logger.info(f"  Spacing: {spacing:.4f} (lower = more uniform)")

    # Hypervolume (quality of Pareto front)
    # Reference point: worst possible values + margin
    ref_point = (
        max(ind.fitness.values[0] for ind in final_pop) * 1.1 + 1,
        max(ind.fitness.values[1] for ind in final_pop) * 1.1 + 1,
    )
    hypervolume = calculate_hypervolume(final_pop, ref_point)
    logger.info(f"  Hypervolume: {hypervolume:.2f} (higher = better)")

    # Population diversity
    diversity = average_pairwise_diversity(
        final_pop, sample_size=min(50, len(final_pop))
    )
    logger.info(f"  Population Diversity: {diversity:.4f} (higher = more diverse)")

    # 6. Summary metrics plot
    _plot_metrics_summary(
        stats,
        spacing,
        hypervolume,
        diversity,
        OUTPUT_DIR / "mode_a_metrics_summary.png",
    )
    logger.info(f"Saved: {OUTPUT_DIR / 'mode_a_metrics_summary.png'}")

    # ==========================================================================
    # EXPORT RESULTS
    # ==========================================================================
    logger.info("Exporting full results...")
    export_paths = export_full_results(
        population=final_pop,
        stats=stats,
        data=data,
        output_dir=OUTPUT_DIR,
        mode_name="mode_a_baseline",
    )

    # Save experiment metadata (convert numpy types to native Python)
    def to_native(val: int | float | np.integer | np.floating | None) -> int | float | None:  # type: ignore[type-arg]
        """Convert numpy scalar to native Python type."""
        if val is None:
            return None
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        return val

    metadata = {
        "experiment": "mode_a_baseline",
        "timestamp": TIMESTAMP,
        "config": {
            "seed": SEED,
            "pop_size": POP_SIZE,
            "ngen": NGEN,
            "cxpb": CXPB,
            "mutpb": MUTPB,
            "fitness_weights": list(FITNESS_WEIGHTS),
        },
        "results": {
            "elapsed_time": to_native(stats.elapsed_time),
            "final_min_hard": to_native(stats.min_hard[-1]) if stats.min_hard else None,
            "final_min_soft": to_native(stats.min_soft[-1]) if stats.min_soft else None,
            "final_feasible_count": (
                to_native(stats.feasible_count[-1]) if stats.feasible_count else 0
            ),
        },
        "nsga2_metrics": {
            "spacing": to_native(spacing),
            "hypervolume": to_native(hypervolume),
            "population_diversity": to_native(diversity),
            "pareto_front_size": len(
                [
                    ind
                    for ind in final_pop
                    if ind.fitness.values[0] == stats.min_hard[-1]
                ]
            ),
        },
        "constraint_breakdown": {k: to_native(v) for k, v in breakdown.items()},
    }

    with open(OUTPUT_DIR / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved: {OUTPUT_DIR / 'experiment_metadata.json'}")

    logger.info("=" * 60)
    logger.info(f"All files saved to: {OUTPUT_DIR}")
    logger.info("MODE A COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
