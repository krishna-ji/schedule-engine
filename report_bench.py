#!/usr/bin/env python3
"""Read bench_compare results and produce a GO/NO-GO migration report.

Reads:
    results/bench_compare/runs.jsonl
    results/bench_compare/summary.json

Prints:
    - Winner on best_soft at final generation (median across seeds)
    - Winner on time-to-feasible (first gen where best_hard == 0)
    - Per-seed detail table
    - Runtime comparison

Generates matplotlib plots (saved to results/bench_compare/):
    - best_hard_over_gens.png
    - best_soft_over_gens.png
    - time_per_gen.png

Usage:
    python report_bench.py
    python report_bench.py --results-dir results/bench_compare
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================================
#  Data loading
# =====================================================================


def load_data(results_dir: Path) -> tuple[list[dict], dict]:
    """Load JSONL rows and summary from bench_compare output."""
    jsonl_path = results_dir / "runs.jsonl"
    summary_path = results_dir / "summary.json"

    if not jsonl_path.exists():
        print(f"ERROR: {jsonl_path} not found. Run bench_compare.py first.")
        sys.exit(1)

    rows: list[dict] = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)

    return rows, summary


# =====================================================================
#  Analysis helpers
# =====================================================================


def group_rows(rows: list[dict]) -> dict[str, dict[int, list[dict]]]:
    """Group rows by solver -> seed -> [rows sorted by gen]."""
    out: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r["solver"]][r["seed"]].append(r)
    # Sort each seed's rows by gen
    for solver in out:
        for seed in out[solver]:
            out[solver][seed].sort(key=lambda x: x["gen"])
    return out


def time_to_feasible(seed_rows: list[dict]) -> int | None:
    """Return first gen where best_hard == 0, or None."""
    for r in seed_rows:
        if r["best_hard"] == 0:
            return int(r["gen"])
    return None


def final_best_soft(seed_rows: list[dict]) -> float:
    """Return best_soft at the last generation."""
    return float(seed_rows[-1]["best_soft"])


def final_best_hard(seed_rows: list[dict]) -> float:
    """Return best_hard at the last generation."""
    return float(seed_rows[-1]["best_hard"])


# =====================================================================
#  Report
# =====================================================================


def print_report(rows: list[dict], summary: dict) -> dict:
    """Print text report and return verdict dict."""
    grouped = group_rows(rows)
    solvers = sorted(grouped.keys())

    if len(solvers) < 2:
        print(f"Only solver found: {solvers}. Need both pymoo and deap for comparison.")
        print("Run bench_compare.py without --pymoo-only or --deap-only.")
        return {"verdict": "INCOMPLETE"}

    config = summary.get("config", {})
    gens = config.get("gens", "?")
    pop = config.get("pop", "?")
    seeds_list = config.get("seeds", [])

    print("=" * 70)
    print("  DEAP vs pymoo Migration Benchmark Report")
    print(f"  gens={gens}  pop={pop}  seeds={seeds_list}")
    print("=" * 70)

    # --- Per-seed table ---
    print(
        f"\n{'Seed':>6} {'Solver':<8} {'Final Hard':>11} {'Final Soft':>11} "
        f"{'TTF (gen)':>10} {'s/gen':>8}"
    )
    print("-" * 62)

    ttf_data: dict[str, list[int | None]] = defaultdict(list)
    soft_data: dict[str, list[float]] = defaultdict(list)
    hard_data: dict[str, list[float]] = defaultdict(list)
    spg_data: dict[str, list[float]] = defaultdict(list)

    for seed in sorted(set(s for sv in grouped.values() for s in sv)):
        for solver in solvers:
            if seed not in grouped[solver]:
                continue
            sr = grouped[solver][seed]
            ttf = time_to_feasible(sr)
            fb_soft = final_best_soft(sr)
            fb_hard = final_best_hard(sr)
            mean_spg = np.mean([r["time_per_gen"] for r in sr if r["gen"] > 0])

            ttf_data[solver].append(ttf)
            soft_data[solver].append(fb_soft)
            hard_data[solver].append(fb_hard)
            spg_data[solver].append(float(mean_spg))

            ttf_str = str(ttf) if ttf is not None else "never"
            print(
                f"{seed:>6} {solver:<8} {fb_hard:>11.0f} {fb_soft:>11.0f} "
                f"{ttf_str:>10} {mean_spg:>8.3f}"
            )

    # --- Median comparison ---
    print(f"\n{'=' * 70}")
    print("MEDIAN ACROSS SEEDS")
    print(f"{'=' * 70}")
    print(f"{'Metric':<30} {'pymoo':>14} {'deap':>14} {'winner':>10}")
    print("-" * 70)

    verdict: dict[str, object] = {}

    # Best soft at final gen
    med_soft_pymoo = float(np.median(soft_data.get("pymoo", [np.nan])))
    med_soft_deap = float(np.median(soft_data.get("deap", [np.nan])))
    winner_soft = "pymoo" if med_soft_pymoo <= med_soft_deap else "deap"
    verdict["soft_winner"] = winner_soft
    print(
        f"{'median final best_soft':<30} {med_soft_pymoo:>14.1f} {med_soft_deap:>14.1f} {winner_soft:>10}"
    )

    # Best hard at final gen
    med_hard_pymoo = float(np.median(hard_data.get("pymoo", [np.nan])))
    med_hard_deap = float(np.median(hard_data.get("deap", [np.nan])))
    winner_hard = "pymoo" if med_hard_pymoo <= med_hard_deap else "deap"
    verdict["hard_winner"] = winner_hard
    print(
        f"{'median final best_hard':<30} {med_hard_pymoo:>14.1f} {med_hard_deap:>14.1f} {winner_hard:>10}"
    )

    # Time to feasible
    def _median_ttf(lst: list[int | None]) -> float:
        nums = [x for x in lst if x is not None]
        return float(np.median(nums)) if nums else float("inf")

    med_ttf_pymoo = _median_ttf(ttf_data.get("pymoo", []))
    med_ttf_deap = _median_ttf(ttf_data.get("deap", []))
    winner_ttf = "pymoo" if med_ttf_pymoo <= med_ttf_deap else "deap"
    if med_ttf_pymoo == float("inf") and med_ttf_deap == float("inf"):
        winner_ttf = "neither"
    verdict["ttf_winner"] = winner_ttf
    ttf_p_s = f"{med_ttf_pymoo:.0f}" if med_ttf_pymoo != float("inf") else "never"
    ttf_d_s = f"{med_ttf_deap:.0f}" if med_ttf_deap != float("inf") else "never"
    print(
        f"{'median time-to-feasible (gen)':<30} {ttf_p_s:>14} {ttf_d_s:>14} {winner_ttf:>10}"
    )

    # Runtime
    med_spg_pymoo = float(np.median(spg_data.get("pymoo", [np.nan])))
    med_spg_deap = float(np.median(spg_data.get("deap", [np.nan])))
    winner_spg = "pymoo" if med_spg_pymoo <= med_spg_deap else "deap"
    verdict["speed_winner"] = winner_spg
    print(
        f"{'median sec/gen':<30} {med_spg_pymoo:>14.3f} {med_spg_deap:>14.3f} {winner_spg:>10}"
    )

    # --- GO / NO-GO ---
    # pymoo is GO if:
    #   1) median best_soft <= deap (pymoo at least as good), OR
    #   2) pymoo reaches feasibility faster
    # AND no uncompensated runtime regression.
    #
    # Runtime rule (from user spec):
    #   "pymoo is >2× slower per generation WITHOUT compensating
    #    quality/feasibility gains"
    # So: runtime_ok if pymoo ≤ 2× deap sec/gen,
    #     OR pymoo achieves ≥2× better hard violations (compensating gain).
    soft_ok = med_soft_pymoo <= med_soft_deap
    ttf_ok = med_ttf_pymoo <= med_ttf_deap
    raw_runtime_ok = med_spg_pymoo <= 2.0 * med_spg_deap
    quality_compensates = med_hard_pymoo < 0.5 * med_hard_deap  # ≥2× better hard
    runtime_ok = raw_runtime_ok or quality_compensates

    go = (soft_ok or ttf_ok) and runtime_ok
    verdict["go"] = go
    verdict["soft_ok"] = soft_ok
    verdict["ttf_ok"] = ttf_ok
    verdict["runtime_ok"] = runtime_ok
    verdict["raw_runtime_ok"] = raw_runtime_ok
    verdict["quality_compensates"] = quality_compensates

    print(f"\n{'=' * 70}")
    status = "GO" if go else "NO-GO"
    print(f"  VERDICT: **{status}**")
    print(f"  soft_ok={soft_ok}  ttf_ok={ttf_ok}  runtime_ok={runtime_ok}")
    if not raw_runtime_ok and quality_compensates:
        print(
            f"  (pymoo {med_spg_pymoo:.1f}s/gen vs deap {med_spg_deap:.1f}s/gen — "
            f"compensated by {med_hard_pymoo:.0f} vs {med_hard_deap:.0f} hard violations)"
        )
    print(f"{'=' * 70}")

    return verdict


# =====================================================================
#  Plots (matplotlib only)
# =====================================================================


def make_plots(rows: list[dict], results_dir: Path) -> None:
    """Generate comparison plots and save as PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots.")
        return

    grouped = group_rows(rows)
    solvers = sorted(grouped.keys())
    colors = {"pymoo": "#2196F3", "deap": "#FF9800"}

    # --- Helper: per-gen arrays averaged across seeds ---
    def _avg_metric(solver: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (gens, mean, std) arrays for a given metric, averaged across seeds."""
        all_seeds = grouped.get(solver, {})
        if not all_seeds:
            return np.array([]), np.array([]), np.array([])
        # Align on gen numbers
        gen_vals: dict[int, list[float]] = defaultdict(list)
        for seed_rows in all_seeds.values():
            for r in seed_rows:
                v = r.get(key)
                if v is not None:
                    gen_vals[r["gen"]].append(float(v))
        gens_sorted = sorted(gen_vals.keys())
        means = [np.mean(gen_vals[g]) for g in gens_sorted]
        stds = [np.std(gen_vals[g]) for g in gens_sorted]
        return np.array(gens_sorted), np.array(means), np.array(stds)

    # ---- Plot 1: best_hard over generations ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for solver in solvers:
        gens, means, stds = _avg_metric(solver, "best_hard")
        if len(gens) == 0:
            continue
        ax.plot(
            gens, means, label=solver, color=colors.get(solver, "gray"), linewidth=2
        )
        ax.fill_between(
            gens,
            means - stds,
            means + stds,
            color=colors.get(solver, "gray"),
            alpha=0.15,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Hard Violations")
    ax.set_title("Hard Constraint Violations (mean +/- std across seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "best_hard_over_gens.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {results_dir / 'best_hard_over_gens.png'}")

    # ---- Plot 2: best_soft over generations ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for solver in solvers:
        gens, means, stds = _avg_metric(solver, "best_soft")
        if len(gens) == 0:
            continue
        ax.plot(
            gens, means, label=solver, color=colors.get(solver, "gray"), linewidth=2
        )
        ax.fill_between(
            gens,
            means - stds,
            means + stds,
            color=colors.get(solver, "gray"),
            alpha=0.15,
        )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Soft Penalty")
    ax.set_title("Soft Penalty (mean +/- std across seeds)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "best_soft_over_gens.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {results_dir / 'best_soft_over_gens.png'}")

    # ---- Plot 3: time per gen (box / bar) ----
    fig, ax = plt.subplots(figsize=(6, 4))
    tpg_by_solver: dict[str, list[float]] = {}
    for solver in solvers:
        vals: list[float] = []
        for seed_rows in grouped.get(solver, {}).values():
            for r in seed_rows:
                if r["gen"] > 0:
                    vals.append(r["time_per_gen"])
        tpg_by_solver[solver] = vals

    positions = list(range(len(solvers)))
    bp_data = [tpg_by_solver.get(s, [0]) for s in solvers]
    bp = ax.boxplot(bp_data, positions=positions, widths=0.5, patch_artist=True)
    for patch, solver in zip(bp["boxes"], solvers):
        patch.set_facecolor(colors.get(solver, "gray"))
        patch.set_alpha(0.6)
    ax.set_xticks(positions)
    ax.set_xticklabels(solvers)
    ax.set_ylabel("Time per Generation (s)")
    ax.set_title("Per-Generation Runtime")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(results_dir / "time_per_gen.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {results_dir / 'time_per_gen.png'}")

    # ---- Plot 4: cv_min over gens (pymoo only) ----
    if "pymoo" in grouped:
        fig, ax = plt.subplots(figsize=(8, 5))
        for seed, seed_rows in sorted(grouped["pymoo"].items()):
            gen_list = [r["gen"] for r in seed_rows if r.get("cv_min") is not None]
            cv_mins = [r["cv_min"] for r in seed_rows if r.get("cv_min") is not None]
            ax.plot(gen_list, cv_mins, label=f"seed={seed}", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Generation")
        ax.set_ylabel("min CV (sum of G)")
        ax.set_title("pymoo Constraint Violation Convergence")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(results_dir / "pymoo_cv_convergence.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {results_dir / 'pymoo_cv_convergence.png'}")


# =====================================================================
#  CLI
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Report on DEAP vs pymoo benchmark results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/bench_compare",
        help="Directory containing runs.jsonl and summary.json",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows, summary = load_data(results_dir)
    print(f"Loaded {len(rows)} per-gen rows from {results_dir / 'runs.jsonl'}")

    verdict = print_report(rows, summary)

    if not args.no_plots:
        print("\nGenerating plots...")
        make_plots(rows, results_dir)

    # Save verdict
    verdict_path = results_dir / "verdict.json"
    with open(verdict_path, "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nVerdict saved -> {verdict_path}")


if __name__ == "__main__":
    main()
