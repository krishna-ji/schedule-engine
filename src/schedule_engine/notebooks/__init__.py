"""
Notebook Support Module.

Provides reusable components for interactive Jupyter notebooks,
enabling DRY architecture across Mode A-E experiments.

Modules:
    - core: Data loading, individual creation, DEAP setup, evolution utilities
    - viz: Plotting and visualization functions
    - strategies: Heuristic selection strategies (local search, round-robin, adaptive, RL)
    - export: Production export utilities

Usage in notebooks:
    from schedule_engine.notebooks.core import load_data, create_random_individual
    from schedule_engine.notebooks.viz import plot_convergence, print_summary
    from schedule_engine.notebooks.strategies import local_search_individual
"""

from __future__ import annotations

from schedule_engine.notebooks.core import (
    EvolutionConfig,
    EvolutionStats,
    NotebookData,
    create_evaluator,
    create_random_individual,
    get_best_individual,
    get_constraint_breakdown,
    load_data,
    run_nsga2,
    setup_deap,
)
from schedule_engine.notebooks.export import (
    export_full_results,
    export_schedule_json,
    export_stats_csv,
)
from schedule_engine.notebooks.strategies import (
    AdaptiveSelector,
    RoundRobinSelector,
    SimpleRLSelector,
    load_trained_agent,
    local_search_individual,
)
from schedule_engine.notebooks.rl_helpers import (
    EvaluationResult,
    build_notebook_config,
    create_env,
    evaluate_agent,
    load_context,
    run_ablation,
    set_global_seed,
    train_agent,
)
from schedule_engine.notebooks.viz import (
    plot_convergence,
    plot_constraint_breakdown,
    print_summary,
)

__all__ = [
    "NotebookData",
    "EvolutionConfig",
    "EvolutionStats",
    "load_data",
    "create_random_individual",
    "create_evaluator",
    "setup_deap",
    "run_nsga2",
    "get_best_individual",
    "get_constraint_breakdown",
    "plot_convergence",
    "plot_constraint_breakdown",
    "print_summary",
    "export_schedule_json",
    "export_stats_csv",
    "export_full_results",
    "local_search_individual",
    "RoundRobinSelector",
    "AdaptiveSelector",
    "SimpleRLSelector",
    "load_trained_agent",
    "EvaluationResult",
    "build_notebook_config",
    "load_context",
    "create_env",
    "train_agent",
    "evaluate_agent",
    "run_ablation",
    "set_global_seed",
]
