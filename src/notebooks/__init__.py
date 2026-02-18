"""
Notebook Support Module.

Provides reusable components for interactive Jupyter notebooks.

Modules:
    - core: Data loading, individual creation, DEAP setup, evolution utilities
    - viz: Plotting and visualization functions
    - strategies: Heuristic selection strategies (local search, round-robin, adaptive)
    - export: Production export utilities
"""

from __future__ import annotations

from src.notebooks.core import (
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
from src.notebooks.export import (
    export_full_results,
    export_schedule_json,
    export_stats_csv,
)
from src.notebooks.strategies import (
    AdaptiveSelector,
    RoundRobinSelector,
    local_search_individual,
)
from src.notebooks.viz import (
    plot_convergence,
    plot_constraint_breakdown,
    print_summary,
)

__all__ = [
    "AdaptiveSelector",
    "EvolutionConfig",
    "EvolutionStats",
    "NotebookData",
    "RoundRobinSelector",
    "create_evaluator",
    "create_random_individual",
    "export_full_results",
    "export_schedule_json",
    "export_stats_csv",
    "get_best_individual",
    "get_constraint_breakdown",
    "load_data",
    "local_search_individual",
    "plot_constraint_breakdown",
    "plot_convergence",
    "print_summary",
    "run_nsga2",
    "setup_deap",
]
