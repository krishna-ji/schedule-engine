"""Notebook helper module - DRY utilities for experiment notebooks.

This module provides reusable functions for Jupyter notebooks running
timetabling experiments (Modes A-E). Keeps notebooks clean and focused
on mode-specific configuration.

Usage:
    from src.notebooks import load_data, run_nsga2, plot_results
    from src.notebooks import export_full_results  # For production outputs
"""

from __future__ import annotations

from src.notebooks.data_loader import ScheduleData, load_data
from src.notebooks.evaluation import (
    ConstraintResult,
    create_evaluator,
    evaluate_constraints,
    get_constraint_breakdown,
)
from src.notebooks.evolution import (
    EvolutionConfig,
    EvolutionStats,
    get_best_individual,
    get_pareto_front,
    run_nsga2,
    setup_deap,
)
from src.notebooks.export import (
    create_output_structure,
    export_comparison_results,
    export_constraint_plots,
    export_csv_data,
    export_full_results,
    export_nsga_plots,
    export_schedule_json,
)
from src.notebooks.heuristics import (
    HEURISTICS,
    AdaptiveSelector,
    RoundRobinSelector,
    fix_group_conflicts,
    fix_instructor_conflicts,
    fix_room_conflicts,
)
from src.notebooks.memetic import local_search_individual, memetic_generation_callback
from src.notebooks.operators import (
    course_aware_crossover,
    smart_mutation,
    uniform_mutation,
)
from src.notebooks.population import (
    create_random_individual,
    create_smart_individual,
    get_subsession_durations,
)
from src.notebooks.rl_helper import SimpleRLSelector, load_trained_agent
from src.notebooks.visualization import (
    plot_comparison,
    plot_constraint_breakdown,
    plot_convergence,
    print_summary,
)

__all__ = [
    # Data loading
    "load_data",
    "ScheduleData",
    # Population
    "create_random_individual",
    "create_smart_individual",
    "get_subsession_durations",
    # Operators
    "course_aware_crossover",
    "smart_mutation",
    "uniform_mutation",
    # Evaluation
    "create_evaluator",
    "evaluate_constraints",
    "get_constraint_breakdown",
    "ConstraintResult",
    # Evolution
    "run_nsga2",
    "setup_deap",
    "get_best_individual",
    "get_pareto_front",
    "EvolutionConfig",
    "EvolutionStats",
    # Export (full production outputs)
    "export_full_results",
    "export_schedule_json",
    "export_csv_data",
    "export_constraint_plots",
    "export_nsga_plots",
    "export_comparison_results",
    "create_output_structure",
    # Memetic
    "local_search_individual",
    "memetic_generation_callback",
    # Heuristics
    "HEURISTICS",
    "RoundRobinSelector",
    "AdaptiveSelector",
    "fix_instructor_conflicts",
    "fix_room_conflicts",
    "fix_group_conflicts",
    # RL
    "SimpleRLSelector",
    "load_trained_agent",
    # Visualization
    "plot_convergence",
    "plot_constraint_breakdown",
    "plot_comparison",
    "print_summary",
]
