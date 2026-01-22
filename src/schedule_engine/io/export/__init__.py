"""Export utilities: JSON, PDF, and plot generation.

Usage:
    from schedule_engine.io.export import export_everything, plot_pareto_front
"""

from __future__ import annotations

from schedule_engine.io.export.exporter import export_everything
from schedule_engine.io.export.plot_convergence import (
    plot_constraint_satisfaction_evolution,
    plot_convergence_dashboard,
    plot_convergence_rate,
    plot_multi_metric_convergence,
)
from schedule_engine.io.export.plot_detailed_constraints import (
    plot_constraint_summary,
    plot_individual_hard_constraints,
    plot_individual_soft_constraints,
)
from schedule_engine.io.export.plot_hypervolume import plot_hypervolume_trend
from schedule_engine.io.export.plot_spacing import plot_spacing_trend
from schedule_engine.io.export.plotdiversity import plot_diversity_trend
from schedule_engine.io.export.plothard import plot_hard_constraint_violation_over_generation
from schedule_engine.io.export.plotpareto import plot_pareto_front
from schedule_engine.io.export.plotsoft import plot_soft_constraint_violation_over_generation
from schedule_engine.io.export.violation_reporter import generate_violation_report

__all__ = [
    # Main exports
    "export_everything",
    # Plots
    "plot_pareto_front",
    "plot_multi_metric_convergence",
    "plot_convergence_dashboard",
    "plot_convergence_rate",
    "plot_constraint_satisfaction_evolution",
    "plot_hard_constraint_violation_over_generation",
    "plot_soft_constraint_violation_over_generation",
    "plot_diversity_trend",
    "plot_hypervolume_trend",
    "plot_spacing_trend",
    "plot_individual_hard_constraints",
    "plot_individual_soft_constraints",
    "plot_constraint_summary",
    # Reports
    "generate_violation_report",
]
