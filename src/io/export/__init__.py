"""Export utilities: JSON, PDF, and plot generation.

Usage:
    from src.io.export import export_everything, plot_pareto_front_from_F
"""

from __future__ import annotations

from src.io.export.exporter import export_everything
from src.io.export.plot_convergence import (
    plot_constraint_satisfaction_evolution,
    plot_convergence_dashboard,
    plot_convergence_rate,
    plot_multi_metric_convergence,
)
from src.io.export.plot_detailed_constraints import (
    plot_constraint_summary,
    plot_individual_hard_constraints,
    plot_individual_soft_constraints,
)
from src.io.export.plot_hypervolume import plot_hypervolume_trend
from src.io.export.plot_igd import plot_igd_trend
from src.io.export.plot_memetic import (
    plot_pareto_repair_shift,
    plot_repair_interventions,
)
from src.io.export.plot_pareto_evolution import plot_pareto_evolution
from src.io.export.plot_spacing import plot_spacing_trend
from src.io.export.plotdiversity import plot_diversity_trend
from src.io.export.plothard import plot_hard_constraint_violation_over_generation
from src.io.export.plotpareto import plot_pareto_front_from_F
from src.io.export.plotsoft import plot_soft_constraint_violation_over_generation
from src.io.export.violation_reporter import generate_violation_report

__all__ = [
    # Main exports
    "export_everything",
    # Reports
    "generate_violation_report",
    "plot_constraint_satisfaction_evolution",
    "plot_constraint_summary",
    "plot_convergence_dashboard",
    "plot_convergence_rate",
    "plot_diversity_trend",
    "plot_hard_constraint_violation_over_generation",
    "plot_hypervolume_trend",
    "plot_igd_trend",
    "plot_individual_hard_constraints",
    "plot_individual_soft_constraints",
    "plot_multi_metric_convergence",
    # Plots
    "plot_pareto_evolution",
    "plot_pareto_front_from_F",
    "plot_pareto_repair_shift",
    "plot_repair_interventions",
    "plot_soft_constraint_violation_over_generation",
    "plot_spacing_trend",
]
