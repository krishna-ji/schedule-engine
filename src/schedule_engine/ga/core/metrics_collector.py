"""Metrics collection for GA evolution.

Extracted from GAScheduler to isolate the metrics-tracking concern.
Handles hypervolume, spacing, IGD, spread, feasibility rate, and
per-constraint breakdown tracking.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from deap import tools

from schedule_engine.ga.core.evaluator import evaluate, evaluate_detailed
from schedule_engine.ga.metrics.diversity import average_pairwise_diversity

if TYPE_CHECKING:
    from schedule_engine.ga.scheduler import GAMetrics

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and records per-generation evolution metrics.

    Parameters
    ----------
    metrics:
        The ``GAMetrics`` instance to accumulate into.
    config:
        The full config snapshot (``self._cfg`` in GAScheduler).
    context:
        ``SchedulingContext`` for fitness evaluation.
    hard_constraint_names / soft_constraint_names:
        Ordered lists of constraint names.
    toolbox:
        DEAP toolbox (for ``clone``).
    """

    def __init__(
        self,
        metrics: GAMetrics,
        context: Any,
        hard_constraint_names: list[str],
        soft_constraint_names: list[str],
        toolbox: Any,
        violation_heatmap: Any | None = None,
        constraint_logger: Any | None = None,
    ) -> None:
        self.metrics = metrics
        self.context = context
        self.hard_constraint_names = hard_constraint_names
        self.soft_constraint_names = soft_constraint_names
        self.toolbox = toolbox
        self.violation_heatmap = violation_heatmap
        self.constraint_logger = constraint_logger

        self._hypervolume_ref_point: Any | None = None
        self.all_time_best: Any | None = None
        self._cached_hard_details: dict[str, int] = {}
        self._cached_soft_details: dict[str, int] = {}

    # ------------------------------------------------------------------

    def track(
        self,
        gen: int,
        population: list,
        *,
        event_tracker: Any | None = None,
        best_individual: Any | None = None,
    ) -> None:
        """Record metrics for *gen*; mirrors ``GAScheduler._track_metrics``."""
        if gen == -1:
            return

        from schedule_engine.ga.metrics.convergence import (
            calculate_constraint_satisfaction_rate,
        )
        from schedule_engine.ga.metrics.hypervolume import (
            calculate_hypervolume,
            get_hypervolume_reference_point,
        )
        from schedule_engine.ga.metrics.pareto_metrics import (
            calculate_inverted_generational_distance,
            calculate_spacing,
            calculate_spread,
            get_pareto_front_size,
        )

        advanced_freq = 10
        is_tracked_gen = (
            gen == 0
            or gen == self.metrics.generation_times.__len__()  # last gen proxy
            or gen % advanced_freq == 0
        )

        # Validate & build fitness array
        for ind in population:
            if (
                not hasattr(ind.fitness, "values")
                or not ind.fitness.valid
                or len(ind.fitness.values) != 2
            ):
                fitness = evaluate(
                    ind,
                    self.context.courses,
                    self.context.instructors,
                    self.context.groups,
                    self.context.rooms,
                )
                ind.fitness.values = fitness

        fitness_array = np.array([ind.fitness.values for ind in population])
        self.metrics.hard_violations.append(float(fitness_array[:, 0].min()))
        self.metrics.soft_penalties.append(float(fitness_array[:, 1].min()))
        diversity = average_pairwise_diversity(population)
        self.metrics.diversity.append(diversity)

        hv = 0.0
        spacing = 0.0
        spread = 0.0

        if is_tracked_gen:
            if gen == 0:
                self._hypervolume_ref_point = get_hypervolume_reference_point(
                    population, margin=0.1
                )
            hv = calculate_hypervolume(population, self._hypervolume_ref_point)
            self.metrics.hypervolume.append(hv)
            spacing = calculate_spacing(population)
            self.metrics.spacing.append(spacing)
            pf_size = get_pareto_front_size(population)
            self.metrics.pareto_front_size.append(pf_size)
            feas_rate = calculate_constraint_satisfaction_rate(population)
            self.metrics.feasibility_rate.append(feas_rate)
            if gen == 0:
                pareto_front = tools.sortNondominated(
                    population, len(population), first_front_only=True
                )[0]
                self.metrics.reference_front = list(pareto_front)
            if self.metrics.reference_front:
                igd = calculate_inverted_generational_distance(
                    population, self.metrics.reference_front
                )
                self.metrics.igd.append(igd)
            else:
                self.metrics.igd.append(0.0)
            spread = calculate_spread(population)
            self.metrics.spread.append(spread)
        else:
            self.metrics.hypervolume.append(
                self.metrics.hypervolume[-1] if self.metrics.hypervolume else 0.0
            )
            hv = self.metrics.hypervolume[-1]
            self.metrics.spacing.append(
                self.metrics.spacing[-1] if self.metrics.spacing else 0.0
            )
            spacing = self.metrics.spacing[-1]
            self.metrics.pareto_front_size.append(
                self.metrics.pareto_front_size[-1]
                if self.metrics.pareto_front_size
                else 0
            )
            self.metrics.feasibility_rate.append(
                self.metrics.feasibility_rate[-1]
                if self.metrics.feasibility_rate
                else 0.0
            )
            self.metrics.igd.append(
                self.metrics.igd[-1] if self.metrics.igd else 0.0
            )
            self.metrics.spread.append(
                self.metrics.spread[-1] if self.metrics.spread else 0.0
            )
            spread = self.metrics.spread[-1]

        # Detailed breakdown
        best = best_individual if best_individual is not None else tools.selBest(population, 1)[0]
        hard_details, soft_details = evaluate_detailed(
            best,
            self.context.courses,
            self.context.instructors,
            self.context.groups,
            self.context.rooms,
        )
        self._cached_hard_details = hard_details
        self._cached_soft_details = soft_details

        # Track all-time best
        if (
            self.all_time_best is None
            or best.fitness.values[0] < self.all_time_best.fitness.values[0]
        ):
            self.all_time_best = self.toolbox.clone(best)
        elif (
            best.fitness.values[0] == self.all_time_best.fitness.values[0]
            and best.fitness.values[1] < self.all_time_best.fitness.values[1]
        ):
            self.all_time_best = self.toolbox.clone(best)

        for name in self.hard_constraint_names:
            self.metrics.detailed_hard[name].append(hard_details[name])
        for name in self.soft_constraint_names:
            self.metrics.detailed_soft[name].append(soft_details[name])

        # Violation heatmap
        if self.violation_heatmap and gen >= 0:
            from schedule_engine.ga.metrics.violation_recorder import (
                record_violations_to_heatmap,
            )
            record_violations_to_heatmap(best, self.context, self.violation_heatmap)
            self.violation_heatmap.record_generation(gen)

        # Constraint logger
        if self.constraint_logger:
            repair_stats = {}
            if 0 <= gen < len(self.metrics.repair_stats):
                repair_stats = self.metrics.repair_stats[gen]

            events: list = []
            if event_tracker and event_tracker.has_events():
                events = event_tracker.get_events()

            notes = ""
            if best.fitness.values[0] == 0:
                notes = "Perfect solution"
                if event_tracker and "perfect_solution" not in events:
                    event_tracker.add("perfect_solution")
                    events = event_tracker.get_events()

            self.constraint_logger.log_generation(
                generation=gen,
                hard_total=best.fitness.values[0],
                soft_total=best.fitness.values[1],
                hard_breakdown=hard_details,
                soft_breakdown=soft_details,
                diversity=diversity,
                time_seconds=0.0,
                hypervolume=hv,
                spacing=spacing,
                igd=self.metrics.igd[-1] if self.metrics.igd else 0.0,
                spread=spread,
                repair_stats=repair_stats,
                events=events,
                notes=notes,
            )
