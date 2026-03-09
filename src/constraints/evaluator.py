"""Unified Evaluator — the single way to evaluate fitness.

Replaces:
- ``ga/evaluator/fitness.py``
- ``ga/evaluator/detailed_fitness.py``
- ``ga/run_helpers.py::create_evaluator()``
- ``ga/run_helpers.py::get_constraint_breakdown()``
- ``constraints/all_constraints.py`` (evaluate_all, evaluate_hard_constraints, etc.)

Usage::

    from src.constraints import Evaluator

    evaluator = Evaluator()  # uses default constraints
    hard, soft = evaluator.fitness(genes, context, qts)
    breakdown = evaluator.breakdown(genes, context, qts)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.constraints.constraints import ALL_CONSTRAINTS, Constraint
from src.domain.timetable import Timetable

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext
    from src.io.time_system import QuantumTimeSystem

__all__ = ["Evaluator"]


class Evaluator:
    """Single, authoritative fitness evaluator.

    Accepts either raw genes or a pre-built ``Timetable`` for both
    aggregate fitness and per-constraint breakdown.

    Args:
        constraints: Custom constraint list. Defaults to all hard + soft
            constraints.
    """

    def __init__(self, constraints: list[Constraint] | None = None) -> None:
        if constraints is None:
            constraints = list(ALL_CONSTRAINTS)
        self.hard = [c for c in constraints if c.kind == "hard"]
        self.soft = [c for c in constraints if c.kind == "soft"]
        self._all = self.hard + self.soft

    # Core: fitness

    def fitness(
        self,
        genes: list[SessionGene],
        context: SchedulingContext,
        qts: QuantumTimeSystem | None = None,
    ) -> tuple[float, float]:
        """Return ``(hard_penalty, soft_penalty)``."""
        tt = Timetable(genes, context, qts)
        return self.fitness_from_timetable(tt)

    def fitness_from_timetable(self, tt: Timetable) -> tuple[float, float]:
        """Evaluate an already-constructed Timetable."""
        hard = sum(c.weight * c.evaluate(tt) for c in self.hard)
        soft = sum(c.weight * c.evaluate(tt) for c in self.soft)
        return hard, soft

    # Core: breakdown

    def breakdown(
        self,
        genes: list[SessionGene],
        context: SchedulingContext,
        qts: QuantumTimeSystem | None = None,
    ) -> dict[str, float]:
        """Return ``{constraint_name: penalty}`` for every constraint."""
        tt = Timetable(genes, context, qts)
        return self.breakdown_from_timetable(tt)

    def breakdown_from_timetable(self, tt: Timetable) -> dict[str, float]:
        """Per-constraint breakdown from an already-constructed Timetable."""
        return {c.name: c.evaluate(tt) for c in self._all}

    # Convenience: hard/soft breakdowns separately

    def hard_breakdown(self, tt: Timetable) -> dict[str, float]:
        """Hard constraint breakdown only."""
        return {c.name: c.evaluate(tt) for c in self.hard}

    def soft_breakdown(self, tt: Timetable) -> dict[str, float]:
        """Soft constraint breakdown only."""
        return {c.name: c.evaluate(tt) for c in self.soft}

    # Convenience: evaluate with a summary like evaluate_all()

    def evaluate_all(
        self, tt: Timetable
    ) -> tuple[float, float, dict[str, float], dict[str, float]]:
        """Full evaluation with breakdowns.

        Returns ``(hard_total, soft_total, hard_breakdown, soft_breakdown)``.
        Drop-in replacement for ``all_constraints.evaluate_all()``.
        """
        hb = self.hard_breakdown(tt)
        sb = self.soft_breakdown(tt)
        return sum(hb.values()), sum(sb.values()), hb, sb
