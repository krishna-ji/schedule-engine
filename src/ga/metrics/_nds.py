"""Non-dominated sorting via pymoo (replaces deap.tools.sortNondominated).

Provides a drop-in replacement for the DEAP non-dominated sorting used
throughout metrics, visualization, and export code.

The ``get_pareto_front`` function accepts a list of objects with
``fitness.values`` (the DEAP convention still used in many places) **or**
plain (hard, soft) tuples, and returns the first Pareto front as a list.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

if TYPE_CHECKING:
    from collections.abc import Callable


def _default_fitness(ind: Any) -> tuple[float, ...]:
    """Extract fitness using the ``fitness.values`` protocol."""
    return ind.fitness.values  # type: ignore[no-any-return]


def get_pareto_front(
    population: list,
    fitness_fn: Callable[[Any], tuple[float, ...]] | None = None,
) -> list:
    """Return the first Pareto front from *population*.

    This is a drop-in replacement for::

        deap.tools.sortNondominated(population, len(population),
                                     first_front_only=True)[0]

    Parameters
    ----------
    population:
        Sequence of individuals.
    fitness_fn:
        Callable ``ind -> (obj1, obj2, …)``.  Defaults to
        ``ind.fitness.values`` for backwards-compatibility with code that
        still passes DEAP-style individuals.

    Returns
    -------
    list
        Individuals belonging to the first non-dominated front, in original
        order.
    """
    if not population:
        return []

    if fitness_fn is None:
        fitness_fn = _default_fitness

    F = np.array([fitness_fn(ind) for ind in population])
    nds = NonDominatedSorting()
    fronts = nds.do(F)
    return [population[i] for i in fronts[0]]
