"""PopulationFactory — single entry point for individual/population creation.

Unifies:
- ``ga/population.py`` (population generation + utilities)
- ``ga/run_helpers.py::create_random_individual()``

Usage::

    from src.ga import PopulationFactory

    factory = PopulationFactory(context)
    pop = factory.create_population(50, strategy="smart")
    ind = factory.random_individual()
    ind = factory.greedy_individual()
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext

__all__ = ["PopulationFactory"]


class PopulationFactory:
    """Unified population creation.

    Wraps the 3 scattered population-creation codepaths behind a single
    class.  Callers no longer need to know which module to import.

    Parameters
    ----------
    context : SchedulingContext
        The scheduling universe (courses, groups, instructors, rooms, etc.).
    parallel : bool
        Use multiprocessing for population generation (default True).
    """

    def __init__(
        self,
        context: SchedulingContext,
        *,
        parallel: bool = True,
    ) -> None:
        self._context = context
        self._parallel = parallel

    @property
    def context(self) -> SchedulingContext:
        return self._context

    # Individual creation

    def random_individual(self, *, conflict_aware: bool = True) -> list[SessionGene]:
        """Create a single random individual.

        Parameters
        ----------
        conflict_aware : bool
            If True (default), use the conflict-avoiding gene placement
            strategy from ``population.py``.  If False, use pure random
            placement (faster but lower quality).
        """
        if conflict_aware:
            from src.ga.core.population import (
                generate_course_group_aware_population,
            )

            pop = generate_course_group_aware_population(
                n=1,
                context=self._context,
                parallel=False,
            )
        else:
            from src.ga.core.population import (
                generate_pure_random_population,
            )

            pop = generate_pure_random_population(
                n=1,
                context=self._context,
            )
        return pop[0] if pop else []

    def greedy_individual(self) -> list[SessionGene]:
        """Create one individual using greedy construction.

        Uses hybrid initialization from ``population.generate_hybrid_population``
        which places the hardest-to-schedule courses first.
        """
        from src.ga.core.population import generate_hybrid_population

        # 1 hybrid = 1 greedy
        pop = generate_hybrid_population(n=1, context=self._context)
        return pop[0] if pop else []

    # Population creation

    def create_population(
        self,
        n: int,
        *,
        strategy: str = "smart",
    ) -> list[list[SessionGene]]:
        """Create a full population.

        Parameters
        ----------
        n : int
            Population size.
        strategy : str
            ``"smart"``   — conflict-aware random (default, best quality)
            ``"hybrid"``  — mix of greedy + random (good diversity)
            ``"random"``  — pure random (fastest, lowest quality)
        """
        if strategy == "hybrid":
            return self._hybrid_population(n)
        if strategy == "random":
            return self._pure_random_population(n)
        return self._smart_population(n)

    # Internal strategies

    def _smart_population(self, n: int) -> list[list[SessionGene]]:
        """Conflict-aware population (delegates to population.py)."""
        from src.ga.core.population import (
            generate_course_group_aware_population,
        )

        return generate_course_group_aware_population(
            n=n,
            context=self._context,
            parallel=self._parallel,
        )

    def _hybrid_population(self, n: int) -> list[list[SessionGene]]:
        """Mix greedy + random (delegates to population.py)."""
        from src.ga.core.population import generate_hybrid_population

        return generate_hybrid_population(n=n, context=self._context)

    def _pure_random_population(self, n: int) -> list[list[SessionGene]]:
        """Pure random population (delegates to population.py)."""
        from src.ga.core.population import generate_pure_random_population

        return generate_pure_random_population(
            n=n,
            context=self._context,
        )
