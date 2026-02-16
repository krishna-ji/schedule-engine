"""RepairPipeline — unified interface for repair operations.

Provides a single class that:
- Wraps the existing repair operators from ``ga/operators/repair_engine.py``
- Uses the Protocol already defined there (``RepairOperator``)
- Supports pluggable selection policies (round-robin, epsilon-greedy)
- Returns structured ``RepairStats`` results

Usage::

    from src.ga import RepairPipeline

    pipeline = RepairPipeline.default(context, fitness_fn)
    repaired, stats = pipeline.repair(individual, max_iterations=10)
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import SchedulingContext

from src.ga.repair.engine import (
    EpsilonGreedyPolicy,
    RepairCandidate,
    RepairOperator,
    RepairStats,
    RepairStepResult,
    RoundRobinPolicy,
    ViolationState,
)

logger = logging.getLogger(__name__)

__all__ = ["RepairPipeline"]

FitnessFn = Callable[[list["SessionGene"]], tuple[float, float]]


class RepairPipeline:
    """Centralised repair interface.

    Composes N ``RepairOperator`` instances with a selection policy.
    Each ``repair()`` call runs up to *max_iterations* steps, selecting
    an operator, generating candidates, and applying the best one if
    it improves fitness.

    Parameters
    ----------
    operators : list[RepairOperator]
        Repair operator instances.
    context : SchedulingContext
        Scheduling data (courses, groups, rooms, etc.).
    fitness_fn : FitnessFn
        ``(individual) -> (hard, soft)`` scorer.
    policy : str
        ``"round_robin"`` or ``"epsilon_greedy"`` (default ``"round_robin"``).
    epsilon : float
        Exploration rate for epsilon-greedy policy (default 0.1).
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        operators: list[RepairOperator],
        context: SchedulingContext,
        fitness_fn: FitnessFn,
        *,
        policy: str = "round_robin",
        epsilon: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self._operators = {op.name: op for op in operators}
        self._context = context
        self._fitness_fn = fitness_fn
        self._rng = random.Random(seed)

        names = [op.name for op in operators]
        if policy == "epsilon_greedy":
            self._policy: RoundRobinPolicy | EpsilonGreedyPolicy = EpsilonGreedyPolicy(
                names, epsilon
            )
        else:
            self._policy = RoundRobinPolicy(names)

    # Public API

    def repair(
        self,
        individual: list[SessionGene],
        *,
        max_iterations: int = 10,
        max_candidates: int = 5,
    ) -> tuple[list[SessionGene], RepairStats]:
        """Apply repair operators to an individual.

        Parameters
        ----------
        individual : list[SessionGene]
            The chromosome to repair (modified in-place AND returned).
        max_iterations : int
            Maximum repair steps to attempt.
        max_candidates : int
            Candidates per operator per step.

        Returns
        -------
        tuple[list[SessionGene], RepairStats]
            The (possibly improved) individual and detailed stats.
        """
        stats = RepairStats()
        current_fitness = self._fitness_fn(individual)

        for _ in range(max_iterations):
            # Select operator
            if isinstance(self._policy, EpsilonGreedyPolicy):
                op_name = self._policy.select(stats.by_operator, self._rng)
            else:
                op_name = self._policy.select()

            op = self._operators.get(op_name)
            if op is None:
                continue

            # Build violation state
            state = self._build_state(individual, current_fitness)

            # Get candidates
            candidates = op.propose(
                individual,
                self._context,
                state,
                max_candidates,
                self._rng,
            )

            if not candidates:
                stats.record(
                    RepairStepResult(
                        applied=False,
                        operator=op_name,
                        delta_hard=0.0,
                        delta_soft=0.0,
                        before=current_fitness,
                        after=current_fitness,
                    )
                )
                continue

            # Try best candidate
            best_candidate = candidates[0]
            applied = self._apply_candidate(individual, best_candidate)

            if applied:
                new_fitness = self._fitness_fn(individual)
                delta_hard = current_fitness[0] - new_fitness[0]
                delta_soft = current_fitness[1] - new_fitness[1]

                # Accept if improvement (lexicographic: hard first)
                if new_fitness[0] < current_fitness[0] or (
                    new_fitness[0] == current_fitness[0]
                    and new_fitness[1] < current_fitness[1]
                ):
                    current_fitness = new_fitness
                    result = RepairStepResult(
                        applied=True,
                        operator=op_name,
                        delta_hard=delta_hard,
                        delta_soft=delta_soft,
                        before=current_fitness,
                        after=new_fitness,
                    )
                else:
                    # Revert
                    self._revert_candidate(individual, best_candidate)
                    result = RepairStepResult(
                        applied=False,
                        operator=op_name,
                        delta_hard=0.0,
                        delta_soft=0.0,
                        before=current_fitness,
                        after=current_fitness,
                    )
            else:
                result = RepairStepResult(
                    applied=False,
                    operator=op_name,
                    delta_hard=0.0,
                    delta_soft=0.0,
                    before=current_fitness,
                    after=current_fitness,
                )

            stats.record(result)

        return individual, stats

    # Factory

    @classmethod
    def default(
        cls,
        context: SchedulingContext,
        fitness_fn: FitnessFn,
        *,
        policy: str = "round_robin",
        seed: int | None = None,
    ) -> RepairPipeline:
        """Create a pipeline with the standard set of repair operators.

        Imports and instantiates operators from ``repair_engine.py``.
        """
        try:
            from src.ga.repair.engine import (
                MoveTimeOperator,
                ReassignInstructorOperator,
                SwapRoomOperator,
            )

            operators: list[RepairOperator] = [
                MoveTimeOperator(),
                SwapRoomOperator(),
                ReassignInstructorOperator(),
            ]
        except ImportError:
            logger.warning(
                "Could not import default repair operators; "
                "pipeline will have no operators."
            )
            operators = []

        return cls(
            operators=operators,
            context=context,
            fitness_fn=fitness_fn,
            policy=policy,
            seed=seed,
        )

    # Internal helpers

    def _build_state(
        self,
        individual: list[SessionGene],
        fitness: tuple[float, float],
    ) -> ViolationState:
        """Build a ViolationState for operator consumption."""
        return ViolationState(
            hard=fitness[0],
            soft=fitness[1],
            gene_scores=[0] * len(individual),
            gene_order=list(range(len(individual))),
            instructor_counts={},
            room_counts={},
            group_counts={},
            available_quanta=set(self._context.available_quanta),
        )

    @staticmethod
    def _apply_candidate(
        individual: list[SessionGene],
        candidate: RepairCandidate,
    ) -> bool:
        """Apply a repair candidate to the individual.

        Returns True if the candidate was applied.
        """
        if candidate.gene_idx >= len(individual):
            return False

        gene = individual[candidate.gene_idx]

        # Save original for potential revert
        gene._repair_backup = (  # type: ignore[attr-defined]
            gene.start_quanta,
            gene.room_id,
            gene.instructor_id,
        )

        if candidate.new_start is not None:
            gene.start_quanta = candidate.new_start
        if candidate.new_room_id is not None:
            gene.room_id = candidate.new_room_id
        if candidate.new_instructor_id is not None:
            gene.instructor_id = candidate.new_instructor_id

        return True

    @staticmethod
    def _revert_candidate(
        individual: list[SessionGene],
        candidate: RepairCandidate,
    ) -> None:
        """Revert a previously applied repair candidate."""
        gene = individual[candidate.gene_idx]
        backup = getattr(gene, "_repair_backup", None)
        if backup:
            gene.start_quanta, gene.room_id, gene.instructor_id = backup
