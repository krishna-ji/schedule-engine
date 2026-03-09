"""
RL-ready repair engine with lexicographic scoring and domain-safe operators.

Design goals:
- Domain safety: never change course/group structure; enforce room type suitability.
- Lexicographic scoring: prioritize hard violations, then soft penalties.
- RL compatibility: operators are actions, policy is pluggable, step returns reward info.
"""

from __future__ import annotations

import logging
import random
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from src.domain.gene import SessionGene

if TYPE_CHECKING:
    from src.domain.types import SchedulingContext

FitnessFn = Callable[[list[SessionGene]], tuple[float, float]]


@dataclass
class RepairCandidate:
    """Proposed change to a single gene."""

    gene_idx: int
    new_start: int | None = None
    new_room_id: str | None = None
    new_instructor_id: str | None = None


@dataclass
class RepairStepResult:
    """Result of a single repair step (one operator application)."""

    applied: bool
    operator: str
    delta_hard: float
    delta_soft: float
    before: tuple[float, float]
    after: tuple[float, float]


@dataclass
class RepairStats:
    """Aggregate statistics for a repair call."""

    steps: int = 0
    applied_steps: int = 0
    total_delta_hard: float = 0.0
    total_delta_soft: float = 0.0
    by_operator: dict[str, dict[str, float]] = field(default_factory=dict)
    step_results: list[RepairStepResult] = field(default_factory=list)

    def record(self, result: RepairStepResult) -> None:
        """Record a single repair step result into aggregate statistics."""
        self.steps += 1
        if result.applied:
            self.applied_steps += 1
            self.total_delta_hard += result.delta_hard
            self.total_delta_soft += result.delta_soft

        op_stats = self.by_operator.setdefault(
            result.operator,
            {
                "steps": 0,
                "applied": 0,
                "delta_hard": 0.0,
                "delta_soft": 0.0,
            },
        )
        op_stats["steps"] += 1
        if result.applied:
            op_stats["applied"] += 1
            op_stats["delta_hard"] += result.delta_hard
            op_stats["delta_soft"] += result.delta_soft

        self.step_results.append(result)


@dataclass
class ViolationState:
    """Compact state for operators and RL policies."""

    hard: float
    soft: float
    gene_scores: list[int]
    gene_order: list[int]
    instructor_counts: dict[int, dict[str, int]]
    room_counts: dict[int, dict[str, int]]
    group_counts: dict[int, dict[str, int]]
    available_quanta: set[int]


class RepairOperator(Protocol):
    """Interface for repair operators (RL action-compatible)."""

    name: str

    def propose(
        self,
        individual: list[SessionGene],
        context: SchedulingContext,
        state: ViolationState,
        max_candidates: int,
        rng: random.Random,
    ) -> list[RepairCandidate]: ...


class RoundRobinPolicy:
    """Simple round-robin policy (stable baseline)."""

    def __init__(self, operator_names: list[str]) -> None:
        self._names = operator_names
        self._idx = 0

    def select(self) -> str:
        name = self._names[self._idx]
        self._idx = (self._idx + 1) % len(self._names)
        return name


class EpsilonGreedyPolicy:
    """Epsilon-greedy policy using average improvement."""

    def __init__(self, operator_names: list[str], epsilon: float = 0.1) -> None:
        self._names = operator_names
        self._epsilon = epsilon

    def select(self, op_stats: dict[str, dict[str, float]], rng: random.Random) -> str:
        if rng.random() < self._epsilon:
            return rng.choice(self._names)

        best_name = self._names[0]
        best_score = float("-inf")
        for name in self._names:
            stats = op_stats.get(name)
            if not stats or stats["applied"] == 0:
                score = 0.0
            else:
                score = stats["delta_hard"] * 1000 + stats["delta_soft"]
            if score > best_score:
                best_score = score
                best_name = name
        return best_name


# Module-level cache for family map (hierarchy-aware group relationships)
_CACHED_FAMILY_MAP: dict[str, set[str]] | None = None


def _get_family_map() -> dict[str, set[str]]:
    """
    Get cached family map for hierarchy-aware group conflict detection.

    Maps each group_id to all related groups (self, siblings, parent).
    """
    global _CACHED_FAMILY_MAP

    if _CACHED_FAMILY_MAP is None:
        try:
            from src.ga.core.population import get_family_map_from_json

            _CACHED_FAMILY_MAP = get_family_map_from_json("data/Groups.json")
        except Exception:
            _CACHED_FAMILY_MAP = {}

    return _CACHED_FAMILY_MAP


def _build_counts(
    individual: list[SessionGene],
) -> tuple[
    dict[int, dict[str, int]],
    dict[int, dict[str, int]],
    dict[int, dict[str, int]],
]:
    """
    Build conflict count maps for violation scoring.

    Now hierarchy-aware: counts related groups (parent/siblings) as conflicts.
    """
    family_map = _get_family_map()

    instructor_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    room_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    group_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for gene in individual:
        for q in range(gene.start_quanta, gene.end_quanta):
            instructor_counts[q][gene.instructor_id] += 1
            room_counts[q][gene.room_id] += 1

            # Hierarchy-aware: count all related groups as occupied
            for gid in gene.group_ids:
                if family_map and gid in family_map:
                    for related_id in family_map[gid]:
                        group_counts[q][related_id] += 1
                else:
                    group_counts[q][gid] += 1

    return instructor_counts, room_counts, group_counts


def _is_instructor_qualified(
    instructor_id: str, gene: SessionGene, context: SchedulingContext
) -> bool:
    instructor = context.instructors.get(instructor_id)
    if instructor is None:
        return False
    course_key = (gene.course_id, gene.course_type)
    qualified = getattr(instructor, "qualified_courses", [])
    return course_key in qualified or gene.course_id in qualified


def _is_instructor_available(
    instructor_id: str, gene: SessionGene, start_q: int, context: SchedulingContext
) -> bool:
    instructor = context.instructors.get(instructor_id)
    if instructor is None:
        return False
    if instructor.is_full_time:
        return True
    end_q = start_q + gene.num_quanta
    return all(q in instructor.available_quanta for q in range(start_q, end_q))


def _max_group_size(gene: SessionGene, context: SchedulingContext) -> int:
    sizes = [
        getattr(context.groups.get(gid), "student_count", 0) for gid in gene.group_ids
    ]
    return max(sizes) if sizes else 0


def _is_room_suitable(
    room_id: str, gene: SessionGene, context: SchedulingContext
) -> bool:
    room = context.rooms.get(room_id)
    course = context.courses.get((gene.course_id, gene.course_type))
    if room is None or course is None:
        return False

    required = getattr(course, "required_room_features", "lecture")
    room_features = getattr(room, "room_features", "lecture")
    room_capacity = getattr(room, "capacity", 0)
    required = str(required).lower().strip()
    room_features = str(room_features).lower().strip()

    from src.utils.room_compatibility import is_room_suitable_for_course

    course_lab_feats = getattr(course, "specific_lab_features", None)
    room_spec_feats = getattr(room, "specific_features", None)
    if not is_room_suitable_for_course(
        required, room_features, course_lab_feats, room_spec_feats
    ):
        return False

    max_size = _max_group_size(gene, context)
    return room_capacity >= max_size if max_size > 0 else True


def _gene_violation_score(
    gene: SessionGene,
    context: SchedulingContext,
    instructor_counts: dict[int, dict[str, int]],
    room_counts: dict[int, dict[str, int]],
    group_counts: dict[int, dict[str, int]],
) -> int:
    score = 0

    if not _is_instructor_qualified(gene.instructor_id, gene, context):
        score += 1

    if not _is_room_suitable(gene.room_id, gene, context):
        score += 1

    if not _is_instructor_available(
        gene.instructor_id, gene, gene.start_quanta, context
    ):
        score += 1

    for q in range(gene.start_quanta, gene.end_quanta):
        if instructor_counts[q].get(gene.instructor_id, 0) > 1:
            score += 1
            break

    for q in range(gene.start_quanta, gene.end_quanta):
        if room_counts[q].get(gene.room_id, 0) > 1:
            score += 1
            break

    for q in range(gene.start_quanta, gene.end_quanta):
        if any(group_counts[q].get(gid, 0) > 1 for gid in gene.group_ids):
            score += 1
            break

    return score


def _build_violation_state(
    individual: list[SessionGene],
    context: SchedulingContext,
    hard: float,
    soft: float,
) -> ViolationState:
    instructor_counts, room_counts, group_counts = _build_counts(individual)
    gene_scores = [
        _gene_violation_score(
            gene, context, instructor_counts, room_counts, group_counts
        )
        for gene in individual
    ]
    gene_order = sorted(
        range(len(individual)), key=lambda i: gene_scores[i], reverse=True
    )
    return ViolationState(
        hard=hard,
        soft=soft,
        gene_scores=gene_scores,
        gene_order=gene_order,
        instructor_counts=instructor_counts,
        room_counts=room_counts,
        group_counts=group_counts,
        available_quanta=set(context.available_quanta),
    )


def _is_time_conflict_free(
    gene: SessionGene,
    start_q: int,
    state: ViolationState,
) -> bool:
    end_q = start_q + gene.num_quanta
    for q in range(start_q, end_q):
        instructor_count = state.instructor_counts[q].get(gene.instructor_id, 0)
        if gene.start_quanta <= q < gene.end_quanta:
            instructor_count -= 1
        if instructor_count > 0:
            return False
        room_count = state.room_counts[q].get(gene.room_id, 0)
        if gene.start_quanta <= q < gene.end_quanta:
            room_count -= 1
        if room_count > 0:
            return False
        for gid in gene.group_ids:
            group_count = state.group_counts[q].get(gid, 0)
            if gene.start_quanta <= q < gene.end_quanta:
                group_count -= 1
            if group_count > 0:
                return False
    return True


def _is_room_conflict_free(
    gene: SessionGene,
    room_id: str,
    state: ViolationState,
) -> bool:
    for q in range(gene.start_quanta, gene.end_quanta):
        count = state.room_counts[q].get(room_id, 0)
        if room_id == gene.room_id:
            count -= 1
        if count > 0:
            return False
    return True


def _is_instructor_conflict_free(
    gene: SessionGene,
    instructor_id: str,
    state: ViolationState,
) -> bool:
    for q in range(gene.start_quanta, gene.end_quanta):
        count = state.instructor_counts[q].get(instructor_id, 0)
        if instructor_id == gene.instructor_id:
            count -= 1
        if count > 0:
            return False
    return True


class MoveTimeOperator:
    """Repair operator that proposes moving a gene to a different time slot."""

    name = "move_time"

    def propose(
        self,
        individual: list[SessionGene],
        context: SchedulingContext,
        state: ViolationState,
        max_candidates: int,
        rng: random.Random,
    ) -> list[RepairCandidate]:
        if not individual:
            return []

        candidates = []
        # Target top-3 violated genes (not just first)
        target_indices = state.gene_order[:3]

        for target_idx in target_indices:
            gene = individual[target_idx]
            num_quanta = gene.num_quanta

            # Build list of possible starts (sampled).
            available = sorted(state.available_quanta)
            if not available:
                continue
            max_available = max(available)
            possible_starts: list[int] = []
            for start_q in available:
                end_q = start_q + num_quanta
                if end_q > max_available + 1:
                    continue
                if any(q not in state.available_quanta for q in range(start_q, end_q)):
                    continue
                # Skip current position
                if start_q == gene.start_quanta:
                    continue
                possible_starts.append(start_q)

            rng.shuffle(possible_starts)

            # Relaxed filtering: only require instructor availability (hard requirement)
            # Let the evaluation decide if conflicts are reduced
            for start_q in possible_starts[: max_candidates // len(target_indices) + 5]:
                if not _is_instructor_available(
                    gene.instructor_id, gene, start_q, context
                ):
                    continue
                # REMOVED: _is_time_conflict_free check - let evaluation decide
                candidates.append(
                    RepairCandidate(gene_idx=target_idx, new_start=start_q)
                )
                if len(candidates) >= max_candidates:
                    break

            if len(candidates) >= max_candidates:
                break

        return candidates


class SwapRoomOperator:
    """Repair operator that proposes reassigning a gene to a different room."""

    name = "swap_room"

    def propose(
        self,
        individual: list[SessionGene],
        context: SchedulingContext,
        state: ViolationState,
        max_candidates: int,
        rng: random.Random,
    ) -> list[RepairCandidate]:
        if not individual:
            return []

        candidates = []
        # Target top-3 violated genes (not just first)
        target_indices = state.gene_order[:3]

        for target_idx in target_indices:
            gene = individual[target_idx]

            room_ids = list(context.rooms.keys())
            rng.shuffle(room_ids)

            for room_id in room_ids:
                if room_id == gene.room_id:
                    continue
                # Room suitability is a hard requirement - keep this check
                if not _is_room_suitable(room_id, gene, context):
                    continue
                # REMOVED: _is_room_conflict_free check - let evaluation decide
                candidates.append(
                    RepairCandidate(gene_idx=target_idx, new_room_id=room_id)
                )
                if len(candidates) >= max_candidates:
                    break

            if len(candidates) >= max_candidates:
                break

        return candidates


class ReassignInstructorOperator:
    """Repair operator that proposes reassigning a gene to a different instructor."""

    name = "reassign_instructor"

    def propose(
        self,
        individual: list[SessionGene],
        context: SchedulingContext,
        state: ViolationState,
        max_candidates: int,
        rng: random.Random,
    ) -> list[RepairCandidate]:
        if not individual:
            return []

        candidates = []
        # Target top-3 violated genes (not just first)
        target_indices = state.gene_order[:3]

        for target_idx in target_indices:
            gene = individual[target_idx]

            instructor_ids = list(context.instructors.keys())
            rng.shuffle(instructor_ids)

            for instructor_id in instructor_ids:
                if instructor_id == gene.instructor_id:
                    continue
                # Qualification is a hard requirement - keep this check
                if not _is_instructor_qualified(instructor_id, gene, context):
                    continue
                # Availability is a hard requirement - keep this check
                if not _is_instructor_available(
                    instructor_id, gene, gene.start_quanta, context
                ):
                    continue
                # REMOVED: _is_instructor_conflict_free check - let evaluation decide
                candidates.append(
                    RepairCandidate(
                        gene_idx=target_idx, new_instructor_id=instructor_id
                    )
                )
                if len(candidates) >= max_candidates:
                    break

            if len(candidates) >= max_candidates:
                break

        return candidates


class RepairEngine:
    """Repair engine with lexicographic scoring and RL-ready interface."""

    def __init__(
        self,
        context: SchedulingContext,
        evaluator: FitnessFn,
        policy: str = "round_robin",
        max_steps: int = 5,
        max_candidates: int = 20,
        budget_ms: float = 50.0,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
        operators: Iterable[RepairOperator] | None = None,
        logger: logging.Logger | None = None,
        log_steps: bool = False,
        log_candidates: bool = True,
    ) -> None:
        self.context = context
        self._evaluate = evaluator
        self.max_steps = max_steps
        self.max_candidates = max_candidates
        self.budget_ms = budget_ms
        self.rng = rng or random.Random()
        self.logger = logger or logging.getLogger(__name__)
        self.log_steps = log_steps
        self.log_candidates = log_candidates

        self.operators: list[RepairOperator] = list(
            operators
            or [
                MoveTimeOperator(),
                SwapRoomOperator(),
                ReassignInstructorOperator(),
            ]
        )
        self.operator_map: dict[str, RepairOperator] = {
            op.name: op for op in self.operators
        }

        if policy == "round_robin":
            self.policy: RoundRobinPolicy | EpsilonGreedyPolicy = RoundRobinPolicy(
                [op.name for op in self.operators]
            )
            self.policy_type = "round_robin"
        else:
            self.policy = EpsilonGreedyPolicy(
                [op.name for op in self.operators], epsilon=epsilon
            )
            self.policy_type = "epsilon_greedy"

        self.operator_stats: dict[str, dict[str, float]] = {}

    def get_action_space(self) -> list[str]:
        """Return the list of available operator names."""
        return [op.name for op in self.operators]

    def repair_individual(
        self,
        individual: list[SessionGene],
        budget_ms: float | None = None,
        max_steps: int | None = None,
        forced_operator: str | None = None,
    ) -> RepairStats:
        """Repair a single individual within budget."""
        budget_ms = self.budget_ms if budget_ms is None else budget_ms
        max_steps = self.max_steps if max_steps is None else max_steps

        stats = RepairStats()
        start_time = time.perf_counter()

        current_hard, current_soft = self._evaluate(individual)

        for step_idx in range(max_steps):
            if budget_ms > 0 and (time.perf_counter() - start_time) * 1000 > budget_ms:
                break

            state = _build_violation_state(
                individual, self.context, current_hard, current_soft
            )

            if forced_operator:
                operator_name = forced_operator
            elif isinstance(self.policy, RoundRobinPolicy):
                operator_name = self.policy.select()
            else:
                operator_name = self.policy.select(self.operator_stats, self.rng)

            operator = self.operator_map.get(operator_name)
            if operator is None:
                break

            # Track attempted steps for success rate reporting
            op_stats = self.operator_stats.setdefault(
                operator_name,
                {"steps": 0.0, "applied": 0.0, "delta_hard": 0.0, "delta_soft": 0.0},
            )
            op_stats["steps"] += 1.0

            if self.log_steps:
                self.logger.debug(
                    "Repair step %d: operator=%s hard=%.2f soft=%.2f",
                    step_idx,
                    operator_name,
                    current_hard,
                    current_soft,
                )

            candidates = operator.propose(
                individual,
                self.context,
                state,
                self.max_candidates,
                self.rng,
            )
            if self.log_candidates:
                self.logger.debug(
                    "Repair operator %s proposed %d candidates",
                    operator_name,
                    len(candidates),
                )

            best: RepairCandidate | None = None
            best_after: tuple[float, float] | None = None

            for candidate in candidates:
                before = (current_hard, current_soft)
                after = self._evaluate_candidate(individual, candidate)
                if after is None:
                    continue
                if self._is_lex_better(after, before) and (
                    best_after is None or self._is_lex_better(after, best_after)
                ):
                    best = candidate
                    best_after = after

            if best is None or best_after is None:
                result = RepairStepResult(
                    applied=False,
                    operator=operator_name,
                    delta_hard=0.0,
                    delta_soft=0.0,
                    before=(current_hard, current_soft),
                    after=(current_hard, current_soft),
                )
                stats.record(result)
                if self.log_steps:
                    self.logger.debug(
                        "Repair operator %s: no improving candidate",
                        operator_name,
                    )
                continue

            self._apply_candidate(individual, best)
            after_hard, after_soft = best_after
            result = RepairStepResult(
                applied=True,
                operator=operator_name,
                delta_hard=current_hard - after_hard,
                delta_soft=current_soft - after_soft,
                before=(current_hard, current_soft),
                after=(after_hard, after_soft),
            )
            stats.record(result)
            current_hard, current_soft = after_hard, after_soft
            if self.log_steps:
                self.logger.debug(
                    "Repair operator %s applied gene=%d delta_hard=%.2f delta_soft=%.2f "
                    "start=%s room=%s instructor=%s",
                    operator_name,
                    best.gene_idx,
                    result.delta_hard,
                    result.delta_soft,
                    best.new_start,
                    best.new_room_id,
                    best.new_instructor_id,
                )

            # Update operator stats for epsilon-greedy
            op_stats = self.operator_stats.setdefault(
                operator_name,
                {"steps": 0.0, "applied": 0.0, "delta_hard": 0.0, "delta_soft": 0.0},
            )
            op_stats["applied"] += 1.0
            op_stats["delta_hard"] += result.delta_hard
            op_stats["delta_soft"] += result.delta_soft

        return stats

    def step(
        self,
        individual: list[SessionGene],
        operator_name: str | None = None,
    ) -> RepairStepResult:
        """Single repair step (RL-friendly)."""
        stats = self.repair_individual(
            individual, max_steps=1, forced_operator=operator_name
        )
        if stats.step_results:
            return stats.step_results[-1]
        return RepairStepResult(
            applied=False,
            operator=operator_name or "none",
            delta_hard=0.0,
            delta_soft=0.0,
            before=(0.0, 0.0),
            after=(0.0, 0.0),
        )

    @staticmethod
    def _is_lex_better(a: tuple[float, float], b: tuple[float, float]) -> bool:
        """Return True if fitness *a* is lexicographically better than *b*."""
        return (a[0], a[1]) < (b[0], b[1])

    def _evaluate_candidate(
        self, individual: list[SessionGene], candidate: RepairCandidate
    ) -> tuple[float, float] | None:
        """Evaluate a candidate repair by temporarily applying it and scoring."""
        gene = individual[candidate.gene_idx]
        old_start = gene.start_quanta
        old_room = gene.room_id
        old_instructor = gene.instructor_id

        if candidate.new_start is not None:
            gene.shift_to(candidate.new_start)
        if candidate.new_room_id is not None:
            gene.room_id = candidate.new_room_id
        if candidate.new_instructor_id is not None:
            gene.instructor_id = candidate.new_instructor_id

        try:
            return self._evaluate(individual)
        finally:
            gene.start_quanta = old_start
            gene.room_id = old_room
            gene.instructor_id = old_instructor

    def _apply_candidate(
        self, individual: list[SessionGene], candidate: RepairCandidate
    ) -> None:
        """Permanently apply a repair candidate to the individual."""
        gene = individual[candidate.gene_idx]
        if candidate.new_start is not None:
            gene.shift_to(candidate.new_start)
        if candidate.new_room_id is not None:
            gene.room_id = candidate.new_room_id
        if candidate.new_instructor_id is not None:
            gene.instructor_id = candidate.new_instructor_id
