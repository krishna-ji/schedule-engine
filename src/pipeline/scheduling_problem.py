"""Pymoo Problem definition for university timetable scheduling.

Implements the scheduling optimization as a pymoo Problem subclass:
- Decision variables: 3×E interleaved chromosome [I0,R0,T0, I1,R1,T1, ...]
- Objectives: F[0] = total hard violations, F[1] = total soft penalty
- Constraints: G[i] = violation count for hard constraint i (G <= 0 means satisfied)

The soft evaluation optionally uses the original Evaluator (via Timetable
construction) when a SchedulingContext is provided, or falls back to a
simplified numeric soft penalty.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .encoding import EncodingSpec, chromosome_views
from .fast_evaluator import fast_evaluate_hard
from .fast_evaluator_batch import (
    BatchEvalData,
    fast_evaluate_hard_batch,
    prepare_batch_data,
)
from .fast_evaluator_vectorized import (
    VectorizedEvalData,
    fast_evaluate_hard_vectorized,
    prepare_vectorized_data,
)

if TYPE_CHECKING:
    from src.domain.types import SchedulingContext
    from src.io.time_system import QuantumTimeSystem

try:
    from pymoo.core.problem import Problem
except ImportError:
    raise ImportError("pymoo is required: pip install pymoo>=0.6")


# Hard constraint names in canonical order
HARD_CONSTRAINT_NAMES = [
    "student_group_exclusivity",
    "instructor_exclusivity",
    "room_exclusivity",
    "instructor_qualifications",
    "room_suitability",
    "instructor_time_availability",
    "room_time_availability",
    "course_completeness",
]


class SchedulingProblem(Problem):
    """Pymoo Problem for university timetable scheduling.

    Objectives:
        F[:, 0] = total hard penalty (weighted sum of all hard constraints)
        F[:, 1] = total soft penalty (from original evaluator or proxy)

    Inequality constraints (G <= 0 means feasible):
        G[:, i] = violation count for hard constraint i

    Parameters
    ----------
    pkl_path : str
        Path to events_with_domains.pkl.
    ctx : SchedulingContext | None
        If provided, enables full soft constraint evaluation via the
        original Evaluator. Without this, soft penalty is 0.
    qts : QuantumTimeSystem | None
        Quantum time system (needed only if ctx is provided).
    """

    def __init__(
        self,
        pkl_path: str = "events_with_domains.pkl",
        ctx: SchedulingContext | None = None,
        qts: QuantumTimeSystem | None = None,
        *,
        vectorized: bool = True,
    ):
        with open(pkl_path, "rb") as f:
            self.pkl_data: dict = pickle.load(f)

        self.spec = EncodingSpec.from_pkl_data(self.pkl_data)
        self.events = self.pkl_data["events"]
        self.allowed_instructors = self.pkl_data["allowed_instructors"]
        self.allowed_rooms = self.pkl_data["allowed_rooms"]
        self.inst_avail = self.pkl_data["instructor_available_quanta"]
        self.room_avail = self.pkl_data["room_available_quanta"]
        self.idx_to_instructor = {
            int(k): v for k, v in self.pkl_data["idx_to_instructor"].items()
        }
        self.idx_to_room = {int(k): v for k, v in self.pkl_data["idx_to_room"].items()}

        self.ctx = ctx
        self.qts = qts

        # Soft evaluator (only when context available)
        self._evaluator = None
        if ctx is not None:
            from src.constraints.evaluator import Evaluator

            self._evaluator = Evaluator()

        # Precomputed evaluation data
        self._vectorized = vectorized
        self._batch_data: BatchEvalData = prepare_batch_data(self.pkl_data)
        self._vec_data: VectorizedEvalData | None = (
            prepare_vectorized_data(self.pkl_data) if vectorized else None
        )

        super().__init__(
            n_var=self.spec.n_vars,
            n_obj=2,  # hard, soft
            n_ieq_constr=len(HARD_CONSTRAINT_NAMES),
            xl=self.spec.xl(),
            xu=self.spec.xu(),
            type_var=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate a population matrix x of shape (pop_size, n_var).

        Uses the batch bitset evaluator for hard constraints (vectorized
        over the full population), then per-individual soft evaluation
        when a SchedulingContext is available.

        Sets:
            out["F"] = objectives (pop_size, 2)
            out["G"] = constraint violations (pop_size, n_constr)
        """
        pop_size = x.shape[0]

        # ---- Hard constraints ----
        if self._vectorized and self._vec_data is not None:
            G = fast_evaluate_hard_vectorized(x, self._vec_data)
        else:
            G = fast_evaluate_hard_batch(x, self._batch_data)

        # ---- Objectives ----
        F = np.zeros((pop_size, 2))
        F[:, 0] = G.sum(axis=1)  # total hard penalty

        # ---- Soft evaluation (per-individual, only when available) ----
        if self._evaluator is not None and self.ctx is not None:
            for i in range(pop_size):
                F[i, 1] = self._evaluate_soft(x[i].astype(int))

        out["F"] = F
        out["G"] = G

    def _evaluate_soft(self, xi: np.ndarray) -> float:
        """Evaluate soft constraints using the original Evaluator.

        Converts numeric chromosome back to SessionGene list, builds
        Timetable, and evaluates soft constraints.
        """
        from src.domain.gene import SessionGene
        from src.domain.timetable import Timetable

        inst, room, time = chromosome_views(xi)
        genes = []
        for e in range(self.spec.n_events):
            ev = self.events[e]
            genes.append(
                SessionGene(
                    course_id=ev["course_id"],
                    course_type=ev["course_type"],
                    instructor_id=self.idx_to_instructor[int(inst[e])],
                    group_ids=list(ev["group_ids"]),
                    room_id=self.idx_to_room[int(room[e])],
                    start_quanta=int(time[e]),
                    num_quanta=ev["num_quanta"],
                )
            )

        assert self.ctx is not None
        assert self._evaluator is not None
        tt = Timetable(genes, self.ctx, self.qts)
        _, soft = self._evaluator.fitness_from_timetable(tt)
        return soft


def create_problem(
    pkl_path: str = "events_with_domains.pkl",
    ctx: SchedulingContext | None = None,
    qts: QuantumTimeSystem | None = None,
) -> SchedulingProblem:
    """Factory function to create SchedulingProblem.

    If ctx/qts are not provided, tries to load from data directory.
    """
    if ctx is None:
        try:
            from src.io.data_store import DataStore
            from src.io.time_system import QuantumTimeSystem as QTS

            store = DataStore.from_json("data")
            ctx = store.to_context()
            qts = QTS()

            # Apply tutorial-practical fix if the pkl was built with it
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            if pkl_data.get("fix_tutorial_practicals", False):
                for _key, course in ctx.courses.items():
                    lab_feats = getattr(course, "specific_lab_features", None)
                    if lab_feats:
                        feats_lower = [
                            (f if isinstance(f, str) else str(f)).lower().strip()
                            for f in lab_feats
                        ]
                        if any(
                            f in ("lecture hall", "seminar room") for f in feats_lower
                        ):
                            course.specific_lab_features = []
        except Exception:
            pass  # No soft evaluation; hard-only mode

    return SchedulingProblem(pkl_path=pkl_path, ctx=ctx, qts=qts)
