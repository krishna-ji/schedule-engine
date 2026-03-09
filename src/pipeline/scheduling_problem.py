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
from typing import TYPE_CHECKING

import numpy as np

from .encoding import EncodingSpec, chromosome_views
from .fast_evaluator_vectorized import (
    VectorizedEvalData,
    fast_evaluate_hard_vectorized,
    prepare_vectorized_data,
)
from .soft_evaluator_vectorized import (
    SoftVectorizedData,
    eval_soft_vectorized_breakdown,
    evaluate_paired_cohorts_vectorized,
    prepare_soft_vectorized_data,
)
from .vectorized_lookups import VectorizedLookups, build_vectorized_lookups

if TYPE_CHECKING:
    from src.domain.types import SchedulingContext
    from src.io.time_system import QuantumTimeSystem

try:
    from pymoo.core.problem import Problem
except ImportError as err:
    raise ImportError("pymoo is required: pip install pymoo>=0.6") from err


# Hard constraint names in canonical order (Academic Nomenclature)
HARD_CONSTRAINT_NAMES = [
    "CTE",  # Cohort Temporal Exclusivity
    "FTE",  # Faculty Temporal Exclusivity
    "SRE",  # Spatial Resource Exclusivity
    "FPC",  # Faculty Pedagogical Congruence
    "FFC",  # Facility Feature Congruence
    "FCA",  # Faculty Chronological Availability
    "CQF",  # Curriculum Quanta Fulfillment
    "ICTD",  # Intra-Course Temporal Dispersion
]

# Indices of G columns that are TOLERATED (excluded from hard objective,
# added to soft instead).  instructor_time_availability (col 5) is
# structurally infeasible for some events, so we treat it as soft.
_TOLERATED_HARD_COLS = frozenset({5})  # FCA
_STRICT_HARD_COLS = np.array(
    [i for i in range(len(HARD_CONSTRAINT_NAMES)) if i not in _TOLERATED_HARD_COLS]
)


class SchedulingProblem(Problem):
    """Pymoo Problem for university timetable scheduling.

    Objectives:
        F[:, 0] = total hard penalty (weighted sum of all hard constraints)
        F[:, 1] = total soft penalty (from original evaluator or proxy)

    Inequality constraints (G <= 0 means feasible):
        G[:, i] = violation count for hard constraint i

    Args:
        pkl_path: Path to .cache/events_with_domains.pkl.
        ctx: If provided, enables full soft constraint evaluation via the
            original Evaluator. Without this, soft penalty is 0.
        qts: Quantum time system (needed only if ctx is provided).
    """

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        ctx: SchedulingContext | None = None,
        qts: QuantumTimeSystem | None = None,
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

        # Precomputed evaluation data — always vectorized (canonical path)
        self._vec_data: VectorizedEvalData = prepare_vectorized_data(self.pkl_data)
        self._soft_data: SoftVectorizedData = prepare_soft_vectorized_data(
            self.pkl_data
        )
        # Unified lookups (superset of _vec_data: includes domain matrices,
        # group_conflict_matrix, event_metadata).
        self.lookups: VectorizedLookups = build_vectorized_lookups(self.pkl_data)

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

        # ---- Hard constraints (vectorized, canonical) ----
        G = fast_evaluate_hard_vectorized(x, self._vec_data)

        # ---- Objectives ----
        F = np.zeros((pop_size, 2))
        # Only strict hard columns contribute to F[0]; tolerated ones
        # (e.g. iAvl) are shifted to the soft objective.
        F[:, 0] = G[:, _STRICT_HARD_COLS].sum(axis=1)  # strict hard penalty
        # Add tolerated hard-constraint violations as soft penalty
        for col in _TOLERATED_HARD_COLS:
            F[:, 1] += G[:, col]

        # ---- Soft evaluation (vectorized over full population) ----
        soft_total, soft_bd = eval_soft_vectorized_breakdown(x, self._soft_data)
        F[:, 1] += soft_total

        # ---- Paired cohort practical alignment (soft) ----
        paired_penalty = evaluate_paired_cohorts_vectorized(x, self.lookups)
        F[:, 1] += paired_penalty
        soft_bd["SSCP"] = paired_penalty

        # Store latest soft breakdown for callback access
        self._last_soft_breakdown = soft_bd

        # ---- Defensive NaN/Inf guard on soft scores ----
        F[:, 1] = np.nan_to_num(F[:, 1], nan=1e6, posinf=1e6, neginf=0.0)

        # ---- Optional: OOP soft eval fallback (legacy, per-individual) ----
        # Uncomment to use original Evaluator instead of vectorized:
        # if self._evaluator is not None and self.ctx is not None:
        #     for i in range(pop_size):
        #         F[i, 1] = self._evaluate_soft(x[i].astype(int))

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
    pkl_path: str = ".cache/events_with_domains.pkl",
    ctx: SchedulingContext | None = None,
    qts: QuantumTimeSystem | None = None,
    run_preflight: bool = True,
) -> SchedulingProblem:
    """Factory function to create SchedulingProblem.

    If ctx/qts are not provided, tries to load from data directory.

    Args:
        run_preflight: Run feasibility checks when loading DataStore.
            Set False in SubprocVecEnv workers to avoid redundant
            O(N²) checks that spam stdout 24× at startup.
    """
    if ctx is None:
        try:
            from src.io.data_store import DataStore
            from src.io.time_system import QuantumTimeSystem as QTS

            store = DataStore.from_json("data", run_preflight=run_preflight)
            ctx = store.to_context()
            qts = QTS()

            # Apply tutorial-practical fix if the pkl was built with it
            with open(pkl_path, "rb") as f:
                pkl_data = pickle.load(f)
            if pkl_data.get("fix_tutorial_practicals", False):
                for course in ctx.courses.values():
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
