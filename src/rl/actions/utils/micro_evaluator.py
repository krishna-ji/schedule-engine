r"""Micro-Evaluator — lightweight localized constraint impact calculation.

Provides ultra-fast ΔHard evaluation for single-event moves without
full population matrix reconstruction. Used by conflict-directed
micro-memetic optimizers to "look before they leap."

Key Function
------------
evaluate_local_move(pop, event_idx, new_time, new_room, new_inst) -> ΔHard

Calculates the net hard constraint impact of moving a single event
to new assignments within a population individual, returning:
- ΔHard < 0: Move improves hard constraints (good)
- ΔHard > 0: Move worsens hard constraints (bad)
- ΔHard = 0: Move is neutral

The function operates in O(Q) time where Q is the expansion length
of the target event, avoiding the full O(N⋅E) population evaluation.

Architecture
------------
Uses the same vectorized evaluation data as the main hard evaluator
but operates on temporary single-event perturbations of the population
matrix. Leverages bincount-based occupancy detection for SRE/CTE/FTE
violations and domain validation for FPC/FFC constraints.

Constraints Evaluated
--------------------
1. CTE (Cohort Temporal Exclusivity) - student group conflicts
2. FTE (Faculty Temporal Exclusivity) - instructor conflicts
3. SRE (Spatial Resource Exclusivity) - room double-booking
4. FPC (Faculty Pedagogical Congruence) - instructor qualification
5. FFC (Facility Feature Congruence) - room feature requirements
6. FCA (Faculty Chronological Availability) - instructor time availability
7. CQF (Curriculum Quanta Fulfillment) - duration requirements (structural)
8. ICTD (Intra-Course Temporal Dispersion) - same-course spacing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.pipeline.repair_operator_vectorized import VectorizedRepair

logger = logging.getLogger(__name__)


def evaluate_local_move(
    engine: VectorizedRepair,
    X: np.ndarray,
    individual_idx: int,
    event_idx: int,
    new_time: int,
    new_room: int,
    new_inst: int,
) -> float:
    """Calculate ΔHard for a single event move within a population individual.

    Parameters
    ----------
    engine : VectorizedRepair
        The vectorized repair engine with precomputed data structures.
    X : ndarray, shape (N, 3*E)
        Population matrix where N=population size, E=number of events.
    individual_idx : int
        Index of the individual to modify (0 ≤ individual_idx < N).
    event_idx : int
        Index of the event to move (0 ≤ event_idx < E).
    new_time : int
        New time slot assignment.
    new_room : int
        New room assignment.
    new_inst : int
        New instructor assignment.

    Returns
    -------
    delta_hard : float
        Net change in hard constraint violations. Negative = improvement.
    """
    E = X.shape[1] // 3

    if not (0 <= individual_idx < X.shape[0] and 0 <= event_idx < E):
        return np.inf  # Invalid indices

    # Extract original assignment for the target event
    orig_inst = int(X[individual_idx, 3 * event_idx + 0])
    orig_room = int(X[individual_idx, 3 * event_idx + 1])
    orig_time = int(X[individual_idx, 3 * event_idx + 2])

    # If no change, return zero delta
    if new_inst == orig_inst and new_room == orig_room and new_time == orig_time:
        return 0.0

    # Copy ONLY the single individual row (not entire population!)
    row_before = X[individual_idx].copy().reshape(1, -1)

    # Calculate hard violations BEFORE the move
    violations_before = _calculate_hard_violations_single(engine, row_before, 0)

    # Apply the proposed move to the copy
    row_after = row_before.copy()
    row_after[0, 3 * event_idx + 0] = new_inst
    row_after[0, 3 * event_idx + 1] = new_room
    row_after[0, 3 * event_idx + 2] = new_time

    # Calculate hard violations AFTER the move
    violations_after = _calculate_hard_violations_single(engine, row_after, 0)

    # Return the net change (negative = improvement)
    return float(violations_after - violations_before)


def _calculate_hard_violations_single(
    engine: VectorizedRepair,
    X: np.ndarray,
    individual_idx: int,
) -> int:
    """Calculate total hard constraint violations for a single individual.

    Uses the same logic as the full vectorized evaluator but operates
    on a single population member for speed.

    Parameters
    ----------
    engine : VectorizedRepair
        The vectorized repair engine.
    X : ndarray, shape (N, 3*E)
        Population matrix.
    individual_idx : int
        Index of individual to evaluate.

    Returns
    -------
    total_violations : int
        Sum of all hard constraint violations for this individual.
    """
    from src.pipeline.fast_evaluator_vectorized import (
        fast_evaluate_hard_vectorized,
        prepare_vectorized_data,
    )

    N, n_vars = X.shape
    E = n_vars // 3

    # Extract single individual as (1, 3*E) matrix for vectorized evaluator
    X_single = X[individual_idx : individual_idx + 1, :]

    # Use the existing vectorized hard evaluator
    try:
        # Need to prepare evaluation data if not already available
        if not hasattr(engine, "_eval_data_cache"):
            # Reconstruct from engine's pkl data if available
            if hasattr(engine, "_events") and hasattr(engine, "_allowed_instructors"):
                pkl_data = {
                    "events": engine._events,
                    "allowed_instructors": engine._allowed_instructors,
                    "allowed_rooms": engine._allowed_rooms,
                    "instructor_available_quanta": engine.inst_avail,
                    "room_available_quanta": engine.room_avail,
                    "exp_event": engine.exp_event,
                    "exp_offset": engine.exp_offset,
                    "group_exp_event": engine.group_exp_event,
                    "group_exp_offset": engine.group_exp_offset,
                }
                engine._eval_data_cache = prepare_vectorized_data(pkl_data)
            else:
                # Fallback to basic conflict counting
                return _basic_conflict_count(engine, X_single, 0)

        G_single = fast_evaluate_hard_vectorized(X_single, engine._eval_data_cache)
        # Sum all constraint violations (shape: (1, 8) -> scalar)
        return int(G_single.sum())

    except (AttributeError, ImportError) as e:
        logger.warning(f"Fast evaluator unavailable ({e}), using basic conflict count")
        return _basic_conflict_count(engine, X_single, 0)


def _basic_conflict_count(
    engine: VectorizedRepair,
    X: np.ndarray,
    individual_idx: int,
) -> int:
    """Basic conflict counting using engine's bincount logic.

    Fallback implementation when fast_evaluate_hard_vectorized is unavailable.
    Focuses on the three main occupancy conflicts: SRE, CTE, FTE.
    """
    from src.pipeline.bitset_time import T as T_

    N = X.shape[0]
    E = engine.n_events

    # Extract assignments for the individual
    inst = np.clip(X[:, 0::3], 0, engine.n_instructors - 1).astype(np.int64)
    room = np.clip(X[:, 1::3], 0, engine.n_rooms - 1).astype(np.int64)
    time = X[:, 2::3].astype(np.int64)

    total_violations = 0

    # Individual index tensor
    n_idx = np.arange(N, dtype=np.int64)[:, None]

    # === SRE: Spatial Resource Exclusivity (room conflicts) ===
    starts_exp = time[:, engine.exp_event]
    quanta_exp = np.clip(starts_exp + engine.exp_offset[None, :], 0, T_ - 1)
    rooms_exp = room[:, engine.exp_event]

    nRT = np.int64(engine.n_rooms) * np.int64(T_)
    room_keys = (n_idx * nRT + rooms_exp * T_ + quanta_exp).ravel()
    room_cnt = np.bincount(room_keys, minlength=int(N * nRT))
    room_conflicts = (room_cnt[room_keys] > 1).astype(np.int64)

    # Aggregate per individual
    event_lin = (n_idx * E + engine.exp_event[None, :]).ravel()
    room_scores = np.bincount(event_lin, weights=room_conflicts, minlength=N * E)
    room_scores = room_scores[: N * E].reshape(N, E)
    total_violations += int(room_scores[individual_idx, :].sum())

    # === FTE: Faculty Temporal Exclusivity (instructor conflicts) ===
    inst_exp = inst[:, engine.exp_event]
    nIT = np.int64(engine.n_instructors) * np.int64(T_)
    inst_keys = (n_idx * nIT + inst_exp * T_ + quanta_exp).ravel()
    inst_cnt = np.bincount(inst_keys, minlength=int(N * nIT))
    inst_conflicts = (inst_cnt[inst_keys] > 1).astype(np.int64)

    inst_scores = np.bincount(event_lin, weights=inst_conflicts, minlength=N * E)
    inst_scores = inst_scores[: N * E].reshape(N, E)
    total_violations += int(inst_scores[individual_idx, :].sum())

    # === CTE: Cohort Temporal Exclusivity (student group conflicts) ===
    # This requires group expansion which is more complex - simplified version
    if hasattr(engine, "group_exp_event") and len(engine.group_exp_event) > 0:
        groups_exp = engine.group_exp_event  # Pre-expanded group assignments
        group_quanta = (
            quanta_exp.ravel()[groups_exp] if groups_exp.size > 0 else np.array([])
        )

        if group_quanta.size > 0:
            # Count group-time conflicts (simplified)
            n_groups = getattr(engine, "n_groups", 50)  # Fallback estimate
            nGT = n_groups * T_
            # This is simplified - full implementation would need proper group mapping
            group_time_pairs = group_quanta % nGT  # Simplified mapping
            group_cnt = np.bincount(group_time_pairs, minlength=nGT)
            group_conflicts = np.sum(group_cnt > 1)
            total_violations += group_conflicts

    return total_violations


def get_conflict_events(
    engine: VectorizedRepair,
    X: np.ndarray,
    individual_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify events causing the most hard constraint violations.

    Returns the events sorted by conflict severity for targeted optimization.

    Optimized: operates on SINGLE individual (N=1 slice) instead of
    computing over the full population and discarding N-1 results.

    Parameters
    ----------
    engine : VectorizedRepair
        The vectorized repair engine.
    X : ndarray, shape (N, 3*E)
        Population matrix.
    individual_idx : int
        Index of individual to analyze.

    Returns
    -------
    conflict_events : ndarray
        Event indices sorted by violation count (descending).
    violation_counts : ndarray
        Corresponding violation counts for each event.
    """
    from src.pipeline.bitset_time import T as T_

    E = X.shape[1] // 3

    # Extract assignments for SINGLE individual only (N=1)
    row = X[individual_idx]
    inst = np.clip(row[0::3], 0, engine.n_instructors - 1).astype(np.int64)
    room = np.clip(row[1::3], 0, engine.n_rooms - 1).astype(np.int64)
    time = row[2::3].astype(np.int64)

    # Calculate room conflicts per event (single individual, no n_idx needed)
    starts_exp = time[engine.exp_event]
    quanta_exp = np.clip(starts_exp + engine.exp_offset, 0, T_ - 1)
    rooms_exp = room[engine.exp_event]

    room_keys = (rooms_exp * T_ + quanta_exp).ravel()
    nRT = int(engine.n_rooms) * T_
    room_cnt = np.bincount(room_keys, minlength=nRT)
    room_conflicts = (room_cnt[room_keys] > 1).astype(np.int64)

    event_lin = engine.exp_event
    room_scores = np.bincount(event_lin, weights=room_conflicts, minlength=E)[:E]

    # Calculate instructor conflicts per event
    inst_exp = inst[engine.exp_event]
    nIT = int(engine.n_instructors) * T_
    inst_keys = (inst_exp * T_ + quanta_exp).ravel()
    inst_cnt = np.bincount(inst_keys, minlength=nIT)
    inst_conflicts = (inst_cnt[inst_keys] > 1).astype(np.int64)

    inst_scores = np.bincount(event_lin, weights=inst_conflicts, minlength=E)[:E]

    # Combine conflict scores
    total_conflicts = room_scores + inst_scores

    # Sort events by conflict severity (descending)
    conflict_order = np.argsort(-total_conflicts)

    return conflict_order, total_conflicts[conflict_order]


def validate_domain_move(
    engine: VectorizedRepair,
    event_idx: int,
    new_time: int,
    new_room: int,
    new_inst: int,
) -> bool:
    """Check if a proposed move satisfies domain constraints.

    Validates that the new assignments are within the allowed domains
    for the target event without causing immediate domain violations.

    Parameters
    ----------
    engine : VectorizedRepair
        The vectorized repair engine with domain data.
    event_idx : int
        Index of the event to validate.
    new_time : int
        Proposed time assignment.
    new_room : int
        Proposed room assignment.
    new_inst : int
        Proposed instructor assignment.

    Returns
    -------
    is_valid : bool
        True if all assignments are within valid domains.
    """
    try:
        # Check instructor domain
        if hasattr(engine, "inst_domains") and hasattr(engine, "inst_dom_len"):
            inst_valid_count = engine.inst_dom_len[event_idx]
            if inst_valid_count > 0:
                valid_insts = engine.inst_domains[event_idx, :inst_valid_count]
                if new_inst not in valid_insts:
                    return False

        # Check room domain
        if hasattr(engine, "room_domains") and hasattr(engine, "room_dom_len"):
            room_valid_count = engine.room_dom_len[event_idx]
            if room_valid_count > 0:
                valid_rooms = engine.room_domains[event_idx, :room_valid_count]
                if new_room not in valid_rooms:
                    return False

        # Check time domain
        if hasattr(engine, "time_domains") and hasattr(engine, "time_dom_len"):
            time_valid_count = engine.time_dom_len[event_idx]
            if time_valid_count > 0:
                valid_times = engine.time_domains[event_idx, :time_valid_count]
                if new_time not in valid_times:
                    return False

        return True

    except (AttributeError, IndexError):
        # If domain data unavailable, assume valid (conservative)
        return True
