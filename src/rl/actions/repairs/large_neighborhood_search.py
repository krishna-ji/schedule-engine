r"""Large Neighborhood Search — Ruin & Recreate Operator.

**STATE-OF-THE-ART META-HEURISTIC** replacing the structurally capped
cohort_temporal_projection (Action 2, Δ=-93 in isolation).

Algorithm (per individual):
1. **Ruin**: Use ``eng._score_all_batch(X)`` to identify the top 5%
   (~40) most-conflicting events.  "Delete" them by extracting their
   assignments from the matrix.
2. **Recreate (Greedy Best-Fit)**: Sort ruined events by domain
   restrictiveness (fewest valid rooms × times first — most constrained
   events placed first).  For each event, scan valid (time, room)
   domain combinations and select the assignment yielding the lowest
   marginal hard constraint increase.

Mathematical Advantage:
  By moving ~40 highly-conflicted events *simultaneously* in a
  constraint-aware reinsertion sequence, we teleport across the
  penalty ridges that trap monotone single-event repair operators.

Complexity: $O(N_s \cdot K \cdot |D_r| \cdot |D_t|)$ where
$N_s$ = selected individuals, $K$ = ruined events, $D_r$/$D_t$ = domain sizes.
Capped via budget parameters for RL step speed.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)


class LargeNeighborhoodSearch(_AtomicRepairBase):
    """Action 2 — Ruin & Recreate LNS meta-heuristic."""

    ACTION_NAME: ClassVar[str] = "large_neighborhood_search"

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        ruin_fraction: float = 0.05,
        max_individuals: int = 5,
        max_room_candidates: int = 5,
        max_time_candidates: int = 8,
    ):
        super().__init__(pkl_path)
        self.ruin_fraction = ruin_fraction
        self.max_individuals = max_individuals
        self.max_room_candidates = max_room_candidates
        self.max_time_candidates = max_time_candidates

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N, n_vars = X.shape
        E = eng.n_events

        # Fix domains first
        eng._fix_domains_vec(X)

        # ── SCORING: identify per-event conflict severity ──────────
        scores = eng._score_all_batch(X)  # (N, E) int32

        # Select top-K worst individuals by total severity
        ind_severity = scores.sum(axis=1)
        nonzero = ind_severity > 0
        if not nonzero.any():
            return
        conflict_individuals = np.where(nonzero)[0]
        severity_order = np.argsort(-ind_severity[conflict_individuals])
        selected = conflict_individuals[severity_order[: self.max_individuals]]

        rng = np.random.default_rng()
        n_ruin = max(1, int(E * self.ruin_fraction))  # ~40 events

        total_improved = 0

        for idx in selected:
            row_scores = scores[idx]  # (E,)
            if row_scores.sum() == 0:
                continue

            # ── RUIN PHASE: extract top-K worst events ─────────────
            worst_events = np.argsort(-row_scores)[:n_ruin]
            # Only ruin events that actually have conflicts
            has_conflict = row_scores[worst_events] > 0
            worst_events = worst_events[has_conflict]
            if len(worst_events) == 0:
                continue

            # Save original assignments (for rollback comparison)
            orig_assignments = np.empty((len(worst_events), 3), dtype=np.int64)
            for i, e in enumerate(worst_events):
                orig_assignments[i, 0] = X[idx, 3 * e + 0]  # inst
                orig_assignments[i, 1] = X[idx, 3 * e + 1]  # room
                orig_assignments[i, 2] = X[idx, 3 * e + 2]  # time

            # ── SORT BY RESTRICTIVENESS ────────────────────────────
            # Most constrained events placed first (smallest domain)
            restrictiveness = np.zeros(len(worst_events), dtype=np.float64)
            for i, e in enumerate(worst_events):
                r_len = max(1, int(eng.room_dom_len[e]))
                t_len = max(1, int(eng.time_dom_len[e]))
                restrictiveness[i] = r_len * t_len
            insertion_order = np.argsort(restrictiveness)
            worst_events = worst_events[insertion_order]
            orig_assignments = orig_assignments[insertion_order]

            # ── "DELETE" RUINED EVENTS ─────────────────────────────
            # Set to sentinel values that won't conflict
            # (place in impossible slot so they don't interfere with
            #  greedy reinsertion of other events)
            for e in worst_events:
                X[idx, 3 * e + 0] = 0  # inst (will be overwritten)
                X[idx, 3 * e + 1] = 0  # room (will be overwritten)
                X[idx, 3 * e + 2] = 0  # time = 0 (minimal interference)

            # ── RECREATE PHASE: greedy best-fit reinsertion ────────
            for i, e in enumerate(worst_events):
                best_score = np.inf
                best_inst = int(orig_assignments[i, 0])
                best_room = int(orig_assignments[i, 1])
                best_time = int(orig_assignments[i, 2])

                # Get valid domains
                n_rooms = int(eng.room_dom_len[e])
                n_times = int(eng.time_dom_len[e])
                n_insts = int(eng.inst_dom_len[e])

                if n_rooms == 0 or n_times == 0 or n_insts == 0:
                    # No valid domain — restore original
                    X[idx, 3 * e + 0] = best_inst
                    X[idx, 3 * e + 1] = best_room
                    X[idx, 3 * e + 2] = best_time
                    continue

                valid_rooms = eng.room_domains[e, :n_rooms]
                valid_times = eng.time_domains[e, :n_times]
                valid_insts = eng.inst_domains[e, :n_insts]

                # Sample candidates to keep runtime bounded
                if len(valid_rooms) > self.max_room_candidates:
                    r_idx = rng.choice(
                        len(valid_rooms), self.max_room_candidates, replace=False
                    )
                    candidate_rooms = valid_rooms[r_idx]
                else:
                    candidate_rooms = valid_rooms

                if len(valid_times) > self.max_time_candidates:
                    t_idx = rng.choice(
                        len(valid_times), self.max_time_candidates, replace=False
                    )
                    candidate_times = valid_times[t_idx]
                else:
                    candidate_times = valid_times

                # Use the first valid instructor (keep original if valid)
                use_inst = best_inst
                if n_insts > 0 and not np.isin(best_inst, valid_insts):
                    use_inst = int(valid_insts[0])

                # ── GREEDY SCAN: find best (time, room) combo ──────
                # Vectorized marginal scoring: temporarily place event
                # at each candidate and measure conflict with current X
                for t_cand in candidate_times:
                    for r_cand in candidate_rooms:
                        score = self._marginal_conflict_score(
                            eng,
                            X,
                            idx,
                            e,
                            int(use_inst),
                            int(r_cand),
                            int(t_cand),
                        )
                        if score < best_score:
                            best_score = score
                            best_inst = int(use_inst)
                            best_room = int(r_cand)
                            best_time = int(t_cand)

                # Commit best placement
                X[idx, 3 * e + 0] = best_inst
                X[idx, 3 * e + 1] = best_room
                X[idx, 3 * e + 2] = best_time

            # ── EVALUATE: did the ruin+recreate improve? ───────────
            # Quick check via _score_all_batch on the single individual
            new_scores = eng._score_all_batch(X[idx : idx + 1])  # (1, E)
            new_total = int(new_scores.sum())
            old_total = int(row_scores.sum())

            if new_total < old_total:
                total_improved += 1
                logger.debug(
                    "LNS: individual %d improved %d → %d (Δ=%d, ruined=%d events)",
                    idx,
                    old_total,
                    new_total,
                    new_total - old_total,
                    len(worst_events),
                )
            else:
                # Rollback — restore original assignments
                for i, e in enumerate(worst_events):
                    X[idx, 3 * e + 0] = orig_assignments[i, 0]
                    X[idx, 3 * e + 1] = orig_assignments[i, 1]
                    X[idx, 3 * e + 2] = orig_assignments[i, 2]
                logger.debug(
                    "LNS: individual %d rollback (old=%d, new=%d)",
                    idx,
                    old_total,
                    new_total,
                )

        logger.debug(
            "LNS Ruin&Recreate: %d/%d individuals improved",
            total_improved,
            len(selected),
        )

    @staticmethod
    def _marginal_conflict_score(
        eng,
        X: np.ndarray,
        ind: int,
        event: int,
        inst: int,
        room: int,
        time: int,
    ) -> int:
        """Fast marginal conflict count for placing one event.

        Counts how many OTHER events in this individual conflict with
        the proposed (inst, room, time) assignment.  O(E) via vectorized
        overlap detection — does NOT call the full evaluator.
        """
        from src.pipeline.bitset_time import T as T_

        E = eng.n_events
        dur_e = int(eng.durations[event])

        # Current assignments for this individual
        all_inst = X[ind, 0::3]  # (E,)
        all_room = X[ind, 1::3]
        all_time = X[ind, 2::3]
        all_dur = eng.durations  # (E,)

        # Event's occupied time range: [time, time + dur_e)
        e_start = time
        e_end = time + dur_e

        # All other events' time ranges
        o_starts = all_time.astype(np.int64)
        o_ends = o_starts + all_dur

        # Time overlap mask (exclude self)
        overlap = (o_starts < e_end) & (e_start < o_ends)
        overlap[event] = False

        # Room conflicts: same room AND time overlap
        room_conflicts = int(np.sum(overlap & (all_room == room)))

        # Instructor conflicts: same instructor AND time overlap
        inst_conflicts = int(np.sum(overlap & (all_inst == inst)))

        # Group conflicts: shared student group AND time overlap
        score = room_conflicts + inst_conflicts

        # Check group conflicts if available
        if hasattr(eng, "_event_groups"):
            my_groups = set(eng._event_groups[event])
            if my_groups:
                for other_e in np.where(overlap)[0]:
                    other_groups = eng._event_groups[int(other_e)]
                    if my_groups.intersection(other_groups):
                        score += 1

        # Domain violations (heavy penalty)
        if hasattr(eng, "inst_avail"):
            for q in range(e_start, min(e_end, T_)):
                if not eng.inst_avail[inst, q]:
                    score += 10

        return score
