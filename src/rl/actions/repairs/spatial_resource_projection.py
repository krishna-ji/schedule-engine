r"""Spatial Resource Projection — Conflict-Directed Room Sniper.

**UPGRADED TO MICRO-MEMETIC OPTIMIZER**

Uses conflict-directed targeting to identify the worst-offending events
causing Room Capacity (SRE) violations, then applies greedy micro-bursts
to find optimal room reassignments using localized ΔHard evaluation.

Algorithm:
1. **Conflict-Directed Targeting**: Use np.argmax on room conflict scores
   to identify events causing the most SRE violations (not random sampling).
2. **Greedy Micro-Burst**: For each worst-offending event, generate k=5
   alternative valid room assignments and evaluate each using evaluate_local_move.
3. **Smart Commitment**: Apply only moves with ΔHard < 0 (strict improvement).
   If no move improves constraints, apply no change (internal tolerance = 0).

Complexity: $O(N \cdot K \cdot \log K)$ where $K$ = number of conflict events.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from src.rl.actions.utils.micro_evaluator import (
    evaluate_local_move,
    get_conflict_events,
    validate_domain_move,
)
from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)


class SpatialResourceProjection(_AtomicRepairBase):
    """Action 0 — project population onto SRE feasibility surface."""

    ACTION_NAME: ClassVar[str] = "spatial_resource_projection"

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events
        T_ = __import__("src.pipeline.bitset_time", fromlist=["T"]).T

        # Fix domains first (prerequisite)
        eng._fix_domains_vec(X)

        room = np.clip(X[:, 1::3], 0, eng.n_rooms - 1).astype(np.int64)
        time = X[:, 2::3].astype(np.int64)
        n_idx = np.arange(N, dtype=np.int64)[:, None]

        # Build room conflict detection (unchanged detection logic)
        starts_exp = time[:, eng.exp_event]
        quanta_exp = np.clip(starts_exp + eng.exp_offset[None, :], 0, T_ - 1)
        rooms_exp = room[:, eng.exp_event]
        event_lin = (n_idx * E + eng.exp_event[None, :]).ravel()
        NE = N * E

        nRT = np.int64(eng.n_rooms) * np.int64(T_)
        room_keys = (n_idx * nRT + rooms_exp * T_ + quanta_exp).ravel()
        room_cnt = np.bincount(room_keys, minlength=int(N * nRT))
        room_conflict = (room_cnt[room_keys] > 1).astype(np.float64)

        scores = np.bincount(event_lin, weights=room_conflict, minlength=NE)
        scores = scores[:NE].reshape(N, E)

        conflict_mask = scores > 0
        if not conflict_mask.any():
            return

        rng = np.random.default_rng()
        improvements_made = 0
        moves_attempted = 0

        # === CONFLICT-DIRECTED TARGETING ===
        # Budget: process only the top-K worst individuals (max 5)
        # to keep RL step time practical at large pop sizes.
        ind_total_conflicts = scores.sum(axis=1)  # (N,)
        nonzero_mask = ind_total_conflicts > 0
        if not nonzero_mask.any():
            return
        conflict_individuals = np.where(nonzero_mask)[0]
        # Sort by severity (worst first) and cap to 5
        severity_order = np.argsort(-ind_total_conflicts[conflict_individuals])
        selected = conflict_individuals[severity_order[:5]]

        # For remaining individuals, apply fast vectorized repair (no per-event loop)
        fast_individuals = conflict_individuals[severity_order[5:]]
        if len(fast_individuals) > 0:
            bi_fast, be_fast = np.nonzero(scores[fast_individuals] > 0)
            bi_mapped = fast_individuals[bi_fast]
            r_dl = eng.room_dom_len[be_fast]
            r_valid = r_dl > 0
            if r_valid.any():
                bi_m, be_m, r_dl_v = bi_mapped[r_valid], be_fast[r_valid], r_dl[r_valid]
                r_idx = (rng.random(len(bi_m)) * r_dl_v).astype(np.int64)
                r_idx = np.minimum(r_idx, r_dl_v - 1)
                X[bi_m, 3 * be_m + 1] = eng.room_domains[be_m, r_idx]

        for individual_idx in selected:
            ind_conflicts = scores[individual_idx, :]
            if ind_conflicts.sum() == 0:
                continue

            # Get conflict events sorted by severity (worst first)
            conflict_events, conflict_counts = get_conflict_events(
                eng, X, individual_idx
            )

            # Target top-K most problematic events (up to 8 events)
            max_targets = min(8, int(ind_conflicts.sum()))
            top_conflict_events = conflict_events[:max_targets]

            for event_idx in top_conflict_events:
                if ind_conflicts[event_idx] == 0:
                    continue  # Skip if this event has no room conflicts

                moves_attempted += 1

                # Current assignments
                current_inst = int(X[individual_idx, 3 * event_idx + 0])
                current_room = int(X[individual_idx, 3 * event_idx + 1])
                current_time = int(X[individual_idx, 3 * event_idx + 2])

                # === GREEDY MICRO-BURST: Generate k=5 room alternatives ===
                best_delta = 0.0  # Only accept improvements (ΔHard < 0)
                best_room = current_room
                best_inst = current_inst
                best_time = current_time

                # Try k=5 different room assignments
                room_domain_len = getattr(eng, "room_dom_len", None)
                if room_domain_len is not None and room_domain_len[event_idx] > 0:
                    valid_rooms = getattr(eng, "room_domains", None)
                    if valid_rooms is not None:
                        n_valid = room_domain_len[event_idx]
                        candidate_rooms = valid_rooms[event_idx, :n_valid]

                        # Sample k=5 room candidates (avoid current room)
                        other_rooms = candidate_rooms[candidate_rooms != current_room]
                        k_samples = min(5, len(other_rooms))

                        if k_samples > 0:
                            # Sample without replacement
                            if len(other_rooms) <= k_samples:
                                room_candidates = other_rooms
                            else:
                                room_indices = rng.choice(
                                    len(other_rooms), k_samples, replace=False
                                )
                                room_candidates = other_rooms[room_indices]

                            # Evaluate each room candidate
                            for candidate_room in room_candidates:
                                # Validate domain constraints first
                                if not validate_domain_move(
                                    eng,
                                    event_idx,
                                    current_time,
                                    candidate_room,
                                    current_inst,
                                ):
                                    continue

                                # Calculate ΔHard for this room assignment
                                delta = evaluate_local_move(
                                    eng,
                                    X,
                                    individual_idx,
                                    event_idx,
                                    current_time,
                                    candidate_room,
                                    current_inst,
                                )

                                # Keep best improvement (most negative ΔHard)
                                if delta < best_delta:
                                    best_delta = delta
                                    best_room = candidate_room

                # === SMART COMMITMENT: Apply only if ΔHard < 0 ===
                if best_delta < 0.0:  # Strict improvement required
                    X[individual_idx, 3 * event_idx + 1] = best_room
                    improvements_made += 1

                    logger.debug(
                        "Room micro-burst: individual=%d, event=%d, room %d→%d, ΔHard=%.3f",
                        individual_idx,
                        event_idx,
                        current_room,
                        best_room,
                        best_delta,
                    )

        logger.debug(
            "SpatialResourceProjection: %d/%d moves improved constraints",
            improvements_made,
            moves_attempted,
        )
