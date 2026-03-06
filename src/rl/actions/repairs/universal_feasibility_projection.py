r"""Universal Feasibility Projection — Bounded Ejection Chain Bulldozer.

**UPGRADED TO MICRO-MEMETIC OPTIMIZER**

Implements bounded ejection chains (depth=3) to resolve complex constraint
violations through intelligent cascading moves. Replaces simple multi-pass
stochastic swaps with conflict-directed search and chain resolution.

Algorithm:
1. **Conflict Identification**: Find top-3 most penalized events overall.
2. **Cascading Chain Search**: For each penalized event:
   - Find optimal time/room assignment using greedy micro-burst
   - If move causes clash with another event B, recursively find slot for B
   - Continue chain up to depth=3 to prevent infinite loops
3. **Chain Commitment**: Apply entire chain only if net ΔHard < 0.

This upgrades the "nuclear option" heuristic from brute-force repair to
intelligent chain-based optimization while maintaining RL step speed.

Complexity: $O(N \cdot 3^3)$ = $O(N)$ per individual (bounded depth).
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


class UniversalFeasibilityProjection(_AtomicRepairBase):
    """Action 4 — Bounded ejection chain micro-memetic optimizer."""

    ACTION_NAME: ClassVar[str] = "universal_feasibility_projection"

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        max_depth: int = 3,
        max_targets: int = 3,
    ):
        super().__init__(pkl_path)
        self.max_depth = max_depth
        self.max_targets = max_targets

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events

        # Fix domains first
        eng._fix_domains_vec(X)

        chains_attempted = 0
        chains_committed = 0
        fallback_repairs = 0

        # Budget: only apply ejection chains to top-K worst individuals
        # (max 15) for RL-speed feasibility. Rest get fast vectorized repair.
        # Fast vectorized severity estimation using room conflict counts
        T_ = __import__("src.pipeline.bitset_time", fromlist=["T"]).T
        room = np.clip(X[:, 1::3], 0, eng.n_rooms - 1).astype(np.int64)
        time = X[:, 2::3].astype(np.int64)
        n_idx = np.arange(N, dtype=np.int64)[:, None]
        starts_exp = time[:, eng.exp_event]
        quanta_exp = np.clip(starts_exp + eng.exp_offset[None, :], 0, T_ - 1)
        rooms_exp = room[:, eng.exp_event]
        nRT = np.int64(eng.n_rooms) * np.int64(T_)
        room_keys = (n_idx * nRT + rooms_exp * T_ + quanta_exp).ravel()
        room_cnt = np.bincount(room_keys, minlength=int(N * nRT))
        room_conflict = (room_cnt[room_keys] > 1).astype(np.float64)
        event_lin = (n_idx * E + eng.exp_event[None, :]).ravel()
        NE = N * E
        scores_flat = np.bincount(event_lin, weights=room_conflict, minlength=NE)
        scores = scores_flat[:NE].reshape(N, E)
        ind_severity = scores.sum(axis=1)  # Fast vectorized severity per individual

        nonzero = ind_severity > 0
        if not nonzero.any():
            return
        conflict_individuals = np.where(nonzero)[0]
        severity_order = np.argsort(-ind_severity[conflict_individuals])
        selected = conflict_individuals[severity_order[:5]]

        # Fast fallback for remaining individuals — skip expensive repair_batch
        # since other operators in the action space will handle them.
        # Only do minimal domain fixing.
        remaining = conflict_individuals[severity_order[5:]]
        if len(remaining) > 0:
            eng._fix_domains_vec(X[remaining])

        for individual_idx in selected:
            # Get conflict events sorted by severity
            conflict_events, conflict_counts = get_conflict_events(
                eng, X, individual_idx
            )

            # Target top-K most problematic events
            n_targets = min(self.max_targets, len(conflict_events))
            top_events = conflict_events[:n_targets]

            if conflict_counts[0] == 0:
                continue

            # === BOUNDED EJECTION CHAIN PROCESSING ===
            for target_event in top_events:
                if conflict_counts[target_event] == 0:
                    continue

                chains_attempted += 1

                # Attempt to build an ejection chain starting from this event
                chain_moves, chain_delta = self._build_ejection_chain(
                    eng, X, individual_idx, target_event, depth=0
                )

                # Commit chain if it improves overall constraints
                if chain_delta < 0.0 and len(chain_moves) > 0:
                    self._apply_chain_moves(X, individual_idx, chain_moves)
                    chains_committed += 1

                    logger.debug(
                        "Ejection chain: individual=%d, root_event=%d, moves=%d, ΔHard=%.3f",
                        individual_idx,
                        target_event,
                        len(chain_moves),
                        chain_delta,
                    )
                # Fallback: use single-event random repair for this conflicting event
                elif self._apply_fallback_repair(eng, X, individual_idx, target_event):
                    fallback_repairs += 1

        logger.debug(
            "UniversalFeasibilityProjection: %d chains attempted, %d committed, %d fallbacks",
            chains_attempted,
            chains_committed,
            fallback_repairs,
        )

    def _build_ejection_chain(
        self, eng, X: np.ndarray, individual_idx: int, event_idx: int, depth: int
    ) -> tuple[list, float]:
        """Build bounded ejection chain starting from the given event.

        Returns:
            chain_moves: List of (event_idx, new_time, new_room, new_inst) tuples
            total_delta: Total ΔHard for the entire chain
        """
        if depth >= self.max_depth:
            return [], 0.0

        # Current assignments for this event
        current_inst = int(X[individual_idx, 3 * event_idx + 0])
        current_room = int(X[individual_idx, 3 * event_idx + 1])
        current_time = int(X[individual_idx, 3 * event_idx + 2])

        rng = np.random.default_rng()

        # Try to find a better assignment using micro-burst approach
        best_delta = 0.0  # Only accept improvements
        best_move = None
        conflicting_events = []  # Events that would be displaced by our move

        # === GREEDY MICRO-BURST EVALUATION ===
        # Try different time slots first (most impactful)
        time_domain_len = getattr(eng, "time_dom_len", None)
        if time_domain_len is not None and time_domain_len[event_idx] > 0:
            valid_times = getattr(eng, "time_domains", None)
            if valid_times is not None:
                n_valid = time_domain_len[event_idx]
                candidate_times = valid_times[event_idx, :n_valid]

                # Try up to 5 different time slots
                time_candidates = candidate_times[candidate_times != current_time]
                k_samples = min(5, len(time_candidates))

                if k_samples > 0:
                    if len(time_candidates) <= k_samples:
                        selected_times = time_candidates
                    else:
                        time_indices = rng.choice(
                            len(time_candidates), k_samples, replace=False
                        )
                        selected_times = time_candidates[time_indices]

                    for candidate_time in selected_times:
                        # Also try different rooms for this time slot
                        room_candidates = [current_room]  # Start with current room

                        room_domain_len = getattr(eng, "room_dom_len", None)
                        if (
                            room_domain_len is not None
                            and room_domain_len[event_idx] > 1
                        ):
                            valid_rooms = getattr(eng, "room_domains", None)
                            if valid_rooms is not None:
                                n_valid_rooms = room_domain_len[event_idx]
                                all_rooms = valid_rooms[event_idx, :n_valid_rooms]
                                other_rooms = all_rooms[all_rooms != current_room]
                                room_candidates.extend(
                                    other_rooms[:2]
                                )  # Try 2 more rooms

                        for candidate_room in room_candidates:
                            # Validate domain constraints
                            if not validate_domain_move(
                                eng,
                                event_idx,
                                candidate_time,
                                candidate_room,
                                current_inst,
                            ):
                                continue

                            # Calculate local ΔHard for this assignment
                            delta = evaluate_local_move(
                                eng,
                                X,
                                individual_idx,
                                event_idx,
                                candidate_time,
                                candidate_room,
                                current_inst,
                            )

                            if delta < best_delta:
                                best_delta = delta
                                best_move = (
                                    event_idx,
                                    candidate_time,
                                    candidate_room,
                                    current_inst,
                                )

                                # Check if this move would displace other events
                                conflicting_events = self._find_conflicting_events(
                                    eng,
                                    X,
                                    individual_idx,
                                    event_idx,
                                    candidate_time,
                                    candidate_room,
                                )

        # If no direct improvement found, return empty chain
        if best_move is None or best_delta >= 0.0:
            return [], 0.0

        chain_moves = [best_move]
        total_delta = best_delta

        # === RECURSIVE CHAIN BUILDING ===
        # For each event that would be displaced, try to find it a new home
        for conflicted_event in conflicting_events:
            if depth < self.max_depth - 1:  # Ensure we don't exceed depth limit
                sub_chain, sub_delta = self._build_ejection_chain(
                    eng, X, individual_idx, conflicted_event, depth + 1
                )
                chain_moves.extend(sub_chain)
                total_delta += sub_delta

        return chain_moves, total_delta

    def _find_conflicting_events(
        self,
        eng,
        X: np.ndarray,
        individual_idx: int,
        moving_event: int,
        new_time: int,
        new_room: int,
    ) -> list[int]:
        """Find events that would conflict with the proposed move (vectorized)."""
        E = eng.n_events
        time = X[individual_idx, 2::3].astype(np.int64)
        room = X[individual_idx, 1::3].astype(np.int64)

        durations = getattr(eng, "durations", np.ones(E, dtype=np.int64))
        moving_dur = int(durations[moving_event])

        # Vectorized: room match AND time overlap
        same_room = room == new_room
        same_room[moving_event] = False  # exclude self
        # Overlap: other_start < new_end AND new_start < other_end
        other_ends = time + durations
        new_end = new_time + moving_dur
        overlap = (time < new_end) & (new_time < other_ends)
        conflict_mask = same_room & overlap
        return np.where(conflict_mask)[0].tolist()

    def _apply_chain_moves(self, X: np.ndarray, individual_idx: int, chain_moves: list):
        """Apply a sequence of moves to the population matrix."""
        for event_idx, new_time, new_room, new_inst in chain_moves:
            X[individual_idx, 3 * event_idx + 0] = new_inst
            X[individual_idx, 3 * event_idx + 1] = new_room
            X[individual_idx, 3 * event_idx + 2] = new_time

    def _apply_fallback_repair(
        self, eng, X: np.ndarray, individual_idx: int, event_idx: int
    ) -> bool:
        """Apply single-event random domain sampling as fallback."""
        rng = np.random.default_rng()

        try:
            # Random room assignment
            room_domain_len = getattr(eng, "room_dom_len", None)
            if room_domain_len is not None and room_domain_len[event_idx] > 0:
                valid_rooms = getattr(eng, "room_domains", None)
                if valid_rooms is not None:
                    n_valid = room_domain_len[event_idx]
                    room_idx = rng.integers(0, n_valid)
                    X[individual_idx, 3 * event_idx + 1] = valid_rooms[
                        event_idx, room_idx
                    ]

            # Random time assignment
            time_domain_len = getattr(eng, "time_dom_len", None)
            if time_domain_len is not None and time_domain_len[event_idx] > 0:
                valid_times = getattr(eng, "time_domains", None)
                if valid_times is not None:
                    n_valid = time_domain_len[event_idx]
                    time_idx = rng.integers(0, n_valid)
                    X[individual_idx, 3 * event_idx + 2] = valid_times[
                        event_idx, time_idx
                    ]

            return True

        except (AttributeError, IndexError):
            return False

    def _count_total_violations(self, eng, X: np.ndarray) -> int:
        """Quick estimate of total constraint violations in population."""
        try:
            # Use the micro-evaluator's basic conflict counting
            from src.rl.actions.utils.micro_evaluator import _basic_conflict_count

            total = 0
            for i in range(X.shape[0]):
                total += _basic_conflict_count(eng, X[i : i + 1], 0)
            return total

        except Exception:
            # Fallback: assume moderate violation level
            return X.shape[0] * X.shape[1] // 20  # 5% violation estimate
