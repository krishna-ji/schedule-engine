r"""Meridian Compaction Heuristic — Feasibility-Gated Soft Optimizer.

**UPGRADED WITH FEASIBILITY GATING**

Targets soft constraint optimization (MIP + CSC) while preventing hard
constraint destruction. All assignment moves are now wrapped with
evaluate_local_move to ensure ΔHard ≤ 0 before commitment.

Algorithm:
1. **Lunch-window evacuation**: Move events from lunch slots to non-lunch
   alternatives, but only if ΔHard ≤ 0 (feasibility gate).
2. **Gap compaction**: Shift events to earliest domain slots to reduce
   schedule gaps, but reject moves that increase hard violations.

This prevents the compaction heuristic from destroying feasibility while
still targeting MIP (Meridian Interval Preservation) and CSC (Cohort
Schedule Contiguity) improvements.

Complexity: $O(N \cdot E \cdot K)$ where $K$ = moves evaluated per event.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from src.rl.actions.utils.micro_evaluator import (
    evaluate_local_move,
    validate_domain_move,
)
from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)

# Time system constants (must match soft_evaluator_vectorized.py)
_QUANTA_PER_DAY = 7
_LUNCH_WITHIN_DAY = np.array([2, 3, 4], dtype=np.int64)  # within-day indices


class MeridianCompactionHeuristic(_AtomicRepairBase):
    """Action 7 — soft-constraint optimisation targeting MIP and CSC."""

    ACTION_NAME: ClassVar[str] = "meridian_compaction"

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        compaction_rate: float = 0.10,
    ):
        super().__init__(pkl_path)
        self.compaction_rate = compaction_rate

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N = X.shape[0]
        E = eng.n_events
        rng = np.random.default_rng()

        # Current time slots — shape (N, E)
        time = X[:, 2::3].astype(np.int64)

        lunch_evacuations = 0
        compaction_moves = 0

        # ── Pass 1: Feasibility-Gated Lunch-window evacuation ─────────
        # Compute within-day quantum: q_wd = time_slot % QUANTA_PER_DAY
        within_day = time % _QUANTA_PER_DAY  # (N, E)

        # Boolean mask: is this event in the lunch window?
        in_lunch = np.isin(within_day, _LUNCH_WITHIN_DAY)  # (N, E)

        if in_lunch.any():
            # Budget: only process top-15 individuals with most lunch conflicts
            ind_lunch_count = in_lunch.sum(axis=1)
            sorted_inds = np.argsort(-ind_lunch_count)
            selected_inds = sorted_inds[: min(5, N)]

            for individual_idx in selected_inds:
                if ind_lunch_count[individual_idx] == 0:
                    break
                for event_idx in range(E):
                    if not in_lunch[individual_idx, event_idx]:
                        continue

                    current_inst = int(X[individual_idx, 3 * event_idx + 0])
                    current_room = int(X[individual_idx, 3 * event_idx + 1])
                    current_time = int(X[individual_idx, 3 * event_idx + 2])

                    # Try to find a non-lunch alternative
                    t_dl = eng.time_dom_len[event_idx]
                    if t_dl <= 1:  # Need at least 2 options to relocate
                        continue

                    # Sample from time domain and test non-lunch candidates
                    best_delta = 0.0  # Only accept non-degrading moves
                    best_time = current_time
                    attempts = 0
                    max_attempts = min(5, t_dl)  # Try up to 5 alternatives

                    for _ in range(max_attempts):
                        attempts += 1
                        t_idx = rng.integers(0, t_dl)
                        candidate_time = eng.time_domains[event_idx, t_idx]

                        # Skip if same as current or still in lunch window
                        if candidate_time == current_time:
                            continue
                        cand_wd = candidate_time % _QUANTA_PER_DAY
                        if cand_wd in _LUNCH_WITHIN_DAY:
                            continue

                        # Validate domain constraints
                        if not validate_domain_move(
                            eng, event_idx, candidate_time, current_room, current_inst
                        ):
                            continue

                        # === FEASIBILITY GATE: Check ΔHard ===
                        delta = evaluate_local_move(
                            eng,
                            X,
                            individual_idx,
                            event_idx,
                            candidate_time,
                            current_room,
                            current_inst,
                        )

                        # Accept if improvement or neutral (ΔHard ≤ 0)
                        if delta <= best_delta:
                            best_delta = delta
                            best_time = candidate_time

                    # Commit the best feasible move (if any improvement/neutral)
                    if best_time != current_time and best_delta <= 0.0:
                        X[individual_idx, 3 * event_idx + 2] = best_time
                        lunch_evacuations += 1

                        logger.debug(
                            "Lunch evacuation: ind=%d, event=%d, time %d→%d, ΔHard=%.3f",
                            individual_idx,
                            event_idx,
                            current_time,
                            best_time,
                            best_delta,
                        )

            logger.debug(
                "MeridianCompaction pass 1: %d/%d events evacuated from lunch",
                lunch_evacuations,
                int(in_lunch.sum()),
            )

        # ── Pass 2: Feasibility-Gated Gap compaction ──────────────────
        compact_mask = rng.random((N, E)) < self.compaction_rate

        if compact_mask.any():
            # Budget: only process top-15 individuals
            ind_compact_count = compact_mask.sum(axis=1)
            sorted_inds = np.argsort(-ind_compact_count)
            selected_inds = sorted_inds[: min(5, N)]

            for individual_idx in selected_inds:
                if ind_compact_count[individual_idx] == 0:
                    break
                for event_idx in range(E):
                    if not compact_mask[individual_idx, event_idx]:
                        continue

                    current_inst = int(X[individual_idx, 3 * event_idx + 0])
                    current_room = int(X[individual_idx, 3 * event_idx + 1])
                    current_time = int(X[individual_idx, 3 * event_idx + 2])

                    # Try earliest domain slot (compaction target)
                    t_dl = eng.time_dom_len[event_idx]
                    if t_dl == 0:
                        continue

                    # Domain arrays are sorted, so index 0 = earliest slot
                    earliest_time = eng.time_domains[event_idx, 0]

                    if earliest_time == current_time:
                        continue  # Already at earliest slot

                    # Validate domain constraints
                    if not validate_domain_move(
                        eng, event_idx, earliest_time, current_room, current_inst
                    ):
                        continue

                    # === FEASIBILITY GATE: Check ΔHard ===
                    delta = evaluate_local_move(
                        eng,
                        X,
                        individual_idx,
                        event_idx,
                        earliest_time,
                        current_room,
                        current_inst,
                    )

                    # Only commit if feasible (ΔHard ≤ 0)
                    if delta <= 0.0:
                        X[individual_idx, 3 * event_idx + 2] = earliest_time
                        compaction_moves += 1

                        logger.debug(
                            "Gap compaction: ind=%d, event=%d, time %d→%d, ΔHard=%.3f",
                            individual_idx,
                            event_idx,
                            current_time,
                            earliest_time,
                            delta,
                        )

            logger.debug(
                "MeridianCompaction pass 2: %d/%d events compacted (rate=%.2f)",
                compaction_moves,
                int(compact_mask.sum()),
                self.compaction_rate,
            )
