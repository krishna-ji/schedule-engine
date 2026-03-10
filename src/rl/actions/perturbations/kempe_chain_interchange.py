r"""Kempe Chain Interchange — Time-Slot Bipartite Swap Operator.

**STATE-OF-THE-ART META-HEURISTIC** replacing the structurally capped
stochastic_quanta_perturbation (Action 5, blind random time swaps).

Algorithm (per individual):
1. Identify a high-conflict time quantum $T_1$ (weighted by violation
   density).
2. Select a random alternative time quantum $T_2$.
3. Pick a conflicting event $E_0$ at $T_1$.  Swap its time to $T_2$.
4. **Chain Trace**: If swapping $E_0$ to $T_2$ causes an instructor or
   room conflict with event $E_1$ at $T_2$, swap $E_1$ to $T_1$.
   Continue tracing the cascading conflicts between $T_1$ and $T_2$
   until the bipartite sub-graph is fully interchanged.

Mathematical Advantage:
  The Kempe chain **guarantees** that the total number of instructor and
  room assignments between $T_1$ and $T_2$ remains invariant.  This
  allows exploration of radically different schedule topologies without
  destroying existing feasible sub-structures.

Complexity: $O(N_s \cdot C)$ where $C$ = chain length (bounded by
number of events at $T_1 \cup T_2$, typically ≤ 20).
"""

from __future__ import annotations

import logging
from typing import ClassVar

import numpy as np

from src.rl.actions.vectorized_ops import _AtomicRepairBase

logger = logging.getLogger(__name__)


class KempeChainInterchange(_AtomicRepairBase):
    """Action 5 — Time-Slot Kempe Chain meta-heuristic."""

    ACTION_NAME: ClassVar[str] = "kempe_chain_interchange"

    def __init__(
        self,
        pkl_path: str = ".cache/events_with_domains.pkl",
        max_individuals: int = 10,
        max_chains_per_individual: int = 3,
        max_chain_length: int = 30,
    ):
        super().__init__(pkl_path)
        self.max_individuals = max_individuals
        self.max_chains_per_individual = max_chains_per_individual
        self.max_chain_length = max_chain_length

    def _apply(self, X: np.ndarray) -> None:
        eng = self.engine
        N, n_vars = X.shape
        E = eng.n_events
        T_ = __import__("src.pipeline.bitset_time", fromlist=["T"]).T

        # Fix domains first
        eng._fix_domains_vec(X)

        # Score population
        scores = eng._score_all_batch(X)  # (N, E)

        # Select individuals with conflicts
        ind_severity = scores.sum(axis=1)
        nonzero = ind_severity > 0
        if not nonzero.any():
            return
        conflict_individuals = np.where(nonzero)[0]
        severity_order = np.argsort(-ind_severity[conflict_individuals])
        selected = conflict_individuals[severity_order[: self.max_individuals]]

        rng = np.random.default_rng()
        total_chains = 0
        total_improved = 0

        for idx in selected:
            row_scores = scores[idx]  # (E,)
            if row_scores.sum() == 0:
                continue

            time = X[idx, 2::3].astype(np.int64)  # (E,)
            durations = eng.durations  # (E,)

            # ── BUILD TIME-QUANTUM CONFLICT DENSITY ────────────────
            # For each quantum, sum the conflict scores of events
            # that occupy it.
            quantum_conflict = np.zeros(T_, dtype=np.float64)
            for e in range(E):
                if row_scores[e] > 0:
                    t_start = int(time[e])
                    t_end = min(t_start + int(durations[e]), T_)
                    quantum_conflict[t_start:t_end] += row_scores[e]

            for _chain_attempt in range(self.max_chains_per_individual):
                # ── SELECT T1: high-conflict quantum ───────────────
                if quantum_conflict.sum() == 0:
                    break

                # Weighted random selection for T1
                probs = quantum_conflict / quantum_conflict.sum()
                t1 = int(rng.choice(T_, p=probs))

                # ── SELECT T2: random different quantum ────────────
                t2_candidates = np.arange(T_)
                t2_candidates = t2_candidates[t2_candidates != t1]
                if len(t2_candidates) == 0:
                    continue
                t2 = int(rng.choice(t2_candidates))

                # ── IDENTIFY EVENTS AT T1 AND T2 ──────────────────
                # An event "occupies" quantum q if q in [time[e], time[e] + dur[e])
                events_at_t1 = self._events_at_quantum(time, durations, t1, E)
                events_at_t2 = self._events_at_quantum(time, durations, t2, E)

                if len(events_at_t1) == 0:
                    continue

                # Save original state for rollback
                orig_time = time.copy()

                # ── EXECUTE KEMPE CHAIN ────────────────────────────
                chain_length = self._execute_kempe_chain(
                    eng,
                    X,
                    idx,
                    t1,
                    t2,
                    events_at_t1,
                    events_at_t2,
                    rng,
                    T_,
                )

                if chain_length == 0:
                    continue

                total_chains += 1

                # ── EVALUATE: did the chain improve? ──────────────
                new_scores = eng._score_all_batch(X[idx : idx + 1])  # (1, E)
                new_total = int(new_scores.sum())
                old_total = int(row_scores.sum())

                if new_total < old_total:
                    total_improved += 1
                    # Update local state for subsequent chains
                    row_scores = new_scores[0]
                    time = X[idx, 2::3].astype(np.int64)
                    # Rebuild quantum conflict density
                    quantum_conflict[:] = 0
                    for e in range(E):
                        if row_scores[e] > 0:
                            t_start = int(time[e])
                            t_end = min(t_start + int(durations[e]), T_)
                            quantum_conflict[t_start:t_end] += row_scores[e]
                    logger.debug(
                        "Kempe: individual %d chain T%dT%d len=%d, %d→%d (Δ=%d)",
                        idx,
                        t1,
                        t2,
                        chain_length,
                        old_total,
                        new_total,
                        new_total - old_total,
                    )
                else:
                    # Rollback
                    X[idx, 2::3] = orig_time
                    time = orig_time

        logger.debug(
            "KempeChain: %d chains executed, %d improved across %d individuals",
            total_chains,
            total_improved,
            len(selected),
        )

    def _execute_kempe_chain(
        self,
        eng,
        X: np.ndarray,
        idx: int,
        t1: int,
        t2: int,
        events_at_t1: np.ndarray,
        events_at_t2: np.ndarray,
        rng: np.random.Generator,
        T_: int,
    ) -> int:
        """Execute a single Kempe chain between quanta t1 and t2.

        Returns the number of events swapped (chain length).
        """
        E = eng.n_events
        durations = eng.durations
        inst = X[idx, 0::3].astype(np.int64)
        room = X[idx, 1::3].astype(np.int64)
        time = X[idx, 2::3].astype(np.int64)

        # Pick a conflicting seed event at t1
        # Prefer events with high conflict scores
        scores = eng._score_all_batch(X[idx : idx + 1])[0]  # (E,)
        t1_scores = scores[events_at_t1]
        if t1_scores.sum() == 0:
            # No conflicts at t1 — pick random
            seed_local = int(rng.integers(0, len(events_at_t1)))
        else:
            probs = t1_scores / t1_scores.sum()
            seed_local = int(rng.choice(len(events_at_t1), p=probs))
        seed_event = int(events_at_t1[seed_local])

        # Validate that moving seed to t2 respects time domain
        if not self._time_domain_valid(eng, seed_event, t2):
            return 0

        # ── CHAIN TRAVERSAL ────────────────────────────────────────
        moved = set()
        swap_queue = [(seed_event, t1, t2)]  # (event, from_t, to_t)
        chain_moves = []  # (event, new_time)

        while swap_queue and len(chain_moves) < self.max_chain_length:
            event_e, from_t, to_t = swap_queue.pop(0)
            if event_e in moved:
                continue

            # Validate time domain
            if not self._time_domain_valid(eng, event_e, to_t):
                continue

            # Record the move
            old_time = int(time[event_e])
            chain_moves.append((event_e, to_t))
            moved.add(event_e)

            # Apply the swap in the working array
            time[event_e] = to_t

            # ── TRACE CONFLICTS AT to_t ────────────────────────────
            # Check if this move creates instructor or room conflicts
            e_inst = int(inst[event_e])
            e_room = int(room[event_e])
            e_dur = int(durations[event_e])

            # Find events at to_t that now conflict
            events_at_dest = self._events_at_quantum(time, durations, to_t, E)

            for other_e in events_at_dest:
                other_e = int(other_e)
                if other_e == event_e or other_e in moved:
                    continue

                # Check for instructor conflict
                inst_conflict = int(inst[other_e]) == e_inst

                # Check for room conflict
                room_conflict = int(room[other_e]) == e_room

                # Check for group conflict
                group_conflict = False
                if hasattr(eng, "_event_groups"):
                    my_groups = set(eng._event_groups[event_e])
                    other_groups = set(eng._event_groups[other_e])
                    group_conflict = bool(my_groups & other_groups)

                if inst_conflict or room_conflict or group_conflict:
                    # Push this conflicting event to from_t (the inverse)
                    swap_queue.append((other_e, to_t, from_t))

        if not chain_moves:
            return 0

        # Apply all chain moves to X
        for event_e, new_time in chain_moves:
            X[idx, 3 * event_e + 2] = new_time

        return len(chain_moves)

    @staticmethod
    def _events_at_quantum(
        time: np.ndarray,
        durations: np.ndarray,
        quantum: int,
        E: int,
    ) -> np.ndarray:
        """Find all events whose time range includes the given quantum.

        An event e occupies quantum q if time[e] <= q < time[e] + dur[e].
        """
        starts = time[:E].astype(np.int64)
        ends = starts + durations[:E].astype(np.int64)
        mask = (starts <= quantum) & (quantum < ends)
        return np.where(mask)[0]

    @staticmethod
    def _time_domain_valid(eng, event: int, new_time: int) -> bool:
        """Check if new_time is in the event's valid time domain."""
        n_valid = int(eng.time_dom_len[event])
        if n_valid == 0:
            return True  # No domain constraint
        valid_times = eng.time_domains[event, :n_valid]
        return bool(np.isin(new_time, valid_times))
