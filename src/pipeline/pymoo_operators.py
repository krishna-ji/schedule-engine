"""Pymoo operators for scheduling: Sampling, Crossover, Mutation, Repair.

These operators work on the 3×E interleaved chromosome format:
    X[3e+0] = instructor_idx, X[3e+1] = room_idx, X[3e+2] = start_quanta

All operators respect event domains (allowed_instructors, allowed_rooms,
allowed_starts) stored in the events_with_domains.pkl.
"""

from __future__ import annotations

import logging
import pickle

import numpy as np

try:
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.sampling import Sampling
except ImportError as err:
    raise ImportError("pymoo is required: pip install pymoo>=0.6") from err

from .repair_operator import SchedulingRepair

# =====================================================================
# Sampling: Constructive Initialization
# =====================================================================


class ConstructiveSampling(Sampling):
    """Generate initial population using constructive heuristic.

    Uses SchedulingRepair.construct_feasible() which greedily places
    events (tightest groups first) with conflict avoidance.
    """

    def __init__(self, pkl_path: str = "events_with_domains.pkl"):
        super().__init__()
        self.repairer = SchedulingRepair(pkl_path)

    def _do(self, problem, n_samples, **kwargs):
        import time as _time

        X = np.zeros((n_samples, problem.n_var), dtype=int)
        t0 = _time.perf_counter()
        for i in range(n_samples):
            rng = np.random.default_rng(i)
            X[i] = self.repairer.construct_feasible(rng)
            elapsed = _time.perf_counter() - t0
            if (i + 1) % 10 == 0 or i == n_samples - 1:
                logging.getLogger(__name__).info(
                    "Constructive sampling: %d/%d (%.1fs)",
                    i + 1,
                    n_samples,
                    elapsed,
                )
        return X


class RandomDomainSampling(Sampling):
    """Generate initial population by sampling random domain-valid chromosomes.

    Fully vectorized — no Python loops over individuals or events.
    Uses padded domain matrices with random index selection.
    """

    def __init__(self, pkl_path: str = "events_with_domains.pkl"):
        super().__init__()
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self._n_events = len(data["events"])
        E = self._n_events

        # Pre-pad ragged domain arrays into dense matrices
        domain_lists = [
            data["allowed_instructors"],
            data["allowed_rooms"],
            data["allowed_starts"],
        ]
        self._dom_padded: list[np.ndarray] = []
        self._dom_len: list[np.ndarray] = []
        for domains in domain_lists:
            max_d = max((len(d) for d in domains), default=1) or 1
            padded = np.zeros((E, max_d), dtype=np.int64)
            lengths = np.zeros(E, dtype=np.int64)
            for e, d in enumerate(domains):
                dl = len(d)
                if dl > 0:
                    lengths[e] = dl
                    padded[e, :dl] = d
            self._dom_padded.append(padded)
            self._dom_len.append(lengths)

    def _do(self, problem, n_samples, **kwargs):
        import time as _time

        t0 = _time.perf_counter()
        E = self._n_events
        X = np.zeros((n_samples, problem.n_var), dtype=int)

        # Vectorized: for each gene type, sample random index into padded domain
        e_idx = np.arange(E, dtype=np.int64)  # (E,) — reused for fancy indexing
        for g in range(3):
            dom_padded = self._dom_padded[g]   # (E, max_dom) int64
            dom_len = self._dom_len[g]         # (E,) int64

            # Random index per (individual, event): shape (N, E)
            safe_len = np.maximum(dom_len, 1)  # avoid div-by-zero
            rand_idx = (
                np.random.random((n_samples, E)) * safe_len[np.newaxis, :]
            ).astype(np.int64)
            rand_idx = np.minimum(rand_idx, safe_len[np.newaxis, :] - 1)

            # Gather: dom_padded[e, rand_idx[n, e]] for all (n, e)
            X[:, g::3] = dom_padded[e_idx[np.newaxis, :], rand_idx]

        elapsed = _time.perf_counter() - t0
        logging.getLogger(__name__).info(
            "Random domain sampling: %d (%.3fs, vectorized)",
            n_samples,
            elapsed,
        )
        return X


# =====================================================================
# Crossover: Event-block crossover
# =====================================================================


class EventBlockCrossover(Crossover):
    """Swap subsets of events (not raw ints) between two parents.

    For each event, randomly choose which parent to inherit from.
    This preserves the (instructor, room, time) triple per event.
    """

    def __init__(self, prob: float = 0.5, **kwargs):
        # 2 parents in, 2 offspring out
        super().__init__(n_parents=2, n_offsprings=2, **kwargs)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        # X shape: (n_parents, n_matings, n_var) — pymoo swaps axes before calling
        _, n_matings, n_var = X.shape
        E = n_var // 3

        # Random boolean mask per (mating, event): True = take from parent 1
        mask_events = np.random.random((n_matings, E)) < self.prob  # (n_matings, E)
        # Expand to gene-level: repeat each bool 3× for the (instr, room, start) triple
        mask_genes = np.repeat(mask_events, 3, axis=1)  # (n_matings, n_var)

        # Build both offspring via np.where (fully vectorized)
        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=X.dtype)
        Y[0] = np.where(mask_genes, X[0], X[1])
        Y[1] = np.where(mask_genes, X[1], X[0])

        swapped = int(mask_events.sum())
        total = n_matings * E
        logging.getLogger(__name__).debug(
            "Crossover: %d matings, %d/%d events swapped (%.1f%%)",
            n_matings, swapped, total, 100 * swapped / total if total else 0,
        )

        return Y


# =====================================================================
# Mutation: Event-local mutation
# =====================================================================


class EventLocalMutation(Mutation):
    """Mutate individual events by changing instructor, room, or time.

    For each event selected for mutation:
    - Randomly reassign 1-3 of (instructor, room, time)
    - Values drawn from the event's allowed domain
    """

    def __init__(
        self,
        pkl_path: str = "events_with_domains.pkl",
        event_prob: float = 0.05,
    ):
        super().__init__()
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.allowed_instructors = data["allowed_instructors"]
        self.allowed_rooms = data["allowed_rooms"]
        self.allowed_starts = data["allowed_starts"]
        self.n_events = len(data["events"])
        self.event_prob = event_prob

    def _do(self, problem, X, **kwargs):
        n_ind, n_var = X.shape
        E = self.n_events
        Y = X.copy()

        # --- Vectorized mask generation ---
        # Which (individual, event) pairs to mutate
        event_mask = np.random.random((n_ind, E)) < self.event_prob  # (n_ind, E)
        n_mutated_events = int(event_mask.sum())

        # Which gene(s) to mutate per selected event: shape (n_ind, E, 3)
        gene_coin = np.random.random((n_ind, E, 3)) < 0.5
        # Ensure at least one gene selected per event
        none_selected = ~gene_coin.any(axis=2)  # (n_ind, E)
        forced_gene = np.random.randint(0, 3, size=(n_ind, E))  # fallback gene
        # Set the forced gene where none were selected
        fi, fe = np.nonzero(none_selected)
        gene_coin[fi, fe, forced_gene[fi, fe]] = True

        # Combine: only care about genes in mutated events
        # gene_mutate[i, e, g] = True  iff  event (i,e) selected AND gene g chosen
        gene_mutate = event_mask[:, :, np.newaxis] & gene_coin  # (n_ind, E, 3)

        # --- Pre-pad ragged domain arrays for vectorized sampling ---
        # For each of the 3 gene types, build a padded domain matrix and sample
        domain_lists = [
            self.allowed_instructors,
            self.allowed_rooms,
            self.allowed_starts,
        ]

        for g, domains in enumerate(domain_lists):
            # gene_mutate[:, :, g] tells us which (ind, event) need a new value
            mutate_g = gene_mutate[:, :, g]  # (n_ind, E)
            if not mutate_g.any():
                continue

            # Build padded domain array for this gene type
            max_dom = max((len(d) for d in domains), default=0)
            if max_dom == 0:
                continue
            dom_padded = np.zeros((E, max_dom), dtype=np.int64)
            dom_lengths = np.empty(E, dtype=np.int64)
            for e_idx in range(E):
                d = domains[e_idx]
                dl = len(d)
                dom_lengths[e_idx] = dl
                if dl > 0:
                    dom_padded[e_idx, :dl] = d

            # For events with empty domains, skip them
            has_domain = dom_lengths > 0  # (E,)
            mutate_g = mutate_g & has_domain[np.newaxis, :]  # mask out empty domains

            if not mutate_g.any():
                continue

            # Get (ind, event) indices of mutations
            mi, me = np.nonzero(mutate_g)

            # Vectorized random index into each event's domain
            rand_idx = (np.random.random(len(mi)) * dom_lengths[me]).astype(np.int64)
            # Clamp to valid range (safety)
            rand_idx = np.minimum(rand_idx, dom_lengths[me] - 1)

            # Gather new values
            new_vals = dom_padded[me, rand_idx]

            # Write back into Y at the correct gene column
            Y[mi, 3 * me + g] = new_vals

        logging.getLogger(__name__).debug(
            "Mutation: %d individuals, %d/%d events mutated (%.1f%%)",
            n_ind, n_mutated_events, n_ind * E,
            100 * n_mutated_events / (n_ind * E) if (n_ind * E) else 0,
        )

        return Y


# =====================================================================
# Algorithm factory
# =====================================================================


def create_algorithm(
    pkl_path: str = "events_with_domains.pkl",
    pop_size: int = 100,
    n_offsprings: int | None = None,
    crossover_prob: float = 0.5,
    mutation_event_prob: float = 0.05,
    algorithm: str = "nsga2",
    seed: int = 42,
    use_repair: bool = True,
):
    """Create a fully-configured pymoo algorithm for scheduling.

    Args:
        pkl_path: Path to events_with_domains.pkl.
        pop_size: Population size.
        n_offsprings: Offspring per generation (default: pop_size).
        crossover_prob: Per-event crossover probability.
        mutation_event_prob: Per-event mutation probability.
        algorithm: Algorithm name ("nsga2" or "ga").
        seed: Random seed.
        use_repair: If True (default), apply PymooVectorizedRepair every
            generation.  Set to False for a pure baseline without any
            repair operator.

    Returns:
        Configured pymoo algorithm instance.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.soo.nonconvex.ga import GA

    sampling = ConstructiveSampling(pkl_path)
    crossover = EventBlockCrossover(prob=crossover_prob)
    mutation = EventLocalMutation(pkl_path=pkl_path, event_prob=mutation_event_prob)

    repair = None
    if use_repair:
        from .repair_operator_vectorized import PymooVectorizedRepair

        repair = PymooVectorizedRepair(pkl_path, passes=5)

    if n_offsprings is None:
        n_offsprings = pop_size

    if algorithm.lower() == "nsga2":
        algo = NSGA2(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            repair=repair,
            seed=seed,
        )
    elif algorithm.lower() == "ga":
        algo = GA(
            pop_size=pop_size,
            n_offsprings=n_offsprings,
            sampling=sampling,
            crossover=crossover,
            mutation=mutation,
            repair=repair,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}. Use 'nsga2' or 'ga'.")

    return algo
