"""Pymoo operators for scheduling: Sampling, Crossover, Mutation, Repair.

These operators work on the 3×E interleaved chromosome format:
    X[3e+0] = instructor_idx, X[3e+1] = room_idx, X[3e+2] = start_quanta

All operators respect event domains (allowed_instructors, allowed_rooms,
allowed_starts) stored in the events_with_domains.pkl.
"""

from __future__ import annotations

import pickle

import numpy as np

from encoding import chromosome_views

try:
    from pymoo.core.crossover import Crossover
    from pymoo.core.mutation import Mutation
    from pymoo.core.sampling import Sampling
except ImportError:
    raise ImportError("pymoo is required: pip install pymoo>=0.6")

from repair_operator import SchedulingRepair

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
        import sys
        import time as _time

        X = np.zeros((n_samples, problem.n_var), dtype=int)
        t0 = _time.perf_counter()
        for i in range(n_samples):
            rng = np.random.default_rng(i)
            X[i] = self.repairer.construct_feasible(rng)
            elapsed = _time.perf_counter() - t0
            print(
                f"\r  Constructive sampling: {i + 1}/{n_samples} "
                f"({elapsed:.1f}s)",
                end="",
                flush=True,
            )
        print()  # newline after progress
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
        # Output shape: (n_offsprings, n_matings, n_var)
        Y = np.zeros((self.n_offsprings, n_matings, n_var), dtype=X.dtype)

        for k in range(n_matings):
            p1 = X[0, k]  # first parent
            p2 = X[1, k]  # second parent

            # Random mask: which events come from parent 1 vs parent 2
            mask = np.random.random(E) < self.prob

            # Offspring 1: events from p1 where mask, else p2
            # Offspring 2: inverse
            o1 = p1.copy()
            o2 = p2.copy()

            for e in range(E):
                if not mask[e]:
                    o1[3 * e : 3 * e + 3] = p2[3 * e : 3 * e + 3]
                    o2[3 * e : 3 * e + 3] = p1[3 * e : 3 * e + 3]

            Y[0, k] = o1
            Y[1, k] = o2

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
        Y = X.copy()
        for i in range(Y.shape[0]):
            for e in range(self.n_events):
                if np.random.random() > self.event_prob:
                    continue

                ai = self.allowed_instructors[e]
                ar = self.allowed_rooms[e]
                at = self.allowed_starts[e]

                # Randomly choose which genes to mutate (at least one)
                which = np.random.random(3) < 0.5
                if not which.any():
                    which[np.random.randint(3)] = True

                if which[0] and ai:
                    Y[i, 3 * e + 0] = np.random.choice(ai)
                if which[1] and ar:
                    Y[i, 3 * e + 1] = np.random.choice(ar)
                if which[2] and at:
                    Y[i, 3 * e + 2] = np.random.choice(at)
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

    Returns:
        Configured pymoo algorithm instance.
    """
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.soo.nonconvex.ga import GA

    from repair_operator import PymooSchedulingRepair

    sampling = ConstructiveSampling(pkl_path)
    crossover = EventBlockCrossover(prob=crossover_prob)
    mutation = EventLocalMutation(pkl_path=pkl_path, event_prob=mutation_event_prob)
    repair = PymooSchedulingRepair(pkl_path)

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
