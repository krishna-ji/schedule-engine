"""GA experiment runners — pymoo-based NSGA-II modes.

Each subclass pre-fills its mode-specific defaults (callback kind,
mutation parameters, repair settings) so the run script only needs to
set user-facing knobs like ``pop_size``, ``ngen``, and ``seed``.

All modes share the same pipeline:
    build_events_with_domains()  →  SchedulingProblem  →  NSGA-II  →  result
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import PROJECT_ROOT, BaseExperiment

__version__ = "3.0.0"  # pymoo runner v3


# =====================================================================
#  GA Experiment (base for all GA modes)
# =====================================================================


class GAExperiment(BaseExperiment):
    """Base GA experiment using pymoo NSGA-II.

    Parameters
    ----------
    mode : str
        Mode name (``"baseline"``, ``"memetic"``, etc.).
    pop_size : int
        Population size.
    ngen : int
        Number of generations.
    crossover_prob : float
        Per-event crossover probability.
    mutation_event_prob : float
        Per-event mutation probability.
    n_offsprings_mult : float
        Offspring multiplier (1.0 = pop_size offspring per gen).
    log_interval : int | None
        Generations between detailed logs.  ``None`` → auto (ngen / 20).
    """

    def __init__(
        self,
        *,
        mode: str,
        pop_size: int = 100,
        ngen: int = 200,
        crossover_prob: float = 0.5,
        mutation_event_prob: float = 0.05,
        n_offsprings_mult: float = 1.0,
        log_interval: int | None = None,
        # BaseExperiment kwargs
        seed: int = 42,
        data_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
        verbose: bool = True,
    ) -> None:
        tag = f"ga_{mode}"
        super().__init__(
            name=f"GA {mode.title()}",
            tag=tag,
            seed=seed,
            data_dir=data_dir,
            output_dir=output_dir,
            verbose=verbose,
        )
        self.mode = mode
        self.pop_size = pop_size
        self.ngen = ngen
        self.crossover_prob = crossover_prob
        self.mutation_event_prob = mutation_event_prob
        self.n_offsprings_mult = n_offsprings_mult
        self.log_interval = log_interval or max(1, ngen // 20)

    # ── Pipeline helpers ──────────────────────────────────────────

    def _ensure_pkl(self) -> str:
        """Build ``events_with_domains.pkl`` if missing; return its path."""
        pkl_path = str(PROJECT_ROOT / "events_with_domains.pkl")
        if not Path(pkl_path).exists():
            self.logger.info("Building events_with_domains.pkl ...")
            from src.pipeline.build_events import build_events_with_domains

            build_events_with_domains(str(self.data_dir))
        return pkl_path

    def _build_callback(self, pkl_path: str) -> Any:
        """Override in subclasses for mode-specific callbacks."""
        return _make_progress_cb(self.log_interval)

    # ── Core execution ─────────────────────────────────────────────

    def _load_data(self) -> tuple[Any, Any, Any]:
        """Load scheduling data and save feasibility report to output dir.

        Returns (store, ctx, qts).
        """
        from src.io.data_store import DataStore
        from src.io.time_system import QuantumTimeSystem

        store = DataStore.from_json(str(self.data_dir))
        ctx = store.to_context()
        qts = QuantumTimeSystem()

        # Save feasibility report to the timestamped output folder
        if store.feasibility_report is not None:
            from src.io.feasibility import generate_feasibility_report_file

            report_path = self.output_dir / "feasibility_report.txt"
            generate_feasibility_report_file(store.feasibility_report, str(report_path))
            self.logger.info(f"Feasibility report -> {report_path}")

        return store, ctx, qts

    def _execute(self) -> dict[str, Any]:
        from pymoo.optimize import minimize

        from src.pipeline.pymoo_operators import create_algorithm
        from src.pipeline.scheduling_problem import create_problem

        pkl_path = self._ensure_pkl()
        _store, ctx, qts = self._load_data()

        # Apply tutorial-practical fix consistent with pkl build
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
                    if any(f in ("lecture hall", "seminar room") for f in feats_lower):
                        course.specific_lab_features = []

        n_offsprings = int(self.pop_size * self.n_offsprings_mult)
        self.logger.info(
            f"Mode: {self.mode}  |  pop={self.pop_size}  "
            f"gens={self.ngen}  seed={self.seed}  "
            f"cx={self.crossover_prob}  mut={self.mutation_event_prob}"
        )

        prob = create_problem(pkl_path, ctx=ctx, qts=qts)
        algo = create_algorithm(
            pkl_path=pkl_path,
            pop_size=self.pop_size,
            n_offsprings=n_offsprings,
            crossover_prob=self.crossover_prob,
            mutation_event_prob=self.mutation_event_prob,
            algorithm="nsga2",
            seed=self.seed,
        )

        callback = self._build_callback(pkl_path)

        t0 = time.time()
        res = minimize(
            prob,
            algo,
            ("n_gen", self.ngen),
            seed=self.seed,
            verbose=False,
            callback=callback,
        )
        elapsed = time.time() - t0

        # Extract best
        F = res.pop.get("F")
        G = res.pop.get("G")
        cv = G.sum(axis=1).clip(0)
        best_idx = int(np.argmin(cv))

        self.logger.info(f"Done in {elapsed:.1f}s  ({elapsed / self.ngen:.2f}s/gen)")
        self.logger.info(
            f"Best: hard={F[best_idx, 0]:.0f}  "
            f"soft={F[best_idx, 1]:.0f}  cv={cv[best_idx]:.0f}"
        )

        return {
            "solver": "pymoo",
            "mode": self.mode,
            "version": __version__,
            "config": self._config_dict(),
            "best_hard": float(F[best_idx, 0]),
            "best_soft": float(F[best_idx, 1]),
            "best_cv": float(cv[best_idx]),
            "n_feasible": int((cv == 0).sum()),
            "elapsed_s": round(elapsed, 2),
            "sec_per_gen": round(elapsed / self.ngen, 3) if self.ngen else 0,
            "convergence_hard": getattr(callback, "best_hards", []),
            "convergence_soft": getattr(callback, "best_softs", []),
            "convergence_constraints": getattr(callback, "best_breakdowns", []),
        }


# =====================================================================
#  Callbacks (shared helpers, kept private)
# =====================================================================

# Short labels for compact per-constraint logging (matches HARD_CONSTRAINT_NAMES order)
_SHORT = ["grp", "inst", "room", "qual", "suit", "iAvl", "rAvl", "comp"]


def _progress_payload(algorithm):
    F = algorithm.pop.get("F")
    G = algorithm.pop.get("G")
    cv = G.sum(axis=1).clip(0)
    best_idx = int(np.argmin(cv))
    return F, G, cv, best_idx


def _constraint_breakdown(G_row: np.ndarray) -> dict[str, int]:
    """Return {short_name: violation_count} for one individual."""
    return {n: int(v) for n, v in zip(_SHORT, G_row, strict=False)}


def _log_gen(algorithm, log_interval):
    F, G, cv, best_idx = _progress_payload(algorithm)
    if algorithm.n_gen == 1 or algorithm.n_gen % log_interval == 0:
        bd = G[best_idx]
        parts = " ".join(f"{n}={int(v)}" for n, v in zip(_SHORT, bd, strict=False))
        print(
            f"  Gen {algorithm.n_gen:4d}: "
            f"hard={F[best_idx, 0]:.0f}  [{parts}]  "
            f"soft={F[best_idx, 1]:.0f}  "
            f"cv={cv.min():.0f}  "
            f"feasible={int((cv == 0).sum())}/{len(algorithm.pop)}"
        )
    return F, G, cv, best_idx


def _make_progress_cb(log_interval: int):
    from pymoo.core.callback import Callback

    class CB(Callback):
        def __init__(self):
            super().__init__()
            self.best_hards: list[float] = []
            self.best_softs: list[float] = []
            self.best_breakdowns: list[dict[str, int]] = []

        def notify(self, algorithm):
            F, G, cv, best_idx = _log_gen(algorithm, log_interval)
            self.best_hards.append(float(F[best_idx, 0]))
            self.best_softs.append(float(F[best_idx, 1]))
            self.best_breakdowns.append(_constraint_breakdown(G[best_idx]))

    return CB()


# =====================================================================
#  Concrete Modes
# =====================================================================


class BaselineExperiment(GAExperiment):
    """Pure NSGA-II — no repair, no local search.

    Default config:
        pop_size=100, ngen=200, cx=0.5, mut=0.05
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("mode", "baseline")
        kwargs.setdefault("crossover_prob", 0.5)
        kwargs.setdefault("mutation_event_prob", 0.05)
        super().__init__(**kwargs)


class MemeticExperiment(GAExperiment):
    """NSGA-II + elite bitset repair (top *elite_pct*%).

    Default config:
        pop_size=80, ngen=150, cx=0.6, mut=0.08, elite_pct=0.05
    """

    def __init__(
        self,
        *,
        elite_pct: float = 0.05,
        repair_iters: int = 5,
        **kwargs,
    ):
        kwargs.setdefault("mode", "memetic")
        kwargs.setdefault("pop_size", 80)
        kwargs.setdefault("ngen", 150)
        kwargs.setdefault("crossover_prob", 0.6)
        kwargs.setdefault("mutation_event_prob", 0.08)
        super().__init__(**kwargs)
        self.elite_pct = elite_pct
        self.repair_iters = repair_iters

    def _build_callback(self, pkl_path: str):
        from pymoo.core.callback import Callback

        from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair

        repairer = BitsetSchedulingRepair(pkl_path)
        log_interval = self.log_interval
        elite_pct = self.elite_pct
        repair_iters = self.repair_iters

        class CB(Callback):
            def __init__(self):
                super().__init__()
                self.best_hards: list[float] = []
                self.best_softs: list[float] = []
                self.best_breakdowns: list[dict[str, int]] = []

            def notify(self, algorithm):
                F, G, cv, best_idx = _log_gen(algorithm, log_interval)
                self.best_hards.append(float(F[best_idx, 0]))
                self.best_softs.append(float(F[best_idx, 1]))
                self.best_breakdowns.append(_constraint_breakdown(G[best_idx]))

                pop = algorithm.pop
                n_elite = max(1, int(len(pop) * elite_pct))
                elite_idxs = np.argsort(cv)[:n_elite]
                for idx in elite_idxs:
                    X = pop[idx].get("X").copy()
                    for _ in range(repair_iters):
                        X = repairer.repair(X)
                    pop[idx].set("X", X)

        return CB()


class AggressiveExperiment(GAExperiment):
    """Large offspring (2x pop), high mutation, full-pop repair.

    Default config:
        pop_size=200, ngen=100, cx=0.7, mut=0.15, 2x offspring
    """

    def __init__(
        self,
        *,
        repair_iters: int = 3,
        **kwargs,
    ):
        kwargs.setdefault("mode", "aggressive")
        kwargs.setdefault("pop_size", 200)
        kwargs.setdefault("ngen", 100)
        kwargs.setdefault("crossover_prob", 0.7)
        kwargs.setdefault("mutation_event_prob", 0.15)
        kwargs.setdefault("n_offsprings_mult", 2.0)
        super().__init__(**kwargs)
        self.repair_iters = repair_iters

    def _build_callback(self, pkl_path: str):
        from pymoo.core.callback import Callback

        from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair

        repairer = BitsetSchedulingRepair(pkl_path)
        log_interval = self.log_interval
        repair_iters = self.repair_iters

        class CB(Callback):
            def __init__(self):
                super().__init__()
                self.best_hards: list[float] = []
                self.best_softs: list[float] = []
                self.best_breakdowns: list[dict[str, int]] = []

            def notify(self, algorithm):
                F, G, cv, best_idx = _log_gen(algorithm, log_interval)
                self.best_hards.append(float(F[best_idx, 0]))
                self.best_softs.append(float(F[best_idx, 1]))
                self.best_breakdowns.append(_constraint_breakdown(G[best_idx]))

                pop = algorithm.pop
                for i in range(len(pop)):
                    X = pop[i].get("X").copy()
                    for _ in range(repair_iters):
                        X = repairer.repair(X)
                    pop[i].set("X", X)

        return CB()


class AdaptiveExperiment(GAExperiment):
    """Stagnation-aware: ramps mutation + elite repair when stuck.

    Starts conservative (mut=0.05), escalates to *mutation_hi* when
    best_hard stalls for *stagnation_window* generations.

    Default config:
        pop_size=100, ngen=300, cx=0.5, mut=0.05→0.20
    """

    def __init__(
        self,
        *,
        stagnation_window: int = 15,
        mutation_hi: float = 0.20,
        elite_pct: float = 0.10,
        repair_iters: int = 5,
        **kwargs,
    ):
        kwargs.setdefault("mode", "adaptive")
        kwargs.setdefault("ngen", 300)
        kwargs.setdefault("crossover_prob", 0.5)
        kwargs.setdefault("mutation_event_prob", 0.05)
        super().__init__(**kwargs)
        self.stagnation_window = stagnation_window
        self.mutation_hi = mutation_hi
        self.elite_pct = elite_pct
        self.repair_iters = repair_iters

    def _build_callback(self, pkl_path: str):
        from pymoo.core.callback import Callback

        from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair

        repairer = BitsetSchedulingRepair(pkl_path)
        log_interval = self.log_interval
        stagnation_window = self.stagnation_window
        mutation_lo = self.mutation_event_prob
        mutation_hi = self.mutation_hi
        elite_pct = self.elite_pct
        repair_iters = self.repair_iters

        class CB(Callback):
            def __init__(self):
                super().__init__()
                self.best_hards: list[float] = []
                self.best_softs: list[float] = []
                self.best_breakdowns: list[dict[str, int]] = []
                self._stagnant = 0
                self._escalated = False

            def notify(self, algorithm):
                F, G, cv, best_idx = _log_gen(algorithm, log_interval)
                cur_hard = float(F[best_idx, 0])
                self.best_hards.append(cur_hard)
                self.best_softs.append(float(F[best_idx, 1]))
                self.best_breakdowns.append(_constraint_breakdown(G[best_idx]))

                if len(self.best_hards) >= 2 and cur_hard >= self.best_hards[-2]:
                    self._stagnant += 1
                else:
                    self._stagnant = 0
                    if self._escalated:
                        self._set_mutation(algorithm, mutation_lo)
                        self._escalated = False

                if self._stagnant >= stagnation_window and not self._escalated:
                    self._set_mutation(algorithm, mutation_hi)
                    self._escalated = True
                    print(
                        f"    >> stagnation @ gen {algorithm.n_gen}"
                        f" — mutation -> {mutation_hi}, elite repair ON"
                    )

                if self._escalated:
                    pop = algorithm.pop
                    n_elite = max(1, int(len(pop) * elite_pct))
                    elite_idxs = np.argsort(cv)[:n_elite]
                    for idx in elite_idxs:
                        X = pop[idx].get("X").copy()
                        for _ in range(repair_iters):
                            X = repairer.repair(X)
                        pop[idx].set("X", X)

            @staticmethod
            def _set_mutation(algorithm, prob):
                mut = algorithm.mating.mutation
                if hasattr(mut, "event_prob"):
                    mut.event_prob = prob

        return CB()


class CPHybridExperiment(GAExperiment):
    """NSGA-II + periodic CP-SAT deep polish (requires ``ortools``).

    Every *cp_interval* generations, converts the best individual to
    ``SessionGene`` list, runs ``CPRepairPipeline``, writes back.

    Default config:
        pop_size=60, ngen=100, cx=0.5, mut=0.05, CP every 10 gens
    """

    def __init__(
        self,
        *,
        cp_interval: int = 10,
        cp_timeout: float = 30.0,
        **kwargs,
    ):
        kwargs.setdefault("mode", "cp_hybrid")
        kwargs.setdefault("pop_size", 60)
        kwargs.setdefault("ngen", 100)
        super().__init__(**kwargs)
        self.cp_interval = cp_interval
        self.cp_timeout = cp_timeout

    def _build_callback(self, pkl_path: str):
        from pymoo.core.callback import Callback

        log_interval = self.log_interval
        cp_interval = self.cp_interval
        cp_timeout = self.cp_timeout

        class CB(Callback):
            def __init__(self):
                super().__init__()
                self.best_hards: list[float] = []
                self.best_softs: list[float] = []
                self.best_breakdowns: list[dict[str, int]] = []
                self._pkl_data = None
                self._ctx = None
                self._cp_pipeline = None
                self._initialised = False

            def _lazy_init(self):
                if self._initialised:
                    return True
                try:
                    from ortools.sat.python import cp_model as _  # noqa: F401
                except ImportError:
                    print("  !! ortools not installed — CP polish disabled")
                    self._initialised = True
                    return False

                from src.ga.repair.cp.pipeline import CPRepairPipeline
                from src.io.data_store import DataStore

                with open(pkl_path, "rb") as f:
                    self._pkl_data = pickle.load(f)

                data_dir = str(PROJECT_ROOT / "data")
                store = DataStore.from_json(data_dir)
                self._ctx = store.to_context()
                self._cp_pipeline = CPRepairPipeline(
                    timeout_global=cp_timeout,
                    timeout_cluster=cp_timeout / 2,
                    num_workers=2,
                )
                self._initialised = True
                return True

            def notify(self, algorithm):
                F, G, cv, best_idx = _log_gen(algorithm, log_interval)
                self.best_hards.append(float(F[best_idx, 0]))
                self.best_softs.append(float(F[best_idx, 1]))
                self.best_breakdowns.append(_constraint_breakdown(G[best_idx]))

                if algorithm.n_gen % cp_interval != 0:
                    return
                if not self._lazy_init() or self._cp_pipeline is None:
                    return

                pop = algorithm.pop
                X = pop[best_idx].get("X").copy()
                genes = self._chromosome_to_genes(X)
                if genes is None:
                    return

                print(f"    CP-SAT polish @ gen {algorithm.n_gen} ...")
                try:
                    repaired_genes, stats = self._cp_pipeline.repair(
                        genes, self._ctx, None
                    )
                    print(
                        f"    CP done: {stats.global_phase_status}, "
                        f"residual_hard={stats.residual_hard:.0f}, "
                        f"time={stats.total_time:.1f}s"
                    )
                    X_new = self._genes_to_chromosome(repaired_genes)
                    if X_new is not None:
                        pop[best_idx].set("X", X_new)
                except Exception as exc:
                    print(f"    CP-SAT error: {exc}")

            def _chromosome_to_genes(self, X):
                try:
                    from src.domain.gene import SessionGene

                    d = self._pkl_data
                    events = d["events"]
                    idx_to_inst = d["idx_to_instructor"]
                    idx_to_room = d["idx_to_room"]
                    E = len(events)
                    genes = []
                    for e in range(E):
                        ev = events[e]
                        genes.append(
                            SessionGene(
                                course_id=ev["course_id"],
                                course_type=ev["course_type"],
                                instructor_id=idx_to_inst[int(X[3 * e])],
                                group_ids=list(ev["group_ids"]),
                                room_id=idx_to_room[int(X[3 * e + 1])],
                                start_quanta=int(X[3 * e + 2]),
                                num_quanta=ev["num_quanta"],
                            )
                        )
                    return genes
                except Exception as exc:
                    print(f"    bridge error (->genes): {exc}")
                    return None

            def _genes_to_chromosome(self, genes):
                try:
                    d = self._pkl_data
                    inst_to_idx = d["instructor_to_idx"]
                    room_to_idx = d["room_to_idx"]
                    E = len(genes)
                    X = np.zeros(3 * E, dtype=int)
                    for e, g in enumerate(genes):
                        X[3 * e] = inst_to_idx[g.instructor_id]
                        X[3 * e + 1] = room_to_idx[g.room_id]
                        X[3 * e + 2] = g.start_quanta
                    return X
                except Exception as exc:
                    print(f"    bridge error (->chromo): {exc}")
                    return None

        return CB()
