"""GA experiment runners — pymoo-based NSGA-II modes.

Each subclass pre-fills its mode-specific defaults (callback kind,
mutation parameters, repair settings) so the run script only needs to
set user-facing knobs like ``pop_size``, ``ngen``, and ``seed``.

All modes share the same pipeline:
    build_events_with_domains()  →  SchedulingProblem  →  NSGA-II  →  result
"""

from __future__ import annotations

import csv
import logging
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

from .base import PROJECT_ROOT, BaseExperiment
from .callback_core import GACallbackBase

logger = logging.getLogger(__name__)

__version__ = "3.0.0"  # pymoo runner v3


# ── Module-level worker for ProcessPoolExecutor (must be picklable) ──


def _repair_single_elite(
    X: np.ndarray,
    pkl_path: str,
    repair_iters: int,
    gen: int,
    idx: int,
) -> np.ndarray:
    """Run the repair loop for one elite individual in a worker process.

    Instantiates its own ``BitsetSchedulingRepair`` (each process needs its
    own copy since the repairer is not fork-safe / not picklable).

    Parameters
    ----------
    X : ndarray, shape (3*E,)
        Chromosome to repair (already copied by the caller).
    pkl_path : str
        Path to events_with_domains.pkl.
    repair_iters : int
        Number of alternating stochastic/deterministic repair passes.
    gen : int
        Current generation (used to seed stochastic passes).
    idx : int
        Individual index (used to seed stochastic passes).

    Returns
    -------
    ndarray
        Repaired chromosome.
    """
    from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair

    repairer = BitsetSchedulingRepair(pkl_path)
    for p in range(repair_iters):
        if p % 2 == 0:
            rng_p = np.random.default_rng([gen, idx, p])
        else:
            rng_p = None
        X_new = repairer.repair(X, rng=rng_p)
        if np.array_equal(X_new, X):
            break
        X = X_new
    return X


def _reeval_modified(algorithm: Any, modified_inds: list) -> None:
    """Clear stale F/G/CV on modified individuals and force re-evaluation.

    Must be called whenever a callback mutates ``pop[i].X`` so that
    pymoo's NSGA-II sorting sees the updated fitness, not the stale
    pre-repair scores.
    """
    if not modified_inds:
        return
    for ind in modified_inds:
        ind.set("F", None)
        ind.set("G", None)
        ind.set("CV", None)

        # Remove from the evaluated set (crucial for pymoo to trigger re-evaluation)
        if "F" in ind.evaluated:
            ind.evaluated.remove("F")
        if "G" in ind.evaluated:
            ind.evaluated.remove("G")
        if "CV" in ind.evaluated:
            ind.evaluated.remove("CV")
    eval_pop = Population.create(*modified_inds)
    Evaluator().eval(algorithm.problem, eval_pop)


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
    export_pdf : bool
        Generate schedule PDFs (calendar, instructor, room).
        Set ``False`` for fast dev iterations (saves ~55s per run).
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
        export_pdf: bool = True,
        force_pdf: bool = False,
        use_repair: bool = True,
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
        self.export_pdf = export_pdf
        self.force_pdf = force_pdf
        self.use_repair = use_repair

    # ── Pipeline helpers ──────────────────────────────────────────

    def _ensure_pkl(self) -> str:
        """Build ``events_with_domains.pkl`` if missing; return its path."""
        from src.pipeline.build_events import PKL_DEFAULT_PATH, ensure_pkl

        pkl_path = str(PROJECT_ROOT / PKL_DEFAULT_PATH)
        ensure_pkl(pkl_path, data_dir=str(self.data_dir))
        return pkl_path

    def _build_callback(self, pkl_path: str) -> Any:
        """Override in subclasses for mode-specific callbacks."""
        return GACallbackBase(self.log_interval)

    # ── Output generation (plots + PDFs) ─────────────────────────

    @staticmethod
    def _chromosome_to_genes(X: np.ndarray, pkl_data: dict) -> list | None:
        """Convert a flat pymoo chromosome back to a list of SessionGene."""
        try:
            from src.domain.gene import SessionGene

            events = pkl_data["events"]
            idx_to_inst = pkl_data["idx_to_instructor"]
            idx_to_room = pkl_data["idx_to_room"]
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
            logger.exception("chromosome -> genes bridge error: %s", exc)
            return None

    def _generate_outputs(
        self,
        *,
        res: Any,
        callback: Any,
        pkl_data: dict,
        ctx: Any,
        qts: Any,
        best_idx: int,
    ) -> None:
        """Generate all plots, schedule PDFs, and reports.

        Called automatically at the end of ``_execute()``.  Failures in
        individual export steps are logged but never propagate — the
        experiment result is always returned.
        """
        import matplotlib as mpl

        mpl.use("Agg")  # non-interactive backend for headless envs

        out = str(self.output_dir)
        F = res.pop.get("F")
        best_hards: list[float] = getattr(callback, "best_hards", [])
        best_softs: list[float] = getattr(callback, "best_softs", [])
        best_breakdowns: list[dict] = getattr(callback, "best_breakdowns", [])

        # ── 1. Convergence plots ───────────────────────────────────
        self._safe_call(
            "hard-violation plot",
            lambda: (
                __import__(
                    "src.io.export.plothard",
                    fromlist=["plot_hard_constraint_violation_over_generation"],
                ).plot_hard_constraint_violation_over_generation(best_hards, out)
            ),
        )
        self._safe_call(
            "soft-penalty plot",
            lambda: (
                __import__(
                    "src.io.export.plotsoft",
                    fromlist=["plot_soft_constraint_violation_over_generation"],
                ).plot_soft_constraint_violation_over_generation(best_softs, out)
            ),
        )

        # ── 2. Per-constraint trend plots ──────────────────────────
        if best_breakdowns:
            # Transpose list[dict] -> dict[str, list[int]]
            all_keys = best_breakdowns[0].keys()
            hard_trends: dict[str, list[int]] = {
                k: [bd.get(k, 0) for bd in best_breakdowns] for k in all_keys
            }
            self._safe_call(
                "individual constraint plots",
                lambda: (
                    __import__(
                        "src.io.export.plot_detailed_constraints",
                        fromlist=["plot_individual_hard_constraints"],
                    ).plot_individual_hard_constraints(hard_trends, out)
                ),
            )

        # ── 2b. Per-soft-constraint trend plots ───────────────────
        best_soft_breakdowns: list[dict] = getattr(callback, "best_soft_breakdowns", [])
        if best_soft_breakdowns and best_soft_breakdowns[0]:
            all_soft_keys = best_soft_breakdowns[0].keys()
            soft_trends: dict[str, list[int]] = {
                k: [bd.get(k, 0) for bd in best_soft_breakdowns] for k in all_soft_keys
            }
            self._safe_call(
                "individual soft constraint plots",
                lambda: (
                    __import__(
                        "src.io.export.plot_detailed_constraints",
                        fromlist=["plot_individual_soft_constraints"],
                    ).plot_individual_soft_constraints(soft_trends, out)
                ),
            )

        # ── 3. Convergence rate analysis ───────────────────────────
        if len(best_hards) >= 11:
            self._safe_call(
                "convergence rate plot",
                lambda: (
                    __import__(
                        "src.io.export.plot_convergence",
                        fromlist=["plot_convergence_rate"],
                    ).plot_convergence_rate(best_hards, out, "Hard Violations")
                ),
            )

        # ── 4. Pareto front (pymoo F matrix) ──────────────────────
        self._safe_call(
            "Pareto front plot",
            lambda: (
                __import__(
                    "src.io.export.plotpareto",
                    fromlist=["plot_pareto_front_from_F"],
                ).plot_pareto_front_from_F(F, out)
            ),
        )

        # ── 4b. MOEA metric plots (HV, spacing, diversity, feasibility) ──
        hv_hist: list[float] = getattr(callback, "hypervolumes", [])
        sp_hist: list[float] = getattr(callback, "spacings", [])
        div_hist: list[float] = getattr(callback, "diversities", [])
        feas_hist: list[float] = getattr(callback, "feasibility_rates", [])
        igd_hist: list[float] = getattr(callback, "igds", [])

        # ── 4a-2. Pareto Evolution scatter (generation-coloured) ──
        f_history: list = getattr(callback, "f_history", [])
        if f_history:
            self._safe_call(
                "Pareto evolution plot",
                lambda: (
                    __import__(
                        "src.io.export.plot_pareto_evolution",
                        fromlist=["plot_pareto_evolution"],
                    ).plot_pareto_evolution(f_history, out)
                ),
            )

        if hv_hist:
            self._safe_call(
                "hypervolume trend",
                lambda: (
                    __import__(
                        "src.io.export.plot_hypervolume",
                        fromlist=["plot_hypervolume_trend"],
                    ).plot_hypervolume_trend(hv_hist, out)
                ),
            )
        if sp_hist:
            self._safe_call(
                "spacing trend",
                lambda: (
                    __import__(
                        "src.io.export.plot_spacing",
                        fromlist=["plot_spacing_trend"],
                    ).plot_spacing_trend(sp_hist, out)
                ),
            )
        if div_hist:
            self._safe_call(
                "diversity trend",
                lambda: (
                    __import__(
                        "src.io.export.plotdiversity",
                        fromlist=["plot_diversity_trend"],
                    ).plot_diversity_trend(div_hist, out)
                ),
            )
        if feas_hist:
            self._safe_call(
                "feasibility rate trend",
                lambda: (
                    __import__(
                        "src.io.export.plot_convergence",
                        fromlist=["plot_constraint_satisfaction_evolution"],
                    ).plot_constraint_satisfaction_evolution(feas_hist, out)
                ),
            )

        # ── 4c. IGD trend (only if reference front was available) ──
        # Filter out nan values to check if any real IGD values exist
        igd_real = [v for v in igd_hist if v == v]  # nan != nan
        if igd_real:
            self._safe_call(
                "IGD trend",
                lambda: (
                    __import__(
                        "src.io.export.plot_igd",
                        fromlist=["plot_igd_trend"],
                    ).plot_igd_trend(igd_hist, out)
                ),
            )

        # Convergence dashboard (needs all 6 inputs)
        if hv_hist and sp_hist and div_hist and feas_hist:
            self._safe_call(
                "convergence dashboard",
                lambda: (
                    __import__(
                        "src.io.export.plot_convergence",
                        fromlist=["plot_convergence_dashboard"],
                    ).plot_convergence_dashboard(
                        best_hards,
                        best_softs,
                        div_hist,
                        hv_hist,
                        sp_hist,
                        feas_hist,
                        out,
                    )
                ),
            )

        # ── 5. Generation-wise CSV log ─────────────────────────────
        self._safe_call(
            "convergence CSV",
            lambda: self._write_convergence_csv(
                out,
                best_hards,
                best_softs,
                best_breakdowns,
                best_soft_breakdowns,
            ),
        )

        # ── 6. Schedule PDFs (calendar, instructor, room) ─────────
        # Always attempt PDF export on the best individual found,
        # even when hard > 0.  This is essential for thesis reporting.
        if self.export_pdf:
            X_best = res.pop[best_idx].get("X")
            genes = self._chromosome_to_genes(X_best, pkl_data)
            if genes is not None:
                # Assign co-instructors for practical genes (pymoo chromosomes
                # don't encode co_instructor_ids so we add them here).
                from src.ga.core.population import _assign_practical_co_instructors

                _assign_practical_co_instructors(genes, ctx)
                self._safe_call(
                    "schedule decode + export",
                    lambda: self._export_schedule_pdfs(genes, ctx, qts, out),
                )
            else:
                self.logger.error(
                    "Skipping schedule PDF export — chromosome-to-gene "
                    "bridge returned None (see traceback above)"
                )
        else:
            self.logger.info("  [skip] PDF export disabled (export_pdf=False)")

        # ── 7. Memetic / repair-specific thesis plots ─────────────
        repair_gens: list[int] = getattr(callback, "repair_gens", [])
        if repair_gens:
            self._safe_call(
                "repair intervention plot",
                lambda: (
                    __import__(
                        "src.io.export.plot_memetic",
                        fromlist=["plot_repair_interventions"],
                    ).plot_repair_interventions(best_hards, repair_gens, out)
                ),
            )
            if f_history:
                self._safe_call(
                    "Pareto repair shift plot",
                    lambda: (
                        __import__(
                            "src.io.export.plot_memetic",
                            fromlist=["plot_pareto_repair_shift"],
                        ).plot_pareto_repair_shift(f_history, repair_gens, out)
                    ),
                )

        self.logger.info(f"Output artefacts written to {self.output_dir}")

    # ── CSV helpers ────────────────────────────────────────────

    @staticmethod
    def _write_convergence_csv(
        output_dir: str,
        best_hards: list[float],
        best_softs: list[float],
        best_breakdowns: list[dict],
        best_soft_breakdowns: list[dict],
    ) -> None:
        """Write ``convergence_history.csv`` with per-gen constraint data."""
        path = Path(output_dir) / "convergence_history.csv"

        # Hard constraint short names (Academic Nomenclature)
        hc_keys = ["CTE", "FTE", "SRE", "FPC", "FFC", "FCA", "CQF", "ICTD", "sib"]
        # Soft constraint short names (Academic Nomenclature)
        sc_keys = ["CSC", "FSC", "MIP", "SSCP"]

        header = (
            ["Gen", "Best_Hard", "Best_Soft"]
            + [f"hc_{k}" for k in hc_keys]
            + [f"sc_{k}" for k in sc_keys]
        )

        n_gens = len(best_hards)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for g in range(n_gens):
                hc_vals = [
                    best_breakdowns[g].get(k, 0) if g < len(best_breakdowns) else 0
                    for k in hc_keys
                ]
                sc_vals = [
                    (
                        best_soft_breakdowns[g].get(k, 0)
                        if g < len(best_soft_breakdowns) and best_soft_breakdowns[g]
                        else 0
                    )
                    for k in sc_keys
                ]
                writer.writerow(
                    [g + 1, int(best_hards[g]), int(best_softs[g])] + hc_vals + sc_vals
                )
        logger.info("  convergence CSV -> %s (%d rows)", path, n_gens)

    def _export_schedule_pdfs(
        self,
        genes: list,
        ctx: Any,
        qts: Any,
        output_dir: str,
    ) -> None:
        """Decode genes -> CourseSession list, then export all PDFs + reports."""
        from src.io.decoder import decode_individual
        from src.io.export.exporter import export_everything
        from src.io.export.schedule_views import (
            generate_instructor_schedules_pdf,
            generate_room_schedules_pdf,
        )
        from src.io.export.violation_reporter import generate_violation_report

        sessions = decode_individual(
            genes,
            ctx.courses,
            ctx.instructors,
            ctx.groups,
            ctx.rooms,
        )

        # Build course lookup {(course_id, course_type): Course}
        course_lookup: dict[tuple[str, str], Any] = ctx.courses

        # 5a. schedule.json + calendar.pdf (group-wise)
        self._safe_call(
            "calendar PDF",
            lambda: (
                export_everything(
                    sessions,
                    output_dir,
                    qts,
                    course_lookup=course_lookup,
                    parallel=False,
                )
            ),
        )

        # 5b. instructor_schedules.pdf
        self._safe_call(
            "instructor PDF",
            lambda: (
                generate_instructor_schedules_pdf(
                    sessions,
                    ctx.instructors,
                    course_lookup,
                    qts,
                    output_dir,
                )
            ),
        )

        # 5c. room_schedules.pdf
        self._safe_call(
            "room PDF",
            lambda: (
                generate_room_schedules_pdf(
                    sessions,
                    ctx.rooms,
                    course_lookup,
                    qts,
                    output_dir,
                    groups=ctx.groups,
                )
            ),
        )

        # 5d. log_violations.log
        self._safe_call(
            "violation report",
            lambda: (
                generate_violation_report(sessions, course_lookup, qts, output_dir)
            ),
        )

    def _safe_call(self, label: str, fn: Any) -> None:
        """Execute *fn* and swallow any exception, logging it with full traceback."""
        try:
            fn()
            self.logger.info(f"  [ok] {label}")
        except Exception as exc:
            self.logger.error("  [FAIL] %s: %s", label, exc, exc_info=True)

    # ── Core execution ─────────────────────────────────────────────

    def _load_data(self) -> tuple[Any, Any, Any]:
        """Load scheduling data and save feasibility report to output dir.

        If the data has known infeasibilities (e.g. instructor qualification
        bottleneck), the GA still runs — it's designed to *optimise toward*
        feasibility.  The feasibility report is saved, but the exception
        is not propagated.

        Returns (store, ctx, qts).
        """
        from src.io.data_store import DataStore
        from src.io.feasibility import (
            InfeasibleProblemError,
            generate_feasibility_report_file,
        )
        from src.io.time_system import QuantumTimeSystem

        feasibility_report = None
        try:
            store = DataStore.from_json(str(self.data_dir))
            feasibility_report = store.feasibility_report
        except InfeasibleProblemError as exc:
            # Capture the report from the exception, then retry without preflight
            feasibility_report = exc.report
            self.logger.warning(
                "Feasibility check failed — running GA anyway "
                "(the optimizer will try to minimise violations)"
            )
            store = DataStore.from_json(str(self.data_dir), run_preflight=False)
        except Exception:
            self.logger.warning("Data loading failed — retrying without preflight")
            store = DataStore.from_json(str(self.data_dir), run_preflight=False)

        ctx = store.to_context()
        qts = QuantumTimeSystem()

        # Always save feasibility report to the timestamped output folder
        if feasibility_report is not None:
            report_path = self.output_dir / "feasibility_report.md"
            generate_feasibility_report_file(feasibility_report, str(report_path))
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

        # ── Pre-feasibility report (runs before GA, survives crashes) ──
        self._safe_call(
            "pre-feasibility report",
            lambda: (
                __import__(
                    "src.io.export.feasibility_reporter",
                    fromlist=["generate_pre_feasibility_report"],
                ).generate_pre_feasibility_report(pkl_data, self.output_dir)
            ),
        )

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
            use_repair=self.use_repair,
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
        # Exclude tolerated columns (iAvl col 5) from cv
        from src.pipeline.scheduling_problem import _TOLERATED_HARD_COLS

        _strict = [i for i in range(G.shape[1]) if i not in _TOLERATED_HARD_COLS]
        cv = G[:, _strict].sum(axis=1).clip(0)
        best_idx = int(np.argmin(cv))

        self.logger.info(f"Done in {elapsed:.1f}s  ({elapsed / self.ngen:.2f}s/gen)")
        self.logger.info(
            f"Best: hard={F[best_idx, 0]:.0f}  "
            f"soft={F[best_idx, 1]:.0f}  cv={cv[best_idx]:.0f}"
        )

        # ── Generate all output artefacts (plots + PDFs) ─────────
        self._generate_outputs(
            res=res,
            callback=callback,
            pkl_data=pkl_data,
            ctx=ctx,
            qts=qts,
            best_idx=best_idx,
        )

        # Per-gen timing summary
        gen_times: list[float] = getattr(callback, "gen_times", [])
        timing_summary: dict[str, Any] = {}
        if gen_times:
            gt = np.array(gen_times)
            timing_summary = {
                "mean_s": round(float(gt.mean()), 4),
                "std_s": round(float(gt.std()), 4),
                "min_s": round(float(gt.min()), 4),
                "max_s": round(float(gt.max()), 4),
                "p50_s": round(float(np.median(gt)), 4),
                "p95_s": round(float(np.percentile(gt, 95)), 4),
            }

        return {
            "solver": "pymoo",
            "mode": self.mode,
            "version": __version__,
            "experiment_class": type(self).__name__,
            "framework": "experiments_v3",
            "config": self._config_dict(),
            "best_hard": float(F[best_idx, 0]),
            "best_soft": float(F[best_idx, 1]),
            "best_cv": float(cv[best_idx]),
            "n_feasible": int((cv == 0).sum()),
            "elapsed_s": round(elapsed, 2),
            "sec_per_gen": round(elapsed / self.ngen, 3) if self.ngen else 0,
            "timing_per_gen": timing_summary,
            "convergence_hard": getattr(callback, "best_hards", []),
            "convergence_soft": getattr(callback, "best_softs", []),
            "convergence_constraints": getattr(callback, "best_breakdowns", []),
            "hypervolumes": getattr(callback, "hypervolumes", []),
            "spacings": getattr(callback, "spacings", []),
            "diversities": getattr(callback, "diversities", []),
            "feasibility_rates": getattr(callback, "feasibility_rates", []),
            "igds": getattr(callback, "igds", []),
            "final_F": F.tolist(),
            "final_G": G.tolist(),
        }


# =====================================================================
#  Concrete Modes
# =====================================================================


class BaselineExperiment(GAExperiment):
    """Pure NSGA-II control group — NO repair operator.

    This is the thesis control: plain NSGA-II with crossover + mutation
    only.  No vectorized repair, no elite repair, no stagnation
    adaptation, no CP.  Provides the unassisted search baseline against
    which Memetic / Aggressive / Adaptive modes are compared.

    Default config:
        pop_size=100, ngen=200, cx=0.5, mut=0.05, use_repair=False
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("mode", "baseline")
        kwargs.setdefault("crossover_prob", 0.5)
        kwargs.setdefault("mutation_event_prob", 0.05)
        kwargs.setdefault("use_repair", False)
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
        repair_frequency: int = 5,
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
        self.repair_frequency = repair_frequency

    def _build_callback(self, pkl_path: str):
        log_interval = self.log_interval
        elite_pct = self.elite_pct
        repair_iters = self.repair_iters
        repair_frequency = self.repair_frequency
        _pkl_path = pkl_path  # captured for worker pickling
        n_workers = max(1, (os.cpu_count() or 1) - 1)

        class CB(GACallbackBase):
            """Memetic callback — applies bitset repair to elite survivors."""

            def _on_generation(self, algorithm, F, G, cv, best_idx):
                gen = algorithm.n_gen or 0
                # Throttle: only run expensive repair every Nth generation
                if gen % repair_frequency != 0:
                    return
                self.repair_gens.append(gen)  # track for thesis plots
                pop = algorithm.pop
                n_elite = max(1, int(len(pop) * elite_pct))
                elite_idxs = np.argsort(cv)[:n_elite]

                # Dispatch independent repair chains to process pool
                X_copies = [pop[idx].get("X").copy() for idx in elite_idxs]
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = [
                        pool.submit(
                            _repair_single_elite,
                            X_copies[j],
                            _pkl_path,
                            repair_iters,
                            gen,
                            int(elite_idxs[j]),
                        )
                        for j in range(len(elite_idxs))
                    ]
                    results = [f.result() for f in futures]

                modified = []
                for j, idx in enumerate(elite_idxs):
                    pop[idx].set("X", results[j])
                    modified.append(pop[idx])
                _reeval_modified(algorithm, modified)

        return CB(log_interval)


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
        log_interval = self.log_interval
        repair_iters = self.repair_iters
        _pkl_path = pkl_path
        n_workers = max(1, (os.cpu_count() or 1) - 1)

        class CB(GACallbackBase):
            """Aggressive callback — repairs entire population every generation."""

            def _on_generation(self, algorithm, F, G, cv, best_idx):
                pop = algorithm.pop
                gen = algorithm.n_gen or 0
                pop_size = len(pop)

                X_copies = [pop[i].get("X").copy() for i in range(pop_size)]
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futures = [
                        pool.submit(
                            _repair_single_elite,
                            X_copies[i],
                            _pkl_path,
                            repair_iters,
                            gen,
                            i,
                        )
                        for i in range(pop_size)
                    ]
                    results = [f.result() for f in futures]

                modified = []
                for i in range(pop_size):
                    pop[i].set("X", results[i])
                    modified.append(pop[i])
                _reeval_modified(algorithm, modified)

        return CB(log_interval)


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
        log_interval = self.log_interval
        stagnation_window = self.stagnation_window
        mutation_lo = self.mutation_event_prob
        mutation_hi = self.mutation_hi
        elite_pct = self.elite_pct
        repair_iters = self.repair_iters
        _pkl_path = pkl_path
        n_workers = max(1, (os.cpu_count() or 1) - 1)

        class CB(GACallbackBase):
            """Adaptive callback — escalates mutation on stagnation."""

            def __init__(self, _log_interval):
                super().__init__(_log_interval)
                self._stagnant = 0
                self._escalated = False

            def _on_generation(self, algorithm, F, G, cv, best_idx):
                cur_hard = self.best_hards[-1]

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
                    logging.getLogger(__name__).info(
                        "stagnation @ gen %d — mutation -> %s, elite repair ON",
                        algorithm.n_gen,
                        mutation_hi,
                    )

                if self._escalated:
                    self.repair_gens.append(algorithm.n_gen or 0)
                    pop = algorithm.pop
                    gen = algorithm.n_gen or 0
                    n_elite = max(1, int(len(pop) * elite_pct))
                    elite_idxs = np.argsort(cv)[:n_elite]

                    X_copies = [pop[idx].get("X").copy() for idx in elite_idxs]
                    with ProcessPoolExecutor(max_workers=n_workers) as pool:
                        futures = [
                            pool.submit(
                                _repair_single_elite,
                                X_copies[j],
                                _pkl_path,
                                repair_iters,
                                gen,
                                int(elite_idxs[j]),
                            )
                            for j in range(len(elite_idxs))
                        ]
                        results = [f.result() for f in futures]

                    modified = []
                    for j, idx in enumerate(elite_idxs):
                        pop[idx].set("X", results[j])
                        modified.append(pop[idx])
                    _reeval_modified(algorithm, modified)

            @staticmethod
            def _set_mutation(algorithm, prob):
                mut = algorithm.mating.mutation
                if hasattr(mut, "event_prob"):
                    mut.event_prob = prob

        return CB(log_interval)


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
        log_interval = self.log_interval
        cp_interval = self.cp_interval
        cp_timeout = self.cp_timeout

        class CB(GACallbackBase):
            """CP-Hybrid callback — invokes CP-SAT solver at fixed intervals."""

            def __init__(self, _log_interval):
                super().__init__(_log_interval)
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
                    logging.getLogger(__name__).warning(
                        "ortools not installed — CP polish disabled"
                    )
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

            def _on_generation(self, algorithm, F, G, cv, best_idx):
                if algorithm.n_gen % cp_interval != 0:
                    return
                if not self._lazy_init() or self._cp_pipeline is None:
                    return

                pop = algorithm.pop
                X = pop[best_idx].get("X").copy()
                genes = self._chromosome_to_genes(X)
                if genes is None:
                    return

                _cb_log = logging.getLogger(__name__)
                _cb_log.info("CP-SAT polish @ gen %d ...", algorithm.n_gen)
                try:
                    repaired_genes, stats = self._cp_pipeline.repair(
                        genes, self._ctx, None
                    )
                    _cb_log.info(
                        "CP done: %s, residual_hard=%.0f, time=%.1fs",
                        stats.global_phase_status,
                        stats.residual_hard,
                        stats.total_time,
                    )
                    X_new = self._genes_to_chromosome(repaired_genes)
                    if X_new is not None:
                        pop[best_idx].set("X", X_new)
                        _reeval_modified(algorithm, [pop[best_idx]])
                except Exception as exc:
                    _cb_log.error("CP-SAT error: %s", exc)

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
                    logging.getLogger(__name__).error("bridge error (->genes): %s", exc)
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
                    logging.getLogger(__name__).error(
                        "bridge error (->chromo): %s", exc
                    )
                    return None

        return CB(log_interval)
