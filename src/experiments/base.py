"""
BaseExperiment - Abstract base class for all scheduling experiments.

Consolidates common functionality from all run files:
- Logging setup
- Data loading
- DEAP toolbox configuration
- Evolution loop boilerplate
- Results finalization

Subclasses implement:
- _run_evolution(): Mode-specific evolution logic
- _get_experiment_name(): Name for logging/output
- _get_extra_config(): Mode-specific config for metadata
- _get_extra_results(): Mode-specific results for metadata
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from deap import base, creator, tools

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.experiments.output.base import BaseExporter

from src.experiments.checks import run_feasibility_checks
from src.ga import PopulationFactory
from src.ga.run_helpers import (
    EvolutionStats,
    NotebookData,
    course_aware_crossover,
    create_evaluator,
    get_best_individual,
    get_constraint_breakdown,
    load_data,
    print_constraint_details,
    setup_deap,
    smart_mutation,
    track_nsga_metrics,
)


class BaseExperiment(ABC):
    """
    Abstract base class for scheduling experiments.

    Provides:
    - Logging setup (file + console)
    - Data loading with QTS configuration
    - DEAP toolbox setup
    - Common evolution loop utilities
    - Results finalization and export

    Constructor Parameters:
    ----------------------
    seed : int
        Random seed for reproducibility
    pop_size : int
        Population size for genetic algorithm
    ngen : int
        Number of generations to evolve
    cxpb : float
        Crossover probability (0.0 to 1.0)
    mutpb : float
        Mutation probability (0.0 to 1.0)
    fitness_weights : tuple[float, float]
        Weights for (hard, soft) objectives. Negative = minimize.
    data_dir : Path | str
        Directory containing input JSON files
    output_dir : Path | str | None
        Output directory (auto-generated if None)
    opening_time : str
        Day start time in "HH:MM" format
    closing_time : str
        Day end time in "HH:MM" format
    closed_days : list[str]
        List of closed days (e.g., ["Saturday"])
    expected_quanta : int | None
        Expected quanta per week for feasibility check.
        If None (default), auto-calculated from opening_time, closing_time, closed_days.
    init_strategy : str
        Population initialization strategy:
        - "smart": Conflict-aware random (default, best quality)
        - "hybrid": Mix of greedy + smart + random (good diversity)
        - "random": Pure random (fastest, lowest quality)
    log_interval : int
        Generations between detailed log output
    verbose : bool
        Enable detailed console output
    """

    def __init__(
        self,
        *,
        # Core GA parameters
        seed: int = 42,
        pop_size: int = 50,
        ngen: int = 100,
        cxpb: float = 0.9,
        mutpb: float = 0.2,
        fitness_weights: tuple[float, float] = (-1.0, -1.0),
        # Data paths
        data_dir: Path | str = "data",
        output_dir: Path | str | None = None,
        # Time configuration
        opening_time: str = "10:00",
        closing_time: str = "17:00",
        closed_days: list[str] | None = None,
        expected_quanta: int | None = None,
        # Population initialization
        init_strategy: str = "smart",
        # Logging
        log_interval: int = 10,
        verbose: bool = True,
    ) -> None:
        """Initialize experiment with configuration."""
        # Store parameters
        self.seed = seed
        self.pop_size = pop_size
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.fitness_weights = fitness_weights
        self.data_dir = Path(data_dir)
        self.opening_time = opening_time
        self.closing_time = closing_time
        self.closed_days = closed_days or ["Saturday"]
        self.expected_quanta = expected_quanta
        self.init_strategy = init_strategy
        self.log_interval = log_interval
        self.verbose = verbose

        # Generate timestamp and output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir is None:
            output_dir = Path("output") / self._get_experiment_name() / self.timestamp
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize state (set during run)
        self._logger: logging.Logger | None = None
        self._data: NotebookData | None = None
        self._evaluate: Callable[[list], tuple[float, float]] | None = None
        self._toolbox: base.Toolbox | None = None
        self._population_factory: PopulationFactory | None = None
        self._pool: multiprocessing.Pool | None = None
        self._final_pop: list[Any] | None = None
        self._stats: EvolutionStats | None = None
        self._best_individual: list | None = None
        self._constraint_breakdown: dict[str, float] | None = None

    # -------------------- Abstract Methods --------------------

    @abstractmethod
    def _get_experiment_name(self) -> str:
        """Return experiment name for output directory and logging."""
        ...

    @abstractmethod
    def _run_evolution(self) -> tuple[list[Any], EvolutionStats]:
        """
        Run mode-specific evolution loop.

        Returns:
            Tuple of (final_population, evolution_stats)
        """
        ...

    def _get_extra_config(self) -> dict[str, Any]:
        """Return mode-specific configuration for metadata. Override in subclass."""
        return {}

    def _get_extra_results(self) -> dict[str, Any]:
        """Return mode-specific results for metadata. Override in subclass."""
        return {}

    # -------------------- Logging --------------------

    @property
    def logger(self) -> logging.Logger:
        """Get logger, creating if needed."""
        if self._logger is None:
            self._logger = self._setup_logging()
        return self._logger

    def _setup_logging(self) -> logging.Logger:
        """Setup logging to file and console.

        File gets full timestamps + DEBUG level.
        Console gets clean, compact output at INFO level.
        """
        log_file = self.output_dir / f"{self._get_experiment_name()}.log"

        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Clean console formatter: just level icon + message (no timestamps)
        console_formatter = logging.Formatter("%(message)s")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        # Use a UTF-8 stream to avoid UnicodeEncodeError on Windows (cp1252)
        utf8_stdout = open(  # noqa: SIM115
            sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False
        )
        console_handler = logging.StreamHandler(utf8_stdout)
        console_handler.setLevel(logging.INFO if self.verbose else logging.WARNING)
        console_handler.setFormatter(console_formatter)

        logger = logging.getLogger(f"{self._get_experiment_name()}_{self.timestamp}")
        logger.handlers.clear()
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    # -------------------- Data & Setup --------------------

    @property
    def data(self) -> NotebookData:
        """Get loaded data."""
        if self._data is None:
            raise RuntimeError("Data not loaded. Call run() first.")
        return self._data

    @property
    def evaluate(self) -> Callable[[list], tuple[float, float]]:
        """Get evaluator function."""
        if self._evaluate is None:
            raise RuntimeError("Evaluator not created. Call run() first.")
        return self._evaluate

    @property
    def toolbox(self) -> base.Toolbox:
        """Get DEAP toolbox."""
        if self._toolbox is None:
            raise RuntimeError("Toolbox not created. Call run() first.")
        return self._toolbox

    def _load_data(self) -> None:
        """Load scheduling data from JSON files."""
        self.logger.info("Loading data...")
        self._data = load_data(
            data_dir=self.data_dir,
            opening_time=self.opening_time,
            closing_time=self.closing_time,
            closed_days=self.closed_days,
        )
        self.logger.info(f"Data loaded: {self._data.summary()}")

        # Auto-calculate expected_quanta if not provided
        actual_quanta = len(self._data.qts.get_all_operating_quanta())
        if self.expected_quanta is None:
            self.expected_quanta = actual_quanta
            self.logger.debug(
                "Auto-calculated expected_quanta=%d from time config", actual_quanta
            )

        # Run feasibility checks
        run_feasibility_checks(
            self._data,
            self.output_dir,
            self.logger,
            expected_quanta=self.expected_quanta,
        )

    def _create_evaluator(self) -> None:
        """Create fitness evaluator."""
        self._evaluate = create_evaluator(self.data)

    def _setup_population_factory(self) -> None:
        """Setup PopulationFactory using SchedulingContext."""
        self._population_factory = PopulationFactory(
            context=self.data.to_context(),
            parallel=self.pop_size >= 10,  # Use parallel for non-trivial pops
        )

    # -------------------- Multiprocessing --------------------

    def _setup_pool(self) -> None:
        """Create multiprocessing pool for parallel fitness evaluation.

        Workers load data from disk once via ``init_worker`` so the heavy
        ``SchedulingContext`` is never pickled per-call.
        """
        from src.utils.parallel_worker import init_worker
        from src.utils.system_info import get_cpu_count

        n_workers = min(get_cpu_count(), max(2, self.pop_size))
        self._pool = multiprocessing.Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(str(self.data_dir), self.seed),
        )
        self.logger.debug("Multiprocessing pool created: %d workers", n_workers)

    def _shutdown_pool(self) -> None:
        """Cleanly shut down the multiprocessing pool."""
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    @property
    def population_factory(self) -> PopulationFactory:
        """Get PopulationFactory instance."""
        if self._population_factory is None:
            raise RuntimeError("PopulationFactory not created. Call run() first.")
        return self._population_factory

    def _setup_toolbox(self) -> None:
        """Setup DEAP toolbox with operators."""
        setup_deap(self.fitness_weights)

        self._toolbox = base.Toolbox()
        self._toolbox.register(
            "individual",
            lambda: creator.Individual(self._create_individual_by_strategy()),
        )
        self._toolbox.register(
            "population",
            self._create_population_by_strategy,
        )
        self._toolbox.register("evaluate", self.evaluate)
        self._toolbox.register("mate", course_aware_crossover)
        self._toolbox.register("mutate", lambda ind: smart_mutation(ind, self.data))
        self._toolbox.register("select", tools.selNSGA2)

    def _init_seeds(self) -> None:
        """Initialize random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _create_individual_by_strategy(self) -> list:
        """Create single individual using configured strategy."""
        if self.init_strategy == "hybrid":
            return self.population_factory.greedy_individual()
        if self.init_strategy == "random":
            return self.population_factory.random_individual(conflict_aware=False)
        # "smart" (default)
        return self.population_factory.random_individual(conflict_aware=True)

    def _create_population_by_strategy(self, n: int) -> list[Any]:
        """Create population using configured strategy."""
        if self.init_strategy == "hybrid":
            # Hybrid: Mix of greedy + smart + random
            raw_pop = self.population_factory.create_population(n, strategy="hybrid")
        elif self.init_strategy == "random":
            # Pure random
            raw_pop = self.population_factory.create_population(n, strategy="random")
        else:  # "smart" (default)
            # Conflict-aware random
            raw_pop = self.population_factory.create_population(n, strategy="smart")

        # Wrap in DEAP Individual
        return [creator.Individual(ind) for ind in raw_pop]

    # -------------------- Evolution Helpers --------------------

    def create_initial_population(self) -> list[Any]:
        """Create and evaluate initial population."""
        pop: list[Any] = self.toolbox.population(n=self.pop_size)
        self.evaluate_offspring(pop)  # Uses parallel pool when available
        return pop

    def apply_crossover(
        self,
        offspring: list[Any],
    ) -> None:
        """Apply crossover to offspring in-place."""
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.cxpb:
                self.toolbox.mate(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

    def apply_mutation(
        self,
        offspring: list[Any],
    ) -> None:
        """Apply mutation to offspring in-place."""
        for ind in offspring:
            if random.random() < self.mutpb:
                self.toolbox.mutate(ind)
                del ind.fitness.values

    def apply_group_repair(
        self,
        offspring: list[Any],
    ) -> None:
        """Repair group-time clashes in-place for all offspring.

        This is a lightweight structural repair that guarantees zero (or
        near-zero) group overlaps after crossover/mutation.  Runs in O(G·Q)
        per individual and is safe to call in every mode.
        """
        from src.ga.repair.group_clash_repair import repair_group_clashes

        ctx = self.data.to_context()
        for ind in offspring:
            repair_group_clashes(ind, ctx)
            # Any repaired gene invalidates fitness
            if ind.fitness.valid:
                del ind.fitness.values

    def evaluate_offspring(
        self,
        offspring: list[Any],
    ) -> None:
        """Evaluate offspring with invalid fitness.

        Uses multiprocessing pool when available for parallel evaluation.
        """
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        if not invalid:
            return

        if self._pool is not None and len(invalid) >= 4:
            # Parallel evaluation via worker pool
            from src.ga.scheduler import _worker_evaluate

            results = self._pool.map(_worker_evaluate, invalid)
            for ind, fit in zip(invalid, results):
                ind.fitness.values = fit
        else:
            # Sequential fallback (small batches or no pool)
            for ind in invalid:
                ind.fitness.values = self.evaluate(ind)

    def record_generation_stats(
        self,
        pop: list[Any],
        stats: EvolutionStats,
        gen: int,
        gen_start: float,
    ) -> None:
        """Record statistics for a generation."""
        hard_vals = [ind.fitness.values[0] for ind in pop]
        soft_vals = [ind.fitness.values[1] for ind in pop]

        stats.generations.append(gen)
        stats.min_hard.append(float(min(hard_vals)))
        stats.avg_hard.append(float(np.mean(hard_vals)))
        stats.max_hard.append(float(max(hard_vals)))
        stats.min_soft.append(float(min(soft_vals)))
        stats.avg_soft.append(float(np.mean(soft_vals)))
        stats.feasible_count.append(sum(1 for h in hard_vals if h == 0))
        stats.generation_times.append(time.time() - gen_start)

        track_nsga_metrics(pop, stats, self.data)

    def log_generation_progress(
        self,
        pop: list[Any],
        gen: int,
    ) -> None:
        """Log detailed progress for a generation."""
        if gen % self.log_interval != 0 and gen != self.ngen - 1:
            return

        best_ind = min(
            pop, key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1])
        )
        breakdown = get_constraint_breakdown(list(best_ind), self.data)

        # Split into hard and soft
        from src.constraints import HARD_CONSTRAINT_NAMES, SOFT_CONSTRAINT_NAMES

        hard_bd = {k: v for k, v in breakdown.items() if k in HARD_CONSTRAINT_NAMES}
        soft_bd = {k: v for k, v in breakdown.items() if k in SOFT_CONSTRAINT_NAMES}

        print_constraint_details(
            hard_bd, soft_bd, gen, logger=self.logger, ngen=self.ngen
        )

    # -------------------- Results --------------------

    def _finalize_results(self) -> None:
        """Finalize results after evolution."""
        if self._final_pop is None or self._stats is None:
            raise RuntimeError("Evolution not completed")

        self._best_individual = get_best_individual(self._final_pop)
        self._constraint_breakdown = get_constraint_breakdown(
            self._best_individual, self.data
        )

    def _build_metadata(self) -> dict[str, Any]:
        """Build experiment metadata for export."""
        if self._stats is None or self._constraint_breakdown is None:
            raise RuntimeError("Results not finalized")

        return {
            "experiment": self._get_experiment_name(),
            "timestamp": self.timestamp,
            "config": {
                "seed": self.seed,
                "pop_size": self.pop_size,
                "ngen": self.ngen,
                "cxpb": self.cxpb,
                "mutpb": self.mutpb,
                "fitness_weights": list(self.fitness_weights),
                "init_strategy": self.init_strategy,
                **self._get_extra_config(),
            },
            "results": {
                "elapsed_time": self._stats.elapsed_time,
                "final_min_hard": (
                    self._stats.min_hard[-1] if self._stats.min_hard else None
                ),
                "final_min_soft": (
                    self._stats.min_soft[-1] if self._stats.min_soft else None
                ),
                "final_feasible_count": (
                    self._stats.feasible_count[-1] if self._stats.feasible_count else 0
                ),
                **self._get_extra_results(),
            },
            "nsga2_metrics": {
                "spacing": self._stats.spacing[-1] if self._stats.spacing else 0.0,
                "hypervolume": (
                    self._stats.hypervolume[-1] if self._stats.hypervolume else 0.0
                ),
                "population_diversity": (
                    self._stats.diversity[-1] if self._stats.diversity else 0.0
                ),
                "pareto_front_size": (
                    self._stats.pareto_front_size[-1]
                    if self._stats.pareto_front_size
                    else 0
                ),
            },
            "constraint_breakdown": self._constraint_breakdown,
            "generation_times": self._stats.generation_times,
        }

    # -------------------- Main Entry Point --------------------

    def run(self) -> dict[str, Any]:
        """
        Run the complete experiment.

        Returns:
            Experiment metadata dictionary
        """
        from src.utils.console_service import get_console
        from src.utils.system_info import get_cpu_count

        console = get_console()

        # Initialize
        self._init_seeds()

        # Rich header
        exp_title = self._get_experiment_name().upper().replace("_", " ")
        console.print()
        console.rule(f"[bold cyan]{exp_title}[/bold cyan]")
        console.print(
            f"  [dim]pop={self.pop_size}  ngen={self.ngen}  "
            f"cxpb={self.cxpb}  mutpb={self.mutpb}  "
            f"seed={self.seed}  init={self.init_strategy}  cpus={get_cpu_count()}[/dim]"
        )
        console.print(f"  [dim]output: {self.output_dir}[/dim]")
        console.print()

        # Log to file (full detail)
        self.logger.debug("=" * 60)
        self.logger.debug(exp_title)
        self.logger.debug("=" * 60)
        self.logger.debug(
            "Config: pop=%d, ngen=%d, cxpb=%.2f, mutpb=%.2f, seed=%d, init=%s",
            self.pop_size,
            self.ngen,
            self.cxpb,
            self.mutpb,
            self.seed,
            self.init_strategy,
        )
        self.logger.debug("Output: %s", self.output_dir)

        # Setup
        self._load_data()
        self._create_evaluator()
        self._setup_population_factory()
        self._setup_pool()
        self._setup_toolbox()

        # Run mode-specific evolution
        console.print("[bold]Starting evolution...[/bold]")
        console.print()
        self.logger.debug("Starting %s evolution...", self._get_experiment_name())
        self._init_seeds()  # Reset seeds before evolution
        try:
            self._final_pop, self._stats = self._run_evolution()
        finally:
            self._shutdown_pool()

        # Completion summary
        elapsed = self._stats.elapsed_time
        mins, secs = divmod(elapsed, 60)
        time_str = f"{int(mins)}m {secs:.0f}s" if mins > 0 else f"{secs:.1f}s"
        console.print()
        console.print(f"  [green bold]Evolution completed in {time_str}[/green bold]")
        self.logger.debug("Evolution completed in %.1fs", elapsed)

        # Finalize
        self._finalize_results()

        # Export
        self._export_results()

        # Completion
        console.print()
        console.rule(f"[bold green]{exp_title} COMPLETE[/bold green]")
        console.print(f"  [dim]All files saved to: {self.output_dir}[/dim]")
        console.print()
        self.logger.debug("All files saved to: %s", self.output_dir)

        return self._build_metadata()

    def _export_results(self) -> None:
        """Export results using exporter. Override for custom export."""

        exporter = self._create_exporter()
        exporter.export_all(
            final_pop=self._final_pop,
            stats=self._stats,
            best_individual=self._best_individual,
            metadata=self._build_metadata(),
        )

    def _create_exporter(self) -> BaseExporter:
        """Create exporter instance. Override for custom exporter."""
        from src.experiments.output.base import BaseExporter

        return BaseExporter(
            output_dir=self.output_dir,
            data=self.data,
            logger=self.logger,
        )

    # -------------------- Convenience Methods --------------------

    @property
    def best_individual(self) -> list:
        """Get best individual after evolution."""
        if self._best_individual is None:
            raise RuntimeError("Evolution not completed")
        return self._best_individual

    @property
    def stats(self) -> EvolutionStats:
        """Get evolution statistics."""
        if self._stats is None:
            raise RuntimeError("Evolution not completed")
        return self._stats

    @property
    def final_population(self) -> list[Any]:
        """Get final population."""
        if self._final_pop is None:
            raise RuntimeError("Evolution not completed")
        return self._final_pop
