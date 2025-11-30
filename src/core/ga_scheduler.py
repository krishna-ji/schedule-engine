"""
GA Scheduler Module

Encapsulates NSGA-II genetic algorithm execution for course scheduling.
Extracted from monolithic main.py for better testability and separation of concerns.
"""

from __future__ import annotations

import random
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from deap import base, tools
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

from src.config import get_config
from src.constraints.registry import (
    get_enabled_hard_constraints,
    get_enabled_soft_constraints,
)
from src.core.types import SchedulingContext
from src.ga.evaluator.detailed_fitness import evaluate_detailed
from src.ga.evaluator.fitness import evaluate
from src.ga.operators.crossover import crossover_course_group_aware
from src.ga.operators.mutation import mutate_individual
from src.ga.population import generate_course_group_aware_population
from src.heuristics.parallel_executor import (
    ParallelHeuristicExecutor,
    get_parallel_executor,
)
from src.heuristics.registry import get_heuristic_statistics_template
from src.metrics.diversity import average_pairwise_diversity
from src.utils.console_service import get_console
from src.utils.parallel_worker import get_worker_context
from src.utils.performance_profiler import get_profiler
from src.utils.structured_logger import StructuredLogger

console = get_console()
logger = StructuredLogger.get_logger(__name__)


def _worker_evaluate(individual: Any) -> tuple[float, float]:
    """
    Evaluate individual using worker-local context.

    This function is called for each evaluation. It retrieves the
    scheduling context from module-level state (set once in init_worker)
    instead of pickling it every time.

    Args:
        individual: GA individual to evaluate

    Returns:
        Tuple of (hard_violations, soft_penalty)
    """
    context = get_worker_context()
    return evaluate(
        individual,
        context["courses"],
        context["instructors"],
        context["groups"],
        context["rooms"],
    )


# ================
# Genetic Operators (Sequential to avoid GIL thrashing)
# ================


def _parallel_crossover(
    offspring: list[Any], cxpb: float, toolbox: Any, max_workers: int | None = None
) -> list[Any]:
    """
    Apply crossover sequentially.

    NOTE: ThreadPoolExecutor removed because Python's GIL prevents true parallelism
    for CPU-bound tasks like crossover. Multiprocessing overhead (pickling)
    often outweighs benefits for simple operators. Sequential is faster and safer.

    CRITICAL FIX: DEAP operators return tuples (ind1, ind2). While most DEAP operators
    modify in-place, we MUST reassign the tuple results to handle edge cases where
    operators return new objects. This prevents GPU evaluation failures caused by
    tuple objects replacing individual contents.
    """
    # Iterate in steps of 2: (0,1), (2,3), etc.
    for i in range(0, len(offspring) - 1, 2):
        if random.random() < cxpb:
            # Call crossover operator (returns tuple of modified individuals)
            result = toolbox.mate(offspring[i], offspring[i + 1])

            # CRITICAL: Must unpack and reassign even if modified in-place
            # Some DEAP operators or custom implementations may return new
            # objects
            offspring[i], offspring[i + 1] = result
            # Failure to reassign causes GPU evaluator to receive
            # tuple-corrupted individuals
            offspring[i], offspring[i + 1] = result

            # CRITICAL FIX: Force fitness invalidation (DEAP bug workaround)
            # Use invalid fitness tuple (inf, inf) instead of deleting
            # DEAP requires tuple length to match fitness weights (2 objectives)
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values

    return offspring


def _parallel_mutation(
    offspring: list[Any], mutpb: float, toolbox: Any, max_workers: int | None = None
) -> list[Any]:
    """
    Apply mutation sequentially.

    NOTE: ThreadPoolExecutor removed because Python's GIL prevents true parallelism
    for CPU-bound tasks. Sequential execution avoids context switching overhead.

    CRITICAL FIX: DEAP mutation returns (individual,) tuple. While most DEAP operators
    modify in-place, we MUST reassign the tuple result to handle edge cases where
    operators return new objects. This prevents GPU evaluation failures.
    """
    for i in range(len(offspring)):
        if random.random() < mutpb:
            # Call mutation operator (returns (individual,) tuple)
            result = toolbox.mutate(offspring[i])

            # CRITICAL: Must unpack and reassign even if modified in-place
            # DEAP convention: mutation returns (ind,) single-element tuple
            # Failure to reassign causes GPU evaluator to receive
            # tuple-corrupted individuals
            offspring[i] = result[0]

            # CRITICAL FIX: Force fitness invalidation
            del offspring[i].fitness.values

    return offspring


class AlwaysShowTimeRemainingColumn(ProgressColumn):
    """
    Custom TimeRemainingColumn with:
    - Always shows an estimate (never blank)
    - Updates only once per second (reduces flicker)
    - Smooths estimates using exponential moving average (reduces wild fluctuations)
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_update_time: float = 0.0
        self._cached_text: Text = Text("~calculating~", style="dim progress.remaining")
        self._ema_remaining: float | None = (
            None  # Exponential moving average for smoothing
        )
        self._alpha: float = 0.3  # Smoothing factor (0.3 = 30% new, 70% old)

    def render(self, task: Task) -> Text:
        """Render remaining time, updating at most once per second."""
        import time as time_module

        current_time = time_module.time()

        # Update only once per second to reduce flicker
        if current_time - self._last_update_time < 1.0:
            return self._cached_text

        self._last_update_time = current_time

        # Task finished
        if task.finished:
            self._cached_text = Text("0:00:00", style="progress.remaining")
            return self._cached_text

        # Calculate raw estimate
        raw_remaining = None

        # Try Rich's built-in calculation first
        if task.time_remaining is not None:
            raw_remaining = task.time_remaining
        # Fallback: Simple extrapolation
        elif task.completed > 0 and task.total and task.total > 0:
            elapsed = task.elapsed or 0
            if elapsed > 0:
                avg_time_per_unit = elapsed / task.completed
                remaining_units = task.total - task.completed
                raw_remaining = avg_time_per_unit * remaining_units

        # Apply exponential moving average for smoothing
        if raw_remaining is not None:
            if self._ema_remaining is None:
                # First estimate - initialize EMA
                self._ema_remaining = raw_remaining
            else:
                # Smooth: EMA = α * new + (1-α) * old
                self._ema_remaining = (
                    self._alpha * raw_remaining
                    + (1 - self._alpha) * self._ema_remaining
                )

            # Format smoothed estimate
            remaining = int(self._ema_remaining)
            hours, remainder = divmod(remaining, 3600)
            minutes, seconds = divmod(remainder, 60)

            if hours > 0:
                time_str = f"~{hours}:{minutes:02d}:{seconds:02d}"
            else:
                time_str = f"~{minutes}:{seconds:02d}"

            self._cached_text = Text(time_str, style="progress.remaining")
            return self._cached_text

        # No estimate available yet
        self._cached_text = Text("~calculating~", style="dim progress.remaining")
        return self._cached_text


@dataclass
class GAConfig:
    """
    GA configuration parameters.

    Attributes:
        pop_size: Population size
        generations: Number of generations to evolve
        crossover_prob: Probability of crossover operation
        mutation_prob: Probability of mutation operation
        repair_config: Repair heuristics configuration dict
                       (from ga_params.get_config().repair)
                       Includes selective_mode, adaptive_repair settings,
                       and enabled heuristics
    """

    pop_size: int
    generations: int
    crossover_prob: float
    mutation_prob: float
    repair_config: dict = field(default_factory=dict)


@dataclass
class GAMetrics:
    """
    Tracks GA evolution metrics per generation.

    Attributes:
        hard_violations: Best hard constraint violations per generation
        soft_penalties: Best soft constraint penalties per generation
        diversity: Population diversity per generation
        detailed_hard: Per-constraint hard violation tracking
        detailed_soft: Per-constraint soft penalty tracking
        repair_stats: Repair statistics per generation
        hypervolume: Hypervolume indicator per generation (multi-objective quality)
        spacing: Spacing metric per generation (Pareto front uniformity)
        pareto_front_size: Number of non-dominated solutions per generation
        igd: Inverted Generational Distance per generation (convergence + coverage)
        spread: Spread metric per generation (extent + distribution)
        feasibility_rate: Percentage of feasible solutions per generation
    """

    hard_violations: list[float] = field(default_factory=list)
    soft_penalties: list[float] = field(default_factory=list)
    diversity: list[float] = field(default_factory=list)
    detailed_hard: dict[str, list[float]] = field(default_factory=dict)
    detailed_soft: dict[str, list[float]] = field(default_factory=dict)
    repair_stats: list[dict] = field(default_factory=list)

    # Phase 1: Essential metrics
    hypervolume: list[float] = field(default_factory=list)
    spacing: list[float] = field(default_factory=list)
    feasibility_rate: list[float] = field(default_factory=list)
    pareto_front_size: list[int] = field(default_factory=list)

    # Phase 2: Advanced metrics
    igd: list[float] = field(default_factory=list)
    spread: list[float] = field(default_factory=list)

    # Reference front for IGD calculation (set once, used throughout)
    reference_front: list = field(default_factory=list)


class GAScheduler:
    """
    Manages NSGA-II genetic algorithm execution for timetabling.

    This class encapsulates the entire GA lifecycle:
    - Toolbox initialization
    - Population generation and validation
    - Evolution loop execution with adaptive repair
    - Metrics tracking
    - Best solution selection

    Adaptive Repair System:
        Implements hybrid repair strategy combining stagnation detection
        and periodic triggers. Dynamically switches between selective (fast)
        and full (intensive) repair modes based on search progress.

        - Stagnation Detection: Monitors HC improvement over rolling window
        - Periodic Triggers: Regular repair at configurable intervals
        - Intensive Triggers: Heavy repair at longer intervals
        - Dynamic Parameters: Adjusts repair_mode and max_iterations per generation

    Usage:
        config = GAConfig(pop_size=50, generations=100, ...)
        scheduler = GAScheduler(config, context, hard_names, soft_names)
        scheduler.setup_toolbox()
        scheduler.initialize_population()
        scheduler.evolve()
        best = scheduler.get_best_solution()
    """

    def __init__(
        self,
        config: GAConfig,
        context: SchedulingContext,
        hard_constraint_names: list[str],
        soft_constraint_names: list[str],
        pool=None,  # NEW: Optional multiprocessing Pool
        logger=None,  # NEW: Optional GALogger for runtime logging
        # NEW: Optional ConstraintLogger for detailed constraint logging
        constraint_logger=None,
        seed: int | None = None,  # NEW: Random seed for worker init
    ):
        """
        Initialize GA scheduler with adaptive repair tracking.

        Args:
            config: GA configuration
                    (includes repair_config with adaptive_repair settings)
            context: Scheduling context with courses, groups, instructors,
                     rooms
            hard_constraint_names: Names of enabled hard constraints
            soft_constraint_names: Names of enabled soft constraints
            pool: Optional multiprocessing.Pool for parallel fitness
                  evaluation
            logger: Optional GALogger for file-based logging
                    (writes to logger.txt, not console)
            constraint_logger: Optional ConstraintLogger for detailed
                               constraint logging
                               (writes to logger_constraints.csv)
            seed: Random seed for reproducibility

        Adaptive Repair:
            Initializes stagnation tracking variables (stagnation_counter, last_best_hc)
            for hybrid trigger logic. Console messages use global console object.
        """
        self.config = config
        self.context = context
        self.hard_constraint_names = hard_constraint_names
        self.soft_constraint_names = soft_constraint_names
        self.pool = pool  # NEW: Store pool for parallel evaluation
        self.logger = logger  # NEW: Store logger for runtime logging
        self.constraint_logger = constraint_logger  # NEW: Store constraint logger
        self.seed = seed  # NEW: Store seed for worker initialization

        # Deterministic short codes (hc1, hc2, ...) for console output/legend
        self.hard_constraint_codes = {
            name: f"hc{i+1}" for i, name in enumerate(self.hard_constraint_names)
        }
        self.soft_constraint_codes = {
            name: f"sc{i+1}" for i, name in enumerate(self.soft_constraint_names)
        }

        self.toolbox: base.Toolbox = base.Toolbox()
        self.population: list[Any] = []
        self.metrics = GAMetrics(
            detailed_hard={name: [] for name in hard_constraint_names},
            detailed_soft={name: [] for name in soft_constraint_names},
        )

        # ADAPTIVE REPAIR: Stagnation tracking for hybrid strategy
        self.stagnation_counter = 0
        self.last_best_hc = float("inf")

        # ENHANCEMENT: Hypermutation tracking
        self.hypermutation_active = False
        self.hypermutation_countdown = 0

        # ENHANCEMENT: Population restart tracking
        self.last_restart_gen = -1000  # Track last restart generation
        self.prolonged_stagnation_counter = 0  # Separate counter for restart

        # Banner tracking (prevent duplicate printing)
        self._banner_printed = False

        # ENHANCEMENT: Violation heatmap for targeted repair
        self.violation_heatmap = None
        enhancement_cfg = get_config().enhancements
        if enhancement_cfg.master_enabled and enhancement_cfg.violation_heatmap.enabled:
            from src.metrics.violation_heatmap import ViolationHeatmap

            self.violation_heatmap = ViolationHeatmap()
            console.print("[dim]   Violation heatmap tracking: ENABLED[/dim]")

        # GPU acceleration REMOVED from GA loop - CPU multiprocessing only
        # GPU is reserved for RL training/inference (better suited for neural networks)
        console.print("[dim]   GA fitness evaluation: CPU multiprocessing only[/dim]")

        # NEW: Hypervolume reference point (initialized during first metric tracking)
        self._hypervolume_ref_point = None

        # PERFORMANCE CACHE: Store detailed constraint breakdown to avoid re-evaluation

        # TRACKING: Initial and best solutions for improvement reporting
        self.initial_best_hard = None
        self.initial_best_soft = None
        self.all_time_best = None  # Track best individual ever seen
        self._cached_hard_details: dict[str, int] = {}
        self._cached_soft_details: dict[str, int] = {}

        # PERFORMANCE CACHE: Store enabled constraints (computed once, used frequently)
        self._enabled_hard_constraints = get_enabled_hard_constraints()
        self._enabled_soft_constraints = get_enabled_soft_constraints()

        # RL INTEGRATION: Components for hyper-heuristic control
        self.rl_enabled = False
        self.rl_controller: Any | None = None  # HybridController (RL integration)
        self.rl_state_encoder: Any | None = None  # StateEncoder (RL integration)
        self.rl_action_mapper: Any | None = None  # ActionMapper (RL integration)

        # HEURISTIC TRACKING: Round-robin tracking and detailed statistics
        from src.ga.heuristic_tracker import HeuristicTracker

        self.heuristic_tracker = HeuristicTracker()
        self.heuristic_stats = get_heuristic_statistics_template()
        self._setup_heuristic_rotation()

        # PERFORMANCE: Parallel heuristic executor (10-16x speedup)
        self.parallel_executor: ParallelHeuristicExecutor | None
        try:
            self.parallel_executor = get_parallel_executor()
            console.print(
                "[dim]   Parallel heuristic executor: ENABLED (10-16x speedup)[/dim]"
            )
        except Exception as e:
            logger.warning(f"Parallel executor init failed: {e}")
            self.parallel_executor = None

    def setup_toolbox(self):
        """Initialize DEAP toolbox with operators."""
        self.toolbox = base.Toolbox()

        # NEW: Register parallel map if pool is provided
        if self.pool is not None:
            self.toolbox.register("map", self.pool.map)

        # Selection operator
        # Use fast NSGA-II for large populations (5-10x faster)
        if get_config().ga.pop_size >= 200:
            from src.ga.operators.fast_nsga2 import sel_nsga2_fast

            self.toolbox.register("select", sel_nsga2_fast)
        else:
            self.toolbox.register("select", tools.selNSGA2)

        # PHASE 3: Hybrid population initialization support
        strategy = get_config().ga.population_strategy

        if strategy == "hybrid":
            from src.ga.hybrid_population import generate_hybrid_population

            self.toolbox.register(
                "population", generate_hybrid_population, context=self.context
            )
        elif strategy == "smart":
            # Original constraint-aware (Phase 1+2 default)
            self.toolbox.register(
                "population",
                generate_course_group_aware_population,
                context=self.context,
            )
        elif strategy == "random":
            # Pure random initialization (no heuristics, no conflict avoidance)
            from src.ga.population import generate_pure_random_population

            self.toolbox.register(
                "population",
                generate_pure_random_population,
                context=self.context,
            )
        else:  # Unknown strategy defaults to smart
            self.toolbox.register(
                "population",
                generate_course_group_aware_population,
                context=self.context,
            )

        # Evaluation operator
        # Use worker initialization pattern for parallel execution
        if self.pool is not None:
            # Parallel mode: use worker evaluation (context already in workers)
            self.toolbox.register("evaluate", _worker_evaluate)
        else:
            # Sequential mode: use direct evaluation with bound context
            self.toolbox.register(
                "evaluate",
                evaluate,
                courses=self.context.courses,
                instructors=self.context.instructors,
                groups=self.context.groups,
                rooms=self.context.rooms,
            )

        # Genetic operators
        self.toolbox.register(
            "mate", crossover_course_group_aware, cx_prob=self.config.crossover_prob
        )

        # PHASE 2: Constraint-guided mutation support
        self.toolbox.register(
            "mutate",
            mutate_individual,
            context=self.context,
            mut_prob=self.config.mutation_prob,
            # Enable constraint-guided mutation
            guided=get_config().ga.use_constraint_guided_mutation,
        )

    def _init_rl(self) -> bool:
        """
        Initialize RL components for hyper-heuristic control.

        Returns:
            True if RL initialized successfully, False otherwise
        """
        rl_config = get_config().rl

        # Check if RL is enabled in configuration
        if not rl_config.enabled:
            return False

        # Check mode (must be 'inference' or 'hybrid' for GA integration)
        if rl_config.mode not in ["inference", "hybrid"]:
            console.print(
                f"[yellow]RL mode '{rl_config.mode}' not compatible with "
                f"GA integration[/yellow]"
            )
            console.print(
                "[dim]   Use mode 'inference' or 'hybrid' for " "production runs[/dim]"
            )
            return False

        try:
            # Import RL components (lazy import to avoid dependency issues)
            from src.rl.deployment.inference import RLInference
            from src.rl.deployment.model_loader import ModelLoader
            from src.rl.gym_env.action_space import ActionMapper
            from src.rl.gym_env.state_encoder import StateEncoder
            from src.rl.hybrid.hybrid_controller import (
                FallbackStrategy,
                HybridController,
                HybridMode,
            )

            console.print("[cyan]Initializing RL Components...[/cyan]")

            # Initialize state encoder
            self.rl_state_encoder = StateEncoder(
                max_generations=self.config.generations,
                history_size=rl_config.environment.observation_history_size,
                normalize=True,
            )
            console.print("   [green][!ok][/green] StateEncoder initialized")

            # Initialize action mapper
            self.rl_action_mapper = ActionMapper(use_config=True)
            console.print(
                f"   [green][!ok][/green] ActionMapper initialized "
                f"({self.rl_action_mapper.n_actions} actions)"
            )

            # Load trained model
            model_path = rl_config.agent.model_path
            if not model_path or model_path == "models/rl_agents/best_model.zip":
                # Try to find best model from manifest
                try:
                    from src.rl.training.checkpoints import CheckpointManager

                    manifest_path = rl_config.training.checkpoint_settings.manifest_path
                    manager = CheckpointManager(manifest_path)
                    best_checkpoint = manager.get_best_checkpoint(
                        metric_name="mean_reward"
                    )
                    if best_checkpoint:
                        model_path = best_checkpoint.model_path
                        console.print(
                            f"   [dim]Using best checkpoint: {model_path}[/dim]"
                        )
                except Exception as e:
                    console.print(
                        f"   [yellow]Could not load best checkpoint: {e}[/yellow]"
                    )

            # Initialize model loader and load model
            loader = ModelLoader(cache_models=True)
            model: BaseAlgorithm = loader.load_model(
                model_path, agent_type=rl_config.agent.type
            )
            console.print(
                f"   [green][!ok][/green] Model loaded: {rl_config.agent.type.upper()}"
            )

            # Initialize inference engine
            inference_engine = RLInference(
                model=model,
                timeout_ms=rl_config.inference.timeout_ms,
            )

            # Initialize hybrid controller
            hybrid_mode = HybridMode(rl_config.hybrid.mode)
            fallback_strategy = FallbackStrategy(rl_config.hybrid.fallback_strategy)
            enable_action_masking = getattr(
                rl_config.hybrid, "enable_action_masking", True
            )

            self.rl_controller = HybridController(
                rl_inference=inference_engine,
                mode=hybrid_mode,
                fallback_strategy=fallback_strategy,
                rl_probability=rl_config.hybrid.rl_probability,
                enable_action_masking=enable_action_masking,
            )
            console.print(
                f"   [green][!ok][/green] HybridController initialized "
                f"(mode: {rl_config.hybrid.mode})"
            )

            self.rl_enabled = True
            console.print("[green]RL Integration: ENABLED[/green]")
            return True

        except ImportError as e:
            console.print(f"[yellow]RL components not available: {e}[/yellow]")
            console.print(
                "[dim]   Install RL dependencies: "
                "uv add gymnasium stable-baselines3[/dim]"
            )
            return False
        except FileNotFoundError as e:
            console.print(f"[yellow]RL model not found: {e}[/yellow]")
            console.print(
                "[dim]   Train model first: "
                "python src/rl/training/train_script.py[/dim]"
            )
            return False
        except Exception as e:
            console.print(f"[red]RL initialization failed: {e}[/red]")
            logger.exception("RL initialization error")
            return False

    def _apply_rl_operators(self, gen: int) -> None:
        """
        Apply RL-selected heuristics to population.

        Uses trained RL agent to select and apply adaptive operators
        based on current population state.

        Args:
            gen: Current generation number
        """
        if not self.rl_enabled or not self.rl_controller:
            return

        # Encode current state
        state = self.rl_state_encoder.encode(  # type: ignore[union-attr]
            population=self.population,
            current_generation=gen,
            generations_without_improvement=self.stagnation_counter,
        )

        # Get valid actions for current state
        valid_actions = self.rl_action_mapper.enabled_actions  # type: ignore[union-attr]

        # Select action using RL controller (with fallback)
        action_id = self.rl_controller.select_action(  # type: ignore[union-attr]
            state=state,
            valid_actions=valid_actions,
            deterministic=True,  # Use deterministic policy for production
        )

        # DEBUG: Log RL decision
        action_info = self.rl_action_mapper.get_action_info(action_id)
        if action_info:
            logger.debug(f" RL selected: {action_info.name} (action_id={action_id})")

        # Record heuristic application for state tracking
        self.rl_state_encoder.record_heuristic_application(action_id)

        # Apply selected heuristic to population
        try:
            # Get action info to determine heuristic type

            # Apply heuristic in parallel to top N individuals for 10-16x speedup
            # (improvement heuristics benefit from parallel application)
            if action_info and action_info.category == "improvement":
                # Select top 4-8 individuals based on population size
                num_targets = min(8, max(4, len(self.population) // 25))
                top_individuals = tools.selBest(self.population, num_targets)
                before_fitness = [tuple(ind.fitness.values) for ind in top_individuals]

                logger.debug(
                    f"Gen {gen}: RL applying '{action_info.name}' to "
                    f"{num_targets} individuals in parallel"
                )

                heuristic_func = action_info.function
                if heuristic_func:
                    parallel_executor = get_parallel_executor()
                    parallel_results = parallel_executor.apply_parallel(
                        heuristic_func=heuristic_func,
                        individuals=top_individuals,
                        context=self.context,
                    )
                    for individual, result in zip(
                        top_individuals, parallel_results or [], strict=True
                    ):
                        if isinstance(result, list):
                            individual[:] = result
                    modified_individuals = top_individuals
                else:
                    modified_individuals = []
                modified_before = before_fitness
            else:
                # For non-improvement heuristics, use single best individual
                best_ind = tools.selBest(self.population, 1)[0]
                modified_before = [tuple(best_ind.fitness.values)]

                # Apply action and get modified individual(s)
                modified_ind, success = self.rl_action_mapper.apply_action(
                    action=action_id,
                    individual=best_ind,
                    context=self.context,
                    population=self.population,
                    generation=gen,
                )
                modified_individuals = [modified_ind] if success else []

            # Evaluate modified individuals
            if modified_individuals:
                fitness_values = list(
                    self.toolbox.map(self.toolbox.evaluate, modified_individuals)
                )
                for ind, fit in zip(modified_individuals, fitness_values, strict=True):
                    ind.fitness.values = fit
                if action_info:
                    for before, fit in zip(
                        modified_before, fitness_values, strict=True
                    ):
                        self._record_heuristic_stat(
                            action_info.name,
                            self._is_improvement(before, fit),
                        )
            elif action_info:
                self._record_heuristic_stat(action_info.name, False)

                # Log action application (optional)
                rl_config = get_config().rl
                if rl_config.logging.log_heuristic_usage:
                    action_info = self.rl_action_mapper.get_action_info(action_id)
                    action_name = (
                        action_info.name if action_info else f"action_{action_id}"
                    )
                    logger.debug(
                        f"Gen {gen}: RL applied '{action_name}' "
                        f"(modified {len(modified_individuals)} individuals)"
                    )

        except Exception as e:
            logger.warning(f"RL action application failed at gen {gen}: {e}")

    def _setup_heuristic_rotation(self) -> None:
        """
        Setup round-robin heuristic rotation from enabled heuristics.

        Builds ordered list based on priority and category.
        Includes REPAIR as a pseudo-heuristic ONLY if repair.enabled=true.
        """
        from src.heuristics import get_enabled_heuristics

        # Get all enabled heuristics sorted by priority
        enabled = get_enabled_heuristics()

        if not enabled:
            console.print("[dim]   No heuristics enabled for round-robin[/dim]")
            return

        # Build rotation order from all enabled heuristics
        heuristic_names = list(enabled.keys())

        # Set rotation order in tracker (includes repair heuristics if enabled)
        self.heuristic_tracker.set_heuristic_order(heuristic_names)

        # Count repair heuristics for display
        repair_heuristics = [h for h in heuristic_names if "repair" in h.lower()]

        if repair_heuristics:
            console.print(
                f"[dim]   Round-robin rotation: {len(heuristic_names)} "
                f"heuristics (including {len(repair_heuristics)} repair "
                f"operators)[/dim]"
            )
        else:
            console.print(
                f"[dim]   Round-robin rotation: {len(heuristic_names)} heuristics (NO repair)[/dim]"
            )

    def _apply_round_robin_heuristics(self, gen: int) -> None:
        """
        Apply heuristics in round-robin order.

        Each generation cycles to the next heuristic in the priority-ordered list.
        Tracks application results for detailed analysis.

        Args:
            gen: Current generation number
        """
        import time

        from deap import tools

        from src.heuristics import get_enabled_heuristics

        # Check if any heuristics are enabled
        if not self.heuristic_tracker.heuristic_order:
            return

        # ADAPTIVE PRIORITY ADJUSTMENT: Reorder heuristics based on recent effectiveness
        # Check if adaptive priority is enabled and it's time to reorder
        full_config = (
            get_config()
        )  # Get full config (not just self.config which is GAConfig)
        adaptive_config = full_config.heuristics.adaptive_priority
        if adaptive_config.get("enabled", False):
            reorder_interval = adaptive_config.get("reorder_interval", 10)

            # Reorder every N generations (and at generation 0 after some data)
            if gen > 0 and gen % reorder_interval == 0:
                order_changed = self.heuristic_tracker.reorder_by_effectiveness(
                    current_generation=gen,
                    window_size=adaptive_config.get("evaluation_window", 10),
                    min_applications=adaptive_config.get("min_applications", 3),
                )

                if order_changed:
                    # Show reordered list with effectiveness scores
                    scores = self.heuristic_tracker.get_effectiveness_summary()
                    console.print(
                        f"[yellow]    Gen {gen}: Reordered heuristics "
                        f"by effectiveness:[/yellow]"
                    )
                    for i, h_name in enumerate(
                        self.heuristic_tracker.heuristic_order[:5], 1
                    ):
                        score = scores.get(h_name, 0.0)
                        console.print(f"      {i}. {h_name}: {score:+.3f}")
                    if len(self.heuristic_tracker.heuristic_order) > 5:
                        console.print(
                            f"      ... and "
                            f"{len(self.heuristic_tracker.heuristic_order) - 5} "
                            f"more"
                        )

        # Get next heuristic in rotation
        heuristic_name = self.heuristic_tracker.get_next_heuristic()

        # Get heuristic metadata
        enabled_heuristics = get_enabled_heuristics()
        heuristic_meta = enabled_heuristics.get(heuristic_name)

        if not heuristic_meta:
            logger.warning(
                f"Heuristic '{heuristic_name}' not found in enabled heuristics"
            )
            return

        # DEBUG: Log heuristic application
        logger.debug(
            f" Applying heuristic: {heuristic_name} ({heuristic_meta.category.value})"
        )

        # Skip construction heuristics (they generate NEW individuals, not modify existing)
        if heuristic_meta.category.value == "construction":
            return

        # Select target individual(s)
        if (
            heuristic_meta.requires_population
            or heuristic_meta.modifies_individual
            and len(self.population) > 1
        ):
            target_size = min(4, len(self.population))
        else:
            target_size = 1

        target_individuals = tools.selBest(self.population, target_size)
        if not target_individuals:
            return

        fitness_before = [tuple(ind.fitness.values) for ind in target_individuals]

        # CONSOLE LOG: Show which heuristic is being applied
        heuristic_start_time = time.time()
        console.print(
            f"[cyan]   -> Gen {gen}: Heuristic '{heuristic_name}' "
            f"({heuristic_meta.category.value}) -> "
            f"{len(target_individuals)} ind(s)...[/cyan]"
        )

        if heuristic_meta.requires_population:
            self._apply_population_heuristic(
                heuristic_name,
                heuristic_meta,
                target_individuals,
                fitness_before,
                gen,
            )
        else:
            self._apply_standard_heuristic_batch(
                heuristic_name,
                heuristic_meta,
                target_individuals,
                fitness_before,
                gen,
            )

        # CONSOLE LOG: Show heuristic completion with timing and improvement
        total_heuristic_time = time.time() - heuristic_start_time
        console.print(
            f"[green]   OK Gen {gen}: '{heuristic_name}' done in {total_heuristic_time:.2f}s[/green]"
        )

    def _apply_population_heuristic(
        self,
        heuristic_name: str,
        heuristic_meta,
        target_individuals,
        fitness_before,
        gen: int,
    ) -> None:
        """Apply heuristics that require population context sequentially."""
        population_snapshot = [list(ind) for ind in self.population]

        for idx, individual in enumerate(target_individuals):
            start_time = time.time()
            before = fitness_before[idx]

            try:
                if heuristic_name == "distance_preserving_crossover":
                    parent2 = tools.selRandom(self.population, 1)[0]
                    result = heuristic_meta.function(
                        parent1=individual,
                        parent2=parent2,
                        context=self.context,
                    )
                    if isinstance(result, tuple) and len(result) >= 2:
                        offspring = result[0]
                        if isinstance(offspring, list):
                            individual[:] = offspring
                elif heuristic_name == "adaptive_diversity_maintenance":
                    result = heuristic_meta.function(
                        individual=individual,
                        population=population_snapshot,
                        context=self.context,
                        generation=gen,
                    )
                    if heuristic_meta.modifies_individual and isinstance(result, list):
                        individual[:] = result
                else:
                    result = heuristic_meta.function(
                        individual=individual,
                        population=population_snapshot,
                        context=self.context,
                    )
                    if heuristic_meta.modifies_individual and isinstance(result, list):
                        individual[:] = result

                if heuristic_meta.modifies_individual:
                    fitness_after = evaluate(
                        individual,
                        self.context.courses,
                        self.context.instructors,
                        self.context.groups,
                        self.context.rooms,
                    )
                    individual.fitness.values = fitness_after
                else:
                    fitness_after = individual.fitness.values

                execution_time = time.time() - start_time
                self.heuristic_tracker.record_application(
                    generation=gen,
                    heuristic_name=heuristic_name,
                    category=heuristic_meta.category.value,
                    fitness_before=before,
                    fitness_after=fitness_after,
                    execution_time=execution_time,
                    individual_id=self.population.index(individual),
                )
                self._record_heuristic_stat(
                    heuristic_name, self._is_improvement(before, fitness_after)
                )

            except Exception as exc:
                logger.warning(
                    f"Heuristic '{heuristic_name}' failed at gen {gen}: {exc}"
                )
                console.print(
                    f"[yellow]   WARN Gen {gen}: Heuristic '{heuristic_name}' "
                    f"failed: {exc}[/yellow]"
                )
                if (
                    not hasattr(individual.fitness, "values")
                    or not individual.fitness.valid
                ):
                    fitness_after = evaluate(
                        individual,
                        self.context.courses,
                        self.context.instructors,
                        self.context.groups,
                        self.context.rooms,
                    )
                    individual.fitness.values = fitness_after
                else:
                    fitness_after = individual.fitness.values

                self.heuristic_tracker.record_application(
                    generation=gen,
                    heuristic_name=heuristic_name,
                    category=heuristic_meta.category.value,
                    fitness_before=before,
                    fitness_after=fitness_after,
                    execution_time=0.0,
                    individual_id=self.population.index(individual),
                )
                self._record_heuristic_stat(heuristic_name, False)

    def _apply_standard_heuristic_batch(
        self,
        heuristic_name: str,
        heuristic_meta,
        target_individuals,
        fitness_before,
        gen: int,
    ) -> None:
        """Apply individual-focused heuristics using parallel executor when possible."""
        if not target_individuals:
            return

        results = None
        per_individual_times = []
        if self.parallel_executor and len(target_individuals) > 1:
            try:
                start = time.time()
                results = self.parallel_executor.apply_parallel(
                    heuristic_func=heuristic_meta.function,
                    individuals=target_individuals,
                    context=self.context,
                )
                elapsed = time.time() - start
                if len(results) == len(target_individuals):
                    per_individual_times = [elapsed / len(target_individuals)] * len(
                        target_individuals
                    )
                else:
                    results = None
            except Exception as exc:
                logger.warning(
                    f"Parallel heuristic '{heuristic_name}' failed, falling back: {exc}"
                )
                results = None

        if not results:
            results = []
            for individual in target_individuals:
                start = time.time()
                try:
                    result = heuristic_meta.function(
                        individual=individual,
                        context=self.context,
                    )
                    results.append(result)
                except Exception as exc:
                    logger.warning(
                        f"Heuristic '{heuristic_name}' failed at gen {gen}: {exc}"
                    )
                    console.print(
                        f"[yellow]   WARN Gen {gen}: Heuristic '{heuristic_name}' failed: {exc}[/yellow]"
                    )
                    results.append(individual)
                finally:
                    per_individual_times.append(time.time() - start)

        for idx, individual in enumerate(target_individuals):
            before = fitness_before[idx]
            result = results[idx] if idx < len(results) else None

            if heuristic_meta.modifies_individual and isinstance(result, list):
                individual[:] = result

            if heuristic_meta.modifies_individual:
                fitness_after = evaluate(
                    individual,
                    self.context.courses,
                    self.context.instructors,
                    self.context.groups,
                    self.context.rooms,
                )
                individual.fitness.values = fitness_after
            else:
                fitness_after = individual.fitness.values

            exec_time = (
                per_individual_times[idx] if idx < len(per_individual_times) else 0.0
            )

            self.heuristic_tracker.record_application(
                generation=gen,
                heuristic_name=heuristic_name,
                category=heuristic_meta.category.value,
                fitness_before=before,
                fitness_after=fitness_after,
                execution_time=exec_time,
                individual_id=self.population.index(individual),
            )

            self._record_heuristic_stat(
                heuristic_name, self._is_improvement(before, fitness_after)
            )

    @staticmethod
    def _is_improvement(
        before: Sequence[float] | None, after: Sequence[float] | None
    ) -> bool:
        """Return True if new fitness dominates previous fitness."""
        if before is None or after is None:
            return False

        if len(before) < 2 or len(after) < 2:
            return False

        hard_before, soft_before = float(before[0]), float(before[1])
        hard_after, soft_after = float(after[0]), float(after[1])

        if hard_after < hard_before:
            return True
        if hard_after > hard_before:
            return False
        return soft_after < soft_before

    def _record_heuristic_stat(self, heuristic_name: str, improved: bool) -> None:
        """Update heuristic stats counters for telemetry."""
        if not heuristic_name:
            return

        applications_key = f"{heuristic_name}_applications"
        improvements_key = f"{heuristic_name}_improvements"

        self.heuristic_stats.setdefault("total_applications", 0)
        self.heuristic_stats.setdefault("total_improvements", 0)
        self.heuristic_stats.setdefault(applications_key, 0)
        self.heuristic_stats.setdefault(improvements_key, 0)

        self.heuristic_stats[applications_key] += 1
        self.heuristic_stats["total_applications"] += 1

        if improved:
            self.heuristic_stats[improvements_key] += 1
            self.heuristic_stats["total_improvements"] += 1

    def initialize_population(self):
        """Create and evaluate initial population."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Initializing Population...", total=2)

            self.population = self.toolbox.population(n=self.config.pop_size)
            progress.advance(task)

            # Validate gene alignment
            self._validate_population_structure()
            progress.advance(task)

        # Evaluate initial population with progress tracking
        console.print("[cyan]Evaluating Initial Population...[/cyan]")
        eval_start = time.time()

        # Use toolbox.map for parallel evaluation when pool is available
        fitness_values = list(self.toolbox.map(self.toolbox.evaluate, self.population))

        for ind, fit in zip(self.population, fitness_values, strict=True):
            ind.fitness.values = fit

        eval_time = time.time() - eval_start
        console.print(
            f"   [green][!ok][/green] Evaluated {len(self.population)} individuals in [cyan]{eval_time:.1f}s[/cyan] "
            f"([dim]{eval_time/len(self.population):.2f}s per individual[/dim])"
        )

        # Show initial best fitness
        best = tools.selBest(self.population, 1)[0]
        console.print(
            f"   [dim]Initial Best:[/dim] Hard=[yellow]{best.fitness.values[0]:.0f}[/yellow], "
            f"Soft=[blue]{best.fitness.values[1]:.2f}[/blue]"
        )
        console.print()

        # Track initial values for improvement calculation
        self.initial_best_hard = abs(best.fitness.values[0])
        self.initial_best_soft = abs(best.fitness.values[1])
        self.all_time_best = self.toolbox.clone(best)

        # Track initial population as Generation 0 (skip expensive metrics for speed)
        # NOTE: We defer expensive metric calculation to generation 0 to avoid 2-min startup delay
        # The initial population metrics are not useful for analysis anyway

        # RL INTEGRATION: Initialize RL components after population is ready
        self._init_rl()

        # Log initial population to logger
        if self.logger:
            diversity = average_pairwise_diversity(self.population)
            self.logger.log_generation(
                generation=-1,
                hard_violations=best.fitness.values[0],
                soft_penalty=best.fitness.values[1],
                time_seconds=eval_time,
                diversity=diversity,
                repairs=0,
                notes="Initial population",
            )

        # Log initial population to constraint logger
        if self.constraint_logger:
            diversity = average_pairwise_diversity(self.population)
            hard_details, soft_details = evaluate_detailed(
                best,
                self.context.courses,
                self.context.instructors,
                self.context.groups,
                self.context.rooms,
            )
            self.constraint_logger.log_generation(
                generation=-1,
                hard_total=best.fitness.values[0],
                soft_total=best.fitness.values[1],
                hard_breakdown=hard_details,
                soft_breakdown=soft_details,
                diversity=diversity,
                time_seconds=eval_time,
                hypervolume=0.0,  # Will be calculated in first _track_metrics call
                spacing=0.0,
                igd=0.0,
                spread=0.0,
                repair_stats={},
                events=[],
                notes="Initial population",
            )

    def _cleanup_rl(self):
        """
        Release RL resources.

        Cleans up RL components (models, caches) to prevent memory leaks
        in long-running or repeated GA executions.
        """
        if not hasattr(self, "rl_controller") or self.rl_controller is None:
            return

        try:
            # Clear inference engine cache
            if hasattr(self.rl_controller, "inference_engine") and hasattr(
                self.rl_controller.inference_engine, "clear_cache"
            ):
                self.rl_controller.inference_engine.clear_cache()

            # Release model references
            if hasattr(self, "rl_action_mapper"):
                del self.rl_action_mapper

            if hasattr(self, "rl_state_encoder"):
                del self.rl_state_encoder

            if hasattr(self, "rl_controller"):
                del self.rl_controller

            logger.debug("RL components cleaned up successfully")

        except Exception as e:
            logger.warning(f"Error during RL cleanup: {e}")

    def _print_startup_banner(self):
        """Print configuration banner before evolution starts."""
        # Prevent duplicate printing
        if self._banner_printed:
            return
        self._banner_printed = True

        from rich.panel import Panel
        from rich.table import Table

        config = get_config()

        # Build feature table - ONLY show enabled features
        feature_table = Table(show_header=False, box=None, padding=(0, 2))
        feature_table.add_column("Feature", style="bold green")
        feature_table.add_column("Value", justify="right", style="cyan")

        enabled_features = []

        # === GENETIC OPERATORS ===
        if config.ga.use_constraint_guided_mutation:
            enabled_features.append(("Constraint-guided mutation", "ON"))

        if config.ga.population_strategy != "random":
            enabled_features.append(
                ("Population strategy", config.ga.population_strategy)
            )

        # === REPAIR SYSTEM ===
        if config.repair.enabled:
            enabled_features.append(("Repair system", "ON"))
            if config.repair.memetic_mode:
                enabled_features.append(("  > Memetic mode", "ON"))
            if config.repair.apply_after_mutation:
                enabled_features.append(("  > Apply after mutation", "ON"))

        # === HEURISTICS ===
        if config.heuristics.master_enabled:
            enabled_features.append(("Heuristics", "ON"))
            if config.heuristics.adaptive_priority.enabled:
                enabled_features.append(("  > Adaptive priority", "ON"))

        # === LARGE NEIGHBORHOOD SEARCH ===
        if config.lns.enabled:
            enabled_features.append(("Large Neighborhood Search", "ON"))

        # === RL SYSTEM ===
        if config.rl.enabled:
            enabled_features.append(("RL hyper-heuristic", "ON"))
            enabled_features.append(("  > Mode", config.rl.mode))

        # === ENHANCEMENTS ===
        if config.enhancements.master_enabled:
            enabled_features.append(("Enhancements", "ON"))
            if config.enhancements.hypermutation.enabled:
                enabled_features.append(("  > Hypermutation", "ON"))
            if config.enhancements.population_restart.enabled:
                enabled_features.append(("  > Population restart", "ON"))

        # Add rows to table
        if enabled_features:
            for feature, value in enabled_features:
                feature_table.add_row(feature, value)
        else:
            # Pure baseline - no enhancements
            feature_table.add_row(
                "[dim]Pure NSGA-II baseline[/dim]", "[dim](no enhancements)[/dim]"
            )

        # Print banner
        console.print()
        console.print(
            Panel(
                feature_table,
                title="[bold white]Active Features[/bold white]",
                border_style="blue",
                padding=(1, 2),
            )
        )
        console.print()

    def evolve(self):
        """Run genetic algorithm evolution loop."""
        try:
            self._run_evolution()
        finally:
            # Always cleanup RL resources
            self._cleanup_rl()

    def _run_evolution(self):
        """Internal evolution loop implementation."""
        gen_times = []
        profiler = get_profiler()

        # ========================================
        # STARTUP CONFIGURATION BANNER
        # ========================================
        self._print_startup_banner()

        # Create elapsed/remaining time bar (shows above progress bar)
        time_bar = Progress(
            TextColumn("[dim]elapsed:[/dim]"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TextColumn("[dim]remaining:[/dim]"),
            AlwaysShowTimeRemainingColumn(),  # Custom column that never shows blank
            TextColumn("•"),
            TextColumn("[dark_red]{task.fields[speed_display]}[/dark_red]"),
            console=console,
            refresh_per_second=10,
        )

        # Create progress bar (evolution progress with distinct thick style)
        progress_bar = Progress(
            SpinnerColumn(),
            TextColumn("[bold white]{task.description}[/bold white]"),
            BarColumn(
                bar_width=40,
                style="dim white",
                complete_style="bold blue",
                finished_style="bold green",
            ),
            TextColumn("[bold white]{task.completed}/{task.total}[/bold white]"),
            console=console,
            refresh_per_second=10,
        )

        # Create spacing row for better terminal display
        spacing_bar = Progress(
            TextColumn(""),
            console=console,
            refresh_per_second=10,
        )

        # Create constraint legend rows (static text that always shows)
        # Display in compact form: 3 constraints per row
        legend_bars = []
        legend_bars.append(
            Progress(TextColumn(""), console=console, refresh_per_second=1)
        )  # spacing
        legend_bars.append(
            Progress(
                TextColumn("[dim]constraint mapping:[/dim]"),
                console=console,
                refresh_per_second=1,
            )
        )

        # Get constraint names for the legend
        best = tools.selBest(self.population, 1)[0]
        hard_details, soft_details = evaluate_detailed(
            best,
            self.context.courses,
            self.context.instructors,
            self.context.groups,
            self.context.rooms,
        )

        # Build hard constraint labels (3 per row) using deterministic order
        hard_items = []
        for name in self.hard_constraint_names:
            clean_name = name.replace("_", " ")
            code = self.hard_constraint_codes.get(name, f"hc{len(hard_items)+1}")
            hard_items.append(f"{code}={clean_name}")

        for name in hard_details:
            if name not in self.hard_constraint_codes:
                clean_name = name.replace("_", " ")
                hard_items.append(f"{name[:4]}={clean_name}")

        # Display hard constraints 3 per row
        for i in range(0, len(hard_items), 3):
            row_items = hard_items[i : i + 3]
            row_text = "  [dim]" + " | ".join(row_items) + "[/dim]"
            legend_bars.append(
                Progress(
                    TextColumn(row_text),
                    console=console,
                    refresh_per_second=1,  # Static text, no need for frequent updates
                )
            )

        # Build soft constraint labels (3 per row)
        soft_items = []
        for name in self.soft_constraint_names:
            clean_name = name.replace("_", " ")
            code = self.soft_constraint_codes.get(name, f"sc{len(soft_items)+1}")
            soft_items.append(f"{code}={clean_name}")

        for name in soft_details:
            if name not in self.soft_constraint_codes:
                clean_name = name.replace("_", " ")
                soft_items.append(f"{name[:4]}={clean_name}")

        # Display soft constraints 3 per row
        for i in range(0, len(soft_items), 3):
            row_items = soft_items[i : i + 3]
            row_text = "  [dim]" + " | ".join(row_items) + "[/dim]"
            legend_bars.append(
                Progress(
                    TextColumn(row_text),
                    console=console,
                    refresh_per_second=1,  # Static text, no need for frequent updates
                )
            )

        # Combine both into a table for multi-line display
        progress_table = Table.grid()
        progress_table.add_row(spacing_bar)  # spacing above
        progress_table.add_row(time_bar)
        progress_table.add_row(progress_bar)
        # Add constraint legend rows
        for legend_bar in legend_bars:
            progress_table.add_row(legend_bar)
        # Add spacing at bottom
        progress_table.add_row(spacing_bar)

        # Use Live display with reduced refresh rate to prevent duplication on resize
        with Live(
            progress_table,
            console=console,
            refresh_per_second=2,  # Reduced from 10 to prevent render issues
            transient=False,  # Don't clear on exit
        ):
            # Helper function to format time as hh:mm:ss
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"

            task1 = progress_bar.add_task(
                "evol prog",
                total=self.config.generations,
            )
            task2 = time_bar.add_task(
                "",
                total=self.config.generations,
                speed_display="--s/gen",
            )

            # Initialize legend bar tasks (static display)
            for legend_bar in legend_bars:
                legend_bar.add_task("")

            # Initialize spacing task
            spacing_bar.add_task("")

            for gen in range(self.config.generations):
                gen_start = time.time()
                self._evolve_generation(gen, progress_bar)
                gen_time = time.time() - gen_start
                gen_times.append(gen_time)

                # Update timing in constraint logger (last logged entry)
                if self.constraint_logger:
                    self.constraint_logger.update_last_generation_time(gen_time)

                progress_bar.advance(task1)
                time_bar.advance(task2)

                # Calculate speed display
                if gen_times:
                    avg_gen_time = sum(gen_times) / len(gen_times)
                    if avg_gen_time < 1.0:
                        speed_display = f"{avg_gen_time*1000:.0f}ms/gen"
                    else:
                        speed_display = f"{avg_gen_time:.1f}s/gen"
                else:
                    speed_display = "--s/gen"

                time_bar.update(task2, speed_display=speed_display)

                # Show progress feedback after EVERY generation completes
                # Display constraint breakdown for non-zero violations
                _display_start = time.time()

                # PERFORMANCE FIX: Use cached detailed evaluation from _track_metrics()
                # This avoids re-evaluating the best individual (saves ~2 seconds per generation!)
                _eval_detailed_start = time.time()
                _best_selection_start = time.time()
                best = tools.selBest(self.population, 1)[0]
                _best_selection_time = time.time() - _best_selection_start

                if hasattr(self, "_cached_hard_details") and hasattr(
                    self, "_cached_soft_details"
                ):
                    # Use cached values (already computed in _track_metrics)
                    hard_details = self._cached_hard_details
                    soft_details = self._cached_soft_details
                    _eval_detailed_time = 0.0  # No re-evaluation needed
                else:
                    # Fallback: evaluate if cache not available (shouldn't happen)
                    hard_details, soft_details = evaluate_detailed(
                        best,
                        self.context.courses,
                        self.context.instructors,
                        self.context.groups,
                        self.context.rooms,
                    )
                    _eval_detailed_time = time.time() - _eval_detailed_start

                # Build compact constraint lists - SHOW RAW VIOLATIONS (not weighted)
                # Get raw violations by dividing by weights
                _constraint_format_start = time.time()
                enabled_hc = (
                    self._enabled_hard_constraints
                )  # Use cached dict (computed once in __init__)
                enabled_sc = (
                    self._enabled_soft_constraints
                )  # Use cached dict (computed once in __init__)

                hc_parts = []
                for name in self.hard_constraint_names:
                    short_name = self.hard_constraint_codes.get(name, name[:4])
                    weighted_val = hard_details.get(name, 0)
                    weight = enabled_hc.get(name, {}).get("weight", 1.0)
                    raw_val = int(weighted_val / weight) if weight > 0 else 0
                    hc_parts.append(f"{short_name}={raw_val}")

                # Include any dynamically added constraints not present when scheduler initialized
                for name, val in hard_details.items():
                    if name not in self.hard_constraint_codes:
                        hc_parts.append(f"{name[:4]}={int(val)}")

                sc_parts = []
                for name in self.soft_constraint_names:
                    short_name = self.soft_constraint_codes.get(name, name[:4])
                    weighted_val = soft_details.get(name, 0.0)
                    weight = enabled_sc.get(name, {}).get("weight", 1.0)
                    raw_val = weighted_val / weight if weight > 0 else 0
                    sc_parts.append(f"{short_name}={raw_val:.1f}")

                for name, val in soft_details.items():
                    if name not in self.soft_constraint_codes:
                        sc_parts.append(f"{name[:4]}={val:.1f}")

                # Build constraint list strings
                hc_list = ", ".join(hc_parts) if hc_parts else ""
                sc_list = ", ".join(sc_parts) if sc_parts else ""
                _constraint_format_time = time.time() - _constraint_format_start

                # DIAGNOSTIC: Verify fitness matches detailed breakdown
                computed_hc = sum(hard_details.values())
                computed_sc = sum(soft_details.values())
                fitness_hc = best.fitness.values[0]
                fitness_sc = best.fitness.values[1]

                if (
                    abs(computed_hc - fitness_hc) > 0.01
                    or abs(computed_sc - fitness_sc) > 0.01
                ):
                    console.print(
                        "[bold red]WARNING: Fitness mismatch detected![/bold red]"
                    )
                    console.print(
                        f"  Fitness HC={fitness_hc:.2f} vs Computed HC={computed_hc:.2f} (diff={abs(fitness_hc-computed_hc):.2f})"
                    )
                    console.print(
                        f"  Fitness SC={fitness_sc:.2f} vs Computed SC={computed_sc:.2f} (diff={abs(fitness_sc-computed_sc):.2f})"
                    )
                    console.print(f"  Hard details: {hard_details}")
                    console.print(f"  Soft details: {soft_details}")

                # Get phase timing breakdown from profiler
                _timing_calc_start = time.time()
                phase_times = {}
                if profiler.enabled and profiler.generation_profiles:
                    last_profile = profiler.generation_profiles[-1]
                    for phase_name, phase in last_profile.phases.items():
                        phase_times[phase_name] = phase.duration

                # Calculate operation percentages
                ops_time = (
                    phase_times.get("selection", 0)
                    + phase_times.get("crossover", 0)
                    + phase_times.get("mutation", 0)
                )
                eval_time = phase_times.get("evaluation", 0)
                replacement_time = phase_times.get("replacement", 0)
                repair_time = phase_times.get("repair_memetic", 0)
                metrics_time = phase_times.get("metrics", 0)

                # New phases
                rl_time = phase_times.get("rl_ops", 0)
                igls_time = phase_times.get("igls_exhaustive", 0) + phase_times.get(
                    "igls_greedy", 0
                )
                lns_time = phase_times.get("lns_repair", 0)
                selective_repair_time = phase_times.get(
                    "selective_repair_cx", 0
                ) + phase_times.get("selective_repair_mut", 0)

                display_time = time.time() - _display_start  # Total display overhead
                other_time = (
                    gen_time
                    - ops_time
                    - eval_time
                    - replacement_time
                    - repair_time
                    - metrics_time
                    - rl_time
                    - igls_time
                    - lns_time
                    - selective_repair_time
                    - display_time  # CRITICAL: Exclude display time from "other" bucket
                )

                # Build timing breakdown string
                timing_parts = []
                if ops_time > 0:
                    timing_parts.append(f"ops={format_time(ops_time)}")
                if eval_time > 0:
                    timing_parts.append(f"eval={format_time(eval_time)}")
                if replacement_time > 0:
                    timing_parts.append(f"replace={format_time(replacement_time)}")
                if metrics_time > 0.1:
                    timing_parts.append(f"metrics={format_time(metrics_time)}")
                if repair_time > 0.01:
                    timing_parts.append(f"repair={format_time(repair_time)}")

                # New phases display
                if rl_time > 0.01:
                    timing_parts.append(f"rl={format_time(rl_time)}")
                if igls_time > 0.01:
                    timing_parts.append(f"igls={format_time(igls_time)}")
                if lns_time > 0.01:
                    timing_parts.append(f"lns={format_time(lns_time)}")
                if selective_repair_time > 0.01:
                    timing_parts.append(
                        f"sel_repair={format_time(selective_repair_time)}"
                    )

                # Show display breakdown if significant
                if display_time > 1.0:
                    timing_parts.append(
                        f"display={format_time(display_time)} "
                        f"(eval_detail={_eval_detailed_time:.1f}s)"
                    )
                if other_time > 0.1:
                    timing_parts.append(f"other={format_time(other_time)}")

                timing_str = ", ".join(timing_parts) if timing_parts else ""
                _timing_calc_time = time.time() - _timing_calc_start

                # Format exactly as requested: [!ok] gen x/y : hc = , sc = , t=4s,  hc1=, hc2=.. sc1=., sc2=...
                # Right-align generation numbers for consistent indentation
                gen_width = len(str(self.config.generations))
                console.print()  # Line break before generation output
                console.print(
                    f"[dim][!ok] gen {gen+1:>{gen_width}}/{self.config.generations} : "
                    f"hc={best.fitness.values[0]:.0f}, sc={best.fitness.values[1]:.2f}, "
                    f"t={format_time(gen_time)} ({timing_str}),  {hc_list} {sc_list}[/dim]"
                )

                # Log generation metrics
                _logging_start = time.time()
                if self.logger:
                    diversity = average_pairwise_diversity(self.population)
                    repairs = 0
                    if gen < len(self.metrics.repair_stats):
                        repairs = self.metrics.repair_stats[gen].get("total_fixes", 0)

                    notes = ""
                    if best.fitness.values[0] == 0:
                        notes = "Perfect solution"

                    self.logger.log_generation(
                        generation=gen,
                        hard_violations=best.fitness.values[0],
                        soft_penalty=best.fitness.values[1],
                        time_seconds=gen_time,
                        diversity=diversity,
                        repairs=repairs,
                        notes=notes,
                    )
                _logging_time = time.time() - _logging_start

                # Log micro-breakdown if display overhead > 0.5s
                if display_time > 0.5:
                    _unaccounted = display_time - (
                        _best_selection_time
                        + _eval_detailed_time
                        + _constraint_format_time
                        + _timing_calc_time
                        + _logging_time
                    )
                    logger.info(
                        f"Gen {gen} display breakdown: total={display_time:.2f}s "
                        f"(best_selection={_best_selection_time:.2f}s, "
                        f"eval_detailed={_eval_detailed_time:.2f}s, "
                        f"constraint_format={_constraint_format_time:.2f}s, "
                        f"timing_calc={_timing_calc_time:.2f}s, "
                        f"logging={_logging_time:.2f}s, "
                        f"unaccounted={_unaccounted:.2f}s)"
                    )

                # Early stopping if perfect solution found
                best = tools.selBest(self.population, 1)[0]
                if best.fitness.values[0] == 0:
                    gen_width = len(str(self.config.generations))
                    console.print(
                        f"\n[!ok] [bold green]Perfect solution found at generation {gen + 1:>{gen_width}}![/bold green]"
                    )

                    # Log early stop
                    if self.logger and gen < len(self.metrics.repair_stats):
                        diversity = average_pairwise_diversity(self.population)
                        repairs = self.metrics.repair_stats[gen].get("total_fixes", 0)
                        self.logger.log_generation(
                            generation=gen,
                            hard_violations=0,
                            soft_penalty=best.fitness.values[1],
                            time_seconds=0,
                            diversity=diversity,
                            repairs=repairs,
                            notes="Early stop - perfect solution",
                        )
                    break

        # Show constraint legend after evolution completes
        # Get constraint names for legend (use best individual from last generation)
        best = tools.selBest(self.population, 1)[0]
        hard_details, soft_details = evaluate_detailed(
            best,
            self.context.courses,
            self.context.instructors,
            self.context.groups,
            self.context.rooms,
        )

        console.print()
        console.print(
            "[dim]constraint mapping (individual values = raw violations, hc/sc totals = weighted sums):[/dim]"
        )

        # Show hard constraints (in configured order)
        for name in self.hard_constraint_names:
            clean_name = name.replace("_", " ")
            code = self.hard_constraint_codes.get(name, name[:4])
            console.print(f"  [dim]{code}:[/dim] {clean_name}")

        for name in hard_details:
            if name not in self.hard_constraint_codes:
                clean_name = name.replace("_", " ")
                console.print(f"  [dim]{name[:4]}:[/dim] {clean_name}")

        # Show soft constraints (in configured order)
        for name in self.soft_constraint_names:
            clean_name = name.replace("_", " ")
            code = self.soft_constraint_codes.get(name, name[:4])
            console.print(f"  [dim]{code}:[/dim] {clean_name}")

        for name in soft_details:
            if name not in self.soft_constraint_codes:
                clean_name = name.replace("_", " ")
                console.print(f"  [dim]{name[:4]}:[/dim] {clean_name}")

        console.print()

        # ENHANCEMENT: Save violation heatmap at end
        if self.violation_heatmap:
            enhancement_cfg = get_config().enhancements
            output_dir = get_config().io.output_dir
            heatmap_file = (
                Path(output_dir) / enhancement_cfg.violation_heatmap.persistence_file
            )
            self.violation_heatmap.save_to_file(str(heatmap_file))
            console.print(f"[dim]   Saved violation heatmap to {heatmap_file}[/dim]")

            # Print summary
            self.violation_heatmap.print_summary(console)

    def _evolve_generation(self, gen: int, progress=None):
        """
        Execute one generation of evolution with adaptive repair.

        Hybrid Repair Strategy:
            1. Stagnation Detection: Track best HC over rolling window
            2. Periodic Triggers: Regular repair every N generations
            3. Intensive Triggers: Heavy repair every M generations (M > N)
            4. Dynamic Parameters: Adjust repair_mode and max_iterations based on trigger type

        Trigger Priority (highest to lowest):
            - Intensive: Every intensive_interval (default 20) → full mode, max_iterations=10
            - Stagnation: Window (default 5) gens without HC improvement → full mode, max_iterations=5
            - Periodic: Every interval (default 10) → full mode, max_iterations=5

        Args:
            gen: Current generation number (0-indexed)
            progress: Optional rich.progress.Progress for UI updates
        """
        # Import EventTracker for event logging
        from src.utils.constraint_logger import EventTracker

        event_tracker = EventTracker()

        repair_config = self.config.repair_config
        igls_config = get_config().repair  # Get IGLS config early

        generation_repair_stats = {
            "instructor_availability_fixes": 0,
            "overlap_fixes": 0,
            "room_fixes": 0,
            "instructor_conflict_fixes": 0,
            "qualification_fixes": 0,
            "room_type_fixes": 0,
            "clustering_fixes": 0,
            "session_count_fixes": 0,
            "total_fixes": 0,
            # NEW: Per-individual tracking
            "individuals_repaired": 0,  # Count of individuals that had repairs
            "crossover_repairs": 0,  # Total repairs after crossover
            "mutation_repairs": 0,  # Total repairs after mutation
            "memetic_repairs": 0,  # Total repairs from memetic search
        }

        # ADAPTIVE REPAIR: Hybrid trigger logic (stagnation + periodic)
        adaptive_config = repair_config.get("adaptive_repair", {})
        stagnation_cfg = adaptive_config.get("stagnation_trigger", {})
        periodic_cfg = adaptive_config.get("periodic_trigger", {})

        # Get enhancement config once (used in multiple places)
        enhancement_cfg = get_config().enhancements

        # Track current best HC for stagnation detection
        if self.population:
            current_best_hc = min(
                ind.fitness.values[0] for ind in self.population if ind.fitness.valid
            )

            # Stagnation detection
            stagnation_detected = False
            if stagnation_cfg.get("enabled", False):
                improvement = self.last_best_hc - current_best_hc
                if improvement <= stagnation_cfg.get("threshold", 0.0):
                    self.stagnation_counter += 1
                    self.prolonged_stagnation_counter += 1  # Track for restart
                else:
                    self.stagnation_counter = 0  # Reset on improvement
                    self.prolonged_stagnation_counter = 0  # Reset on improvement
                    self.last_best_hc = current_best_hc

                if self.stagnation_counter >= stagnation_cfg.get("window", 5):
                    stagnation_detected = True
                    event_tracker.add("stagnation_detected")

                    # ENHANCEMENT: Trigger hypermutation on stagnation
                    if (
                        enhancement_cfg.master_enabled
                        and enhancement_cfg.hypermutation.enabled
                        and enhancement_cfg.hypermutation.trigger_on_stagnation
                    ):
                        # Activate hypermutation
                        self.hypermutation_active = True
                        self.hypermutation_countdown = (
                            enhancement_cfg.hypermutation.duration_generations
                        )
                        console.print(
                            f"[bold magenta] [!hurray] Gen {gen}: HYPERMUTATION activated "
                            f"(mutpb: {self.config.mutation_prob:.1f} → "
                            f"{enhancement_cfg.hypermutation.mutation_rate:.1f} "
                            f"for {self.hypermutation_countdown} gens)[/bold magenta]"
                        )
                        event_tracker.add("hypermutation_start")

                # ENHANCEMENT: Check for population restart (RISKY - last resort)
                restart_cfg = enhancement_cfg.population_restart
                if (
                    enhancement_cfg.master_enabled
                    and restart_cfg.enabled
                    and self.prolonged_stagnation_counter
                    >= restart_cfg.trigger_stagnation_gens
                ):
                    # Check minimum interval since last restart
                    gens_since_restart = gen - self.last_restart_gen
                    if gens_since_restart >= restart_cfg.min_interval_gens:
                        self._restart_population(gen)
                        event_tracker.add("population_restart")
                        self.last_best_hc = float("inf")  # Force re-evaluation

            # Periodic trigger detection
            is_periodic_gen = periodic_cfg.get("enabled", False) and (
                gen > 0 and gen % periodic_cfg.get("interval", 10) == 0
            )
            is_intensive_gen = periodic_cfg.get("enabled", False) and (
                gen > 0 and gen % periodic_cfg.get("intensive_interval", 20) == 0
            )

            # Apply dynamic repair parameters based on triggers
            if is_intensive_gen:
                # INTENSIVE REPAIR (every 20 gens): HARD/STRICT mode
                # - Full mode (not selective) for thorough checking
                # - High max_iterations (can take time)
                # - Enable memetic mode for deep local search on elite
                intensive_action = adaptive_config.get("intensive_action", {})
                repair_config["selective_mode"] = (
                    intensive_action.get("repair_mode", "full") == "selective"
                )
                repair_config["max_iterations"] = intensive_action.get(
                    "max_iterations", 10
                )
                repair_config["memetic_iterations"] = intensive_action.get(
                    "max_iterations", 10
                )
                repair_config["memetic_mode"] = True  # Enable memetic for intensive
                event_tracker.add("intensive_repair")
                console.print(
                    f"[bold red][!info] Gen {gen}: INTENSIVE REPAIR triggered (every {periodic_cfg.get('intensive_interval', 20)} gens) "
                    f"- HARD mode: full scan, max_iterations={repair_config['max_iterations']}, memetic=ON[/bold red]"
                )
            elif stagnation_detected:
                # STAGNATION REPAIR: SOFT mode
                # - Selective mode (fast, targeted)
                # - Limited iterations
                # - No memetic mode (keep it lightweight)
                trigger_action = adaptive_config.get("trigger_action", {})
                repair_config["selective_mode"] = (
                    trigger_action.get("repair_mode", "full") == "selective"
                )
                repair_config["max_iterations"] = trigger_action.get(
                    "max_iterations", 5
                )
                repair_config["memetic_iterations"] = trigger_action.get(
                    "max_iterations", 5
                )
                repair_config["memetic_mode"] = False  # Disable memetic for stagnation
                event_tracker.add("stagnation_repair")
                console.print(
                    f"[bold yellow]️ Gen {gen}: STAGNATION repair "
                    f"triggered ({self.stagnation_counter} gens) - SOFT mode: "
                    f"selective, max_iterations={repair_config['max_iterations']}, "
                    f"memetic=OFF[/bold yellow]"
                )
                self.stagnation_counter = 0  # Reset after applying repair
            elif is_periodic_gen:
                # PERIODIC REPAIR (every 10 gens): SOFT mode
                # - Selective mode (fast, targeted)
                # - Limited iterations
                # - No memetic mode
                trigger_action = adaptive_config.get("trigger_action", {})
                repair_config["selective_mode"] = (
                    trigger_action.get("repair_mode", "full") == "selective"
                )
                repair_config["max_iterations"] = trigger_action.get(
                    "max_iterations", 5
                )
                repair_config["memetic_iterations"] = trigger_action.get(
                    "max_iterations", 5
                )
                repair_config["memetic_mode"] = False  # Disable memetic for periodic
                event_tracker.add("periodic_repair")
                console.print(
                    f"[bold cyan] [!info] Gen {gen}: PERIODIC repair triggered "
                    f"(every {periodic_cfg.get('interval', 10)} gens) - "
                    f"SOFT mode: selective, "
                    f"max_iterations={repair_config['max_iterations']}, "
                    f"memetic=OFF[/bold cyan]"
                )
            else:
                # NO TRIGGER: Reset memetic mode to base config value
                # This ensures memetic doesn't carry over from previous trigger
                base_memetic = get_config().repair.memetic_mode
                repair_config["memetic_mode"] = base_memetic

        # PHASE 1.3: Get adaptive probabilities based on search progress
        profiler = get_profiler()
        profiler.start_generation(gen)

        cxpb, mutpb = self._get_adaptive_probabilities(gen)

        # DEBUG: Log probabilities to identify the issue
        if gen <= 5:  # Only log first few generations to avoid spam
            console.print(
                f"[dim]   DEBUG Gen {gen}: cxpb={cxpb:.2f}, mutpb={mutpb:.2f} (config: cx={self.config.crossover_prob:.2f}, mut={self.config.mutation_prob:.2f})[/dim]"
            )

        # ENHANCEMENT: Override mutation probability if hypermutation active
        if self.hypermutation_active:
            enhancement_cfg = get_config().enhancements
            mutpb = enhancement_cfg.hypermutation.mutation_rate
            event_tracker.add("hypermutation_active")
            self.hypermutation_countdown -= 1
            if self.hypermutation_countdown <= 0:
                self.hypermutation_active = False
                event_tracker.add("hypermutation_ended")
                console.print(
                    f"[dim]   Gen {gen}: Hypermutation ended, returning to normal mutpb[/dim]"
                )

        # MICRO-TIMING: Track unaccounted overhead between phases
        import time as time_module

        _phase_start = time_module.time()
        _selection_prep_time = 0
        _crossover_prep_time = 0
        _mutation_prep_time = 0
        _eval_prep_time = 0
        _replace_prep_time = 0

        # Selection
        _before = time_module.time()
        profiler.start_phase("selection", items_to_process=len(self.population))
        _selection_prep_time = time_module.time() - _before

        offspring = self.toolbox.select(self.population, len(self.population))
        offspring = list(map(self.toolbox.clone, offspring))
        profiler.end_phase()

        # PERFORMANCE: Parallel Crossover (8-12x faster for large populations)
        # Uses ThreadPoolExecutor to apply crossover to pairs concurrently
        _before = time_module.time()
        profiler.start_phase("crossover", items_to_process=len(offspring) // 2)
        _crossover_prep_time = time_module.time() - _before

        offspring = _parallel_crossover(offspring, cxpb, self.toolbox)
        profiler.end_phase()

        # PERFORMANCE FIX: Selective repairs after crossover disabled by default
        # Previously consumed 30-40s per generation with minimal quality improvement
        # Repair is better applied strategically during stagnation (see stagnation_repair below)
        # Can be re-enabled via igls.selective_repair.apply_after_crossover=true
        if (
            repair_config.get("enabled", False)
            and igls_config.selective_repair.enabled
            and igls_config.selective_repair.apply_after_crossover
        ):
            profiler.start_phase("selective_repair_cx")
            for i in range(0, len(offspring), 2):
                if (
                    i + 1 < len(offspring)
                    and not offspring[i].fitness.valid
                    and random.random() < igls_config.selective_repair.apply_probability
                ):
                    from src.ga.operators.intensive_local_search import (
                        apply_selective_probabilistic,
                    )

                    offspring[i], was_repaired1 = apply_selective_probabilistic(
                        individual=offspring[i],
                        context=self.context,
                        apply_probability=1.0,
                    )
                    offspring[i + 1], was_repaired2 = apply_selective_probabilistic(
                        individual=offspring[i + 1],
                        context=self.context,
                        apply_probability=1.0,
                    )

                    if was_repaired1 or was_repaired2:
                        if "crossover_repair_applied" not in list(event_tracker.events):
                            event_tracker.add("crossover_repair_applied")

                        if was_repaired1:
                            generation_repair_stats["individuals_repaired"] += 1
                            generation_repair_stats["crossover_repairs"] += 1
                        if was_repaired2:
                            generation_repair_stats["individuals_repaired"] += 1
                            generation_repair_stats["crossover_repairs"] += 1
            profiler.end_phase()

        # PERFORMANCE: Parallel Mutation (8-12x faster for large populations)
        # Uses ThreadPoolExecutor to apply mutation concurrently
        profiler.start_phase("mutation", items_to_process=len(offspring))
        offspring = _parallel_mutation(offspring, mutpb, self.toolbox)
        profiler.end_phase()

        # PERFORMANCE FIX: Selective repairs after mutation disabled by default
        # Previously consumed 30-40s per generation with minimal quality improvement
        # Natural selection already filters out poor mutations - repair not needed here
        # Can be re-enabled via igls.selective_repair.apply_after_mutation=true
        if (
            repair_config.get("enabled", False)
            and igls_config.selective_repair.enabled
            and igls_config.selective_repair.apply_after_mutation
        ):
            profiler.start_phase("selective_repair_mut")
            for mutant in offspring:
                if (
                    not mutant.fitness.valid
                    and random.random() < igls_config.selective_repair.apply_probability
                ):
                    from src.ga.operators.intensive_local_search import (
                        apply_selective_probabilistic,
                    )

                    mutant, was_repaired = apply_selective_probabilistic(
                        individual=mutant,
                        context=self.context,
                        apply_probability=1.0,  # Already gated above
                    )

                    if was_repaired:
                        if "mutation_repair_applied" not in list(event_tracker.events):
                            event_tracker.add("mutation_repair_applied")

                        generation_repair_stats["individuals_repaired"] += 1
                        generation_repair_stats["mutation_repairs"] += 1
            profiler.end_phase()

        # CRITICAL FIX: Comprehensive fitness invalidation check
        # Check multiple conditions to detect DEAP invalidation bugs
        _before_invalid = time_module.time()
        profiler.start_phase("invalidation_check")

        invalid = []
        for ind in offspring:
            # Multi-condition check (DEAP sometimes leaves fitness.valid=True after delete)
            is_invalid = (
                not hasattr(ind, "fitness")
                or not hasattr(ind.fitness, "valid")
                or not ind.fitness.valid
                or not hasattr(ind.fitness, "values")
                or len(ind.fitness.values) == 0
            )

            if is_invalid:
                invalid.append(ind)

        profiler.end_phase()
        _invalid_check_time = time_module.time() - _before_invalid

        # Calculate expected invalidation count
        expected_invalid = int(len(offspring) * (cxpb * 0.75 + mutpb * 0.25))

        # DEBUG: Log invalidation results (always show for first 5 gens)
        if gen <= 5:
            console.print(
                f"[dim]   Gen {gen}: {len(invalid)}/{len(offspring)} "
                f"individuals invalidated (expected ~{expected_invalid}, "
                f"{len(invalid)/len(offspring)*100:.1f}%)[/dim]"
            )

        # WARNING: Detect invalidation failure
        if len(invalid) < expected_invalid * 0.5 and gen > 0:
            console.print(
                f"[bold yellow]   WARNING Gen {gen}: Only "
                f"{len(invalid)}/{len(offspring)} individuals marked "
                f"invalid! Expected ~{expected_invalid}. Fitness "
                f"invalidation may be broken.[/bold yellow]"
            )

        # PARALLEL FITNESS EVALUATION (PRIMARY PARALLELIZATION TARGET)
        if invalid:
            profiler.start_phase("evaluation", items_to_process=len(invalid))

            # Use multiprocessing pool via toolbox.map (32 cores parallelized)
            fitness_values = list(self.toolbox.map(self.toolbox.evaluate, invalid))

            for ind, fit in zip(invalid, fitness_values, strict=True):
                ind.fitness.values = fit

            profiler.end_phase()

            # DEBUG: Verify fitness assignment (always show for first 5 gens)
            if gen <= 5:
                evaluated_count = sum(
                    1
                    for ind in invalid
                    if hasattr(ind, "fitness")
                    and hasattr(ind.fitness, "values")
                    and len(ind.fitness.values) > 0
                )
                console.print(
                    f"[dim]   Gen {gen}: {evaluated_count}/{len(invalid)} "
                    f"individuals evaluated successfully[/dim]"
                )
        else:
            # CRITICAL ERROR: No individuals to evaluate (fitness invalidation completely failed)
            if gen > 0:  # Skip generation 0 (initial population already evaluated)
                console.print(
                    f"[bold red]   ERROR Gen {gen}: NO individuals marked for re-evaluation! "
                    f"Fitness invalidation is BROKEN. GA is NOT evolving![/bold red]"
                )
                console.print(
                    f"[yellow]   Emergency fallback: Force re-evaluating ALL {len(offspring)} individuals...[/yellow]"
                )

                # Emergency fallback: Re-evaluate ENTIRE population
                profiler.start_phase(
                    "evaluation_emergency", items_to_process=len(offspring)
                )

                fitness_values = list(
                    self.toolbox.map(self.toolbox.evaluate, offspring)
                )

                for ind, fit in zip(offspring, fitness_values, strict=True):
                    ind.fitness.values = fit

                profiler.end_phase()

                console.print(
                    f"[green]   Emergency re-evaluation complete: {len(offspring)} individuals[/green]"
                )

        # Track overhead
        _eval_prep_time = _invalid_check_time

        # PHASE 1.2: Explicit Elitism - preserve top solutions
        _before_replace = time_module.time()
        profiler.start_phase("replacement", items_to_process=len(self.population))
        _replace_prep_time = time_module.time() - _before_replace

        # Replacement: (μ + λ) selection - combine parents + offspring only
        # Elite are already in population, no need to add separately
        combined = self.population + offspring  # 200 + 200 = 400 (vs 410 before)
        self.population[:] = self.toolbox.select(combined, len(self.population))
        profiler.end_phase()

        # Calculate overhead timing
        _total_overhead = (
            _selection_prep_time
            + _crossover_prep_time
            + _mutation_prep_time
            + _eval_prep_time
            + _replace_prep_time
        )

        # Log if overhead is significant (>100ms)
        if _total_overhead > 0.1:
            console.print(
                f"[dim]   Phase overhead: {_total_overhead:.3f}s "
                f"(sel={_selection_prep_time:.3f}s, cx={_crossover_prep_time:.3f}s, "
                f"mut={_mutation_prep_time:.3f}s, eval_check={_eval_prep_time:.3f}s, "
                f"replace={_replace_prep_time:.3f}s)[/dim]"
            )

        # RL INTEGRATION: Apply RL-selected heuristics
        if self.rl_enabled:
            profiler.start_phase("rl_ops")
            self._apply_rl_operators(gen)
            profiler.end_phase()
            event_tracker.add("rl_operators_applied")
        # ROUND-ROBIN: Apply heuristics in fixed rotation (when RL disabled)
        elif len(self.heuristic_tracker.heuristic_order) > 0:
            profiler.start_phase("roundrobin_heuristics")
            self._apply_round_robin_heuristics(gen)
            profiler.end_phase()
            event_tracker.add("roundrobin_heuristic_applied")

        # Memetic mode: Apply intensive local search to elite individuals
        if repair_config.get("enabled", False) and repair_config.get(
            "memetic_mode", False
        ):
            from src.ga.operators.repair import repair_individual_unified

            event_tracker.add("memetic_repair_applied")

            # Use selective mode from config
            selective_mode = repair_config.get("selective_mode", True)

            elite_percentage = repair_config.get("elite_percentage", 0.2)
            elite_count = max(1, int(elite_percentage * len(self.population)))
            elite_individuals = tools.selBest(self.population, elite_count)

            profiler.start_phase("repair_memetic", items_to_process=elite_count)
            for individual in elite_individuals:
                stats = repair_individual_unified(
                    individual,
                    self.context,
                    max_iterations=repair_config.get("memetic_iterations", 5),
                    selective=selective_mode,
                )

                # Track memetic repairs
                total_fixes = stats.get("total_fixes", 0)
                if total_fixes > 0:
                    generation_repair_stats["individuals_repaired"] += 1
                    generation_repair_stats["memetic_repairs"] += total_fixes

                # Invalidate fitness after repair
                del individual.fitness.values

                # Aggregate all memetic stats with proper key mapping
                self._accumulate_repair_stats(generation_repair_stats, stats)

            # Re-evaluate elite after memetic repair
            # Use toolbox.map for parallel evaluation when pool is available
            fitness_values = list(
                self.toolbox.map(self.toolbox.evaluate, elite_individuals)
            )
            for ind, fit in zip(elite_individuals, fitness_values, strict=True):
                ind.fitness.values = fit
            profiler.end_phase()

        # Finalize generation repair totals
        # Sum category fixes (after key mapping) plus phase-specific counts
        category_total = (
            generation_repair_stats["instructor_availability_fixes"]
            + generation_repair_stats["overlap_fixes"]
            + generation_repair_stats["room_fixes"]
            + generation_repair_stats["instructor_conflict_fixes"]
            + generation_repair_stats["qualification_fixes"]
            + generation_repair_stats["room_type_fixes"]
            + generation_repair_stats["clustering_fixes"]
            + generation_repair_stats["session_count_fixes"]
        )
        phase_total = (
            generation_repair_stats["crossover_repairs"]
            + generation_repair_stats["mutation_repairs"]
            + generation_repair_stats["memetic_repairs"]
        )
        generation_repair_stats["total_fixes"] = max(category_total, phase_total)
        # ============
        # HEURISTIC TOOLBOX ARCHITECTURE (Nov 2025)
        # ============
        # ALL repair/improvement operations are now unified heuristics:
        #   - igls_repair, lns_repair, selective_repair, exhaustive_search
        #   - Applied via round-robin rotation OR RL-guided selection
        #   - No hardcoded generation triggers - mode-specific config
        #   - Managed through heuristics.repair.* in configs
        #
        # Legacy hardcoded triggers REMOVED:
        #    Exhaustive search at gens [3, 25] - use heuristic
        #    LNS periodic trigger every 50 gens - use heuristic
        #    Stagnation-triggered repairs - migrate to heuristics (future)
        #
        # Migration: Enable via configs/heuristics/repair/*.enabled=true
        # ============

        # Store generation repair stats
        self.metrics.repair_stats.append(generation_repair_stats)

        # Track metrics (also logs to constraint logger)
        profiler.start_phase("metrics", items_to_process=len(self.population))
        # Pass best individual to avoid re-evaluation in display code
        best = tools.selBest(self.population, 1)[0]
        self._track_metrics(gen, event_tracker, best_individual=best)
        profiler.end_phase()

        # End profiler generation and display breakdown
        profiler.end_generation()

    def _get_adaptive_probabilities(self, gen: int) -> tuple[float, float]:
        """
        Adjust crossover and mutation probabilities based on search phase.

        PHASE 1.3: Adaptive Operator Probabilities

        Strategy:
        - Early (0-30%): High exploration (more mutation, less crossover)
        - Mid (30-70%): Balanced (use config defaults)
        - Late (70-100%): High exploitation (more crossover, less mutation)

        Args:
            gen: Current generation number

        Returns:
            Tuple of (crossover_prob, mutation_prob)
        """
        progress = gen / self.config.generations

        if progress < 0.3:
            # Early phase: explore aggressively
            crossover_prob = 0.7
            mutation_prob = 0.4
        elif progress < 0.7:
            # Mid phase: balanced (use config defaults)
            # Use explicit values from our config instead of potentially unset config object values
            config = get_config()
            crossover_prob = config.ga.cxpb  # Use explicit ga.cxpb
            mutation_prob = config.ga.mutpb  # Use explicit ga.mutpb
        else:
            # Late phase: exploit (refine good solutions)
            crossover_prob = 0.9
            mutation_prob = 0.2

        return crossover_prob, mutation_prob

    def _track_metrics(self, gen: int, event_tracker=None, best_individual=None):
        """
        Record metrics for current generation.
        OPTIMIZED: Skip expensive metrics on non-tracked generations.
        CRITICAL FIX: Skip ALL metrics for initial population (gen=-1) to avoid 2-min startup delay.

        Args:
            gen: Generation number (-1 for initial population, 0+ for evolved generations)
            event_tracker: Optional EventTracker with events from this generation
            best_individual: Optional pre-selected best individual to cache detailed evaluation
        """
        # PERFORMANCE FIX: Skip initial population metrics entirely (gen=-1)
        # These metrics are not useful for analysis and cause 2-min startup delay
        if gen == -1:
            return

        # Import new metrics modules
        from src.metrics.convergence import calculate_constraint_satisfaction_rate
        from src.metrics.hypervolume import (
            calculate_hypervolume,
            get_hypervolume_reference_point,
        )
        from src.metrics.pareto_metrics import (
            calculate_inverted_generational_distance,
            calculate_spacing,
            calculate_spread,
            get_pareto_front_size,
        )

        # Determine if this is a tracked generation for expensive metrics
        metrics_config = get_config().metrics
        advanced_freq = metrics_config.advanced_metrics_frequency
        # Always track: gen 0, last gen, or every Nth generation
        is_tracked_gen = (
            gen == 0 or gen == self.config.generations - 1 or gen % advanced_freq == 0
        )

        # ALWAYS calculate basic metrics (fast, essential for progress tracking)
        # Optimized with numpy for 2-10x speedup on large populations

        # Validate all individuals have valid fitness before numpy conversion
        for ind in self.population:
            if (
                not hasattr(ind.fitness, "values")
                or not ind.fitness.valid
                or len(ind.fitness.values) != 2
            ):
                # Re-evaluate invalid fitness (use direct evaluate function)
                fitness = evaluate(
                    ind,
                    self.context.courses,
                    self.context.instructors,
                    self.context.groups,
                    self.context.rooms,
                )
                ind.fitness.values = fitness

        fitness_array = np.array([ind.fitness.values for ind in self.population])
        self.metrics.hard_violations.append(float(fitness_array[:, 0].min()))
        self.metrics.soft_penalties.append(float(fitness_array[:, 1].min()))
        diversity = average_pairwise_diversity(self.population)
        self.metrics.diversity.append(diversity)

        # Initialize variables that will be used in constraint logging
        hv = 0.0
        spacing = 0.0
        spread = 0.0

        # CONDITIONAL: Calculate expensive multi-objective metrics
        if is_tracked_gen:
            # Phase 1: Essential multi-objective metrics
            # Calculate hypervolume (use consistent reference point)
            if gen == 0:
                # First generation: establish reference point
                self._hypervolume_ref_point = get_hypervolume_reference_point(
                    self.population, margin=0.1
                )

            hv = calculate_hypervolume(self.population, self._hypervolume_ref_point)
            self.metrics.hypervolume.append(hv)

            # Calculate spacing (Pareto front uniformity)
            spacing = calculate_spacing(self.population)
            self.metrics.spacing.append(spacing)

            # Count Pareto front size
            pf_size = get_pareto_front_size(self.population)
            self.metrics.pareto_front_size.append(pf_size)

            # Calculate feasibility rate
            feas_rate = calculate_constraint_satisfaction_rate(self.population)
            self.metrics.feasibility_rate.append(feas_rate)

            # Phase 2: Advanced metrics (IGD, Spread)
            # IGD requires reference front - use initial population as reference
            if gen == 0:
                # Store initial Pareto front as reference
                pareto_front = tools.sortNondominated(
                    self.population, len(self.population), first_front_only=True
                )[0]
                self.metrics.reference_front = list(pareto_front)

            # Calculate IGD if reference front exists
            if self.metrics.reference_front:
                igd = calculate_inverted_generational_distance(
                    self.population, self.metrics.reference_front
                )
                self.metrics.igd.append(igd)
            else:
                self.metrics.igd.append(0.0)

            # Calculate spread
            spread = calculate_spread(self.population)
            self.metrics.spread.append(spread)
        else:
            # Skip expensive metrics, use placeholder values (reuse last known value)
            if self.metrics.hypervolume:
                hv = self.metrics.hypervolume[-1]
                self.metrics.hypervolume.append(hv)
            else:
                hv = 0.0
                self.metrics.hypervolume.append(0.0)

            if self.metrics.spacing:
                spacing = self.metrics.spacing[-1]
                self.metrics.spacing.append(spacing)
            else:
                spacing = 0.0
                self.metrics.spacing.append(0.0)

            if self.metrics.pareto_front_size:
                self.metrics.pareto_front_size.append(
                    self.metrics.pareto_front_size[-1]
                )
            else:
                self.metrics.pareto_front_size.append(0)

            if self.metrics.feasibility_rate:
                self.metrics.feasibility_rate.append(self.metrics.feasibility_rate[-1])
            else:
                self.metrics.feasibility_rate.append(0.0)

            if self.metrics.igd:
                self.metrics.igd.append(self.metrics.igd[-1])
            else:
                self.metrics.igd.append(0.0)

            if self.metrics.spread:
                spread = self.metrics.spread[-1]
                self.metrics.spread.append(spread)
            else:
                spread = 0.0
                self.metrics.spread.append(0.0)

        # Detailed constraint breakdown
        # PERFORMANCE: Use cached best individual if provided to avoid re-evaluation
        if best_individual is None:
            best = tools.selBest(self.population, 1)[0]
        else:
            best = best_individual

        hard_details, soft_details = evaluate_detailed(
            best,
            self.context.courses,
            self.context.instructors,
            self.context.groups,
            self.context.rooms,
        )

        # Cache the detailed breakdown for display (avoid re-evaluation)
        self._cached_hard_details = hard_details
        self._cached_soft_details = soft_details

        # Update all-time best individual
        if (
            self.all_time_best is None
            or best.fitness.values[0] < self.all_time_best.fitness.values[0]
        ):
            # Better hard constraint score
            self.all_time_best = self.toolbox.clone(best)
        elif (
            best.fitness.values[0] == self.all_time_best.fitness.values[0]
            and best.fitness.values[1] < self.all_time_best.fitness.values[1]
        ):
            # Same hard constraint score, better soft constraint
            self.all_time_best = self.toolbox.clone(best)

        for name in self.hard_constraint_names:
            self.metrics.detailed_hard[name].append(hard_details[name])

        for name in self.soft_constraint_names:
            self.metrics.detailed_soft[name].append(soft_details[name])

        # ENHANCEMENT: Record violations to heatmap
        if self.violation_heatmap and gen >= 0:  # Skip initial pop
            from src.metrics.violation_recorder import record_violations_to_heatmap

            record_violations_to_heatmap(best, self.context, self.violation_heatmap)
            self.violation_heatmap.record_generation(gen)

        # Log to constraint logger if available
        if self.constraint_logger:
            # Get repair stats for this generation
            repair_stats = {}
            if gen >= 0 and gen < len(self.metrics.repair_stats):
                repair_stats = self.metrics.repair_stats[gen]
            elif gen == -1:  # Initial population
                repair_stats = {}

            # Get events from event tracker
            events = []
            if event_tracker and event_tracker.has_events():
                events = event_tracker.get_events()

            # Determine notes
            notes = ""
            if best.fitness.values[0] == 0:
                notes = "Perfect solution"
                if event_tracker and "perfect_solution" not in events:
                    event_tracker.add("perfect_solution")
                    events = event_tracker.get_events()  # Refresh events list

            # Log to constraint CSV (crash-safe - flushes immediately)
            self.constraint_logger.log_generation(
                generation=gen,
                hard_total=best.fitness.values[0],
                soft_total=best.fitness.values[1],
                hard_breakdown=hard_details,
                soft_breakdown=soft_details,
                diversity=diversity,
                time_seconds=0.0,  # Will be updated by evolve() loop
                hypervolume=hv,
                spacing=spacing,
                igd=self.metrics.igd[-1] if self.metrics.igd else 0.0,
                spread=spread,
                repair_stats=repair_stats,
                events=events,
                notes=notes,
            )

        # Detailed periodic logging disabled (user prefers compact gen-by-gen format)
        # User requested removal of verbose GEN X Hard=... breakdown
        # All info now shown in compact format: [!ok] gen x/y : hc=, sc=, t=Xs, hc1=, hc2=...
        # if gen >= 0 and (
        #     gen == 0 or (gen + 1) % 4 == 0 or gen == self.config.generations - 1
        # ):
        #     self._log_generation_details(gen, best, hard_details, soft_details)

    def _log_generation_details(
        self, gen: int, best, hard_details: dict, soft_details: dict
    ):
        """Print detailed constraint breakdown."""
        console.print(
            f"\n[cyan]GEN {gen+1}[/cyan] Hard=[yellow]{best.fitness.values[0]:.0f}[/yellow], "
            f"Soft=[blue]{best.fitness.values[1]:.2f}[/blue]"
        )

        # Display repair statistics if enabled
        if self.config.repair_config.get("enabled", False) and gen < len(
            self.metrics.repair_stats
        ):
            repair_stats = self.metrics.repair_stats[gen]
            if repair_stats["total_fixes"] > 0:
                # Build repair summary with all non-zero categories
                repair_parts = []
                if repair_stats.get("instructor_availability_fixes", 0) > 0:
                    repair_parts.append(
                        f"instr_avail:{repair_stats['instructor_availability_fixes']}"
                    )
                if repair_stats.get("overlap_fixes", 0) > 0:
                    repair_parts.append(f"group:{repair_stats['overlap_fixes']}")
                if repair_stats.get("room_fixes", 0) > 0:
                    repair_parts.append(f"room:{repair_stats['room_fixes']}")
                if repair_stats.get("instructor_conflict_fixes", 0) > 0:
                    repair_parts.append(
                        f"instr:{repair_stats['instructor_conflict_fixes']}"
                    )
                if repair_stats.get("qualification_fixes", 0) > 0:
                    repair_parts.append(f"qual:{repair_stats['qualification_fixes']}")
                if repair_stats.get("room_type_fixes", 0) > 0:
                    repair_parts.append(f"type:{repair_stats['room_type_fixes']}")
                if repair_stats.get("clustering_fixes", 0) > 0:
                    repair_parts.append(f"cluster:{repair_stats['clustering_fixes']}")
                if repair_stats.get("session_count_fixes", 0) > 0:
                    repair_parts.append(f"count:{repair_stats['session_count_fixes']}")

                repair_summary = ", ".join(repair_parts) if repair_parts else "misc"
                console.print(
                    f"   [green]Repairs: {repair_stats['total_fixes']} fixes[/green] ({repair_summary})"
                )

        if best.fitness.values[0] > 0:
            console.print(
                f"   [yellow]HARD Total: {best.fitness.values[0]:.0f}[/yellow]"
            )
            for name, value in hard_details.items():
                if value > 0:
                    console.print(f"      • {name}: {value}")

        if best.fitness.values[1] > 0:
            console.print(f"   [blue]SOFT Total: {best.fitness.values[1]:.2f}[/blue]")
            for name, value in soft_details.items():
                if value > 0:
                    console.print(f"      • {name}: {value:.2f}")

    def _accumulate_repair_stats(self, agg: dict, stats: dict) -> None:
        """
        Accumulate repair stats from a single repair call into generation totals.

        Maps detailed repair keys from repair.py to the consolidated keys used by
        the scheduler and loggers.

        Example mappings:
        - group_overlaps_fixes -> overlap_fixes
        - room_conflicts_fixes -> room_fixes
        - instructor_conflicts_fixes -> instructor_conflict_fixes
        - instructor_qualifications_fixes -> qualification_fixes
        - room_type_mismatches_fixes -> room_type_fixes
        - session_clustering_fixes -> clustering_fixes
        - incomplete_or_extra_sessions_fixes -> session_count_fixes
        - instructor_availability_fixes -> instructor_availability_fixes
        """
        key_map = {
            "group_overlaps_fixes": "overlap_fixes",
            "room_conflicts_fixes": "room_fixes",
            "instructor_conflicts_fixes": "instructor_conflict_fixes",
            "instructor_qualifications_fixes": "qualification_fixes",
            "room_type_mismatches_fixes": "room_type_fixes",
            "session_clustering_fixes": "clustering_fixes",
            "incomplete_or_extra_sessions_fixes": "session_count_fixes",
            "instructor_availability_fixes": "instructor_availability_fixes",
        }

        normalized = {}
        for key, value in stats.items():
            if not key.endswith("_fixes"):
                continue
            normalized[key] = value
            if key.startswith("repair_"):
                normalized[key[len("repair_") :]] = value

        for src_key, dst_key in key_map.items():
            if src_key in normalized:
                agg[dst_key] += normalized.get(src_key, 0)

    def _restart_population(self, gen: int):
        """
        Population restart: Replace worst individuals with new random ones.

        RISKY OPERATION: Destroys genetic information but reintroduces diversity.
        Should only trigger as last resort after prolonged stagnation.

        Strategy:

        1. Sort population by fitness (worst first)
        2. Keep best X% (elite preservation)
        3. Generate new X% with hybrid strategy
        4. Re-evaluate new individuals

        Args:
            gen: Current generation number (for logging)
        """
        enhancement_cfg = get_config().enhancements
        restart_cfg = enhancement_cfg.population_restart

        if not restart_cfg.enabled:
            return

        # Calculate how many to replace
        restart_count = int(len(self.population) * restart_cfg.restart_percentage)
        elite_count = len(self.population) - restart_count

        console.print(
            f"\n[bold red] [!info] Gen {gen}: POPULATION RESTART triggered![/bold red]"
        )
        console.print(
            f"   [dim]Replacing worst {restart_count}/{len(self.population)} individuals "
            f"({restart_cfg.restart_percentage*100:.0f}%)[/dim]"
        )

        # Sort by fitness (best first for NSGA-II multi-objective)
        # Use lexicographic sort: hard constraints first, then soft
        sorted_pop = sorted(
            self.population,
            key=lambda ind: (ind.fitness.values[0], ind.fitness.values[1]),
        )

        # Keep elite (best individuals)
        elite = sorted_pop[:elite_count]

        # Generate new individuals using hybrid strategy
        new_individuals = self.toolbox.population(n=restart_count)

        # Evaluate new individuals
        console.print(f"   [cyan]Evaluating {restart_count} new individuals...[/cyan]")
        fitness_values = list(self.toolbox.map(self.toolbox.evaluate, new_individuals))
        for ind, fit in zip(new_individuals, fitness_values, strict=True):
            ind.fitness.values = fit

        # Replace population
        self.population[:] = elite + new_individuals

        # Calculate diversity improvement
        from src.metrics.diversity import average_pairwise_diversity

        new_diversity = average_pairwise_diversity(self.population)

        console.print(
            f"   [green][!ok] Restart complete! New diversity: "
            f"{new_diversity:.4f}[/green]"
        )

        # Update tracking
        self.last_restart_gen = gen
        self.prolonged_stagnation_counter = 0  # Reset counter

    def get_best_solution(self):
        """
        Select best solution from all generations (hall of fame).

        Returns the all-time best individual seen during evolution.
        Prefers feasible solutions (hard constraints satisfied) with
        lowest soft constraint penalty. If no feasible solution exists,
        returns the solution with fewest hard constraint violations.

        Returns:
            Best individual from all generations
        """
        # Return all-time best if available (tracked during evolution)
        if self.all_time_best is not None:
            return self.all_time_best

        # Fallback: Search current population (shouldn't reach here in normal execution)
        pareto_front = tools.sortNondominated(
            self.population, len(self.population), first_front_only=True
        )[0]

        # Prefer feasible solutions (no hard constraint violations)
        feasible = [ind for ind in pareto_front if ind.fitness.values[0] == 0]

        if feasible:
            # Among feasible solutions, select one with minimum soft penalty
            return min(feasible, key=lambda ind: ind.fitness.values[1])
        else:
            # No feasible solution, return best from Pareto front
            return pareto_front[0]

    def _validate_population_structure(self):
        """
        Validate gene alignment across population.

        Ensures all individuals have matching (course, group) pairs at each
        gene position. This is critical for position-independent crossover.

        Raises:
            ValueError: If population structure is invalid
        """
        if not self.population:
            return

        reference = [
            (gene.course_id, gene.course_type, tuple(sorted(gene.group_ids)))
            for gene in self.population[0]
        ]
        reference_set = set(reference)

        for idx, individual in enumerate(self.population[1:], start=1):
            current = [
                (gene.course_id, gene.course_type, tuple(sorted(gene.group_ids)))
                for gene in individual
            ]
            current_set = set(current)

            if current_set != reference_set:
                missing = reference_set - current_set
                extra = current_set - reference_set
                raise ValueError(
                    f"[X] Gene alignment validation FAILED!\n"
                    f"   Individual {idx} has different structure than Individual 0.\n"
                    f"   Missing pairs: {missing}\n"
                    f"   Extra pairs: {extra}\n"
                    f"   This indicates a bug in population generation."
                )

            # NOTE: Duplicates are now ALLOWED for theory courses split into multiple sessions
            # E.g., ENME 152 theory may have 3 genes (one per 2-hour session)
            # Only validate that both individuals have the same TOTAL structure
            pass
