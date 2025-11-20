"""
GA Scheduler Module

Encapsulates NSGA-II genetic algorithm execution for course scheduling.
Extracted from monolithic main.py for better testability and separation of concerns.
"""

from typing import List, Dict, Optional
import logging
from dataclasses import dataclass, field
from pathlib import Path
from deap import base, tools
import random
import time
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    ProgressColumn,
    Task,
)
from rich.table import Table
from rich.live import Live
from rich.text import Text

# from concurrent.futures import ThreadPoolExecutor  # Removed: GIL limits CPU parallelism

from src.ga.population import generate_course_group_aware_population
from src.ga.operators.crossover import crossover_course_group_aware
from src.ga.operators.mutation import mutate_individual
from src.ga.evaluator.fitness import evaluate
from src.config import get_config
from src.ga.evaluator.detailed_fitness import evaluate_detailed
from src.ga.evaluator.gpu_batch_evaluator import GPUConstraintEvaluator
from src.metrics.diversity import average_pairwise_diversity
from src.core.types import SchedulingContext
from src.utils.console_service import get_console
from src.heuristics.parallel_executor import get_parallel_executor
from src.utils.parallel_worker import get_worker_context
from src.utils.performance_profiler import get_profiler

console = get_console()
logger = logging.getLogger(__name__)


def _worker_evaluate(individual):
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


# ============================================================================
# Genetic Operators (Sequential to avoid GIL thrashing)
# ============================================================================


def _parallel_crossover(offspring, cxpb, toolbox, max_workers=None):
    """
    Apply crossover sequentially.

    NOTE: ThreadPoolExecutor removed because Python's GIL prevents true parallelism
    for CPU-bound tasks like crossover. Multiprocessing overhead (pickling)
    often outweighs benefits for simple operators. Sequential is faster and safer.
    """
    # Iterate in steps of 2: (0,1), (2,3), etc.
    for i in range(0, len(offspring) - 1, 2):
        if random.random() < cxpb:
            toolbox.mate(offspring[i], offspring[i + 1])
            del offspring[i].fitness.values
            del offspring[i + 1].fitness.values

    return offspring


def _parallel_mutation(offspring, mutpb, toolbox, max_workers=None):
    """
    Apply mutation sequentially.

    NOTE: ThreadPoolExecutor removed because Python's GIL prevents true parallelism
    for CPU-bound tasks. Sequential execution avoids context switching overhead.
    """
    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    return offspring


class AlwaysShowTimeRemainingColumn(ProgressColumn):
    """
    Custom TimeRemainingColumn with:
    - Always shows an estimate (never blank)
    - Updates only once per second (reduces flicker)
    - Smooths estimates using exponential moving average (reduces wild fluctuations)
    """

    def __init__(self):
        super().__init__()
        self._last_update_time = 0.0
        self._cached_text = Text("~calculating~", style="dim progress.remaining")
        self._ema_remaining = None  # Exponential moving average for smoothing
        self._alpha = 0.3  # Smoothing factor (0.3 = 30% new, 70% old)

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
        repair_config: Repair heuristics configuration dict (from ga_params.get_config().repair)
                       Includes selective_mode, adaptive_repair settings, and enabled heuristics
    """

    pop_size: int
    generations: int
    crossover_prob: float
    mutation_prob: float
    repair_config: Dict = field(default_factory=dict)


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

    hard_violations: List[float] = field(default_factory=list)
    soft_penalties: List[float] = field(default_factory=list)
    diversity: List[float] = field(default_factory=list)
    detailed_hard: Dict[str, List[float]] = field(default_factory=dict)
    detailed_soft: Dict[str, List[float]] = field(default_factory=dict)
    repair_stats: List[Dict] = field(default_factory=list)

    # Phase 1: Essential metrics
    hypervolume: List[float] = field(default_factory=list)
    spacing: List[float] = field(default_factory=list)
    feasibility_rate: List[float] = field(default_factory=list)
    pareto_front_size: List[int] = field(default_factory=list)

    # Phase 2: Advanced metrics
    igd: List[float] = field(default_factory=list)
    spread: List[float] = field(default_factory=list)

    # Reference front for IGD calculation (set once, used throughout)
    reference_front: List = field(default_factory=list)


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
        hard_constraint_names: List[str],
        soft_constraint_names: List[str],
        pool=None,  # NEW: Optional multiprocessing Pool
        logger=None,  # NEW: Optional GALogger for runtime logging
        constraint_logger=None,  # NEW: Optional ConstraintLogger for detailed constraint logging
        seed: Optional[int] = None,  # NEW: Random seed for worker initialization
    ):
        """
        Initialize GA scheduler with adaptive repair tracking.

        Args:
            config: GA configuration (includes repair_config with adaptive_repair settings)
            context: Scheduling context with courses, groups, instructors, rooms
            hard_constraint_names: Names of enabled hard constraints
            soft_constraint_names: Names of enabled soft constraints
            pool: Optional multiprocessing.Pool for parallel fitness evaluation
            logger: Optional GALogger for file-based logging (writes to logger.txt, not console)
            constraint_logger: Optional ConstraintLogger for detailed constraint logging (writes to logger_constraints.csv)
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

        self.toolbox = None
        self.population = None
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

        # ENHANCEMENT: Violation heatmap for targeted repair
        self.violation_heatmap = None
        enhancement_cfg = get_config().enhancements
        if enhancement_cfg.master_enabled and enhancement_cfg.violation_heatmap.enabled:
            from src.metrics.violation_heatmap import ViolationHeatmap

            self.violation_heatmap = ViolationHeatmap()
            console.print("[dim]   Violation heatmap tracking: ENABLED[/dim]")

        # GPU Batch Evaluator for 10-50x speedup
        try:
            self.gpu_evaluator = GPUConstraintEvaluator(
                device="auto", auto_tune_batch_size=True
            )
            if self.gpu_evaluator.enabled:
                console.print(
                    "[green]\u2713 GPU acceleration enabled for fitness evaluation (10-50x speedup)[/green]"
                )
        except Exception as e:
            logger.warning(f"GPU evaluator initialization failed: {e}")
            self.gpu_evaluator = None

        # NEW: Hypervolume reference point (initialized during first metric tracking)
        self._hypervolume_ref_point = None

        # RL INTEGRATION: Components for hyper-heuristic control
        self.rl_enabled = False
        self.rl_controller = None
        self.rl_state_encoder = None
        self.rl_action_mapper = None

        # PERFORMANCE: Parallel heuristic executor (10-16x speedup)
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
        self.toolbox.register("select", tools.selNSGA2)

        # PHASE 3: Hybrid population initialization support
        if get_config().ga.population_strategy == "hybrid":
            from src.ga.hybrid_population import generate_hybrid_population

            self.toolbox.register(
                "population", generate_hybrid_population, context=self.context
            )
        elif get_config().ga.population_strategy == "smart":
            # Original constraint-aware (Phase 1+2 default)
            self.toolbox.register(
                "population",
                generate_course_group_aware_population,
                context=self.context,
            )
        else:  # "random" or any other value defaults to smart
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
            guided=get_config().ga.use_constraint_guided_mutation,  # Enable constraint-guided mutation
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
                f"[yellow]RL mode '{rl_config.mode}' not compatible with GA integration[/yellow]"
            )
            console.print(
                "[dim]   Use mode 'inference' or 'hybrid' for production runs[/dim]"
            )
            return False

        try:
            # Import RL components (lazy import to avoid dependency issues)
            from src.rl.gym_env.state_encoder import StateEncoder
            from src.rl.gym_env.action_space import ActionMapper
            from src.rl.hybrid.hybrid_controller import HybridController
            from src.rl.deployment.model_loader import ModelLoader
            from src.rl.deployment.inference import RLInference

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
                f"   [green][!ok][/green] ActionMapper initialized ({self.rl_action_mapper.n_actions} actions)"
            )

            # Load trained model
            model_path = rl_config.agent.model_path
            if not model_path or model_path == "models/rl_agents/best_model.zip":
                # Try to find best model from manifest
                try:
                    from src.rl.training.checkpoints import CheckpointManager

                    manifest_path = rl_config.training.checkpoint_settings.manifest_path
                    manager = CheckpointManager(manifest_path)
                    best_checkpoint = manager.get_best_checkpoint(metric="mean_reward")
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
            model, metadata = loader.load_model(
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
            self.rl_controller = HybridController(
                inference_engine=inference_engine,
                action_mapper=self.rl_action_mapper,
                mode=rl_config.hybrid.mode,
                fallback_strategy=rl_config.hybrid.fallback_strategy,
                rl_probability=rl_config.hybrid.rl_probability,
            )
            console.print(
                f"   [green][!ok][/green] HybridController initialized (mode: {rl_config.hybrid.mode})"
            )

            self.rl_enabled = True
            console.print("[green]RL Integration: ENABLED[/green]")
            return True

        except ImportError as e:
            console.print(f"[yellow]RL components not available: {e}[/yellow]")
            console.print(
                "[dim]   Install RL dependencies: uv add gymnasium stable-baselines3[/dim]"
            )
            return False
        except FileNotFoundError as e:
            console.print(f"[yellow]RL model not found: {e}[/yellow]")
            console.print(
                f"[dim]   Train model first: python src/rl/training/train_script.py[/dim]"
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
        state = self.rl_state_encoder.encode(
            population=self.population,
            current_generation=gen,
            generations_without_improvement=self.stagnation_counter,
        )

        # Get valid actions for current state
        valid_actions = self.rl_action_mapper.enabled_actions

        # Select action using RL controller (with fallback)
        action_id = self.rl_controller.select_action(
            state=state,
            valid_actions=valid_actions,
            deterministic=True,  # Use deterministic policy for production
        )

        # Record heuristic application for state tracking
        self.rl_state_encoder.record_heuristic_application(action_id)

        # Apply selected heuristic to population
        try:
            # Get action info to determine heuristic type
            action_info = self.rl_action_mapper.get_action_info(action_id)

            # Apply heuristic in parallel to top N individuals for 10-16x speedup
            # (improvement heuristics benefit from parallel application)
            if action_info and action_info.category == "improvement":
                # Select top 4-8 individuals based on population size
                num_targets = min(8, max(4, len(self.population) // 25))
                top_individuals = tools.selBest(self.population, num_targets)

                logger.debug(
                    f"Gen {gen}: RL applying '{action_info.name}' to {num_targets} individuals in parallel"
                )

                # Get heuristic function
                heuristic_func = action_info.function
                if heuristic_func:
                    # Apply in parallel using parallel executor
                    parallel_executor = get_parallel_executor()
                    modified_individuals = parallel_executor.apply_parallel(
                        heuristic_func=heuristic_func,
                        individuals=top_individuals,
                        context=self.context,
                    )
                else:
                    modified_individuals = []
            else:
                # For non-improvement heuristics, use single best individual
                best_ind = tools.selBest(self.population, 1)[0]

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
                for ind, fit in zip(modified_individuals, fitness_values):
                    ind.fitness.values = fit

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

        for ind, fit in zip(self.population, fitness_values):
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

        # Track initial population as Generation 0
        self._track_metrics(gen=-1)  # Will be recorded as generation 0

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
            if hasattr(self.rl_controller, "inference_engine"):
                if hasattr(self.rl_controller.inference_engine, "clear_cache"):
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
            Progress(TextColumn(""), console=console, refresh_per_second=10)
        )  # spacing
        legend_bars.append(
            Progress(
                TextColumn("[dim]constraint mapping:[/dim]"),
                console=console,
                refresh_per_second=10,
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

        # Build hard constraint labels (3 per row)
        hard_items = []
        hc_counter = 1
        for name in hard_details.keys():
            clean_name = name.replace("_", " ")
            hard_items.append(f"hc{hc_counter}={clean_name}")
            hc_counter += 1

        # Display hard constraints 3 per row
        for i in range(0, len(hard_items), 3):
            row_items = hard_items[i : i + 3]
            row_text = "  [dim]" + " | ".join(row_items) + "[/dim]"
            legend_bars.append(
                Progress(
                    TextColumn(row_text),
                    console=console,
                    refresh_per_second=10,
                )
            )

        # Build soft constraint labels (3 per row)
        soft_items = []
        sc_counter = 1
        for name in soft_details.keys():
            clean_name = name.replace("_", " ")
            soft_items.append(f"sc{sc_counter}={clean_name}")
            sc_counter += 1

        # Display soft constraints 3 per row
        for i in range(0, len(soft_items), 3):
            row_items = soft_items[i : i + 3]
            row_text = "  [dim]" + " | ".join(row_items) + "[/dim]"
            legend_bars.append(
                Progress(
                    TextColumn(row_text),
                    console=console,
                    refresh_per_second=10,
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

        with Live(progress_table, console=console, refresh_per_second=10):
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
                best = tools.selBest(self.population, 1)[0]

                # Get detailed constraint breakdown
                hard_details, soft_details = evaluate_detailed(
                    best,
                    self.context.courses,
                    self.context.instructors,
                    self.context.groups,
                    self.context.rooms,
                )

                # Build compact constraint lists with short names
                hc_parts = []
                hc_counter = 1
                for name, val in hard_details.items():
                    if val > 0:
                        hc_parts.append(f"hc{hc_counter}={int(val)}")
                        hc_counter += 1

                sc_parts = []
                sc_counter = 1
                for name, val in soft_details.items():
                    if val > 0:
                        sc_parts.append(f"sc{sc_counter}={val:.1f}")
                        sc_counter += 1

                # Build constraint list strings
                hc_list = ", ".join(hc_parts) if hc_parts else ""
                sc_list = ", ".join(sc_parts) if sc_parts else ""

                # Get phase timing breakdown from profiler
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
                repair_time = gen_time - ops_time - eval_time  # Remaining time

                # Build timing breakdown string
                timing_parts = []
                if ops_time > 0:
                    timing_parts.append(f"ops={ops_time:.2f}s")
                if eval_time > 0:
                    eval_device = (
                        "GPU"
                        if (
                            self.gpu_evaluator
                            and self.gpu_evaluator.enabled
                            and len(invalid) >= 50
                        )
                        else "CPU"
                    )
                    timing_parts.append(f"eval={eval_time:.2f}s({eval_device})")
                if repair_time > 0.01:
                    timing_parts.append(f"repair={repair_time:.2f}s")

                timing_str = ", ".join(timing_parts) if timing_parts else ""

                # Format exactly as requested: [!ok] gen x/y : hc = , sc = , t=4s,  hc1=, hc2=.. sc1=., sc2=...
                console.print(
                    f"[dim][!ok] gen {gen+1}/{self.config.generations} : "
                    f"hc={best.fitness.values[0]:.0f}, sc={best.fitness.values[1]:.2f}, "
                    f"t={gen_time:.1f}s ({timing_str}),  {hc_list} {sc_list}[/dim]"
                )

                # Log generation metrics
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

                # Early stopping if perfect solution found
                best = tools.selBest(self.population, 1)[0]
                if best.fitness.values[0] == 0:
                    console.print(
                        f"\n[!ok] [bold green]Perfect solution found at generation {gen + 1}![/bold green]"
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
        console.print("[dim]constraint mapping:[/dim]")

        # Show hard constraints
        hc_counter = 1
        for name, val in hard_details.items():
            clean_name = name.replace("_", " ")
            console.print(f"  [dim]hc{hc_counter}:[/dim] {clean_name}")
            hc_counter += 1

        # Show soft constraints
        sc_counter = 1
        for name, val in soft_details.items():
            clean_name = name.replace("_", " ")
            console.print(f"  [dim]sc{sc_counter}:[/dim] {clean_name}")
            sc_counter += 1

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
                    f"[bold yellow]⚠️ Gen {gen}: STAGNATION repair triggered ({self.stagnation_counter} gens) "
                    f"- SOFT mode: selective, max_iterations={repair_config['max_iterations']}, memetic=OFF[/bold yellow]"
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
                    f"[bold cyan] [!info] Gen {gen}: PERIODIC repair triggered (every {periodic_cfg.get('interval', 10)} gens) "
                    f"- SOFT mode: selective, max_iterations={repair_config['max_iterations']}, memetic=OFF[/bold cyan]"
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

        # ENHANCEMENT: Override mutation probability if hypermutation is active
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

        # Selection
        profiler.start_phase("selection", items_to_process=len(self.population))
        offspring = self.toolbox.select(self.population, len(self.population))
        offspring = list(map(self.toolbox.clone, offspring))
        profiler.end_phase()

        # PERFORMANCE: Parallel Crossover (8-12x faster for large populations)
        # Uses ThreadPoolExecutor to apply crossover to pairs concurrently
        profiler.start_phase("crossover", items_to_process=len(offspring) // 2)
        offspring = _parallel_crossover(offspring, cxpb, self.toolbox)
        profiler.end_phase()

        # Apply selective repairs after crossover (if enabled)
        if (
            repair_config.get("enabled", False)
            and igls_config.selective_repair.enabled
            and igls_config.selective_repair.apply_after_crossover
        ):
            for i in range(0, len(offspring), 2):
                if i + 1 < len(offspring) and not offspring[i].fitness.valid:
                    if random.random() < igls_config.selective_repair.apply_probability:
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
                            if "crossover_repair_applied" not in [
                                e for e in event_tracker.events
                            ]:
                                event_tracker.add("crossover_repair_applied")

                            if was_repaired1:
                                generation_repair_stats["individuals_repaired"] += 1
                                generation_repair_stats["crossover_repairs"] += 1
                            if was_repaired2:
                                generation_repair_stats["individuals_repaired"] += 1
                                generation_repair_stats["crossover_repairs"] += 1

        # PERFORMANCE: Parallel Mutation (8-12x faster for large populations)
        # Uses ThreadPoolExecutor to apply mutation concurrently
        profiler.start_phase("mutation", items_to_process=len(offspring))
        offspring = _parallel_mutation(offspring, mutpb, self.toolbox)
        profiler.end_phase()

        # Apply selective repairs after mutation (if enabled)
        if (
            repair_config.get("enabled", False)
            and igls_config.selective_repair.enabled
            and igls_config.selective_repair.apply_after_mutation
        ):
            for mutant in offspring:
                if not mutant.fitness.valid:
                    # Probabilistic gate
                    if random.random() < igls_config.selective_repair.apply_probability:
                        from src.ga.operators.intensive_local_search import (
                            apply_selective_probabilistic,
                        )

                        mutant, was_repaired = apply_selective_probabilistic(
                            individual=mutant,
                            context=self.context,
                            apply_probability=1.0,  # Already gated above
                        )

                        if was_repaired:
                            if "mutation_repair_applied" not in [
                                e for e in event_tracker.events
                            ]:
                                event_tracker.add("mutation_repair_applied")

                            generation_repair_stats["individuals_repaired"] += 1
                            generation_repair_stats["mutation_repairs"] += 1

        # Evaluate invalid individuals with GPU acceleration when available
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        if invalid:
            profiler.start_phase("evaluation", items_to_process=len(invalid))

            # GPU batch evaluation (10-50x faster for large populations)
            if self.gpu_evaluator and self.gpu_evaluator.enabled and len(invalid) >= 50:
                try:
                    fitness_values = self.gpu_evaluator.evaluate_batch(
                        invalid,
                        self.context.courses,
                        self.context.instructors,
                        self.context.groups,
                        self.context.rooms,
                    )
                    for ind, fit in zip(invalid, fitness_values):
                        ind.fitness.values = fit
                except Exception as e:
                    logger.warning(f"GPU evaluation failed, falling back to CPU: {e}")
                    # Fallback to CPU
                    fitness_values = list(
                        self.toolbox.map(self.toolbox.evaluate, invalid)
                    )
                    for ind, fit in zip(invalid, fitness_values):
                        ind.fitness.values = fit
            else:
                # CPU evaluation for small batches or when GPU unavailable
                fitness_values = list(self.toolbox.map(self.toolbox.evaluate, invalid))
                for ind, fit in zip(invalid, fitness_values):
                    ind.fitness.values = fit

            profiler.end_phase()

        # PHASE 1.2: Explicit Elitism - preserve top solutions
        elite_size = max(1, int(0.05 * len(self.population)))  # Top 5%
        elite = tools.selBest(self.population, elite_size)

        # Replacement: combine parents, offspring, AND elite
        combined = (
            self.population + offspring + elite
        )  # Elite ensures monotonic improvement
        self.population[:] = self.toolbox.select(combined, len(self.population))

        # RL INTEGRATION: Apply RL-selected heuristics
        if self.rl_enabled:
            self._apply_rl_operators(gen)
            event_tracker.add("rl_operators_applied")

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
            for ind, fit in zip(elite_individuals, fitness_values):
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

        # ========================================================================
        # NEW: INTENSIVE GLOBAL LOCAL SEARCH (IGLS) SYSTEM
        # ========================================================================
        # Three-tier repair strategy with priority resolution:
        #   Tier 1: Exhaustive search (fixed generations: 3, 25)
        #   Tier 2: Greedy full search (stagnation-triggered)
        #   Tier 3: Selective probabilistic (post-mutation cleanup)
        # ========================================================================

        repair_triggered = None  # Track which repair was applied
        igls_metrics = {}

        # TIER 1: Exhaustive Search (Fixed Generations)
        if (
            igls_config.exhaustive_search.enabled
            and gen in igls_config.exhaustive_search.generations
        ):
            console.print(
                f"\n[bold red][!info] exhaustive search triggered on  gen {gen}: "
                f"(steepest descent on top {igls_config.exhaustive_search.population_coverage*100:.0f}%)[/bold red]"
            )

            from src.ga.operators.intensive_local_search import apply_exhaustive_search

            self.population, igls_metrics = apply_exhaustive_search(
                population=self.population,
                context=self.context,
                population_coverage=igls_config.exhaustive_search.population_coverage,
                max_neighborhood_size=igls_config.exhaustive_search.max_neighborhood_size,
                timeout_seconds=igls_config.exhaustive_search.timeout_seconds,
            )

            # Re-evaluate population after exhaustive search
            fitnesses = self.toolbox.map(self.toolbox.evaluate, self.population)
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness.values = fit

            repair_triggered = "exhaustive"
            event_tracker.add("igls_exhaustive_search")

            console.print(
                f"[bold green][!done] exhaustive search complete: "
                f"{igls_metrics['genes_improved']} genes improved, "
                f"total reduction: {igls_metrics['total_improvement']}, "
                f"time: {igls_metrics['execution_time']:.1f}s"
                f"{' [TIMED OUT]' if igls_metrics.get('timed_out') else ''}[/bold green]"
            )

        # TIER 2: Greedy Full Search (Stagnation-Triggered)
        elif (
            igls_config.stagnation_repair.enabled
            and gen >= igls_config.stagnation_repair.min_generation
            and self.stagnation_counter >= igls_config.stagnation_repair.patience
            and (gen - getattr(self, "_last_stagnation_repair_gen", -999))
            >= igls_config.stagnation_repair.cooldown
        ):
            console.print(
                f"\n[bold yellow] [!info] Gen {gen}: STAGNATION REPAIR triggered "
                f"(greedy search on top {igls_config.stagnation_repair.population_coverage*100:.0f}%, "
                f"{self.stagnation_counter} gens stagnant)[/bold yellow]"
            )

            from src.ga.operators.intensive_local_search import apply_greedy_search

            self.population, igls_metrics = apply_greedy_search(
                population=self.population,
                context=self.context,
                population_coverage=igls_config.stagnation_repair.population_coverage,
                max_iterations=igls_config.stagnation_repair.max_iterations,
                timeout_seconds=igls_config.stagnation_repair.timeout_seconds,
            )

            # Re-evaluate population after greedy search
            fitnesses = self.toolbox.map(self.toolbox.evaluate, self.population)
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness.values = fit

            repair_triggered = "greedy_stagnation"
            event_tracker.add("igls_stagnation_repair")

            # Reset stagnation counter and update last repair generation
            self.stagnation_counter = 0
            self._last_stagnation_repair_gen = gen

            console.print(
                f"[bold green]   ✓ Stagnation repair complete: "
                f"{igls_metrics['genes_improved']} genes improved, "
                f"total reduction: {igls_metrics['total_improvement']}, "
                f"time: {igls_metrics['execution_time']:.1f}s"
                f"{' [TIMED OUT]' if igls_metrics.get('timed_out') else ''}[/bold green]"
            )

        # Store IGLS metrics if repair was triggered
        if repair_triggered:
            igls_metrics["repair_type"] = repair_triggered
            igls_metrics["generation"] = gen
            if not hasattr(self.metrics, "igls_history"):
                self.metrics.igls_history = []
            self.metrics.igls_history.append(igls_metrics)

        # ========================================================================
        # END: INTENSIVE GLOBAL LOCAL SEARCH (IGLS) SYSTEM
        # ========================================================================

        # ========================================================================
        # LNS-IGLS REPAIR SYSTEM
        # ========================================================================
        # Apply LNS-IGLS repair to best individuals when triggered
        lns_config = get_config().lns
        if lns_config.enabled:
            from src.lns.lns_operator import should_trigger_lns_repair, lns_igls_repair

            # Check if LNS should be triggered
            should_trigger = should_trigger_lns_repair(
                generation=gen,
                trigger_interval=lns_config.trigger_interval,
                stagnation_counter=self.stagnation_counter,
                stagnation_threshold=lns_config.stagnation_threshold,
                force_trigger_generations=lns_config.force_trigger_generations,
            )

            if should_trigger:
                event_tracker.add("lns_repair_triggered")
                console.print(
                    f"\n[bold blue][!info] LNS-IGLS repair triggered on gen {gen}[/bold blue]"
                )

                # Get best individuals
                num_to_repair = min(lns_config.apply_to_best_n, len(self.population))
                best_individuals = tools.selBest(self.population, num_to_repair)

                # Apply LNS-IGLS repair to each
                repaired_count = 0
                for idx, individual in enumerate(best_individuals):
                    console.print(
                        f"[dim]   Repairing individual {idx+1}/{num_to_repair}...[/dim]"
                    )

                    repaired = lns_igls_repair(
                        individual=individual,
                        courses=self.context.courses,
                        instructors=self.context.instructors,
                        groups=self.context.groups,
                        rooms=self.context.rooms,
                        max_subproblem_size=lns_config.max_subproblem_size,
                        min_subproblem_size=lns_config.min_subproblem_size,
                        expand_hops=lns_config.expand_neighborhood_hops,
                        igls_max_iterations=lns_config.igls_max_iterations,
                        igls_time_limit=lns_config.igls_time_limit,
                        enable_diagnostics=lns_config.enable_diagnostics,
                    )

                    # If repair was successful (returned different individual), update
                    if repaired is not individual:
                        # Replace in population
                        pop_idx = self.population.index(individual)
                        self.population[pop_idx] = repaired
                        repaired_count += 1
                        # Invalidate fitness
                        del repaired.fitness.values

                # Re-evaluate repaired individuals
                if repaired_count > 0:
                    invalid = [ind for ind in self.population if not ind.fitness.valid]
                    fitness_values = list(
                        self.toolbox.map(self.toolbox.evaluate, invalid)
                    )
                    for ind, fit in zip(invalid, fitness_values):
                        ind.fitness.values = fit

                    event_tracker.add("lns_igls_repair_applied")
                    console.print(
                        f"[bold green]   ✓ LNS-IGLS repair complete: "
                        f"{repaired_count}/{num_to_repair} individuals repaired[/bold green]"
                    )

                    # Reset stagnation counter after successful repair
                    # This prevents immediate re-triggering on next generation
                    self.stagnation_counter = 0
                    logger.info(
                        f"Stagnation counter reset after LNS-IGLS repair (gen {gen})"
                    )
                else:
                    console.print(
                        "[yellow]   LNS-IGLS repair: no improvements found[/yellow]"
                    )

        # ========================================================================
        # END: LNS-IGLS REPAIR SYSTEM
        # ========================================================================

        # Store generation repair stats
        self.metrics.repair_stats.append(generation_repair_stats)

        # Track metrics (also logs to constraint logger)
        self._track_metrics(gen, event_tracker)

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
            crossover_prob = self.config.crossover_prob
            mutation_prob = self.config.mutation_prob
        else:
            # Late phase: exploit (refine good solutions)
            crossover_prob = 0.9
            mutation_prob = 0.2

        return crossover_prob, mutation_prob

    def _track_metrics(self, gen: int, event_tracker=None):
        """
        Record metrics for current generation.

        Args:
            gen: Generation number (-1 for initial population, 0+ for evolved generations)
            event_tracker: Optional EventTracker with events from this generation
        """
        # Import new metrics modules
        from src.metrics.hypervolume import (
            calculate_hypervolume,
            get_hypervolume_reference_point,
        )
        from src.metrics.pareto_metrics import (
            calculate_spacing,
            calculate_inverted_generational_distance,
            calculate_spread,
            get_pareto_front_size,
        )
        from src.metrics.convergence import calculate_constraint_satisfaction_rate

        # Basic metrics
        self.metrics.hard_violations.append(
            min(ind.fitness.values[0] for ind in self.population)
        )
        self.metrics.soft_penalties.append(
            min(ind.fitness.values[1] for ind in self.population)
        )
        diversity = average_pairwise_diversity(self.population)
        self.metrics.diversity.append(diversity)

        # Phase 1: Essential multi-objective metrics
        # Calculate hypervolume (use consistent reference point)
        if gen == 0 or gen == -1:
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
        if gen == -1 or gen == 0:
            # Store initial Pareto front as reference
            pareto_front = tools.sortNondominated(
                self.population, len(self.population), first_front_only=True
            )[0]
            self.metrics.reference_front = [ind for ind in pareto_front]

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

        # Detailed constraint breakdown
        best = tools.selBest(self.population, 1)[0]
        hard_details, soft_details = evaluate_detailed(
            best,
            self.context.courses,
            self.context.instructors,
            self.context.groups,
            self.context.rooms,
        )

        for name in self.hard_constraint_names:
            self.metrics.detailed_hard[name].append(hard_details[name])

        for name in self.soft_constraint_names:
            self.metrics.detailed_soft[name].append(soft_details[name])

        # ENHANCEMENT: Record violations to heatmap
        if self.violation_heatmap and gen >= 0:  # Skip initial population
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
        self, gen: int, best, hard_details: Dict, soft_details: Dict
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

    def _accumulate_repair_stats(self, agg: Dict, stats: Dict) -> None:
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

        for src_key, dst_key in key_map.items():
            if src_key in stats:
                agg[dst_key] += stats.get(src_key, 0)

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
        for ind, fit in zip(new_individuals, fitness_values):
            ind.fitness.values = fit

        # Replace population
        self.population[:] = elite + new_individuals

        # Calculate diversity improvement
        from src.metrics.diversity import average_pairwise_diversity

        new_diversity = average_pairwise_diversity(self.population)

        console.print(
            f"   [green][!ok] Restart complete! New diversity: {new_diversity:.4f}[/green]"
        )

        # Update tracking
        self.last_restart_gen = gen
        self.prolonged_stagnation_counter = 0  # Reset counter

    def get_best_solution(self):
        """
        Select best solution from final population.

        Prefers feasible solutions (hard constraints satisfied) with
        lowest soft constraint penalty. If no feasible solution exists,
        returns the solution with fewest hard constraint violations.

        Returns:
            Best individual from the final population
        """
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

            # Check for duplicates within individual
            if len(current) != len(current_set):
                duplicates = [x for x in current if current.count(x) > 1]
                raise ValueError(
                    f"[X] Individual {idx} contains DUPLICATE (course, course_type, group) pairs!\n"
                    f"   Duplicates: {set(duplicates)}"
                )
