"""
GA Scheduler Module

Encapsulates NSGA-II genetic algorithm execution for course scheduling.
Extracted from monolithic main.py for better testability and separation of concerns.
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from deap import base, tools
import random
import time
import psutil
import threading
import os
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    ProgressColumn,
    Task,
)
from rich.table import Table
from rich.live import Live
from rich.text import Text

from src.ga.population import generate_course_group_aware_population
from src.ga.operators.crossover import crossover_course_group_aware
from src.ga.operators.mutation import mutate_individual
from src.ga.evaluator.fitness import evaluate
from src.config import get_config
from src.ga.evaluator.detailed_fitness import evaluate_detailed
from src.metrics.diversity import average_pairwise_diversity
from src.core.types import SchedulingContext

console = Console()


# ============================================================================
# Worker Initialization for Multiprocessing
# ============================================================================
# Module-level worker context (set once per worker process)
_WORKER_CONTEXT = None


def _worker_init(data_dir: str, seed: int):
    """
    Initialize worker process by loading data from JSON files.

    This function is called once when each worker process starts.
    It sets up DEAP creator types and loads scheduling context from disk.

    This approach avoids pickling complex objects - workers just read the
    same JSON files that the main process read.

    Args:
        data_dir: Directory containing input JSON files
        seed: Random seed for reproducibility

    Fixes:
        - Bug #1: No pickling overhead (workers load from disk once)
        - Bug #2: Random seed propagation (seed set in each worker)
        - Bug #4: Creator types missing (types created in each worker)
    """
    global _WORKER_CONTEXT
    import os
    import sys
    from io import StringIO
    from deap import creator, base
    from src.encoder.input_encoder import (
        load_courses,
        load_groups,
        load_instructors,
        load_rooms,
        link_courses_and_groups,
        link_courses_and_instructors,
    )
    from src.encoder.quantum_time_system import QuantumTimeSystem

    # Set up DEAP creator types (required for Windows spawn)
    if not hasattr(creator, "FitnessMulti"):
        # Use two-objective minimization: hard penalties, soft penalties
        # Both objectives are minimized equally in terms of direction; relative
        # importance is controlled via constraint weights and soft_weight_factor
        # in configuration, not by magnitudes here.
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    # Set environment variable to indicate we're in a worker process
    # This allows other modules to suppress warnings
    os.environ["_GA_WORKER_PROCESS"] = "1"

    # Suppress all print output from data loading (workers should be silent)
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        # Load data from JSON files (same as main process)
        qts = QuantumTimeSystem()
        groups = load_groups(os.path.join(data_dir, "Groups.json"), qts)

        # Get enrolled course codes
        enrolled_course_codes = set()
        for group in groups.values():
            enrolled_course_codes.update(group.enrolled_courses)

        # Load and filter courses
        all_courses = load_courses(os.path.join(data_dir, "Course.json"))
        courses = {
            key: course
            for key, course in all_courses.items()
            if key[0] in enrolled_course_codes
        }

        instructors = load_instructors(os.path.join(data_dir, "Instructors.json"), qts)
        rooms = load_rooms(os.path.join(data_dir, "Rooms.json"), qts)

        # Link relationships (suppress output)
        link_courses_and_groups(courses, groups)
        link_courses_and_instructors(courses, instructors)
    finally:
        # Restore stdout
        sys.stdout = old_stdout

    # Store scheduling context in module-level variable
    _WORKER_CONTEXT = {
        "courses": courses,
        "instructors": instructors,
        "groups": groups,
        "rooms": rooms,
    }

    # Propagate random seed to worker
    random.seed(seed)


def _worker_evaluate(individual):
    """
    Evaluate individual using worker-local context.

    This function is called for each evaluation. It retrieves the
    scheduling context from module-level state (set once in _worker_init)
    instead of pickling it every time.

    Args:
        individual: GA individual to evaluate

    Returns:
        Tuple of (hard_violations, soft_penalty)
    """
    return evaluate(
        individual,
        _WORKER_CONTEXT["courses"],
        _WORKER_CONTEXT["instructors"],
        _WORKER_CONTEXT["groups"],
        _WORKER_CONTEXT["rooms"],
    )


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

        # NEW: Hypervolume reference point (initialized during first metric tracking)
        self._hypervolume_ref_point = None

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

    def evolve(self):
        """Run genetic algorithm evolution loop."""
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

        # Get CPU count for per-core monitoring
        cpu_count = psutil.cpu_count(logical=True)

        # Get current process for process-specific monitoring
        process = psutil.Process(os.getpid())

        # Create CPU monitoring bars (one per core)
        cpu_bars = []
        cpu_tasks = []
        for i in range(cpu_count):
            cpu_bar = Progress(
                TextColumn(f"[dim]cpu {i:2d}:[/dim]"),
                BarColumn(bar_width=20),
                TextColumn("[yellow]{task.fields[cpu_percent]:>5.1f}%[/yellow]"),
                console=console,
                refresh_per_second=10,
            )
            cpu_bars.append(cpu_bar)

        # Create Memory monitoring bar (process-specific) - WIDER bar
        memory_bar = Progress(
            TextColumn("[dim]memory:[/dim]"),
            BarColumn(bar_width=20),
            TextColumn("[magenta]{task.fields[mem_used]}[/magenta]"),
            TextColumn("[dim](Peak: {task.fields[mem_peak]})[/dim]"),
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
        # Add Memory bar
        progress_table.add_row(memory_bar)
        # Add CPU bars
        for cpu_bar in cpu_bars:
            progress_table.add_row(cpu_bar)
        # Add constraint legend rows below CPU
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

            # Initialize CPU monitoring tasks
            for i, cpu_bar in enumerate(cpu_bars):
                cpu_task = cpu_bar.add_task(
                    "",
                    total=100,
                    cpu_percent=0.0,
                )
                cpu_tasks.append(cpu_task)

            # Initialize Memory monitoring task (process-specific)
            mem_info = process.memory_info()
            memory_task = memory_bar.add_task(
                "",
                total=100,
                mem_used=f"{mem_info.rss / (1024**3):.2f}GB",
                mem_peak=f"{mem_info.rss / (1024**3):.2f}GB",
            )

            # Initialize legend bar tasks (static display)
            for legend_bar in legend_bars:
                legend_bar.add_task("")

            # Initialize spacing task
            spacing_bar.add_task("")

            # Tracking variables for resource monitoring
            monitoring_active = True
            peak_memory = mem_info.rss

            def update_resource_monitors():
                """Background thread to update CPU and Memory metrics every second"""
                nonlocal peak_memory
                while monitoring_active:
                    try:
                        # Update CPU monitoring (per-core, process-specific)
                        # Get per-core CPU percentages (system-wide)
                        cpu_percentages = psutil.cpu_percent(interval=0.5, percpu=True)
                        for i, (cpu_bar, cpu_task) in enumerate(
                            zip(cpu_bars, cpu_tasks)
                        ):
                            if i < len(cpu_percentages):
                                cpu_bar.update(
                                    cpu_task,
                                    completed=cpu_percentages[i],
                                    cpu_percent=cpu_percentages[i],
                                )

                        # Update Memory monitoring (process-specific)
                        current_mem_info = process.memory_info()
                        current_mem = current_mem_info.rss
                        if current_mem > peak_memory:
                            peak_memory = current_mem

                        # Calculate percentage of system memory used by this process
                        system_total = psutil.virtual_memory().total
                        mem_percent = (current_mem / system_total) * 100

                        memory_bar.update(
                            memory_task,
                            completed=mem_percent,
                            mem_used=f"{current_mem / (1024**3):.2f}GB",
                            mem_peak=f"{peak_memory / (1024**3):.2f}GB",
                        )

                        time.sleep(0.5)  # Update twice per second for smoother display
                    except Exception:
                        pass  # Silently handle any errors in monitoring thread

            # Start background monitoring thread
            monitor_thread = threading.Thread(
                target=update_resource_monitors, daemon=True
            )
            monitor_thread.start()

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

                # Format exactly as requested: [!ok] gen x/y : hc = , sc = , t=4s,  hc1=, hc2=.. sc1=., sc2=...
                console.print(
                    f"[dim][!ok] gen {gen+1}/{self.config.generations} : "
                    f"hc={best.fitness.values[0]:.0f}, sc={best.fitness.values[1]:.2f}, "
                    f"t={gen_time:.1f}s,  {hc_list} {sc_list}[/dim]"
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

            # Stop monitoring thread
            monitoring_active = False
            monitor_thread.join(timeout=1.0)

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
        offspring = self.toolbox.select(self.population, len(self.population))
        offspring = list(map(self.toolbox.clone, offspring))

        # Crossover (using adaptive probability)
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:  # ← Use adaptive crossover probability
                self.toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

                # Apply PROBABILISTIC repairs after crossover if enabled (Tier 3: Selective)
                # Only applies to a fraction of crossover offspring
                if (
                    repair_config.get("enabled", False)
                    and igls_config.selective_repair.enabled
                    and igls_config.selective_repair.apply_after_crossover
                ):

                    # Probabilistic gate
                    if random.random() < igls_config.selective_repair.apply_probability:
                        from src.ga.operators.intensive_local_search import (
                            apply_selective_probabilistic,
                        )

                        offspring[i - 1], was_repaired1 = apply_selective_probabilistic(
                            individual=offspring[i - 1],
                            context=self.context,
                            apply_probability=1.0,  # Already gated above
                        )
                        offspring[i], was_repaired2 = apply_selective_probabilistic(
                            individual=offspring[i],
                            context=self.context,
                            apply_probability=1.0,  # Already gated above
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

        # Mutation (using adaptive probability)
        for mutant in offspring:
            if random.random() < mutpb:  # ← Use adaptive mutation probability
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

                # Apply PROBABILISTIC repairs after mutation if enabled (Tier 3: Selective)
                # Only applies to a fraction of mutated offspring (e.g., 30%)
                if (
                    repair_config.get("enabled", False)
                    and igls_config.selective_repair.enabled
                    and igls_config.selective_repair.apply_after_mutation
                ):

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

        # Evaluate invalid individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        if invalid:
            # Only log when evaluating many individuals (helpful for debugging)
            # Removed frequent logging to reduce console clutter

            # Use toolbox.map for parallel evaluation when pool is available
            fitness_values = list(self.toolbox.map(self.toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitness_values):
                ind.fitness.values = fit

        # PHASE 1.2: Explicit Elitism - preserve top solutions
        elite_size = max(1, int(0.05 * len(self.population)))  # Top 5%
        elite = tools.selBest(self.population, elite_size)

        # Replacement: combine parents, offspring, AND elite
        combined = (
            self.population + offspring + elite
        )  # Elite ensures monotonic improvement
        self.population[:] = self.toolbox.select(combined, len(self.population))

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

            start_time = time.time()
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

            start_time = time.time()
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

        # Store generation repair stats
        self.metrics.repair_stats.append(generation_repair_stats)

        # Track metrics (also logs to constraint logger)
        self._track_metrics(gen, event_tracker)

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
