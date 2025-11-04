"""
CP-SAT Scheduler

Main orchestrator for generating feasible solutions using Google OR-Tools CP-SAT solver.

Features:
    - Multi-solution generation with diversity
    - Configurable solve time limits
    - Solution quality metrics
    - Random seed support for reproducibility
    - Comprehensive runtime logging
"""

from typing import List, Optional
import time
import logging
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from functools import partial
from ortools.sat.python import cp_model  # type: ignore
from rich.console import Console

from src.core.types import SchedulingContext
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.entities.decoded_session import CourseSession
from src.ortools.model_builder import ModelBuilder
from src.ortools.solution_decoder import decode_cp_solution, merge_consecutive_sessions

console = Console()


# Setup logging
def setup_cp_logger(output_dir: str = "output") -> logging.Logger:
    """Setup dedicated logger for CP-SAT operations with timestamps."""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / f"cpsat_runtime_{timestamp}.log"

    logger = logging.getLogger("cp_scheduler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers

    # File handler with detailed formatting
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"CP-SAT Runtime Log - Started at {datetime.now()}")
    logger.info("=" * 80)

    return logger


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    """
    Callback to collect multiple solutions during CP-SAT search.
    """

    def __init__(
        self,
        session_vars,
        var_factory,
        context,
        max_solutions: int = 50,
        logger: Optional[logging.Logger] = None,
    ):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.session_vars = session_vars
        self.var_factory = var_factory
        self.context = context
        self.max_solutions = max_solutions
        self.solutions = []
        self.solution_count = 0
        self._start_time = None
        self.logger = logger

    def on_solution_callback(self):
        """Called each time a solution is found."""
        import time

        if self._start_time is None:
            self._start_time = time.time()

        self.solution_count += 1
        elapsed = time.time() - self._start_time

        # Decode and store solution
        sessions = decode_cp_solution(
            self, self.session_vars, self.var_factory, self.context
        )

        # Merge consecutive sessions
        merged_sessions = merge_consecutive_sessions(sessions)

        self.solutions.append(merged_sessions)

        console.print(
            f"  [cyan]✓ Solution {self.solution_count}/{self.max_solutions}[/cyan] "
            f"found in {elapsed:.1f}s "
            f"[dim]({len(merged_sessions)} sessions)[/dim]"
        )

        if self.logger:
            self.logger.info(
                f"SOLUTION {self.solution_count} FOUND - "
                f"Elapsed: {elapsed:.2f}s, Sessions: {len(merged_sessions)}"
            )

        # Stop if we have enough solutions
        if self.solution_count >= self.max_solutions:
            console.print(f"\n  [green]Target reached! Stopping search...[/green]")
            if self.logger:
                self.logger.info(
                    f"Target of {self.max_solutions} solutions reached. Stopping search."
                )
            self.StopSearch()


class CPScheduler:
    """
    CP-SAT based scheduler for generating feasible course schedules.

    Generates multiple diverse feasible solutions that satisfy all hard constraints.
    """

    def __init__(
        self,
        context: SchedulingContext,
        qts: QuantumTimeSystem,
        time_limit_seconds: int = 300,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize CP-SAT scheduler.

        Args:
            context: SchedulingContext with all entities
            qts: QuantumTimeSystem for time calculations
            time_limit_seconds: Maximum time to spend solving (default: 5 minutes)
            random_seed: Random seed for reproducibility (default: None)
        """
        self.context = context
        self.qts = qts
        self.time_limit = time_limit_seconds
        self.random_seed = random_seed

    def generate_feasible_solutions(
        self, num_solutions: int = 50, logger: Optional[logging.Logger] = None
    ) -> List[List[CourseSession]]:
        """
        Generate multiple feasible solutions using CP-SAT.

        Uses solution enumeration to find diverse feasible schedules.

        Args:
            num_solutions: Target number of solutions to generate
            logger: Optional logger for runtime tracking

        Returns:
            List of solutions, where each solution is a List[CourseSession]

        Raises:
            ValueError: If problem is infeasible
        """
        import threading

        if logger is None:
            logger = setup_cp_logger()

        console.print("\n[bold cyan]═══ CP-SAT Solver ═══[/bold cyan]\n")

        logger.info("CONFIGURATION:")
        logger.info(f"  Target Solutions: {num_solutions}")
        logger.info(f"  Time Limit: {self.time_limit}s")
        logger.info(f"  Random Seed: {self.random_seed}")
        logger.info("-" * 80)

        start_time = time.time()

        # Build model
        console.print("[cyan]Building CP-SAT model...[/cyan]")
        logger.info("MODEL BUILDING - Started")

        builder = ModelBuilder(self.context, self.qts)
        model, session_vars, var_factory = builder.build_model()

        build_time = time.time() - start_time
        console.print(f"  [green]✓ Model built in {build_time:.2f}s[/green]")
        logger.info(f"MODEL BUILDING - Completed in {build_time:.2f}s")
        logger.info(
            f"  Variables: {len(session_vars) * 3}"
        )  # start, instructor, room per session
        logger.info(f"  Constraints: {len(model.Proto().constraints)}")
        logger.info("-" * 80)

        # Configure solver with aggressive optimization
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.log_search_progress = True

        # Parallel search with all cores
        num_workers = cpu_count()
        solver.parameters.num_search_workers = num_workers

        # Advanced optimization parameters
        solver.parameters.cp_model_presolve = True  # Enable presolve
        solver.parameters.symmetry_level = 2  # Maximum symmetry breaking
        solver.parameters.linearization_level = 2  # Aggressive linearization
        solver.parameters.optimize_with_core = True  # Use core-guided search
        solver.parameters.use_branching_in_lp = True  # Better branching
        solver.parameters.exploit_all_precedences = True  # Better precedence inference
        solver.parameters.instantiate_all_variables = (
            False  # Lazy instantiation for speed
        )
        solver.parameters.use_sat_inprocessing = True  # SAT-level optimizations
        solver.parameters.min_orthogonality_for_lp_constraints = (
            0.05  # Better LP relaxations
        )

        console.print(
            f"  [cyan]Using {num_workers} parallel workers with advanced optimizations[/cyan]"
        )
        logger.info(f"  Parallel Workers: {num_workers}")
        logger.info(f"  Presolve: Enabled")
        logger.info(f"  Symmetry Breaking: Level 2")
        logger.info(f"  Linearization: Level 2")
        logger.info(f"  Core-guided Search: Enabled")

        if self.random_seed is not None:
            solver.parameters.random_seed = self.random_seed

        # Search for multiple solutions
        console.print(
            f"[bold]Searching for {num_solutions} feasible solutions...[/bold]"
        )
        console.print(f"Time limit: {self.time_limit} seconds")
        console.print(f"[dim]Solver will report progress every few seconds...[/dim]\n")

        solution_collector = SolutionCollector(
            session_vars,
            var_factory,
            self.context,
            max_solutions=num_solutions,
            logger=logger,
        )

        # Progress monitoring thread
        stop_monitoring = threading.Event()

        def monitor_progress():
            """Print periodic solver stats"""
            last_count = 0
            while not stop_monitoring.is_set():
                stop_monitoring.wait(5)  # Check every 5 seconds
                if not stop_monitoring.is_set():
                    elapsed = time.time() - start_time
                    if solution_collector.solution_count > last_count:
                        last_count = solution_collector.solution_count
                    else:
                        # No new solutions in last 5s - show we're still working
                        console.print(
                            f"  [dim]⏳ Searching... {elapsed:.0f}s elapsed, "
                            f"{solution_collector.solution_count} solutions so far[/dim]"
                        )
                        if logger:
                            logger.info(
                                f"Progress: {elapsed:.0f}s elapsed, {solution_collector.solution_count} solutions"
                            )

        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        # Solve
        logger.info("SOLVING - Started")
        solve_start = time.time()

        try:
            status = solver.Solve(model, solution_collector)
        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=1)

        solve_time = time.time() - solve_start
        elapsed_time = time.time() - start_time

        # Process results
        console.print(f"\n[bold]Solver Status:[/bold] {solver.StatusName(status)}")
        console.print(
            f"[bold]Solutions Found:[/bold] {solution_collector.solution_count}"
        )
        console.print(f"[bold]Solve Time:[/bold] {elapsed_time:.2f}s")
        console.print(f"[bold]Branches:[/bold] {solver.NumBranches()}")
        console.print(f"[bold]Conflicts:[/bold] {solver.NumConflicts()}")

        logger.info(f"SOLVING - Completed in {solve_time:.2f}s")
        logger.info(f"  Status: {solver.StatusName(status)}")
        logger.info(f"  Solutions Found: {solution_collector.solution_count}")
        logger.info(f"  Wall Time: {solver.WallTime():.2f}s")
        logger.info(f"  User Time: {solver.UserTime():.2f}s")
        logger.info(f"  Branches: {solver.NumBranches()}")
        logger.info(f"  Conflicts: {solver.NumConflicts()}")
        logger.info("-" * 80)
        logger.info(f"TOTAL RUNTIME: {elapsed_time:.2f}s")
        logger.info("=" * 80)

        if status == cp_model.INFEASIBLE:
            console.print("\n[bold red]✗ Problem is INFEASIBLE[/bold red]")
            console.print("No valid schedule exists that satisfies all constraints.")
            logger.error("Problem is INFEASIBLE - no valid schedule exists")
            raise ValueError("Problem is infeasible - no valid schedule exists")

        if solution_collector.solution_count == 0:
            console.print(
                "\n[bold yellow]⚠ No solutions found within time limit[/bold yellow]"
            )
            logger.warning("No solutions found within time limit")
            raise ValueError("No solutions found within time limit")

        console.print(
            f"\n[bold green]✓ Generated {solution_collector.solution_count} feasible solutions[/bold green]\n"
        )
        logger.info(
            f"SUCCESS: Generated {solution_collector.solution_count} feasible solution(s)"
        )

        return solution_collector.solutions

    def generate_single_solution(
        self, logger: Optional[logging.Logger] = None
    ) -> List[CourseSession]:
        """
        Generate a single feasible solution quickly.

        Args:
            logger: Optional logger for runtime tracking

        Returns:
            List of CourseSession objects representing the schedule

        Raises:
            ValueError: If problem is infeasible
        """
        import threading

        if logger is None:
            logger = setup_cp_logger()

        console.print(
            "\n[bold cyan]═══ CP-SAT Solver (Single Solution) ═══[/bold cyan]\n"
        )

        # Check for unlimited time mode (24+ hours is effectively unlimited)
        is_unlimited = self.time_limit <= 0 or self.time_limit >= 86400
        if is_unlimited:
            console.print(
                "[bold yellow]⚠️ EXTENDED TIME MODE - This may run for many hours/days![/bold yellow]"
            )
            console.print("[dim]Press Ctrl+C to abort if needed[/dim]\n")

        logger.info("CONFIGURATION:")
        logger.info(f"  Mode: Single Solution (hard constraints only)")
        logger.info(
            f"  Time Limit: {'UNLIMITED' if is_unlimited else f'{self.time_limit}s'}"
        )
        logger.info(f"  Random Seed: {self.random_seed}")
        logger.info("-" * 80)

        start_time = time.time()

        # Build model
        console.print("[cyan]Building CP-SAT model...[/cyan]")
        logger.info("MODEL BUILDING - Started")

        builder = ModelBuilder(self.context, self.qts)
        model, session_vars, var_factory = builder.build_model()

        build_time = time.time() - start_time
        console.print(f"  [green]✓ Model built in {build_time:.2f}s[/green]")
        logger.info(f"MODEL BUILDING - Completed in {build_time:.2f}s")
        logger.info(
            f"  Variables: {len(session_vars) * 3}"
        )  # start, instructor, room per session
        logger.info(f"  Constraints: {len(model.Proto().constraints)}")

        # Warn about large models
        num_constraints = len(model.Proto().constraints)
        if num_constraints > 1_000_000:
            console.print(
                f"\n[bold yellow]⚠️ WARNING: Large model detected![/bold yellow]"
            )
            console.print(f"  Constraints: {num_constraints:,}")
            console.print(f"  Estimated solve time: [bold]1-12+ hours[/bold]")
            if not is_unlimited:
                console.print(
                    f"  [red]Your time limit of {self.time_limit}s may be insufficient![/red]"
                )

        logger.info("-" * 80)

        # Configure solver - MINIMAL presolve for large problems
        solver = cp_model.CpSolver()

        # Set time limit (0 = unlimited, don't set parameter at all)
        if self.time_limit > 0:
            solver.parameters.max_time_in_seconds = self.time_limit
        else:
            # No time limit - let it run forever
            pass

        solver.parameters.log_search_progress = True

        # Parallel search with all cores
        num_workers = cpu_count()
        solver.parameters.num_search_workers = num_workers

        # MINIMAL optimization - avoid constraint explosion
        solver.parameters.cp_model_presolve = True
        solver.parameters.symmetry_level = 0  # Disable (causes explosion)
        solver.parameters.linearization_level = 0  # Disable (creates 40M constraints)
        solver.parameters.optimize_with_core = False  # Disable for large problems
        solver.parameters.stop_after_first_solution = True  # Stop at first solution
        solver.parameters.max_presolve_iterations = 3  # Limit presolve passes

        console.print(
            f"  [cyan]Using {num_workers} parallel workers (MINIMAL presolve)[/cyan]"
        )
        logger.info(f"  Parallel Workers: {num_workers}")
        logger.info(f"  Presolve: MINIMAL (avoid explosion)")
        logger.info(f"  Symmetry Breaking: Disabled")
        logger.info(f"  Linearization: Disabled")

        if self.random_seed is not None:
            solver.parameters.random_seed = self.random_seed

        # Solve
        console.print(f"[bold]Searching for feasible solution...[/bold]")
        if is_unlimited:
            console.print("[yellow]Time limit: UNLIMITED[/yellow]")
        else:
            console.print(f"Time limit: {self.time_limit} seconds")
        console.print(f"[dim]Progress updates every 10 seconds...[/dim]\n")

        # Progress monitoring with human-readable time
        stop_monitoring = threading.Event()

        def monitor_progress():
            """Print periodic status updates with human-readable time"""
            warning_shown = False
            while not stop_monitoring.is_set():
                stop_monitoring.wait(10)  # Check every 10 seconds
                if not stop_monitoring.is_set():
                    elapsed = time.time() - start_time
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    seconds = int(elapsed % 60)

                    time_str = (
                        f"{hours}h {minutes}m {seconds}s"
                        if hours > 0
                        else f"{minutes}m {seconds}s"
                    )
                    console.print(
                        f"  [dim]⏳ Still searching... {time_str} elapsed[/dim]"
                    )

                    # Show warning after 30 minutes with no solution
                    if elapsed > 1800 and not warning_shown:
                        console.print(
                            f"\n[bold yellow]⚠️ No solution after 30 minutes![/bold yellow]"
                        )
                        console.print(
                            "  Problem may be infeasible or extremely constrained"
                        )
                        console.print(
                            "  Consider pressing Ctrl+C and reducing problem size\n"
                        )
                        warning_shown = True

                    if logger:
                        logger.info(
                            f"Progress: {elapsed:.0f}s elapsed, still searching..."
                        )

        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        logger.info("SOLVING - Started")
        solve_start = time.time()

        try:
            status = solver.Solve(model)
        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=1)

        solve_time = time.time() - solve_start
        elapsed_time = time.time() - start_time

        # Process results
        console.print(f"\n[bold]Solver Status:[/bold] {solver.StatusName(status)}")
        console.print(
            f"[bold]Total Runtime:[/bold] {elapsed_time:.2f}s ({elapsed_time/3600:.2f} hours)"
        )
        console.print(f"[bold]Branches:[/bold] {solver.NumBranches():,}")
        console.print(f"[bold]Conflicts:[/bold] {solver.NumConflicts():,}")

        logger.info(f"SOLVING - Completed in {solve_time:.2f}s")
        logger.info(f"  Status: {solver.StatusName(status)}")
        logger.info(f"  Wall Time: {solver.WallTime():.2f}s")
        logger.info(f"  User Time: {solver.UserTime():.2f}s")
        logger.info(f"  Branches: {solver.NumBranches():,}")
        logger.info(f"  Conflicts: {solver.NumConflicts():,}")
        logger.info("-" * 80)
        logger.info(
            f"TOTAL RUNTIME: {elapsed_time:.2f}s ({elapsed_time/3600:.2f} hours)"
        )
        logger.info("=" * 80)

        if status == cp_model.INFEASIBLE:
            console.print("\n[bold red]✗ Problem is INFEASIBLE[/bold red]")
            logger.error("Problem is INFEASIBLE - no valid schedule exists")
            raise ValueError("Problem is infeasible")

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            console.print("\n[bold yellow]⚠ No solution found[/bold yellow]")
            logger.warning("No solution found within time limit")
            raise ValueError("No solution found within time limit")

        # Decode solution
        logger.info("DECODING SOLUTION - Started")
        decode_start = time.time()

        sessions = decode_cp_solution(solver, session_vars, var_factory, self.context)
        merged_sessions = merge_consecutive_sessions(sessions)

        decode_time = time.time() - decode_start
        logger.info(f"DECODING SOLUTION - Completed in {decode_time:.2f}s")
        logger.info(f"  Total Sessions: {len(merged_sessions)}")
        logger.info("-" * 80)

        console.print(
            f"\n[bold green]✓ Found feasible solution with {len(merged_sessions)} sessions[/bold green]\n"
        )
        logger.info(
            f"SUCCESS: Generated feasible solution with {len(merged_sessions)} sessions"
        )

        return merged_sessions
