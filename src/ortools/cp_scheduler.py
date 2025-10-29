"""
CP-SAT Scheduler

Main orchestrator for generating feasible solutions using Google OR-Tools CP-SAT solver.

Features:
    - Multi-solution generation with diversity
    - Configurable solve time limits
    - Solution quality metrics
    - Random seed support for reproducibility
"""

from typing import List, Optional
import time
from ortools.sat.python import cp_model
from rich.console import Console

from src.core.types import SchedulingContext
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.entities.decoded_session import CourseSession
from src.ortools.model_builder import ModelBuilder
from src.ortools.solution_decoder import decode_cp_solution, merge_consecutive_sessions

console = Console()


class SolutionCollector(cp_model.CpSolverSolutionCallback):
    """
    Callback to collect multiple solutions during CP-SAT search.
    """

    def __init__(self, session_vars, var_factory, context, max_solutions: int = 50):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.session_vars = session_vars
        self.var_factory = var_factory
        self.context = context
        self.max_solutions = max_solutions
        self.solutions = []
        self.solution_count = 0
        self._start_time = None

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

        # Stop if we have enough solutions
        if self.solution_count >= self.max_solutions:
            console.print(f"\n  [green]Target reached! Stopping search...[/green]")
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
        self, num_solutions: int = 50
    ) -> List[List[CourseSession]]:
        """
        Generate multiple feasible solutions using CP-SAT.

        Uses solution enumeration to find diverse feasible schedules.

        Args:
            num_solutions: Target number of solutions to generate

        Returns:
            List of solutions, where each solution is a List[CourseSession]

        Raises:
            ValueError: If problem is infeasible
        """
        import threading

        console.print("\n[bold cyan]═══ CP-SAT Solver ═══[/bold cyan]\n")

        start_time = time.time()

        # Build model
        builder = ModelBuilder(self.context, self.qts)
        model, session_vars, var_factory = builder.build_model()

        # Configure solver
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.enumerate_all_solutions = True
        solver.parameters.log_search_progress = True  # Enable logging

        if self.random_seed is not None:
            solver.parameters.random_seed = self.random_seed

        # Search for multiple solutions
        console.print(
            f"[bold]Searching for {num_solutions} feasible solutions...[/bold]"
        )
        console.print(f"Time limit: {self.time_limit} seconds")
        console.print(f"[dim]Solver will report progress every few seconds...[/dim]\n")

        solution_collector = SolutionCollector(
            session_vars, var_factory, self.context, max_solutions=num_solutions
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

        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        # Solve
        try:
            status = solver.Solve(model, solution_collector)
        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=1)

        elapsed_time = time.time() - start_time

        # Process results
        console.print(f"\n[bold]Solver Status:[/bold] {solver.StatusName(status)}")
        console.print(
            f"[bold]Solutions Found:[/bold] {solution_collector.solution_count}"
        )
        console.print(f"[bold]Solve Time:[/bold] {elapsed_time:.2f}s")
        console.print(f"[bold]Branches:[/bold] {solver.NumBranches()}")
        console.print(f"[bold]Conflicts:[/bold] {solver.NumConflicts()}")

        if status == cp_model.INFEASIBLE:
            console.print("\n[bold red]✗ Problem is INFEASIBLE[/bold red]")
            console.print("No valid schedule exists that satisfies all constraints.")
            raise ValueError("Problem is infeasible - no valid schedule exists")

        if solution_collector.solution_count == 0:
            console.print(
                "\n[bold yellow]⚠ No solutions found within time limit[/bold yellow]"
            )
            raise ValueError("No solutions found within time limit")

        console.print(
            f"\n[bold green]✓ Generated {solution_collector.solution_count} feasible solutions[/bold green]\n"
        )

        return solution_collector.solutions

    def generate_single_solution(self) -> List[CourseSession]:
        """
        Generate a single feasible solution quickly.

        Returns:
            List of CourseSession objects representing the schedule

        Raises:
            ValueError: If problem is infeasible
        """
        import threading

        console.print(
            "\n[bold cyan]═══ CP-SAT Solver (Single Solution) ═══[/bold cyan]\n"
        )

        start_time = time.time()

        # Build model
        builder = ModelBuilder(self.context, self.qts)
        model, session_vars, var_factory = builder.build_model()

        # Configure solver for single solution
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.log_search_progress = True

        if self.random_seed is not None:
            solver.parameters.random_seed = self.random_seed

        # Solve
        console.print("[bold]Searching for feasible solution...[/bold]")
        console.print(f"Time limit: {self.time_limit} seconds")
        console.print(f"[dim]Solver will report progress periodically...[/dim]\n")

        # Progress monitoring
        stop_monitoring = threading.Event()

        def monitor_progress():
            """Print periodic status updates"""
            while not stop_monitoring.is_set():
                stop_monitoring.wait(10)  # Check every 10 seconds
                if not stop_monitoring.is_set():
                    elapsed = time.time() - start_time
                    console.print(
                        f"  [dim]⏳ Still searching... {elapsed:.0f}s elapsed[/dim]"
                    )

        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        try:
            status = solver.Solve(model)
        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=1)

        elapsed_time = time.time() - start_time

        # Process results
        console.print(f"\n[bold]Solver Status:[/bold] {solver.StatusName(status)}")
        console.print(f"[bold]Solve Time:[/bold] {elapsed_time:.2f}s")
        console.print(f"[bold]Branches:[/bold] {solver.NumBranches()}")
        console.print(f"[bold]Conflicts:[/bold] {solver.NumConflicts()}")

        if status == cp_model.INFEASIBLE:
            console.print("\n[bold red]✗ Problem is INFEASIBLE[/bold red]")
            raise ValueError("Problem is infeasible")

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            console.print("\n[bold yellow]⚠ No solution found[/bold yellow]")
            raise ValueError("No solution found within time limit")

        # Decode solution
        sessions = decode_cp_solution(solver, session_vars, var_factory, self.context)
        merged_sessions = merge_consecutive_sessions(sessions)

        console.print(
            f"\n[bold green]✓ Found feasible solution with {len(merged_sessions)} sessions[/bold green]\n"
        )

        return merged_sessions
