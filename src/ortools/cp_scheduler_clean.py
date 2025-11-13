"""
CP-SAT Scheduler - Pure Constraint Programming Implementation

Clean implementation with:
- Hard constraints only (no soft constraints)
- Proper logging with Python logging module
- Memory-safe parallel search
- Minimal presolve to avoid constraint explosion
"""

import logging
import time
import threading
from typing import List
from multiprocessing import cpu_count
from ortools.sat.python import cp_model

from src.core.types import SchedulingContext
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.entities.decoded_session import CourseSession
from src.ortools.model_builder import ModelBuilder
from src.ortools.solution_decoder import decode_cp_solution, merge_consecutive_sessions


logger = logging.getLogger(__name__)


class CPScheduler:
    """
    CP-SAT based scheduler for hard constraint satisfaction.

    Generates a single feasible schedule that satisfies all hard constraints.
    Uses Google OR-Tools CP-SAT solver with minimal presolve strategy.
    """

    def __init__(
        self,
        context: SchedulingContext,
        qts: QuantumTimeSystem,
        time_limit_seconds: int = 0,
        num_workers: int = 4,
        random_seed: int = None,
    ):
        """
        Initialize CP-SAT scheduler.

        Args:
            context: SchedulingContext with all entities
            qts: QuantumTimeSystem for time calculations
            time_limit_seconds: Maximum solve time (0 = unlimited)
            num_workers: Number of parallel search workers (default: 4)
            random_seed: Random seed for reproducibility
        """
        self.context = context
        self.qts = qts
        self.time_limit = time_limit_seconds
        self.num_workers = min(num_workers, cpu_count())
        self.random_seed = random_seed

    def generate_single_solution(self) -> List[CourseSession]:
        """
        Generate a single feasible solution using CP-SAT.

        Returns:
            List of CourseSession objects representing the schedule

        Raises:
            ValueError: If problem is infeasible or no solution found
        """
        start_time = time.time()

        # Build CP-SAT model
        logger.info("Building CP-SAT model...")
        build_start = time.time()

        builder = ModelBuilder(self.context, self.qts)
        model, session_vars, var_factory = builder.build_model()

        build_time = time.time() - build_start
        num_constraints = len(model.Proto().constraints)

        logger.info(f"Model built in {build_time:.2f}s")
        logger.info(f"  Variables: {len(session_vars) * 3}")
        logger.info(f"  Constraints: {num_constraints:,}")

        # Warn about large models
        if num_constraints > 1_000_000:
            logger.warning(f"Large model detected ({num_constraints:,} constraints)")
            logger.warning("Expected solve time: 1-12+ hours")

        # Configure solver
        solver = cp_model.CpSolver()

        # Time limit (0 = unlimited)
        if self.time_limit > 0:
            solver.parameters.max_time_in_seconds = self.time_limit
            logger.info(f"Time limit: {self.time_limit}s")
        else:
            logger.info("Time limit: UNLIMITED")

        # Parallel search workers (memory-safe limit)
        solver.parameters.num_search_workers = self.num_workers
        logger.info(f"Parallel workers: {self.num_workers}")

        # Minimal presolve to avoid constraint explosion
        solver.parameters.cp_model_presolve = True
        solver.parameters.symmetry_level = 0  # Disable (memory expensive)
        solver.parameters.linearization_level = (
            0  # Disable (creates auxiliary constraints)
        )
        solver.parameters.optimize_with_core = False  # Disable for large problems
        solver.parameters.max_presolve_iterations = 3  # Limit passes
        solver.parameters.log_search_progress = True  # Enable solver logging

        logger.info("Presolve: MINIMAL (avoid constraint explosion)")

        if self.random_seed is not None:
            solver.parameters.random_seed = self.random_seed
            logger.info(f"Random seed: {self.random_seed}")

        # Progress monitoring thread
        stop_monitoring = threading.Event()

        def monitor_progress():
            """Log periodic progress updates."""
            while not stop_monitoring.is_set():
                stop_monitoring.wait(60)  # Update every 60 seconds
                if not stop_monitoring.is_set():
                    elapsed = time.time() - start_time
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)

                    if hours > 0:
                        time_str = f"{hours}h {minutes}m"
                    else:
                        time_str = f"{minutes}m"

                    logger.info(f"Still searching... {time_str} elapsed")

        monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
        monitor_thread.start()

        # Solve
        logger.info("=" * 80)
        logger.info("Starting CP-SAT search...")
        logger.info("=" * 80)

        solve_start = time.time()

        try:
            status = solver.Solve(model)
        finally:
            stop_monitoring.set()
            monitor_thread.join(timeout=1)

        solve_time = time.time() - solve_start
        total_time = time.time() - start_time

        # Log results
        logger.info("=" * 80)
        logger.info(f"Solver status: {solver.StatusName(status)}")
        logger.info(f"Solve time: {solve_time:.2f}s ({solve_time/3600:.2f}h)")
        logger.info(f"Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
        logger.info(f"Branches: {solver.NumBranches():,}")
        logger.info(f"Conflicts: {solver.NumConflicts():,}")
        logger.info(f"Wall time: {solver.WallTime():.2f}s")
        logger.info("=" * 80)

        # Check status
        if status == cp_model.INFEASIBLE:
            logger.error("Problem is INFEASIBLE - no valid schedule exists")
            raise ValueError("Problem is infeasible")

        if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            logger.error("No solution found within time limit")
            raise ValueError("No solution found")

        # Decode solution
        logger.info("Decoding solution...")
        decode_start = time.time()

        sessions = decode_cp_solution(solver, session_vars, var_factory, self.context)
        merged_sessions = merge_consecutive_sessions(sessions)

        decode_time = time.time() - decode_start
        logger.info(f"Solution decoded in {decode_time:.2f}s")
        logger.info(f"Total sessions: {len(merged_sessions)}")

        logger.info("=" * 80)
        logger.info("[OK] SOLUTION FOUND SUCCESSFULLY")
        logger.info("=" * 80)

        return merged_sessions
