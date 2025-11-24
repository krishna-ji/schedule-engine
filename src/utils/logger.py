"""
Runtime Logger Module

Logs generation-by-generation metrics and configuration to output/run.log.
Provides detailed runtime analysis for each GA run.
"""

import os
from datetime import datetime
from typing import Dict, List


class GALogger:
    """
    Logger for GA execution metrics.

    Logs:
    - Configuration parameters (population, generations, probabilities, etc.)
    - Per-generation metrics (hard violations, soft penalties, time, diversity)
    - Runtime statistics (total time, avg time per generation, etc.)
    """

    def __init__(self, output_dir: str, config: Dict):
        """
        Initialize logger.

        Args:
            output_dir: Directory to write run.log
            config: Configuration dictionary with GA parameters
        """
        self.output_dir = output_dir
        self.log_path = os.path.join(output_dir, "run.log")
        self.config = config
        self.generation_logs: List[Dict] = []
        self.start_time = None
        self.end_time = None

        # Ensure output directory exists (os.makedirs creates parents by default)
        os.makedirs(output_dir, exist_ok=True)

        # Initialize log file with header
        self._write_header()

    def _write_header(self):
        """Write configuration and header to log file."""
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("SCHEDULE ENGINE - GENETIC ALGORITHM RUN LOG\n")
            f.write("=" * 80 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Output Directory: {self.output_dir}\n")
            f.write("\n")

            # Configuration Section
            f.write("-" * 80 + "\n")
            f.write("CONFIGURATION\n")
            f.write("-" * 80 + "\n")

            # GA Parameters
            f.write(f"Population Size:        {self.config.get('pop_size', 'N/A')}\n")
            f.write(
                f"Generations:            {self.config.get('generations', 'N/A')}\n"
            )
            f.write(
                f"Crossover Probability:  {self.config.get('crossover_prob', 'N/A')}\n"
            )
            f.write(
                f"Mutation Probability:   {self.config.get('mutation_prob', 'N/A')}\n"
            )
            f.write(f"Random Seed:            {self.config.get('seed', 'N/A')}\n")
            f.write(
                f"Multiprocessing:        {self.config.get('use_multiprocessing', 'N/A')}\n"
            )
            f.write(
                f"Worker Processes:       {self.config.get('num_workers', 'N/A')}\n"
            )
            f.write("\n")

            # Population Strategy
            f.write(
                f"Population Strategy:    {self.config.get('population_strategy', 'N/A')}\n"
            )
            f.write(
                f"Adaptive Operators:     {self.config.get('adaptive_operators', 'N/A')}\n"
            )
            f.write(
                f"Elite Preservation:     {self.config.get('elite_preservation', 'N/A')}\n"
            )
            f.write(f"Elite Size:             {self.config.get('elite_size', 'N/A')}\n")
            f.write("\n")

            # Constraints
            f.write(
                f"Hard Constraints:       {self.config.get('num_hard_constraints', 'N/A')}\n"
            )
            f.write(
                f"Soft Constraints:       {self.config.get('num_soft_constraints', 'N/A')}\n"
            )
            f.write("\n")

            # Repair Configuration
            repair_enabled = self.config.get("repair_enabled", False)
            f.write(
                f"Repair Heuristics:      {'Enabled' if repair_enabled else 'Disabled'}\n"
            )
            if repair_enabled:
                f.write(
                    f"  Max Iterations:       {self.config.get('repair_max_iterations', 'N/A')}\n"
                )
                f.write(
                    f"  After Mutation:       {self.config.get('repair_after_mutation', 'N/A')}\n"
                )
                f.write(
                    f"  After Crossover:      {self.config.get('repair_after_crossover', 'N/A')}\n"
                )
                f.write(
                    f"  Memetic Mode:         {self.config.get('repair_memetic_mode', 'N/A')}\n"
                )
                if self.config.get("repair_memetic_mode"):
                    f.write(
                        f"  Memetic Iterations:   {self.config.get('repair_memetic_iterations', 'N/A')}\n"
                    )
            f.write("\n")

            # Data Statistics
            f.write(
                f"Courses:                {self.config.get('num_courses', 'N/A')}\n"
            )
            f.write(f"Groups:                 {self.config.get('num_groups', 'N/A')}\n")
            f.write(
                f"Instructors:            {self.config.get('num_instructors', 'N/A')}\n"
            )
            f.write(f"Rooms:                  {self.config.get('num_rooms', 'N/A')}\n")
            f.write(f"Time Quanta:            {self.config.get('num_quanta', 'N/A')}\n")
            f.write("\n")

            # Generation Log Header
            f.write("=" * 80 + "\n")
            f.write("GENERATION LOG\n")
            f.write("=" * 80 + "\n")
            f.write(
                f"{'Gen':<6} {'Hard':<8} {'Soft':<10} {'Time(s)':<8} {'Diversity':<10} {'Repairs':<8} {'Notes'}\n"
            )
            f.write("-" * 80 + "\n")

    def log_generation(
        self,
        generation: int,
        hard_violations: float,
        soft_penalty: float,
        time_seconds: float,
        diversity: float,
        repairs: int = 0,
        notes: str = "",
    ):
        """
        Log metrics for a single generation.

        Args:
            generation: Generation number (-1 for initial population)
            hard_violations: Best hard constraint violations
            soft_penalty: Best soft constraint penalty
            time_seconds: Time taken for this generation (seconds)
            diversity: Population diversity metric
            repairs: Number of repairs performed
            notes: Optional notes (e.g., "perfect solution", "early stop")
        """
        # Store for later analysis
        self.generation_logs.append(
            {
                "generation": generation,
                "hard_violations": hard_violations,
                "soft_penalty": soft_penalty,
                "time_seconds": time_seconds,
                "diversity": diversity,
                "repairs": repairs,
                "notes": notes,
            }
        )

        # Append to log file
        with open(self.log_path, "a", encoding="utf-8") as f:
            gen_str = "INIT" if generation == -1 else str(generation + 1)
            f.write(
                f"{gen_str:<6} {hard_violations:<8.0f} {soft_penalty:<10.2f} "
                f"{time_seconds:<8.3f} {diversity:<10.4f} {repairs:<8} {notes}\n"
            )

    def start_run(self):
        """Mark the start of the GA run."""
        self.start_time = datetime.now()

    def end_run(self, best_hard: float, best_soft: float, final_schedule_sessions: int):
        """
        Mark the end of the GA run and write summary.

        Args:
            best_hard: Final best hard violations
            best_soft: Final best soft penalty
            final_schedule_sessions: Number of sessions in final schedule
        """
        self.end_time = datetime.now()
        total_time = (self.end_time - self.start_time).total_seconds()

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("RUN SUMMARY\n")
            f.write("=" * 80 + "\n")

            # Time statistics
            f.write(
                f"Total Runtime:          {total_time:.2f}s ({total_time/60:.2f} minutes)\n"
            )

            if self.generation_logs:
                # Filter out initial population (gen=-1) from timing stats
                evolution_logs = [
                    log for log in self.generation_logs if log["generation"] >= 0
                ]

                if evolution_logs:
                    avg_gen_time = sum(
                        log["time_seconds"] for log in evolution_logs
                    ) / len(evolution_logs)
                    min_gen_time = min(log["time_seconds"] for log in evolution_logs)
                    max_gen_time = max(log["time_seconds"] for log in evolution_logs)

                    f.write(f"Generations Completed:  {len(evolution_logs)}\n")
                    f.write(f"Avg Time per Gen:       {avg_gen_time:.3f}s\n")
                    f.write(f"Min Time per Gen:       {min_gen_time:.3f}s\n")
                    f.write(f"Max Time per Gen:       {max_gen_time:.3f}s\n")

                # Initial vs Final comparison
                initial = self.generation_logs[0]  # gen=-1 (initial pop)
                final = self.generation_logs[-1]  # last generation

                f.write("\n")
                f.write(f"Initial Hard Violations: {initial['hard_violations']:.0f}\n")
                f.write(f"Final Hard Violations:   {final['hard_violations']:.0f}\n")

                if initial["hard_violations"] > 0:
                    improvement = (
                        (initial["hard_violations"] - final["hard_violations"])
                        / initial["hard_violations"]
                        * 100
                    )
                    f.write(f"Hard Improvement:        {improvement:.1f}%\n")

                f.write("\n")
                f.write(f"Initial Soft Penalty:    {initial['soft_penalty']:.2f}\n")
                f.write(f"Final Soft Penalty:      {final['soft_penalty']:.2f}\n")

                if initial["soft_penalty"] > 0:
                    improvement = (
                        (initial["soft_penalty"] - final["soft_penalty"])
                        / initial["soft_penalty"]
                        * 100
                    )
                    f.write(f"Soft Improvement:        {improvement:.1f}%\n")

                # Diversity analysis
                f.write("\n")
                f.write(f"Initial Diversity:       {initial['diversity']:.4f}\n")
                f.write(f"Final Diversity:         {final['diversity']:.4f}\n")

                # Repair statistics
                total_repairs = sum(log["repairs"] for log in self.generation_logs)
                f.write("\n")
                f.write(f"Total Repairs:           {total_repairs}\n")

            # Final solution quality
            f.write("\n")
            f.write(f"Final Schedule Sessions: {final_schedule_sessions}\n")
            f.write(f"Final Hard Violations:   {best_hard:.0f}\n")
            f.write(f"Final Soft Penalty:      {best_soft:.2f}\n")

            if best_hard == 0:
                f.write(
                    "\n[[!ok]]FEASIBLE SOLUTION FOUND (No hard constraint violations)\n"
                )
            else:
                f.write("\nINFEASIBLE SOLUTION (Hard constraints violated)\n")

            f.write("\n")
            f.write(
                f"Log completed at: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("=" * 80 + "\n")

    def get_log_path(self) -> str:
        """Return the path to the log file."""
        return self.log_path
