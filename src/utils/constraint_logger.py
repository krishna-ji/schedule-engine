"""
Constraint Logger Module

Logs detailed constraint breakdowns, diversity metrics, and GA events to CSV.
Separate from run.log - provides granular per-generation constraint analysis.

Features:
- Detailed hard/soft constraint breakdown per generation
- Diversity metrics tracking
- Advanced metrics (hypervolume, spacing, IGD, spread)
- Event logging (repair, hypermutation, stagnation, etc.)
- **Enhanced repair tracking:** individuals repaired, crossover/mutation/memetic counts
- Timing metrics (time per generation)
- Crash-safe: Flushes after each generation (no data loss on crash)
- CSV format for easy analysis in Excel/Python

Output: data/metrics.csv in output directory

CSV Columns:
- generation: Generation number (INIT for initial population)
- hard_total: Sum of all hard constraint violations
- soft_total: Sum of all soft constraint penalties
- hard_<constraint_name>: Individual hard constraint values
- soft_<constraint_name>: Individual soft constraint values
- diversity: Population diversity metric (0-1)
- hypervolume: NSGA-II quality indicator
- spacing: Solution distribution uniformity
- igd: Inverted Generational Distance
- spread: Solution spread metric
- time_seconds: Time taken for this generation
- repairs_total: Total number of repairs performed
- repairs_individuals_count: Number of individuals that received repairs
- repairs_crossover_count: Total repairs applied after crossover
- repairs_mutation_count: Total repairs applied after mutation
- repairs_memetic_count: Total repairs from memetic local search
- repairs_<type>: Breakdown by repair type (availability, overlap, room, etc.)
- events: Semicolon-separated list of events (repair, stagnation, hypermutation, etc.)
- notes: Optional notes (e.g., "Initial population", "Perfect solution")
"""

import os
import csv
from typing import Dict, Optional, List


class ConstraintLogger:
    """
    Logger for detailed per-generation constraint analysis.

    Logs to CSV file with columns:
    - Generation number
    - Total hard violations
    - Total soft penalty
    - Individual hard constraint values (one column per constraint)
    - Individual soft constraint values (one column per constraint)
    - Diversity metric
    - Time per generation (seconds)
    - Events (repair, hypermutation, stagnation, etc.)
    - Repair statistics breakdown
    """

    def __init__(
        self,
        output_dir: str,
        hard_constraint_names: List[str],
        soft_constraint_names: List[str],
    ):
        """
        Initialize constraint logger.

        Args:
            output_dir: Directory to write data/metrics.csv
            hard_constraint_names: List of enabled hard constraint names
            soft_constraint_names: List of enabled soft constraint names
        """
        self.output_dir = output_dir
        # Create data/ subdirectory for CSV files
        data_dir = os.path.join(output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        self.log_path = os.path.join(data_dir, "metrics.csv")
        self.hard_names = hard_constraint_names
        self.soft_names = soft_constraint_names

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize CSV file with header
        self._write_header()

    def _write_header(self):
        """Write CSV header with all constraint columns."""
        # Build column names dynamically based on enabled constraints
        columns = [
            "generation",
            "hard_total",
            "soft_total",
        ]

        # Add individual hard constraint columns
        for name in self.hard_names:
            columns.append(f"hard_{name}")

        # Add individual soft constraint columns
        for name in self.soft_names:
            columns.append(f"soft_{name}")

        # Add metrics and event columns
        columns.extend(
            [
                "diversity",
                "hypervolume",  # NEW: NSGA-II quality metric
                "spacing",  # NEW: Solution distribution metric
                "igd",  # NEW: Inverted Generational Distance
                "spread",  # NEW: Solution spread metric
                "time_seconds",
                # Repair totals
                "repairs_total",
                "repairs_individuals_count",  # NEW: How many individuals were repaired
                "repairs_crossover_count",  # NEW: Repairs after crossover
                "repairs_mutation_count",  # NEW: Repairs after mutation
                "repairs_memetic_count",  # NEW: Repairs from memetic search
                # Repair breakdowns by type
                "repairs_instructor_availability",
                "repairs_overlap",
                "repairs_room",
                "repairs_instructor_conflict",
                "repairs_qualification",
                "repairs_room_type",
                "repairs_clustering",
                "repairs_session_count",
                "events",  # Comma-separated list of events in this generation
                "notes",
            ]
        )

        # Write header
        with open(self.log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    def log_generation(
        self,
        generation: int,
        hard_total: float,
        soft_total: float,
        hard_breakdown: Dict[str, float],
        soft_breakdown: Dict[str, float],
        diversity: float,
        time_seconds: float,
        hypervolume: float = 0.0,
        spacing: float = 0.0,
        igd: float = 0.0,
        spread: float = 0.0,
        repair_stats: Optional[Dict[str, int]] = None,
        events: Optional[List[str]] = None,
        notes: str = "",
    ):
        """
        Log constraint data for a single generation.

        Args:
            generation: Generation number (-1 for initial population, 0+ for evolved)
            hard_total: Total hard constraint violations
            soft_total: Total soft constraint penalty
            hard_breakdown: Dict mapping constraint names to individual values
            soft_breakdown: Dict mapping constraint names to individual values
            diversity: Population diversity metric
            time_seconds: Time taken for this generation
            hypervolume: Hypervolume indicator (NSGA-II quality metric)
            spacing: Spacing metric (solution distribution)
            igd: Inverted Generational Distance
            spread: Solution spread metric
            repair_stats: Optional dict with repair statistics (from GAScheduler.metrics.repair_stats)
            events: Optional list of event strings (e.g., ["repair", "stagnation", "hypermutation"])
            notes: Optional notes string
        """
        # Build row data
        row = [
            generation if generation >= 0 else "INIT",
            f"{hard_total:.1f}",
            f"{soft_total:.2f}",
        ]

        # Add hard constraint breakdown (in order of self.hard_names)
        for name in self.hard_names:
            value = hard_breakdown.get(name, 0)
            row.append(f"{value:.1f}")

        # Add soft constraint breakdown (in order of self.soft_names)
        for name in self.soft_names:
            value = soft_breakdown.get(name, 0)
            row.append(f"{value:.2f}")

        # Add diversity and timing
        row.append(f"{diversity:.4f}")
        row.append(f"{hypervolume:.6f}")
        row.append(f"{spacing:.6f}")
        row.append(f"{igd:.6f}")
        row.append(f"{spread:.6f}")
        row.append(f"{time_seconds:.3f}")

        # Add repair statistics
        if repair_stats is None:
            repair_stats = {}

        row.append(str(repair_stats.get("total_fixes", 0)))
        row.append(str(repair_stats.get("individuals_repaired", 0)))  # NEW
        row.append(str(repair_stats.get("crossover_repairs", 0)))  # NEW
        row.append(str(repair_stats.get("mutation_repairs", 0)))  # NEW
        row.append(str(repair_stats.get("memetic_repairs", 0)))  # NEW
        row.append(str(repair_stats.get("instructor_availability_fixes", 0)))
        row.append(str(repair_stats.get("overlap_fixes", 0)))
        row.append(str(repair_stats.get("room_fixes", 0)))
        row.append(str(repair_stats.get("instructor_conflict_fixes", 0)))
        row.append(str(repair_stats.get("qualification_fixes", 0)))
        row.append(str(repair_stats.get("room_type_fixes", 0)))
        row.append(str(repair_stats.get("clustering_fixes", 0)))
        row.append(str(repair_stats.get("session_count_fixes", 0)))

        # Add events (comma-separated)
        if events:
            row.append("; ".join(events))
        else:
            row.append("")

        # Add notes
        row.append(notes)

        # Write row to CSV (append mode)
        # IMPORTANT: Flush immediately for crash safety
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            f.flush()  # Force write to disk (crash-safe)

    def get_log_path(self) -> str:
        """Return the path to the constraint log CSV file."""
        return self.log_path

    def update_last_generation_time(self, time_seconds: float):
        """
        Update the time_seconds for the most recently logged generation.

        This is a workaround because time is tracked in the evolve() loop
        but logging happens in _track_metrics() which doesn't have access to timing.

        NOTE: This is not truly crash-safe since it requires re-reading and rewriting
        the entire CSV file. However, the main constraint data is already safely written,
        so this only affects the timing column.

        Args:
            time_seconds: Time taken for the generation
        """
        import shutil

        # Read all rows
        try:
            with open(self.log_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                rows = list(reader)

            if len(rows) <= 1:  # Only header or empty
                return

            # Update time in last row (column index depends on structure)
            # Find time_seconds column index
            header = rows[0]
            try:
                time_idx = header.index("time_seconds")
            except ValueError:
                return  # Column not found

            # Update last row's time
            rows[-1][time_idx] = f"{time_seconds:.3f}"

            # Write to temporary file first (safer)
            temp_path = self.log_path + ".tmp"
            with open(temp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                f.flush()

            # Replace original with temp file
            shutil.move(temp_path, self.log_path)

        except Exception:
            # Silently fail - timing update is not critical
            pass


class EventTracker:
    """
    Helper class to track events during a generation.

    Events tracked (logged in events column of logger_all.csv):

    **Repair Events:**
    - `crossover_repair_applied` - Repair applied after crossover operations
    - `mutation_repair_applied` - Repair applied after mutation operations
    - `memetic_repair_applied` - Memetic local search applied to elite individuals
    - `periodic_repair` - Regular periodic repair triggered (every N generations)
    - `stagnation_repair` - Repair triggered due to stagnation detection
    - `intensive_repair` - High-intensity repair triggered (every M generations)

    **Stagnation Events:**
    - `stagnation_detected` - No improvement detected for X consecutive generations

    **Hypermutation Events:**
    - `hypermutation_start` - Hypermutation activated (increased mutation rate)
    - `hypermutation_active` - Hypermutation is currently active
    - `hypermutation_ended` - Hypermutation period ended, returning to normal

    **Population Events:**
    - `population_restart` - Partial population restart due to prolonged stagnation

    **Solution Events:**
    - `perfect_solution` - Perfect solution found (hard violations = 0)

    **Format:** Events are semicolon-separated in the CSV (e.g., "stagnation_detected; hypermutation_start; crossover_repair_applied")
    """

    def __init__(self):
        self.events: List[str] = []

    def add(self, event: str):
        """Add an event to the tracker."""
        self.events.append(event)

    def has_events(self) -> bool:
        """Check if any events were recorded."""
        return len(self.events) > 0

    def get_events(self) -> List[str]:
        """Get list of events."""
        return self.events

    def clear(self):
        """Clear all events."""
        self.events.clear()
