"""
Experiment manager for organizing outputs and tracking experiments.

Provides structured output organization by experiment ID, automatic
experiment logging, and comparison tools for research workflows.
"""

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.table import Table

from src.utils.console_service import get_console

console = get_console()

# Known experiment IDs and their display names
EXPERIMENTS = {
    "a": "Experiment A: Pure NSGA-II",
    "b": "Experiment B: Memetic",
    "c": "Experiment C: Round-Robin",
    "d": "Experiment D: Adaptive",
    "e": "Experiment E: RL-Guided",
}


@dataclass
class ExperimentRun:
    """
    Metadata for a single experiment run.

    Tracks all relevant information for reproducibility and comparison.
    """

    run_id: str  # Unique identifier (timestamp-based)
    runtime_mode: str  # Experiment ID (e.g., "a", "b", "c", "d", "e")
    config_reference: str  # Identifier for the config/blueprint used
    output_path: str  # Path to output directory
    seed: int  # Random seed
    timestamp: str  # ISO format timestamp
    experiment_name: str | None = None  # User-provided name
    duration_seconds: float | None = None  # Total runtime
    generations: int | None = None  # Number of generations
    population_size: int | None = None  # Population size
    best_hard_violations: float | None = None  # Best hard constraint violations
    best_soft_penalty: float | None = None  # Best soft penalty
    final_hypervolume: float | None = None  # Final hypervolume (if available)
    notes: str | None = None  # Optional notes

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRun":
        """Load from dictionary with backward compatibility for legacy fields."""
        data = data.copy()

        # Handle legacy field names
        if "config_path" in data:
            data["config_reference"] = data.pop("config_path")

        # Set defaults for new fields
        data.setdefault("config_reference", "legacy")
        data.setdefault("experiment_name", None)

        # Remove any unknown fields
        valid_fields = {
            "run_id",
            "runtime_mode",
            "config_reference",
            "output_path",
            "seed",
            "timestamp",
            "experiment_name",
            "duration_seconds",
            "generations",
            "population_size",
            "best_hard_violations",
            "best_soft_penalty",
            "final_hypervolume",
            "notes",
        }
        data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**data)

    @property
    def is_complete(self) -> bool:
        """Check if run has complete results (not just registered)."""
        return (
            self.duration_seconds is not None
            and self.best_hard_violations is not None
            and self.best_soft_penalty is not None
        )


class ExperimentManager:
    """
    Manages experiment runs with automatic output organization.

    Features:
    - Structured output directories by runtime mode
    - Automatic experiment logging (manifest.json)
    - Comparison tools for benchmarking
    - Easy retrieval of past runs
    """

    def __init__(self, base_output_dir: str = "output"):
        """
        Initialize experiment manager.

        Args:
            base_output_dir: Base directory for all experiment outputs
        """
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.base_dir / "experiment_manifest.json"
        self._load_manifest()

    def _load_manifest(self) -> None:
        """Load experiment manifest from disk."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    content = f.read().strip()
                    if not content:
                        # Empty file - initialize with empty manifest
                        self.runs = []
                        self._save_manifest()  # Create valid JSON
                    else:
                        data = json.loads(content)
                        self.runs = [
                            ExperimentRun.from_dict(r) for r in data.get("runs", [])
                        ]
            except (json.JSONDecodeError, ValueError):
                # Corrupted manifest - backup and reinitialize
                backup_path = self.manifest_path.with_suffix(".json.backup")
                self.manifest_path.rename(backup_path)
                console.print(
                    f"[yellow]️  Corrupted manifest backed up to {backup_path}[/yellow]"
                )
                self.runs = []
                self._save_manifest()  # Create fresh manifest
        else:
            self.runs = []
            self._save_manifest()  # Create manifest if it doesn't exist

    def _save_manifest(self) -> None:
        """Save experiment manifest to disk."""
        data = {"runs": [run.to_dict() for run in self.runs], "version": "1.0"}
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def create_output_dir(
        self,
        runtime_mode: str,
        experiment_name: str | None = None,
        timestamp: datetime | None = None,
        output_subdir: str | None = None,
    ) -> Path:
        """
        Create structured output directory for experiment.

        Directory structure:
        output/
          {mode_category}/           # e.g., baseline, nsga, rl, hybrid
            {mode_name}/             # e.g., pure-nsga, nsga-full, rl-guided
              evaluation_{timestamp}_{exp_name}/

        Args:
            runtime_mode: Runtime mode enum
            experiment_name: Optional experiment name
            timestamp: Optional timestamp (defaults to now)
            output_subdir: Optional custom subdirectory (e.g., "f-construction")

        Returns:
            Path to created output directory
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Parse mode value
        mode_value = runtime_mode  # e.g., "a", "b", "c", "d", "e"

        # Build clean folder name for experiment
        # Experiment IDs: a, b, c, d, e, f
        folder_map = {
            "a": "a-baseline-nsga-only",
            "b": "b-nsga-memetic",
            "c": "c-roundrobin",
            "d": "d-adaptive",
            "e": "e-rl-guided",
            "e5": "e5-rl-training",
            "f": "f-heuristic-testing",
        }

        # Use custom subdirectory if provided, otherwise use mode folder
        if output_subdir:
            mode_folder = output_subdir
        else:
            mode_folder = folder_map.get(mode_value, f"experiment-{mode_value}")

        # Build directory path (flat structure)
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        dir_name = f"evaluation_{timestamp_str}"
        if experiment_name:
            dir_name = f"{dir_name}_{experiment_name}"

        output_path = self.base_dir / mode_folder / dir_name

        output_path.mkdir(parents=True, exist_ok=True)

        return output_path

    def register_run(
        self,
        runtime_mode: str,
        config_reference: str,
        output_path: Path,
        experiment_name: str | None = None,
        seed: int = 69,
        notes: str | None = None,
    ) -> ExperimentRun:
        """
        Register a new experiment run.

        Args:
            runtime_mode: Runtime mode used
            config_reference: Identifier for the config blueprint
            output_path: Path to output directory
            experiment_name: Optional experiment name
            seed: Random seed
            notes: Optional notes

        Returns:
            ExperimentRun object
        """
        timestamp = datetime.now()
        run_id = timestamp.strftime("%Y%m%d_%H%M%S")

        run = ExperimentRun(
            run_id=run_id,
            runtime_mode=runtime_mode,
            experiment_name=experiment_name,
            config_reference=config_reference,
            output_path=str(output_path),
            seed=seed,
            timestamp=timestamp.isoformat(),
            notes=notes,
        )

        self.runs.append(run)
        self._save_manifest()

        return run

    def update_run_results(
        self,
        run: ExperimentRun,
        duration_seconds: float | None = None,
        generations: int | None = None,
        population_size: int | None = None,
        best_hard_violations: float | None = None,
        best_soft_penalty: float | None = None,
        final_hypervolume: float | None = None,
    ) -> None:
        """
        Update experiment run with results.

        Args:
            run: ExperimentRun object to update
            duration_seconds: Total runtime
            generations: Number of generations
            population_size: Population size
            best_hard_violations: Best hard constraint violations
            best_soft_penalty: Best soft penalty
            final_hypervolume: Final hypervolume indicator
        """
        if duration_seconds is not None:
            run.duration_seconds = duration_seconds
        if generations is not None:
            run.generations = generations
        if population_size is not None:
            run.population_size = population_size
        if best_hard_violations is not None:
            run.best_hard_violations = best_hard_violations
        if best_soft_penalty is not None:
            run.best_soft_penalty = best_soft_penalty
        if final_hypervolume is not None:
            run.final_hypervolume = final_hypervolume

        self._save_manifest()

    def get_runs_for_mode(
        self, runtime_mode: str, complete_only: bool = False
    ) -> list[ExperimentRun]:
        """
        Get all runs for a specific runtime mode.

        Args:
            runtime_mode: Runtime mode to filter by
            complete_only: If True, only return runs with complete results

        Returns:
            List of ExperimentRun objects
        """
        runs = [r for r in self.runs if r.runtime_mode == runtime_mode]
        if complete_only:
            runs = [r for r in runs if r.is_complete]
        return runs

    def get_complete_runs(self) -> list[ExperimentRun]:
        """Get all runs with complete results."""
        return [r for r in self.runs if r.is_complete]

    def get_incomplete_runs(self) -> list[ExperimentRun]:
        """Get all runs with incomplete results."""
        return [r for r in self.runs if not r.is_complete]

    def archive_incomplete_runs(self) -> int:
        """
        Archive incomplete runs to separate file and remove from main manifest.

        This cleans up the manifest by moving runs that were started but never
        completed (missing duration/fitness data) to an archive file.

        Returns:
            Number of runs archived
        """
        incomplete = self.get_incomplete_runs()
        if not incomplete:
            console.print("[green]No incomplete runs to archive.[/green]")
            return 0

        # Save incomplete runs to archive
        archive_path = self.base_dir / "experiment_manifest_incomplete.json"
        archive_data = {"runs": [run.to_dict() for run in incomplete], "version": "1.0"}

        with open(archive_path, "w") as f:
            json.dump(archive_data, f, indent=2)

        # Keep only complete runs in main manifest
        self.runs = self.get_complete_runs()
        self._save_manifest()

        console.print(
            f"[yellow]Archived {len(incomplete)} incomplete runs to:[/yellow] {archive_path}"
        )
        console.print(
            f"[green]Main manifest now has {len(self.runs)} complete runs.[/green]"
        )

        return len(incomplete)

    def get_best_run(self, runtime_mode: str | None = None) -> ExperimentRun | None:
        """
        Get most recent experiment run.

        Args:
            runtime_mode: Optional runtime mode filter

        Returns:
            Latest ExperimentRun or None
        """
        runs = self.get_runs_for_mode(runtime_mode) if runtime_mode else self.runs
        if not runs:
            return None
        return max(runs, key=lambda r: r.timestamp)

    def compare_modes(self, modes: list[str] | None = None, top_n: int = 5) -> Table:
        """
        Generate comparison table for experiments.

        Args:
            modes: Optional list of experiment IDs to compare (defaults to all)
            top_n: Number of recent runs per experiment to include

        Returns:
            Rich Table object for display
        """
        if modes is None:
            modes = list(EXPERIMENTS.keys())

        table = Table(title="Experiment Comparison")
        table.add_column("Experiment", style="cyan")
        table.add_column("Runs", justify="right")
        table.add_column("Best Hard", justify="right", style="green")
        table.add_column("Best Soft", justify="right", style="yellow")
        table.add_column("Avg Duration", justify="right")
        table.add_column("Latest", style="dim")

        for mode in modes:
            runs = self.get_runs_for_mode(mode)
            display_name = EXPERIMENTS.get(mode, mode)
            if not runs:
                table.add_row(display_name, "0", "-", "-", "-", "-")
                continue

            # Recent runs
            recent = sorted(runs, key=lambda r: r.timestamp, reverse=True)[:top_n]

            # Best results
            valid_hard = [
                r.best_hard_violations
                for r in runs
                if r.best_hard_violations is not None
            ]
            valid_soft = [
                r.best_soft_penalty for r in runs if r.best_soft_penalty is not None
            ]
            valid_duration = [
                r.duration_seconds for r in runs if r.duration_seconds is not None
            ]

            best_hard = f"{min(valid_hard):.1f}" if valid_hard else "-"
            best_soft = f"{min(valid_soft):.2f}" if valid_soft else "-"
            avg_duration = (
                f"{sum(valid_duration) / len(valid_duration):.1f}s"
                if valid_duration
                else "-"
            )

            latest = recent[0].timestamp.split("T")[0] if recent else "-"

            table.add_row(
                display_name,
                str(len(runs)),
                best_hard,
                best_soft,
                avg_duration,
                latest,
            )

        return table

    def get_manifest_stats(self) -> dict[str, Any]:
        """
        Get statistics about the current manifest.

        Returns:
            Dictionary with manifest statistics
        """
        complete = self.get_complete_runs()
        incomplete = self.get_incomplete_runs()

        stats = {
            "total_runs": len(self.runs),
            "complete_runs": len(complete),
            "incomplete_runs": len(incomplete),
            "completion_rate": (
                f"{len(complete) / len(self.runs) * 100:.1f}%" if self.runs else "N/A"
            ),
        }

        # Per-experiment statistics
        mode_stats = {}
        for exp_id in EXPERIMENTS:
            mode_runs = self.get_runs_for_mode(exp_id)
            if mode_runs:
                mode_complete = [r for r in mode_runs if r.is_complete]
                mode_stats[exp_id] = {
                    "total": len(mode_runs),
                    "complete": len(mode_complete),
                    "incomplete": len(mode_runs) - len(mode_complete),
                }

        stats["by_mode"] = mode_stats
        return stats

    def print_manifest_stats(self) -> None:
        """Print manifest statistics to console."""
        stats = self.get_manifest_stats()

        console.print("\n[bold cyan]Manifest Statistics[/bold cyan]")
        console.print(f"  Total runs: {stats['total_runs']}")
        console.print(f"  Complete: {stats['complete_runs']} [green]✓[/green]")
        console.print(f"  Incomplete: {stats['incomplete_runs']} [yellow]![/yellow]")
        console.print(f"  Completion rate: {stats['completion_rate']}")

        if stats["incomplete_runs"] > 0:
            console.print(
                "\n[yellow]Tip:[/yellow] Run [cyan]manager.archive_incomplete_runs()[/cyan] to clean manifest."
            )

    def export_comparison_csv(
        self, output_path: Path, modes: list[str] | None = None
    ) -> None:
        """
        Export comparison data to CSV for analysis.

        Args:
            output_path: Path to CSV file
            modes: Optional list of experiment IDs to include (defaults to all)
        """
        import csv

        if modes is None:
            modes = list(EXPERIMENTS.keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "run_id",
                    "runtime_mode",
                    "experiment_name",
                    "timestamp",
                    "generations",
                    "population_size",
                    "best_hard_violations",
                    "best_soft_penalty",
                    "duration_seconds",
                    "final_hypervolume",
                ]
            )

            for mode in modes:
                runs = self.get_runs_for_mode(mode)
                for run in runs:
                    writer.writerow(
                        [
                            run.run_id,
                            run.runtime_mode,
                            run.experiment_name or "",
                            run.timestamp,
                            run.generations or "",
                            run.population_size or "",
                            run.best_hard_violations or "",
                            run.best_soft_penalty or "",
                            run.duration_seconds or "",
                            run.final_hypervolume or "",
                        ]
                    )

        console.print(f"[green]Exported comparison data to {output_path}[/green]")

    def cleanup_old_runs(
        self, keep_last_n: int = 10, runtime_mode: str | None = None
    ) -> None:
        """
        Clean up old experiment outputs to save disk space.

        Args:
            keep_last_n: Number of recent runs to keep per experiment
            runtime_mode: Optional experiment ID filter (defaults to all experiments)
        """
        modes = [runtime_mode] if runtime_mode else list(EXPERIMENTS.keys())

        for mode in modes:
            runs = sorted(
                self.get_runs_for_mode(mode), key=lambda r: r.timestamp, reverse=True
            )

            if len(runs) <= keep_last_n:
                continue

            # Delete old runs
            for run in runs[keep_last_n:]:
                output_path = Path(run.output_path)
                if output_path.exists():
                    shutil.rmtree(output_path)
                    mode_name = EXPERIMENTS.get(mode, mode)
                    console.print(
                        f"[dim]Deleted old run: {run.run_id} ({mode_name})[/dim]"
                    )

            # Remove from manifest
            self.runs = [r for r in self.runs if r not in runs[keep_last_n:]]

        self._save_manifest()
        console.print(
            f"[green]Cleaned old runs (kept last {keep_last_n} per mode)[/green]"
        )
