"""
Experiment manager for organizing outputs and tracking runtime modes.

Provides structured output organization by runtime mode, automatic
experiment logging, and comparison tools for research workflows.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from rich.table import Table

from src.config.runtime_mode import RuntimeMode
from src.utils.console_service import get_console

console = get_console()


@dataclass
class ExperimentRun:
    """
    Metadata for a single experiment run.

    Tracks all relevant information for reproducibility and comparison.
    """

    run_id: str  # Unique identifier (timestamp-based)
    runtime_mode: str  # RuntimeMode value (e.g., "1-pure-nsga")
    experiment_name: Optional[str]  # User-provided name
    config_path: str  # Path to config file used
    output_path: str  # Path to output directory
    seed: int  # Random seed
    timestamp: str  # ISO format timestamp
    duration_seconds: Optional[float] = None  # Total runtime
    generations: Optional[int] = None  # Number of generations
    population_size: Optional[int] = None  # Population size
    best_hard_violations: Optional[float] = None  # Best hard constraint violations
    best_soft_penalty: Optional[float] = None  # Best soft penalty
    final_hypervolume: Optional[float] = None  # Final hypervolume (if available)
    notes: Optional[str] = None  # Optional notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentRun":
        """Load from dictionary."""
        return cls(**data)


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

    def _load_manifest(self):
        """Load experiment manifest from disk."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                data = json.load(f)
                self.runs = [ExperimentRun.from_dict(r) for r in data.get("runs", [])]
        else:
            self.runs = []

    def _save_manifest(self):
        """Save experiment manifest to disk."""
        data = {"runs": [run.to_dict() for run in self.runs], "version": "1.0"}
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def create_output_dir(
        self,
        runtime_mode: RuntimeMode,
        experiment_name: Optional[str] = None,
        timestamp: Optional[datetime] = None,
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

        Returns:
            Path to created output directory
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Parse mode value
        mode_value = runtime_mode.value  # e.g., "1-pure-nsga" or "a-pure-nsga"
        mode_prefix = mode_value.split("-")[0]  # "1" or "a"
        
        # Build clean folder name with mode prefix
        # Progressive modes (a-e): Use simple descriptive names
        # Numbered modes (1-10): Use original structure with categories
        if mode_prefix in ["a", "b", "c", "d", "e"]:
            # Progressive thesis experiments - flat structure with mode prefix
            folder_map = {
                "a": "a-baseline-nsga-only",
                "b": "b-nsga-memetic",
                "c": "c-roundrobin",
                "d": "d-adaptive",
                "e": "e-rl-guided",
            }
            mode_folder = folder_map.get(mode_prefix, mode_value)
            
            # Build directory path (flat structure)
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            dir_name = f"evaluation_{timestamp_str}"
            if experiment_name:
                dir_name = f"{dir_name}_{experiment_name}"
            
            output_path = self.base_dir / mode_folder / dir_name
        else:
            # Numbered modes (1-10): Keep category-based structure
            mode_number, mode_name = mode_value.split("-", 1)
            category_map = {
                "1": "baseline",
                "2": "nsga",
                "3": "nsga",
                "4": "nsga",
                "5": "rl",
                "6": "hybrid",
                "7": "rl",
                "8": "hybrid",
                "9": "rl",
                "10": "rl",
            }
            category = category_map.get(mode_number, "other")
            
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            dir_name = f"evaluation_{timestamp_str}"
            if experiment_name:
                dir_name = f"{dir_name}_{experiment_name}"
            
            output_path = self.base_dir / category / mode_name / dir_name
        
        output_path.mkdir(parents=True, exist_ok=True)

        return output_path

    def register_run(
        self,
        runtime_mode: RuntimeMode,
        config_path: Path,
        output_path: Path,
        experiment_name: Optional[str] = None,
        seed: int = 69,
        notes: Optional[str] = None,
    ) -> ExperimentRun:
        """
        Register a new experiment run.

        Args:
            runtime_mode: Runtime mode used
            config_path: Path to config file
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
            runtime_mode=runtime_mode.value,
            experiment_name=experiment_name,
            config_path=str(config_path),
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
        duration_seconds: Optional[float] = None,
        generations: Optional[int] = None,
        population_size: Optional[int] = None,
        best_hard_violations: Optional[float] = None,
        best_soft_penalty: Optional[float] = None,
        final_hypervolume: Optional[float] = None,
    ):
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

    def get_runs_by_mode(self, runtime_mode: RuntimeMode) -> List[ExperimentRun]:
        """
        Get all runs for a specific runtime mode.

        Args:
            runtime_mode: Runtime mode to filter by

        Returns:
            List of ExperimentRun objects
        """
        return [r for r in self.runs if r.runtime_mode == runtime_mode.value]

    def get_latest_run(
        self, runtime_mode: Optional[RuntimeMode] = None
    ) -> Optional[ExperimentRun]:
        """
        Get most recent experiment run.

        Args:
            runtime_mode: Optional runtime mode filter

        Returns:
            Latest ExperimentRun or None
        """
        runs = self.get_runs_by_mode(runtime_mode) if runtime_mode else self.runs
        if not runs:
            return None
        return max(runs, key=lambda r: r.timestamp)

    def compare_modes(
        self, modes: Optional[List[RuntimeMode]] = None, top_n: int = 5
    ) -> Table:
        """
        Generate comparison table for runtime modes.

        Args:
            modes: Optional list of modes to compare (defaults to all)
            top_n: Number of recent runs per mode to include

        Returns:
            Rich Table object for display
        """
        if modes is None:
            modes = list(RuntimeMode)

        table = Table(title="Runtime Mode Comparison")
        table.add_column("Mode", style="cyan")
        table.add_column("Runs", justify="right")
        table.add_column("Best Hard", justify="right", style="green")
        table.add_column("Best Soft", justify="right", style="yellow")
        table.add_column("Avg Duration", justify="right")
        table.add_column("Latest", style="dim")

        for mode in modes:
            runs = self.get_runs_by_mode(mode)
            if not runs:
                table.add_row(mode.display_name, "0", "-", "-", "-", "-")
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
                mode.display_name,
                str(len(runs)),
                best_hard,
                best_soft,
                avg_duration,
                latest,
            )

        return table

    def export_comparison_csv(
        self, output_path: Path, modes: Optional[List[RuntimeMode]] = None
    ):
        """
        Export comparison data to CSV for analysis.

        Args:
            output_path: Path to CSV file
            modes: Optional list of modes to include (defaults to all)
        """
        import csv

        if modes is None:
            modes = list(RuntimeMode)

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
                runs = self.get_runs_by_mode(mode)
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

    def clean_old_runs(
        self, keep_last_n: int = 10, runtime_mode: Optional[RuntimeMode] = None
    ):
        """
        Clean up old experiment outputs to save disk space.

        Args:
            keep_last_n: Number of recent runs to keep per mode
            runtime_mode: Optional mode filter (defaults to all modes)
        """
        modes = [runtime_mode] if runtime_mode else list(RuntimeMode)

        for mode in modes:
            runs = sorted(
                self.get_runs_by_mode(mode), key=lambda r: r.timestamp, reverse=True
            )

            if len(runs) <= keep_last_n:
                continue

            # Delete old runs
            for run in runs[keep_last_n:]:
                output_path = Path(run.output_path)
                if output_path.exists():
                    shutil.rmtree(output_path)
                    console.print(
                        f"[dim]Deleted old run: {run.run_id} ({mode.display_name})[/dim]"
                    )

            # Remove from manifest
            self.runs = [r for r in self.runs if r not in runs[keep_last_n:]]

        self._save_manifest()
        console.print(
            f"[green]Cleaned old runs (kept last {keep_last_n} per mode)[/green]"
        )
