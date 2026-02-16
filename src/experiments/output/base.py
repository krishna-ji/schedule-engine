"""
BaseExporter - Handles exporting experiment results to disk.

Exports:
- Best individual info (JSON)
- Evolution statistics (JSON + CSV)
- Experiment metadata (JSON)
- Diagnostic plots (PNG)
- Decoded schedule (JSON + PDF)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.ga.run_helpers import EvolutionStats, NotebookData


class BaseExporter:
    """Exports experiment results: stats, metadata, schedule, and plots."""

    def __init__(
        self,
        output_dir: Path | str,
        data: NotebookData,
        logger: logging.Logger,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.data = data
        self.logger = logger

        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.csv_dir = self.output_dir / "csv"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

    def export_all(
        self,
        final_pop: list[Any] | None,
        stats: EvolutionStats | None,
        best_individual: list | None,
        metadata: dict[str, Any] | None,
        **kwargs: Any,
    ) -> None:
        """Export all results to disk."""
        if metadata is not None:
            self._export_metadata(metadata)

        if stats is not None:
            self._export_stats(stats)

        if best_individual is not None:
            self._export_schedule(best_individual)

        if final_pop is not None and stats is not None:
            self._export_plots(final_pop, stats)

    # -------------------- Individual Exports --------------------

    def _export_metadata(self, metadata: dict[str, Any]) -> None:
        """Save experiment metadata."""
        path = self.output_dir / "metadata.json"
        self._write_json(path, metadata)
        self.logger.debug("Saved metadata → %s", path)

    def _export_stats(self, stats: EvolutionStats) -> None:
        """Save evolution statistics as JSON and CSV."""
        # JSON
        stats_path = self.output_dir / "stats.json"
        stats_dict: dict[str, Any] = {
            "generations": stats.generations,
            "min_hard": stats.min_hard,
            "avg_hard": stats.avg_hard,
            "max_hard": stats.max_hard,
            "min_soft": stats.min_soft,
            "avg_soft": stats.avg_soft,
            "feasible_count": stats.feasible_count,
            "generation_times": stats.generation_times,
            "elapsed_time": stats.elapsed_time,
            "diversity": stats.diversity,
            "hypervolume": stats.hypervolume,
            "spacing": stats.spacing,
            "pareto_front_size": stats.pareto_front_size,
        }
        self._write_json(stats_path, stats_dict)
        self.logger.debug("Saved stats → %s", stats_path)

        # CSV
        csv_path = self.csv_dir / "evolution_stats.csv"
        self._write_stats_csv(csv_path, stats)
        self.logger.debug("Saved stats CSV → %s", csv_path)

    def _export_schedule(self, best_individual: list) -> None:
        """Save best individual's fitness and decode schedule to JSON + PDF."""
        try:
            # 1. Save basic fitness info
            path = self.output_dir / "best_individual.json"
            info: dict[str, Any] = {
                "num_genes": len(best_individual),
            }
            if hasattr(best_individual, "fitness") and best_individual.fitness.valid:
                info["fitness"] = list(best_individual.fitness.values)
            self._write_json(path, info)
            self.logger.debug("Saved best individual info → %s", path)

            # 2. Decode and export full schedule (schedule.json + calendar.pdf)
            from src.io.decoder import decode_individual
            from src.io.export.exporter import export_everything

            decoded_schedule = decode_individual(
                best_individual,
                self.data.courses,
                self.data.instructors,
                self.data.groups,
                self.data.rooms,
            )

            export_everything(
                schedule=decoded_schedule,
                output_path=str(self.output_dir),
                qts=self.data.qts,
                course_lookup=self.data.courses,
                parallel=True,
            )
            self.logger.info("Exported schedule.json and calendar.pdf")
        except Exception:
            self.logger.warning("Could not export schedule", exc_info=True)

    def _export_plots(self, final_pop: list[Any], stats: EvolutionStats) -> None:
        """Generate and save diagnostic plots."""
        try:
            from src.io.export.plothard import \
                plot_hard_constraint_violation_over_generation
            from src.io.export.plotpareto import plot_pareto_front
            from src.io.export.plotsoft import \
                plot_soft_constraint_violation_over_generation

            plot_dir = str(self.plots_dir)

            plot_hard_constraint_violation_over_generation(
                [int(v) for v in stats.min_hard], plot_dir
            )
            plot_soft_constraint_violation_over_generation(
                stats.min_soft, plot_dir
            )
            plot_pareto_front(final_pop, plot_dir)

            self.logger.debug("Saved diagnostic plots → %s", plot_dir)
        except Exception:
            self.logger.warning("Could not generate plots", exc_info=True)

        # Advanced plots (optional, may not always be available)
        try:
            from src.io.export.plot_hypervolume import plot_hypervolume_trend
            from src.io.export.plot_spacing import plot_spacing_trend

            plot_dir = str(self.plots_dir)

            if stats.hypervolume:
                plot_hypervolume_trend(stats.hypervolume, plot_dir)
            if stats.spacing:
                plot_spacing_trend(stats.spacing, plot_dir)
        except Exception:
            self.logger.debug("Advanced plots skipped (optional)")

    # -------------------- Helpers --------------------

    def _write_json(self, path: Path, data: Any) -> None:
        """Write data to a JSON file."""
        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)

    def _write_stats_csv(self, path: Path, stats: EvolutionStats) -> None:
        """Write evolution stats to CSV."""
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "generation",
                    "min_hard",
                    "avg_hard",
                    "max_hard",
                    "min_soft",
                    "avg_soft",
                    "feasible_count",
                    "gen_time",
                ]
            )
            for i, gen in enumerate(stats.generations):
                writer.writerow(
                    [
                        gen,
                        stats.min_hard[i] if i < len(stats.min_hard) else "",
                        stats.avg_hard[i] if i < len(stats.avg_hard) else "",
                        stats.max_hard[i] if i < len(stats.max_hard) else "",
                        stats.min_soft[i] if i < len(stats.min_soft) else "",
                        stats.avg_soft[i] if i < len(stats.avg_soft) else "",
                        (
                            stats.feasible_count[i]
                            if i < len(stats.feasible_count)
                            else ""
                        ),
                        (
                            f"{stats.generation_times[i]:.3f}"
                            if i < len(stats.generation_times)
                            else ""
                        ),
                    ]
                )
