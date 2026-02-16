"""
RepairExporter - Exports experiment results with repair history analysis.

Extends BaseExporter with repair-specific plots and data.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.experiments.output.base import BaseExporter

if TYPE_CHECKING:
    from src.ga.run_helpers import EvolutionStats


class RepairExporter(BaseExporter):
    """Exports experiment results including repair operator analysis."""

    def export_all(
        self,
        final_pop: list[Any] | None = None,
        stats: EvolutionStats | None = None,
        best_individual: list | None = None,
        metadata: dict[str, Any] | None = None,
        repair_history: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        """Export all results including repair history."""
        super().export_all(
            final_pop=final_pop,
            stats=stats,
            best_individual=best_individual,
            metadata=metadata,
        )

        if repair_history is not None:
            self._export_repair_history(repair_history)
            if final_pop is not None and stats is not None:
                self._export_repair_plots(repair_history)

    def _export_repair_history(self, repair_history: list[dict[str, Any]]) -> None:
        """Save repair history as JSON."""
        path = self.output_dir / "repair_history.json"
        self._write_json(path, repair_history)
        self.logger.debug("Saved repair history → %s", path)

    def _export_repair_plots(self, repair_history: list[dict[str, Any]]) -> None:
        """Generate repair-specific diagnostic plots."""
        try:
            from src.io.export.plot_repair_analysis import (
                plot_repair_efficacy_over_generations,
            )

            plot_dir = str(self.plots_dir)
            plot_repair_efficacy_over_generations(repair_history, plot_dir)
            self.logger.debug("Saved repair analysis plots → %s", plot_dir)
        except Exception:
            self.logger.warning("Could not generate repair plots", exc_info=True)
