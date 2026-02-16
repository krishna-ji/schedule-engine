"""
RLExporter - Exports experiment results with RL training history.

Extends BaseExporter with Q-learning/RL-specific data and plots.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.experiments.output.base import BaseExporter

if TYPE_CHECKING:
    from src.ga.run_helpers import EvolutionStats


class RLExporter(BaseExporter):
    """Exports experiment results including RL training data."""

    def export_all(
        self,
        final_pop: list[Any] | None = None,
        stats: EvolutionStats | None = None,
        best_individual: list | None = None,
        metadata: dict[str, Any] | None = None,
        q_table_history: list[dict[str, float]] | None = None,
        epsilon_history: list[float] | None = None,
        rewards_history: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Export all results including RL training history."""
        super().export_all(
            final_pop=final_pop,
            stats=stats,
            best_individual=best_individual,
            metadata=metadata,
        )

        rl_data: dict[str, Any] = {}
        if q_table_history is not None:
            rl_data["q_table_history"] = q_table_history
        if epsilon_history is not None:
            rl_data["epsilon_history"] = epsilon_history
        if rewards_history is not None:
            rl_data["rewards_history"] = rewards_history

        if rl_data:
            self._export_rl_history(rl_data)

    def _export_rl_history(self, rl_data: dict[str, Any]) -> None:
        """Save RL training history as JSON."""
        path = self.output_dir / "rl_history.json"
        self._write_json(path, rl_data)
        self.logger.debug("Saved RL history → %s", path)
