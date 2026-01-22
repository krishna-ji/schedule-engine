"""
Checkpoint metadata management for RL training.

Provides utilities for:
- Recording checkpoint metadata (seed, config, validation metrics, timestamp)
- Managing manifest.json for checkpoint tracking
- Selecting best checkpoint based on validation performance
- Checkpoint versioning and reproducibility
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from schedule_engine.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointMetadata:
    """
    Metadata for a training checkpoint.

    Attributes:
        checkpoint_id: Unique checkpoint identifier
        model_path: Path to saved model file
        timestep: Training timestep when checkpoint was saved
        timestamp: ISO timestamp when checkpoint was created
        stage: Training stage (for curriculum learning)
        seed: Random seed used
        config_hash: Hash of config used for training
        validation_metrics: Validation performance metrics
        training_metrics: Training metrics at checkpoint time
        status: Checkpoint status (checkpoint, validated, promoted, archived)
        notes: Optional notes about this checkpoint
    """

    checkpoint_id: str
    model_path: str
    timestep: int
    timestamp: str
    stage: str | None = None
    seed: int | None = None
    config_hash: str | None = None
    validation_metrics: dict[str, float] | None = None
    training_metrics: dict[str, float] | None = None
    status: str = "checkpoint"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.validation_metrics is None:
            self.validation_metrics = {}
        if self.training_metrics is None:
            self.training_metrics = {}


class CheckpointManager:
    """
    Manages checkpoint metadata and manifest file.

    Provides functionality for:
    - Adding checkpoints to manifest
    - Querying checkpoints by criteria
    - Selecting best checkpoint
    - Updating checkpoint status
    """

    def __init__(self, manifest_path: str = "models/rl_agents/manifest.json"):
        """
        Initialize checkpoint manager.

        Args:
            manifest_path: Path to manifest JSON file
        """
        self.manifest_path = Path(manifest_path)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing manifest
        self.checkpoints: list[CheckpointMetadata] = self._load_manifest()

        logger.info(
            f"Initialized CheckpointManager with {len(self.checkpoints)} existing checkpoints"
        )

    def _load_manifest(self) -> list[CheckpointMetadata]:
        """Load manifest from file."""
        if not self.manifest_path.exists():
            logger.info(f"No existing manifest found at {self.manifest_path}")
            return []

        try:
            with open(self.manifest_path) as f:
                data = json.load(f)

            checkpoints = [CheckpointMetadata(**entry) for entry in data]
            logger.info(f"Loaded {len(checkpoints)} checkpoints from manifest")
            return checkpoints
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return []

    def _save_manifest(self) -> None:
        """Save manifest to file."""
        try:
            data = [asdict(cp) for cp in self.checkpoints]

            with open(self.manifest_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved manifest with {len(self.checkpoints)} checkpoints")
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            raise

    def add_checkpoint(
        self,
        model_path: str,
        timestep: int,
        stage: str | None = None,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
        validation_metrics: dict[str, float] | None = None,
        training_metrics: dict[str, float] | None = None,
        notes: str = "",
    ) -> CheckpointMetadata:
        """
        Add new checkpoint to manifest.

        Args:
            model_path: Path to saved model
            timestep: Training timestep
            stage: Training stage
            seed: Random seed
            config: Training configuration
            validation_metrics: Validation performance
            training_metrics: Training metrics
            notes: Optional notes

        Returns:
            Created checkpoint metadata
        """
        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(model_path, timestep)

        # Hash config for reproducibility
        config_hash = None
        if config:
            config_hash = self._hash_config(config)

        # Create metadata
        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            model_path=model_path,
            timestep=timestep,
            timestamp=datetime.now().isoformat(),
            stage=stage,
            seed=seed,
            config_hash=config_hash,
            validation_metrics=validation_metrics or {},
            training_metrics=training_metrics or {},
            status="checkpoint",
            notes=notes,
        )

        self.checkpoints.append(metadata)
        self._save_manifest()

        logger.info(
            f"Added checkpoint: {checkpoint_id} (timestep={timestep}, stage={stage})"
        )

        return metadata

    def _generate_checkpoint_id(self, model_path: str, timestep: int) -> str:
        """Generate unique checkpoint ID."""
        base = Path(model_path).stem
        return f"{base}_t{timestep}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _hash_config(self, config: dict[str, Any]) -> str:
        """Generate hash of configuration for reproducibility."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointMetadata | None:
        """Get checkpoint by ID."""
        for cp in self.checkpoints:
            if cp.checkpoint_id == checkpoint_id:
                return cp
        return None

    def query_checkpoints(
        self,
        stage: str | None = None,
        status: str | None = None,
        min_timestep: int | None = None,
        max_timestep: int | None = None,
    ) -> list[CheckpointMetadata]:
        """
        Query checkpoints by criteria.

        Args:
            stage: Filter by training stage
            status: Filter by status
            min_timestep: Minimum timestep
            max_timestep: Maximum timestep

        Returns:
            List of matching checkpoints
        """
        results = self.checkpoints

        if stage:
            results = [cp for cp in results if cp.stage == stage]

        if status:
            results = [cp for cp in results if cp.status == status]

        if min_timestep is not None:
            results = [cp for cp in results if cp.timestep >= min_timestep]

        if max_timestep is not None:
            results = [cp for cp in results if cp.timestep <= max_timestep]

        return results

    def get_best_checkpoint(
        self,
        metric_name: str = "mean_reward",
        stage: str | None = None,
        status: str = "checkpoint",
        maximize: bool = True,
    ) -> CheckpointMetadata | None:
        """
        Select best checkpoint based on validation metric.

        Args:
            metric_name: Validation metric to optimize
            stage: Filter by stage
            status: Filter by status
            maximize: True to maximize metric, False to minimize

        Returns:
            Best checkpoint or None if no checkpoints found
        """
        candidates = self.query_checkpoints(stage=stage, status=status)

        # Filter checkpoints with the metric
        candidates = [
            cp
            for cp in candidates
            if cp.validation_metrics and metric_name in cp.validation_metrics
        ]

        if not candidates:
            logger.warning(f"No checkpoints found with metric '{metric_name}'")
            return None

        # Find best
        best = max(
            candidates,
            key=lambda cp: (
                cp.validation_metrics[metric_name]
                if cp.validation_metrics
                else float("-inf")
            ),
        )
        if not maximize:
            best = min(
                candidates,
                key=lambda cp: (
                    cp.validation_metrics[metric_name]
                    if cp.validation_metrics
                    else float("inf")
                ),
            )

        logger.info(
            f"Best checkpoint: {best.checkpoint_id} "
            f"({metric_name}={best.validation_metrics[metric_name] if best.validation_metrics else 'N/A':.4f})"
        )

        return best

    def update_checkpoint_status(
        self,
        checkpoint_id: str,
        new_status: str,
        notes: str | None = None,
    ) -> None:
        """
        Update checkpoint status.

        Args:
            checkpoint_id: Checkpoint to update
            new_status: New status (validated, promoted, archived)
            notes: Optional notes
        """
        checkpoint = self.get_checkpoint(checkpoint_id)

        if checkpoint is None:
            logger.error(f"Checkpoint not found: {checkpoint_id}")
            return

        old_status = checkpoint.status
        checkpoint.status = new_status

        if notes:
            checkpoint.notes = notes

        self._save_manifest()

        logger.info(f"Updated checkpoint {checkpoint_id}: {old_status} -> {new_status}")

    def get_statistics(self) -> dict[str, Any]:
        """Get checkpoint statistics."""
        total = len(self.checkpoints)

        if total == 0:
            return {"total": 0}

        # Count by status
        status_counts: dict[str, int] = {}
        for cp in self.checkpoints:
            status_counts[cp.status] = status_counts.get(cp.status, 0) + 1

        # Count by stage
        stage_counts: dict[str, int] = {}
        for cp in self.checkpoints:
            if cp.stage:
                stage_counts[cp.stage] = stage_counts.get(cp.stage, 0) + 1

        return {
            "total": total,
            "by_status": status_counts,
            "by_stage": stage_counts,
            "latest_timestep": max(cp.timestep for cp in self.checkpoints),
        }


def create_checkpoint_metadata(
    model_path: str,
    timestep: int,
    validation_metrics: dict[str, float],
    **kwargs: float | str | dict[str, float] | int | None,  # type: ignore[misc]
) -> CheckpointMetadata:
    """
    Convenience function to create checkpoint metadata.

    Args:
        model_path: Path to saved model
        timestep: Training timestep
        validation_metrics: Validation metrics
        **kwargs: Additional metadata fields

    Returns:
        CheckpointMetadata instance
    """
    checkpoint_id = Path(model_path).stem + f"_t{timestep}"

    return CheckpointMetadata(
        checkpoint_id=checkpoint_id,
        model_path=model_path,
        timestep=timestep,
        timestamp=datetime.now().isoformat(),
        validation_metrics=validation_metrics,
        **kwargs,  # type: ignore[arg-type]
    )
