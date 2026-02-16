"""
Model registry for managing RL agent versions in production.

Provides atomic metadata tracking for RL agent promotions and rollbacks.
Configuration updates are now handled dynamically through Python presets, so the
registry is the single source of truth for active deployments.
"""

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelRegistration:
    """Registration record for a deployed model."""

    model_id: str
    model_path: str
    agent_type: str
    deployed_at: str
    promoted_from_checkpoint: str | None
    validation_metrics: dict[str, float]
    config_snapshot: dict[str, Any]
    deployed_by: str
    notes: str = ""
    status: str = "active"  # active, deprecated, rolled_back

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRegistration":
        """Create from dictionary."""
        return cls(**data)


class ModelRegistry:
    """
    Manages production model deployments with atomic updates.

    Features:
    - Version history tracking
    - Rollback support
    - Thread-safe operations
    - Validation before promotion

    Usage:
        registry = ModelRegistry("models/rl_agents/registry.json")
        registry.promote_model(
            model_path="models/rl_agents/best_model.zip",
            agent_type="ppo",
            validation_metrics={"mean_reward": -2.5, "success_rate": 0.85},
            promoted_by="user",
            notes="Trained with curriculum on 500K timesteps"
        )
    """

    def __init__(
        self,
        registry_path: str | Path = "models/rl_agents/registry.json",
    ):
        """
        Initialize model registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path)
        self.lock = threading.Lock()

        # Ensure registry file exists
        if not self.registry_path.exists():
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_registry({"deployments": [], "rollback_history": []})

    def promote_model(
        self,
        model_path: str | Path,
        agent_type: str,
        validation_metrics: dict[str, float],
        promoted_by: str,
        notes: str = "",
        checkpoint_id: str | None = None,
    ) -> ModelRegistration:
        """
        Promote a model to production and mark its metadata as active.

        Args:
            model_path: Path to trained model file
            agent_type: Agent type (ppo, dqn)
            validation_metrics: Validation metrics (e.g., mean_reward, success_rate)
            promoted_by: Username or system identifier
            notes: Optional deployment notes
            checkpoint_id: Optional checkpoint ID from manifest

        Returns:
            ModelRegistration record

        Raises:
            FileNotFoundError: Model file doesn't exist
        """
        with self.lock:
            model_path = Path(model_path)

            # Validate model exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Create deployment record
            model_id = f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            registration = ModelRegistration(
                model_id=model_id,
                model_path=str(model_path),
                agent_type=agent_type,
                deployed_at=datetime.now().isoformat(),
                promoted_from_checkpoint=checkpoint_id,
                validation_metrics=validation_metrics,
                config_snapshot={
                    "agent": {
                        "model_path": str(model_path),
                        "type": agent_type,
                    }
                },
                deployed_by=promoted_by,
                notes=notes,
                status="active",
            )

            # Load registry
            registry_data = self._load_registry()

            # Mark previous active deployment as deprecated
            for deployment in registry_data["deployments"]:
                if deployment.get("status") == "active":
                    deployment["status"] = "deprecated"
                    deployment["deprecated_at"] = datetime.now().isoformat()

            # Add new deployment
            registry_data["deployments"].append(registration.to_dict())

            # Save registry
            self._save_registry(registry_data)
            logger.info(f"Registered deployment: {model_id}")

            return registration

    def rollback_to_previous(self) -> ModelRegistration | None:
        """
        Rollback to previous deployment.

        Returns:
            Previous deployment record if successful, None otherwise
        """
        with self.lock:
            registry_data = self._load_registry()
            deployments = registry_data["deployments"]

            # Find latest active and latest non-active deployment (deprecated/rolled_back)
            active_idx = None
            previous_idx = None

            for i in range(len(deployments) - 1, -1, -1):
                dep = deployments[i]
                status = dep.get("status")
                if active_idx is None and status == "active":
                    active_idx = i
                    continue
                if previous_idx is None and status in {"deprecated", "rolled_back"}:
                    previous_idx = i

                if active_idx is not None and previous_idx is not None:
                    break

            if previous_idx is None:
                logger.warning("Cannot rollback: no previous deployment found")
                return None

            # Update statuses in-place without creating new record
            now = datetime.now().isoformat()
            active_id = None
            if active_idx is not None:
                deployments[active_idx]["status"] = "rolled_back"
                deployments[active_idx]["rolled_back_at"] = now
                active_id = deployments[active_idx]["model_id"]

            previous = deployments[previous_idx]
            previous["status"] = "active"
            previous["reactivated_at"] = now

            registry_data["rollback_history"].append(
                {
                    "from": active_id,
                    "to": previous["model_id"],
                    "rolled_back_at": now,
                }
            )
            self._save_registry(registry_data)

            logger.info(f"Rolled back to {previous['model_id']}")
            return ModelRegistration.from_dict(previous)

    def get_active_deployment(self) -> ModelRegistration | None:
        """Get currently active deployment."""
        registry_data = self._load_registry()
        for dep in registry_data["deployments"]:
            if dep.get("status") == "active":
                return ModelRegistration.from_dict(dep)
        return None

    def get_deployment_history(self, limit: int = 10) -> list[ModelRegistration]:
        """
        Get deployment history (most recent first).

        Args:
            limit: Maximum number of deployments to return

        Returns:
            List of deployment records
        """
        registry_data = self._load_registry()
        deployments = registry_data["deployments"][-limit:]
        deployments.reverse()
        return [ModelRegistration.from_dict(d) for d in deployments]

    def _load_registry(self) -> dict[str, Any]:
        """Load registry from JSON file."""
        if not self.registry_path.exists():
            return {"deployments": [], "rollback_history": []}

        with self.registry_path.open() as f:
            data: dict[str, Any] = json.load(f)

        if "deployments" not in data:
            data["deployments"] = []
        if "rollback_history" not in data:
            data["rollback_history"] = []

        return data

    def _save_registry(self, data: dict[str, Any]) -> None:
        """Save registry to JSON file (atomic write)."""
        temp_path = self.registry_path.with_suffix(".json.tmp")
        with temp_path.open("w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(self.registry_path)
