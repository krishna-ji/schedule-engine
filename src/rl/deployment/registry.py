"""
Model registry for managing RL agent versions in production.

Provides atomic updates to production configuration and model metadata tracking.
Ensures safe model promotion with rollback capability.
"""

import json
import shutil
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

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
    - Atomic config updates (write to temp, then rename)
    - Version history tracking
    - Rollback support
    - Thread-safe operations
    - Validation before promotion

    Usage:
        registry = ModelRegistry("configs/prod.yaml", "models/rl_agents/registry.json")
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
        prod_config_path: str | Path,
        registry_path: str | Path = "models/rl_agents/registry.json",
    ):
        """
        Initialize model registry.

        Args:
            prod_config_path: Path to production config file (e.g., configs/prod.yaml)
            registry_path: Path to registry JSON file
        """
        self.prod_config_path = Path(prod_config_path)
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
        Promote a model to production with atomic config update.

        Steps:
        1. Validate model file exists
        2. Load current prod config
        3. Update rl.agent.model_path and rl.agent.type
        4. Write to temp file
        5. Atomic rename (overwrites prod config)
        6. Record deployment in registry

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
            ValueError: Invalid configuration
        """
        with self.lock:
            model_path = Path(model_path)

            # Validate model exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load current prod config
            with open(self.prod_config_path) as f:
                config = yaml.safe_load(f)

            # Backup current config
            backup_path = self.prod_config_path.with_suffix(".yaml.backup")
            shutil.copy(self.prod_config_path, backup_path)
            logger.info(f"Backed up config to {backup_path}")

            # Update config with new model
            if "rl" not in config:
                config["rl"] = {}
            if "agent" not in config["rl"]:
                config["rl"]["agent"] = {}

            config["rl"]["agent"]["model_path"] = str(model_path)
            config["rl"]["agent"]["type"] = agent_type

            # Atomic write: write to temp, then rename
            temp_path = self.prod_config_path.with_suffix(".yaml.tmp")
            with open(temp_path, "w") as f:
                yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

            # Atomic rename (overwrites prod config)
            temp_path.replace(self.prod_config_path)
            logger.info(f"Updated prod config: {self.prod_config_path}")

            # Create deployment record
            model_id = f"{agent_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            registration = ModelRegistration(
                model_id=model_id,
                model_path=str(model_path),
                agent_type=agent_type,
                deployed_at=datetime.now().isoformat(),
                promoted_from_checkpoint=checkpoint_id,
                validation_metrics=validation_metrics,
                config_snapshot=config.get("rl", {}),
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

            # Find current active and previous deprecated
            active_idx = None
            previous_idx = None

            for i, dep in enumerate(deployments):
                if dep.get("status") == "active":
                    active_idx = i
                elif dep.get("status") == "deprecated" and previous_idx is None:
                    previous_idx = i

            if active_idx is None or previous_idx is None:
                logger.warning("Cannot rollback: no previous deployment found")
                return None

            # Get previous deployment
            previous = ModelRegistration.from_dict(deployments[previous_idx])

            # Promote previous model
            registration = self.promote_model(
                model_path=previous.model_path,
                agent_type=previous.agent_type,
                validation_metrics=previous.validation_metrics,
                promoted_by="system_rollback",
                notes=f"Rolled back from {deployments[active_idx]['model_id']}",
            )

            # Record rollback
            registry_data["rollback_history"].append(
                {
                    "from": deployments[active_idx]["model_id"],
                    "to": previous.model_id,
                    "rolled_back_at": datetime.now().isoformat(),
                }
            )
            self._save_registry(registry_data)

            logger.info(f"Rolled back to {previous.model_id}")
            return registration

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

        with open(self.registry_path) as f:
            return json.load(f)  # type: ignore[no-any-return]

    def _save_registry(self, data: dict[str, Any]) -> None:
        """Save registry to JSON file (atomic write)."""
        temp_path = self.registry_path.with_suffix(".json.tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2)
        temp_path.replace(self.registry_path)
