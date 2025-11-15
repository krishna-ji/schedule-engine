"""
Unit tests for model registry and promotion system.
"""

import pytest
import json
import yaml
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rl.deployment.registry import ModelRegistry, ModelRegistration


class TestModelRegistry:
    """Test ModelRegistry functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        with TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_config(self, temp_dir):
        """Create sample prod.yaml config."""
        config_path = temp_dir / "prod.yaml"
        config = {
            "rl": {
                "enabled": False,
                "agent": {
                    "type": "ppo",
                    "model_path": "models/rl_agents/old_model.zip",
                },
            },
        }
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        return config_path

    @pytest.fixture
    def sample_model(self, temp_dir):
        """Create sample model file."""
        model_path = temp_dir / "test_model.zip"
        model_path.write_text("fake model data")
        return model_path

    def test_registry_creation(self, temp_dir, sample_config):
        """Test registry initialization."""
        registry_path = temp_dir / "registry.json"
        registry = ModelRegistry(sample_config, registry_path)

        assert registry.prod_config_path == sample_config
        assert registry.registry_path == registry_path
        assert registry_path.exists()

    def test_promote_model(self, temp_dir, sample_config, sample_model):
        """Test model promotion."""
        registry_path = temp_dir / "registry.json"
        registry = ModelRegistry(sample_config, registry_path)

        # Promote model
        registration = registry.promote_model(
            model_path=sample_model,
            agent_type="ppo",
            validation_metrics={"mean_reward": -2.5, "success_rate": 0.85},
            promoted_by="test_user",
            notes="Test promotion",
        )

        assert registration.model_path == str(sample_model)
        assert registration.agent_type == "ppo"
        assert registration.status == "active"
        assert registration.promoted_by == "test_user"

        # Verify config was updated
        with open(sample_config, "r") as f:
            config = yaml.safe_load(f)

        assert config["rl"]["agent"]["model_path"] == str(sample_model)
        assert config["rl"]["agent"]["type"] == "ppo"

        # Verify registry was updated
        with open(registry_path, "r") as f:
            registry_data = json.load(f)

        assert len(registry_data["deployments"]) == 1
        assert registry_data["deployments"][0]["model_id"] == registration.model_id

    def test_multiple_promotions(self, temp_dir, sample_config, sample_model):
        """Test multiple promotions (should deprecate old ones)."""
        registry_path = temp_dir / "registry.json"
        registry = ModelRegistry(sample_config, registry_path)

        # First promotion
        reg1 = registry.promote_model(
            model_path=sample_model,
            agent_type="ppo",
            validation_metrics={"mean_reward": -3.0},
            promoted_by="test_user",
        )

        # Second promotion
        model2 = temp_dir / "model2.zip"
        model2.write_text("model 2 data")
        reg2 = registry.promote_model(
            model_path=model2,
            agent_type="dqn",
            validation_metrics={"mean_reward": -2.0},
            promoted_by="test_user",
        )

        # Verify only second is active
        assert reg2.status == "active"

        with open(registry_path, "r") as f:
            registry_data = json.load(f)

        assert len(registry_data["deployments"]) == 2
        assert registry_data["deployments"][0]["status"] == "deprecated"
        assert registry_data["deployments"][1]["status"] == "active"

    def test_get_active_deployment(self, temp_dir, sample_config, sample_model):
        """Test getting active deployment."""
        registry_path = temp_dir / "registry.json"
        registry = ModelRegistry(sample_config, registry_path)

        # No deployment yet
        assert registry.get_active_deployment() is None

        # Promote model
        registration = registry.promote_model(
            model_path=sample_model,
            agent_type="ppo",
            validation_metrics={"mean_reward": -2.5},
            promoted_by="test_user",
        )

        # Get active
        active = registry.get_active_deployment()
        assert active is not None
        assert active.model_id == registration.model_id
        assert active.status == "active"

    def test_rollback(self, temp_dir, sample_config):
        """Test rollback to previous deployment."""
        registry_path = temp_dir / "registry.json"
        registry = ModelRegistry(sample_config, registry_path)

        # Promote two models
        model1 = temp_dir / "model1.zip"
        model1.write_text("model 1")
        reg1 = registry.promote_model(
            model_path=model1,
            agent_type="ppo",
            validation_metrics={"mean_reward": -3.0},
            promoted_by="test_user",
        )

        model2 = temp_dir / "model2.zip"
        model2.write_text("model 2")
        reg2 = registry.promote_model(
            model_path=model2,
            agent_type="ppo",
            validation_metrics={"mean_reward": -2.0},
            promoted_by="test_user",
        )

        # Rollback
        rolled_back = registry.rollback_to_previous()
        assert rolled_back is not None

        # Verify config points to model1
        with open(sample_config, "r") as f:
            config = yaml.safe_load(f)
        assert config["rl"]["agent"]["model_path"] == str(model1)

        # Verify rollback history
        with open(registry_path, "r") as f:
            registry_data = json.load(f)
        assert len(registry_data["rollback_history"]) == 1

    def test_deployment_history(self, temp_dir, sample_config):
        """Test getting deployment history."""
        registry_path = temp_dir / "registry.json"
        registry = ModelRegistry(sample_config, registry_path)

        # Promote multiple models
        for i in range(5):
            model = temp_dir / f"model{i}.zip"
            model.write_text(f"model {i}")
            registry.promote_model(
                model_path=model,
                agent_type="ppo",
                validation_metrics={"mean_reward": float(-3 + i * 0.5)},
                promoted_by="test_user",
            )

        # Get history (most recent first)
        history = registry.get_deployment_history(limit=3)
        assert len(history) == 3
        assert history[0].status == "active"  # Most recent is active
        assert history[1].status == "deprecated"
        assert history[2].status == "deprecated"

    def test_model_not_found(self, temp_dir, sample_config):
        """Test promotion fails if model doesn't exist."""
        registry_path = temp_dir / "registry.json"
        registry = ModelRegistry(sample_config, registry_path)

        with pytest.raises(FileNotFoundError):
            registry.promote_model(
                model_path="nonexistent_model.zip",
                agent_type="ppo",
                validation_metrics={},
                promoted_by="test_user",
            )

    def test_atomic_write(self, temp_dir, sample_config, sample_model):
        """Test that config updates are atomic."""
        registry_path = temp_dir / "registry.json"
        registry = ModelRegistry(sample_config, registry_path)

        # Promote model
        registry.promote_model(
            model_path=sample_model,
            agent_type="ppo",
            validation_metrics={"mean_reward": -2.5},
            promoted_by="test_user",
        )

        # Verify no temp files left behind
        temp_files = list(temp_dir.glob("*.tmp"))
        assert len(temp_files) == 0

        # Verify backup exists
        backup_files = list(temp_dir.glob("*.backup"))
        assert len(backup_files) == 1


class TestModelRegistration:
    """Test ModelRegistration dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        reg = ModelRegistration(
            model_id="test_model_123",
            model_path="models/test.zip",
            agent_type="ppo",
            deployed_at="2025-01-15T12:00:00",
            promoted_from_checkpoint="checkpoint_1",
            validation_metrics={"mean_reward": -2.5},
            config_snapshot={},
            deployed_by="test_user",
            notes="Test",
            status="active",
        )

        data = reg.to_dict()
        assert isinstance(data, dict)
        assert data["model_id"] == "test_model_123"
        assert data["agent_type"] == "ppo"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "model_id": "test_model_123",
            "model_path": "models/test.zip",
            "agent_type": "ppo",
            "deployed_at": "2025-01-15T12:00:00",
            "promoted_from_checkpoint": "checkpoint_1",
            "validation_metrics": {"mean_reward": -2.5},
            "config_snapshot": {},
            "deployed_by": "test_user",
            "notes": "Test",
            "status": "active",
        }

        reg = ModelRegistration.from_dict(data)
        assert reg.model_id == "test_model_123"
        assert reg.agent_type == "ppo"
        assert reg.status == "active"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
