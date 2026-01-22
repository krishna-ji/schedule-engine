"""
Promote validated RL model to production.

This script promotes a validated checkpoint to production by:
1. Validating the trained policy/checkpoint
2. Recording deployment metadata in models/rl_agents/registry.json
3. Letting Python presets automatically pick the latest active agent

Usage:
    # Promote best checkpoint from manifest
    python scripts/promote_model_to_prod.py --checkpoint-id ppo_stage3_20250115_123045

    # Promote specific model file
    python scripts/promote_model_to_prod.py --model-path models/rl_agents/best_model.zip --agent-type ppo

    # Rollback to previous deployment
    python scripts/promote_model_to_prod.py --rollback
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

# Add src/ to path for local package imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

REGISTRY_PATH = project_root / "models" / "rl_agents" / "registry.json"

from schedule_engine.config import get_config
from schedule_engine.rl.deployment.registry import ModelRegistry
from schedule_engine.rl.training.checkpoints import CheckpointManager, CheckpointMetadata
from schedule_engine.utils.logging_config import get_logger

logger = get_logger(__name__)


def _resolve_manifest_path() -> Path:
    """Best-effort resolution of the checkpoint manifest path."""
    try:
        config = get_config()
        return Path(config.rl.training.save_dir).resolve() / "manifest.json"
    except Exception:
        # Fallback to default models directory if config is unavailable
        return project_root / "models" / "rl_agents" / "manifest.json"


def _load_model_metadata(model_path: Path) -> dict[str, Any] | None:
    """Load adjacent JSON metadata for a saved RL model, if present."""
    metadata_path = model_path.with_suffix(".json")
    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path) as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        logger.warning(f"Failed to parse model metadata {metadata_path}: {exc}")
        return None

    if isinstance(data, dict):
        return cast(dict[str, Any], data)

    logger.warning(
        "Model metadata %s is not a JSON object (found %s)",
        metadata_path,
        type(data).__name__,
    )
    return None


def _infer_agent_type(
    *,
    checkpoint: CheckpointMetadata | None = None,
    metadata: dict[str, Any] | None = None,
    override: str | None = None,
) -> str:
    """Infer agent type from override, metadata, or config defaults."""
    if override:
        return override.lower()

    if checkpoint and checkpoint.training_metrics:
        metric_agent = checkpoint.training_metrics.get("agent_type")
        if isinstance(metric_agent, str) and metric_agent.strip():
            return metric_agent.strip().lower()

    if metadata:
        metadata_agent = metadata.get("agent_type")
        if isinstance(metadata_agent, str) and metadata_agent.strip():
            return metadata_agent.strip().lower()

    if checkpoint:
        stem = Path(checkpoint.model_path).stem.lower()
        for candidate in ("ppo", "dqn"):
            if candidate in stem:
                return candidate

    try:
        config = get_config()
        config_agent = getattr(config.rl.agent, "type", None)
        if isinstance(config_agent, str) and config_agent.strip():
            return config_agent.strip().lower()
    except Exception:
        pass

    return "ppo"


def promote_from_checkpoint(
    checkpoint_id: str,
    promoted_by: str = "user",
    notes: str = "",
) -> None:
    """
    Promote a checkpoint from manifest to production.

    Args:
        checkpoint_id: Checkpoint ID from manifest
        promoted_by: User identifier
        notes: Optional deployment notes
    """
    manifest_path = _resolve_manifest_path()
    manager = CheckpointManager(str(manifest_path))

    checkpoint = manager.get_checkpoint(checkpoint_id)
    if not checkpoint:
        logger.error(f"Checkpoint not found: {checkpoint_id}")
        print(f" Checkpoint not found: {checkpoint_id}")
        print("\nAvailable checkpoints:")
        for ckpt in manager.checkpoints:
            print(f"  - {ckpt.checkpoint_id} ({ckpt.stage}, {ckpt.status})")
        sys.exit(1)

    # Validate checkpoint file exists
    checkpoint_path = Path(checkpoint.model_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint.model_path}")
        print(f" Checkpoint file not found: {checkpoint.model_path}")
        sys.exit(1)

    metadata = _load_model_metadata(checkpoint_path)
    agent_type = _infer_agent_type(checkpoint=checkpoint, metadata=metadata)

    # Validate checkpoint is a valid model
    try:
        from stable_baselines3 import DQN, PPO

        if agent_type == "ppo":
            _ = PPO.load(checkpoint.model_path)
        elif agent_type == "dqn":
            _ = DQN.load(checkpoint.model_path)
        else:
            logger.error(f"Unknown agent type: {agent_type}")
            print(f" Unknown agent type: {agent_type}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Invalid checkpoint file: {e}")
        print(f" Invalid checkpoint file: {e}")
        sys.exit(1)

    # Validate checkpoint status
    if checkpoint.status != "validated":
        logger.warning(
            f"Checkpoint {checkpoint_id} has status '{checkpoint.status}' (not validated)"
        )
        confirm = input(
            f"[?] Checkpoint status is '{checkpoint.status}'. Continue? (y/N): "
        )
        if confirm.lower() != "y":
            print("Promotion cancelled")
            sys.exit(0)

    print(f" Promoting checkpoint: {checkpoint_id}")
    print(f"   Model: {checkpoint.model_path}")
    print(f"   Agent: {agent_type}")
    print(f"   Stage: {checkpoint.stage}")
    mean_reward = (
        checkpoint.validation_metrics.get("mean_reward")
        if checkpoint.validation_metrics
        else None
    )
    if isinstance(mean_reward, float | int):
        mean_reward_display = f"{mean_reward:.4f}"
    else:
        mean_reward_display = "N/A"
    print(f"   Mean Reward: {mean_reward_display}")
    print()

    # Initialize registry
    registry = ModelRegistry(REGISTRY_PATH)

    # Promote model
    try:
        validation_metrics = checkpoint.validation_metrics or {}
        registration = registry.promote_model(
            model_path=checkpoint.model_path,
            agent_type=agent_type,
            validation_metrics=validation_metrics,
            promoted_by=promoted_by,
            notes=notes or f"Promoted from checkpoint {checkpoint_id}",
            checkpoint_id=checkpoint_id,
        )

        print(" Model promoted successfully!")
        print(f"   Deployment ID: {registration.model_id}")
        print(f"   Registry updated: {REGISTRY_PATH}")
        print()
        print(" Active agent will be used automatically by Python presets.")
        print(" To run production workload:")
        print("   uv run prod")

    except Exception as e:
        logger.exception("Promotion failed")
        print(f" Promotion failed: {e}")
        sys.exit(1)


def promote_from_file(
    model_path: str | Path,
    agent_type: str,
    validation_metrics: dict[str, float] | None,
    promoted_by: str = "user",
    notes: str = "",
) -> None:
    """
    Promote a model file to production.

    Args:
        model_path: Path to model file
        agent_type: Agent type (ppo, dqn)
        validation_metrics: Validation metrics dict
        promoted_by: User identifier
        notes: Optional deployment notes
    """
    model_path_path = Path(model_path)

    if not model_path_path.exists():
        logger.error(f"Model file not found: {model_path_path}")
        print(f" Model file not found: {model_path_path}")
        sys.exit(1)

    agent_type = agent_type.lower()

    print(f" Promoting model file: {model_path_path}")
    print(f"   Agent: {agent_type}")
    print()

    # Initialize registry
    registry = ModelRegistry(REGISTRY_PATH)

    # Promote model
    try:
        metrics = validation_metrics or {"mean_reward": 0.0}

        registration = registry.promote_model(
            model_path=model_path_path,
            agent_type=agent_type,
            validation_metrics=metrics,
            promoted_by=promoted_by,
            notes=notes,
        )

        print(" Model promoted successfully!")
        print(f"   Deployment ID: {registration.model_id}")
        print(f"   Registry updated: {REGISTRY_PATH}")
        print()
        print(" Active agent will be used automatically by Python presets.")
        print(" To run production workload:")
        print("   uv run prod")

    except Exception as e:
        logger.exception("Promotion failed")
        print(f" Promotion failed: {e}")
        sys.exit(1)


def rollback_deployment() -> None:
    """Rollback to previous deployment."""
    registry = ModelRegistry(REGISTRY_PATH)

    # Get current deployment
    current = registry.get_active_deployment()
    if not current:
        print(" No active deployment found")
        sys.exit(1)

    print(f"Current deployment: {current.model_id}")
    print(f"   Model: {current.model_path}")
    print(f"   Deployed: {current.deployed_at}")
    print()

    confirm = input("[?] Rollback to previous deployment? (y/N): ")
    if confirm.lower() != "y":
        print("Rollback cancelled")
        sys.exit(0)

    try:
        registration = registry.rollback_to_previous()
        if registration:
            print(" Rolled back successfully!")
            print(f"   Now using: {registration.model_id}")
            print(f"   Model: {registration.model_path}")
        else:
            print(" Rollback failed: No previous deployment found")
            sys.exit(1)
    except Exception as e:
        logger.exception("Rollback failed")
        print(f" Rollback failed: {e}")
        sys.exit(1)


def list_deployments(limit: int = 10) -> None:
    """List recent deployments."""
    registry = ModelRegistry(REGISTRY_PATH)

    deployments = registry.get_deployment_history(limit=limit)

    if not deployments:
        print("No deployments found")
        return

    print(f"Recent deployments (showing last {len(deployments)}):")
    print()

    for dep in deployments:
        status_icon = "" if dep.status == "active" else ""
        print(f"{status_icon} {dep.model_id}")
        print(f"   Model: {dep.model_path}")
        print(f"   Agent: {dep.agent_type}")
        print(f"   Deployed: {dep.deployed_at}")
        print(f"   Metrics: {dep.validation_metrics}")
        print(f"   Status: {dep.status}")
        if dep.notes:
            print(f"   Notes: {dep.notes}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Promote validated RL model to production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Promote best checkpoint from manifest
  python scripts/promote_model_to_prod.py --checkpoint-id ppo_stage3_20250115_123045

  # Promote specific model file
  python scripts/promote_model_to_prod.py --model-path models/rl_agents/best_model.zip --agent-type ppo

  # Rollback to previous deployment
  python scripts/promote_model_to_prod.py --rollback

  # List deployment history
  python scripts/promote_model_to_prod.py --list
        """,
    )

    parser.add_argument(
        "--checkpoint-id",
        type=str,
        help="Checkpoint ID from manifest to promote",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to model file to promote",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        choices=["ppo", "dqn"],
        help="Agent type (required with --model-path)",
    )
    parser.add_argument(
        "--promoted-by",
        type=str,
        default="user",
        help="User identifier (default: 'user')",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional deployment notes",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to previous deployment",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent deployments",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of deployments to list (default: 10)",
    )

    args = parser.parse_args()

    # Handle list command
    if args.list:
        list_deployments(limit=args.limit)
        return

    # Handle rollback command
    if args.rollback:
        rollback_deployment()
        return

    # Handle promotion commands
    if args.checkpoint_id:
        promote_from_checkpoint(
            checkpoint_id=args.checkpoint_id,
            promoted_by=args.promoted_by,
            notes=args.notes,
        )
    elif args.model_path:
        if not args.agent_type:
            parser.error("--agent-type is required with --model-path")

        # Use default metrics if not provided
        validation_metrics = {"mean_reward": 0.0}

        promote_from_file(
            model_path=args.model_path,
            agent_type=args.agent_type,
            validation_metrics=validation_metrics,
            promoted_by=args.promoted_by,
            notes=args.notes,
        )
    else:
        parser.error(
            "Must specify --checkpoint-id, --model-path, --rollback, or --list"
        )


if __name__ == "__main__":
    main()
