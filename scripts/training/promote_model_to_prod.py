"""
Promote validated RL model to production.

This script promotes a validated checkpoint to production by:
1. Updating configs/prod.yaml with new model path
2. Recording deployment in registry.json
3. Creating backup of previous configuration

Usage:
    # Promote best checkpoint from manifest
    python scripts/promote_model_to_prod.py --checkpoint-id ppo_stage3_20250115_123045

    # Promote specific model file
    python scripts/promote_model_to_prod.py --model-path models/rl_agents/best_model.zip --agent-type ppo

    # Rollback to previous deployment
    python scripts/promote_model_to_prod.py --rollback
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rl.deployment.registry import ModelRegistry
from src.rl.training.checkpoints import CheckpointManager
from src.config import get_config
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)


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
    config = get_config()

    # Load checkpoint from manifest
    manifest_path = config.rl.training.checkpoint_settings.manifest_path
    manager = CheckpointManager(manifest_path)

    checkpoint = manager.get_checkpoint(checkpoint_id)
    if not checkpoint:
        logger.error(f"Checkpoint not found: {checkpoint_id}")
        print(f" Checkpoint not found: {checkpoint_id}")
        print(f"\nAvailable checkpoints:")
        for ckpt in manager.list_checkpoints():
            print(f"  - {ckpt.checkpoint_id} ({ckpt.stage}, {ckpt.status})")
        sys.exit(1)

    # Validate checkpoint file exists
    checkpoint_path = Path(checkpoint.model_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint file not found: {checkpoint.model_path}")
        print(f" Checkpoint file not found: {checkpoint.model_path}")
        sys.exit(1)

    # Validate checkpoint is a valid model
    try:
        from stable_baselines3 import PPO, DQN

        if checkpoint.agent_type.lower() == "ppo":
            _ = PPO.load(checkpoint.model_path)
        elif checkpoint.agent_type.lower() == "dqn":
            _ = DQN.load(checkpoint.model_path)
        else:
            logger.error(f"Unknown agent type: {checkpoint.agent_type}")
            print(f" Unknown agent type: {checkpoint.agent_type}")
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
            f"⚠️  Checkpoint status is '{checkpoint.status}'. Continue? (y/N): "
        )
        if confirm.lower() != "y":
            print("Promotion cancelled")
            sys.exit(0)

    print(f" Promoting checkpoint: {checkpoint_id}")
    print(f"   Model: {checkpoint.model_path}")
    print(f"   Agent: {checkpoint.agent_type}")
    print(f"   Stage: {checkpoint.stage}")
    print(f"   Mean Reward: {checkpoint.validation_metrics.get('mean_reward', 'N/A')}")
    print()

    # Initialize registry
    prod_config_path = project_root / "configs" / "prod.yaml"
    registry_path = project_root / "models" / "rl_agents" / "registry.json"
    registry = ModelRegistry(prod_config_path, registry_path)

    # Promote model
    try:
        registration = registry.promote_model(
            model_path=checkpoint.model_path,
            agent_type=checkpoint.agent_type,
            validation_metrics=checkpoint.validation_metrics,
            promoted_by=promoted_by,
            notes=notes or f"Promoted from checkpoint {checkpoint_id}",
            checkpoint_id=checkpoint_id,
        )

        print(f" Model promoted successfully!")
        print(f"   Deployment ID: {registration.model_id}")
        print(f"   Config updated: {prod_config_path}")
        print(f"   Registry updated: {registry_path}")
        print()
        print(f" To use in production, run:")
        print(f"   uv run prod")

    except Exception as e:
        logger.exception("Promotion failed")
        print(f" Promotion failed: {e}")
        sys.exit(1)


def promote_from_file(
    model_path: str,
    agent_type: str,
    validation_metrics: dict,
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
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        print(f" Model file not found: {model_path}")
        sys.exit(1)

    print(f" Promoting model file: {model_path}")
    print(f"   Agent: {agent_type}")
    print()

    # Initialize registry
    prod_config_path = project_root / "configs" / "prod.yaml"
    registry_path = project_root / "models" / "rl_agents" / "registry.json"
    registry = ModelRegistry(prod_config_path, registry_path)

    # Promote model
    try:
        registration = registry.promote_model(
            model_path=model_path,
            agent_type=agent_type,
            validation_metrics=validation_metrics,
            promoted_by=promoted_by,
            notes=notes,
        )

        print(f" Model promoted successfully!")
        print(f"   Deployment ID: {registration.model_id}")
        print(f"   Config updated: {prod_config_path}")
        print(f"   Registry updated: {registry_path}")
        print()
        print(f" To use in production, run:")
        print(f"   uv run prod")

    except Exception as e:
        logger.exception("Promotion failed")
        print(f" Promotion failed: {e}")
        sys.exit(1)


def rollback_deployment() -> None:
    """Rollback to previous deployment."""
    prod_config_path = project_root / "configs" / "prod.yaml"
    registry_path = project_root / "models" / "rl_agents" / "registry.json"
    registry = ModelRegistry(prod_config_path, registry_path)

    # Get current deployment
    current = registry.get_active_deployment()
    if not current:
        print(" No active deployment found")
        sys.exit(1)

    print(f"Current deployment: {current.model_id}")
    print(f"   Model: {current.model_path}")
    print(f"   Deployed: {current.deployed_at}")
    print()

    confirm = input("⚠️  Rollback to previous deployment? (y/N): ")
    if confirm.lower() != "y":
        print("Rollback cancelled")
        sys.exit(0)

    try:
        registration = registry.rollback_to_previous()
        if registration:
            print(f" Rolled back successfully!")
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
    prod_config_path = project_root / "configs" / "prod.yaml"
    registry_path = project_root / "models" / "rl_agents" / "registry.json"
    registry = ModelRegistry(prod_config_path, registry_path)

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
        validation_metrics = {
            "mean_reward": 0.0,
            "note": "Promoted manually without validation metrics",
        }

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
