"""
Select best checkpoint from training run.

Analyzes checkpoint manifest and selects the best checkpoint
based on validation metrics for production deployment.

Usage:
    python scripts/select_best_checkpoint.py --metric mean_reward --stage hard
    python scripts/select_best_checkpoint.py --metric mean_reward --maximize
"""

import argparse
import json
import sys
from pathlib import Path

# Add src/ to path for local package imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root / "src"))

from schedule_engine.rl.training.checkpoints import CheckpointManager
from schedule_engine.utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Select best checkpoint from training run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default="models/rl_agents/manifest.json",
        help="Path to checkpoint manifest file",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="mean_reward",
        help="Validation metric to optimize",
    )

    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="Filter by curriculum stage (easy, medium, hard)",
    )

    parser.add_argument(
        "--status",
        type=str,
        default="checkpoint",
        help="Filter by checkpoint status",
    )

    parser.add_argument(
        "--maximize",
        action="store_true",
        help="Maximize metric (default: maximize)",
    )

    parser.add_argument(
        "--minimize",
        dest="maximize",
        action="store_false",
        help="Minimize metric",
    )

    parser.add_argument(
        "--promote",
        action="store_true",
        help="Promote selected checkpoint to 'validated' status",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save selection result (JSON)",
    )

    parser.set_defaults(maximize=True)

    return parser.parse_args()


def display_checkpoint_info(checkpoint):
    """Display checkpoint information."""
    logger.info("\n" + "=" * 60)
    logger.info("SELECTED CHECKPOINT")
    logger.info("=" * 60)
    logger.info(f"Checkpoint ID: {checkpoint.checkpoint_id}")
    logger.info(f"Model Path: {checkpoint.model_path}")
    logger.info(f"Timestep: {checkpoint.timestep:,}")
    logger.info(f"Stage: {checkpoint.stage}")
    logger.info(f"Seed: {checkpoint.seed}")
    logger.info(f"Status: {checkpoint.status}")
    logger.info(f"Timestamp: {checkpoint.timestamp}")

    if checkpoint.validation_metrics:
        logger.info("\nValidation Metrics:")
        for key, value in checkpoint.validation_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

    if checkpoint.training_metrics:
        logger.info("\nTraining Metrics:")
        for key, value in checkpoint.training_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

    if checkpoint.notes:
        logger.info(f"\nNotes: {checkpoint.notes}")


def main():
    """Main function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("CHECKPOINT SELECTION")
    logger.info("=" * 60)
    logger.info(f"Manifest: {args.manifest}")
    logger.info(
        f"Metric: {args.metric} ({'maximize' if args.maximize else 'minimize'})"
    )

    if args.stage:
        logger.info(f"Stage filter: {args.stage}")

    logger.info(f"Status filter: {args.status}")

    try:
        # Load checkpoint manager
        manager = CheckpointManager(manifest_path=args.manifest)

        # Get statistics
        stats = manager.get_statistics()
        logger.info(f"\nTotal checkpoints: {stats['total']}")

        if stats["total"] == 0:
            logger.error("No checkpoints found in manifest")
            sys.exit(1)

        logger.info(f"By status: {stats['by_status']}")

        if "by_stage" in stats:
            logger.info(f"By stage: {stats['by_stage']}")

        # Find best checkpoint
        logger.info("\nSearching for best checkpoint...")

        best = manager.get_best_checkpoint(
            metric_name=args.metric,
            stage=args.stage,
            status=args.status,
            maximize=args.maximize,
        )

        if best is None:
            logger.error("No suitable checkpoint found")
            sys.exit(1)

        # Display result
        display_checkpoint_info(best)

        # Promote if requested
        if args.promote:
            logger.info("\n" + "=" * 60)
            logger.info("PROMOTING CHECKPOINT")
            logger.info("=" * 60)

            manager.update_checkpoint_status(
                checkpoint_id=best.checkpoint_id,
                new_status="validated",
                notes=f"Selected as best by {args.metric}={best.validation_metrics[args.metric]:.4f}",
            )

            logger.info("✓ Promoted checkpoint to 'validated' status")

        # Save to output file
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            from dataclasses import asdict

            result = {
                "selected_checkpoint": asdict(best),
                "selection_criteria": {
                    "metric": args.metric,
                    "maximize": args.maximize,
                    "stage": args.stage,
                    "status": args.status,
                },
            }

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            logger.info(f"\n✓ Result saved to: {output_path}")

        logger.info("\n✓ Checkpoint selection complete!")

        # Return path for easy use in scripts
        print(f"\nBest model path: {best.model_path}")

    except Exception as e:
        logger.error(f"Failed to select checkpoint: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
