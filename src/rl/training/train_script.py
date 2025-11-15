"""
Training entry point script.

Trains RL agent to learn heuristic selection for GA scheduler.

Usage:
    python src/rl/training/train_script.py --timesteps 50000 --agent ppo
    python src/rl/training/train_script.py --timesteps 100000 --agent dqn --save-path models/my_model
    python src/rl/training/train_script.py --help
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.encoder import load_scheduling_data
from src.rl.gym_env import ScheduleEnv, StateEncoder, ActionMapper, RewardCalculator
from src.rl.training import RLTrainer
from src.config import get_config
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RL agent for heuristic selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Number of training timesteps",
    )

    parser.add_argument(
        "--agent",
        "--agent-type",
        dest="agent_type",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="RL agent type",
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Model save path (default: auto-generated)",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save models",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory with JSON files",
    )

    parser.add_argument(
        "--max-generations",
        type=int,
        default=100,
        help="Max generations per episode",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max steps per episode",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="GA population size",
    )

    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes after training",
    )

    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after training",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level (0=none, 1=info, 2=debug)",
    )

    return parser.parse_args()


def create_environment(args, context):
    """
    Create RL environment for training.

    Args:
        args: Command-line arguments
        context: Scheduling context

    Returns:
        ScheduleEnv instance
    """
    logger.info("Creating RL environment...")

    # Create components
    state_encoder = StateEncoder()
    action_mapper = ActionMapper()
    reward_calculator = RewardCalculator()

    # Create environment
    env = ScheduleEnv(
        context=context,
        state_encoder=state_encoder,
        action_mapper=action_mapper,
        reward_calculator=reward_calculator,
        max_generations=args.max_generations,
        max_steps_per_episode=args.max_steps,
        population_size=args.population_size,
    )

    logger.info(
        f"Environment created: max_gen={args.max_generations}, max_steps={args.max_steps}"
    )
    logger.info(f"Observation space: {env.observation_space}")
    logger.info(f"Action space: {env.action_space}")

    return env


def main():
    """Main training function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("RL AGENT TRAINING")
    logger.info("=" * 60)
    logger.info(f"Agent type: {args.agent_type.upper()}")
    logger.info(f"Training timesteps: {args.timesteps:,}")
    logger.info(f"Data directory: {args.data_dir}")

    try:
        # Load scheduling data
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Load Scheduling Data")
        logger.info("=" * 60)

        context = load_scheduling_data(args.data_dir)

        logger.info(f"Loaded {len(context.courses)} courses")
        logger.info(f"Loaded {len(context.instructors)} instructors")
        logger.info(f"Loaded {len(context.rooms)} rooms")
        logger.info(f"Loaded {len(context.groups)} groups")

        # Create environment
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Create RL Environment")
        logger.info("=" * 60)

        env = create_environment(args, context)

        # Create trainer
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Initialize Trainer")
        logger.info("=" * 60)

        trainer = RLTrainer(
            env=env,
            agent_type=args.agent_type,
            save_dir=args.save_dir,
            verbose=args.verbose,
        )

        # Train agent
        logger.info("\n" + "=" * 60)
        logger.info("STEP 4: Train Agent")
        logger.info("=" * 60)

        trainer.train(
            total_timesteps=args.timesteps,
            progress_bar=True,
        )

        # Evaluate agent
        if not args.no_eval and args.eval_episodes > 0:
            logger.info("\n" + "=" * 60)
            logger.info("STEP 5: Evaluate Agent")
            logger.info("=" * 60)

            metrics = trainer.evaluate(n_eval_episodes=args.eval_episodes)

            logger.info("\nEvaluation Results:")
            logger.info(
                f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}"
            )
            logger.info(
                f"  Min/Max Reward: {metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}"
            )
            logger.info(f"  Mean Episode Length: {metrics['mean_length']:.1f}")

        # Save model
        logger.info("\n" + "=" * 60)
        logger.info("STEP 6: Save Model")
        logger.info("=" * 60)

        if args.save_path:
            save_path = args.save_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{args.agent_type}_scheduler_{args.timesteps}_{timestamp}"

        model_path = trainer.save_model(
            filename=save_path,
            metadata={
                "timesteps": args.timesteps,
                "agent_type": args.agent_type,
                "max_generations": args.max_generations,
                "max_steps": args.max_steps,
                "population_size": args.population_size,
            },
        )

        logger.info(f"\n✓ Model saved to: {model_path}")

        # Print training statistics
        stats = trainer.get_training_statistics()
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total timesteps: {stats['total_timesteps']:,}")
        logger.info(
            f"Total training time: {stats['total_training_time']:.1f}s ({stats['total_training_time']/60:.1f} min)"
        )
        logger.info(f"Training runs: {stats['num_training_runs']}")

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)

        # View TensorBoard logs
        logger.info(f"\nTo view training logs in TensorBoard:")
        logger.info(f"  tensorboard --logdir {trainer.tensorboard_log}")
        logger.info(f"  Then open: http://localhost:6006")

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nTraining failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
