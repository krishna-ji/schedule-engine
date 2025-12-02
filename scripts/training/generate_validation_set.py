"""
Generate validation dataset for RL training.

Creates hold-out validation problems for each curriculum stage
to enable unbiased evaluation of checkpoint performance.

Usage:
    python scripts/generate_validation_set.py --stage easy --num-problems 10
    python scripts/generate_validation_set.py --stage all --output data/validation/
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.encoder import SchedulingContext, load_scheduling_data  # noqa: E402
from src.rl.training.curriculum import create_default_curriculum  # noqa: E402
from src.utils.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate validation dataset for RL training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["easy", "medium", "hard", "all"],
        help="Curriculum stage to generate validation set for",
    )

    parser.add_argument(
        "--num-problems",
        type=int,
        default=10,
        help="Number of validation problems per stage",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Input data directory with JSON files",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/validation",
        help="Output directory for validation datasets",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def generate_validation_problems(
    context: SchedulingContext,
    num_courses: int,
    num_problems: int,
    seed: int,
) -> list:
    """
    Generate validation problems of specified difficulty.

    Args:
        context: Full scheduling context
        num_courses: Target number of courses per problem
        num_problems: Number of validation problems
        seed: Random seed

    Returns:
        List of problem configurations
    """
    random.seed(seed)
    all_courses = context.courses

    if len(all_courses) < num_courses:
        logger.warning(
            f"Requested {num_courses} courses but only {len(all_courses)} available. "
            f"Using all courses."
        )
        num_courses = len(all_courses)

    problems = []

    for i in range(num_problems):
        # Sample courses randomly
        sampled_courses = random.sample(all_courses, num_courses)

        # Create problem configuration
        problem = {
            "problem_id": f"val_{num_courses}c_p{i + 1:03d}",
            "num_courses": num_courses,
            "course_ids": [c.id for c in sampled_courses],
            "seed": seed + i,
            "description": (
                f"Validation problem {i + 1}/{num_problems} with {num_courses} courses"
            ),
        }

        problems.append(problem)

    return problems


def main():
    """Main function."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("VALIDATION SET GENERATION")
    logger.info("=" * 60)
    logger.info(f"Stage: {args.stage}")
    logger.info(f"Problems per stage: {args.num_problems}")
    logger.info(f"Random seed: {args.seed}")

    try:
        # Load data
        logger.info("\nLoading scheduling data...")
        context = load_scheduling_data(args.data_dir)

        logger.info(f"Loaded {len(context.courses)} courses")
        logger.info(f"Loaded {len(context.instructors)} instructors")
        logger.info(f"Loaded {len(context.rooms)} rooms")

        # Get curriculum stages
        curriculum_config = create_default_curriculum()

        # Determine which stages to generate
        if args.stage == "all":
            stages_to_generate = curriculum_config
        else:
            stages_to_generate = [
                s for s in curriculum_config if s["name"] == args.stage
            ]

        if not stages_to_generate:
            logger.error(f"Stage '{args.stage}' not found in curriculum")
            sys.exit(1)

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate validation sets
        all_results = {}

        for stage_config in stages_to_generate:
            stage_name = stage_config["name"]
            num_courses = stage_config["sample_config"]["num_courses"]

            logger.info(
                f"\nGenerating validation set for stage '{stage_name}' ({num_courses} courses)..."
            )

            problems = generate_validation_problems(
                context=context,
                num_courses=num_courses,
                num_problems=args.num_problems,
                seed=args.seed,
            )

            # Save to file
            output_file = output_dir / f"validation_{stage_name}.json"

            validation_data = {
                "stage": stage_name,
                "num_courses": num_courses,
                "num_problems": len(problems),
                "seed": args.seed,
                "problems": problems,
            }

            with open(output_file, "w") as f:
                json.dump(validation_data, f, indent=2)

            logger.info(f"âœ“ Saved {len(problems)} problems to {output_file}")

            all_results[stage_name] = {
                "num_problems": len(problems),
                "output_file": str(output_file),
            }

        # Save summary
        summary_file = output_dir / "validation_summary.json"

        summary = {
            "total_stages": len(all_results),
            "total_problems": sum(r["num_problems"] for r in all_results.values()),
            "seed": args.seed,
            "stages": all_results,
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total stages: {summary['total_stages']}")
        logger.info(f"Total problems: {summary['total_problems']}")
        logger.info(f"Summary saved to: {summary_file}")

        for stage_name, result in all_results.items():
            logger.info(f"  {stage_name}: {result['num_problems']} problems")

        logger.info("\nâœ“ Validation set generation complete!")

    except Exception as e:
        logger.error(f"Failed to generate validation set: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
