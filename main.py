"""
Pure CP-SAT Schedule Engine Entry Point

Hard constraint satisfaction using Google OR-Tools CP-SAT solver only.
No genetic algorithms, no soft constraints, no overhead.
"""

import sys
import json
import logging
import yaml
import argparse
from pathlib import Path
from datetime import datetime

from src.ortools.cp_scheduler_clean import CPScheduler
from src.workflows.standard_run import load_input_data
from src.validation.input_validator import validate_input


def load_config(config_file: str = "configs/cpsat.prod.yaml") -> dict:
    """
    Load configuration from YAML file with base config support.

    If config contains 'base' key, loads base config first and merges.
    """
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Check if this config extends a base config
    if "base" in config:
        base_file = config.pop("base")
        base_path = Path(config_file).parent / base_file

        with open(base_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)

        # Deep merge: config overrides base
        def merge_dict(base, override):
            result = base.copy()
            for key, value in override.items():
                if (
                    key in result
                    and isinstance(result[key], dict)
                    and isinstance(value, dict)
                ):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result

        config = merge_dict(base_config, config)

    return config


def setup_logging(output_dir: Path, config: dict) -> logging.Logger:
    """Configure clean logging."""
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"cpsat_{timestamp}.log"

    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s | %(levelname)-8s | %(message)s")
    date_format = log_config.get("date_format", "%Y-%m-%d %H:%M:%S")

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("CP-SAT Schedule Engine - Pure Constraint Programming")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    return logger


def export_schedule_json(schedule, output_file: str) -> None:
    """Export schedule to JSON."""
    schedule_data = []
    for session in schedule:
        schedule_data.append(
            {
                "course_id": session.course_id,
                "course_name": session.course_name,
                "group_ids": session.group_ids,
                "instructor_name": session.instructor_name,
                "room_name": session.room_name,
                "quanta": session.quanta,
                "duration_minutes": session.duration_minutes,
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(schedule_data, f, indent=2, ensure_ascii=False)


def main():
    """Run pure CP-SAT scheduler."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="CP-SAT Schedule Engine")
    parser.add_argument(
        "--config",
        default="configs/cpsat.prod.yaml",
        help="Configuration file (default: configs/cpsat.prod.yaml)",
    )
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    io_config = config.get("io", {})
    solver_config = config.get("solver", {})

    DATA_DIR = io_config.get("data_dir", "data")
    OUTPUT_DIR = Path(io_config.get("output_dir", "output"))
    TIME_LIMIT = solver_config.get("time_limit", 0)
    WORKERS = solver_config.get("num_workers", 4)
    RANDOM_SEED = solver_config.get("random_seed")

    # Setup
    logger = setup_logging(OUTPUT_DIR, config)
    logger.info(f"Configuration: {args.config}")
    logger.info("-" * 80)

    try:
        # Step 1: Load input data
        logger.info("STEP 1: Loading input data")
        logger.info("-" * 80)

        qts, context = load_input_data(DATA_DIR)

        logger.info(f"  Courses: {len(context.courses)}")
        logger.info(f"  Groups: {len(context.groups)}")
        logger.info(f"  Instructors: {len(context.instructors)}")
        logger.info(f"  Rooms: {len(context.rooms)}")
        logger.info(f"  Time quanta: {len(qts.get_all_operating_quanta())}")
        logger.info("-" * 80)

        # Step 2: Validate input
        logger.info("STEP 2: Input validation")
        logger.info("-" * 80)

        if not validate_input(context):
            logger.error("Input validation failed!")
            return 1

        logger.info("  [OK] Input validation passed")
        logger.info("-" * 80)

        # Step 3: Run CP-SAT solver
        logger.info("STEP 3: Running CP-SAT solver")
        logger.info(
            f"  Time limit: {'UNLIMITED' if TIME_LIMIT == 0 else f'{TIME_LIMIT}s'}"
        )
        logger.info(f"  Parallel workers: {WORKERS}")
        logger.info("-" * 80)

        scheduler = CPScheduler(
            context=context,
            qts=qts,
            time_limit_seconds=TIME_LIMIT,
            num_workers=WORKERS,
            random_seed=RANDOM_SEED,
        )

        schedule = scheduler.generate_single_solution()

        logger.info("-" * 80)
        logger.info(f"  [OK] Solution found with {len(schedule)} sessions")
        logger.info("-" * 80)

        # Step 4: Export results
        logger.info("STEP 4: Exporting results")
        logger.info("-" * 80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"schedule_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)

        # Export JSON
        json_file = output_path / "schedule.json"
        export_schedule_json(schedule, str(json_file))
        logger.info(f"  [OK] JSON: {json_file}")

        logger.info("-" * 80)
        logger.info("=" * 80)
        logger.info("CP-SAT SCHEDULER COMPLETED SUCCESSFULLY")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("\n" + "=" * 80)
        logger.warning("Interrupted by user (Ctrl+C)")
        logger.warning("=" * 80)
        return 130

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        logger.error("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
