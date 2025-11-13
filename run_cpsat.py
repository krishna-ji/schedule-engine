"""
Pure CP-SAT Runner - No Configuration Required
Clean command-line interface for constraint programming
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

from src.ortools.cp_scheduler_clean import CPScheduler
from src.encoder.input_encoder import encode_input
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.validation.input_validator import validate_input


def setup_logging(output_dir: Path) -> None:
    """Configure clean logging."""
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"cpsat_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("CP-SAT Pure Constraint Programming Scheduler")
    logger.info("=" * 80)


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

    # Simple configuration
    DATA_DIR = "data"
    OUTPUT_DIR = Path("output")
    TIME_LIMIT = 0  # unlimited
    WORKERS = 4  # memory-safe
    QUANTUM_MINUTES = 60

    # Setup
    setup_logging(OUTPUT_DIR)
    logger = logging.getLogger(__name__)

    try:
        # Load data
        logger.info("Loading input data...")
        qts = QuantumTimeSystem(quantum_minutes=QUANTUM_MINUTES)
        context = encode_input(DATA_DIR, qts)

        logger.info(f"  Courses: {len(context.courses)}")
        logger.info(f"  Groups: {len(context.groups)}")
        logger.info(f"  Instructors: {len(context.instructors)}")
        logger.info(f"  Rooms: {len(context.rooms)}")
        logger.info(f"  Quanta: {len(qts.get_all_operating_quanta())}")
        logger.info("-" * 80)

        # Validate
        logger.info("Validating input...")
        if not validate_input(context):
            logger.error("Validation failed!")
            return 1
        logger.info("  ✓ Validation passed")
        logger.info("-" * 80)

        # Solve
        scheduler = CPScheduler(
            context=context, qts=qts, time_limit_seconds=TIME_LIMIT, num_workers=WORKERS
        )

        schedule = scheduler.generate_single_solution()
        logger.info("-" * 80)

        # Export
        logger.info("Exporting results...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"schedule_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)

        json_file = output_path / "schedule.json"
        export_schedule_json(schedule, str(json_file))
        logger.info(f"  JSON: {json_file}")

        logger.info("=" * 80)
        logger.info(f"✓ SUCCESS - Output: {output_path}")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 130

    except Exception as e:
        logger.exception(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
