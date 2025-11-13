"""
Pure CP-SAT Scheduler - Clean Implementation
No GA, no soft constraints, just hard constraint satisfaction
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from src.ortools.cp_scheduler import CPScheduler
from src.encoder.input_encoder import encode_input
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.validation.input_validator import validate_input
from src.exporter.json_exporter import export_schedule_json
from src.exporter.pdf_exporter import PDFScheduleExporter


def setup_logging(output_dir: str = "output") -> logging.Logger:
    """Setup clean logging configuration."""
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(output_dir) / f"cpsat_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("CP-SAT Scheduler - Pure Constraint Programming")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    return logger


def main():
    """Pure CP-SAT scheduling workflow."""
    parser = argparse.ArgumentParser(description="Pure CP-SAT Course Scheduler")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Input data directory (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output/)",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=0,
        help="Time limit in seconds (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--quantum-minutes",
        type=int,
        default=60,
        help="Time quantum in minutes (default: 60)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.output_dir)

    try:
        # Step 1: Load input data
        logger.info("STEP 1: Loading input data")
        logger.info("-" * 80)

        qts = QuantumTimeSystem(quantum_minutes=args.quantum_minutes)
        context = encode_input(args.data_dir, qts)

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

        logger.info("  ✓ Input validation passed")
        logger.info("-" * 80)

        # Step 3: Run CP-SAT solver
        logger.info("STEP 3: Running CP-SAT solver")
        logger.info(
            f"  Time limit: {'UNLIMITED' if args.time_limit == 0 else f'{args.time_limit}s'}"
        )
        logger.info(f"  Parallel workers: {args.workers}")
        logger.info("-" * 80)

        scheduler = CPScheduler(
            context=context,
            qts=qts,
            time_limit_seconds=args.time_limit,
            num_workers=args.workers,
        )

        schedule = scheduler.generate_single_solution()

        logger.info("-" * 80)
        logger.info(f"  ✓ Solution found with {len(schedule)} sessions")
        logger.info("-" * 80)

        # Step 4: Export results
        logger.info("STEP 4: Exporting results")
        logger.info("-" * 80)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output_dir) / f"schedule_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)

        # Export JSON
        json_file = output_path / "schedule.json"
        export_schedule_json(schedule, str(json_file))
        logger.info(f"  ✓ JSON exported: {json_file}")

        # Export PDF
        pdf_file = output_path / "schedule.pdf"
        exporter = PDFScheduleExporter(context, qts)
        exporter.export(schedule, str(pdf_file))
        logger.info(f"  ✓ PDF exported: {pdf_file}")

        logger.info("-" * 80)
        logger.info("=" * 80)
        logger.info("CP-SAT SCHEDULER COMPLETED SUCCESSFULLY")
        logger.info(f"Output directory: {output_path}")
        logger.info("=" * 80)

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C)")
        return 130

    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
