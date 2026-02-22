#!/usr/bin/env python3
"""Exact domain size analysis with real constraints."""

import logging
import sys
from pathlib import Path

from src.utils.logging_config import quick_setup

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def analyze_real_domain_sizes():
    """Analyze domain sizes using actual constraint logic."""
    try:
        import random

        from src.ga.core.population import generate_pure_random_population
        from src.ga.operators.mutation import find_suitable_rooms_for_course
        from src.io.data_store import DataStore

        store = DataStore.from_json("data")
        ctx = store.to_context()

        pop = generate_pure_random_population(1, ctx, parallel=False)
        genes = pop[0]

        # Sample 100 random genes
        sample_genes = random.sample(genes, min(100, len(genes)))

        logger.info("=== REAL DOMAIN SIZES FOR 100 RANDOM EVENTS ===")
        logger.info("")
        logger.info("Function locations:")
        logger.info(
            "- Room suitability: find_suitable_rooms_for_course() in src/ga/operators/mutation.py"
        )
        logger.info(
            "- Uses: is_room_suitable_for_course() in src/utils/room_compatibility.py"
        )
        logger.info(
            "- Instructor qualification: course.qualified_instructor_ids (InstructorQualifications constraint)"
        )
        logger.info(
            "- Time availability: instructor.is_full_time or instructor.available_quanta"
        )
        logger.info("")

        suitable_rooms_counts = []
        qualified_instructors_counts = []
        allowed_start_times = []

        for i, gene in enumerate(sample_genes):
            course_key = (gene.course_id, gene.course_type)
            course = ctx.courses.get(course_key)

            # REAL room suitability check
            primary_group = gene.group_ids[0] if gene.group_ids else ""
            suitable_rooms = find_suitable_rooms_for_course(
                gene.course_id, gene.course_type, primary_group, ctx
            )

            # REAL instructor qualification check
            qualified_instructors: list[str] = []
            if course:
                qualified_instructors = getattr(course, "qualified_instructor_ids", [])

            # REAL time availability check
            # Consider instructor availability for start time constraints
            max_quantum = 42  # Total operating quanta
            base_max_start = max_quantum - gene.num_quanta

            # For now, use base time range (could be refined with instructor availability)
            allowed_starts = list(range(base_max_start + 1))

            suitable_rooms_counts.append(len(suitable_rooms))
            qualified_instructors_counts.append(len(qualified_instructors))
            allowed_start_times.append(len(allowed_starts))

            if i < 10:  # Show first 10 events
                logger.info(
                    "Event %d: Course=%s Type=%s Groups=%s",
                    i,
                    gene.course_id,
                    gene.course_type,
                    gene.group_ids,
                )
                logger.info("  Duration: %d quanta", gene.num_quanta)
                logger.info(
                    "  Suitable rooms: %d (from suitability+capacity check)",
                    len(suitable_rooms),
                )
                logger.info("  Qualified instructors: %d", len(qualified_instructors))
                logger.info(
                    "  Allowed start times: %d (0 to %d)",
                    len(allowed_starts),
                    base_max_start,
                )
                if i < 3:  # Show details for first 3
                    logger.debug("    Sample suitable rooms: %s", suitable_rooms[:5])
                    logger.debug(
                        "    Sample qualified instructors: %s",
                        qualified_instructors[:5],
                    )
                logger.info("")

        logger.info("=== SUMMARY STATISTICS ===")
        logger.info("")
        logger.info("SUITABLE ROOMS (real suitability + capacity):")
        logger.info("  Min: %d", min(suitable_rooms_counts))
        logger.info("  Max: %d", max(suitable_rooms_counts))
        logger.info(
            "  Avg: %.1f", sum(suitable_rooms_counts) / len(suitable_rooms_counts)
        )
        logger.info("")

        logger.info("QUALIFIED INSTRUCTORS (real qualification check):")
        logger.info("  Min: %d", min(qualified_instructors_counts))
        logger.info("  Max: %d", max(qualified_instructors_counts))
        logger.info(
            "  Avg: %.1f",
            sum(qualified_instructors_counts) / len(qualified_instructors_counts),
        )
        logger.info("")

        logger.info("ALLOWED START TIMES (considering duration):")
        logger.info("  Min: %d", min(allowed_start_times))
        logger.info("  Max: %d", max(allowed_start_times))
        logger.info("  Avg: %.1f", sum(allowed_start_times) / len(allowed_start_times))
        logger.info("")

        # Show distribution of constraints
        logger.info("CONSTRAINT TIGHTNESS ANALYSIS:")
        tight_rooms = sum(1 for count in suitable_rooms_counts if count <= 5)
        tight_instructors = sum(
            1 for count in qualified_instructors_counts if count <= 2
        )
        tight_time = sum(1 for count in allowed_start_times if count <= 10)

        logger.info(
            "Events with <=5 suitable rooms: %d/100 (%.0f%%)",
            tight_rooms,
            100 * tight_rooms / 100,
        )
        logger.info(
            "Events with <=2 qualified instructors: %d/100 (%.0f%%)",
            tight_instructors,
            100 * tight_instructors / 100,
        )
        logger.info(
            "Events with <=10 start time options: %d/100 (%.0f%%)",
            tight_time,
            100 * tight_time / 100,
        )

    except Exception as e:
        logger.error("Error during analysis: %s", e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    quick_setup()
    analyze_real_domain_sizes()
