#!/usr/bin/env python3
"""Simple individual inspection script - no multiprocessing."""

import logging
import sys
from pathlib import Path

from src.utils.logging_config import quick_setup

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def inspect_individual():
    """Create and inspect a single individual."""
    try:
        # Import required modules
        from src.ga.core.evaluator import evaluate
        from src.ga.core.population import generate_pure_random_population
        from src.io.data_store import DataStore
        from src.io.decoder import decode_individual
        from src.io.time_system import QuantumTimeSystem

        logger.info("Loading data...")
        data_store = DataStore.from_json(PROJECT_ROOT / "data")
        courses, instructors, groups, rooms = (
            data_store.courses,
            data_store.instructors,
            data_store.groups,
            data_store.rooms,
        )
        context = data_store.to_context()

        logger.info("Data loaded:")
        logger.info("  Courses: %d", len(courses))
        logger.info("  Instructors: %d", len(instructors))
        logger.info("  Groups: %d", len(groups))
        logger.info("  Rooms: %d", len(rooms))

        logger.info("\nGenerating individual...")
        population = generate_pure_random_population(1, context, parallel=False)
        individual = population[0]
        logger.info("Individual size (genes): %d", len(individual))

        # Print first few genes
        logger.info("\nFirst 5 genes:")
        for i, gene in enumerate(individual[:5]):
            logger.info(
                "  Gene %d: Course=%s, Groups=%s, Instructor=%s, Room=%s, Start=%s, Num=%s",
                i, gene.course_id, gene.group_ids,
                gene.instructor_id, gene.room_id,
                gene.start_quanta, gene.num_quanta
            )

        # Quick evaluation test
        logger.info("\nSingle evaluation test...")
        hard, soft = evaluate(individual, courses, instructors, groups, rooms)
        logger.info("Fitness: Hard=%s, Soft=%s", hard, soft)

        # Print schedule snippet for first few genes
        logger.info("\nDecoded schedule (first 5 sessions):")
        decode_individual(individual, courses, instructors, groups, rooms)
        qts = QuantumTimeSystem()
        for i, gene in enumerate(individual[:5]):
            day, start_time = qts.quanta_to_time(gene.start_quanta)
            _, end_time = qts.quanta_to_time(gene.start_quanta + gene.num_quanta - 1)
            time_str = f"{day} {start_time}-{end_time}"
            logger.info(
                "  %s - %s - %s - %s - %s",
                gene.course_id, gene.group_ids, gene.instructor_id, gene.room_id, time_str
            )

    except Exception as e:
        logger.error("Error during inspection: %s", e)
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    quick_setup()
    inspect_individual()
