#!/usr/bin/env python3
"""Debug why CP-SAT returns INFEASIBLE even with 0 frozen genes."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import quick_setup

quick_setup()
logger = logging.getLogger(__name__)

from src.domain.types import SchedulingContext
from src.ga.core.population import generate_hybrid_population, get_family_map_from_json
from src.ga.repair.cp.solver import CPSATSolver
from src.ga.repair.detector import detect_violated_genes
from src.io.loaders import load_data

# Load data
data_dir = PROJECT_ROOT / "data"
ctx: SchedulingContext = load_data(data_dir, "10:00", "17:00", ["Saturday"])
family_map = get_family_map_from_json(str(data_dir / "Groups.json"))

# Create one individual
logger.info("=== Creating test individual ===")
ind = generate_hybrid_population(1, ctx)[0]  # Get first individual from population
logger.info("Total genes: %d", len(ind))

# Detect violations
logger.info("=== Detecting violations ===")
violations = detect_violated_genes(ind, ctx, strategy="hybrid")
logger.info("Violated genes: %d", len(violations))
if violations:
    violated_indices = sorted(violations.keys())
    logger.info("First 10 violated indices: %s", violated_indices[:10])

    # Show violation types
    from collections import Counter

    vtype_counts: Counter[str] = Counter()
    for vtypes in violations.values():
        for vt in vtypes:
            vtype_counts[vt] += 1
    logger.info("Violation type counts: %s", dict(vtype_counts))

# Try CP-SAT with 0 frozen on violated genes
logger.info("=== Testing CP-SAT with 0 frozen genes ===")
if violations:
    violated_indices = sorted(violations.keys())[:50]  # Test with first 50
    logger.info("Testing with %d violated genes", len(violated_indices))

    solver = CPSATSolver(ctx, family_map, timeout_seconds=10, num_workers=4)
    result = solver.solve(ind, violated_indices, frozen=[], warm_start=True)

    logger.info("Status: %s", result.status)
    logger.info("Success: %s", result.success)
    logger.info("Wall time: %.1fs", result.wall_time)

    if not result.success:
        logger.info("=== Why INFEASIBLE? ===")
        # Check for conflicting resource requirements
        violated_genes = [ind[i] for i in violated_indices]

        # Count unique resources needed
        instructors_needed = {g.instructor_id for g in violated_genes}
        rooms_needed = {g.room_id for g in violated_genes}
        groups_affected = set()
        for g in violated_genes:
            groups_affected.update(g.group_ids)

        logger.info("Genes to solve: %d", len(violated_genes))
        logger.info("Unique instructors: %d", len(instructors_needed))
        logger.info("Unique rooms: %d", len(rooms_needed))
        logger.info("Unique groups: %d", len(groups_affected))

        # Check if total duration fits in available quanta
        total_quanta_needed = sum(g.num_quanta for g in violated_genes)
        available_quanta = len(ctx.available_quanta)
        logger.info("Total quanta needed: %d", total_quanta_needed)
        logger.info("Available quanta: %d", available_quanta)
        logger.info("Utilization: %.1f%%", 100 * total_quanta_needed / available_quanta)

        # Check instructor availability mismatches
        part_time_conflicts = 0
        for g in violated_genes:
            instr = ctx.instructors.get(g.instructor_id)
            if instr and not instr.is_full_time:
                occupied = set(range(g.start_quanta, g.start_quanta + g.num_quanta))
                available = instr.available_quanta
                if not occupied.issubset(available):
                    part_time_conflicts += 1
        logger.info("Part-time instructor conflicts: %d", part_time_conflicts)
