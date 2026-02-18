#!/usr/bin/env python3
"""Debug why CP-SAT returns INFEASIBLE even with 0 frozen genes."""

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.DEBUG, format="%(message)s")

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
print("\n=== Creating test individual ===")
ind = generate_hybrid_population(1, ctx)[0]  # Get first individual from population
print(f"Total genes: {len(ind)}")

# Detect violations
print("\n=== Detecting violations ===")
violations = detect_violated_genes(ind, ctx, strategy="hybrid")
print(f"Violated genes: {len(violations)}")
if violations:
    violated_indices = sorted(violations.keys())
    print(f"First 10 violated indices: {violated_indices[:10]}")

    # Show violation types
    from collections import Counter

    vtype_counts: Counter[str] = Counter()
    for vtypes in violations.values():
        for vt in vtypes:
            vtype_counts[vt] += 1
    print(f"Violation type counts: {dict(vtype_counts)}")

# Try CP-SAT with 0 frozen on violated genes
print("\n=== Testing CP-SAT with 0 frozen genes ===")
if violations:
    violated_indices = sorted(violations.keys())[:50]  # Test with first 50
    print(f"Testing with {len(violated_indices)} violated genes")

    solver = CPSATSolver(ctx, family_map, timeout_seconds=10, num_workers=4)
    result = solver.solve(ind, violated_indices, frozen=[], warm_start=True)

    print(f"Status: {result.status}")
    print(f"Success: {result.success}")
    print(f"Wall time: {result.wall_time:.1f}s")

    if not result.success:
        print("\n=== Why INFEASIBLE? ===")
        # Check for conflicting resource requirements
        violated_genes = [ind[i] for i in violated_indices]

        # Count unique resources needed
        instructors_needed = {g.instructor_id for g in violated_genes}
        rooms_needed = {g.room_id for g in violated_genes}
        groups_affected = set()
        for g in violated_genes:
            groups_affected.update(g.group_ids)

        print(f"Genes to solve: {len(violated_genes)}")
        print(f"Unique instructors: {len(instructors_needed)}")
        print(f"Unique rooms: {len(rooms_needed)}")
        print(f"Unique groups: {len(groups_affected)}")

        # Check if total duration fits in available quanta
        total_quanta_needed = sum(g.num_quanta for g in violated_genes)
        available_quanta = len(ctx.available_quanta)
        print(f"Total quanta needed: {total_quanta_needed}")
        print(f"Available quanta: {available_quanta}")
        print(f"Utilization: {100*total_quanta_needed/available_quanta:.1f}%")

        # Check instructor availability mismatches
        part_time_conflicts = 0
        for g in violated_genes:
            instr = ctx.instructors.get(g.instructor_id)
            if instr and not instr.is_full_time:
                occupied = set(range(g.start_quanta, g.start_quanta + g.num_quanta))
                available = instr.available_quanta
                if not occupied.issubset(available):
                    part_time_conflicts += 1
        print(f"Part-time instructor conflicts: {part_time_conflicts}")
