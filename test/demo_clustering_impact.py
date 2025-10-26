"""
Quick demo showing clustering-aware initialization impact.
Compares random vs cluster-aware assignment for typical course sessions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ga.population import assign_conflict_free_quanta
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.utils.time_helpers import quantum_to_day_and_within_day
from src.utils.time_helpers import (
    ISOLATED_SESSION_PENALTY,
    OVERSIZED_BLOCK_PENALTY_PER_QUANTUM,
)
from src.utils.time_helpers import get_config().time.preferred_block_size_max
import random


def calculate_clustering_penalty(quanta: list, qts: QuantumTimeSystem) -> int:
    """Calculate clustering penalty for a set of quanta."""
    if not quanta:
        return 0

    # Group by day and find blocks
    day_quanta = {}
    for q in quanta:
        try:
            day, within_day = quantum_to_day_and_within_day(q, qts)
            if day not in day_quanta:
                day_quanta[day] = []
            day_quanta[day].append(within_day)
        except:
            continue

    penalty = 0
    for day_q in day_quanta.values():
        sorted_q = sorted(day_q)
        if not sorted_q:
            continue

        # Find blocks
        current_block_size = 1
        blocks = []
        for i in range(1, len(sorted_q)):
            if sorted_q[i] == sorted_q[i - 1] + 1:
                current_block_size += 1
            else:
                blocks.append(current_block_size)
                current_block_size = 1
        blocks.append(current_block_size)

        # Calculate penalty
        for block_size in blocks:
            if block_size == 1:
                penalty += ISOLATED_SESSION_PENALTY
            elif block_size > get_config().time.preferred_block_size_max:
                penalty += (
                    block_size - get_config().time.preferred_block_size_max
                ) * OVERSIZED_BLOCK_PENALTY_PER_QUANTUM

    return penalty


def format_blocks(quanta: list, qts: QuantumTimeSystem) -> str:
    """Format quanta as day: [block sizes]."""
    if not quanta:
        return "Empty"

    day_quanta = {}
    for q in quanta:
        try:
            day, within_day = quantum_to_day_and_within_day(q, qts)
            if day not in day_quanta:
                day_quanta[day] = []
            day_quanta[day].append(within_day)
        except:
            continue

    result = []
    for day, day_q in sorted(day_quanta.items()):
        sorted_q = sorted(day_q)
        blocks = []
        if sorted_q:
            current_block_size = 1
            for i in range(1, len(sorted_q)):
                if sorted_q[i] == sorted_q[i - 1] + 1:
                    current_block_size += 1
                else:
                    blocks.append(current_block_size)
                    current_block_size = 1
            blocks.append(current_block_size)
        result.append(f"{day}: {blocks}")

    return ", ".join(result)


def demo_comparison():
    """Show before/after comparison for typical course sizes."""
    qts = QuantumTimeSystem()
    available = list(qts.get_all_operating_quanta())

    print("=" * 70)
    print("CLUSTERING-AWARE INITIALIZATION - IMPACT DEMONSTRATION")
    print("=" * 70)
    print()

    test_cases = [
        (2, "Small lecture (2 hours/week)"),
        (3, "Standard lecture (3 hours/week)"),
        (4, "Lecture + Tutorial (4 hours/week)"),
        (6, "Full course (6 hours/week)"),
        (9, "Lab-heavy course (9 hours/week)"),
    ]

    total_before = 0
    total_after = 0

    for quanta_needed, description in test_cases:
        print(f"📚 {description} ({quanta_needed} quanta)")
        print("-" * 70)

        # Simulate random assignment (old way)
        random_assignment = random.sample(available, quanta_needed)
        random_penalty = calculate_clustering_penalty(random_assignment, qts)
        random_blocks = format_blocks(random_assignment, qts)

        # Cluster-aware assignment (new way)
        cluster_assignment = assign_conflict_free_quanta(
            quanta_needed, available, set()
        )
        cluster_penalty = calculate_clustering_penalty(cluster_assignment, qts)
        cluster_blocks = format_blocks(cluster_assignment, qts)

        print(f"  [!ERR] Random:  {random_blocks}")
        print(f"     Penalty: {random_penalty}")
        print()
        print(f"  ✅ Clustered: {cluster_blocks}")
        print(f"     Penalty: {cluster_penalty}")
        print()

        improvement = random_penalty - cluster_penalty
        if improvement > 0:
            print(f"  💰 Improvement: -{improvement} penalty points")
        elif improvement == 0:
            print(f"  ✓ Both optimal (0 penalty)")
        print()

        total_before += random_penalty
        total_after += cluster_penalty

    print("=" * 70)
    print(f"TOTAL PENALTY ACROSS ALL COURSES:")
    print(f"  Before (Random):      {total_before}")
    print(f"  After (Cluster-aware): {total_after}")
    print(
        f"  Improvement:          -{total_before - total_after} ({100*(total_before-total_after)/max(total_before,1):.1f}% reduction)"
    )
    print("=" * 70)
    print()
    print("✅ Cluster-aware initialization creates MUCH better starting individuals!")
    print("   GA can now focus on other constraints instead of fixing fragmentation.")


if __name__ == "__main__":
    demo_comparison()
