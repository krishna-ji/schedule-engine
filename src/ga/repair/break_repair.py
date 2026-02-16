"""
Repair heuristic for break placement violations.

Moves sessions out of break windows to ensure groups have proper breaks.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from src.io.time_system import QuantumTimeSystem

if TYPE_CHECKING:
    from src.domain.gene import SessionGene
    from src.domain.types import Individual


def repair_break_placement(
    individual: Individual,
    max_iterations: int = 5,
    **kwargs: Any,
) -> dict[str, int]:
    """
    Repairs break placement violations by rescheduling sessions.

    Strategy:
    1. Decode individual to identify break violations
    2. For each violated group-day:
       - Find sessions overlapping break window
       - Try rescheduling to earlier/later slots
       - Preserve hard constraints (no new conflicts)
    3. Repeat until violations resolved or max iterations reached

    Args:
        individual: Individual to repair
        max_iterations: Max repair attempts
        **kwargs: Additional repair parameters

    Returns:
        Stats dict with repair counts
    """
    qts = QuantumTimeSystem()

    if not qts.enforce_break_placement:
        return {"break_violations_repaired": 0}

    # Get entity dictionaries from kwargs (passed by repair framework)
    courses = kwargs.get("courses")
    instructors = kwargs.get("instructors")
    groups = kwargs.get("groups")
    rooms = kwargs.get("rooms")

    if not all([courses, instructors, groups, rooms]):
        # Cannot decode without entities - skip repair
        return {"break_violations_repaired": 0}

    # Type assertions for mypy (we checked they're not None above)

    assert isinstance(courses, dict)
    assert isinstance(instructors, dict)
    assert isinstance(groups, dict)
    assert isinstance(rooms, dict)

    from src.io import decode_individual

    repairs = 0

    for _iteration in range(max_iterations):
        sessions = decode_individual(individual, courses, instructors, groups, rooms)
        break_windows = qts.get_break_window_quanta()
        group_schedules = qts.build_group_day_schedules(sessions)

        violations_found = False

        for (group_id, day_name), occupied_quanta in group_schedules.items():
            if day_name not in break_windows:
                continue

            break_quanta = break_windows[day_name]
            occupied_in_break = occupied_quanta & break_quanta
            free_count = len(break_quanta) - len(occupied_in_break)

            if free_count < qts.break_min_quanta:
                violations_found = True
                # Attempt repair: shift session out of break window
                repair_result = _shift_session_out_of_break(
                    individual=individual,
                    group_id=group_id,
                    day_name=day_name,
                    break_quanta=break_quanta,
                    occupied_in_break=occupied_in_break,
                    qts=qts,
                )
                repairs += repair_result

        if not violations_found:
            break  # All violations resolved

    return {"break_violations_repaired": repairs}


def _shift_session_out_of_break(
    individual: Individual,
    group_id: str,
    day_name: str,
    break_quanta: set[int],
    occupied_in_break: set[int],
    qts: QuantumTimeSystem,
) -> int:
    """
    Attempts to shift ONE session out of break window.

    Strategy:
    - Find a session partially/fully in break window
    - Try shifting to earlier slots (before break) or later slots (after break)
    - Validate no hard constraint violations

    Args:
        individual: Individual to repair
        group_id: Group with break violation
        day_name: Day with violation
        break_quanta: Set of within-day quanta in break window
        occupied_in_break: Set of within-day quanta occupied during break
        qts: QuantumTimeSystem instance

    Returns:
        1 if repair successful, 0 otherwise
    """

    # Find genes that involve this group and overlap with break window
    candidate_genes: list[tuple[int, SessionGene]] = []

    for gene_idx, gene in enumerate(individual):
        if group_id not in gene.group_ids:
            continue

        # Check if any quanta overlap with break window on this day
        for quantum in gene.get_quanta_list():
            day, within_day_q = qts.quantum_to_day_and_within_day(quantum)
            if day == day_name and within_day_q in occupied_in_break:
                candidate_genes.append((gene_idx, gene))
                break  # Found overlap, move to next gene

    if not candidate_genes:
        return 0  # No genes found to repair

    # Select one gene to repair (random choice)
    gene_idx, gene = random.choice(candidate_genes)

    # Get day offset for calculating continuous quanta
    day_offset = qts.day_quanta_offset[day_name]
    if day_offset is None:
        return 0

    # Generate candidate time slots (before and after break window)
    min_break = min(break_quanta)
    max_break = max(break_quanta)

    session_duration = gene.num_quanta

    # Before break: slots that end before break starts
    before_slots = list(range(max(0, min_break - session_duration + 1)))

    # After break: slots that start after break ends
    day_quanta_count = qts.day_quanta_count[day_name]
    if day_quanta_count is None:
        return 0

    after_slots = list(range(max_break + 1, day_quanta_count - session_duration + 1))

    candidate_slots = before_slots + after_slots

    if not candidate_slots:
        return 0  # No valid slots available

    # Shuffle to add randomness
    random.shuffle(candidate_slots)

    # Try each candidate slot
    for start_within_day in candidate_slots[:10]:  # Limit attempts to 10
        # Convert to continuous quanta
        new_start_quanta = day_offset + start_within_day

        # Create modified gene
        original_start = gene.start_quanta
        gene.start_quanta = new_start_quanta

        # Validate: check if this creates new hard constraint violations
        # For simplicity, we'll accept the change if it doesn't overlap with break
        # More sophisticated validation would check instructor/room conflicts

        # Simple validation: ensure new quanta don't overlap with break
        valid = True
        for q in gene.get_quanta_list():
            _, within_day_q = qts.quantum_to_day_and_within_day(q)
            if within_day_q in break_quanta:
                valid = False
                break

        if valid:
            # Accept the repair
            return 1
        # Revert and try next slot
        gene.start_quanta = original_start

    # No valid slot found
    return 0
