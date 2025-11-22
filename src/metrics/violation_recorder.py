"""
Violation Recorder for Heatmap Integration

Records constraint violations during fitness evaluation.
Integrates with ViolationHeatmap to track hot genes.
"""

from typing import List
from src.ga.sessiongene import SessionGene
from src.core.types import SchedulingContext
from src.metrics.violation_heatmap import ViolationHeatmap


def record_violations_to_heatmap(
    individual: List[SessionGene], context: SchedulingContext, heatmap: ViolationHeatmap
):
    """
    Evaluate individual and record all violations to heatmap.

    This function checks each gene for constraint violations WITHOUT
    computing penalties (lightweight check for tracking only).

    Args:
        individual: GA individual (chromosome)
        context: Scheduling context
        heatmap: ViolationHeatmap to record violations
    """
    if heatmap is None:
        return

    # Check each gene for violations
    for gene in individual:
        # 1. Instructor availability
        instructor = context.instructors.get(gene.instructor_id)
        if instructor:
            # Check if any quantum in the contiguous block is unavailable
            if any(
                q not in instructor.available_quanta
                for q in range(gene.start_quanta, gene.end_quanta)
            ):
                heatmap.record_violation(gene, "availability")

        # 2. Instructor qualification
        course_key = (gene.course_id, gene.course_type)
        course = context.courses.get(course_key)
        if course and instructor:
            if instructor.instructor_id not in course.qualified_instructor_ids:
                heatmap.record_violation(gene, "qualification")

        # 3. Room type mismatch (simplified check)
        room = context.rooms.get(gene.room_id)
        if room and course:
            required = getattr(course, "required_room_features", "classroom").lower()
            room_feat = getattr(room, "room_features", "classroom").lower()
            if required not in room_feat and room_feat not in required:
                heatmap.record_violation(gene, "room_type")

    # 4. Check overlaps (requires comparing genes)
    from collections import defaultdict

    # Group overlaps
    group_schedule = defaultdict(lambda: defaultdict(list))
    for gene in individual:
        for group_id in gene.group_ids:
            for q in range(gene.start_quanta, gene.end_quanta):
                group_schedule[group_id][q].append(gene)

    for group_id, quanta_map in group_schedule.items():
        for quantum, genes in quanta_map.items():
            if len(genes) > 1:
                # Record all genes involved in overlap
                for gene in genes:
                    heatmap.record_violation(gene, "overlap")

    # Instructor conflicts
    instructor_schedule = defaultdict(lambda: defaultdict(list))
    for gene in individual:
        for q in range(gene.start_quanta, gene.end_quanta):
            instructor_schedule[gene.instructor_id][q].append(gene)

    for instructor_id, quanta_map in instructor_schedule.items():
        for quantum, genes in quanta_map.items():
            if len(genes) > 1:
                for gene in genes:
                    heatmap.record_violation(gene, "instructor_conflict")

    # Room conflicts
    room_schedule = defaultdict(lambda: defaultdict(list))
    for gene in individual:
        for q in range(gene.start_quanta, gene.end_quanta):
            room_schedule[gene.room_id][q].append(gene)

    for room_id, quanta_map in room_schedule.items():
        for quantum, genes in quanta_map.items():
            if len(genes) > 1:
                for gene in genes:
                    heatmap.record_violation(gene, "room_conflict")
