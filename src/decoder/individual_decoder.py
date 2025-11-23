"""
Module: individual_decoder

This module provides functionality to decode a genetic algorithm (GA) individual—
represented as a list of `SessionGene` objects—into a list of rich, semantically
meaningful `CourseSession` objects.

The decoded structure is used for constraint evaluation, visualization,
and schedule analysis in the University Course Timetabling Problem (UCTP).
"""

from typing import List, Dict
from src.ga.sessiongene import SessionGene
from src.entities.decoded_session import CourseSession
from src.entities.course import Course
from src.entities.instructor import Instructor
from src.entities.group import Group
from src.entities.room import Room


def decode_individual(
    individual: List[SessionGene],
    courses: Dict[tuple, Course],  # Keys are (course_code, course_type) tuples
    instructors: Dict[str, Instructor],
    groups: Dict[str, Group],
    rooms: Dict[str, Room],
) -> List[CourseSession]:
    """Decodes a GA individual (chromosome) into a list of CourseSession objects.

    This function translates each `SessionGene` into a `CourseSession`, enriching
    the basic encoded representation with full instructor and group references,
    along with room and course metadata. The output is suitable for use in
    constraint checking and visualization.

    Architecture Note (Nov 2025): SessionGene uses contiguous representation
    (start_quanta + num_quanta) instead of array-based quanta list. This enforces
    structural continuity and reduces memory footprint by 60%.

    Args:
        individual (List[SessionGene]): The chromosome to decode; each gene represents
            a single course session assignment with time (start_quanta + num_quanta),
            room, and entity assignments.
        courses (Dict[tuple, Course]): Mapping from (course_code, course_type) to Course
            objects, providing metadata like required room features.
        instructors (Dict[str, Instructor]): Mapping from instructor ID to Instructor objects.
        groups (Dict[str, Group]): Mapping from group ID to Group objects, including
            availability and enrollment data.
        rooms (Dict[str, Room]): Mapping from room ID to Room objects, including
            capacity and features data.

    Returns:
        List[CourseSession]: A list of fully populated CourseSession objects derived
        from the input chromosome.
    """
    decoded_sessions = []

    # Get actual valid quantum range from QuantumTimeSystem
    from src.encoder.quantum_time_system import QuantumTimeSystem

    MAX_VALID_QUANTUM = QuantumTimeSystem().total_quanta

    for gene in individual:
        # Validate and clip quanta before decoding to prevent ValueError in constraints
        # This handles cases where crossover/mutation bypassed SessionGene validation
        valid_quanta = [
            q
            for q in range(gene.start_quanta, gene.end_quanta)
            if 0 <= q < MAX_VALID_QUANTUM
        ]
        if not valid_quanta:
            # All quanta invalid - skip this gene entirely
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                f"Skipping gene {gene.course_id} - all quanta invalid: start={gene.start_quanta}, num={gene.num_quanta}"
            )
            continue

        # Update gene with valid quanta (modify in-place to fix the chromosome)
        if len(valid_quanta) != gene.num_quanta:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                f"Clipped gene {gene.course_id} quanta from {gene.num_quanta} to {len(valid_quanta)}"
            )
            # Update to valid range
            gene.start_quanta = valid_quanta[0] if valid_quanta else 0
            gene.num_quanta = len(valid_quanta) if valid_quanta else 1
        # Look up course using tuple key (course_id, course_type)
        course_key = (gene.course_id, gene.course_type)
        course = courses[course_key]

        instructor = instructors[gene.instructor_id]
        # Get primary group (first group in the list)
        group = groups[gene.group_ids[0]] if gene.group_ids else None
        room = rooms[gene.room_id]

        session = CourseSession(
            course_id=gene.course_id,
            instructor_id=gene.instructor_id,
            group_ids=gene.group_ids,
            room_id=gene.room_id,
            session_quanta=gene.get_quanta_list(),
            required_room_features=course.required_room_features,
            course_type=gene.course_type,  # Use gene's course_type
            instructor=instructor,
            group=group,  # Primary group (first in list)
            room=room,
        )

        decoded_sessions.append(session)

    return decoded_sessions
