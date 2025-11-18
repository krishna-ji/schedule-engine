"""
Encoder module for loading and encoding scheduling data.

Provides functions to load courses, groups, instructors, and rooms from JSON files
and convert them into internal representations with quantum time encoding.
"""

from src.encoder.input_encoder import (
    load_courses,
    load_groups,
    load_instructors,
    load_rooms,
    link_courses_and_groups,
    link_courses_and_instructors,
    encode_availability,
)
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.core.types import SchedulingContext


def load_scheduling_data(data_dir: str) -> SchedulingContext:
    """
    Load all scheduling data from JSON files and create context.
    
    Args:
        data_dir: Directory containing JSON data files
        
    Returns:
        SchedulingContext with all loaded data
    """
    import os
    
    qts = QuantumTimeSystem()
    
    # Load all entities
    courses = load_courses(os.path.join(data_dir, "Course.json"))
    groups = load_groups(os.path.join(data_dir, "Groups.json"), qts)
    instructors = load_instructors(os.path.join(data_dir, "Instructors.json"), qts)
    rooms = load_rooms(os.path.join(data_dir, "Rooms.json"), qts)
    
    # Link entities
    link_courses_and_groups(courses, groups)
    link_courses_and_instructors(courses, instructors)
    
    # Create context
    context = SchedulingContext(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        available_quanta=qts.get_all_operating_quanta(),
        qts=qts
    )
    
    return context


__all__ = [
    'load_courses',
    'load_groups',
    'load_instructors',
    'load_rooms',
    'link_courses_and_groups',
    'link_courses_and_instructors',
    'encode_availability',
    'QuantumTimeSystem',
    'SchedulingContext',
    'load_scheduling_data',
]
