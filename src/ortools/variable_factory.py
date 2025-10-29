"""
CP-SAT Variable Factory

Creates decision variables for the course scheduling problem.

Decision Variables:
    For each (course, group, session_index):
        - start_quantum: When does the session start? (integer quantum index)
        - instructor_id: Which instructor teaches? (integer ID index)
        - room_id: Which room is used? (integer ID index)

Variable Indexing:
    - Convert string IDs to integer indices for CP-SAT compatibility
    - Maintain bidirectional mappings for decoding
"""

from typing import Dict, List, Tuple
from ortools.sat.python import cp_model

from src.core.types import SchedulingContext
from src.entities.course import Course


class VariableFactory:
    """
    Factory for creating CP-SAT decision variables for scheduling problem.

    Creates variables for each session that needs to be scheduled,
    along with domain constraints (valid ranges for each variable).
    """

    def __init__(self, context: SchedulingContext):
        """
        Initialize variable factory with scheduling context.

        Args:
            context: SchedulingContext containing all entities
        """
        self.context = context

        # Create index mappings (string ID <-> integer index)
        self.instructor_to_idx = {
            iid: idx for idx, iid in enumerate(context.instructors.keys())
        }
        self.idx_to_instructor = {
            idx: iid for iid, idx in self.instructor_to_idx.items()
        }

        self.room_to_idx = {rid: idx for idx, rid in enumerate(context.rooms.keys())}
        self.idx_to_room = {idx: rid for rid, idx in self.room_to_idx.items()}

        # Store variables by session key
        self.session_vars: Dict[Tuple[str, str, str, int], Dict] = {}

    def create_session_variables(
        self, model: cp_model.CpModel
    ) -> Dict[Tuple[str, str, str, int], Dict]:
        """
        Create decision variables for all sessions.

        For each (course_code, course_type, group_id, session_index):
            - start_quantum: start time
            - instructor: assigned instructor
            - room: assigned room

        Args:
            model: CP-SAT model to add variables to

        Returns:
            Dictionary mapping session keys to variable dictionaries
        """
        session_vars = {}

        # Iterate through all courses
        for course_key, course in self.context.courses.items():
            course_code, course_type = course_key

            # For each group enrolled in this course
            for group_id in course.enrolled_group_ids:
                # Calculate number of sessions needed
                quanta_per_week = course.quanta_per_week

                # Create variables for each session
                # Strategy: Flexible scheduling - each session can be any duration
                # We'll create variables for individual quanta and group them via constraints
                for session_idx in range(quanta_per_week):
                    session_key = (course_code, course_type, group_id, session_idx)

                    # Variable: Start quantum (when does this quantum slot occur?)
                    # Domain: All available quanta in the week
                    start_var = model.NewIntVar(
                        min(self.context.available_quanta),
                        max(self.context.available_quanta),
                        f"start_{course_code}_{course_type}_{group_id}_s{session_idx}",
                    )

                    # Variable: Instructor (who teaches this quantum?)
                    # Domain: All qualified instructors for this course
                    qualified_instructors = self._get_qualified_instructor_indices(
                        course
                    )

                    if not qualified_instructors:
                        # No qualified instructors - problem is infeasible
                        raise ValueError(
                            f"No qualified instructors for course {course_code} ({course_type})"
                        )

                    instructor_var = model.NewIntVarFromDomain(
                        cp_model.Domain.FromValues(qualified_instructors),
                        f"instructor_{course_code}_{course_type}_{group_id}_s{session_idx}",
                    )

                    # Variable: Room (which room is used?)
                    # Domain: All rooms with appropriate type and capacity
                    suitable_rooms = self._get_suitable_room_indices(course, group_id)

                    if not suitable_rooms:
                        # No suitable rooms - problem is infeasible
                        raise ValueError(
                            f"No suitable rooms for course {course_code} ({course_type}) "
                            f"with group {group_id}"
                        )

                    room_var = model.NewIntVarFromDomain(
                        cp_model.Domain.FromValues(suitable_rooms),
                        f"room_{course_code}_{course_type}_{group_id}_s{session_idx}",
                    )

                    # Store variables
                    session_vars[session_key] = {
                        "start_quantum": start_var,
                        "instructor": instructor_var,
                        "room": room_var,
                        "course_code": course_code,
                        "course_type": course_type,
                        "group_id": group_id,
                        "session_index": session_idx,
                    }

        self.session_vars = session_vars
        return session_vars

    def _get_qualified_instructor_indices(self, course: Course) -> List[int]:
        """
        Get list of instructor indices qualified for this course.

        Args:
            course: Course entity

        Returns:
            List of instructor indices
        """
        qualified_ids = course.qualified_instructor_ids
        indices = []

        for iid in qualified_ids:
            if iid in self.instructor_to_idx:
                indices.append(self.instructor_to_idx[iid])

        return indices

    def _get_suitable_room_indices(self, course: Course, group_id: str) -> List[int]:
        """
        Get list of room indices suitable for this course and group.

        Filters by:
            - Room type matches course requirements
            - Room capacity >= group size

        Args:
            course: Course entity
            group_id: Group ID

        Returns:
            List of room indices
        """
        group = self.context.groups[group_id]
        required_features = course.required_room_features
        group_size = group.student_count

        suitable_indices = []

        for room_id, room in self.context.rooms.items():
            # Check capacity
            if room.capacity < group_size:
                continue

            # Check room type compatibility
            if self._room_type_matches(required_features, room.room_features):
                suitable_indices.append(self.room_to_idx[room_id])

        return suitable_indices

    def _room_type_matches(self, required: str, room_type: str) -> bool:
        """
        Check if room type satisfies requirement with flexible compatibility.

        Args:
            required: Required room type (e.g., "lecture", "practical")
            room_type: Actual room type

        Returns:
            True if compatible
        """
        required_lower = required.lower().strip()
        room_lower = room_type.lower().strip()

        # Exact match
        if required_lower == room_lower:
            return True

        # Lecture compatibility
        if required_lower in ["lecture", "classroom", "theory"]:
            if room_lower in [
                "lecture",
                "classroom",
                "auditorium",
                "seminar",
                "tutorial",
            ]:
                return True

        # Practical compatibility
        if required_lower in ["practical", "lab", "laboratory"]:
            if room_lower in [
                "practical",
                "lab",
                "laboratory",
                "computer_lab",
                "science_lab",
            ]:
                return True

        return False
