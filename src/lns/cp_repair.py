"""
CP-SAT based repair for conflicted sessions.

This module uses Google OR-Tools CP-SAT solver to repair hard constraint
violations in a subset of sessions while respecting the fixed partial schedule.
"""

from typing import List, Dict, Optional
from collections import defaultdict
import logging

from ortools.sat.python import cp_model

from src.ga.sessiongene import SessionGene
from src.entities.course import Course
from src.entities.instructor import Instructor
from src.entities.group import Group
from src.entities.room import Room
from src.encoder.quantum_time_system import QuantumTimeSystem

# Logger setup
logger = logging.getLogger(__name__)


class CPRepairSolver:
    """
    CP-SAT solver for repairing conflicted sessions.

    This solver creates a CP model for a subproblem consisting of conflicted
    sessions, enforces hard constraints internally and against the fixed
    partial schedule, and optimizes soft constraints.
    """

    def __init__(
        self,
        time_limit_seconds: float = 10.0,
        quantum_time_system: QuantumTimeSystem = None,
    ):
        """
        Initialize CP repair solver.

        Args:
            time_limit_seconds: Maximum time for CP-SAT solver
            quantum_time_system: Time system for quantum conversion
        """
        self.time_limit_seconds = time_limit_seconds
        self.qts = quantum_time_system or QuantumTimeSystem()

    def repair_sessions(
        self,
        conflicted_sessions: List[SessionGene],
        partial_schedule: List[SessionGene],
        courses: Dict[tuple, Course],
        instructors: Dict[str, Instructor],
        groups: Dict[str, Group],
        rooms: Dict[str, Room],
    ) -> Optional[List[SessionGene]]:
        """
        Repair conflicted sessions using CP-SAT solver.

        Args:
            conflicted_sessions: Sessions to repair
            partial_schedule: Fixed sessions (already scheduled)
            courses: Course dictionary
            instructors: Instructor dictionary
            groups: Group dictionary
            rooms: Room dictionary

        Returns:
            List of repaired SessionGenes if successful, None if failed
        """
        if not conflicted_sessions:
            return []

        logger.info(
            f"CP-SAT repair: attempting to repair {len(conflicted_sessions)} sessions"
        )

        # Create CP model
        model = cp_model.CpModel()

        # Create variables and collect domains
        session_vars = []
        for idx, session in enumerate(conflicted_sessions):
            course_key = (session.course_id, session.course_type)
            course = courses[course_key]
            instructor = instructors[session.instructor_id]

            # Get domains for this session
            start_domain = self._get_start_time_domain(
                course, instructor, groups, session.group_ids
            )
            room_domain = self._get_room_domain(
                course, rooms, session.group_ids, groups
            )

            # Create variables
            start_var = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(start_domain),
                f"start_{idx}",
            )
            room_var = model.NewIntVarFromDomain(
                cp_model.Domain.FromValues(room_domain),
                f"room_{idx}",
            )

            # Calculate duration from session quanta
            duration = len(session.quanta)

            session_vars.append(
                {
                    "index": idx,
                    "session": session,
                    "start": start_var,
                    "room": room_var,
                    "duration": duration,
                    "course_key": course_key,
                }
            )

        # Add internal constraints (among conflicted sessions)
        self._add_internal_constraints(
            model, session_vars, conflicted_sessions, instructors, groups
        )

        # Add constraints against fixed partial schedule
        self._add_partial_schedule_constraints(
            model, session_vars, partial_schedule, instructors, groups, rooms
        )

        # Add soft constraint optimization
        soft_penalty = self._add_soft_constraints(
            model, session_vars, conflicted_sessions, groups
        )

        # Minimize soft constraint penalty
        model.Minimize(soft_penalty)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit_seconds
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Extract solution
            repaired_sessions = self._extract_solution(
                solver, session_vars, conflicted_sessions, rooms
            )
            logger.info(
                f"CP-SAT repair: SUCCESS (status={solver.StatusName(status)}, "
                f"time={solver.WallTime():.2f}s)"
            )
            return repaired_sessions
        else:
            logger.warning(
                f"CP-SAT repair: FAILED (status={solver.StatusName(status)}, "
                f"time={solver.WallTime():.2f}s)"
            )
            return None

    def _get_start_time_domain(
        self,
        course: Course,
        instructor: Instructor,
        groups: Dict[str, Group],
        group_ids: List[str],
    ) -> List[int]:
        """Get valid start time quanta for a session."""
        # Start with all operating quanta
        all_quanta = self.qts.get_all_operating_quanta()

        # Intersect with instructor availability
        valid_quanta = set(instructor.available_quanta) & all_quanta

        # Intersect with all group availabilities
        for gid in group_ids:
            group = groups[gid]
            valid_quanta &= set(group.available_quanta)

        # Filter to ensure session fits (start + duration must be valid)
        # Use quanta_per_week / sessions as approximation for duration
        duration = max(1, course.quanta_per_week // 2)  # Approximate session duration
        filtered = []
        for q in sorted(valid_quanta):
            # Check if all quanta from q to q+duration-1 are available
            session_quanta = list(range(q, q + duration))
            if all(sq in valid_quanta for sq in session_quanta):
                filtered.append(q)

        return filtered if filtered else [0]  # Fallback to avoid empty domain

    def _get_room_domain(
        self,
        course: Course,
        rooms: Dict[str, Room],
        group_ids: List[str],
        groups: Dict[str, Group],
    ) -> List[int]:
        """Get valid room indices for a session."""
        # Create room ID to index mapping
        room_list = list(rooms.keys())

        # Calculate total group size
        total_size = sum(groups[gid].student_count for gid in group_ids)

        # Filter rooms by capacity and features
        valid_indices = []
        for idx, room_id in enumerate(room_list):
            room = rooms[room_id]
            if room.capacity >= total_size and room.is_suitable_for_course_type(
                course.required_room_features
            ):
                valid_indices.append(idx)

        return valid_indices if valid_indices else [0]  # Fallback

    def _add_internal_constraints(
        self,
        model: cp_model.CpModel,
        session_vars: List[Dict],
        sessions: List[SessionGene],
        instructors: Dict[str, Instructor],
        groups: Dict[str, Group],
    ):
        """Add constraints among conflicted sessions (no overlaps)."""
        n = len(session_vars)

        # Instructor conflicts: same instructor cannot teach two sessions at the same time
        instructor_sessions = defaultdict(list)
        for var_dict in session_vars:
            session = var_dict["session"]
            instructor_sessions[session.instructor_id].append(var_dict)

        for instructor_id, vars_list in instructor_sessions.items():
            if len(vars_list) > 1:
                # Create NoOverlap constraint for this instructor
                intervals = []
                for var_dict in vars_list:
                    interval = model.NewIntervalVar(
                        var_dict["start"],
                        var_dict["duration"],
                        var_dict["start"] + var_dict["duration"],
                        f"instructor_{instructor_id}_interval_{var_dict['index']}",
                    )
                    intervals.append(interval)
                model.AddNoOverlap(intervals)

        # Student group conflicts: same group cannot attend two sessions at the same time
        group_sessions = defaultdict(list)
        for var_dict in session_vars:
            session = var_dict["session"]
            for gid in session.group_ids:
                group_sessions[gid].append(var_dict)

        for group_id, vars_list in group_sessions.items():
            if len(vars_list) > 1:
                intervals = []
                for var_dict in vars_list:
                    interval = model.NewIntervalVar(
                        var_dict["start"],
                        var_dict["duration"],
                        var_dict["start"] + var_dict["duration"],
                        f"group_{group_id}_interval_{var_dict['index']}",
                    )
                    intervals.append(interval)
                model.AddNoOverlap(intervals)

        # Room conflicts: same room cannot host two sessions at the same time
        # This is trickier because room is a variable, not fixed
        # We need to add conditional NoOverlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                var_i = session_vars[i]
                var_j = session_vars[j]

                # Create boolean: same_room = (room_i == room_j)
                same_room = model.NewBoolVar(f"same_room_{i}_{j}")
                model.Add(var_i["room"] == var_j["room"]).OnlyEnforceIf(same_room)
                model.Add(var_i["room"] != var_j["room"]).OnlyEnforceIf(same_room.Not())

                # If same room, then no temporal overlap
                # End_i <= Start_j OR End_j <= Start_i
                end_i = var_i["start"] + var_i["duration"]
                end_j = var_j["start"] + var_j["duration"]

                # Create boolean for the two non-overlap cases
                i_before_j = model.NewBoolVar(f"i_before_j_{i}_{j}")
                j_before_i = model.NewBoolVar(f"j_before_i_{i}_{j}")

                model.Add(end_i <= var_j["start"]).OnlyEnforceIf(
                    [same_room, i_before_j]
                )
                model.Add(end_j <= var_i["start"]).OnlyEnforceIf(
                    [same_room, j_before_i]
                )

                # If same room, at least one must be true
                model.AddBoolOr([i_before_j, j_before_i, same_room.Not()])

    def _add_partial_schedule_constraints(
        self,
        model: cp_model.CpModel,
        session_vars: List[Dict],
        partial_schedule: List[SessionGene],
        instructors: Dict[str, Instructor],
        groups: Dict[str, Group],
        rooms: Dict[str, Room],
    ):
        """Add constraints to avoid conflicts with the fixed partial schedule."""
        # For each session in partial schedule, forbid overlaps with repaired sessions

        for fixed_session in partial_schedule:
            fixed_start = min(fixed_session.quanta)
            fixed_duration = len(fixed_session.quanta)
            fixed_end = fixed_start + fixed_duration
            fixed_room_id = fixed_session.room_id

            # Check instructor conflicts
            for var_dict in session_vars:
                if var_dict["session"].instructor_id == fixed_session.instructor_id:
                    # This repaired session has same instructor
                    # Must not overlap with fixed session
                    var_start = var_dict["start"]
                    var_end = var_start + var_dict["duration"]

                    # No overlap: var_end <= fixed_start OR fixed_end <= var_start
                    no_overlap_1 = model.NewBoolVar(
                        f"no_overlap_inst_1_{var_dict['index']}"
                    )
                    no_overlap_2 = model.NewBoolVar(
                        f"no_overlap_inst_2_{var_dict['index']}"
                    )

                    model.Add(var_end <= fixed_start).OnlyEnforceIf(no_overlap_1)
                    model.Add(fixed_end <= var_start).OnlyEnforceIf(no_overlap_2)
                    model.AddBoolOr([no_overlap_1, no_overlap_2])

            # Check group conflicts
            for gid in fixed_session.group_ids:
                for var_dict in session_vars:
                    if gid in var_dict["session"].group_ids:
                        var_start = var_dict["start"]
                        var_end = var_start + var_dict["duration"]

                        no_overlap_1 = model.NewBoolVar(
                            f"no_overlap_grp_1_{var_dict['index']}_{gid}"
                        )
                        no_overlap_2 = model.NewBoolVar(
                            f"no_overlap_grp_2_{var_dict['index']}_{gid}"
                        )

                        model.Add(var_end <= fixed_start).OnlyEnforceIf(no_overlap_1)
                        model.Add(fixed_end <= var_start).OnlyEnforceIf(no_overlap_2)
                        model.AddBoolOr([no_overlap_1, no_overlap_2])

            # Check room conflicts
            room_list = list(rooms.keys())
            if fixed_room_id in room_list:
                fixed_room_idx = room_list.index(fixed_room_id)

                for var_dict in session_vars:
                    # If this repaired session uses the same room, no temporal overlap
                    same_room = model.NewBoolVar(f"same_room_fixed_{var_dict['index']}")
                    model.Add(var_dict["room"] == fixed_room_idx).OnlyEnforceIf(
                        same_room
                    )
                    model.Add(var_dict["room"] != fixed_room_idx).OnlyEnforceIf(
                        same_room.Not()
                    )

                    var_start = var_dict["start"]
                    var_end = var_start + var_dict["duration"]

                    no_overlap_1 = model.NewBoolVar(
                        f"no_overlap_room_1_{var_dict['index']}"
                    )
                    no_overlap_2 = model.NewBoolVar(
                        f"no_overlap_room_2_{var_dict['index']}"
                    )

                    model.Add(var_end <= fixed_start).OnlyEnforceIf(
                        [same_room, no_overlap_1]
                    )
                    model.Add(fixed_end <= var_start).OnlyEnforceIf(
                        [same_room, no_overlap_2]
                    )
                    model.AddBoolOr([no_overlap_1, no_overlap_2, same_room.Not()])

    def _add_soft_constraints(
        self,
        model: cp_model.CpModel,
        session_vars: List[Dict],
        sessions: List[SessionGene],
        groups: Dict[str, Group],
    ) -> cp_model.LinearExpr:
        """Add soft constraints and return penalty expression."""
        # For now, implement a simple soft constraint: minimize schedule span
        # This encourages compact schedules

        penalties = []

        # Compactness: penalize sessions scheduled late in the week
        for var_dict in session_vars:
            # Penalty proportional to start time
            penalties.append(var_dict["start"])

        # Sum all penalties
        if penalties:
            return sum(penalties)
        else:
            return model.NewConstant(0)

    def _extract_solution(
        self,
        solver: cp_model.CpSolver,
        session_vars: List[Dict],
        original_sessions: List[SessionGene],
        rooms: Dict[str, Room],
    ) -> List[SessionGene]:
        """Extract repaired sessions from CP-SAT solution."""
        room_list = list(rooms.keys())
        repaired = []

        for var_dict in session_vars:
            idx = var_dict["index"]
            original = original_sessions[idx]

            # Get solution values
            start_quantum = solver.Value(var_dict["start"])
            room_idx = solver.Value(var_dict["room"])
            duration = var_dict["duration"]

            # Create quanta list
            quanta = list(range(start_quantum, start_quantum + duration))

            # Get room ID
            room_id = (
                room_list[room_idx] if room_idx < len(room_list) else original.room_id
            )

            # Create repaired SessionGene
            repaired_gene = SessionGene(
                course_id=original.course_id,
                course_type=original.course_type,
                instructor_id=original.instructor_id,
                group_ids=original.group_ids,
                room_id=room_id,
                quanta=quanta,
            )
            repaired.append(repaired_gene)

        return repaired


def repair_with_cp_sat(
    conflicted_sessions: List[SessionGene],
    partial_schedule: List[SessionGene],
    courses: Dict[tuple, Course],
    instructors: Dict[str, Instructor],
    groups: Dict[str, Group],
    rooms: Dict[str, Room],
    time_limit_seconds: float = 10.0,
) -> Optional[List[SessionGene]]:
    """
    Convenience function to repair sessions using CP-SAT.

    Args:
        conflicted_sessions: Sessions with hard constraint violations
        partial_schedule: Already scheduled (fixed) sessions
        courses: Course dictionary
        instructors: Instructor dictionary
        groups: Group dictionary
        rooms: Room dictionary
        time_limit_seconds: Time limit for CP-SAT solver

    Returns:
        List of repaired sessions if successful, None otherwise
    """
    solver = CPRepairSolver(time_limit_seconds=time_limit_seconds)
    return solver.repair_sessions(
        conflicted_sessions,
        partial_schedule,
        courses,
        instructors,
        groups,
        rooms,
    )
