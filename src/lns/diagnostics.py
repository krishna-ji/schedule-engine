"""
LNS Diagnostics and Pre-Feasibility Checks.

This module provides tools to diagnose why CP-SAT fails (INFEASIBLE) and
to pre-check subproblem feasibility before invoking expensive solvers.
"""

import logging
from collections import defaultdict

from src.entities.course import Course
from src.entities.group import Group
from src.entities.instructor import Instructor
from src.entities.room import Room
from src.ga.sessiongene import SessionGene

logger = logging.getLogger(__name__)


class SubproblemDiagnostics:
    """Diagnostics for LNS subproblems."""

    def __init__(
        self,
        conflicted_sessions: list[SessionGene],
        partial_schedule: list[SessionGene],
        courses: dict[tuple, Course],
        instructors: dict[str, Instructor],
        groups: dict[str, Group],
        rooms: dict[str, Room],
    ):
        self.conflicted_sessions = conflicted_sessions
        self.partial_schedule = partial_schedule
        self.courses = courses
        self.instructors = instructors
        self.groups = groups
        self.rooms = rooms

    def compute_domain_sizes(self) -> dict[int, dict[str, int]]:
        """Compute domain sizes for each conflicted session.

        Returns:
            Dict mapping session index to domain sizes:
            {
                session_idx: {
                    "start_time_options": int,
                    "room_options": int,
                    "instructor_available": bool,
                    "groups_available": int
                }
            }
        """
        domain_info = {}

        for idx, session in enumerate(self.conflicted_sessions):
            course_key = (session.course_id, session.course_type)
            course = self.courses[course_key]
            instructor = self.instructors[session.instructor_id]

            # Get instructor available quanta
            instructor_quanta = set(instructor.available_quanta)

            # Get intersection of all group availabilities
            common_quanta = instructor_quanta.copy()
            for gid in session.group_ids:
                group = self.groups[gid]
                common_quanta &= set(group.available_quanta)

            # Filter out quanta already occupied in partial schedule by this instructor/groups
            occupied_instructor: set[int] = set()
            occupied_groups: dict[str, set[int]] = {
                gid: set() for gid in session.group_ids
            }

            for fixed_session in self.partial_schedule:
                if fixed_session.instructor_id == session.instructor_id:
                    # Add all quanta in the contiguous block to occupied set
                    occupied_instructor.update(
                        range(fixed_session.start_quanta, fixed_session.end_quanta)
                    )
                for gid in session.group_ids:
                    if gid in fixed_session.group_ids:
                        occupied_groups[gid].update(
                            range(fixed_session.start_quanta, fixed_session.end_quanta)
                        )

            # Available quanta (sessions can be non-contiguous across the week)
            session_duration = session.num_quanta

            # Filter out occupied quanta
            available_quanta = common_quanta - occupied_instructor
            for gid in session.group_ids:
                available_quanta -= occupied_groups[gid]

            # Count available quanta (not requiring contiguous blocks)
            num_available = len(available_quanta)

            # Count suitable rooms
            total_group_size = sum(
                self.groups[gid].student_count for gid in session.group_ids
            )
            suitable_rooms = 0
            for room in self.rooms.values():
                if (
                    room.capacity >= total_group_size
                    and room.is_suitable_for_course_type(course.required_room_features)
                ):
                    suitable_rooms += 1

            domain_info[idx] = {
                "available_quanta": (
                    num_available
                ),  # Total available quanta (can be non-contiguous)
                "required_quanta": session_duration,  # Quanta needed for session
                "room_options": suitable_rooms,
                "instructor_available": len(instructor_quanta) > 0,
                "groups_common_quanta": len(common_quanta),
            }

        return domain_info

    def pre_check_feasibility(self) -> tuple[bool, str]:
        """Run pre-feasibility check before invoking CP-SAT.

        Returns:
            (is_feasible, reason_if_not)
        """
        domain_info = self.compute_domain_sizes()

        # Check for zero-domain sessions
        infeasible_sessions = []
        reasons = []

        for idx, info in domain_info.items():
            session = self.conflicted_sessions[idx]
            # Check if enough available quanta exist (not requiring contiguity)
            if info["available_quanta"] < info["required_quanta"]:
                infeasible_sessions.append(idx)
                reasons.append(
                    f"Session {idx} (course {session.course_id}, instructor {session.instructor_id}): "
                    f"insufficient available quanta ({info['available_quanta']}/{info['required_quanta']} needed, "
                    f"groups_common={info['groups_common_quanta']})"
                )
            elif info["room_options"] == 0:
                infeasible_sessions.append(idx)
                reasons.append(
                    f"Session {idx} (course {session.course_id}): no suitable rooms"
                )
            elif not info["instructor_available"]:
                infeasible_sessions.append(idx)
                reasons.append(
                    f"Session {idx} (course {session.course_id}): instructor {session.instructor_id} has no availability"
                )

        if infeasible_sessions:
            reason_str = "; ".join(reasons[:3])  # Show first 3
            if len(reasons) > 3:
                reason_str += f" ... and {len(reasons) - 3} more"
            return False, reason_str

        return True, ""

    def log_subproblem_summary(self) -> None:
        """Log detailed summary of subproblem for diagnostics."""
        domain_info = self.compute_domain_sizes()

        logger.info(
            f"Subproblem diagnostics: {len(self.conflicted_sessions)} sessions, "
            f"{len(self.partial_schedule)} fixed"
        )

        available_quanta_list = [
            info["available_quanta"] for info in domain_info.values()
        ]
        required_quanta_list = [
            info["required_quanta"] for info in domain_info.values()
        ]
        room_options = [info["room_options"] for info in domain_info.values()]

        if available_quanta_list:
            logger.info(
                f"  Available quanta: min={min(available_quanta_list)}, "
                f"max={max(available_quanta_list)}, avg={sum(available_quanta_list) / len(available_quanta_list):.1f}"
            )
            logger.info(
                f"  Required quanta: min={min(required_quanta_list)}, "
                f"max={max(required_quanta_list)}, avg={sum(required_quanta_list) / len(required_quanta_list):.1f}"
            )
        if room_options:
            logger.info(
                f"  Room options: min={min(room_options)}, "
                f"max={max(room_options)}, avg={sum(room_options) / len(room_options):.1f}"
            )

        # Log sessions with insufficient domains
        insufficient_quanta = [
            idx
            for idx, info in domain_info.items()
            if info["available_quanta"] < info["required_quanta"]
        ]
        zero_room = [
            idx for idx, info in domain_info.items() if info["room_options"] == 0
        ]

        if insufficient_quanta:
            logger.warning(
                f"  {len(insufficient_quanta)} sessions with insufficient available quanta: {insufficient_quanta[:5]}"
            )
        if zero_room:
            logger.warning(
                f"  {len(zero_room)} sessions with NO suitable rooms: {zero_room[:5]}"
            )


def build_conflict_graph(
    individual: list[SessionGene],
    courses: dict[tuple, Course],
    instructors: dict[str, Instructor],
    groups: dict[str, Group],
    rooms: dict[str, Room],
) -> dict[int, set[int]]:
    """Build conflict graph for the entire schedule.

    Nodes: session indices
    Edges: sessions that share instructor, group, room, or have temporal overlap

    Returns:
        Adjacency dict: {session_idx: set(neighbor_indices)}
    """
    adjacency = defaultdict(set)

    # Build index by resource
    instructor_sessions = defaultdict(list)
    group_sessions = defaultdict(list)
    room_sessions = defaultdict(list)

    for idx, session in enumerate(individual):
        instructor_sessions[session.instructor_id].append(idx)
        for gid in session.group_ids:
            group_sessions[gid].append(idx)
        room_sessions[session.room_id].append(idx)

    # Add edges for shared resources
    for sessions_list in instructor_sessions.values():
        for i in sessions_list:
            for j in sessions_list:
                if i != j:
                    adjacency[i].add(j)

    for sessions_list in group_sessions.values():
        for i in sessions_list:
            for j in sessions_list:
                if i != j:
                    adjacency[i].add(j)

    for sessions_list in room_sessions.values():
        for i in sessions_list:
            for j in sessions_list:
                if i != j:
                    adjacency[i].add(j)

    # Add edges for temporal overlaps (same time quanta)
    for i in range(len(individual)):
        for j in range(i + 1, len(individual)):
            # Check if contiguous blocks overlap
            gene_i = individual[i]
            gene_j = individual[j]
            # Two ranges overlap if: start_i < end_j AND start_j < end_i
            if (
                gene_i.start_quanta < gene_j.end_quanta
                and gene_j.start_quanta < gene_i.end_quanta
            ):
                adjacency[i].add(j)
                adjacency[j].add(i)

    return dict(adjacency)


def expand_neighborhood_bfs(
    initial_indices: list[int],
    conflict_graph: dict[int, set[int]],
    max_size: int,
    hops: int = 1,
) -> list[int]:
    """Expand neighborhood using BFS on conflict graph.

    Args:
        initial_indices: Starting session indices
        conflict_graph: Adjacency dict from build_conflict_graph
        max_size: Maximum neighborhood size
        hops: Number of BFS hops to expand

    Returns:
        Expanded list of session indices
    """
    if hops == 0 or len(initial_indices) >= max_size:
        return initial_indices

    visited = set(initial_indices)
    frontier = set(initial_indices)

    for _ in range(hops):
        new_frontier = set()
        for node in frontier:
            neighbors = conflict_graph.get(node, set())
            for neighbor in neighbors:
                if neighbor not in visited and len(visited) < max_size:
                    new_frontier.add(neighbor)
                    visited.add(neighbor)
        frontier = new_frontier
        if not frontier:
            break

    return sorted(visited)
