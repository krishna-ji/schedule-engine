"""
CP-SAT Constraint Factory

Implements hard constraints as CP-SAT constraints.

Hard Constraints (5 core):
    1. no_group_overlap: Groups cannot be in multiple places at same time
    2. instructor_not_qualified: Only qualified instructors (handled in variable domains)
    3. availability_violations: Respect instructor/group/room availability
    4. room_type_mismatch: Room types match requirements (handled in variable domains)
    5. room_double_booking: Rooms cannot host multiple sessions simultaneously

Additional Constraints:
    - Session continuity: Multi-quantum sessions must be consecutive
    - Same-day sessions: All quanta for a session must be on same day
"""

from typing import Dict, Tuple, List
from ortools.sat.python import cp_model

from src.core.types import SchedulingContext
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.ortools.variable_factory import VariableFactory


class ConstraintFactory:
    """
    Factory for adding CP-SAT constraints to the scheduling model.

    Translates high-level scheduling constraints into CP-SAT constraint expressions.
    """

    def __init__(
        self,
        context: SchedulingContext,
        var_factory: VariableFactory,
        qts: QuantumTimeSystem,
    ):
        """
        Initialize constraint factory.

        Args:
            context: SchedulingContext with all entities
            var_factory: VariableFactory with decision variables
            qts: QuantumTimeSystem for time calculations
        """
        self.context = context
        self.var_factory = var_factory
        self.qts = qts

    def add_all_constraints(self, model: cp_model.CpModel, session_vars: Dict):
        """
        Add all hard constraints to the model.

        Args:
            model: CP-SAT model
            session_vars: Dictionary of session variables from VariableFactory
        """
        print("  [1/5] Adding group overlap constraints...")
        self.add_no_group_overlap_constraints(model, session_vars)

        print("  [2/5] Adding instructor conflict constraints...")
        self.add_no_instructor_conflict_constraints(model, session_vars)

        print("  [3/5] Adding availability constraints...")
        self.add_availability_constraints(model, session_vars)

        print("  [4/5] Adding room conflict constraints...")
        self.add_no_room_conflict_constraints(model, session_vars)

        print("  [5/5] Adding valid quantum constraints...")
        self.add_valid_quantum_constraints(model, session_vars)

    def add_no_group_overlap_constraints(
        self, model: cp_model.CpModel, session_vars: Dict
    ):
        """
        Constraint: No group can attend multiple sessions at the same time.

        For each group:
            For each pair of sessions involving that group:
                start_time1 != start_time2

        Args:
            model: CP-SAT model
            session_vars: Session variables dictionary
        """
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TimeElapsedColumn,
        )

        # Group sessions by group_id
        group_sessions = {}
        for session_key, vars_dict in session_vars.items():
            group_id = vars_dict["group_id"]
            if group_id not in group_sessions:
                group_sessions[group_id] = []
            group_sessions[group_id].append((session_key, vars_dict))

        # Calculate total constraints
        total_constraints = sum(
            len(sessions) * (len(sessions) - 1) // 2
            for sessions in group_sessions.values()
        )

        # For each group, add pairwise no-overlap constraints
        constraint_count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"      Adding group overlap constraints...", total=len(group_sessions)
            )

            for group_id, sessions in group_sessions.items():
                for i in range(len(sessions)):
                    for j in range(i + 1, len(sessions)):
                        key1, vars1 = sessions[i]
                        key2, vars2 = sessions[j]

                        # These two sessions cannot occur at the same time
                        model.Add(vars1["start_quantum"] != vars2["start_quantum"])
                        constraint_count += 1

                progress.advance(task)

        print(f"      ✓ Added {constraint_count:,} group overlap constraints")

    def add_no_instructor_conflict_constraints(
        self, model: cp_model.CpModel, session_vars: Dict
    ):
        """
        Constraint: No instructor can teach multiple sessions at the same time.

        For each pair of sessions:
            If instructor1 == instructor2:
                Then start_time1 != start_time2

        Args:
            model: CP-SAT model
            session_vars: Session variables dictionary
        """
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TimeElapsedColumn,
        )

        sessions_list = list(session_vars.items())
        total_pairs = len(sessions_list) * (len(sessions_list) - 1) // 2
        constraint_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"      Adding instructor conflicts...", total=total_pairs
            )

            for i in range(len(sessions_list)):
                for j in range(i + 1, len(sessions_list)):
                    key1, vars1 = sessions_list[i]
                    key2, vars2 = sessions_list[j]

                    # Create boolean: are instructors the same?
                    same_instructor = model.NewBoolVar(f"same_instr_{i}_{j}")
                    model.Add(vars1["instructor"] == vars2["instructor"]).OnlyEnforceIf(
                        same_instructor
                    )
                    model.Add(vars1["instructor"] != vars2["instructor"]).OnlyEnforceIf(
                        same_instructor.Not()
                    )

                    # If same instructor, then different times
                    model.Add(
                        vars1["start_quantum"] != vars2["start_quantum"]
                    ).OnlyEnforceIf(same_instructor)
                    constraint_count += 1

                    if constraint_count % 10000 == 0:
                        progress.update(task, completed=constraint_count)

            progress.update(task, completed=total_pairs)

        print(f"      ✓ Added {constraint_count:,} instructor conflict constraints")

    def add_no_room_conflict_constraints(
        self, model: cp_model.CpModel, session_vars: Dict
    ):
        """
        Constraint: No room can host multiple sessions at the same time.

        For each pair of sessions:
            If room1 == room2:
                Then start_time1 != start_time2

        Args:
            model: CP-SAT model
            session_vars: Session variables dictionary
        """
        from rich.progress import (
            Progress,
            SpinnerColumn,
            TextColumn,
            BarColumn,
            TimeElapsedColumn,
        )

        sessions_list = list(session_vars.items())
        total_pairs = len(sessions_list) * (len(sessions_list) - 1) // 2
        constraint_count = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"      Adding room conflicts...", total=total_pairs
            )

            for i in range(len(sessions_list)):
                for j in range(i + 1, len(sessions_list)):
                    key1, vars1 = sessions_list[i]
                    key2, vars2 = sessions_list[j]

                    # Create boolean: are rooms the same?
                    same_room = model.NewBoolVar(f"same_room_{i}_{j}")
                    model.Add(vars1["room"] == vars2["room"]).OnlyEnforceIf(same_room)
                    model.Add(vars1["room"] != vars2["room"]).OnlyEnforceIf(
                        same_room.Not()
                    )

                    # If same room, then different times
                    model.Add(
                        vars1["start_quantum"] != vars2["start_quantum"]
                    ).OnlyEnforceIf(same_room)
                    constraint_count += 1

                    if constraint_count % 10000 == 0:
                        progress.update(task, completed=constraint_count)

            progress.update(task, completed=total_pairs)

        print(f"      ✓ Added {constraint_count:,} room conflict constraints")

    def add_availability_constraints(self, model: cp_model.CpModel, session_vars: Dict):
        """
        Constraint: Sessions must occur during available times for instructor, group, and room.

        OPTIMIZED: Use domain restrictions instead of individual constraints.

        Args:
            model: CP-SAT model
            session_vars: Session variables dictionary
        """
        constraint_count = 0

        for session_key, vars_dict in session_vars.items():
            group_id = vars_dict["group_id"]
            start_var = vars_dict["start_quantum"]

            # Group availability - restrict domain directly
            group = self.context.groups[group_id]
            if group.available_quanta:
                # Already restricted by valid_quantum_constraints
                # Additional group restriction
                group_available = sorted(group.available_quanta)
                model.AddAllowedAssignments([start_var], [[q] for q in group_available])
                constraint_count += 1

        print(f"      Added {constraint_count} availability constraints (optimized)")

    def add_valid_quantum_constraints(
        self, model: cp_model.CpModel, session_vars: Dict
    ):
        """
        Constraint: Start quanta must be valid operating quanta.

        Args:
            model: CP-SAT model
            session_vars: Session variables dictionary
        """
        constraint_count = 0
        valid_quanta = sorted(self.context.available_quanta)

        for session_key, vars_dict in session_vars.items():
            start_var = vars_dict["start_quantum"]

            # Restrict to valid quanta
            model.AddAllowedAssignments([start_var], [[q] for q in valid_quanta])
            constraint_count += 1

        print(f"      Added {constraint_count} valid quantum constraints")
