"""
OR-Tools Proof-of-Concept for Course Scheduling

This is a SIMPLIFIED demonstration comparing OR-Tools CP-SAT solver approach
with the current DEAP-based genetic algorithm implementation.

Purpose: Educational comparison for thesis, NOT a replacement for current system.

Note: This is a minimal example showing constraint programming approach.
A full implementation would be 500-1000 lines with all constraints.
"""

from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class SimpleCourse:
    """Simplified course representation."""
    id: str
    name: str
    hours_needed: int
    course_type: str  # "theory" or "practical"
    
    
@dataclass
class SimpleGroup:
    """Simplified group representation."""
    id: str
    enrolled_courses: List[str]
    

@dataclass
class SimpleInstructor:
    """Simplified instructor representation."""
    id: str
    name: str
    qualifications: List[str]  # Course IDs they can teach


@dataclass
class SimpleRoom:
    """Simplified room representation."""
    id: str
    capacity: int
    room_type: str  # "theory" or "practical"


@dataclass
class ScheduleSlot:
    """A scheduled session."""
    course_id: str
    group_id: str
    instructor_id: str
    room_id: str
    time_slot: int
    day: int
    hour: int


def solve_with_ortools_simple(
    courses: List[SimpleCourse],
    groups: List[SimpleGroup],
    instructors: List[SimpleInstructor],
    rooms: List[SimpleRoom],
    num_days: int = 5,
    hours_per_day: int = 8
) -> Tuple[List[ScheduleSlot], Dict[str, any]]:
    """
    Solve course scheduling using Google OR-Tools CP-SAT solver.
    
    This is a SIMPLIFIED example showing the constraint programming approach.
    A complete implementation would include all constraints from the DEAP version.
    
    Args:
        courses: List of courses to schedule
        groups: List of student groups
        instructors: List of available instructors
        rooms: List of available rooms
        num_days: Number of days in the schedule
        hours_per_day: Hours per day
        
    Returns:
        Tuple of (schedule, metrics)
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        print("⚠️  OR-Tools not installed. Install with: pip install ortools")
        print("This is expected - OR-Tools is not in requirements (proof-of-concept only)")
        return [], {"status": "NOT_INSTALLED"}
    
    print("\n" + "="*70)
    print("OR-TOOLS CP-SAT SOLVER - PROOF OF CONCEPT")
    print("="*70)
    print("\nNote: This is a SIMPLIFIED demonstration.")
    print("A full implementation would be 500-1000 lines with all constraints.\n")
    
    start_time = time.time()
    
    model = cp_model.CpModel()
    
    # Time slots: days × hours_per_day
    time_slots = range(num_days * hours_per_day)
    
    print(f"Problem size:")
    print(f"  - Courses: {len(courses)}")
    print(f"  - Groups: {len(groups)}")
    print(f"  - Instructors: {len(instructors)}")
    print(f"  - Rooms: {len(rooms)}")
    print(f"  - Time slots: {len(time_slots)} ({num_days} days × {hours_per_day} hours)")
    print(f"  - Decision variables: ~{len(courses) * len(groups) * len(time_slots) * len(instructors) * len(rooms):,}")
    
    # ========================================================================
    # DECISION VARIABLES
    # ========================================================================
    # session[c, g, t, i, r] = 1 if course c for group g is scheduled
    # at time t with instructor i in room r
    
    print("\n[1/5] Creating decision variables...")
    sessions = {}
    var_count = 0
    
    for course in courses:
        for group in groups:
            if course.id not in group.enrolled_courses:
                continue  # Skip if group not enrolled
                
            for t in time_slots:
                for instructor in instructors:
                    if course.id not in instructor.qualifications:
                        continue  # Skip unqualified instructors
                        
                    for room in rooms:
                        if course.course_type != room.room_type:
                            continue  # Skip type mismatch
                            
                        var_name = f's_c{course.id}_g{group.id}_t{t}_i{instructor.id}_r{room.id}'
                        sessions[(course.id, group.id, t, instructor.id, room.id)] = \
                            model.NewBoolVar(var_name)
                        var_count += 1
    
    print(f"   Created {var_count:,} boolean variables (reduced from full combinatorial space)")
    
    # ========================================================================
    # HARD CONSTRAINTS
    # ========================================================================
    
    print("\n[2/5] Adding hard constraints...")
    constraint_count = 0
    
    # Constraint 1: Each course-group pair must be scheduled for required hours
    print("   - Course hour requirements...")
    for course in courses:
        for group in groups:
            if course.id not in group.enrolled_courses:
                continue
                
            # Sum all sessions for this course-group pair
            relevant_sessions = [
                sessions[key] for key in sessions
                if key[0] == course.id and key[1] == group.id
            ]
            
            if relevant_sessions:
                model.Add(sum(relevant_sessions) == course.hours_needed)
                constraint_count += 1
    
    # Constraint 2: No group overlap (group can't be in two places at once)
    print("   - No group time conflicts...")
    for group in groups:
        for t in time_slots:
            conflicting_sessions = [
                sessions[key] for key in sessions
                if key[1] == group.id and key[2] == t
            ]
            
            if conflicting_sessions:
                model.Add(sum(conflicting_sessions) <= 1)
                constraint_count += 1
    
    # Constraint 3: No instructor conflict (instructor can't teach two classes)
    print("   - No instructor conflicts...")
    for instructor in instructors:
        for t in time_slots:
            conflicting_sessions = [
                sessions[key] for key in sessions
                if key[3] == instructor.id and key[2] == t
            ]
            
            if conflicting_sessions:
                model.Add(sum(conflicting_sessions) <= 1)
                constraint_count += 1
    
    # Constraint 4: No room double-booking
    print("   - No room conflicts...")
    for room in rooms:
        for t in time_slots:
            conflicting_sessions = [
                sessions[key] for key in sessions
                if key[4] == room.id and key[2] == t
            ]
            
            if conflicting_sessions:
                model.Add(sum(conflicting_sessions) <= 1)
                constraint_count += 1
    
    print(f"   Added {constraint_count:,} hard constraints")
    
    # ========================================================================
    # SOFT CONSTRAINTS (simplified as penalties)
    # ========================================================================
    
    print("\n[3/5] Adding soft constraints (as penalties)...")
    
    # This is where OR-Tools becomes complex for soft constraints
    # Each soft constraint needs auxiliary variables and penalty modeling
    
    # Example: Gap penalty (simplified)
    gap_penalties = []
    
    for group in groups:
        for day in range(num_days):
            day_start = day * hours_per_day
            day_end = (day + 1) * hours_per_day
            
            # Create variables to track if group has class in each hour
            has_class = {}
            for hour in range(day_start, day_end):
                has_class[hour] = model.NewBoolVar(f'has_class_g{group.id}_t{hour}')
                
                # has_class[hour] = 1 if any session scheduled
                sessions_at_hour = [
                    sessions[key] for key in sessions
                    if key[1] == group.id and key[2] == hour
                ]
                
                if sessions_at_hour:
                    # has_class = 1 if sum(sessions) >= 1
                    model.Add(sum(sessions_at_hour) >= 1).OnlyEnforceIf(has_class[hour])
                    model.Add(sum(sessions_at_hour) == 0).OnlyEnforceIf(has_class[hour].Not())
            
            # Count gaps (hours between first and last class with no class)
            # This is complex to model - simplified version
            # In practice, would need many auxiliary variables
            
    print("   Soft constraint modeling in OR-Tools is complex")
    print("   (requires auxiliary variables and penalty engineering)")
    
    # ========================================================================
    # OBJECTIVE FUNCTION
    # ========================================================================
    
    print("\n[4/5] Setting objective function...")
    
    # For this demo, minimize number of sessions (simple objective)
    # Real objective would include soft constraint penalties
    total_sessions = sum(sessions.values())
    
    # In practice, would be: minimize(hard_violations + soft_penalties)
    # But since hard constraints are enforced, we optimize soft
    model.Minimize(total_sessions)  # Placeholder
    
    print("   Objective: Minimize total sessions (simplified)")
    print("   (Real objective would include gap penalties, preferences, etc.)")
    
    # ========================================================================
    # SOLVE
    # ========================================================================
    
    print("\n[5/5] Solving...")
    solver = cp_model.CpSolver()
    
    # Set time limit (5 minutes)
    solver.parameters.max_time_in_seconds = 300.0
    
    # Solve
    solve_start = time.time()
    status = solver.Solve(model)
    solve_time = time.time() - solve_start
    
    print("\n" + "="*70)
    print("SOLUTION STATUS")
    print("="*70)
    
    status_names = {
        cp_model.OPTIMAL: "OPTIMAL",
        cp_model.FEASIBLE: "FEASIBLE",
        cp_model.INFEASIBLE: "INFEASIBLE",
        cp_model.MODEL_INVALID: "MODEL_INVALID",
        cp_model.UNKNOWN: "UNKNOWN"
    }
    
    print(f"\nStatus: {status_names.get(status, 'UNKNOWN')}")
    print(f"Solve time: {solve_time:.2f} seconds")
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Objective value: {solver.ObjectiveValue():.2f}")
        
        # Extract solution
        schedule = []
        for key, var in sessions.items():
            if solver.Value(var) == 1:
                course_id, group_id, time_slot, instructor_id, room_id = key
                day = time_slot // hours_per_day
                hour = time_slot % hours_per_day
                
                schedule.append(ScheduleSlot(
                    course_id=course_id,
                    group_id=group_id,
                    instructor_id=instructor_id,
                    room_id=room_id,
                    time_slot=time_slot,
                    day=day,
                    hour=hour
                ))
        
        print(f"\nScheduled sessions: {len(schedule)}")
        print(f"Variables: {var_count:,}")
        print(f"Constraints: {constraint_count:,}")
        
        # Calculate metrics
        metrics = {
            "status": status_names.get(status),
            "solve_time": solve_time,
            "total_time": time.time() - start_time,
            "objective_value": solver.ObjectiveValue(),
            "num_sessions": len(schedule),
            "num_variables": var_count,
            "num_constraints": constraint_count,
            "hard_violations": 0  # OR-Tools guarantees this
        }
        
        return schedule, metrics
    
    else:
        print("\n⚠️  No solution found!")
        print("\nPossible reasons:")
        print("  - Problem is infeasible (impossible to satisfy all constraints)")
        print("  - Time limit exceeded (increase max_time_in_seconds)")
        print("  - Model error (check constraint definitions)")
        
        metrics = {
            "status": status_names.get(status),
            "solve_time": solve_time,
            "total_time": time.time() - start_time,
            "hard_violations": None
        }
        
        return [], metrics


def compare_approaches():
    """
    Compare OR-Tools vs DEAP approaches.
    
    This demonstrates the key differences in methodology.
    """
    print("\n" + "="*70)
    print("METHODOLOGY COMPARISON: OR-TOOLS vs DEAP")
    print("="*70)
    
    comparison = """
╔════════════════════════════════════════════════════════════════════════╗
║                        OR-TOOLS CP-SAT                                 ║
╠════════════════════════════════════════════════════════════════════════╣
║ Approach:        Constraint Programming (exact solver)                ║
║ Algorithm:       SAT-based branch-and-bound                            ║
║ Search:          Complete search of solution space                     ║
║ Optimality:      Guaranteed (when found)                               ║
║ Runtime:         Deterministic                                         ║
║                                                                        ║
║ Strengths:                                                             ║
║   ✓ Provably optimal solutions                                         ║
║   ✓ Fast for hard constraint satisfaction                              ║
║   ✓ Proves infeasibility                                               ║
║   ✓ Deterministic (same input → same output)                           ║
║   ✓ Scales well to large problems                                      ║
║                                                                        ║
║ Weaknesses:                                                            ║
║   ✗ Complex soft constraint modeling                                   ║
║   ✗ No multi-objective Pareto optimization                             ║
║   ✗ Black-box (less explainable)                                       ║
║   ✗ Requires constraint programming expertise                          ║
║   ✗ Lower academic novelty (standard tool)                             ║
╚════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════╗
║                      DEAP NSGA-II (Current)                            ║
╠════════════════════════════════════════════════════════════════════════╣
║ Approach:        Evolutionary Computation (metaheuristic)              ║
║ Algorithm:       Non-dominated Sorting Genetic Algorithm II            ║
║ Search:          Stochastic exploration of solution space              ║
║ Optimality:      Near-optimal (no guarantee)                           ║
║ Runtime:         Variable (stochastic)                                 ║
║                                                                        ║
║ Strengths:                                                             ║
║   ✓ Natural soft constraint handling (penalties)                       ║
║   ✓ Multi-objective Pareto optimization                                ║
║   ✓ Highly customizable (domain heuristics)                            ║
║   ✓ Explainable (evolution plots, diversity)                           ║
║   ✓ High academic/research value                                       ║
║   ✓ Excellent for trade-off exploration                                ║
║                                                                        ║
║ Weaknesses:                                                            ║
║   ✗ No optimality guarantee                                            ║
║   ✗ Slower for large problems                                          ║
║   ✗ Variable results (run-to-run)                                      ║
║   ✗ May get stuck in local optima                                      ║
║   ✗ Requires many generations                                          ║
╚════════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════════╗
║                    HYBRID APPROACH (Best of Both)                      ║
╠════════════════════════════════════════════════════════════════════════╣
║ Phase 1: OR-Tools finds feasible solution (hard constraints)          ║
║ Phase 2: DEAP optimizes soft constraints from feasible start          ║
║                                                                        ║
║ Strengths:                                                             ║
║   ✓ Guaranteed hard constraint satisfaction                            ║
║   ✓ Optimal soft constraint optimization                               ║
║   ✓ Fast convergence (warm start)                                      ║
║   ✓ Best solution quality                                              ║
║   ✓ High research value (comparative study)                            ║
║                                                                        ║
║ Weaknesses:                                                            ║
║   ✗ Higher implementation complexity                                   ║
║   ✗ Two systems to maintain                                            ║
║   ✗ Longer development time                                            ║
╚════════════════════════════════════════════════════════════════════════╝
"""
    
    print(comparison)


def demonstrate_ortools():
    """
    Run a simple demonstration of OR-Tools approach.
    
    This shows the methodology, not a complete implementation.
    """
    print("\n" + "="*70)
    print("DEMONSTRATION: OR-TOOLS PROOF-OF-CONCEPT")
    print("="*70)
    
    # Create simple test data
    courses = [
        SimpleCourse("CS101", "Programming I", 3, "theory"),
        SimpleCourse("CS102", "Data Structures", 3, "theory"),
        SimpleCourse("CS103L", "Programming Lab", 2, "practical"),
    ]
    
    groups = [
        SimpleGroup("BCE1", ["CS101", "CS102", "CS103L"]),
        SimpleGroup("BCE2", ["CS101", "CS103L"]),
    ]
    
    instructors = [
        SimpleInstructor("INS001", "Dr. Smith", ["CS101", "CS102"]),
        SimpleInstructor("INS002", "Dr. Jones", ["CS102", "CS103L"]),
        SimpleInstructor("INS003", "Dr. Brown", ["CS101", "CS103L"]),
    ]
    
    rooms = [
        SimpleRoom("R101", 50, "theory"),
        SimpleRoom("R102", 50, "theory"),
        SimpleRoom("LAB1", 30, "practical"),
    ]
    
    # Solve
    schedule, metrics = solve_with_ortools_simple(
        courses, groups, instructors, rooms,
        num_days=5, hours_per_day=8
    )
    
    if metrics["status"] == "NOT_INSTALLED":
        print("\n💡 To try OR-Tools:")
        print("   pip install ortools")
        print("   python docs/ortools_poc.py")
        return
    
    # Print results
    if schedule:
        print("\n" + "="*70)
        print("SAMPLE SCHEDULE (first 10 sessions)")
        print("="*70)
        
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        
        for i, slot in enumerate(schedule[:10]):
            day_name = days[slot.day]
            hour = 8 + slot.hour  # Assume 8 AM start
            
            print(f"\n{i+1}. {slot.course_id} for {slot.group_id}")
            print(f"   Time: {day_name} {hour}:00-{hour+1}:00")
            print(f"   Instructor: {slot.instructor_id}")
            print(f"   Room: {slot.room_id}")
        
        if len(schedule) > 10:
            print(f"\n   ... and {len(schedule) - 10} more sessions")
    
    # Show comparison
    compare_approaches()
    
    # Final recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS FOR YOUR PROJECT")
    print("="*70)
    
    recommendations = """
Based on this demonstration and your current implementation:

✅ KEEP YOUR DEAP IMPLEMENTATION

Why?
1. Your solution already works well
2. DEAP is better suited for your constraint profile (many soft constraints)
3. Higher academic value (novel algorithms vs standard tool)
4. Multi-objective optimization (Pareto fronts)
5. Explainable results (evolution plots)
6. Already invested ~19K LOC (working code)

🟡 OPTIONAL: Add OR-Tools as Enhancement

Consider hybrid approach:
1. Phase 1: OR-Tools checks feasibility (hard constraints)
2. Phase 2: DEAP optimizes from feasible start
3. Result: Best of both worlds

Development time: 2-4 weeks
Value: High (comparative study for thesis)

❌ DON'T: Full rewrite to OR-Tools

Why not?
1. High risk (4-8 weeks development)
2. May lose soft constraint optimization
3. Lower academic value (using standard tool)
4. Your DEAP solution already produces quality results
5. Not worth the effort if current solution works

📚 FOR THESIS: Add Comparison Section

Include this in your thesis:
1. Related work comparison (OR-Tools, OptaPlanner, etc.)
2. Why you chose evolutionary approach
3. Advantages of DEAP for your problem type
4. Optional: Quick OR-Tools benchmark results

This shows you evaluated alternatives (scholarly approach).

BOTTOM LINE: You made the right choice! Keep building! 🚀
"""
    
    print(recommendations)


if __name__ == "__main__":
    demonstrate_ortools()
    
    print("\n" + "="*70)
    print("For detailed comparison, see:")
    print("  - docs/LIBRARY_COMPARISON.md (comprehensive analysis)")
    print("  - docs/WHEN_TO_USE_WHAT.md (decision guide)")
    print("="*70 + "\n")
