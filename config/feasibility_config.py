"""
Feasibility Checker Configuration

Controls which feasibility checks are enabled before running the GA.
These checks help identify unsolvable problems early, saving computational time.
"""

# ============================================================================
# GLOBAL FEASIBILITY CHECKING
# ============================================================================

# Master switch - set to False to skip all feasibility checks
ENABLE_FEASIBILITY_CHECKS = True

# Stop execution if any check fails (recommended for production)
# Set to False to see all warnings without stopping
FAIL_ON_INFEASIBILITY = True

# ============================================================================
# INDIVIDUAL FEASIBILITY CHECKS
# ============================================================================

FEASIBILITY_CHECKS = {
    # Check 1: Instructor Workload vs. Availability
    # Verifies that total teaching demand doesn't exceed instructor availability
    "instructor_workload": {
        "enabled": True,
        "description": "Total teaching demand ≤ Total instructor availability",
        "severity": "critical",  # critical, warning, info
    },
    # Check 2: Instructor Qualification Bottleneck
    # Per-course check: Are there enough qualified instructors for each course?
    "instructor_qualification_bottleneck": {
        "enabled": True,
        "description": "Each course has sufficient qualified instructor availability",
        "severity": "critical",
    },
    # Check 3: Room Capacity Bottleneck
    # Verifies total seat-hours can accommodate total student-hours
    "room_capacity_bottleneck": {
        "enabled": True,
        "description": "Total room capacity ≥ Total student enrollment demand",
        "severity": "critical",
    },
    # Check 4: Room Feature Bottleneck
    # Per-feature check: Are there enough rooms with required features?
    "room_feature_bottleneck": {
        "enabled": True,
        "description": "Rooms with required features have sufficient availability",
        "severity": "critical",
    },
    # Check 5: Group Availability Pigeonhole Problem
    # Verifies no student group is overloaded (more courses than time slots)
    "group_pigeonhole": {
        "enabled": True,
        "description": "Each group's course load fits within available time slots",
        "severity": "critical",
    },
}

# ============================================================================
# REPORTING OPTIONS
# ============================================================================

# Generate detailed feasibility report (saved to output directory)
GENERATE_FEASIBILITY_REPORT = True

# Show feasibility results in console (with rich formatting)
SHOW_CONSOLE_OUTPUT = True

# Save feasibility report even if all checks pass
SAVE_REPORT_ON_SUCCESS = True

# ============================================================================
# TOLERANCE SETTINGS
# ============================================================================

# Allow small margin of error (e.g., 5% over-capacity might still be solvable)
# Set to 0.0 for strict checking, 0.05 for 5% tolerance
TOLERANCE_MARGIN = 0.02  # 2% tolerance

# ============================================================================
# NOTES
# ============================================================================
# - Set ENABLE_FEASIBILITY_CHECKS=False to skip all checks (not recommended)
# - Individual checks can be disabled by setting "enabled": False
# - Severity levels: critical (stops execution), warning (shows warning), info (informational)
# - Tolerance margin applies to resource capacity checks (rooms, instructors)
