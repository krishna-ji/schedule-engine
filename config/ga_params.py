# This File Contains Genetic Algorithm Parameters


# Number of generations - adjust based on population size
# Larger populations often need fewer generations to converge
NGEN = 100  # Optimized for multiprocessing (was 50)

# Population size - INCREASED for better multiprocessing utilization
# Larger populations keep all CPU cores busy during parallel fitness evaluation
# Use POP_SIZE=10 for quick testing, 100+ for production runs
POP_SIZE = 10  # Optimized for multiprocessing


# Crossover and mutation probabilities optimized for constraint-aware population
CXPB, MUTPB = 0.8, 0.3  # Reduced mutation to preserve good constraint relationships

# Parallelization Settings
USE_MULTIPROCESSING = True  # Set to False for debugging (single-threaded execution)
NUM_WORKERS = None  # None = use all available CPU cores, or specify manually (e.g., 4)

# ============================================================================
# POPULATION INTEGRITY VALIDATION
# ============================================================================
# Enable strict validation that checks if individuals maintain the same course-group pairs
# during crossover. This catches population corruption bugs but may be disabled for performance
# or to allow experimental operators that intentionally modify population structure.

VALIDATE_POPULATION_INTEGRITY = False  # Set to True to enable strict validation checks

# ============================================================================
# REPAIR HEURISTICS CONFIGURATION
# ============================================================================
# Registry-based repair system - enable/disable individual repair heuristics
# Similar to constraints configuration in config/constraints.py

REPAIR_HEURISTICS_CONFIG = {
    # ========================================
    # Global Repair Settings
    # ========================================
    "enabled": True,  # Master switch - set to False to disable ALL repairs
    "max_iterations": 2,  # REDUCED: from 3 to 2 for faster evolution (1-5 recommended)
    # When to apply repairs
    "apply_after_mutation": True,  # Fix violations after mutation (recommended)
    "apply_after_crossover": False,  # DISABLED: Crossover rarely creates violations (optimization)
    # Memetic mode - apply intensive local search to elite solutions
    "memetic_mode": False,  # Enable for elite-only iterative refinement
    "elite_percentage": 0.2,  # Top 20% get extra repair passes
    "memetic_iterations": 5,  # Extra iterations for elite individuals
    # Threshold-based repair (optional)
    "violation_threshold": None,  # Only repair if violations > threshold (None = always)
    # ========================================
    # Individual Repair Heuristics
    # ========================================
    # Format: "heuristic_name": {"enabled": bool, "priority": int}
    # Priority: Lower number = higher priority (executed first)
    # Set enabled=False to disable specific repairs
    "heuristics": {
        "repair_instructor_availability": {
            "enabled": True,
            "priority": 1,
            "description": "Fix instructor availability violations",
        },
        "repair_group_overlaps": {
            "enabled": True,
            "priority": 2,
            "description": "Fix group schedule overlaps",
        },
        "repair_room_conflicts": {
            "enabled": True,
            "priority": 3,
            "description": "Fix room double-booking conflicts",
        },
        "repair_instructor_conflicts": {
            "enabled": True,
            "priority": 4,
            "description": "Fix instructor double-booking conflicts",
        },
        "repair_instructor_qualifications": {
            "enabled": True,
            "priority": 5,
            "description": "Reassign unqualified instructors",
        },
        "repair_room_type_mismatches": {
            "enabled": True,
            "priority": 6,
            "description": "Fix room type mismatches (lab/lecture)",
        },
        "repair_session_clustering": {
            "enabled": True,  # Rearranges quanta to form blocks (preserves total count)
            "priority": 7,
            "description": "Improve session block clustering (move isolated sessions)",
        },
        "repair_incomplete_or_extra_sessions": {
            "enabled": True,
            "priority": 8,
            "description": "Add missing or remove extra sessions",
            "warning": "Can modify individual length - use with caution",
        },
    },
}

# ============================================================================
# NOTES
# ============================================================================
# - Smaller population works better with constraint-aware initialization
# - Lower mutation rate preserves the structural integrity of solutions
# - Higher crossover rate allows good solutions to spread quickly
# - Multiprocessing provides 3-6× speedup by parallelizing fitness evaluation
# - Set USE_MULTIPROCESSING=False when debugging to simplify error tracking
# - Repair heuristics fix hard violations after genetic operations
# - Configure repairs via REPAIR_HEURISTICS_CONFIG above
