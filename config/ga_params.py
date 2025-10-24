# This File Contains Genetic Algorithm Parameters


# Number of generations - adjust based on population size
# Larger populations often need fewer generations to converge
NGEN = 100  # Optimized for multiprocessing (was 50)

# Population size - INCREASED for better multiprocessing utilization
# Larger populations keep all CPU cores busy during parallel fitness evaluation
# Use POP_SIZE=10 for quick testing, 100+ for production runs
POP_SIZE = 20  # Optimized for multiprocessing


# Crossover and mutation probabilities optimized for constraint-aware population
CXPB, MUTPB = 0.8, 0.3  # Reduced mutation to preserve good constraint relationships

# Parallelization Settings
USE_MULTIPROCESSING = True  # Set to False for debugging (single-threaded execution)
NUM_WORKERS = None  # None = use all available CPU cores, or specify manually (e.g., 4)

# ============================================================================
# PHASE 1 ENHANCEMENTS (Metaheuristic Enhancement Strategy)
# ============================================================================
# These settings implement Priority 1-5 from enhance_metaheuristic.md

# Priority 5: Explicit Elitism - Always preserve top 5% of solutions
ELITE_PRESERVATION = (
    True  # Guarantees monotonic improvement (best fitness never degrades)
)
ELITE_SIZE = 0.05  # Percentage of population (5% = top solutions always survive)

# Priority 4: Adaptive Operator Probabilities - Adjust CX/MUT during evolution
USE_ADAPTIVE_PROBABILITIES = (
    True  # Explore early (more mutation), exploit late (more crossover)
)
# Note: Base CXPB/MUTPB below are used in mid-phase (30-70% progress)
# Early phase (0-30%): CX=0.7, MUT=0.4 (exploration)
# Late phase (70-100%): CX=0.9, MUT=0.2 (exploitation)

# Priority 2: Constraint-Guided Mutation - Target violations instead of random mutation
USE_CONSTRAINT_GUIDED_MUTATION = (
    True  # 80% target violations, 20% random (for diversity)
)
# Expected impact: 20-30% faster convergence to zero violations

# Priority 3: Hybrid Population Initialization - Mix greedy, smart, and random
POPULATION_STRATEGY = "hybrid"  # Options: "hybrid", "smart", "random"
# "hybrid" = 25% greedy + 50% smart + 25% random (RECOMMENDED for Phase 3)
# "smart" = 100% constraint-aware (Phase 1+2 default)
# "random" = 100% random (baseline, not recommended)
# Expected impact: 15-25% better initial quality → faster convergence

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
    "max_iterations": 5,  # ENHANCED: from 2 → 5 for aggressive repair (Phase 1.1)
    # When to apply repairs
    "apply_after_mutation": True,  # Fix violations after mutation (recommended)
    "apply_after_crossover": True,  # ENHANCED: Re-enabled for thorough repair (Phase 1.1)
    # Memetic mode - apply intensive local search to elite solutions
    "memetic_mode": True,  # ENHANCED: Enabled for elite refinement (Phase 1.1 - CRITICAL!)
    "elite_percentage": 0.2,  # Top 20% get extra repair passes
    "memetic_iterations": 10,  # ENHANCED: from 5 → 10 for intensive local search (Phase 1.1)
    # Threshold-based repair (optional)
    "violation_threshold": None,  # Always repair (no threshold)
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
