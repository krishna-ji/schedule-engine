# This File Contains Genetic Algorithm Parameters


# Number of generations - adjust based on population size
# Larger populations often need fewer generations to converge
NGEN = 200  # Production run

# Population size - INCREASED for better multiprocessing utilization
# Larger populations keep all CPU cores busy during parallel fitness evaluation
# Use POP_SIZE=10 for quick testing, 100+ for production runs
POP_SIZE = 20  # Production run - INCREASED from 10


# Crossover and mutation probabilities optimized for constraint-aware population
CXPB, MUTPB = 0.8, 0.3  # Reduced mutation to preserve good constraint relationships

# Parallelization Settings
USE_MULTIPROCESSING = True  # Now using serialized context - safe for Windows spawn
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
    "enabled": True,  # Production test with repairs enabled
    "max_iterations": 3,  # CRITICAL: Increased from 2 → 3 for better availability violation fixing
    # When to apply repairs
    "apply_after_mutation": True,  # Fix violations after mutation (recommended)
    "apply_after_crossover": True,  # CRITICAL FIX: Enable to prevent crossover from breaking repairs
    # Memetic mode - apply intensive local search to elite solutions
    "memetic_mode": False,  # DISABLED: Not helping with stuck violations, redirect effort to global repair
    "elite_percentage": 0.1,  # OPTIMIZED: Top 10% only (reduced from 20%) - less overhead
    "memetic_iterations": 5,  # OPTIMIZED: reduced from 10 → 5 for balance between quality and speed
    # Threshold-based repair (optional)
    "violation_threshold": None,  # Always repair (no threshold)
    # ========================================
    # NEW: Selective Repair Optimization (Option B)
    # ========================================
    # Only repair genes with violations (3-4× faster than full repair)
    "selective_mode": True,  # Enable selective repair (RECOMMENDED for performance)
    "detection_strategy": "hybrid",  # "fast", "full", or "hybrid" (recommended)
    "recheck_after_repair": True,  # Re-detect violations after each iteration
    # ========================================
    # NEW: Adaptive Repair Strategy (Hybrid: Stagnation + Periodic)
    # ========================================
    # Dynamically switches between selective (fast) and full (intensive) repair
    # based on search progress. Combines stagnation detection with periodic triggers.
    #
    # Trigger Priority (highest to lowest):
    #   1. Intensive: Every intensive_interval (e.g., gen 20, 40, 60...) → max_iterations=10
    #   2. Stagnation: No HC improvement for 'window' gens → max_iterations=5
    #   3. Periodic: Every interval (e.g., gen 10, 30, 50...) → max_iterations=5
    #
    # Expected Behavior:
    #   - Gen 10: Periodic trigger (if not intensive)
    #   - Gen 15: Stagnation trigger (if HC plateaus for 5 gens)
    #   - Gen 20: Intensive trigger (overrides periodic)
    #
    # See docs/ADAPTIVE_REPAIR_HYBRID_STRATEGY.md for detailed explanation
    "adaptive_repair": {
        "enabled": True,  # Enable hybrid adaptive strategy
        # Stagnation-based trigger
        "stagnation_trigger": {
            "enabled": True,  # Enable stagnation detection
            "window": 3,  # OPTIMIZED: Reduced from 5 → 3 to detect stagnation faster and trigger repairs sooner
            "metric": "best_hc",  # Metric to track: "best_hc", "avg_hc", "best_fitness"
            "threshold": 0.0,  # Minimum improvement required (0 = any improvement counts)
        },
        # Periodic trigger (generation-based)
        "periodic_trigger": {
            "enabled": True,  # Enable periodic deep repair
            "interval": 10,  # Apply full repair every N generations (None = disable)
            "intensive_interval": 20,  # Extra intensive repair every N gens (None = disable)
        },
        # Action when triggered (normal trigger)
        "trigger_action": {
            "repair_mode": "full",  # Switch to "full" repair (from "selective")
            "max_iterations": 8,  # More intensive than normal (normal=2) - INCREASED from 5 to 8
        },
        # Action for intensive trigger (every intensive_interval)
        "intensive_action": {
            "repair_mode": "full",  # Full repair mode
            "max_iterations": 15,  # Maximum intensity - INCREASED from 10 to 15
        },
    },
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
