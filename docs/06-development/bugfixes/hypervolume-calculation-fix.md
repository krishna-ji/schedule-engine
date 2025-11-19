# Hypervolume Calculation Bug Fix

**Date**: November 18, 2025  
**Status**:  Fixed  
**Severity**: High (Core NSGA-II metric completely broken)

---

## Problem

Hypervolume indicator was always returning 0, making it impossible to track multi-objective optimization quality.

## Root Cause

The implementation was calling `tools.hypervolume()` from DEAP, which is **NOT** a hypervolume calculator - it's actually a **selector function** that returns the **index** of the individual with the **least hypervolume contribution** (used for archive pruning).

```python
# WRONG (old code)
hv = tools.hypervolume(pareto_front, list(ref_point))  # Returns index, not volume!
```

The function signature:
```python
tools.hypervolume(front, **kargs)  # Returns INDEX of worst contributor, not volume
```

This explains why it was returning 0 - it was returning **index 0**, not the actual hypervolume value!

---

## Solution

Implemented a proper 2D hypervolume calculator using the **sweep-line algorithm** with O(n log n) complexity:

### Algorithm Steps

1. **Extract Pareto Front**: Filter to non-dominated solutions only
2. **Sort** by first objective (hard constraints) in ascending order
3. **For each point**, calculate rectangle contribution:
   - Width: from current point to next point (or reference if last)
   - Height: from current point to reference in second objective
   - Area: width × height
4. **Sum** all contributions

### Implementation

```python
def calculate_hypervolume(population: List, ref_point: Tuple[float, float] = None) -> float:
    """Calculate 2D hypervolume using sweep-line algorithm."""
    if not population:
        return 0.0
    
    # Extract Pareto front
    pareto_front = tools.sortNondominated(
        population, len(population), first_front_only=True
    )[0]
    
    if not pareto_front:
        return 0.0
    
    # Extract fitness values
    fitnesses = np.array([ind.fitness.values for ind in pareto_front])
    
    # Auto-compute reference point if not provided
    if ref_point is None:
        max_hard = np.max(fitnesses[:, 0])
        max_soft = np.max(fitnesses[:, 1])
        ref_point = (max_hard * 1.1 + 1.0, max_soft * 1.1 + 1.0)
    
    ref_hard, ref_soft = ref_point
    
    # Validate reference point dominates all points
    if np.any(fitnesses[:, 0] >= ref_hard) or np.any(fitnesses[:, 1] >= ref_soft):
        ref_hard = max(ref_hard, np.max(fitnesses[:, 0]) * 1.2 + 10.0)
        ref_soft = max(ref_soft, np.max(fitnesses[:, 1]) * 1.2 + 10.0)
    
    # Sort by first objective
    sorted_indices = np.argsort(fitnesses[:, 0])
    sorted_fitnesses = fitnesses[sorted_indices]
    
    hypervolume = 0.0
    
    # Calculate rectangle contributions
    for i, point in enumerate(sorted_fitnesses):
        obj1, obj2 = point
        
        # Width: to next point or reference
        if i < len(sorted_fitnesses) - 1:
            next_obj1 = sorted_fitnesses[i + 1, 0]
        else:
            next_obj1 = ref_hard
        
        width = next_obj1 - obj1
        height = ref_soft - obj2
        
        if width > 0 and height > 0:
            hypervolume += width * height
    
    return float(hypervolume)
```

---

## Verification Tests

### Test 1: Single-Point Pareto Front
- **Fitness**: `(5.0, 50.0)`
- **Reference**: `(10.0, 100.0)`
- **Expected**: `(10-5) × (100-50) = 5 × 50 = 250`
- **Result**: `250.0` 

### Test 2: Multi-Point Pareto Front (3 solutions)
- **Fitness**: `[(2.0, 80.0), (5.0, 50.0), (8.0, 30.0)]`
- **Reference**: `(10.0, 100.0)`
- **Manual Calculation**:
  - Rectangle 1: `(5-2) × (100-80) = 3 × 20 = 60`
  - Rectangle 2: `(8-5) × (100-50) = 3 × 50 = 150`
  - Rectangle 3: `(10-8) × (100-30) = 2 × 70 = 140`
  - Total: `60 + 150 + 140 = 350`
- **Result**: `350.0` 

### Test 3: Dominated Solutions (Filtered Correctly)
- **Population**: `[(2.0, 80.0), (5.0, 50.0), (8.0, 30.0), (6.0, 60.0)]`
- **Pareto Front**: `[(2.0, 80.0), (5.0, 50.0), (8.0, 30.0)]` (4th dominated)
- **Result**: Same as Test 2 (`350.0`) 

### Test 4: Feasible vs Infeasible Populations
- **Feasible** (hard=0): Properly calculated 
- **Infeasible** (hard>0): Properly calculated 

---

## Files Modified

- `src/metrics/hypervolume.py` - Complete rewrite of `calculate_hypervolume()` function
  - Added `import numpy as np`
  - Implemented sweep-line algorithm
  - Enhanced documentation with algorithm description

---

## Impact

### Before Fix
-  Hypervolume always 0 in `logger_constraints.csv`
-  Plots showed flat zero line
-  No way to assess multi-objective convergence
-  Could not compare NSGA-II performance across runs

### After Fix
-  Hypervolume properly tracks Pareto front quality over generations
-  Enables evaluation of convergence (proximity to optimal front)
-  Enables evaluation of diversity (spread of solutions)
-  Can compare algorithm configurations and parameters

---

## Other Metrics Verified

Tested all other NSGA-II metrics - **all working correctly**:

| Metric | Status | Purpose |
|--------|--------|---------|
| **Hypervolume** |  **FIXED** | Multi-objective quality (convergence + diversity) |
| **Spacing** |  Working | Uniformity of Pareto front distribution |
| **Spread** |  Working | Extent and distribution quality |
| **IGD** |  Working | Inverted Generational Distance (convergence + coverage) |
| **Pareto Front Size** |  Working | Number of non-dominated solutions |
| **Feasibility Rate** |  Working | Percentage of solutions with hard=0 |

---

## Lessons Learned

1. **Never assume library function names are obvious** - `tools.hypervolume()` does NOT calculate hypervolume
2. **Always verify metrics with manual calculations** - caught the bug immediately in testing
3. **DEAP documentation can be misleading** - need to check actual function signatures
4. **2D hypervolume is simple** - sweep-line algorithm is straightforward and fast
5. **Higher dimensions need different algorithms** - WFG algorithm required for 3+ objectives

---

## Future Improvements

1. **Optional: Use pymoo library** - Has battle-tested hypervolume implementation
2. **Add hypervolume contribution calculation** - For archive maintenance
3. **Add hypervolume difference tracking** - Generation-to-generation improvement
4. **Validate against known benchmarks** - Test on ZDT/DTLZ problems

---

## References

- **DEAP Source Code**: `deap/tools/indicator.py` - Confirms `hypervolume()` is a selector
- **Hypervolume Algorithm**: Sweep-line for 2D, WFG for higher dimensions
- **NSGA-II Paper**: Deb et al. (2002) - Original multi-objective algorithm
