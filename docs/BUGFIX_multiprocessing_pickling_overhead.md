# CRITICAL BUG: Multiprocessing is 2.7× SLOWER Due to Pickling Overhead

## Executive Summary

**Status:** 🔴 **CRITICAL BUG CONFIRMED**  
**Impact:** Multiprocessing is **2.7× slower** than sequential execution  
**Root Cause:** Partial function with bound context pickled on every `pool.map()` call  
**Fix:** Use worker initialization with process-local context  
**Expected Improvement:** **2.76× speedup** (from 0.37× to 2.76×)

---

## Problem Description

Despite `USE_MULTIPROCESSING=True` and proper `toolbox.map()` usage, the GA runs **slower** with multiprocessing enabled:

```
Sequential execution:  0.090s (3.7ms per individual)
Parallel execution:    0.239s (10.0ms per individual)
Speedup:               0.37× (2.7× SLOWER!)
```

---

## Root Cause Analysis

### Current Implementation (BUGGY)

```python
# src/core/ga_scheduler.py, Line ~245
toolbox.register(
    "evaluate",
    evaluate,
    courses=self.context.courses,        # ← 36 KB
    instructors=self.context.instructors, # ← 49 KB
    groups=self.context.groups,          # ← 14 KB
    rooms=self.context.rooms,            # ← 15 KB
)

# When pool.map() is called:
# 1. toolbox.evaluate is a partial(evaluate, courses=..., groups=..., rooms=...)
# 2. This partial function (~120 KB) must be pickled
# 3. On Windows (spawn), pickled function sent to EVERY worker
# 4. Pickling happens on EVERY pool.map() call
# 5. Total overhead: ~11ms per call (2.7× the evaluation time!)
```

### Why This Fails on Windows

**Linux (fork):**
- Workers inherit parent memory via copy-on-write
- Partial function already in memory
- Low overhead

**Windows (spawn):**
- Workers are fresh processes
- Everything must be pickled and sent via IPC
- **120 KB context pickled on every `pool.map()` call**
- Overhead dominates for fast evaluations (< 10ms)

---

## Proof of Bug

### Test Results

```
Test 1: Current Approach (partial function)
  Time: 0.239s (10.0ms per individual)
  Speedup: 0.37× (SLOWER than sequential!)

Test 2: Worker Initialization (proposed fix)
  Time: 0.032s (1.4ms per individual)
  Speedup: 2.76× (FASTER than sequential!)

Test 3: Sequential Baseline
  Time: 0.090s (3.7ms per individual)
```

**Improvement:** Worker initialization is **86.4% faster** than partial approach!

---

## Proposed Fix

### Solution: Worker Initialization

Initialize each worker **once** with the context, avoiding repeated pickling:

```python
# src/core/ga_scheduler.py

# Module-level worker context
_WORKER_CONTEXT = None

def _worker_init(courses, instructors, groups, rooms):
    """Initialize worker process with evaluation context."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        'courses': courses,
        'instructors': instructors,
        'groups': groups,
        'rooms': rooms,
    }

def _worker_evaluate(individual):
    """Worker evaluation function using process-local context."""
    from src.ga.evaluator.fitness import evaluate
    return evaluate(
        individual,
        courses=_WORKER_CONTEXT['courses'],
        instructors=_WORKER_CONTEXT['instructors'],
        groups=_WORKER_CONTEXT['groups'],
        rooms=_WORKER_CONTEXT['rooms'],
    )

class GAScheduler:
    def __init__(self, config, context, hard_names, soft_names, pool=None):
        # ... existing code ...
        
        # NEW: If pool provided, it should be initialized with context
        if pool is not None:
            # Pool must be created with initializer in main.py
            self.pool = pool
        else:
            self.pool = None

    def setup_toolbox(self):
        """Initialize DEAP toolbox with operators."""
        self.toolbox = base.Toolbox()

        # NEW: Register worker evaluation or regular evaluation
        if self.pool is not None:
            # Use worker-based evaluation (context already in workers)
            self.toolbox.register("evaluate", _worker_evaluate)
            self.toolbox.register("map", self.pool.map)
        else:
            # Use direct evaluation (single-threaded)
            self.toolbox.register(
                "evaluate",
                evaluate,
                courses=self.context.courses,
                instructors=self.context.instructors,
                groups=self.context.groups,
                rooms=self.context.rooms,
            )
        
        # ... rest of setup ...
```

### Changes to main.py

```python
# main.py

def main():
    pool = None

    if USE_MULTIPROCESSING:
        import multiprocessing
        from src.core.ga_scheduler import _worker_init
        
        # NEW: Create pool with initializer
        pool = multiprocessing.Pool(
            processes=NUM_WORKERS,
            initializer=_worker_init,
            initargs=(courses, instructors, groups, rooms),
        )
    
    # ... rest of workflow ...
```

**Problem:** `courses`, `instructors`, etc. are not available in `main()` yet!

### Better Solution: Initialize in workflow

```python
# src/workflows/standard_run.py

def run_standard_workflow(..., pool=None):
    # ... load data ...
    
    # NEW: If pool provided, initialize workers with context
    if pool is not None:
        from src.core.ga_scheduler import _worker_init
        # Close the uninit pool and recreate with init
        pool.close()
        pool.join()
        
        pool = multiprocessing.Pool(
            processes=pool._processes,
            initializer=_worker_init,
            initargs=(context.courses, context.instructors, context.groups, context.rooms),
        )
    
    # ... run GA with initialized pool ...
```

---

## Implementation Plan

### Phase 1: Add Worker Functions (src/core/ga_scheduler.py)

1. Add module-level `_WORKER_CONTEXT` variable
2. Add `_worker_init()` function
3. Add `_worker_evaluate()` function
4. Modify `setup_toolbox()` to use worker evaluation when pool provided

### Phase 2: Update Pool Creation (main.py)

1. Import `_worker_init`
2. Create pool with `initializer=_worker_init`
3. Pass `initargs` with context (AFTER loading data)

### Phase 3: Fix Workflow Ordering (src/workflows/standard_run.py)

1. Accept pool parameter
2. Re-create pool with initializer after loading context
3. Pass initialized pool to GAScheduler

---

## Expected Results

### Performance

| Population | Sequential | Parallel (current) | Parallel (fixed) | Improvement |
|------------|------------|-------------------|------------------|-------------|
| 10         | 0.04s      | 0.22s (0.18×)     | 0.02s (2.0×)     | **11× faster** |
| 24         | 0.09s      | 0.24s (0.38×)     | 0.03s (3.0×)     | **8× faster** |
| 50         | 0.19s      | 0.50s (0.38×)     | 0.06s (3.2×)     | **8× faster** |
| 100        | 0.38s      | 1.00s (0.38×)     | 0.12s (3.2×)     | **8× faster** |

### CPU Utilization

- **Before:** 100% on 1 core (multiprocessing not working)
- **After:** 70-85% on all cores (proper parallelization)

---

## Testing Protocol

1. Apply fixes to `ga_scheduler.py`, `main.py`, `standard_run.py`
2. Run `python test/test_worker_init.py` → Should show 2.5-3× speedup
3. Run `python main.py` with `POP_SIZE=50, NGEN=10`
4. Monitor Task Manager → Should see all cores active
5. Compare time with `USE_MULTIPROCESSING=False` → Should be 3-4× faster

---

## Affected Files

1. `src/core/ga_scheduler.py` - Add worker functions, modify setup_toolbox()
2. `main.py` - Create pool with initializer (AFTER context loaded)
3. `src/workflows/standard_run.py` - Re-initialize pool with context
4. `docs/BUGFIX_multiprocessing_pickling_overhead.md` - This document

---

## Alternative Solutions (Not Recommended)

### 1. Switch to `fork` start method
```python
multiprocessing.set_start_method('fork')  # NOT AVAILABLE ON WINDOWS
```
❌ Not viable - Windows doesn't support fork

### 2. Use threading instead
```python
from multiprocessing.pool import ThreadPool
pool = ThreadPool(processes=8)
```
❌ Python GIL limits parallelism for CPU-bound tasks

### 3. Use shared memory
```python
from multiprocessing import Manager
manager = Manager()
shared_context = manager.dict(...)
```
❌ High IPC overhead, complex implementation

### 4. Increase evaluation complexity
❌ Doesn't solve the architectural issue

---

## Conclusion

The current multiprocessing implementation has a **critical bug** that makes it **2.7× slower** than sequential execution on Windows. The fix (worker initialization) is straightforward and provides **2.76× speedup**, making multiprocessing actually useful.

**Priority:** 🔴 **CRITICAL** - Fix immediately before production use

---

**Date:** October 24, 2025  
**Discovered by:** Deep code audit  
**Status:** Ready for implementation
