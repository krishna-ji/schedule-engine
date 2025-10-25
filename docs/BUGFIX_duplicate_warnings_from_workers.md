# Bugfix: Duplicate Warnings During Initialization

## Problem

During population initialization, warnings about missing courses were printed hundreds of times:

```
[!] Warning: Course ME706 not found for group BME7
[!] Warning: Course ENCE 256 not found for group BCE4
[!] Warning: Course CE604 not found for group BCE5
[!] Warning: Course ENIE 254 not found for group BIE4
[!] Warning: Course ME706 not found for group BME7
... (repeated 100+ times)
```

**User question**: "these same errors are also shown by all threads or what?"

## Root Cause Analysis

The warnings were **NOT** from worker threads - they were from the **main process** calling `generate_course_group_pairs()` multiple times:

1. **Hybrid initialization** creates 25 greedy + 50 smart + 25 random individuals
2. **Greedy construction** calls `_greedy_construction()` 25 times (once per individual)
3. Each call generated the course-group pairs **independently** and printed warnings
4. Result: Same 4 warnings printed **25 times** for greedy + **50 times** for smart = **75 duplicates**

### Why This Happened

**Before fix:**
```python
# hybrid_population.py
for i in range(greedy_count):
    pairs = generate_course_group_pairs(...)  # ← Generates pairs + prints warnings
    individual = _greedy_construction(pairs, ...)
```

**The pairs were identical every time** but generated from scratch in a loop!

## Solution

**Cache the pairs** - generate them once, reuse for all individuals:

### Changes Made

#### 1. `src/ga/hybrid_population.py`
- Generate pairs **once** before the loop
- Pass cached pairs to construction functions
- Reduced pair generation from 75× to **1×**

**Before:**
```python
def generate_hybrid_population(...):
    for i in range(greedy_count):
        pairs = generate_course_group_pairs(...)  # Called 25 times
        individual = _greedy_construction(pairs, ...)
```

**After:**
```python
def generate_hybrid_population(...):
    # Generate pairs ONCE (prints warnings only once)
    all_pairs = generate_course_group_pairs(courses, groups, hierarchy, silent=False)
    
    for i in range(greedy_count):
        individual = _greedy_construction(all_pairs, ...)  # Reuse cached pairs
```

#### 2. `src/ga/course_group_pairs.py`
- Added `silent` parameter to suppress warnings when needed
- **Removed unused test code** that was importing non-existent `src.utils.console`
- Fixed import error that was causing warnings

**Key change:**
```python
def generate_course_group_pairs(..., silent: bool = False):
    if course not in courses:
        if not silent:  # ← Only print if not silenced
            print(f"[!] Warning: Course {course_code} not found for group {group_id}")
```

## Results

**Before:**
- 75+ duplicate warnings during initialization
- Confusing output that looked like worker thread spam
- Unnecessary computation (regenerating identical pairs)

**After:**
- Warnings printed **once** per missing course
- Clean, readable output
- Performance improvement (pairs generated 1× instead of 75×)

## Side Effects

### Positive
- **Performance boost**: Pair generation is computationally cheap but not free
- **Cleaner console**: No more spam-like output
- **Better debugging**: Actual issues are now visible

### None Negative
- All functionality preserved
- API is backward compatible (silent defaults to False)

## Files Modified

1. **src/ga/hybrid_population.py**
   - Cache pairs before construction loops
   - Pass cached pairs to `_greedy_construction()` and `_random_construction()`

2. **src/ga/course_group_pairs.py**
   - Add `silent` parameter to `generate_course_group_pairs()`
   - Remove unused test code (fixed import error)
   - Conditional warning printing

## Verification

Run the GA:
```powershell
python main.py
```

**Expected output:**
```
Hybrid initialization: 25 greedy, 50 smart, 25 random
[!] Warning: Course ME706 not found for group BME7
[!] Warning: Course ENCE 256 not found for group BCE4
[!] Warning: Course CE604 not found for group BCE5
[!] Warning: Course ENIE 254 not found for group BIE4
Found 527 course-group pairs to schedule
Generated 100 individuals with average 527.0 genes each
```

**Notice**: Each warning appears **once** instead of 75+ times!

## Related Issues

- **Previous bug**: Worker processes printing duplicate tables → Fixed with environment variables
- **This bug**: Main process printing duplicate warnings → Fixed with pair caching
- **Import error**: `src.utils.console` not existing → Fixed by removing unused test code

## Lessons Learned

1. **Cache expensive operations**: Don't regenerate identical data in loops
2. **Profile before blaming threads**: The warnings weren't from workers!
3. **Remove dead code**: Unused test code can cause import errors
4. **Add silent flags**: Useful for batch operations where warnings aren't needed
