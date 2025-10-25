# Bug Fix: Duplicate Worker Warnings

## Problem

When running with multiprocessing enabled, warning messages were printed multiple times (once per worker process):

```
[!] Warning: Course ME706 not found for group BME7
[!] Warning: Course ENCE 256 not found for group BCE4
[!] Warning: Course CE604 not found for group BCE5
[!] Warning: Course ENIE 254 not found for group BIE4
[!] Warning: Course ME706 not found for group BME7  # Repeated by worker 2
[!] Warning: Course ENCE 256 not found for group BCE4
[!] Warning: Course CE604 not found for group BCE5
[!] Warning: Course ENIE 254 not found for group BIE4
... (repeated 5+ times for 5 workers)
```

Also, informational messages like "Found 527 course-group pairs" and "Generated X individuals" were printed multiple times.

---

## Root Cause

**Worker processes load data independently** and call population initialization functions, which print warnings and info messages to stdout. Since there are typically 5-8 worker processes, each message appears 5-8 times.

**Why this happens:**
1. Main process initializes data → prints warnings once ✓
2. Worker processes initialize independently (to avoid pickling overhead)
3. Each worker calls `generate_course_group_pairs()` → prints same warnings ✗
4. Each worker calls `generate_course_group_aware_population()` → prints info messages ✗

---

## Solution

### Strategy: Environment Variable Flag

Added environment variable `_GA_WORKER_PROCESS="1"` to identify worker processes and suppress their output.

### Implementation

**1. Worker Initialization** (`src/core/ga_scheduler.py`)
```python
def _worker_init(data_dir: str, seed: int):
    # Set flag to indicate we're in a worker process
    os.environ["_GA_WORKER_PROCESS"] = "1"
    
    # ... rest of initialization
```

**2. Course-Group Pair Generation** (`src/ga/course_group_pairs.py`)
```python
def generate_course_group_pairs(
    courses: Dict, groups: Dict, hierarchy: Dict, silent: bool = False
) -> List[Tuple]:
    # ... pair generation logic
    
    if not matching_courses:
        if not silent:  # Only print in main process
            print(f"[!] Warning: Course {course_code} not found for group {parent_prefix}")
        continue
```

**3. Population Generators** (`src/ga/population.py`, `src/ga/hybrid_population.py`)
```python
def generate_course_group_aware_population(n: int, context: SchedulingContext) -> List:
    # Detect if we're in a worker process
    silent = os.environ.get("_GA_WORKER_PROCESS") == "1"
    
    # Generate pairs with silent flag
    pair_tuples = generate_course_group_pairs(
        context.courses, context.groups, hierarchy, silent=silent
    )
    
    # Suppress info messages in workers
    if not silent:
        print(f"Found {len(course_group_pairs)} course-group pairs to schedule")
    
    # ... generation logic
    
    if not silent:
        print(f"Generated {len(population)} individuals with average {avg} genes each")
```

---

## Files Modified

1. **`src/core/ga_scheduler.py`**
   - Added `os.environ["_GA_WORKER_PROCESS"] = "1"` in `_worker_init()`

2. **`src/ga/course_group_pairs.py`**
   - Added `silent: bool = False` parameter to `generate_course_group_pairs()`
   - Wrapped warning print with `if not silent:` check

3. **`src/ga/population.py`**
   - Added `import os`
   - Detect worker with `silent = os.environ.get("_GA_WORKER_PROCESS") == "1"`
   - Pass `silent=silent` to `generate_course_group_pairs()`
   - Wrapped info prints with `if not silent:` checks

4. **`src/ga/hybrid_population.py`**
   - Added `import os`
   - Detect worker with `silent = os.environ.get("_GA_WORKER_PROCESS") == "1"`
   - Pass `silent=silent` to `generate_course_group_pairs()`
   - Wrapped "Hybrid initialization" print with `if not silent:` check

---

## Result

### Before
```
Hybrid initialization: 25 greedy, 50 smart, 25 random
[!] Warning: Course ME706 not found for group BME7
[!] Warning: Course ENCE 256 not found for group BCE4
... (repeated 100+ times across all workers)
Found 527 course-group pairs to schedule
Generated 50 individuals with average 527.0 genes each
... (repeated 25+ times for each worker)
```

### After
```
Hybrid initialization: 25 greedy, 50 smart, 25 random
[!] Warning: Course ME706 not found for group BME7
[!] Warning: Course ENCE 256 not found for group BCE4
[!] Warning: Course CE604 not found for group BCE5
[!] Warning: Course ENIE 254 not found for group BIE4
Found 527 course-group pairs to schedule
Generated 50 individuals with average 527.0 genes each
```

**Clean, single output from main process only!** ✅

---

## Why This Approach?

### Alternative Considered: Redirect stdout in workers
```python
# Could do this in _worker_init():
sys.stdout = open(os.devnull, 'w')
```

**Problems:**
- Suppresses ALL output, including important error messages
- Makes debugging worker issues impossible
- Too blunt an instrument

### Chosen Approach: Selective Suppression
- Only suppresses informational/warning messages
- Allows errors and exceptions to still print
- Surgical fix targeting specific print statements
- Easy to extend to other modules if needed

---

## Testing

**Run with multiprocessing:**
```powershell
python main.py
```

**Expected behavior:**
- Warnings appear ONCE at startup
- "Found X course-group pairs" appears ONCE
- "Generated X individuals" appears ONCE per population generation call
- No duplicate messages from worker processes

---

## Related Bugs

- **Multiprocessing pool deadlock** → FIXED (workers load JSON directly)
- **Repair performance issues** → FIXED (optimized iteration counts)
- **Duplicate worker output** → FIXED (this document)

---

## Design Notes

**Why not suppress in `_worker_init()` stdout redirect?**

The existing stdout suppression in `_worker_init()` only covers **data loading** (linking courses/groups). Population generation happens LATER during evaluation, after stdout is restored. That's why we need the environment variable approach.

**Flow:**
```
_worker_init():
    stdout = StringIO()  # Suppress
    load_courses()       # Silent ✓
    load_groups()        # Silent ✓
    stdout = original    # Restore

_worker_evaluate(individual):
    # NOW population functions might be called
    generate_course_group_pairs()  # Would print without our fix ✗
```

---

## Conclusion

Clean, targeted fix that:
- ✅ Eliminates duplicate warnings (100+ lines → 4 lines)
- ✅ Preserves error visibility for debugging
- ✅ Maintains clean console output
- ✅ Easy to extend to other modules
- ✅ No performance impact
