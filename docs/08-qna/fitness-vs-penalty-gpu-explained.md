# Fitness vs Penalty & GPU Evaluation Explained

**Date**: 2025-11-23  
**Topic**: Terminology confusion and GPU acceleration benefits

## Core Confusion: Fitness vs Penalty

### Terminology (They Mean the Same Thing!)

In this codebase, **"fitness" and "penalty" are used interchangeably**:

```python
def evaluate(individual) -> Tuple[int, int]:
    """Returns: (hard_penalty_score, soft_penalty_score)"""
    hard_penalty = 0
    soft_penalty = 0
    # ... calculate violations ...
    return (hard_penalty, soft_penalty)  # ← This IS the fitness!
```

**Key Point**: Your fitness function **IS** a penalty function. We're minimizing violations (penalties), so:
- **Lower fitness = Better solution**
- **Fitness weights in DEAP**: `(-1.0, -1.0)` means "minimize both objectives"

### What Does "Fitness Evaluation" Mean?

**Fitness evaluation** = The process of:
1. **Decoding** individual genes → concrete schedule sessions
2. **Checking constraints** (hard + soft) against the schedule
3. **Counting violations** for each constraint
4. **Weighting** violations by importance
5. **Summing** into two totals: `(hard_penalty, soft_penalty)`

**Example**:
```python
individual = [SessionGene(...), SessionGene(...), ...]  # Chromosome (100 genes)
                    ↓ decode
sessions = [CourseSession(...), CourseSession(...), ...]  # Concrete schedule
                    ↓ check constraints
violations = {
    "instructor_exclusivity": 772 raw conflicts,  # Same instructor, same time
    "room_exclusivity": 1043 raw conflicts,       # Same room, same time
    # ... 8 total hard constraints
}
                    ↓ weight and sum
hard_penalty = 3.0×772 + 3.0×1043 + ... = 12,778  # Weighted sum
soft_penalty = 1.5×16 + 1.0×4 + ... = 1165.6

fitness = (12778, 1165.6)  # ← Tuple of (hard, soft) penalties
```

## GPU Acceleration - What Does It Actually Speed Up?

### ❌ What GPU Does NOT Speed Up

**GPU does NOT accelerate most of your fitness evaluation** because:

1. **Decoding is CPU-only**: Converting `SessionGene` → `CourseSession` requires Python object creation, dict lookups, string operations (not GPU-friendly)

2. **Most constraints are CPU-only**: Complex logic like:
   - Instructor qualification checking (string matching, set operations)
   - Room suitability validation (type checking, capacity comparisons)
   - Course completeness checks (counting sessions per course)
   - Schedule compactness calculations (gap detection, graph traversal)

3. **Small data structures**: Your problem has ~100 genes, ~50 courses, ~20 instructors - **too small for GPU benefit**

### ✅ What GPU DOES Speed Up (Limited!)

The GPU evaluator (`gpu_batch_evaluator.py`) **only accelerates 3 simple constraints**:

```python
# GPU-friendly (vectorizable) constraints:
1. Instructor double-booking: Same instructor at overlapping times
2. Room double-booking: Same room at overlapping times  
3. Group conflicts: Same group at overlapping times
```

**Why these work on GPU**: They're pure arithmetic comparisons on integers (time slots, IDs).

### GPU Code Architecture

```python
# GPU evaluator converts genes to tensor:
tensor[individual, gene, :] = [start_time, instructor_hash, room_hash, num_groups, duration]
                               ↓ GPU vectorized operations
# Parallel pairwise comparison of ALL genes against ALL genes
# Finds overlaps using: time_i < time_j + duration_j AND time_j < time_i + duration_i
```

**The problem**: This is only checking **3 out of 12 constraints**!

## Your Actual Performance Bottleneck

Looking at your timing breakdown:
```
t=00:07:57 (ops=00:00:10, eval=00:03:55, replace=00:00:04, metrics=00:00:01, other=00:03:47)
```

**Breakdown**:
- `eval=00:03:55` (49% of time) - **Fitness evaluation**
- `other=00:03:47` (47% of time) - **Unaccounted overhead**
- `ops=00:00:10` (2% of time) - Selection/crossover/mutation
- `replace=00:00:04` (<1%) - Population replacement
- `metrics=00:00:01` (<1%) - Metric calculation

### Why is GPU NOT helping you?

**Batch size threshold**: GPU only kicks in for `len(population) >= 50`:

```python
# src/core/ga_scheduler.py line 1504
if self.gpu_evaluator and self.gpu_evaluator.enabled and len(invalid) >= 50:
    # Use GPU
else:
    # Use CPU
```

**The issue**: Your evaluation happens in two places:

1. **Initial population** (gen 0): Evaluates 200 individuals → GPU SHOULD activate
2. **Per-generation offspring**: Evaluates ~200 offspring → GPU SHOULD activate
3. **But**: Invalid individuals after crossover/mutation might be < 50 → CPU fallback!

**Check your logs**: Is GPU actually being used? The evaluator should log:
```
✓ GPU Evaluator initialized: NVIDIA GeForce RTX 3060 (12GB)
  Optimal batch size: 256
```

## Why GPU Acceleration is Limited for Timetabling

### Fundamental Problem: Complex Constraint Logic

Most timetabling constraints require **complex Python logic** that doesn't vectorize:

```python
# Example: Instructor qualifications (NOT GPU-friendly)
def instructor_qualifications(sessions, courses):
    violations = 0
    for session in sessions:
        instructor = session.instructor
        course = courses[session.course_id]
        
        # Complex string/set operations (CPU-only)
        if course.required_qualification not in instructor.qualifications:
            violations += 1
    return violations
```

**Why GPU can't help**:
- Python object traversal (`session.instructor`, `instructor.qualifications`)
- Dictionary lookups (`courses[session.course_id]`)
- String comparisons (`qualification not in qualifications`)
- Variable-length data structures (different instructors have different quals)

### What WOULD Benefit from GPU

Problems with **homogeneous numerical operations**:
- Image processing (matrix convolutions)
- Neural network training (matrix multiplications)
- Physics simulations (particle interactions)
- Graph algorithms on LARGE graphs (>10K nodes)

**Timetabling is NOT this type of problem**.

## The Real Speedup Opportunity: Parallelization

### Current Setup (Already Implemented)

```yaml
# configs/base.yaml
parallel:
  use_multiprocessing: true
  num_workers: null  # Auto: uses your CPU core count
```

**This parallelizes**:
- Fitness evaluation across multiple CPU cores
- Crossover/mutation operators (parallel processing)
- Feasibility checks (concurrent validation)

**Your system**: If you have 32 cores, you're evaluating 32 individuals simultaneously on CPU.

### Why This is Better Than GPU (For Your Problem)

**CPU parallelization benefits**:
- ✅ Works for ALL constraints (not just 3 simple ones)
- ✅ Full Python logic support (dicts, strings, objects)
- ✅ No data transfer overhead (CPU ↔ GPU is slow)
- ✅ Better for small-medium populations (200-500)

**GPU would only help if**:
- ✅ Population size = 10,000+ individuals
- ✅ Constraints are purely arithmetic (no string/dict operations)
- ✅ Data structures are homogeneous (fixed-size tensors)

## Recommendations

### 1. Verify GPU is Actually Being Used

Run diagnostics:
```powershell
uv run diagnose
```

Check for:
```
✓ CUDA Available: True
✓ GPU: NVIDIA GeForce RTX 3060 (12GB)
✓ PyTorch CUDA: True
```

Then check your run logs for:
```
✓ GPU Evaluator initialized: NVIDIA GeForce RTX 3060 (12GB)
```

If you see "GPU Evaluator disabled" → GPU is NOT being used!

### 2. Profile Where Time is Actually Spent

The `eval=00:03:55` time includes:
1. Decoding (gene → sessions)
2. Constraint checking (ALL 12 constraints)
3. Weighting and summing

**To find the bottleneck**, add detailed timing:
```python
# In fitness.py, add timing per constraint
import time
for constraint_name, constraint_info in enabled_hard_constraints.items():
    start = time.time()
    penalty = constraint_func(sessions)
    duration = time.time() - start
    if duration > 0.01:
        print(f"{constraint_name}: {duration:.4f}s")
```

### 3. Understand the `other=00:03:47` Mystery

**47% of your time is unaccounted!** This could be:
- Memory allocation/garbage collection
- Python interpreter overhead
- Logging/console output
- Data structure copying

**Enable detailed profiling**:
```yaml
# configs/base.yaml
performance:
  enable_profiling: true
  show_per_generation: true
```

This breaks down timing into phases: selection, crossover, mutation, evaluation, replacement, repair, metrics.

### 4. Realistic GPU Expectations

**IF** GPU is working correctly, expect:
- **Speedup**: 2-3x at best (only 3/12 constraints accelerated)
- **NOT**: 10-50x (that's for pure numerical problems)

**Why?**:
```
Total fitness time: 3:55
  - GPU-friendly constraints: ~30s (instructor/room/group conflicts)
  - CPU-only constraints: ~3:25 (qualifications, completeness, compactness, etc.)
  
GPU speedup: 30s → 1s (30x faster)
BUT: Total speedup: 3:55 → 3:26 (only 12% faster overall)
```

## Bottom Line

### Is GPU Worth It?

**For your problem size (200 pop, ~100 genes)**: **NO**, CPU multiprocessing is better.

**GPU would help if**:
- Population = 5,000-10,000 individuals
- Most constraints were arithmetic (they're not)
- You had GPU-friendly constraint implementations (you don't)

### What's Actually Slow?

Your **516 seconds per generation** is NOT normal. Something else is wrong:

**Expected times** (200 pop, 32 CPU cores):
- Selection: <1s
- Crossover: 2-3s
- Mutation: 2-3s
- Evaluation: 10-20s (parallelized across 32 cores)
- Replacement: <1s
- Metrics: <1s
- **Total**: ~20-30 seconds per generation (NOT 516!)

**Your actual times suggest**:
- Multiprocessing is NOT working (evaluating sequentially?)
- Memory leak or thrashing (garbage collection pauses)
- Excessive logging/IO operations
- Some constraint is extremely slow (10x slower than others)

## Next Steps

1. **Run diagnostics**: `uv run diagnose`
2. **Check multiprocessing**: Look for "32 workers" in logs
3. **Profile constraints**: Add timing to each constraint function
4. **Investigate "other" time**: Enable detailed performance profiling
5. **Compare expected vs actual**: 516s/gen is **17x slower** than it should be!

The GPU is a red herring - your real problem is elsewhere.
