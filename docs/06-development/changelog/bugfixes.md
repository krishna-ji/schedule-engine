# Bugfix Changelog

Chronological record of bug fixes with brief summaries. For detailed technical analysis, see `docs/06-development/bugfixes/`.

---

## [2025-11-23] GPU Removed from GA Loop (CPU-Only Fitness Evaluation)

**Severity**: Enhancement (Performance Clarification)  
**Component**: `ga_scheduler.py` fitness evaluation  
**Impact**: Simplified architecture, clearer separation of concerns

**Rationale**: GPU acceleration provided minimal benefit for GA fitness evaluation due to:
1. Only 3/12 constraints were GPU-friendly (instructor/room/group conflicts)
2. Most constraints require complex Python logic (dicts, strings, sets) - not vectorizable
3. Small problem size (~200 pop, ~100 genes) - GPU overhead > benefit
4. CPU multiprocessing already provides excellent parallelization (32 cores)

**Change**: Removed GPU evaluator from GA loop entirely:
- Deleted GPU batch evaluation code path
- Removed GPU evaluator initialization
- Simplified to CPU-only multiprocessing

**GPU Still Used For**: RL training and inference (where it truly excels - neural network computations)

**Performance Impact**: None (GPU was providing <5% speedup at best, often slower due to overhead)

**Files**: `src/core/ga_scheduler.py` (removed GPU import, init, and evaluation branching)

**Verification**: Run `uv run nsga --test` - should show "GA fitness evaluation: CPU multiprocessing only"

---

## [2025-11-23] Constraint Display Confusion (hc total vs individual values)

**Severity**: Medium (Display Issue)  
**Component**: Console output in `ga_scheduler.py`  
**Impact**: User confusion - displayed constraint breakdown didn't sum to total

**Problem**: Console showed `hc=27` (total) but individual values like `hc8=3129` that summed to 12,778. User thought constraint calculation was broken.

**Root Cause**: Display code showed **weighted penalties** (`weight × raw_violations`) as individual constraint values, but users interpreted them as raw violation counts. The total `hc` correctly showed the weighted sum, but the breakdown was confusing.

**Example**:
- Room exclusivity: 1043 raw violations, weight=3.0
- Old display: `hc8=3129` (user thinks: 3129 violations)
- New display: `hc8=1043` (shows raw count, contributes 3129 to weighted total)

**Fix**:
1. Changed display to show **raw violation counts** by dividing weighted values by constraint weights
2. Added diagnostic check to verify fitness matches detailed breakdown
3. Updated legend: "individual values = raw violations, hc/sc totals = weighted sums"

**Files**: `src/core/ga_scheduler.py` (display logic, lines ~1006-1038)

**Verification**: Run `uv run nsga --test` - individual constraint values now sum correctly when multiplied by weights.

**Detailed docs**: `docs/06-development/bugfixes/constraint-display-confusion.md`

---

## [2025-11-22] Windows Multiprocessing Handle Limit

**Severity**: Critical  
**Component**: Multiprocessing pools in `standard_run.py`, `parallel_executor.py`, `lns_operator.py`, `train_script.py`  
**Impact**: Complete crash on Windows with 32+ core CPUs (ValueError: need at most 63 handles)

**Problem**: Windows `WaitForMultipleObjects` has a hard limit of 63 handles. With auto-detection using `CPU cores * 2`, a 32-core system creates 64 workers + 2 management handles = 66 total, exceeding the limit.

**Root Cause**: Overly aggressive worker count calculation:
```python
# BROKEN:
num_workers = multiprocessing.cpu_count() * 2  # 32 * 2 = 64 workers + 2 handles = 66!

# FIXED:
num_workers = multiprocessing.cpu_count()  # 32 workers (well under 63 limit)
```

**Fix**:
1. Changed auto-detection from `CPU * 2` to `CPU count` (32 workers instead of 64)
2. Updated all multiprocessing components: `standard_run.py`, `parallel_executor.py`, `lns_operator.py`, `train_script.py`
3. Updated `configs/base.yaml` documentation

**Impact**: Engine runs successfully on Windows with any core count. Still excellent parallelism with 32 workers.

**Verification**: Run on Windows: `uv run nsga --test` (should show "32 workers" instead of "64 workers")

---

## [2025-11-22] GPU Tuple Corruption - DEAP Operator Tuple Reassignment

**Severity**: Critical  
**Component**: `src/core/ga_scheduler.py` (lines 77-130)  
**Impact**: GPU acceleration completely broken (10-50x speedup lost)

**Problem**: GPU batch evaluation failed with `'tuple' object has no attribute 'course_id'`. DEAP operators return tuples `(ind1, ind2)` for crossover and `(individual,)` for mutation, but `_parallel_crossover` and `_parallel_mutation` were NOT reassigning these tuples back to the offspring list, assuming in-place modification only.

**Root Cause**: Missing tuple unpacking:
```python
# BROKEN:
result = toolbox.mate(offspring[i], offspring[i + 1])
# No reassignment - tuple corruption leaks into individual

# FIXED:
result = toolbox.mate(offspring[i], offspring[i + 1])
offspring[i], offspring[i + 1] = result  # CRITICAL: Unpack and reassign
```

**Fix**:
1. `_parallel_crossover`: Explicitly unpack `offspring[i], offspring[i + 1] = result`
2. `_parallel_mutation`: Change to index-based loop and unpack `offspring[i] = result[0]`
3. Enhanced GPU evaluator error logging for diagnosis

**Impact**: GPU evaluation now works correctly, restoring 10-50x speedup. Combined with pymoo optimization, achieves 139x total speedup (50s → 0.36s per generation).

**Verification**: `uv run python scripts/diagnostics/test_deap_operators.py`

**Details**: `docs/06-development/bugfixes/gpu-tuple-corruption-fix.md`

---

## [2025-11-22] Mutation Operator Violating Course Completeness

**Severity**: Critical  
**Component**: `src/ga/operators/mutation.py`  
**Constraint**: `hc8` (course_completeness)

**Problem**: `mutate_time_quanta()` fallback used `min(num_quanta, len(available_quanta))` which could return fewer quanta than required. Combined with `quanta_list_to_contiguous()` using list length as `num_quanta`, this permanently corrupted gene duration.

**Root Cause**: 
```python
# OLD (BUGGY):
return random.sample(available_quanta, min(num_quanta, len(available_quanta)))
# Could return 2 quanta when course needs 5!
```

**Fix**: 
1. Check `len(available_quanta) < num_quanta` before mutation
2. If insufficient, keep original time slots (don't mutate)
3. Fallback always returns EXACTLY `num_quanta` items (no `min()`)

**Impact**: Course completeness violations (`hc8`) should now be 0 throughout evolution (structural invariant preserved).

**Constraint Mapping**: hc1=student_group_exclusivity, hc2=instructor_exclusivity, hc3=instructor_qualifications, hc4=room_suitability, hc5=instructor_time_availability, hc6=room_time_availability, hc7=course_completeness, hc8=room_exclusivity

**Details**: `docs/06-development/bugfixes/mutation-quanta-count-preservation.md`

---

## [2025-11-21] Repair Operators Using Deprecated SessionGene API

**Severity**: Critical  
**Component**: `src/ga/operators/repair.py`  
**Architecture**: SessionGene Nov 2025 migration

**Problem**: Entire repair system (2537 lines, 19+ operators) used OLD `gene.quanta` list API from before Nov 2025 architecture migration to contiguous representation (`start_quanta + num_quanta`).

**Fix**: Complete rewrite of repair.py (2537→370 lines):
- Updated to use `start_quanta + num_quanta` API
- Removed `repair_incomplete_or_extra_sessions` (unnecessary with correct init/mutation)
- Streamlined to 2 core repairs: instructor availability, group overlaps

**Details**: `docs/06-development/bugfixes/repair-operator-architecture-mismatch.md`

---

## Template for Future Entries

```markdown
## [YYYY-MM-DD] Brief Title (3-7 words)

**Severity**: Critical/High/Medium/Low  
**Component**: File path  
**Related**: Constraint/Feature

**Problem**: One-sentence description of bug behavior

**Root Cause**: Code snippet or explanation

**Fix**: Key changes made

**Impact**: Expected behavior after fix

**Details**: Link to detailed bugfix doc
```
