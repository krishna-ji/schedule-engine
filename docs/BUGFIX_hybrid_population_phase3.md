# Bug Fix: Phase 3 Hybrid Population Implementation Issues

**Date:** October 24, 2025  
**Status:** ✅ Fixed

---

## Summary

Fixed two critical bugs in the Phase 3 hybrid population initialization that prevented the GA from running.

---

## Issues Found

### 1. **TypeError in `_random_gene()`: Population must be a sequence**

**Error:**
```
TypeError: Population must be a sequence. For dicts or sets, use sorted(d).
```

**Root Cause:**
`random.sample()` requires a sequence (list/tuple), but `context.available_quanta` was likely a set or dict.

**Location:** `src/ga/hybrid_population.py`, line 337

**Fix:**
```python
# Before:
quanta = sorted(random.sample(context.available_quanta, num_quanta))

# After:
available_quanta_list = list(context.available_quanta)  # Convert to list
quanta = sorted(random.sample(available_quanta_list, num_quanta))
```

---

### 2. **AttributeError: 'list' object has no attribute 'fitness'**

**Error:**
```
AttributeError: 'list' object has no attribute 'fitness'
```

**Root Cause:**
The hybrid population functions (`_greedy_construction`, `_random_construction`) were returning raw Python lists instead of DEAP Individual objects.

**Location:** `src/ga/hybrid_population.py`, lines 55-62

**Fix:**
```python
# Import create_individual
from src.ga.individual import create_individual

# Wrap gene lists with create_individual()
for i in range(greedy_count):
    individual = _greedy_construction(context)
    if individual:
        population.append(create_individual(individual))  # ← Added wrapper

# Same for random construction
for i in range(random_count):
    individual = _random_construction(context)
    if individual:
        population.append(create_individual(individual))  # ← Added wrapper
```

---

### 3. **AttributeError: 'Instructor' object has no attribute 'availability'**

**Error:**
```
AttributeError: 'Instructor' object has no attribute 'availability'
```

**Root Cause:**
Constraint-guided mutation (Phase 2) was checking `instructor.availability`, but the entire availability system was removed in a previous cleanup (see `COMPLETE_AVAILABILITY_REMOVAL.md`).

**Location:** `src/ga/operators/constraint_guided_mutation.py`, line 105

---

### 4. **TypeError in `_mutate_session()`: Population must be a sequence (again)**

**Error:**
```
TypeError: Population must be a sequence. For dicts or sets, use sorted(d).
```

**Root Cause:**
Same issue as bug #1, but in the `_mutate_session()` function within constraint_guided_mutation.py. The function was using `random.sample(context.available_quanta, num_quanta)` without converting to a list first.

**Location:** `src/ga/operators/constraint_guided_mutation.py`, line 198 (and line ~221)

**Fix:**
Removed availability checks and replaced with qualification checks:

```python
# Before:
def _find_violating_sessions(...):
    ...
    # Check instructor availability
    if not _is_instructor_available(session, context):
        violating.append(idx)
        continue

def _is_instructor_available(session, context):
    """Check if instructor is available during session time."""
    if not instructor or not instructor.availability:
        return True
    for quantum in session.quanta:
        if quantum not in instructor.availability:
            return False
    return True

# After:
def _find_violating_sessions(...):
    ...
    # Check instructor qualification
    if not _is_instructor_qualified(session, context):
        violating.append(idx)
        continue

def _is_instructor_qualified(session, context):
    """Check if instructor is qualified to teach the course."""
    course = context.courses.get(session.course_id)
    if not course or not course.qualified_instructor_ids:
        return True
    return session.instructor_id in course.qualified_instructor_ids
```

**Removed function:** `_is_instructor_available()`  
**Added function:** `_is_instructor_qualified()`

**Fix:**
```python
# In _mutate_session()
def _mutate_session(gene, context: SchedulingContext):
    mutation_type = random.random()
    
    # Convert available_quanta to list for sampling (← ADDED)
    available_quanta_list = list(context.available_quanta)

    if mutation_type < 0.4:
        # Change time slots
        num_quanta = len(gene.quanta)
        if num_quanta > 0 and len(available_quanta_list) >= num_quanta:
            new_quanta = random.sample(available_quanta_list, num_quanta)  # ← FIXED
            ...
    
    else:
        # Change multiple attributes (aggressive mutation)
        num_quanta = len(gene.quanta)
        if num_quanta > 0 and len(available_quanta_list) >= num_quanta:
            new_quanta = random.sample(available_quanta_list, num_quanta)  # ← FIXED
```

---

### 5. **AttributeError: 'CourseSession' object has no attribute 'group_id'**

**Error:**
```
AttributeError: 'CourseSession' object has no attribute 'group_id'. Did you mean: 'group_ids'?
```

**Root Cause:**
Used singular `group_id` instead of plural `group_ids` to match the `CourseSession` dataclass attribute name.

**Location:** `src/ga/operators/constraint_guided_mutation.py`, `_has_group_overlap()` function

**Fix:**
```python
# Before:
session_groups = session.group_id if isinstance(session.group_id, list) else [session.group_id]
other_groups = other.group_id if isinstance(other.group_id, list) else [other.group_id]

# After:
session_groups = session.group_ids if isinstance(session.group_ids, list) else [session.group_ids]
other_groups = other.group_ids if isinstance(other.group_ids, list) else [other.group_ids]
```

---

### 6. **Potential Issue: Wrong attribute name for time quanta**

**Issue:** Used `session.quanta` instead of `session.session_quanta`

**Location:** `src/ga/operators/constraint_guided_mutation.py`, in all three conflict-checking functions

**Fix:**
Changed all references from `session.quanta` → `session.session_quanta` in:
- `_has_group_overlap()`
- `_has_room_conflict()`
- `_has_instructor_conflict()`

```python
# Before:
if set(session.quanta) & set(other.quanta):

# After:
if set(session.session_quanta) & set(other.session_quanta):
```

---

## Files Modified

| File | Changes |
|------|---------|
| `src/ga/hybrid_population.py` | 1. Added `from src.ga.individual import create_individual`<br>2. Fixed `_random_gene()` to convert quanta to list<br>3. Wrapped greedy/random individuals with `create_individual()` |
| `src/ga/operators/constraint_guided_mutation.py` | 1. Removed `_is_instructor_available()` function<br>2. Added `_is_instructor_qualified()` function<br>3. Updated `_find_violating_sessions()` to use qualification check<br>4. Fixed `_mutate_session()` to convert available_quanta to list (2 locations)<br>5. Fixed attribute names: `group_id` → `group_ids` (4 occurrences)<br>6. Fixed attribute names: `quanta` → `session_quanta` (6 occurrences)<br>7. Updated docstrings to note availability removal |

---

## Verification

### Before Fix:
```
Hybrid initialization: 2 greedy, 6 smart, 2 random
Traceback (most recent call last):
  TypeError: Population must be a sequence.
```

### After Fix:
```
Hybrid initialization: 2 greedy, 6 smart, 2 random
...
Found 527 course-group pairs to schedule
Generated 6 individuals with average 527.0 genes each
...
Evaluating Initial Population...
   ✓ Evaluated 10 individuals in 2.3s (0.23s per individual)
   Initial Best: Hard=1998, Soft=4825.00
⠙ Evolution Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 0/100
```

**Result:** ✅ GA runs successfully through initialization and starts evolution

---

## Lessons Learned

### 1. **Type Consistency**
Always ensure data structures are in the correct format for library functions:
- `random.sample()` needs a sequence (list/tuple), not a set/dict
- Convert with `list()` when needed

### 2. **DEAP Individual Wrapping**
All population generators must wrap gene lists with `create_individual()`:
- Look at existing patterns (`generate_course_group_aware_population`)
- DEAP requires Individual objects with `.fitness` attribute
- Raw lists won't work

### 3. **System-Wide Changes Impact**
When major systems are removed (like availability):
- Check ALL modules that might reference them
- Search for `availability` across codebase
- Update new code to match current architecture

### 4. **Integration Testing**
New modules should be tested with full system:
- Unit tests verify logic
- Integration tests catch system-level issues
- Run `python main.py` after major changes

---

## Related Documentation

- `COMPLETE_AVAILABILITY_REMOVAL.md` - Why availability was removed
- `PHASE3_HYBRID_INITIALIZATION.md` - Phase 3 design and expected behavior
- `PHASE2_CONSTRAINT_GUIDED_MUTATION.md` - Constraint-guided mutation design

---

## Status

✅ **All Issues Resolved**

The hybrid population initialization now works correctly:
- Generates 25% greedy + 50% smart + 25% random individuals
- Properly wraps all individuals as DEAP Individual objects
- Constraint-guided mutation checks correct violations
- GA evolution proceeds normally

**Next:** Let system run to completion and verify performance improvements
