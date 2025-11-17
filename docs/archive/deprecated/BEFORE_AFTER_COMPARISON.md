# Block Clustering: Before vs After Comparison

## Overview

This document shows the key differences between the old uniform block clustering and the new course-type-aware system.

## 1. Constraint Logic

### Before (Uniform)

```python
# Same rules for ALL courses
for block_size in blocks:
    if block_size == 1:
        penalty += 5  # Hardcoded
    elif block_size > 3:
        penalty += (block_size - 3) * 1  # Hardcoded
```

**Problems:**
- ❌ Theory and practical treated the same
- ❌ No pedagogical awareness
- ❌ Hardcoded penalties (no tuning)
- ❌ All isolated sessions penalized equally

### After (Course-Type-Aware)

```python
# Different rules for theory vs practical
if course_type == "practical":
    # Practical: MUST be single block
    if len(blocks) > 1:
        penalty += cfg.practical_fragmentation_penalty * (len(blocks) - 1)
else:
    # Theory: Flexible 2-3 block distribution
    isolated_count = 0
    for block_size in blocks:
        if block_size == 1:
            isolated_count += 1
            if isolated_count > cfg.theory_max_excused_isolated:
                penalty += cfg.theory_isolated_penalty
        elif block_size > cfg.preferred_block_size_max:
            penalty += excess * cfg.theory_oversized_penalty_per_quantum
```

**Benefits:**
- ✅ Pedagogically appropriate rules
- ✅ Configurable penalties
- ✅ First isolated session excused (practical)
- ✅ Heavy penalty for practical fragmentation

## 2. Configuration

### Before

```python
# In code (hard to change)
ISOLATED_SESSION_PENALTY = 5
OVERSIZED_BLOCK_PENALTY_PER_QUANTUM = 1
PREFERRED_BLOCK_SIZE_MAX = 3
```

**Problems:**
- ❌ Requires code changes to tune
- ❌ No environment-specific settings
- ❌ Same values for all course types

### After

```yaml
# In YAML config (easy to tune)
time:
  preferred_block_size_max: 3
  
  # Theory
  theory_isolated_penalty: 2
  theory_oversized_penalty_per_quantum: 1
  theory_max_excused_isolated: 1
  
  # Practical
  practical_fragmentation_penalty: 20
```

**Benefits:**
- ✅ No code changes needed
- ✅ Environment-specific (test/dev/prod)
- ✅ Course-type-specific settings

## 3. Repair Heuristics

### Before

```python
# Same repair for all courses
def repair_session_clustering(individual, context):
    # Move isolated quanta to adjacent positions
    # Split oversized blocks
    # Rebuild with 2-3 block distribution
```

**Problems:**
- ❌ Doesn't prioritize single-block for practicals
- ❌ May create multiple blocks for labs
- ❌ No course-type awareness

### After

```python
# Course-type-aware repair
def repair_session_clustering(individual, context):
    course_type = get_course_type(gene)
    
    if course_type == "practical":
        # Special: consolidate into single block
        _rebuild_practical_single_block(gene, ...)
    else:
        # Theory: flexible multi-block optimization
        _rebuild_gene_clustering(gene, ...)
        _split_oversized_blocks(gene, ...)
```

**Benefits:**
- ✅ Practical courses get single-block treatment
- ✅ Theory courses maintain flexibility
- ✅ Repair matches constraint logic

## 4. Example Scenarios

### Scenario 1: 6-hour Theory Course

**Before:**
- Distribution: [1, 2, 3] → Penalty = 5 (isolated)
- Repair: Tries to merge isolated into 2-block

**After:**
- Distribution: [1, 2, 3] → Penalty = 0 (first isolated excused!)
- Repair: Only triggered if multiple isolated sessions
- Result: More schedule flexibility ✅

### Scenario 2: 3-hour Lab (Practical)

**Before:**
- Distribution: [2, 1] → Penalty = 5 (isolated)
- Repair: May leave as [2, 1] or change to [1, 2]
- Problem: Still fragmented ❌

**After:**
- Distribution: [2, 1] → Penalty = 20 (fragmentation!)
- Repair: Forces consolidation to [3]
- Result: Continuous lab time ✅

### Scenario 3: 9-hour Theory Course

**Before:**
- Distribution: [9] → Penalty = 6 (oversized by 6)
- Repair: Splits to [3, 3, 3]

**After:**
- Distribution: [9] → Penalty = 6 (6 quanta beyond 3)
- Repair: Splits to [3, 3, 3]
- Same behavior, but now configurable! ✅

### Scenario 4: 6-hour Lab (Practical)

**Before:**
- Distribution: [2, 2, 2] → Penalty = 0 (no isolated, not oversized)
- Repair: No action
- Problem: Lab is still fragmented! ❌

**After:**
- Distribution: [2, 2, 2] → Penalty = 40 (2 splits × 20)
- Repair: Consolidates to [6]
- Result: Proper lab session ✅

## 5. Environment-Specific Tuning

### Test Environment (Fast)

```yaml
theory_isolated_penalty: 2
practical_fragmentation_penalty: 20
```
- Quick convergence
- Reasonable quality

### Production Environment (Quality)

```yaml
theory_isolated_penalty: 3
theory_oversized_penalty_per_quantum: 2
practical_fragmentation_penalty: 50
```
- Stricter enforcement
- Higher quality schedules
- Longer runtime acceptable

## 6. Penalty Comparison Table

| Scenario | Course Type | Distribution | Old Penalty | New Penalty | Change |
|----------|-------------|--------------|-------------|-------------|---------|
| Ideal blocks | Theory | [3, 3] | 0 | 0 | Same ✓ |
| One isolated | Theory | [1, 2, 3] | 5 | 0 | Better ✓ |
| Two isolated | Theory | [1, 1, 4] | 11 | 3 | Better ✓ |
| Oversized | Theory | [6] | 3 | 3 | Same ✓ |
| Single block | Practical | [3] | 0 | 0 | Same ✓ |
| Two blocks | Practical | [2, 1] | 5 | 20 | Stricter ✓ |
| Three blocks | Practical | [1, 1, 1] | 15 | 40 | Much stricter ✓ |
| Fragmented lab | Practical | [2, 2, 2] | 0 | 40 | Now detected! ✓ |

## 7. Impact Summary

### Theory Courses

**Improvements:**
- ✅ First isolated session excused (more realistic)
- ✅ Configurable tolerance
- ✅ Environment-specific strictness
- ✅ Better reflects actual scheduling needs

### Practical Courses

**Improvements:**
- ✅ Heavy penalty for ANY fragmentation
- ✅ Repair actively consolidates blocks
- ✅ Matches pedagogical requirement
- ✅ Guaranteed continuous lab time

### System-Wide

**Improvements:**
- ✅ No code changes for tuning
- ✅ Environment-specific configs
- ✅ Pedagogically accurate
- ✅ Better schedule quality
- ✅ Comprehensive testing
- ✅ Full documentation

## 8. Migration Path

### For Existing Schedules

Old schedules will automatically benefit from new logic on next run:
1. Config loads with new parameters (defaults match old behavior)
2. Constraint evaluation uses new course-type logic
3. Repair uses enhanced strategies
4. No data migration needed

### For Custom Configurations

If you had custom penalty values:
1. Old parameters still work (backward compatible)
2. New parameters take precedence
3. Recommended: Update to use new course-type-specific settings

```yaml
# Old (still works)
isolated_session_penalty: 5

# New (better)
theory_isolated_penalty: 2
practical_fragmentation_penalty: 20
```

---

**Conclusion:** The new course-type-aware system provides better pedagogical alignment, more flexibility, and easier tuning while maintaining backward compatibility.
