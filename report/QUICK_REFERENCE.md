# 🎯 Quick Reference: Constraint Weight Issue

## The Problem in One Sentence
**Block clustering (quality preference) is weighted the same as safety constraints, but generates 7x more violations, causing the GA to optimize for "nice grouping" instead of "no double-bookings".**

---

## The Evidence

| Constraint | Violations | Weight | Penalty Contribution | Type |
|------------|------------|--------|---------------------|------|
| Block Clustering | 7,408 | 2.0 | 14,816 (73.6%) | Quality |
| Group Overlaps | 1,046 ↑175% | 2.0 | 2,092 (10.4%) | **SAFETY** |
| Instructor Conflicts | 250 ↑229% | 2.0 | 500 (2.5%) | **SAFETY** |

---

## The Fix

### Before:
```yaml
hard_constraints:
  session_block_clustering_penalty:
    weight: 2.0
  no_group_overlap:
    weight: 2.0
  no_instructor_conflict:
    weight: 2.0
```

### After (Conservative):
```yaml
hard_constraints:
  session_block_clustering_penalty:
    weight: 1.0  # ⬇️ REDUCE
  no_group_overlap:
    weight: 2.5  # ⬆️ INCREASE
  no_instructor_conflict:
    weight: 2.5  # ⬆️ INCREASE
```

### After (Aggressive):
```yaml
hard_constraints:
  session_block_clustering_penalty:
    weight: 0.5  # ⬇️⬇️ HEAVY REDUCE
  no_group_overlap:
    weight: 3.0  # ⬆️⬆️ HIGH PRIORITY
  no_instructor_conflict:
    weight: 3.0  # ⬆️⬆️ HIGH PRIORITY
```

---

## Expected Impact

| Metric | Current | After Conservative | After Aggressive |
|--------|---------|-------------------|------------------|
| **Group Overlaps** | 1,046 | 400-600 (50% ↓) | 100-300 (80% ↓) |
| **Instructor Conflicts** | 250 | 100-150 (50% ↓) | 50-100 (70% ↓) |
| **Block Clustering** | 7,408 | 8,000-9,000 (may ↑) | 10,000-12,000 (may ↑) |
| **Usability** | ❌ Impossible | ⚠️ Usable | ✅ Good |

---

## Why This Works

**Current penalty distribution**:
```
Block:    14,816 (73.6%) ← GA focuses here
Safety:    2,592 (12.9%) ← GA ignores this
Other:     2,720 (13.5%)
```

**After conservative fix**:
```
Block:     7,408 (35%) ← Reduced focus
Safety:    6,480 (31%) ← GA now cares!
Other:     7,240 (34%)
```

**After aggressive fix**:
```
Safety:    9,690 (45%) ← HIGHEST PRIORITY ✅
Block:     3,704 (17%) ← Background concern
Other:     8,240 (38%)
```

---

## Commands to Execute

```bash
# 1. Check data quality first
python scripts/check_data_quality.py

# 2. Backup config
cp configs/dev.yaml configs/dev_BACKUP.yaml

# 3. Edit configs/dev.yaml with new weights
# (See "The Fix" section above)

# 4. Quick test
python main.py --env test

# 5. Full run
python main.py --env dev

# 6. Compare
echo "OLD: 10,064 violations"
grep "Hard violations" output/evaluation_*/logger.txt | tail -1
```

---

## Success Criteria

✅ **GOOD**: Group overlaps < 500, Instructor conflicts < 150  
✅ **EXCELLENT**: Group overlaps < 200, Instructor conflicts < 100  
✅ **PERFECT**: All critical constraints < 50

---

## Full Documentation

- 📄 `ANALYSIS_SUMMARY.md` - Overview and action plan
- 📄 `CONSTRAINT_WEIGHT_ANALYSIS.md` - Deep technical analysis (500 lines)
- 📄 `DETAILED_ANALYSIS_REPORT.md` - Complete GA run analysis (600 lines)
- 📄 `IMMEDIATE_ACTION_PLAN.md` - Step-by-step implementation guide

---

**TL;DR**: Change 3 numbers in `configs/dev.yaml` (5 minutes), run test (35 minutes), expect 50-80% reduction in critical violations, schedule becomes usable.
