# 🔧 IMMEDIATE ACTION PLAN
## Quick Fix Guide - Priority Ordered

---

## 🚨 **STOP! Read This First**

Your GA produced **10,064 hard violations** after 150 generations. This is **completely unusable**. 

**The good news**: The problems are **fixable**. Follow this guide step-by-step.

---

## 🔥 **CRITICAL BUG: Repair System Not Running**

### **Problem**
```
repairs_total: 0 (EVERY GENERATION!)
```

Despite having repair enabled in your config, **zero repairs were executed** during the entire run. This is your **#1 problem**.

### **Fix Steps**

#### Step 1: Verify Repair Integration

Check your `src/core/ga_scheduler.py`. Search for where repair should be called. Look for patterns like:
- `repair_individual()`
- `apply_repair()`
- Memetic local search

**If you don't find any repair calls in your GA evolution loop, that's the bug!**

#### Step 2: Check Repair Module

```bash
# From project root
python -c "from src.ga.operators.repair import repair_individual; print('Repair module loads OK')"
```

If this fails, check:
- `src/ga/operators/repair.py` exists
- `repair_registry.py` is properly configured
- Import paths are correct

#### Step 3: Add Debug Logging

Add this to your repair function:
```python
def repair_individual(individual, context, **kwargs):
    print(f"[DEBUG] Repair called for individual with {len(individual)} genes")
    # ... rest of repair logic
```

Run test config and watch for debug messages.

#### Step 4: Enable Memetic Repair

Your config has `memetic_mode: true` but repairs aren't running. Check if memetic repair is actually implemented in your GA loop.

**Expected behavior**: After each generation, elite individuals should undergo local search/repair.

---

## 🔴 **PRIORITY 1: Data Quality Check**

### **Problem**: Duplicate Course Enrollments

The violation report shows groups scheduled for the **same course multiple times simultaneously**:

```
Group BAR5A @ Sunday 08:00:
- AR604 @ Lecture Room [B-207]
- AR604 @ Lecture Room [E-302]  ← DUPLICATE!
- CE607 @ Mechanical Workshop
- CE607 @ Lecture Room [B-103]   ← DUPLICATE!
```

### **Fix Steps**

#### Create Data Diagnostic Script

```python
# scripts/check_data_quality.py
import json
from collections import Counter

def check_duplicate_enrollments(groups_file="data/Groups.json"):
    with open(groups_file, 'r') as f:
        data = json.load(f)
    
    issues = []
    for group in data:
        group_id = group.get('group_id') or group.get('id')
        enrolled = group.get('enrolled_courses', [])
        
        # Check for duplicates
        duplicates = [course for course, count in Counter(enrolled).items() if count > 1]
        
        if duplicates:
            issues.append({
                'group': group_id,
                'duplicates': duplicates,
                'count': {course: Counter(enrolled)[course] for course in duplicates}
            })
    
    if issues:
        print(f"\n❌ Found {len(issues)} groups with duplicate enrollments:\n")
        for issue in issues:
            print(f"  Group {issue['group']}:")
            for course, count in issue['count'].items():
                print(f"    - {course} appears {count} times")
    else:
        print("\n✅ No duplicate enrollments found!")
    
    return issues

if __name__ == "__main__":
    check_duplicate_enrollments()
```

#### Run the Check

```bash
python scripts/check_data_quality.py
```

#### Fix Any Issues

If duplicates are found, **manually edit `data/Groups.json`** to remove them, OR clarify if they're legitimate (e.g., separate theory and lab sections).

---

## 🟠 **PRIORITY 2: Quick Config Fixes**

### **Fix 1: Reduce Block Clustering Weight**

**File**: `configs/dev.yaml`

```yaml
hard_constraints:
  session_block_clustering_penalty:
    enabled: true
    weight: 1.0  # Changed from 2.0
```

**Why**: This constraint accounts for 79% of all violations. Reducing its weight lets the GA focus on more critical constraints first.

---

### **Fix 2: Reduce Mutation Rate**

**File**: `configs/dev.yaml`

```yaml
ga:
  ngen: 100
  pop_size: 100
  cxpb: 0.85
  mutpb: 0.15  # Changed from 0.25
```

**Why**: High mutation (25%) is too disruptive—it breaks clustered time blocks and introduces violations faster than repair can fix them.

---

### **Fix 3: Increase Elite Preservation**

**File**: `configs/common.yaml`

```yaml
ga:
  elite_preservation: true
  elite_size: 0.10  # Changed from 0.05
```

**Why**: Protects best solutions from being mutated away. 10% elites (10 individuals in pop of 100) ensures good genes survive longer.

---

### **Fix 4: Enable Population Restart**

**File**: `configs/dev.yaml`

```yaml
enhancements:
  population_restart:
    enabled: true  # Changed from false
    trigger_stagnation_gens: 10
    restart_percentage: 0.3
    min_interval_gens: 50
```

**Why**: Your diversity collapsed from 0.69 to 0.15 (78% loss). Population restart injects fresh solutions when stuck.

---

### **Fix 5: Increase Repair Iterations**

**File**: `configs/dev.yaml`

```yaml
repair:
  max_iterations: 15  # Changed from 7
  memetic_mode: true
  selective_mode: true
```

**Why**: Once repair is working, more iterations = better violation reduction.

---

## ✅ **TESTING PROTOCOL**

After implementing fixes, follow this sequence:

### **Test 1: Smoke Test (5 minutes)**

```bash
python main.py --env test
```

**Check for**:
- ✅ No crashes
- ✅ `repairs_total > 0` in logger
- ✅ Hard violations decreasing

---

### **Test 2: Dev Run (35 minutes)**

```bash
python main.py --env dev
```

**Success criteria**:
- ✅ Hard violations < 5,000 by gen 100
- ✅ Diversity stays > 0.30
- ✅ Repairs executed > 1,000 total
- ✅ No stagnation > 10 generations

---

### **Test 3: Extended Run (If Test 2 passes)**

```yaml
# configs/dev.yaml - temporarily extend
ga:
  ngen: 200  # From 100
```

```bash
python main.py --env dev
```

**Target**: Hard violations < 1,000 (ideally zero!)

---

## 📊 **How to Read Results**

### **Check Logger File**

```bash
# After run completes
tail -n 50 output/evaluation_<timestamp>/logger.txt
```

Look for:
```
Gen 100: Hard=XXXX, Soft=YYYY, Repairs=ZZZZ
```

### **Check Violation Report**

```bash
head -n 100 output/evaluation_<timestamp>/violation_report.txt
```

Count violations by type. Prioritize fixing the most frequent ones.

### **Check CSV Data**

```bash
# View final generation
tail -n 5 output/evaluation_<timestamp>/CSVs/hard_constraints_all.csv
```

---

## 🎯 **Success Metrics**

| Metric | Current | Target (Good) | Excellent |
|--------|---------|---------------|-----------|
| Hard Violations | 10,064 | < 1,000 | **0** |
| Repairs Executed | **0 (BUG)** | > 1,000 | > 5,000 |
| Diversity @ Gen 50 | 0.32 | > 0.35 | > 0.45 |
| Block Clustering | 7,928 | < 2,000 | < 500 |
| Group Overlaps | 996 | < 100 | **0** |

---

## 🔍 **Debugging Checklist**

If issues persist after fixes:

### **Problem: Still No Repairs**

- [ ] Check repair imports in `ga_scheduler.py`
- [ ] Verify repair functions are registered in `repair_registry.py`
- [ ] Add print statements inside repair functions
- [ ] Check if `apply_after_mutation/crossover` flags are being read

### **Problem: High Block Clustering**

- [ ] Reduce weight further (try 0.5)
- [ ] Check if time slots are actually available for 2-3 hour blocks
- [ ] Verify `max_session_coalescence` is set correctly
- [ ] Consider temporarily disabling this constraint

### **Problem: Group Overlaps Won't Decrease**

- [ ] Check for duplicate enrollments (run data quality script)
- [ ] Verify group availability windows aren't too restrictive
- [ ] Check crossover implementation (may be breaking course-group pairing)
- [ ] Add repair heuristic specifically for group overlaps

### **Problem: Diversity Still Collapsing**

- [ ] Increase population size (try 150)
- [ ] Enable fitness sharing or niching
- [ ] Reduce crossover probability (try 0.75)
- [ ] Add periodic random individual injection

---

## 📞 **Need More Help?**

### **Share These Files**:
1. `output/evaluation_<timestamp>/logger.txt`
2. `output/evaluation_<timestamp>/violation_report.txt` (first 200 lines)
3. `output/evaluation_<timestamp>/CSVs/hard_constraints_all.csv`
4. Your modified config files

### **Report Format**:
```
Configuration used: [dev/prod]
Fixes applied: [list]
Results:
  - Hard violations: XXXX
  - Repairs executed: YYYY
  - Stuck at generation: ZZ
Specific issue: [describe]
```

---

## 🚀 **Timeline Estimate**

| Task | Time | Status |
|------|------|--------|
| Fix repair system | 2-4 hours | 🔴 CRITICAL |
| Data quality check | 30 min | 🔴 CRITICAL |
| Apply config fixes | 15 min | 🟠 EASY |
| Run test + dev | 45 min | ⚪ WAITING |
| Analyze results | 30 min | ⚪ WAITING |
| Iterate if needed | 2-3 hours | ⚪ WAITING |
| **TOTAL** | **4-8 hours** | **TODAY** |

---

## ✅ **Quick Start (Right Now!)**

```bash
# 1. Create diagnostic script
mkdir -p scripts
cat > scripts/check_data_quality.py << 'EOF'
# [paste script from above]
EOF

# 2. Check for data issues
python scripts/check_data_quality.py

# 3. Apply config fixes
# Edit configs/dev.yaml with changes above

# 4. Run quick test
python main.py --env test

# 5. Check if repairs are running
grep "repairs_total" output/evaluation_*/logger.txt

# 6. If repairs still zero, DEBUG REPAIR SYSTEM FIRST
# Otherwise proceed to full dev run
```

---

**Good luck! The fixes are straightforward—you'll have this working soon! 🚀**