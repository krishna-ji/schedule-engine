# 🔬 Schedule Engine - Detailed Analysis Report
## Evaluation Run: 2025-10-27 08:34:47

---

## 📊 Executive Summary

**Run Configuration**: Dev environment (100 generations → Extended to 150)  
**Total Runtime**: 33.61 minutes (2016.42 seconds)  
**Final Best Solution**: **10,064 hard violations** | **1,221 soft penalties**

### 🚨 **Critical Finding**
**The GA is NOT converging to a feasible solution.** After 150 generations, the schedule still has **10,064 hard constraint violations**, indicating **severe structural problems** that prevent the algorithm from finding valid schedules.

---

## 🎯 Key Metrics

| Metric | Initial | Final | Improvement | Status |
|--------|---------|-------|-------------|--------|
| **Hard Violations** | 12,662 | 10,064 | ↓ 20.5% | ❌ **UNACCEPTABLE** |
| **Soft Penalties** | 1,665 | 1,221 | ↓ 26.7% | ⚠️ Secondary concern |
| **Diversity** | 0.6934 | 0.1525 | ↓ 78.0% | ❌ **COLLAPSED** |
| **Avg Time/Gen** | - | 8.766s | - | ✅ Acceptable |

---

## 📈 Convergence Analysis

### 1. **Overall Trend: Slow Degrading Plateau**

```
Generation Milestones:
┌─────────┬──────────┬───────────┬────────────┐
│ Gen     │ Hard     │ Soft      │ Event      │
├─────────┼──────────┼───────────┼────────────┤
│ INIT    │ 12,662   │ 1,665     │ Baseline   │
│ 50      │ 11,380   │ 1,376     │ -10%       │
│ 100     │ 10,786   │ 1,397     │ -15%       │
│ 150     │ 10,064   │ 1,221     │ -20% FINAL │
└─────────┴──────────┴───────────┴────────────┘
```

**Analysis**: The GA shows **marginal improvement** but **never reaches feasibility**. The best solution still has over 10,000 violations—completely unusable.

---

### 2. **Stagnation Windows**

| Generations | Hard Violations | Status | Duration |
|-------------|----------------|--------|----------|
| 50-60 | 11,380 → 11,292 | Minimal progress | 10 gens |
| 66-72 | 11,242 (stuck) | **STAGNANT** | 7 gens |
| 85-90 | 10,992 (stuck) | **STAGNANT** | 6 gens |
| 107-113 | 10,544 (stuck) | **STAGNANT** | 7 gens |
| 137-148 | 10,236 (stuck) | **STAGNANT** | 12 gens |

**Critical**: Multiple prolonged stagnation periods indicate the GA is **trapped in local optima** and cannot escape.

---

### 3. **Diversity Collapse** ⚠️

```
Diversity Trajectory:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Gen   0: ████████████████████ 0.693 (69%)
Gen  25: ███████████████ 0.562 (56%)
Gen  50: ██████ 0.322 (32%)
Gen  75: ████ 0.236 (24%)
Gen 100: ███ 0.195 (19%)
Gen 125: ██ 0.184 (18%)
Gen 150: ██ 0.153 (15%) ← COLLAPSED
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Critical Issue**: Diversity dropped by **78%** from 0.693 to 0.153. This severe premature convergence indicates:
- Population became **genetically homogeneous**
- Search space **prematurely narrowed**
- Lost ability to explore alternative solutions

---

## 🔍 Constraint Breakdown

### **Hard Constraints Analysis**

#### Final Violations (Generation 150):

| Constraint | Violations | Weight | Weighted Penalty | % of Total |
|-----------|------------|--------|------------------|----------|
| **Session Block Clustering** | 7,928 | 2.0 | 15,856 | **78.8%** |
| **Group Overlap** | 996 | 2.0 | 1,992 | 9.9% |
| **Availability Violations** | 918 | 2.0 | 1,836 | 9.1% |
| **Room Type Mismatch** | 464 | 2.0 | 928 | 4.6% |
| **Instructor Conflict** | 248 | 2.0 | 496 | 2.5% |
| **Instructor Not Qualified** | 0 | 2.0 | 0 | 0.0% |
| **Incomplete/Extra Sessions** | 0 | 1.0 | 0 | 0.0% |
| **TOTAL** | **10,554** | - | **21,108** | **100%** |

---

### 🚨 **Root Cause #1: Session Block Clustering Penalty**

**This is the PRIMARY culprit—accounting for 79% of all violations!**

#### What is it?
The block clustering constraint ensures that theory courses are scheduled in contiguous blocks (2-3 hour chunks) rather than fragmented single hours. Practical courses should not be fragmented either.

#### Why is it failing?
```
Violation Progression:
Gen   0: 10,618 violations (Initial chaos)
Gen  50:  8,776 violations (-17%)
Gen 100:  8,210 violations (-23%)
Gen 150:  7,928 violations (-25%)
```

**Even after 150 generations, this constraint is barely improving!**

#### Likely Causes:
1. **Time Slot Scarcity**: Not enough consecutive time slots available
2. **Competing Constraints**: Trying to avoid group overlaps forces fragmentation
3. **Mutation Damage**: Random mutations break carefully clustered blocks
4. **Weak Repair**: Repair heuristics not targeting block reconstruction

---

### 🚨 **Root Cause #2: Group Overlap Violations**

**996 instances where multiple courses are scheduled for the same group at the same time.**

#### Pattern Analysis from Violation Report:

**Most Affected Groups** (Top 10):
1. **BCE7A-F** (6 groups): 3-8 overlapping sessions per time slot
2. **BCE4A-F** (6 groups): 2-7 overlapping sessions per time slot
3. **BCE5A-F** (6 groups): 3-7 overlapping sessions per time slot
4. **BCT5A-B**: 6-9 overlapping sessions
5. **BAR5A-B**: 7-8 overlapping sessions

#### Example Critical Case:
```
Group BAR5A @ Sunday 08:00 has 8 OVERLAPPING sessions:
- AR601 @ Lecture Room [B-308]
- AR605 @ Material Testing Laboratory
- AR602 @ Lecture Room [B-309]
- AR604 @ Lecture Room [B-207]
- CE607 @ Mechanical Workshop
- EE604 @ Drawing Room #2
- AR604 @ Lecture Room [E-302]  ← DUPLICATE COURSE!
- CE607 @ Lecture Room [B-103]  ← DUPLICATE COURSE!
```

**Critical Discovery**: Some courses appear **twice** for the same group (e.g., AR604, CE607), suggesting **data duplication** or **encoding errors**.

#### Why It's Happening:
1. **Data Quality Issue**: Groups may be enrolled in too many courses
2. **Insufficient Time Windows**: Competing courses fight for same slots
3. **Cross-Department Conflicts**: Shared service courses (CE607, EE604) create bottlenecks
4. **Weak Crossover**: Course-group pairing gets disrupted

---

### 🚨 **Root Cause #3: Availability Violations**

**918 violations where sessions are scheduled outside instructor/group/room availability windows.**

#### Why This Is Still High:
- **Initial Population**: 910 violations (already present!)
- **Final**: 918 violations (actually got WORSE)

This suggests:
1. **Availability data is too restrictive**
2. **Mutation introduces violations faster than repair fixes them**
3. **No pressure to satisfy availability early** (other constraints dominate)

---

### 🚨 **Root Cause #4: Room Type Mismatch**

**464 violations where courses are assigned to wrong room types.**

#### Examples:
- Theory course in Lab
- Lab course in Lecture room
- Drawing course in Seminar hall

**Why**: Random mutations assign any available room without type checking. Repair doesn't prioritize this.

---

### 🚨 **Root Cause #5: Instructor Conflicts**

**248 violations where one instructor is scheduled in multiple sessions simultaneously.**

Lower than group overlaps, but still significant. Shows instructors are being overbooked.

---

## 🛑 **What's NOT Working**

### 1. **Repair Heuristics: Zero Activity** ⚠️

```
repairs_total: 0 (every generation!)
repairs_memetic_count: 0
repairs_crossover_count: 0
repairs_mutation_count: 0
```

**CRITICAL BUG**: Despite `repair.enabled: true` and `memetic_mode: true`, **NO REPAIR OPERATIONS WERE EXECUTED** during the entire 150-generation run!

**This is a SEVERE configuration or code bug.** Repair is the primary mechanism for fixing violations, and it's completely inactive.

---

### 2. **Population Strategy: Inadequate**

Your config uses **hybrid population initialization**:
- 25% greedy
- 50% smart (constraint-aware)
- 25% random

But the initial population still started with **12,662 violations**—indicating even the "smart" initialization is producing terrible solutions.

---

### 3. **Crossover: Disrupting Course-Group Pairings**

Group overlap violations **increased** in early generations (380 → 666), suggesting crossover is breaking valid course-group assignments.

---

### 4. **Mutation: Too Disruptive**

With **mutpb = 0.25** (25% mutation rate), the algorithm is aggressively mutating individuals, which is:
- Breaking clustered time blocks
- Introducing availability violations
- Disrupting instructor assignments

---

### 5. **Selection Pressure: Insufficient**

The GA is using NSGA-II (multi-objective) with only **5% elite preservation**. This may not be enough to protect the best solutions from being mutated away.

---

## 💡 **Root Cause Summary**

### **Primary Culprits** (in order of impact):

1. **🔴 Block Clustering Penalty (79% of violations)**
   - Time slot fragmentation
   - Competing constraints prevent block formation
   - No effective repair mechanism

2. **🔴 Repair System Failure (0 repairs executed)**
   - Configuration bug or code issue
   - Critical mechanism completely inactive

3. **🟠 Group Overlap Violations (10% of violations)**
   - Data quality issues (duplicate enrollments)
   - Time window scarcity
   - Weak crossover protection

4. **🟠 Diversity Collapse (78% loss)**
   - Population became too homogeneous
   - Trapped in local optima
   - Cannot explore better solutions

5. **🟡 Availability & Room Type Violations**
   - Too restrictive constraints
   - Weak mutation/repair balance

---

## 🎯 **Actionable Recommendations**

### **🔥 CRITICAL - Must Fix Immediately**

#### 1. **Fix Repair System (HIGHEST PRIORITY)**

**Problem**: Zero repairs executed despite being enabled.

**Actions**:
```yaml
# Verify repair configuration
repair:
  enabled: true
  apply_after_mutation: true
  apply_after_crossover: true
  memetic_mode: true
  selective_mode: true
  max_iterations: 10  # Increase from 7
```

**Debug Steps**:
- Add logging inside repair functions to confirm they're being called
- Check if `repair_individual()` is properly integrated into GA loop
- Verify repair registry is loading functions correctly
- Test repair heuristics in isolation

---

#### 2. **Address Block Clustering Constraint**

**Option A**: Relax the constraint weight temporarily
```yaml
hard_constraints:
  session_block_clustering_penalty:
    enabled: true
    weight: 1.0  # Reduce from 2.0
```

**Option B**: Add block-aware initialization
- Generate initial population with pre-clustered time blocks
- Ensure 2-3 hour chunks are always contiguous

**Option C**: Add block-aware mutation
- When mutating time slots, keep sessions of same course together
- Swap entire blocks instead of individual quanta

---

#### 3. **Fix Data Quality Issues**

**Check for duplicate enrollments**:
```python
# Run this diagnostic on your Groups.json
for group in groups:
    enrolled_courses = group.enrolled_courses
    duplicates = [c for c in set(enrolled_courses) 
                  if enrolled_courses.count(c) > 1]
    if duplicates:
        print(f"Group {group.id} has duplicates: {duplicates}")
```

**Remove duplicate enrollments** or clarify if they're legitimate (e.g., multiple theory and lab sections).

---

### **⚠️ HIGH PRIORITY**

#### 4. **Increase Diversity Maintenance**

```yaml
enhancements:
  population_restart:
    enabled: true
    trigger_stagnation_gens: 10  # Restart if stuck for 10 gens
    restart_percentage: 0.3  # Replace 30% of population
```

**Add diversity-promoting mechanisms**:
- **Fitness sharing**: Penalize similar solutions
- **Niching**: Maintain subpopulations
- **Periodic injection**: Add random individuals every N generations

---

#### 5. **Adjust Mutation Rate**

```yaml
ga:
  mutpb: 0.15  # Reduce from 0.25
```

Lower mutation rate will:
- Preserve clustered blocks better
- Reduce introduced violations
- Allow repair to catch up

---

#### 6. **Strengthen Elite Preservation**

```yaml
ga:
  elite_size: 0.10  # Increase from 0.05 (10% instead of 5%)
```

This ensures the best solutions survive longer.

---

### **📌 MEDIUM PRIORITY**

#### 7. **Investigate Availability Constraints**

Check if availability windows are too restrictive:
```python
# Analyze availability data
for instructor in instructors:
    available_quanta = len(instructor.availability)
    required_quanta = sum(course.quanta_per_week 
                          for course in instructor.qualified_courses)
    if available_quanta < required_quanta * 1.2:
        print(f"Instructor {instructor.id} may be overbooked")
```

**Consider**: Relaxing availability as a **soft constraint** instead of hard.

---

#### 8. **Add Constraint-Specific Repair Heuristics**

Once repair system is working, ensure these repairs exist:
- `repair_block_clustering()`: Merge fragmented sessions
- `repair_group_overlap()`: Shift conflicting sessions
- `repair_availability()`: Move sessions to valid time windows
- `repair_room_type()`: Swap to correct room types

---

#### 9. **Tune NSGA-II Parameters**

```yaml
ga:
  pop_size: 150  # Increase from 100
  ngen: 200  # More generations with better mechanisms
```

Larger population provides more diversity and search coverage.

---

#### 10. **Add Heatmap-Based Targeted Mutation**

You have `violation_heatmap` enabled, but ensure it's actually being used:
```yaml
enhancements:
  violation_heatmap:
    enabled: true
    target_hot_genes: true
    top_n_hotspots: 50  # Increase from 30
```

This focuses repair/mutation on problematic genes.

---

## 📋 **Detailed Constraint Analysis**

### **Soft Constraints (Final Values)**

| Constraint | Penalty | Observations |
|-----------|---------|--------------|
| Group Gaps | 605 | Acceptable - students have some idle time |
| Instructor Gaps | 425 | Acceptable - instructors have breaks |
| Midday Break Violations | 367 | Moderate - some sessions during lunch |
| **Total** | **1,397** | Secondary concern until hard constraints fixed |

**Note**: Soft constraints are irrelevant until hard violations are eliminated. Don't optimize these yet.

---

## 🔧 **Implementation Priority Matrix**

| Priority | Action | Impact | Effort | Time |
|----------|--------|--------|--------|------|
| 🔥 **P0** | Fix repair system | **CRITICAL** | Medium | 2-4 hours |
| 🔥 **P0** | Clean duplicate enrollments | **HIGH** | Low | 30 min |
| ⚠️ **P1** | Reduce block clustering weight | **HIGH** | Low | 5 min |
| ⚠️ **P1** | Add block-aware mutation | **HIGH** | High | 4-8 hours |
| ⚠️ **P1** | Increase diversity (restart) | **MEDIUM** | Medium | 2-3 hours |
| 📌 **P2** | Reduce mutation rate | **MEDIUM** | Low | 5 min |
| 📌 **P2** | Strengthen elite preservation | **MEDIUM** | Low | 5 min |
| 📌 **P2** | Audit availability constraints | **MEDIUM** | Medium | 1-2 hours |

---

## 🧪 **Suggested Experiment Sequence**

### **Experiment 1: Repair System Validation** (URGENT)
```bash
# Fix repair bug, then run:
python main.py --env test
# Expected: repairs_total > 0 in logger
```

### **Experiment 2: Quick Wins** (After repair fix)
```yaml
# configs/dev.yaml modifications:
hard_constraints:
  session_block_clustering_penalty:
    weight: 1.0  # Halve weight
ga:
  mutpb: 0.15  # Reduce mutation
  elite_size: 0.10  # Increase elites
```
```bash
python main.py --env dev
# Target: < 5,000 hard violations
```

### **Experiment 3: Data Quality** (Parallel task)
```bash
# Run data diagnostic script (create one to check for duplicates)
python scripts/check_data_quality.py
# Fix any issues found in Groups.json
```

### **Experiment 4: Full Production Run** (After above fixes)
```bash
python main.py --env prod
# Target: < 1,000 hard violations (or zero!)
```

---

## 📊 **Comparison: What Success Looks Like**

| Metric | Current (Bad) | Target (Good) | Stretch Goal |
|--------|--------------|---------------|--------------|
| **Hard Violations** | 10,064 | < 1,000 | **0** |
| **Soft Penalties** | 1,221 | < 800 | < 500 |
| **Diversity @ Gen 100** | 0.195 | > 0.30 | > 0.40 |
| **Stagnation Windows** | 12 gens | < 5 gens | None |
| **Repairs Executed** | **0 (BUG)** | > 1,000 | > 5,000 |

---

## 🎓 **Lessons Learned**

### What Went Wrong:
1. **Repair system completely failed** (0 repairs) - most critical issue
2. **Block clustering constraint** dominates (79% of violations)
3. **Data quality issues** (duplicate enrollments)
4. **Diversity collapsed** (78% loss) - premature convergence
5. **Initial population** was already terrible (12K violations)

### What Went Right:
1. **No qualification violations** - instructor-course matching works!
2. **No incomplete sessions** - course requirements satisfied!
3. **Feasibility checks passed** - problem is theoretically solvable
4. **Consistent performance** - 8.7s per generation is stable
5. **Multiprocessing worked** - parallel evaluation successful

---

## 🚀 **Next Steps (Actionable Plan)**

### **Phase 1: Critical Fixes (Today)**
1. ✅ Debug and fix repair system
2. ✅ Check Groups.json for duplicate enrollments
3. ✅ Reduce block clustering weight to 1.0
4. ✅ Reduce mutation rate to 0.15
5. ✅ Run quick test (`--env test`)

### **Phase 2: Validation (Tomorrow)**
1. ✅ Run dev config with fixes
2. ✅ Verify repairs > 0 in logger
3. ✅ Check hard violations < 5,000
4. ✅ Monitor diversity stays > 0.30

### **Phase 3: Advanced Improvements (This Week)**
1. ✅ Implement block-aware mutation
2. ✅ Add population restart mechanism
3. ✅ Tune availability constraints
4. ✅ Run full production config

### **Phase 4: Final Validation (Next Week)**
1. ✅ Achieve zero hard violations
2. ✅ Optimize soft constraints
3. ✅ Generate thesis-quality schedules
4. ✅ Document results

---

## 📝 **Conclusion**

**Current Status**: ❌ **System is NOT operational**

The schedule engine is currently unable to produce feasible schedules. The primary bottlenecks are:

1. **Non-functional repair system** (zero repairs executed)
2. **Overwhelming block clustering violations** (79% of all problems)
3. **Data quality issues** (duplicate enrollments)
4. **Severe diversity collapse** (population homogeneity)

**Good News**: The problem is theoretically solvable (feasibility checks passed), and the issues are **fixable** with targeted code and configuration changes.

**Estimated Time to Fix**: **1-2 weeks** with focused effort on:
- Repair system debugging
- Data cleaning
- Block clustering constraint refinement
- Diversity maintenance mechanisms

---

**Report Generated**: 2025-10-27  
**Analysis Duration**: 33 minutes of GA execution  
**Analyst**: GitHub Copilot  
**Confidence Level**: HIGH (based on comprehensive metrics and logs)

---

