# Production (prod.yaml) Runtime Breakdown

## What Happens When You Run prod.yaml Directly

### Configuration Overview
```yaml
ngen: 500 generations
pop_size: 100 individuals
multiprocessing: ON (auto-detect cores)
exhaustive_triggers: 6 times at [3, 30, 100, 200, 350, 480]
exhaustive_timeout: 10 minutes (600s) per trigger
```

---

## Timeline Breakdown (12-24 Hour Run)

### Phase 1: Initial Evolution (Gens 1-30)
**Duration: 1-2 hours**

```
Gen 1:   Population initialization (hybrid: 25% greedy, 50% smart, 25% random)
Gen 2:   Normal evolution (~5-10s per gen with 100 pop)
Gen 3:    EXHAUSTIVE SEARCH TRIGGER #1
         ├─ Top 15% (15 individuals)
         ├─ 120 max neighbors per gene
         ├─ Timeout: 10 minutes
         └─ Expected: 3-8 minutes (early violations are easy to fix)
         
Gens 4-29: Normal evolution
Gen 30:   EXHAUSTIVE SEARCH TRIGGER #2
         ├─ Top 15% (15 individuals)
         ├─ Expected: 5-10 minutes (more optimized already)
         └─ Major consolidation of solutions
```

**Phase 1 Result**: Hard violations reduced 40-60%

---

### Phase 2: Mid Evolution (Gens 31-199)
**Duration: 3-5 hours**

```
Gens 31-99:   Normal evolution
              ├─ May trigger stagnation repair if stuck
              └─ Selective repair cleaning 40% of offspring

Gen 100:  EXHAUSTIVE SEARCH TRIGGER #3
         ├─ Mid-evolution deep optimization
         ├─ Expected: 8-10 minutes (harder to improve now)
         └─ Refinement of established solutions

Gens 101-199: Continued evolution
              ├─ Population converging
              └─ Quality improvements slowing down
```

**Phase 2 Result**: Hard violations reduced 60-80% from initial

---

### Phase 3: Late Mid Evolution (Gens 200-349)
**Duration: 3-5 hours**

```
Gen 200:  EXHAUSTIVE SEARCH TRIGGER #4
         ├─ Late mid-evolution push
         ├─ Expected: 9-10 minutes (diminishing returns)
         └─ Fine-tuning near-optimal solutions

Gens 201-349: Refinement phase
              ├─ Small incremental improvements
              ├─ Possible stagnation repair triggers
              └─ Population highly converged
```

**Phase 3 Result**: Approaching local optima

---

### Phase 4: Final Refinement (Gens 350-500)
**Duration: 5-8 hours**

```
Gen 350:  EXHAUSTIVE SEARCH TRIGGER #5
         ├─ Late evolution refinement
         ├─ Expected: 9-10 minutes (very small improvements)
         └─ Polishing best solutions

Gens 351-479: Final evolution
              ├─ Minimal changes
              ├─ Population fully converged
              └─ Quality plateau reached

Gen 480:  EXHAUSTIVE SEARCH TRIGGER #6
         ├─ Final polish before end
         ├─ Expected: 9-10 minutes
         └─ Last chance optimization

Gens 481-500: Completion
```

**Phase 4 Result**: Maximum quality achieved

---

## Total Runtime Estimation

### Time Budget Breakdown

| Component | Time | Notes |
|-----------|------|-------|
| **Normal Evolution** | 10-15 hours | 494 gens × 1-2 min/gen with 100 pop |
| **6 Exhaustive Triggers** | 1-1.5 hours | 6 × 8-10 minutes average |
| **Stagnation Repairs** | 0.5-1 hour | 3-5 triggers × 5-10 minutes |
| **Overhead** | 0.5-1 hour | I/O, evaluation, logging |
| **TOTAL** | **12-18 hours** | Typical range |

### Best Case (Fast Hardware)
- 8+ cores, all utilized
- Fast CPU, plenty of RAM
- No swapping, no timeouts
- **~12 hours**

### Worst Case (Slower Hardware)
- 4-6 cores
- Limited RAM (8-12GB)
- Some timeouts occur
- **~24 hours**

### Realistic Case (Medium Hardware)
- 6-8 cores
- 16GB RAM
- No major issues
- **~15-18 hours**

---

## Why Not Run Prod Directly? (The Real Reasons)

### ⚠️ Issue #1: You Don't Know If It Will Complete

**Problem**: With 100 pop, exhaustive search at gen 3 might:
- Take 3 minutes (good!)
- Take 10 minutes and timeout (bad - incomplete optimization)
- Take > 10 minutes and abort (worst - wasted 10 min + incomplete)

**Solution**: Run `prod_test.yaml` first (30 gens, 30-60 min) to measure:
```bash
python main.py --config configs/prod_test.yaml
```

Watch for:
```
 Gen 3: EXHAUSTIVE SEARCH triggered
    Exhaustive search complete: N genes improved, time: X.Xs
```

If X < 300s: You can run prod.yaml safely 
If X > 300s: Use prod_safe.yaml instead (50 pop) ⚠️

---

### ⚠️ Issue #2: Resource Requirements Unknown

**Your VM specs unknown:**
- CPU cores: ??
- RAM: ??
- Other processes: ??

**prod.yaml needs:**
- 8+ cores (optimal)
- 16+ GB RAM
- 12-24 hours continuous runtime

**What if your VM has:**
- Only 4 cores? → Multiprocessing less effective, 2x slower
- Only 8GB RAM? → May OOM with 100 pop × 527 genes
- Other jobs running? → Resource contention

---

### ⚠️ Issue #3: Wasted Time on Wrong Settings

**Scenario**: You start prod.yaml, it runs for 8 hours, then:

1. **Exhaustive timeouts at gen 100**
   - Wasted 8 hours getting there
   - Now stuck with incomplete optimization

2. **Out of memory at gen 250**
   - Half-complete run
   - No usable results

3. **Taking too long (> 24 hours estimated)**
   - Can't wait that long
   - Have to kill process
   - No results

**Solution**: Test first, validate, then commit to long run

---

## The Smart Approach: Incremental Testing

### Step 1: Quick Smoke Test (5 min)
```bash
python main.py --env test
```
**Validates**:
- IGLS system works
- No crashes
- Config loads

 Already done!

---

### Step 2: Scale Test (30-60 min) ← **DO THIS NEXT**
```bash
python main.py --config configs/prod_test.yaml
```
**Measures**:
- Time for exhaustive with 100 pop (gen 3, 25)
- Memory usage with 100 pop
- Multiprocessing effectiveness

**Decision criteria**:
- Gen 3 exhaustive < 5 min:  Run prod.yaml
- Gen 3 exhaustive 5-10 min: ⚠️ Run prod_safe.yaml
- Gen 3 exhaustive > 10 min:  Need to tune (reduce pop or coverage)

---

### Step 3: Medium Production Run (4-6 hours)
```bash
python main.py --config configs/prod_safe.yaml
```
**Features**:
- 250 gens (enough for convergence)
- 50 pop (VM-friendly)
- 5 exhaustive triggers
- Completes in work-day time

**Validates**:
- Full evolution cycle
- IGLS effectiveness
- Stagnation repair works

---

### Step 4: Full Production (12-24 hours)
```bash
python main.py --env prod
```
**Only if**:
-  prod_test.yaml completed successfully
-  Exhaustive took < 5 minutes with 100 pop
-  Memory usage was OK (< 80% peak)
-  You have 12-24 hours to spare

---

## Direct Answer: Can You Run prod.yaml Now?

### Technical Answer: YES
```bash
# This will work (no errors, will run)
python main.py --env prod
```

### Practical Answer: NOT RECOMMENDED YET

**Why?**
1.  You haven't validated 100 pop performance
2.  Don't know your VM's actual capacity
3.  Don't know if exhaustive will timeout
4.  12-24 hour commitment without validation
5.  Might waste time on wrong settings

**Instead, do this:**
```bash
# 1. Test 100 pop scale (30-60 min)
python main.py --config configs/prod_test.yaml

# 2. If that works well, decide:
#    - Gen 3 exhaustive < 5 min? → Run prod.yaml (12-24h)
#    - Gen 3 exhaustive > 5 min? → Run prod_safe.yaml (4-6h)
```

---

## Summary Table

| Config | Runtime | Purpose | When to Use |
|--------|---------|---------|-------------|
| **test** | 5 min | Smoke test |  Always first (done!) |
| **prod_test** | 30-60 min | Scale validation |  **DO THIS NEXT** |
| **prod_safe** | 4-6 hours | VM-friendly production | After prod_test, if 100 pop is slow |
| **prod** | 12-24 hours | Maximum quality | After prod_test, if 100 pop is fast |

---

## Recommended Action

```bash
# Run this now (30-60 minutes):
python main.py --config configs/prod_test.yaml

# Then check the output for gen 3 exhaustive time
# If < 5 minutes: You can run prod.yaml
# If > 5 minutes: Use prod_safe.yaml instead
```

**After prod_test completes, tell me:**
1. How long did gen 3 exhaustive take?
2. How long did gen 25 exhaustive take?
3. What's your VM specs (cores/RAM)?

Then I'll tell you which config is optimal for your setup! 
