# Production Run Guide - IGLS System

## Quick Decision Matrix

| Your VM Specs | Recommended Config | Expected Runtime | Command |
|---------------|-------------------|------------------|---------|
| 2-4 cores, 4-8GB RAM | **test** | 5-10 min | `python main.py --env test` |
| 4-8 cores, 8-16GB RAM | **prod_safe** | 4-6 hours | `python main.py --config configs/prod_safe.yaml` |
| 8-16 cores, 16-32GB RAM | **prod** | 12-24 hours | `python main.py --env prod` |
| Testing 100 pop scale | **prod_test** | 30-60 min | `python main.py --config configs/prod_test.yaml` |

## Configuration Comparison

### test.yaml (✓ Already Verified)
```
ngen: 30
pop_size: 10
multiprocessing: OFF
exhaustive: [3, 25]
runtime: ~5 min
```
**Use for**: Quick smoke tests, debugging

### prod_safe.yaml (✅ Recommended for VM)
```
ngen: 250
pop_size: 50
multiprocessing: ON
exhaustive: [3, 30, 100, 180, 240] (5 triggers)
runtime: ~4-6 hours
```
**Use for**: First production run, VM-friendly, balanced quality

### prod_test.yaml (🔬 For Testing)
```
ngen: 30
pop_size: 100
multiprocessing: ON
exhaustive: [3, 25]
runtime: ~30-60 min
```
**Use for**: Test if your VM can handle 100 pop before long run

### prod.yaml (🚀 Full Production)
```
ngen: 500
pop_size: 100
multiprocessing: ON
exhaustive: [3, 30, 100, 200, 350, 480] (6 triggers)
runtime: ~12-24 hours
```
**Use for**: Final production runs on powerful hardware

## Step-by-Step VM Run Guide

### Step 1: Quick Sanity Check (5 min)
```bash
python main.py --env test
```
✓ Verifies IGLS system working
✓ Confirms no environment issues

### Step 2: Test Production Scale (Optional, 30-60 min)
```bash
python main.py --config configs/prod_test.yaml
```
✓ Tests 100 pop performance
✓ Measures time/memory usage
✓ Validates exhaustive timeout settings

**Monitor during this run:**
- CPU usage (should be 80-100%)
- Memory usage (watch for OOM)
- Time at gen 3 exhaustive trigger
- Time at gen 25 exhaustive trigger

**Decision criteria:**
- If gen 3 exhaustive < 300s: Can proceed to prod.yaml
- If gen 3 exhaustive > 300s: Use prod_safe.yaml instead
- If memory > 12GB: Reduce pop_size to 50

### Step 3: Production Run

#### Option A: Safe Production (Recommended)
```bash
# Start background run with logging
python main.py --config configs/prod_safe.yaml 2>&1 | Tee-Object output/prod_safe_run.log

# Or for long run, use nohup equivalent:
Start-Job -ScriptBlock { python main.py --config configs/prod_safe.yaml }
```

**Expected:**
- Runtime: 4-6 hours
- Triggers: 5 exhaustive searches
- Quality: Good (250 gens sufficient for convergence)

#### Option B: Full Production (If You Have Time)
```bash
# Start background run
python main.py --env prod 2>&1 | Tee-Object output/prod_full_run.log

# Or use screen/tmux on Linux, or scheduled task on Windows
```

**Expected:**
- Runtime: 12-24 hours
- Triggers: 6 exhaustive searches
- Quality: Best (500 gens)

## Monitoring Your Run

### Check Progress
```bash
# Watch the log
Get-Content output/prod_safe_run.log -Tail 20 -Wait

# Or check latest output directory
Get-ChildItem output -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
```

### Key Things to Watch

1. **IGLS Triggers**
```
🔥 Gen X: EXHAUSTIVE SEARCH triggered
   ✅ Exhaustive search complete: N genes improved, total reduction: X, time: Ys
```
- Time should be < timeout (300s for prod_safe, 600s for prod)
- If timing out: Reduce coverage or increase timeout

2. **Stagnation Repair**
```
🔄 Stagnation detected! Triggering GREEDY REPAIR
   ✅ Greedy repair complete: ...
```
- Should trigger if stuck
- If triggering too often (< cooldown): Increase patience

3. **Generation Progress**
```
...OK!... Gen 25/250: Hard=2306, Soft=487.00, Time=1.2s
```
- Normal gens: < 2s
- Exhaustive gens: 30-300s
- If regular gens > 5s: Multiprocessing may not be working

4. **Memory Issues**
```
MemoryError: ...
```
- Reduce pop_size
- Disable multiprocessing (slower but safer)

## Troubleshooting

### Problem: Exhaustive Search Times Out
**Symptom**: `⚠ Exhaustive search timed out`

**Solution**:
```yaml
# In your config, increase timeout or reduce coverage:
exhaustive_search:
  population_coverage: 0.15  # Reduce from 0.25
  timeout_seconds: 600  # Increase from 300
```

### Problem: Out of Memory
**Symptom**: Process killed or MemoryError

**Solution**:
```yaml
# Reduce population size:
ga:
  pop_size: 30  # Reduce from 50

# Or disable multiprocessing:
parallel:
  use_multiprocessing: false
```

### Problem: Too Slow (> 10s per gen)
**Symptom**: Regular generations taking > 5s

**Solution**:
```yaml
# Ensure multiprocessing is working:
parallel:
  use_multiprocessing: true
  num_workers: 4  # Set explicitly to match your cores
```

### Problem: Run Interrupted
**Symptom**: Process stopped mid-run

**Recovery**: 
- Check `output/` for partial results
- Restart with fewer generations
- Use `prod_safe.yaml` instead of `prod.yaml`

## Post-Run Analysis

After run completes, check:

```bash
# Find output directory
$latest = Get-ChildItem output -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1

# View summary
Get-Content "$latest/logger.txt" | Select-Object -Last 50

# Check IGLS effectiveness
Get-Content "$latest/logger_all.csv" | Select-String "igls_exhaustive"

# View results
Start-Process "$latest/ScheduleCalendar.pdf"
```

## Summary: Can You Run in VM Now?

✅ **Yes, with the right config:**

1. **For immediate testing**: Use `test` (5 min)
2. **For first production run**: Use `prod_safe.yaml` (4-6 hours)
3. **For scale testing**: Use `prod_test.yaml` (30-60 min)
4. **For final quality**: Use `prod.yaml` (12-24 hours, requires good hardware)

**My recommendation**: Start with `prod_safe.yaml` - it's optimized for typical VM resources and will complete in 4-6 hours with good quality.

```bash
# Run this now:
python main.py --config configs/prod_safe.yaml
```

Then monitor the log and adjust based on performance!
