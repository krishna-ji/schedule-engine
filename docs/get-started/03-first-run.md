# First Run Guide

## Quick Start (5 minutes)

### 1. Run Smoke Test

```powershell
# Navigate to project root
cd schedule-engine

# Run quick smoke test (30 generations, ~5 minutes)
uv run test
```

**What happens:**
1. Loads data from `data/` directory
2. Validates input data integrity
3. Runs feasibility checks
4. Initializes population (10 individuals)
5. Evolves for 30 generations
6. Generates best schedule
7. Exports results to `output/evaluation_<timestamp>/`

**Expected output:**
```
[12:34:56] Loading configuration...
[12:34:56] Environment: test
[12:34:56] Generations: 30, Population: 10

[12:34:57] Loading data...
✓ Courses: 150
✓ Groups: 30
✓ Instructors: 50
✓ Rooms: 40

[12:34:58] Validating input data...
✓ All validation checks passed

[12:34:59] Running feasibility checks...
✓ Instructor workload: PASS
✓ Room capacity: PASS
✓ All checks passed

[12:35:00] Initializing population (hybrid strategy)...
✓ 10 individuals created

[12:35:00] Starting NSGA-II evolution...
Gen 0 | Best: (-150, -45.2) | Diversity: 0.67
Gen 10 | Best: (-120, -38.1) | Diversity: 0.65
Gen 20 | Best: (-85, -32.5) | Diversity: 0.62
Gen 30 | Best: (-50, -28.7) | Diversity: 0.60

[12:40:12] Evolution complete (5m 12s)

[12:40:15] Decoding best schedule...
✓ 450 sessions scheduled

[12:40:18] Exporting results...
✓ JSON: output/evaluation_20251120_123418/schedule.json
✓ PDF: output/evaluation_20251120_123418/calendar.pdf
✓ Plots: output/evaluation_20251120_123418/plots/

results
✓ perfect schedule (no hard violations)
  soft penalty: 28.7
  sessions: 450
  output: output/evaluation_20251120_123418
  runtime: 312.5s
```

### 2. View Results

```powershell
# Navigate to output directory
cd output/evaluation_20251120_123418/

# View files
ls

# Expected files:
# - schedule.json          # Detailed schedule data
# - calendar.pdf           # Visual timetable
# - report.txt             # Text summary
# - plots/
#   - fitness_evolution.png
#   - diversity_evolution.png
#   - pareto_front.png
```

### 3. Inspect Schedule

**View Calendar PDF:**
- Open `calendar.pdf` in browser or PDF viewer
- Shows weekly timetable with color-coded sessions

**View JSON Data:**
```powershell
# Pretty-print JSON
cat schedule.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

**View Summary Report:**
```powershell
cat report.txt
```

Example report:
```
============================================================
Schedule Engine - Experiment Report
============================================================

Experiment: evaluation_20251120_123418
Environment: test
Runtime Mode: baseline (Pure NSGA-II)
Timestamp: 2025-11-20 12:34:18

------------------------------------------------------------
Configuration
------------------------------------------------------------
Generations: 30
Population Size: 10
Crossover Probability: 0.75
Mutation Probability: 0.25
Elite Preservation: True (10%)

------------------------------------------------------------
Final Results
------------------------------------------------------------
Best Fitness: (-50, -28.7)
  Hard Violations: 50
  Soft Penalty: 28.7

Total Sessions: 450
Runtime: 5m 12s

------------------------------------------------------------
Constraint Violations
------------------------------------------------------------
Hard Constraints:
  student_group_exclusivity: 15 violations
  instructor_exclusivity: 20 violations
  room_exclusivity: 15 violations
  (Other constraints: 0 violations)

Soft Constraints:
  avoid_early_sessions: 12 violations
  avoid_late_sessions: 8 violations
  instructor_preferences: 5 violations
  (Total soft penalty: 28.7)

------------------------------------------------------------
Convergence Statistics
------------------------------------------------------------
Initial Best: (-220, -65.3)
Final Best: (-50, -28.7)
Improvement: 77% (hard), 56% (soft)

Generations to first feasible: N/A (not achieved)
Generations at best: 28

Average Diversity: 0.63
Final Diversity: 0.60
```

## Running Different Modes

### Baseline Experiment (Pure NSGA-II)

```powershell
# Short test run
uv run exp1 --env test

# Full production run (2000 generations, ~1-2 hours)
uv run exp1 --env prod
```

### With Repairs (IGLS)

```powershell
# Test run
uv run exp2 --env test

# Production run
uv run exp2 --env prod
```

### With Heuristics

```powershell
# Test run
uv run exp3 --env test

# Production run
uv run exp3 --env prod
```

### Full GA (Best Non-RL)

```powershell
# Test run
uv run exp4 --env test

# Production run
uv run exp4 --env prod
```

### RL-Guided (Advanced)

```powershell
# Requires trained RL model (see RL Training Guide)
uv run exp5 --env test
uv run exp5 --env prod
```

## Understanding Output

### Fitness Values

Schedule Engine uses two-objective minimization:

- **First objective**: Hard constraint violations (must be 0 for feasible schedule)
- **Second objective**: Soft constraint penalty (minimize for better quality)

Example: `(-50, -28.7)`
- 50 hard violations (schedule is infeasible)
- 28.7 soft penalty

**Goal:** Achieve `(0, X)` where X is minimized.

### Diversity Metric

- Measures genetic diversity in population
- Range: [0, 1] (1 = maximum diversity)
- Typical range: 0.6-0.8
- Too high (>0.9): Population too dispersed, slow convergence
- Too low (<0.4): Population too similar, premature convergence

### Evolution Plots

**Fitness Evolution** (`plots/fitness_evolution.png`):
- Shows best/median/worst fitness over generations
- Should show downward trend (minimization)

**Diversity Evolution** (`plots/diversity_evolution.png`):
- Shows population diversity over time
- Gradual decrease expected (convergence)

**Pareto Front** (`plots/pareto_front.png`):
- Shows trade-off between objectives
- Points closer to origin are better

## Common First Run Issues

### Issue 1: No GPU detected

**Symptom:**
```
[WARN] CUDA not available, using CPU fallback
```

**Impact:** Slower evaluation (10-50x), but still works.

**Solution:**
- Install NVIDIA drivers
- Verify: `nvidia-smi`
- Reinstall PyTorch with CUDA: `uv sync --frozen --reinstall-package torch`

### Issue 2: High hard violations

**Symptom:**
```
Best: (-150, -42.1)
```

**Reason:** Smoke test (30 generations) insufficient for convergence.

**Solution:** Run longer:
```powershell
# Medium run (500 generations, ~30 minutes)
python main.py --env test --config configs/test.yaml
# Modify configs/test.yaml: ngen: 500

# Production run (2000 generations)
uv run prod
```

### Issue 3: Out of memory

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solution:**
- Reduce population size in config
- Disable GPU (`gpu.enabled: false` in config)
- Close other applications

### Issue 4: Data file errors

**Symptom:**
```
FileNotFoundError: data/Course.json not found
```

**Solution:**
```powershell
# Verify data files exist
ls data/

# Check data integrity
uv run check-data

# If missing, restore from archive
cp data/archive/Course.json data/
```

## Next Steps

### Run All Thesis Experiments

```powershell
# Run all 5 progressive experiments (6-10 hours total)
uv run exp1 --env prod  # Baseline
uv run exp2 --env prod  # + Repairs
uv run exp3 --env prod  # + Heuristics
uv run exp4 --env prod  # + Local search
uv run exp5 --env prod  # + RL-guided (requires training)
```

### Analyze Results

```powershell
# Compare experiments
uv run compare-experiments

# Generate thesis plots
uv run generate-thesis-plots

# Export metrics to CSV
uv run export-thesis-data
```

### Train RL Agent

```powershell
# See RL Training Guide for detailed instructions
uv run train-rl --timesteps 100000 --env prod
```

## Additional Resources

- [UV Commands Reference](04-uv-commands.md) - All available commands
- [Runtime Modes Guide](../02-user-guides/runtime-modes.md) - Detailed mode documentation
- [Configuration Guide](02-setup.md) - Customize settings
- [Architecture Overview](../architecture/01-high-level-architecture.md) - System design
- [Troubleshooting Guide](../troubleshooting/01-common-issues.md) - Common issues
