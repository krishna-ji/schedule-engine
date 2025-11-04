# CP-SAT Runtime Logging System

## Overview

Comprehensive runtime logging has been added to the CP-SAT solver to track performance, identify bottlenecks, and monitor solving progress. Every CP-SAT run now automatically generates a timestamped log file with detailed timing information.

## Log File Location

```
output/cpsat_runtime_<timestamp>.log
```

Example: `output/cpsat_runtime_20250127_152535.log`

## Quick Start

### Run CP-SAT with Logging (Automatic)

```bash
# Single feasible solution (120s timeout)
uv run python main.py --config configs/cpsat_only.yaml --mode cpsat

# Check the log file
cat output/cpsat_runtime_*.log
```

### What Gets Logged

**1. Configuration**
- Mode (single solution vs multiple solutions)
- Time limit
- Random seed
- Number of target solutions

**2. Data Loading**
- Total time
- Number of courses, groups, instructors, rooms

**3. Model Building**
- Variable creation time
- Per-constraint-type timing (5 constraint types)
- Total variables and constraints count
- Total model build time

**4. Solving**
- Periodic progress updates (every 10s)
- Solution discovery timestamps
- Solver statistics (branches, conflicts, wall/user time)
- Final status (OPTIMAL, FEASIBLE, INFEASIBLE, etc.)

**5. Solution Decoding**
- Time to convert CP solution to CourseSession objects
- Number of sessions

**6. Total Runtime**
- Overall elapsed time from start to finish

## Log Format

```
[timestamp] | [level] | [message]
```

Example:
```
2025-01-27 15:25:35 | INFO     | CP-SAT MODE - Started
2025-01-27 15:25:35 | INFO     | CONFIGURATION:
2025-01-27 15:25:35 | INFO     |   Time Limit: 120s
2025-01-27 15:25:35 | INFO     |   Random Seed: 42
2025-01-27 15:25:35 | INFO     | ------------------------------------------------------------
2025-01-27 15:25:36 | INFO     | DATA LOADING - Completed in 1.23s
2025-01-27 15:25:36 | INFO     |   Courses: 45
2025-01-27 15:25:36 | INFO     |   Groups: 23
2025-01-27 15:25:36 | INFO     | ------------------------------------------------------------
2025-01-27 15:25:37 | INFO     | MODEL BUILDING - Started
2025-01-27 15:25:37 | INFO     |   Variables created: 2558 in 0.12s
2025-01-27 15:25:37 | INFO     |   [1/5] Adding group overlap constraints...
2025-01-27 15:25:38 | INFO     |       ✓ Group overlap constraints added in 1.45s
2025-01-27 15:25:38 | INFO     |   [2/5] Adding instructor conflict constraints...
2025-01-27 15:25:52 | INFO     |       ✓ Instructor conflict constraints added in 14.23s
2025-01-27 15:25:52 | INFO     |   [3/5] Adding availability constraints...
2025-01-27 15:25:53 | INFO     |       ✓ Availability constraints added in 0.89s
2025-01-27 15:25:53 | INFO     |   [4/5] Adding room conflict constraints...
2025-01-27 15:26:07 | INFO     |       ✓ Room conflict constraints added in 14.12s
2025-01-27 15:26:07 | INFO     |   [5/5] Adding valid quantum constraints...
2025-01-27 15:26:07 | INFO     |       ✓ Valid quantum constraints added in 0.05s
2025-01-27 15:26:07 | INFO     | MODEL BUILDING - Completed in 30.45s
2025-01-27 15:26:07 | INFO     |   Variables: 7674
2025-01-27 15:26:07 | INFO     |   Constraints: 3342789
2025-01-27 15:26:07 | INFO     | ------------------------------------------------------------
2025-01-27 15:26:07 | INFO     | SOLVING - Started
2025-01-27 15:26:17 | INFO     | Progress: 10s elapsed, still searching...
2025-01-27 15:26:42 | INFO     | SOLUTION 1 FOUND - Elapsed: 35.23s, Sessions: 127
2025-01-27 15:26:42 | INFO     | SOLVING - Completed in 35.45s
2025-01-27 15:26:42 | INFO     |   Status: OPTIMAL
2025-01-27 15:26:42 | INFO     |   Solutions Found: 1
2025-01-27 15:26:42 | INFO     |   Wall Time: 35.45s
2025-01-27 15:26:42 | INFO     |   User Time: 34.89s
2025-01-27 15:26:42 | INFO     |   Branches: 12458
2025-01-27 15:26:42 | INFO     |   Conflicts: 3421
2025-01-27 15:26:42 | INFO     | ------------------------------------------------------------
2025-01-27 15:26:42 | INFO     | DECODING SOLUTION - Completed in 0.12s
2025-01-27 15:26:42 | INFO     |   Total Sessions: 127
2025-01-27 15:26:42 | INFO     | ------------------------------------------------------------
2025-01-27 15:26:42 | INFO     | TOTAL RUNTIME: 67.12s
2025-01-27 15:26:42 | INFO     | ============================================================
2025-01-27 15:26:42 | INFO     | SUCCESS: Generated feasible solution with 127 sessions
```

## Performance Analysis

### Expected Timing (Sample Dataset ~45 courses)

| Phase | Expected Time | Alert If > |
|-------|--------------|------------|
| Data Loading | 0.5-2s | 5s |
| Variable Creation | 0.1-0.5s | 2s |
| Group Overlap Constraints | 0.5-3s | 10s |
| **Instructor Conflict Constraints** | **5-20s** | **60s** |
| Availability Constraints | 0.5-2s | 10s |
| **Room Conflict Constraints** | **5-20s** | **60s** |
| Valid Quantum Constraints | 0.05-0.2s | 1s |
| **Model Building Total** | **15-50s** | **120s** |
| **Solving** | **30-120s** | **300s** |
| Solution Decoding | 0.1-0.5s | 2s |
| **Overall Runtime** | **50-180s** | **400s** |

**Bottlenecks** (in order of impact):
1. Instructor conflict constraints (O(n²) pairwise checks)
2. Room conflict constraints (O(n²) pairwise checks)
3. Solving phase (depends on problem complexity)

### VM Performance Expectations

On VM with better resources:
- Model building: 30-50% faster
- Solving: 2-5x faster (depends on CP-SAT's internal parallelization)
- Instructor/room conflicts may still dominate model building time

## Analyzing Logs

### Finding the Slowest Phase

```bash
# Extract timing information
grep "Completed in" output/cpsat_runtime_*.log

# Sort by time
grep "added in" output/cpsat_runtime_*.log | sort -t: -k4 -n
```

### Monitoring Long-Running Solves

```bash
# Watch log in real-time
tail -f output/cpsat_runtime_*.log

# Check if solver is making progress
grep "Progress:" output/cpsat_runtime_*.log | tail -10
```

### Checking for Issues

**Infeasibility:**
```bash
grep "INFEASIBLE" output/cpsat_runtime_*.log
```
→ Problem has no valid solution, check constraints

**Timeout:**
```bash
grep "No solution found within time limit" output/cpsat_runtime_*.log
```
→ Increase `time_limit` or simplify problem

**Slow Model Building:**
```bash
grep "MODEL BUILDING - Completed" output/cpsat_runtime_*.log
```
→ If > 120s, constraint generation may be inefficient

## Configuration

### Adjust Time Limit

Edit `configs/cpsat_only.yaml`:
```yaml
ortools:
  time_limit: 300  # 5 minutes instead of 120s
```

### Generate Multiple Solutions

For hybrid mode (CP-SAT + GA):
```yaml
ortools:
  time_limit: 600  # 10 minutes
  num_solutions: 10  # Generate 10 feasible solutions
```

Log will show each solution as it's found:
```
SOLUTION 1 FOUND - Elapsed: 35.23s, Sessions: 127
SOLUTION 2 FOUND - Elapsed: 48.12s, Sessions: 125
...
```

## Troubleshooting

### Log File Not Created

**Check output directory exists:**
```bash
ls -la output/
```

**Test logger manually:**
```bash
uv run python -c "from src.ortools.cp_scheduler import setup_cp_logger; logger = setup_cp_logger(); logger.info('Test'); print('OK')"
```

### Constraint Count Explosion

**If log shows:**
```
Constraints: 30000000+
```

**Issue:** Availability constraints not optimized (should be ~2500)

**Fix:** Verify `constraint_factory.py` uses `AddAllowedAssignments()` domain restrictions instead of individual boolean constraints per quantum

### Solving Never Completes

**If stuck at:**
```
Progress: 300s elapsed, 0 solutions
```

**Options:**
1. **Increase time limit**: Set `time_limit: 600` or higher
2. **Simplify problem**: Reduce courses, relax availability
3. **Check feasibility**: Run feasibility check first
4. **Inspect constraints**: Verify constraint logic is correct

### Out of Memory

**If solver crashes:**
- Reduce number of constraints (simplify availability)
- Increase VM RAM
- Try sequential solving (disable parallelization if enabled)

## Files Modified

- `src/ortools/cp_scheduler.py` - Added `setup_cp_logger()`, logging in `generate_feasible_solutions()` and `generate_single_solution()`
- `src/ortools/model_builder.py` - Added logger parameter, timing tracking
- `src/ortools/constraint_factory.py` - Added logger parameter, per-constraint-type timing
- `main.py` - Added time import, integrated logger in `run_cpsat_mode()`

## Benefits

✅ **Identify Bottlenecks** - See exactly which phase takes longest  
✅ **Track Progress** - Monitor long-running solves in real-time  
✅ **Debug Issues** - Detailed context for infeasibility/timeouts  
✅ **Compare Runs** - Benchmark performance across different configs  
✅ **VM Deployment** - Monitor performance on remote machines  
✅ **Zero Configuration** - Automatic logging for all CP-SAT runs  

## Next Steps

1. **Run test:** `uv run python main.py --config configs/cpsat_only.yaml --mode cpsat`
2. **Check log:** `cat output/cpsat_runtime_*.log`
3. **Identify bottleneck:** Which phase takes longest?
4. **Optimize:** Based on timing data (if needed)
5. **Deploy to VM:** Test with production dataset

## See Also

- Full deployment guide: `DEPLOYMENT_CHECKLIST.md`
- OR-Tools documentation: `docs/for_report/`
- Configuration guide: `docs/CONFIG_VISUAL_GUIDE.md`
