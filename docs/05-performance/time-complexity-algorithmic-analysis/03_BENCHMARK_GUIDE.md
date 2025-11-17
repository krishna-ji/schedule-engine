# Benchmark and Profiling Guide

**Document Version:** 1.0  
**Date:** November 17, 2025  
**Status:** Implementation Guide

---

## Overview

This guide provides instructions for benchmarking and profiling constraint checking performance to validate complexity analysis and measure optimization impact.

---

## 1. Benchmark Script Usage

### 1.1 Basic Benchmark

Run basic performance benchmark on current dataset:

```bash
# Using UV (recommended)
uv run python scripts/bench_constraint_check.py

# Or direct Python
python scripts/bench_constraint_check.py
```

**Output:**
```
============================================================
Benchmarking: data
============================================================
Loading scheduling context...
Generating population (size=50)...
Individual: 148 sessions, ~592 quanta

Benchmarking decoding...
  Decode: 2.345 ± 0.123 ms
Benchmarking full evaluation...
  Total: 38.567 ± 1.234 ms
  Range: [36.123, 42.345] ms
  Hard constraints: 12.456 ms
  Soft constraints: 23.789 ms

Top 5 slowest constraints:
  soft.session_continuity                 : 12.345 ms
  soft.student_schedule_compactness       :  6.789 ms
  soft.instructor_schedule_compactness    :  4.567 ms
  hard.course_completeness                :  3.456 ms
  hard.student_group_exclusivity          :  2.345 ms
```

### 1.2 Custom Parameters

```bash
# Larger population
python scripts/bench_constraint_check.py --pop 200

# More timing runs (better statistical accuracy)
python scripts/bench_constraint_check.py --runs 100

# Save results to JSON
python scripts/bench_constraint_check.py --output output/benchmark_results.json
```

### 1.3 Dataset Comparison

Compare performance across different dataset sizes:

```bash
# Requires data organized as: data/small/, data/medium/, data/large/
python scripts/bench_constraint_check.py --compare

# Output:
# ================================================================================
# COMPARISON SUMMARY
# ================================================================================
# Dataset         Sessions   Decode     Hard       Soft       Total     
# --------------------------------------------------------------------------------
# small           52         1.23       5.67       8.90       15.80     
# medium          148        2.34       12.45      23.78      38.57     
# large           312        5.67       28.90      56.78      91.35     
```

---

## 2. Profiling with cProfile

### 2.1 Basic Profiling

```bash
# Generate profile
python -m cProfile -o output/profile.out scripts/bench_constraint_check.py

# Analyze profile
python -m pstats output/profile.out
```

**In pstats shell:**
```python
# Top 20 functions by cumulative time
stats> sort cumulative
stats> stats 20

# Top 20 by total time
stats> sort tottime
stats> stats 20

# Filter to constraint functions
stats> sort cumulative
stats> stats constraint

# Print callers of a function
stats> callers student_group_exclusivity

# Print callees
stats> callees evaluate
```

### 2.2 Generate Call Graph

```bash
# Install gprof2dot
pip install gprof2dot

# Convert profile to dot graph
gprof2dot -f pstats output/profile.out | dot -Tpng -o output/profile_graph.png

# View graph
start output/profile_graph.png  # Windows
open output/profile_graph.png   # macOS
xdg-open output/profile_graph.png  # Linux
```

---

## 3. Profiling with py-spy

### 3.1 Top View (Real-Time)

```bash
# Install py-spy
pip install py-spy

# Live top view
py-spy top -- python scripts/bench_constraint_check.py
```

**Output:**
```
Total Samples: 1000
%Own   %Total  Function
40.2%  45.3%   student_schedule_compactness - soft.py:45
15.6%  18.2%   session_continuity - soft.py:182
12.3%  14.1%   instructor_exclusivity - hard.py:67
...
```

### 3.2 Flame Graph

```bash
# Record flame graph (SVG)
py-spy record -o output/flamegraph.svg -- python scripts/bench_constraint_check.py

# With higher sampling rate
py-spy record -r 500 -o output/flamegraph.svg -- python scripts/bench_constraint_check.py

# View in browser
start output/flamegraph.svg
```

### 3.3 Speedscope Format

```bash
# Record in speedscope format (better interactive visualization)
py-spy record --format speedscope -o output/profile.speedscope.json -- python scripts/bench_constraint_check.py

# Upload to https://www.speedscope.app/ for visualization
```

---

## 4. Profiling with pyinstrument

### 4.1 Basic Usage

```bash
# Install pyinstrument
pip install pyinstrument

# Profile script
pyinstrument scripts/bench_constraint_check.py
```

**Output:**
```
  _     ._   __/__   _ _  _  _ _/_   Recorded: 10:45:32  Samples:  1234
 /_//_/// /_\ / //_// / //_'/ //     Duration: 5.234     CPU time: 5.123
/   _/                      v3.4.2

Program: scripts/bench_constraint_check.py

5.234 <module>  bench_constraint_check.py:1
└─ 5.123 main  bench_constraint_check.py:245
   ├─ 2.345 benchmark_full_evaluation  bench_constraint_check.py:156
   │  └─ 2.123 evaluate  fitness.py:18
   │     ├─ 1.234 student_schedule_compactness  soft.py:45
   │     ├─ 0.456 session_continuity  soft.py:182
   │     └─ 0.234 instructor_exclusivity  hard.py:67
   └─ 1.234 generate_population  population.py:34
```

### 4.2 HTML Report

```bash
# Generate HTML report
pyinstrument -r html -o output/profile_report.html scripts/bench_constraint_check.py

# Open in browser
start output/profile_report.html
```

---

## 5. Line-by-Line Profiling

### 5.1 Setup line_profiler

```bash
# Install line_profiler
pip install line_profiler
```

### 5.2 Add Decorators

Add `@profile` decorator to functions you want to profile:

```python
# In src/constraints/soft.py
@profile  # Add this line
def student_schedule_compactness(sessions: List[CourseSession]) -> int:
    penalty = 0
    # ... rest of function
```

### 5.3 Run Profiler

```bash
# Profile with kernprof
kernprof -l -v scripts/bench_constraint_check.py

# Output shows time per line:
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#     45                                           def student_schedule_compactness(...):
#     46         1        123.0    123.0      5.2      penalty = 0
#     47         1       1234.0   1234.0     52.3      group_day_quanta = defaultdict(...)
#     48       150        567.0      3.8     24.0      for session in sessions:
# ...
```

---

## 6. Memory Profiling

### 6.1 memory_profiler

```bash
# Install memory_profiler
pip install memory_profiler

# Add @profile decorator to functions
# Then run:
python -m memory_profiler scripts/bench_constraint_check.py
```

### 6.2 memray (Modern Alternative)

```bash
# Install memray
pip install memray

# Profile memory
memray run scripts/bench_constraint_check.py

# Generate flame graph
memray flamegraph memray-*.bin

# View in browser
start memray-flamegraph-*.html
```

---

## 7. Benchmarking Before/After Optimization

### 7.1 Baseline Benchmark

Before optimization:

```bash
# Run 3 times and save results
for i in {1..3}; do
    python scripts/bench_constraint_check.py \
        --runs 100 \
        --output output/baseline_run_$i.json
done

# Calculate average
python scripts/analyze_benchmark_results.py output/baseline_run_*.json
```

### 7.2 Post-Optimization Benchmark

After optimization:

```bash
# Run same benchmark
for i in {1..3}; do
    python scripts/bench_constraint_check.py \
        --runs 100 \
        --output output/optimized_run_$i.json
done
```

### 7.3 Compare Results

```bash
# Create comparison script (scripts/compare_benchmarks.py)
python scripts/compare_benchmarks.py \
    --baseline output/baseline_run_*.json \
    --optimized output/optimized_run_*.json
```

**Expected Output:**
```
OPTIMIZATION IMPACT ANALYSIS
============================

Overall Performance:
  Baseline:   38.57 ± 1.23 ms
  Optimized:  19.23 ± 0.87 ms
  Speedup:    2.00× faster
  Improvement: 50.1%

Per-Constraint Impact:
  student_schedule_compactness: 12.34ms → 6.12ms (2.02× faster)
  session_continuity:           12.35ms → 5.67ms (2.18× faster)
  instructor_exclusivity:        2.34ms → 2.31ms (1.01× faster)
  ...
```

---

## 8. Continuous Performance Monitoring

### 8.1 Add to CI/CD Pipeline

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmark

on:
  pull_request:
    branches: [main, dev]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install uv
          uv sync
      
      - name: Run benchmark
        run: |
          uv run python scripts/bench_constraint_check.py \
            --runs 50 \
            --output benchmark_results.json
      
      - name: Compare with baseline
        run: |
          # Compare with main branch baseline
          git checkout main
          uv run python scripts/bench_constraint_check.py \
            --runs 50 \
            --output baseline_results.json
          
          git checkout -
          
          python scripts/compare_benchmarks.py \
            --baseline baseline_results.json \
            --current benchmark_results.json \
            --fail-on-regression 20  # Fail if >20% slower
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: benchmark_results.json
```

### 8.2 Performance Dashboard

Track performance over time:

```bash
# Append to historical log
echo "$(date +%Y-%m-%d),$(jq '.results[0].timings.total_ms' benchmark_results.json)" \
    >> performance_history.csv

# Generate plot
python scripts/plot_performance_history.py performance_history.csv
```

---

## 9. Profiling RL Training

### 9.1 Profile Full Episode

```bash
# Add profiling to RL training script
py-spy record -o output/rl_training_profile.svg -- \
    python scripts/train_rl_agent.py --episodes 10
```

### 9.2 Profile Per-Step Overhead

```python
# In src/rl/gym_env/schedule_env.py
import cProfile
import pstats

def step(self, action: int):
    profiler = cProfile.Profile()
    profiler.enable()
    
    # ... step logic ...
    
    profiler.disable()
    
    # Save profile every 100 steps
    if self.current_step % 100 == 0:
        stats = pstats.Stats(profiler)
        stats.dump_stats(f"output/rl_step_{self.current_step}.prof")
```

---

## 10. Interpreting Results

### 10.1 Key Metrics to Track

1. **Total Evaluation Time** - Overall constraint checking speed
2. **Per-Constraint Time** - Identify bottlenecks
3. **Standard Deviation** - Measure consistency
4. **Scaling Behavior** - How time grows with dataset size

### 10.2 Red Flags

⚠️ **Watch for:**
- Constraints taking >10ms (potential optimization target)
- High std deviation (inconsistent performance)
- Super-linear growth (O(n²) or worse)
- Memory leaks (growing memory usage)

### 10.3 Success Criteria

✅ **Optimization successful if:**
- Total time reduced by >20%
- No correctness regressions (unit tests pass)
- Consistent performance (std dev <10% of mean)
- Scales sub-linearly (close to O(n) or O(n log n))

---

## 11. Troubleshooting

### 11.1 Benchmark is Too Fast

If timing is <1ms, increase workload:

```bash
# Larger population
python scripts/bench_constraint_check.py --pop 500

# More runs
python scripts/bench_constraint_check.py --runs 1000
```

### 11.2 Profiler Overhead

Profiling adds overhead. For accurate timing:
1. Use py-spy (sampling, minimal overhead)
2. Disable profiler for final benchmarks
3. Run multiple times and take average

### 11.3 Inconsistent Results

```bash
# Disable CPU frequency scaling (Linux)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Increase process priority (Windows)
start /HIGH python scripts/bench_constraint_check.py

# Disable background tasks
# Close other applications before benchmarking
```

---

## 12. Profiling Commands Quick Reference

```bash
# cProfile - deterministic profiling
python -m cProfile -o profile.out script.py
python -m pstats profile.out

# py-spy - sampling profiler (minimal overhead)
py-spy top -- python script.py
py-spy record -o flamegraph.svg -- python script.py

# pyinstrument - call stack profiler
pyinstrument script.py
pyinstrument -r html -o report.html script.py

# line_profiler - line-by-line profiling
kernprof -l -v script.py

# memory_profiler - memory usage
python -m memory_profiler script.py

# memray - modern memory profiler
memray run script.py
memray flamegraph memray-*.bin
```

---

## 13. Next Steps

After benchmarking:

1. ✅ Identify bottleneck constraints (>10ms)
2. ✅ Analyze complexity (see `01_COMPLEXITY_ANALYSIS.md`)
3. ✅ Implement optimizations (see `02_OPTIMIZATION_STRATEGIES.md`)
4. ✅ Re-benchmark to validate improvements
5. ✅ Add unit tests to prevent regressions
6. ✅ Update documentation with results

---

## References

- Benchmark Script: `scripts/bench_constraint_check.py`
- Complexity Analysis: `01_COMPLEXITY_ANALYSIS.md`
- Optimization Strategies: `02_OPTIMIZATION_STRATEGIES.md`
- Python Profiling Docs: https://docs.python.org/3/library/profile.html
- py-spy: https://github.com/benfred/py-spy
- pyinstrument: https://github.com/joerick/pyinstrument
