# Time Complexity & Algorithmic Analysis

**Last Updated:** November 17, 2025  
**Status:** Complete Initial Analysis

---

## Overview

Comprehensive algorithmic complexity analysis and optimization guide for the schedule-engine constraint checking system. This analysis addresses performance bottlenecks in constraint evaluation affecting both GA evolution and RL training.

---

## 📚 Documents

### [01_COMPLEXITY_ANALYSIS.md](./01_COMPLEXITY_ANALYSIS.md)
**Complete Big-O complexity analysis of all constraint functions**

- Per-function complexity analysis with concrete examples
- Scaling behavior for different dataset sizes
- Performance estimates and bottleneck identification
- Hard constraints: O(S × Q) typical
- Soft constraints: O(S × Q + N × D × Q_d × log(Q_d))
- Total evaluation: ~40ms for medium dataset (150 sessions)

**Key Findings:**
- ✅ Most constraints are already optimal (hash-based, single-pass)
- ⚠️ Minor optimizations possible (10-20% improvement)
- 🔴 Primary bottleneck is NOT in constraints (only 4% of RL time)

### [02_OPTIMIZATION_STRATEGIES.md](./02_OPTIMIZATION_STRATEGIES.md)
**Concrete optimization proposals with code examples**

**Phase 1: Quick Wins (1-2 days)**
- Convert qualification lists to sets (O(S × I) → O(S))
- Remove unnecessary sorting (O(Q log Q) → O(Q))
- Add fitness caching for duplicates
- **Expected Impact:** 15-20% speedup

**Phase 2: Delta Evaluation (1 week)**
- Incremental constraint checking
- Track changes between individuals
- Only re-evaluate affected constraints
- **Expected Impact:** 5-10× speedup for mutations

**Phase 3: Advanced (2-3 weeks)**
- Parallel constraint evaluation (2-4× speedup)
- Interval trees for temporal indexing
- Cython compilation (2-3× speedup)

### [03_BENCHMARK_GUIDE.md](./03_BENCHMARK_GUIDE.md)
**Complete guide to benchmarking and profiling**

**Includes:**
- Benchmark script usage and examples
- Profiling with cProfile, py-spy, pyinstrument
- Line-by-line profiling with line_profiler
- Memory profiling with memray
- Before/after comparison methodology
- CI/CD integration examples

---

## 🚀 Quick Start

### Run Benchmark

```bash
# Basic benchmark
uv run python scripts/bench_constraint_check.py

# With custom parameters
python scripts/bench_constraint_check.py --pop 100 --runs 100 --output results.json

# Compare datasets
python scripts/bench_constraint_check.py --compare
```

### Profile Constraint Checking

```bash
# Call graph profiling
python -m cProfile -o profile.out scripts/bench_constraint_check.py
python -m pstats profile.out

# Flame graph
py-spy record -o flamegraph.svg -- python scripts/bench_constraint_check.py

# Interactive HTML report
pyinstrument -r html -o report.html scripts/bench_constraint_check.py
```

---

## 📊 Key Results

### Current Performance (Medium Dataset)

| Component | Sessions | Time | Percentage |
|-----------|----------|------|------------|
| Decode | 150 | 2ms | 5% |
| Hard Constraints | 150 | 12ms | 30% |
| Soft Constraints | 150 | 26ms | 65% |
| **Total** | **150** | **~40ms** | **100%** |

### Constraint Complexity Summary

| Constraint | Complexity | Time | Status |
|------------|-----------|------|---------|
| `student_group_exclusivity` | O(S × Q) | 2-5ms | ✅ Optimal |
| `instructor_exclusivity` | O(S × Q) | 2-5ms | ✅ Optimal |
| `room_exclusivity` | O(S × Q) | 2-5ms | ✅ Optimal |
| `instructor_qualifications` | O(S × I) | 1-3ms | ⚠️ Minor opt |
| `course_completeness` | O(S × G + C × G) | 2-5ms | ✅ Optimal |
| `student_schedule_compactness` | O(S × Q + G × D × Q_d) | 3-8ms | ⚠️ Minor opt |
| `session_continuity` | O(S × Q + C × D × Q_d × log(Q_d)) | 5-15ms | ⚠️ Remove sort |

### Optimization Potential

| Optimization | Effort | Impact | Risk |
|-------------|--------|--------|------|
| Lists → Sets | Low | +2ms (5%) | Very Low |
| Remove Sorting | Low | +5-10ms (12-25%) | Very Low |
| Delta Evaluation | High | 5-10× (mutations) | Medium |
| Parallel Eval | Medium | 2-4× | Medium |

---

## 🎯 Recommendations

### Immediate Actions

1. ✅ **Profile RL environment** - constraint evaluation is only 4% of time
   - Focus on state encoding, action mapping overhead
   - RL training: 1000ms/step, evaluation: 40ms/step
   
2. ✅ **Implement Phase 1 optimizations** - low-hanging fruit
   - Convert lists to sets in Course entity
   - Remove sorting in soft constraints
   - Add basic fitness caching

### Medium-Term

3. **Delta evaluation for mutations** - highest ROI
   - Implement `compute_delta()` and delta-aware evaluation
   - Integrate with GA mutation operators
   - Target 5-10× speedup for typical mutations

### Long-Term Research

4. **Constraint propagation** (CP-SAT inspired)
5. **Spatial indexing** (R-trees, interval trees)
6. **Compiled extensions** (Cython)

---

## 📈 Scaling Analysis

### Time Complexity Growth

```
Dataset     Sessions    Total Time    Per Individual
Small       50          15ms          15ms
Medium      150         40ms          40ms
Large       300         90ms          90ms
X-Large     500         180ms         180ms

Growth rate: ~O(S^1.3) empirical (sub-quadratic)
```

### Population Impact

```
Population  Medium Dataset  Large Dataset
50          2s/gen          4.5s/gen
100         4s/gen          9s/gen
200         8s/gen          18s/gen

Full prod run (200 pop × 2000 gens):
  Medium: 4,000s = 67 min
  Large:  36,000s = 600 min = 10 hours
```

---

## 🔬 For Stack Overflow / External Review

### Short Version (Quick Ask)

See: **Prompts in `prompt.md`** Section 1

**Title:** "Algorithmic complexity analysis for constraint checking in DEAP timetabling engine"

**Summary:** DEAP timetabling with ~150 sessions, constraint checking takes 40ms. Already using hash maps for O(S × Q) conflict detection. Looking for incremental/delta evaluation patterns for RL per-step evaluation.

### Detailed Version (Full Review)

See: **Prompts in `prompt.md`** Section 2

**Request:** Full complexity analysis + optimization plan with:
- Big-O per constraint
- Delta evaluation code sketch
- Benchmark harness
- Unit test outline

### For LLMs (Deep Analysis)

See: **Prompts in `prompt.md`** Section 3

---

## 📝 Related Documents

**In This Directory:**
- `01_COMPLEXITY_ANALYSIS.md` - Complete Big-O analysis
- `02_OPTIMIZATION_STRATEGIES.md` - Optimization proposals with code
- `03_BENCHMARK_GUIDE.md` - Profiling and benchmarking guide

**Project Documentation:**
- `docs/QUICKREF.md` - General project quick reference
- `docs/PROD_RUN_GUIDE.md` - Production run guidance
- `docs/PHASE_2.1_SUMMARY.md` - RL environment implementation
- `.github/copilot-instructions.md` - Project structure and guidelines

**Implementation:**
- `src/constraints/hard.py` - Hard constraint implementations
- `src/constraints/soft.py` - Soft constraint implementations
- `src/constraints/registry.py` - Constraint registration system
- `src/ga/evaluator/fitness.py` - Main evaluation function
- `scripts/bench_constraint_check.py` - Benchmark script

---

## 🧪 Testing Optimizations

After implementing optimizations:

1. **Run unit tests** - ensure correctness
   ```bash
   uv run pytest test/constraints/
   ```

2. **Benchmark before/after**
   ```bash
   python scripts/bench_constraint_check.py --runs 100 --output baseline.json
   # ... make changes ...
   python scripts/bench_constraint_check.py --runs 100 --output optimized.json
   python scripts/compare_benchmarks.py baseline.json optimized.json
   ```

3. **Validate with prod run**
   ```bash
   uv run test  # Quick validation
   uv run prod  # Full production run
   ```

---

## 📞 Contact / Questions

For questions about this analysis:
1. Review documents in this directory
2. Check related project documentation
3. Run benchmarks to reproduce results
4. Refer to source code comments

---

**Note:** This analysis is based on the current implementation as of November 17, 2025. Future optimizations and refactoring may change these results. Always benchmark your specific use case!
