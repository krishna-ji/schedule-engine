# Vectorization Progress

**Status:** All 4 phases COMPLETE  
**Date:** 2025-01-XX  
**Test suite:** 669 passed, 2 pre-existing failures (unrelated)

---

## Profiling Results (Baseline)

| Component | Time (pop=200) | % of gen |
|---|---|---|
| Hard eval (vectorized) | 0.031s | 0.05% |
| Hard eval (batch/OOP) | 0.134s | 0.22% |
| Soft eval (OOP loop) | ~10s est | ~16% |
| **Repair** | **61.3s** | **99.7%** |
| Sampling | 27.5s (one-time) | — |

**Key finding:** Repair dominates the GA loop at 0.307s/individual.
Evaluation vectorization gives huge *relative* speedups but limited
end-to-end impact because repair is the bottleneck.

---

## Phase A — Hard Evaluator Canonicalization ✅

**Goal:** Remove dual evaluation paths, always use vectorized hard eval.

**Changes:**

- Removed `vectorized` flag from `SchedulingProblem.__init__`
- Removed `fast_evaluator_batch` import from `scheduling_problem.py`
- `_evaluate()` now always calls `fast_evaluate_hard_vectorized`
- Added deprecation notice to `fast_evaluator_batch.py`
- Created `src/pipeline/batch_api.py` with `BatchContext` and 4 entry points

**Results:**

| Metric | Value |
|---|---|
| Hard eval vectorized | 0.032s (pop=200) |
| Hard eval batch (deprecated) | 0.128s (pop=200) |
| **Speedup** | **4.06×** |
| Equivalence | EXACT_MATCH |
| Tests | 11/11 passed (test_batch_api.py) |
| Benchmark | results/bench_phase_a.json |

---

## Phase B — Vectorized Soft Evaluator ✅

**Goal:** Replace per-individual OOP Timetable→Evaluator loop for the top 3
soft constraints with a fully vectorized numpy kernel.

**Constraints implemented:**

1. `StudentScheduleCompactness` — gap penalty per group per day
2. `InstructorScheduleCompactness` — gap penalty per instructor per day
3. `StudentLunchBreak` — free quanta in lunch window per group per day

**Algorithm:** np.bincount scatter into 4D occupancy tensor
`(N, entity, day, quantum)`, then direct range-mask gap computation
with midday break exclusion.

**Key fix:** Day-boundary clamping — the OOP `SessionGene.__post_init__`
clamps events that would spill past end-of-day back into the starting day.
The vectorized kernel must replicate this to match OOP behaviour.

**Changes:**

- Created `src/pipeline/soft_evaluator_vectorized.py`
- Wired into `batch_api.py` (`eval_soft_batch`)
- Wired into `scheduling_problem.py` (`_evaluate` uses vectorized soft eval)

**Results:**

| Metric | Value |
|---|---|
| Soft eval vectorized | 12.92ms (pop=50) |
| Soft eval OOP loop | 257.74ms (pop=50) |
| **Speedup** | **19.95×** |
| Equivalence | EXACT_MATCH (atol=1e-6) |
| Tests | 6/6 passed (test_soft_eval_vectorized.py) |
| Benchmark | results/bench_phase_b.json |

---

## Phase C — Repair Analysis Vectorization ✅

**Goal:** Vectorize the analysis substeps of repair (domain clamping,
occupancy count construction, conflict detection) across the full population.
The actual placement search remains sequential per-individual.

**Changes:**

- Created `src/pipeline/repair_analysis_vectorized.py` with:
  - `fix_domains_batch()` — vectorized domain clamping with instructor availability
  - `build_counts_batch()` — 3D count arrays via np.add.at scatter
  - `count_conflicts_batch()` — batch conflict detection
  - `repair_summary_batch()` — population-level analysis
- Updated `batch_api.py` `repair_batch()` to use vectorized domain fix

**Results:**

| Operation | Batch (ms) | Loop (ms) | Speedup |
|---|---|---|---|
| build_counts | 12.21 | 84.43 | **6.92×** |
| count_conflicts | 24.97 | 117.28 | **4.70×** |
| fix_domains | exact match | — | — |
| Tests | 5/5 passed | | |
| Benchmark | results/bench_phase_c.json | | |

**Note:** The placement search (`_find_placement`) accounts for ~70% of
per-individual repair time and is inherently sequential (each placement
changes the conflict landscape for subsequent events). Full vectorization
would require a wavefront/batch-placement architecture, which is a significant
rewrite beyond the scope of this release.

---

## Phase D — Metrics Arrays-Only ✅

**Goal:** Ensure metrics API uses pure numpy input/output, wrapping
pymoo/scipy C-backed indicators.

**Changes:**

- `metrics_batch(F, ref_point)` in `batch_api.py` — already implemented
- Uses pymoo `HV`, `IGD`, `NonDominatedSorting` and scipy `pdist`
- Created `tests/test_metrics_batch.py` with 6 tests

**Results:**

| Metric | Value |
|---|---|
| metrics_batch | 2.27ms (pop=200) |
| Tests | 6/6 passed |
| Benchmark | results/bench_phase_d.json |

---

## Files Created / Modified

### New files

| File | Purpose |
|---|---|
| `src/pipeline/batch_api.py` | Canonical batch API (eval_hard, eval_soft, repair, metrics) |
| `src/pipeline/soft_evaluator_vectorized.py` | Vectorized soft eval for top 3 constraints |
| `src/pipeline/repair_analysis_vectorized.py` | Vectorized repair analysis (counts, conflicts, domains) |
| `docs/VECTORIZATION_PLAN.md` | Profiling results + phased plan |
| `docs/VECTORIZATION_PROGRESS.md` | This file |
| `tests/test_batch_api.py` | Phase A tests (11 tests) |
| `tests/test_soft_eval_vectorized.py` | Phase B tests (6 tests) |
| `tests/test_repair_analysis_vectorized.py` | Phase C tests (5 tests) |
| `tests/test_metrics_batch.py` | Phase D tests (6 tests) |
| `scripts/micro_bench.py` | Micro-benchmarking script |
| `scripts/debug_soft_eval.py` | Diagnostic script (debug only) |
| `results/bench_phase_a.json` | Phase A benchmark data |
| `results/bench_phase_b.json` | Phase B benchmark data |
| `results/bench_phase_c.json` | Phase C benchmark data |
| `results/bench_phase_d.json` | Phase D benchmark data |

### Modified files

| File | Change |
|---|---|
| `src/pipeline/scheduling_problem.py` | Removed `vectorized` flag, wired soft eval, always vectorized |
| `src/pipeline/fast_evaluator_batch.py` | Added deprecation notice |

---

## Test Summary

| Test file | Tests | Status |
|---|---|---|
| test_batch_api.py | 11 | ✅ All pass |
| test_soft_eval_vectorized.py | 6 | ✅ All pass |
| test_repair_analysis_vectorized.py | 5 | ✅ All pass |
| test_metrics_batch.py | 6 | ✅ All pass |
| **Full suite** | **669** | **✅ Pass** (2 pre-existing failures) |

---

## Speedup Summary

| Component | Before | After | Speedup |
|---|---|---|---|
| Hard eval (pop=200) | 0.128s (batch) | 0.032s (vec) | **4.06×** |
| Soft eval (pop=50) | 257.7ms (OOP) | 12.9ms (vec) | **19.95×** |
| build_counts (pop=50) | 84.4ms (loop) | 12.2ms (batch) | **6.92×** |
| count_conflicts (pop=50) | 117.3ms (loop) | 25.0ms (batch) | **4.70×** |
| metrics (pop=200) | — | 2.3ms | — |

---

## Remaining Bottleneck

Repair placement (`_find_placement`) at 0.307s/individual (61.3s for pop=200)
remains the dominant bottleneck at 99.7% of generation time. This is
inherently sequential — each event placement changes the conflict landscape
for subsequent placements. Potential future approaches:

1. **C extension** — move the inner placement loop to C/Cython
2. **Multiprocessing** — repair individuals in parallel (GIL-free)
3. **Approximation** — reduce repair passes (currently up to 8+4 rounds)
4. **Smarter initialization** — better constructive heuristic to start closer to feasible
