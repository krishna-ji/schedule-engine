# Vectorization Plan

**Date:** 2026-02-19
**Problem:** 549 events, 75 rooms, 189 instructors, 42 quanta, 92 groups
**Baseline run:** pop=200, gens=50 (estimated ~55 min wall time)

---

## 1  Profiling Results (Micro-Benchmarks)

| Component | pop=200 time | per-ind | % of gen | Notes |
|---|---|---|---|---|
| Hard eval (vectorized) | 0.031 s | 0.15 ms | 0.05% | numpy bincount, **already fast** |
| Hard eval (batch) | 0.134 s | 0.67 ms | 0.22% | bitset loops, **redundant path** |
| **Repair (bitset)** | **61.3 s** | **307 ms** | **99.7%** | per-ind Python loop, **#1 target** |
| Full `_evaluate()` | 0.008 s | 0.04 ms | 0.01% | vectorized hard, no soft (ctx=None) |
| Soft eval (OOP) | N/A in hot path | ~50 ms | 0% | only when ctx provided |
| Crossover | ~10 ms | — | < 0.1% | event-block swap |
| Mutation | ~15 ms | — | < 0.1% | local perturbation |
| Constructive sampling | 27.5 s | 137 ms | one-time | initial population only |

**Hard eval vectorized vs batch:** 4.3× speedup, numerically exact.

### Generation Time Breakdown (estimated, pop=200)

```
Repair:      61.3 s  (99.7%)
Hard eval:    0.03 s  ( 0.05%)
Crossover:    0.01 s
Mutation:     0.02 s
Selection:   ~0.01 s
─────────────────────
Total:       ~61.4 s/gen  ×  50 gens  =  ~51 min
```

**Conclusion:** Repair dominates so thoroughly that vectorizing evaluation
gives negligible end-to-end improvement. The highest-impact work is
(1) repair analysis vectorization, (2) eliminating the redundant batch
evaluator, (3) adding a vectorized soft evaluator for when ctx is used.

---

## 2  Existing Evaluation Pipelines (Dedup Needed)

| Module | API | Used by | Status |
|---|---|---|---|
| `fast_evaluator.py` | `fast_evaluate_hard()` scalar | Tests only | **Dead code** — delete |
| `fast_evaluator_batch.py` | `fast_evaluate_hard_batch(X,data)->G` | Fallback path | **Redundant** — vectorized is 4× faster |
| `fast_evaluator_vectorized.py` | `fast_evaluate_hard_vectorized(X,vdata)->G` | Default hot path | **Canonical** |
| `constraints/evaluator.py` | `Evaluator.fitness_from_timetable(tt)` | `_evaluate_soft()` | OOP soft eval |

**Action:** Keep vectorized as canonical hard eval. Build vectorized soft
eval. Remove batch evaluator from hot path (retain for testing).

---

## 3  Phased Plan

### Phase A — Hard Evaluator Canonicalization (Low risk, 1 day)

**Impact:** Remove confusion, ensure single path, ~0 speedup (already vectorized).

- [x] Verify `fast_evaluate_hard_vectorized` is numerically identical to batch (✓ micro-bench)
- Make `SchedulingProblem._evaluate()` always use vectorized (remove `vectorized` flag)
- Add `eval_hard_batch(X) -> G` canonical API with shape/dtype assertions
- Add equivalence test in `tests/test_vectorized_equivalence.py`
- Deprecation warning on `fast_evaluate_hard_batch` import

### Phase B — Soft Evaluator Vectorization (Medium risk, 2-3 days)

**Impact:** Currently 0% of hot path (ctx=None default), but **critical when
soft constraints are enabled** (~50ms/ind → 10s/gen for pop=200).

Port top 3 soft constraints to numpy:

1. **StudentScheduleCompactness** — group-day gap penalty via np.diff on sorted quanta
2. **InstructorScheduleCompactness** — same pattern over instructor-day
3. **StudentLunchBreak** — boolean mask intersection

Define `eval_soft_batch(X) -> S` shape `(N,)` float64.

### Phase C — Repair Substep Vectorization (High risk, 3-5 days)

**Impact:** **99.7% of generation time.** Even 2× speedup saves ~25 min/run.

Current repair flow (per individual):

1. `_fix_domains()` — clamp to allowed sets (per-event, **vectorizable**)
2. `_fix_conflicts_incremental()` — 8 passes, conflict detection + re-placement
3. `_fix_group_conflicts()` — 4 rounds, deconflict by group

**Vectorizable substeps:**

- **Conflict detection** (currently `_make_counts()` builds count arrays per individual)
  → Build 3D count tensors `room_cnt[N, R, T]`, `inst_cnt[N, I, T]`, `grp_cnt[N, G, T]`
  for the entire population at once using `np.add.at` or `np.bincount`.
  Then `conflict_events = np.where(cnt > 1)` identifies which events to fix.
- **Domain clamping** (`_fix_domains`) → vectorized `np.isin` + `np.searchsorted`
  over population matrix.

**Non-vectorizable (keep scalar):**

- Move application in `_find_placement()` — inherently sequential per-event
  (each placement changes the count arrays for subsequent events).

**Strategy:** Vectorize analysis (find conflicts for all individuals at once),
then apply moves per-individual but with pre-computed conflict sets.

Define `repair_batch(X) -> X_repaired` shape `(N, 3E)`.

### Phase D — Metrics (Low risk, 1 day)

**Impact:** Not in GA hot path, but used by experiment callbacks and
post-processing. Already using pymoo/scipy internals.

- Wrap pymoo `HV`, `IGD`, `GD`, `NonDominatedSorting` into arrays-only API
- `metrics_batch(F) -> dict[str, float]` with `hv`, `igd`, `spacing`, `n_fronts`
- Remove DEAP `fitness.values` dependency in metrics module
- Pure numpy input/output

---

## 4  Risk Assessment

| Phase | Risk | Reason |
|---|---|---|
| A | Low | Removing redundant code, adding asserts |
| B | Medium | New evaluation kernel, must match OOP exactly |
| C | High | Repair correctness is critical; sequential nature |
| D | Low | Wrapping existing pymoo indicators |

---

## 5  Gate Criteria (per phase)

1. `pytest` passes (exclude RL tests)
2. Batch vs scalar equivalence: hard constraints **exact**, soft **≤ 1e-6** tolerance
3. Benchmark shows ≥2× speedup for that phase's target (or proves no regression)
4. No new duplicate pipelines/modules
5. Updated `docs/VECTORIZATION_PROGRESS.md`

---

## 6  ETA

| Phase | Estimated | Cumulative |
|---|---|---|
| A — Hard eval canonical | 0.5 day | 0.5 day |
| B — Soft eval vectorized | 2 days | 2.5 days |
| C — Repair analysis vectorized | 3 days | 5.5 days |
| D — Metrics arrays-only | 0.5 day | 6 days |

---

## 7  Files Affected

```
src/pipeline/scheduling_problem.py     — remove vectorized flag, add batch API
src/pipeline/fast_evaluator.py         — deprecate or delete
src/pipeline/fast_evaluator_batch.py   — deprecate (keep for equivalence tests)
src/pipeline/fast_evaluator_vectorized.py — canonical, add asserts
src/pipeline/repair_operator_bitset.py — vectorize conflict detection
src/pipeline/soft_evaluator_vectorized.py — NEW: vectorized soft eval
src/pipeline/batch_api.py             — NEW: canonical batch API entry points
src/ga/metrics/batch_metrics.py       — NEW: arrays-only metrics wrapper
tests/test_vectorized_equivalence.py  — NEW: batch vs scalar tests
tests/test_batch_api.py              — NEW: shape/dtype contract tests
```
