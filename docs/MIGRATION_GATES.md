# DEAP → pymoo Migration: GO/NO-GO Gates & Full Migration

## Status: FULL MIGRATION — pymoo is DEFAULT

| Milestone | Status | Evidence |
| --------- | ------ | -------- |
| Encoding roundtrip | PASS | `tests/test_migration_gates.py::TestEncodingRoundtrip` |
| Evaluator equivalence | PASS | `tests/test_migration_gates.py::TestEvaluatorEquivalence` |
| Offspring repair | PASS | `test_repair_offspring.py` — 100% at k=5,10,20,50 |
| PKL integrity guards | PASS | Schema v2, data hash, event key recomputation |
| Structural infeasibility fix | PASS | 0 events with empty room domains |
| pymoo Problem + operators | PASS | Smoke test, cv_min dropping |
| Vectorized hard evaluator | PASS | 0.162 ms/ind, 5.92× speedup, 0 mismatches |
| Benchmark tooling | PASS | `bench_compare.py` + `report_bench.py` |
| Default solver flipped | DONE | `solve.py --solver pymoo` (default) |
| Kill-switch | DONE | `SCHED_SOLVER=deap` env var, `--solver deap` CLI |
| Solver metadata in output | DONE | `data_hash`, `schema_version`, `solver`, timestamps |
| DEAP deprecation warning | DONE | `DeprecationWarning` when `--solver deap` |

---

## Gate Definitions

### Gate 1 — Equivalence Tests Pass

```bash
pytest tests/test_migration_gates.py -v
```

**Pass criterion:** 0 failures, 0 errors.

### Gate 2 — Solution Quality (disjunctive)

| Sub-gate | Metric                    | Rule                                                |
| -------- | ------------------------- | --------------------------------------------------- |
| 2a       | `median_final_best_soft`  | pymoo ≤ DEAP across N seeds                         |
| 2b       | `median_time_to_feasible` | pymoo reaches `best_hard == 0` at earlier/equal gen |

At least **one** of 2a or 2b must hold.

### Gate 3 — No Uncompensated Runtime Regression

**Primary rule:** pymoo `sec/gen` ≤ 2× DEAP `sec/gen` (median across seeds).

**Compensation clause:** If pymoo is >2× slower per-gen but achieves ≥2×
better hard violations (i.e. `pymoo_hard < 0.5 × deap_hard`), the runtime
penalty is considered compensated by quality gains. This follows the user
rule: *"pymoo is >2× slower per generation **without** compensating
quality/feasibility gains"*.

### Composite

```text
GO   = Gate1 AND (Gate2a OR Gate2b) AND Gate3
NO-GO = anything else
```

---

## Running the Gate Check

```bash
# Gate 1
pytest tests/test_migration_gates.py -v

# Gates 2 & 3
python bench_compare.py --gens 50 --pop 50 --seeds 5
python report_bench.py

# Review
cat results/bench_compare/verdict.json
```

---

## Kill-Switch

If a regression is found, revert instantly without any code change:

```bash
# Option 1: env var (no code change needed)
SCHED_SOLVER=deap python solve.py --gens 100 --pop 50

# Option 2: CLI flag
python solve.py --solver deap --gens 100 --pop 50

# Option 3: config (instance_config.py)
# Change DEFAULT_SOLVER = "deap"
```

All three mechanisms are live. Priority: CLI > env var > config > "pymoo".

---

## Output Metadata

Every `solve.py --output results/run.json` now includes:

```json
{
  "solver": "pymoo",
  "solve_version": "2.0.0",
  "timestamp": "2026-02-19T06:16:55.304992+00:00",
  "data_hash": "97624942a61825a1...",
  "schema_version": 2,
  "n_events": 549,
  "config": {"gens": 100, "pop": 50, "seed": 42},
  "best_hard": 0,
  "best_soft": 1234,
  "elapsed_s": 45.2,
  "sec_per_gen": 0.452
}
```

This enables post-hoc comparison of runs across solvers and data versions.

---

## Parallel Validation Protocol

For the first 1-2 weeks after the default flip:

1. **Primary runs** use pymoo (the default).
2. **Shadow runs** periodically re-run with `--solver deap` on same seeds.
3. Compare via `bench_compare.py` + `report_bench.py`.
4. If any `verdict.json` says NO-GO on realistic configs, flip back immediately.

---

## DEAP Deprecation Timeline

| Phase | Action | Trigger |
| --- | --- | --- |
| Now | DEAP emits `DeprecationWarning` | Automatic |
| +2 weeks | Mark DEAP code with `# DEPRECATED` comments | After confidence |
| +4 weeks | Remove DEAP from `solve.py` (optional) | After no regressions |
| Never | Delete `src/ga/`, `src/experiments/` | Keep as reference |

---

## Escalation

If `report_bench.py` says NO-GO:

1. Check which sub-gate failed.
2. If runtime: profile evaluator with `bench_eval_vectorized.py`.
3. If quality: increase `--gens`/`--pop` and rerun.
4. Rollback: `SCHED_SOLVER=deap` — instant, no code change.
