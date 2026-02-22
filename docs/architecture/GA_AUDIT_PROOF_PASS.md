# GA Architecture Audit — Proof Pass Baseline

> **Date**: 2026-02-22  
> **Branch**: `feat/pymoo-only`  
> **Scope**: `runs/ga_01..05`, `src/experiments/{base,ga_experiment,moea_metrics}.py`

---

## 1. Mode-File Verdicts

All 5 run-mode files are **THIN** entry points.

| File | Lines | Imports (project) | Experiment class | Verdict |
|---|---|---|---|---|
| `runs/ga_01_baseline.py` | 55 | `BaselineExperiment` (L16) | L39–L49 + `exp.run()` L50 | **THIN** |
| `runs/ga_02_memetic.py` | 64 | `MemeticExperiment` (L19) | L46–L58 + `exp.run()` L59 | **THIN** |
| `runs/ga_03_aggressive.py` | 65 | `AggressiveExperiment` (L20) | L47–L59 + `exp.run()` L60 | **THIN** |
| `runs/ga_04_adaptive.py` | 69 | `AdaptiveExperiment` (L20) | L49–L63 + `exp.run()` L64 | **THIN** |
| `runs/ga_05_cp_hybrid.py` | 67 | `CPHybridExperiment` (L22) | L49–L61 + `exp.run()` L62 | **THIN** |

**Forbidden patterns searched** (all 0 matches across all 5 files):  
`minimize(`, `NSGA2(`, `Callback`, `src.pipeline`, `plot_`, `export_everything`, `generate_`, `DataStore.from_json`, `decode_`

---

## 2. Top-10 Symbol Spans (ga_experiment.py, pre-refactor)

| # | Symbol | Start | End | Length | Indent |
|---|---|---|---|---|---|
| 1 | `class GAExperiment` | 30 | 586 | 557 | 0 |
| 2 | `def _generate_outputs` | 141 | 335 | 195 | 4 |
| 3 | `class CPHybridExperiment` | 961 | 1115 | 155 | 0 |
| 4 | `def _build_callback` (CPHybrid) | 985 | 1115 | 131 | 4 |
| 5 | `def _execute` | 459 | 586 | 128 | 4 |
| 6 | `class CB` (CPHybrid inner) | 992 | 1115 | 124 | 8 |
| 7 | `class AdaptiveExperiment` | 860 | 960 | 101 | 0 |
| 8 | `def _export_schedule_pdfs` | 336 | 413 | 78 | 4 |
| 9 | `def _build_callback` (Adaptive) | 889 | 960 | 72 | 4 |
| 10 | `class MemeticExperiment` | 732 | 797 | 66 | 0 |

---

## 3. Callback Duplication Proof (pre-refactor)

### Duplicated `__init__` body (5 copies)

| Mode | Lines | Extra fields |
|---|---|---|
| baseline | L690–L698 | — |
| memetic | L766–L774 | — |
| aggressive | L830–L838 | — |
| adaptive | L903–L913 | `_stagnant`, `_escalated` |
| cp_hybrid | L993–L1005 | `_pkl_data`, `_ctx`, `_cp_pipeline`, `_initialised` |

Common block (identical in all 5):

```python
self.best_hards: list[float] = []
self.best_softs: list[float] = []
self.best_breakdowns: list[dict[str, int]] = []
self.gen_times: list[float] = []
self._gen_t0: float = time.time()
_init_moea_lists(self)
```

### Duplicated `notify` core (5 copies)

| Mode | Lines |
|---|---|
| baseline | L700–L708 |
| memetic | L776–L784 |
| aggressive | L840–L848 |
| adaptive | L915–L924 |
| cp_hybrid | L1034–L1042 |

Common block (identical except adaptive names `cur_hard`):

```python
now = time.time()
self.gen_times.append(now - self._gen_t0)
self._gen_t0 = now
F, G, cv, best_idx = _log_gen(algorithm, log_interval)
self.best_hards.append(float(F[best_idx, 0]))
self.best_softs.append(float(F[best_idx, 1]))
self.best_breakdowns.append(_constraint_breakdown(G[best_idx]))
_record_moea_metrics(self, algorithm, F, G)
```

---

## 4. Refactor Plan Tiers

### Tier 1 — Immediate safe fixes

- Extract shared callback base class (deduplicate 5× init + notify)
- Replace `print()` in callbacks with logger where possible
- Move `_generate_outputs` + `_export_schedule_pdfs` to output pipeline module

### Tier 2 — Medium refactors

- Strategy objects for mode behavior (repair/escalation/CP polish)
- Split `_execute` into staged methods (prepare / build / optimize / summarize / output)
- Config dataclasses (`GAParams`, `OutputConfig`, `MetricsConfig`)

### Tier 3 — Risky / high-impact

- Break `GAExperiment` into orchestrator + collaborators
- Extract CP bridge logic to dedicated module
- Tighten exception policy (structured error metadata)

---

## 5. Key SE Findings

| Finding | Location | Severity |
|---|---|---|
| God class tendency | `GAExperiment` L30–L586 (557 lines, 7+ responsibilities) | Medium |
| Side effects in `_generate_outputs` | L141–L335 (195 lines, 12+ `__import__` calls) | Medium |
| Side effects in `_execute` | L475–L482 (mutates `course.specific_lab_features`) | Low |
| Silent exception swallowing | `_safe_call` L414–L420, CP callback L1068–L1069 | Low |
| Nested CB classes untestable | 5 inner classes in `_build_callback` methods | Medium |
| Boolean flag soup (early) | `export_pdf` + `force_pdf` L64–L65 | Low |

---

*This document is a frozen baseline. Compare after refactors to verify no behavioral regression.*
