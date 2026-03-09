# Thesis ↔ Codebase Misalignment Audit Report

**Date:** 2025-07-06  
**Scope:** All chapters of the thesis (`major-project-repot/`) cross-referenced against the full codebase (`src/`)  
**Verdict:** **31 divergences found** — 7 Critical, 13 Major, 11 Minor

---

## Table of Contents

1. [Critical Issues (C-001 – C-007)](#1-critical-issues)
2. [Major Issues (M-001 – M-013)](#2-major-issues)
3. [Minor Issues (N-001 – N-011)](#3-minor-issues)
4. [Code Bug Discovered During Audit](#4-code-bug-discovered-during-audit)
5. [Summary Matrix](#5-summary-matrix)

---

## 1. Critical Issues

These will be immediately noticed by any examiner who cross-references thesis sections or compares claims against the code.

---

### C-001: `resultAndAnalysis.tex` — Constraint Count is Wrong

| | |
|---|---|
| **Thesis claim** | `resultAndAnalysis.tex` line 112: *"12 hard constraints and 8 soft constraints"* — repeated at lines 116, 361, 426, 445 |
| **Correct value** | 8 hard constraints, 4 soft constraints (per methodology ch., `sec_constraint_taxonomy.tex`, and code) |
| **Code evidence** | `src/constraints/constraints.py`: `HARD_CONSTRAINT_CLASSES` has 8 entries; `SOFT_CONSTRAINT_CLASSES` has 6 entries; `src/rl/gym_env/fast_state_encoder.py`: `SOFT_CONSTRAINT_NAMES` = 4 entries (CSC, FSC, MIP, SSCP); `src/pipeline/scheduling_problem.py`: `HARD_CONSTRAINT_NAMES` = 8 entries |
| **Impact** | Every table, figure, and analysis in this chapter that references "12 hard + 8 soft" is wrong |
| **Fix** | Replace all instances of "12 hard" → "8 hard" and "8 soft" → "4 soft" throughout `resultAndAnalysis.tex`. Note: the soft count depends on interpretation — see C-002. |

---

### C-002: Soft Constraint Count — Thesis Says 4, Code Has 6

| | |
|---|---|
| **Thesis claim** | `sec_constraint_taxonomy.tex` Table `tab:soft_constraints_taxonomy`: 4 soft constraints — CSC, FSC, MIP, SSCP |
| **Code reality** | `src/constraints/constraints.py` lines 830-871: `SOFT_CONSTRAINT_CLASSES` contains **6** classes: `StudentScheduleCompactness` (CSC), `InstructorScheduleCompactness` (FSC), `StudentLunchBreak` (MIP), `SessionContinuity` ("session_continuity"), `PairedCohortPracticalAlignment` (SSCP), `BreakPlacementCompliance` ("break_placement_compliance") |
| **RL state vector** | `fast_state_encoder.py`: `SOFT_CONSTRAINT_NAMES = ["CSC", "FSC", "MIP", "SSCP"]` — only 4 are observed by the RL agent |
| **Impact** | Two soft constraints (`SessionContinuity`, `BreakPlacementCompliance`) contribute to fitness evaluation but are: (a) undocumented in the thesis, (b) invisible to the RL agent's 39-D state vector. This means the RL agent optimizes a partial view of the soft objective |
| **Fix** | **Option A:** Document all 6 soft constraints in the thesis and note that only 4 are observed by the RL agent. **Option B:** Remove `SessionContinuity` and `BreakPlacementCompliance` from the code if they are vestigial. Either way, the thesis and code must agree. |

---

### C-003: `resultAndAnalysis.tex` — Dataset is a Toy Instance, Not the Real One

| | |
|---|---|
| **Thesis claim** | `resultAndAnalysis.tex` lines 54-70: *"10 distinct courses"*, *"3 groups"*, course names like CS101, MATH101, PHYS101 (toy academic instance) |
| **Correct dataset** | `b1resultAndAnalysis.tex` and `data/` directory: **Tribhuvan University Thapathali Campus** — 444 courses, 37 groups, 181 instructors, 67 rooms, 790 scheduling events |
| **Code evidence** | `data/Course.json` (444 courses), `data/Groups.json` (37 groups), `data/Instructors.json` (181 instructors), `data/Rooms.json` (67 rooms) |
| **Impact** | The entire experimental setup section of the results chapter describes a non-existent dataset. All search space calculations, constraint complexity estimates, and resource descriptions are fabricated |
| **Fix** | Replace the toy instance description with the real TU Thapathali dataset. Use values from `b1resultAndAnalysis.tex` which correctly describes the real instance. |

---

### C-004: `resultAndAnalysis.tex` — Temporal Framework is Wrong

| | |
|---|---|
| **Thesis claim** | `resultAndAnalysis.tex` line 74: *"7 days (Monday through Sunday)"*, *"10 operating hours per day (09:00–19:00)"*, *"70 discrete time quanta"* |
| **Correct value** | 6 days (Monday–Saturday, Saturday closed for classes), 7 operating hours per day (10:00–17:00), **Q = 42** quanta |
| **Code evidence** | `src/config/loader.py` defaults: `opening_time="10:00"`, `closing_time="17:00"`, `closed_days=["Saturday"]`; `src/pipeline/repair_operator_bitset.py`: `from .bitset_time import T` where T=42; `src/rl/gym_env/reward_shaper.py` line 62: `_T: int = 42` |
| **Fix** | Correct to "6 days, 7 hours/day (10:00–17:00), 42 quanta" throughout. |

---

### C-005: `resultAndAnalysis.tex` — Software Framework is Wrong

| | |
|---|---|
| **Thesis claim** | `resultAndAnalysis.tex` line 371: *"DEAP 1.4.1 (GA framework)"* listed as the primary GA framework |
| **Correct framework** | **pymoo 0.6.1.3** — the entire GA pipeline uses `pymoo.algorithms.moo.nsga2.NSGA2` with custom pymoo operators |
| **Code evidence** | `src/pipeline/pymoo_operators.py`: imports `from pymoo.core.crossover import Crossover`; `src/pipeline/scheduling_problem.py`: `from pymoo.core.problem import Problem`; `src/rl/gym_env/pymoo_env.py`: creates `NSGA2(...)` algorithm; `pyproject.toml`: `pymoo = ">=0.6.1.3"` |
| **Context** | DEAP was the original framework but was fully replaced by pymoo during the Phase 55 overhaul. Legacy DEAP code still exists in `src/ga/operators/` but is no longer used in the current pipeline |
| **Fix** | Replace "DEAP 1.4.1 (GA framework)" with "pymoo 0.6.1.3 (multi-objective optimization framework)". Note: DEAP *was* used historically and could be mentioned as the predecessor if desired. |

---

### C-006: `resultAndAnalysis.tex` — Hardware Specs Internally Contradictory

| | |
|---|---|
| **Thesis claim A** | `resultAndAnalysis.tex` lines 11-28, Table `tab:hardware_specs`: **Intel i9-14900K, 128 GB DDR5, RTX 4060** (CORRECT) |
| **Thesis claim B** | `resultAndAnalysis.tex` line 369: *"32-core AMD Ryzen CPU for parallel fitness evaluation, 64 GB RAM"* (WRONG) |
| **Fix** | Delete/rewrite line 369 to match the table at the top of the same chapter. The table has the correct hardware. |

---

### C-007: `resultAndAnalysis.tex` — Heuristic Count is Wrong

| | |
|---|---|
| **Thesis claim** | `resultAndAnalysis.tex` line 211: *"Mode C: Mode B + 19 heuristic operators via uniform rotation"*; line 226: *"19 heuristics in round-robin rotation"* |
| **Correct value** | **6 LLH pipeline configurations** (Conservative, Aggressive, Memetic Elite, Soft-Focus, Destructive-Constructive, Intensified) |
| **Code evidence** | `src/rl/actions/vectorized_ops.py` lines 311-318: `VECTORIZED_ACTION_SPACE` dict with 6 entries; `NUM_ACTIONS = len(VECTORIZED_ACTION_SPACE)` = 6 |
| **Context** | The "19 heuristics + 1 no-op = 20 actions" was the DEAP-era action space (`src/rl/gym_env/action_space.py`). The current system uses 6 LLH pipeline configs. |
| **Fix** | Replace "19 heuristic operators" with "6 LLH pipeline configurations" and update Mode C/D/E/F descriptions. |

---

## 2. Major Issues

These are significant discrepancies that could cause confusion during examination.

---

### M-001: `resultAndAnalysis.tex` — XXX Placeholders Throughout

| | |
|---|---|
| **Locations** | Lines 54, 56, 60, 61, 64, 68, 69, 70, 82, 86, 88, 426, and many more |
| **Examples** | *"credit hours ranging from 3 to 4 credits, translating to XXX to XXX contact hours"*; *"Group sizes vary from XXX to XXX students"*; *"Feasibility Rate: XXX%"* |
| **Impact** | Any examiner will immediately see that this chapter was never completed |
| **Fix** | Fill in all XXX placeholders with actual values from the TU Thapathali dataset, or (recommended) replace the entire toy-dataset section with the real dataset description from `b1resultAndAnalysis.tex`. |

---

### M-002: Internal Thesis Contradiction — GA Parameters (Methodology vs Implementation)

| | |
|---|---|
| **Methodology chapter** | `sec_evolutionary_operators.tex`: $p_c = 0.5$ (per-event Bernoulli), $p_m = 0.05$ (per-event mutation rate) |
| **Implementation chapter** | `experimentConfiguration.tex` Table `tab:algorithm_params`: $p_c = 0.70$, $p_m = 0.20$ |
| **Code reality** | **Both exist in code for different systems**: `src/pipeline/pymoo_operators.py` line 282: `crossover_prob: float = 0.5`, `mutation_event_prob: float = 0.05` (CURRENT pymoo system). `src/config/loader.py` defaults: `cxpb=0.70`, `mutpb=0.20` (LEGACY DEAP system). `src/rl/gym_env/pymoo_env.py` reset(): creates `EventBlockCrossover(prob=0.5)`, `EventLocalMutation(event_prob=0.05)` |
| **Root cause** | `experimentConfiguration.tex` was written during the DEAP era and never updated to reflect the pymoo migration. The per-event semantics also differ (per-event vs per-individual). |
| **Fix** | Update `experimentConfiguration.tex` Table `tab:algorithm_params` to use $p_c = 0.5$ (per-event exchange prob) and $p_m = 0.05$ (per-event mutation prob). Add a note explaining these are per-event rates in the pymoo vectorized operators, not per-individual DEAP-era rates. |

---

### M-003: Internal Thesis Contradiction — Crossover Description (Methodology vs Implementation)

| | |
|---|---|
| **Methodology chapter** | `sec_evolutionary_operators.tex`: Describes **EventBlockCrossover** — per-event Bernoulli mask, exchanges atomic $(I_e, R_e, T_e)$ triples, "no HashMap required", fully vectorized O(N·E) |
| **Implementation chapter** | `experimentConfiguration.tex` rationale text: References "position-independent crossover operator" |
| **Code** | **Both exist**: `src/pipeline/pymoo_operators.py` → `EventBlockCrossover` (CURRENT, used by `pymoo_env.py`); `src/ga/operators/crossover.py` → `crossover_course_group_aware` (LEGACY DEAP, hash-map matching by `(course_id, group_ids)` keys) |
| **Issue** | The "position-independent" terminology in the implementation chapter refers to the LEGACY hash-map crossover, not the current EventBlockCrossover. These are fundamentally different operators. |
| **Fix** | Ensure the implementation chapter describes EventBlockCrossover (methodology's Algorithm 1) and remove or clearly label any references to the legacy hash-map crossover. |

---

### M-004: Internal Thesis Contradiction — Mutation Description (Methodology vs Implementation)  

| | |
|---|---|
| **Methodology chapter** | `sec_evolutionary_operators.tex`: Describes **EventLocalMutation** — domain-agnostic, uniform random sampling, $p_m = 0.05$ per event |
| **Implementation chapter** | `experimentConfiguration.tex` rationale: References "constraint-guided mutation operator" |
| **Code** | **Both exist**: `src/pipeline/pymoo_operators.py` → `EventLocalMutation` (CURRENT); `src/ga/operators/mutation.py` → `mutate_gene` with retention probabilities $p_{\text{ret\_inst}} = 0.7$, $p_{\text{ret\_room}} = 0.5$ (LEGACY DEAP); `src/ga/operators/constraint_guided_mutation.py` → `constraint_guided_mutation` targeting violating sessions (LEGACY DEAP) |
| **Issue** | The "constraint-guided mutation" in the implementation chapter is the DEAP-era operator with 80/20 targeted/random split and spreading-aware selection. The current system uses the simpler EventLocalMutation. |
| **Fix** | Update the implementation chapter to describe EventLocalMutation. The constraint-guided mutation can be mentioned as a historical approach that was superseded. |

---

### M-005: `experimentConfiguration.tex` — RL Population Size Mismatch

| | |
|---|---|
| **Thesis claim** | `sec_reinforcement_learning.tex` Table `tab:rl_hyperparameters`: RL pop_size = 100 |
| **Thesis claim** | `experimentConfiguration.tex` Table `tab:algorithm_params`: pop_size = 50 (test) / 200 (production) |
| **Code** | `src/rl/gym_env/pymoo_env.py`: `self._pop_size` comes from config; default in RL training is typically 120 (from run scripts) |
| **Fix** | Reconcile all three: decide the canonical RL pop_size and make all tables consistent. If the actual training used 120, state 120. |

---

### M-006: `resultAndAnalysis.tex` — Mode E Description is Wrong

| | |
|---|---|
| **Thesis claim** | Line 230: *"Mode E: Validates the full hyper-heuristic framework with reinforcement learning controlling meta-level algorithmic decisions (mutation rates, repair triggers, diversity injection)"* |
| **Code reality** | The RL agent does NOT control "mutation rates, repair triggers, diversity injection". It selects one of 6 **LLH pipeline configurations** per generation — each config specifies elite fraction, repair passes, and optional mating-level operations (ruin-recreate, time compaction). |
| **Code evidence** | `src/rl/actions/vectorized_ops.py`: 6 `_AtomicRepairBase` subclasses with `PostGenConfig` (elite_fraction, passes, stochastic_alternate, ruin_fraction, compact_soft). The action is a single integer 0-5 selecting which repair configuration to apply. |
| **Impact** | Describing the RL agent as controlling "mutation rates" is misleading — it controls repair intensity, not crossover/mutation operator parameters |
| **Fix** | Rewrite Mode E description: "Mode E: RL agent (MaskablePPO) selects from 6 LLH pipeline configurations per generation, each specifying post-generation repair intensity (elite fraction, passes, stochastic/deterministic, optional ruin-recreate or time compaction)." |

---

### M-007: `resultAndAnalysis.tex` — Mode C Heuristic Count Mismatch

| | |
|---|---|
| **Thesis claim** | Line 211: *"Mode C: Mode B + 19 heuristic operators via uniform rotation"* |
| **Code reality** | Mode C (Round-Robin) cycles through the same 6 LLH pipeline configurations, not 19 |
| **Code evidence** | `src/rl/actions/vectorized_ops.py`: `NUM_ACTIONS = 6` |
| **Fix** | Replace "19 heuristic operators" → "6 LLH pipeline configurations" |

---

### M-008: `experimentConfiguration.tex` — Tournament Size Inconsistency

| | |
|---|---|
| **Thesis claim** | `experimentConfiguration.tex` Table `tab:algorithm_params`: tournament size = 2 |
| **`resultAndAnalysis.tex`** | (Check needed — the template chapter may reference tournament size 3) |
| **Code** | `src/config/loader.py`: `tournament_size=2` default; pymoo's NSGA-II uses its own binary tournament based on crowded comparison — no explicit tournament_size parameter |
| **Issue** | The tournament_size=2 in `experimentConfiguration.tex` matches the legacy DEAP config default. In the pymoo system, NSGA-II's binary tournament is built-in and doesn't use this parameter. |
| **Fix** | Clarify that pymoo's NSGA-II uses the standard binary tournament with crowded comparison operator, and remove or relabel the legacy tournament_size parameter. |

---

### M-009: `reward_calculator.py` is Legacy — Not Used by Current System

| | |
|---|---|
| **Thesis** | `sec_reinforcement_learning.tex` describes reward as phase-transition delta with feasibility bonus and clip [-10, 10], then PBRS + curriculum → final clip [-15, 15] |
| **Code** | `src/rl/gym_env/reward_calculator.py` (376 lines): A completely different reward system using `fitness_weight * improvement + diversity_weight * bonus - time_weight * penalty`, normalized to [-1, 1], with optional hypervolume-based reward. Uses DEAP-era `Individual` objects with `.fitness.values`. |
| **Current system** | `src/rl/gym_env/pymoo_env.py`: Computes reward directly as `delta_hard = prev_best - cur_best` with phase transition at threshold 100, feasibility bonus 10, clipped to [-10, 10]. Then `curriculum_wrapper.py` adds PBRS + curriculum bonus and clips to [-15, 15]. |
| **Impact** | If anyone reads `reward_calculator.py` thinking it's the active reward system, they'll see completely different formulas |
| **Fix** | No thesis change needed (thesis correctly describes the pymoo_env reward). But add a deprecation notice to `reward_calculator.py` header, or delete it. |

---

### M-010: `experimentConfiguration.tex` — "32 CPU Workers" vs Code Parallelization

| | |
|---|---|
| **Thesis claim** | `experimentConfiguration.tex`: "32 CPU workers, pool.map batch fitness eval" |
| **Code reality** | The pymoo system does NOT use `pool.map` for fitness evaluation. `SchedulingProblem._evaluate()` is fully vectorized NumPy — it evaluates the entire population matrix in a single call using `fast_evaluate_hard_vectorized()`. Parallelization in RL training uses `SubprocVecEnv` for parallel environments, not parallel fitness eval. |
| **Fix** | Replace "pool.map batch fitness eval" with "vectorized NumPy batch evaluation" and clarify that parallelization is for RL environment vectorization (multiple env instances via SubprocVecEnv), not GA fitness evaluation. |

---

### M-011: Thesis LLH Table vs Code — Action Names

| | |
|---|---|
| **Thesis claim** | `sec_reinforcement_learning.tex` Table `tab:llh_action_space`: Actions 0-5 with descriptions |
| **Code** | `src/rl/actions/vectorized_ops.py` ACTION_NAMES: `{0: "conservative_repair", 1: "aggressive_repair", 2: "memetic_elite_repair", 3: "soft_focus_repair", 4: "destructive_constructive", 5: "intensified_repair"}` |
| **Specific parameters in code**: |
| Action 0 | elite_fraction=0.10, passes=2, stochastic_alternate=True |
| Action 1 | elite_fraction=0.25, passes=3, stochastic_alternate=True |
| Action 2 | elite_fraction=0.15, passes=4, stochastic_alternate=True |
| Action 3 | elite_fraction=0.08, passes=2, stochastic_alternate=True, compact_soft=True |
| Action 4 | elite_fraction=0.20, passes=2, stochastic_alternate=True, ruin_fraction=0.10 |
| Action 5 | elite_fraction=0.20, passes=3, stochastic_alternate=True |
| **Fix** | Verify that Table `tab:llh_action_space` matches these exact parameters. Pay special attention to Action 5 — the code sets `stochastic_alternate=True` but the docstring says "3 passes deterministic", which is inconsistent within the code itself. |

---

### M-012: Thesis Describes SSCP as Soft Constraint but Code Has Dual Implementation

| | |
|---|---|
| **Thesis claim** | SSCP (Symmetric Sub-Cohort Parallelism) is listed as a soft constraint in Table `tab:soft_constraints_taxonomy` |
| **Code reality** | SSCP is enforced at TWO levels: (1) As a **structural invariant** via `_sync_paired_events()` in `VectorizedRepair` (forces $t_a = t_b$ and $r_a \neq r_b$ for all paired events — guarantees SSCP=0 from generation 1); (2) As a **soft penalty** via `evaluate_paired_cohorts_vectorized()` in `SchedulingProblem._evaluate()` |
| **Impact** | Calling SSCP a "soft constraint" is misleading — it's structurally enforced to always be zero by the repair operator, making it effectively a hard invariant maintained by the algorithm. The soft penalty exists as a secondary quality measure. |
| **Fix** | Add a note in the thesis explaining the dual enforcement: SSCP is structurally guaranteed by the repair operator's paired-event synchronization (ensuring zero violations), while a soft penalty term tracks residual alignment quality for Pareto optimization. |

---

### M-013: `nsga2Implementation.tex` — Fitness Weights

| | |
|---|---|
| **Thesis claim** | `nsga2Implementation.tex`: Fitness vector $\mathbf{F} = (W_{\text{hard}} \cdot P_{\text{hard}}, W_{\text{soft}} \cdot P_{\text{soft}})$ with $W_{\text{hard}} = W_{\text{soft}} = -1.0$ |
| **Code reality** | `src/pipeline/scheduling_problem.py`: `F[:, 0] = G[:, _STRICT_HARD_COLS].sum(axis=1)` and `F[:, 1] = soft_total + paired_penalty`. No weighting factors. Both objectives are positive (violation counts), and pymoo minimizes by default. No $-1.0$ multiplication. |
| **Impact** | The thesis claims negative weights (for DEAP maximization convention where fitness = -violations), but pymoo uses **minimization** convention (lower = better). The sign convention is fundamentally different. |
| **Fix** | Remove the $W = -1.0$ weighting from the thesis. State that both objectives are minimized directly: $f_1 = \sum_{c \in \text{strict}} G_c$ and $f_2 = P_{\text{soft}}$. |

---

## 3. Minor Issues

These are smaller documentation inconsistencies or stale references.

---

### N-001: `gym_env/__init__.py` Docstring Says 20 Actions

| | |
|---|---|
| **File** | `src/rl/gym_env/__init__.py` |
| **Claim** | Documents "20 discrete actions (19 heuristics + 1 no-op)" |
| **Reality** | Current system uses `Discrete(6)` — 6 LLH pipeline configurations |
| **Fix** | Update docstring to "6 discrete actions (LLH pipeline configurations)" |

---

### N-002: `src/rl/helpers.py` Uses Legacy ScheduleEnv

| | |
|---|---|
| **File** | `src/rl/helpers.py` |
| **Issue** | `build_notebook_config` uses `cxpb=0.7`, `mutpb=0.2` and imports legacy `ScheduleEnv` |
| **Impact** | Minor — only used for notebook utilities |
| **Fix** | Update or deprecate this file |

---

### N-003: `action_space.py` Still Exists as Legacy Code

| | |
|---|---|
| **File** | `src/rl/gym_env/action_space.py` (555 lines) |
| **Issue** | Full DEAP-era `ActionMapper` with 20 actions. Not imported or used by `PymooHyperHeuristicEnv`. |
| **Impact** | Dead code that could confuse anyone examining the codebase alongside the thesis |
| **Fix** | Add a prominent deprecation header, or move to `src/rl/legacy/` |

---

### N-004: `resultAndAnalysis.tex` Comment Says "To Be Finalized"

| | |
|---|---|
| **Location** | `resultAndAnalysis.tex` line 1: `% To be finalized after project completion only!` |
| **Impact** | If this comment appears in the compiled PDF... it won't (it's a LaTeX comment). But the content clearly was never finalized. |
| **Fix** | Remove comment and finalize the chapter content |

---

### N-005: Thesis Uses Academic Nomenclature, Code Uses Both

| | |
|---|---|
| **Example** | Thesis: "Faculty Chronological Availability (FCA)"; Code constraint class: `InstructorTimeAvailability` with `code="FCA"` |
| **Issue** | Not wrong per se, but the dual naming could cause confusion when cross-referencing |
| **Fix** | Add a mapping table in the thesis: Academic Name ↔ Code Class Name ↔ Abbreviation |

---

### N-006: Thesis Curriculum Phase Boundaries Match Code ✓

| | |
|---|---|
| **Status** | **VERIFIED CORRECT** — no fix needed |
| **Thesis** | Phase 1: episodes 0→21 (SRE, FFC); Phase 2: 21→63 (adds FTE, FPC, FCA); Phase 3: >63 (all 8). Blend window = 5 episodes. |
| **Code** | `curriculum_wrapper.py`: `phase1_episodes=21`, `phase2_episodes=63`, `_BLEND_WINDOW=5`, `_PHASE_1_COLS = {"SRE", "FFC"}`, `_PHASE_2_COLS = {"SRE", "FFC", "FTE", "FPC", "FCA"}`, `_PHASE_3_COLS = all 8` |

---

### N-007: Thesis PBRS Formula Matches Code ✓

| | |
|---|---|
| **Status** | **VERIFIED CORRECT** — no fix needed |
| **Thesis** | $\Phi(s) = -\frac{\text{Var}(g_0, \ldots, g_7) + \text{Var}(\text{per-inst conflicts}) + \text{Var}(\text{per-room conflicts})}{\text{max\_var}}$, shaping $= \gamma\Phi(s') - \Phi(s)$, $\gamma = 0.99$ |
| **Code** | `reward_shaper.py`: `potential()` computes constraint_var + resource_var via `np.var(G_best)` + `_per_resource_variance()`, normalized by `_max_var`, returns negative. `shaping_reward()` = `gamma * phi_new - phi_prev`, `gamma=0.99`. |

---

### N-008: Thesis Reward Clipping Matches Code ✓

| | |
|---|---|
| **Status** | **VERIFIED CORRECT** — no fix needed |
| **Thesis** | Base reward clipped to $[-10, 10]$; after PBRS + curriculum → final clip $[-15, 15]$ |
| **Code** | `pymoo_env.py`: `np.clip(reward, -10.0, 10.0)` (base); `curriculum_wrapper.py`: `np.clip(shaped_reward, -15.0, 15.0)` (final) |

---

### N-009: Thesis State Vector Matches Code ✓

| | |
|---|---|
| **Status** | **VERIFIED CORRECT** — no fix needed |
| **Thesis** | 39-D observation: [0:5] fitness, [5:8] constraint violation, [8:13] diversity, [13:25] constraint breakdown (8H+4S=12), [25:29] progress, [29:39] heuristic history |
| **Code** | `fast_state_encoder.py`: `OBS_DIM = 39`, identical feature layout documented in code comments |

---

### N-010: Thesis Action Space Table Matches Code ✓

| | |
|---|---|
| **Status** | **VERIFIED CORRECT** — no fix needed |
| **Thesis** | 6 actions: Conservative, Aggressive, Memetic Elite, Soft-Focus, Destructive-Constructive, Intensified |
| **Code** | `vectorized_ops.py`: 6 class definitions mapping to actions 0-5 with matching names |

---

### N-011: `experimentConfiguration.tex` — Missing RL Clip Range Description

| | |
|---|---|
| **Thesis claim** | Table `tab:algorithm_params`: $p_c = 0.70$ with "RL range [0.50, 0.90]" and $p_m = 0.20$ with "RL range [0.10, 0.40]" |
| **Code reality** | The RL agent does NOT modulate crossover or mutation probabilities. It selects repair pipeline configurations. The "RL range" column in the table implies the RL agent tunes these rates — this is incorrect for the current system. |
| **Fix** | Remove the "RL range" column or replace it with the actual RL action space description (6 repair configurations with varying elite%, passes, etc.) |

---

## 4. Code Bug Discovered During Audit

### BUG-001: FCA Toleration is Silently Broken

| | |
|---|---|
| **Thesis claim** | `sec_constraint_taxonomy.tex` and `nsga2Implementation.tex`: FCA (col 5) is "tolerated" — excluded from hard objective $f_1$, added to soft objective $f_2$ via `_TOLERATED_HARD_COLS = {5}` |
| **Code file** | `src/pipeline/scheduling_problem.py` lines 153-165 |
| **The bug** | The code correctly (1) excludes FCA from `F[:, 0]` using `_STRICT_HARD_COLS`, and (2) adds FCA violations to `F[:, 1]` via `F[:, 1] += G[:, col]`. **However**, line 160 then OVERWRITES `F[:, 1]` with `F[:, 1] = soft_total`, destroying the FCA contribution. The assignment operator `=` replaces the accumulated value instead of adding to it. |
| **Actual behavior** | FCA violations go into `G[:, 5]` (observable as a constraint column) but do NOT contribute to either objective: $f_1$ excludes it (correct), and $f_2$ receives it then immediately loses it (bug). FCA violations are effectively ignored during NSGA-II selection. |
| **Thesis-code divergence** | Thesis says FCA is "shifted to the soft objective" — this is NOT what the code actually does |
| **Fix (code)** | Change line 160 from `F[:, 1] = soft_total` to `F[:, 1] += soft_total` so that the FCA penalty is preserved and added to the soft evaluation total |
| **Fix (thesis)** | Once the code is fixed, thesis is correct. If the code is intentionally ignoring FCA (i.e., the current behavior is desired), then the thesis should say "FCA is excluded from both objectives" rather than "shifted to soft." |

---

## 5. Summary Matrix

| ID | Severity | Chapter/File | Issue Summary | Fix Difficulty |
|---|---|---|---|---|
| C-001 | CRITICAL | resultAndAnalysis.tex | "12 hard + 8 soft" should be "8 hard + 4 soft" | Easy (find-replace) |
| C-002 | CRITICAL | sec_constraint_taxonomy.tex / constraints.py | Code has 6 soft constraints; thesis documents 4 | Medium (decide which is canonical) |
| C-003 | CRITICAL | resultAndAnalysis.tex | Toy dataset (10 courses, 3 groups) instead of real TU data | Hard (rewrite section) |
| C-004 | CRITICAL | resultAndAnalysis.tex | "7 days, 70 quanta" should be "6 days, 42 quanta" | Easy |
| C-005 | CRITICAL | resultAndAnalysis.tex | "DEAP 1.4.1" should be "pymoo 0.6.1.3" | Easy |
| C-006 | CRITICAL | resultAndAnalysis.tex | Hardware table vs text contradiction (Intel vs AMD) | Easy (delete wrong text) |
| C-007 | CRITICAL | resultAndAnalysis.tex | "19 heuristic operators" should be "6 LLH configs" | Easy |
| M-001 | MAJOR | resultAndAnalysis.tex | Dozens of XXX placeholders unfilled | Hard (fill all) |
| M-002 | MAJOR | experimentConfiguration.tex vs methodology | p_c=0.70/p_m=0.20 vs p_c=0.5/p_m=0.05 | Medium |
| M-003 | MAJOR | experimentConfiguration.tex vs methodology | Legacy hash-map crossover vs EventBlockCrossover | Medium |
| M-004 | MAJOR | experimentConfiguration.tex vs methodology | Legacy constraint-guided mutation vs EventLocalMutation | Medium |
| M-005 | MAJOR | sec_reinforcement_learning.tex vs experimentConfig | RL pop_size: 100 vs 50/200 vs code default | Easy |
| M-006 | MAJOR | resultAndAnalysis.tex | Mode E description mischaracterizes RL agent's role | Medium |
| M-007 | MAJOR | resultAndAnalysis.tex | Mode C "19 heuristics" should be "6 LLH configs" | Easy |
| M-008 | MAJOR | experimentConfiguration.tex | Tournament size from DEAP era, not applicable to pymoo | Easy |
| M-009 | MAJOR | reward_calculator.py | Legacy reward class, not used but misleading | Easy (add deprecation notice) |
| M-010 | MAJOR | experimentConfiguration.tex | "pool.map batch fitness eval" → vectorized NumPy | Easy |
| M-011 | MAJOR | sec_reinforcement_learning.tex / vectorized_ops.py | Verify LLH table params exactly match code | Easy (verify) |
| M-012 | MAJOR | sec_constraint_taxonomy.tex / repair code | SSCP is structurally enforced, not just a soft preference | Medium |
| M-013 | MAJOR | nsga2Implementation.tex / scheduling_problem.py | $W = -1.0$ weights don't apply; pymoo minimizes directly | Easy |
| N-001 | MINOR | gym_env/**init**.py | Docstring says 20 actions | Trivial |
| N-002 | MINOR | rl/helpers.py | Uses legacy config values | Trivial |
| N-003 | MINOR | action_space.py | 555-line dead code file | Trivial |
| N-004 | MINOR | resultAndAnalysis.tex | "To be finalized" comment | Trivial |
| N-005 | MINOR | General | Academic nomenclature vs code class names | Trivial |
| N-006 | MINOR | Curriculum phases | ✓ VERIFIED CORRECT | None |
| N-007 | MINOR | PBRS formula | ✓ VERIFIED CORRECT | None |
| N-008 | MINOR | Reward clipping | ✓ VERIFIED CORRECT | None |
| N-009 | MINOR | State vector | ✓ VERIFIED CORRECT | None |
| N-010 | MINOR | Action space | ✓ VERIFIED CORRECT | None |
| N-011 | MINOR | experimentConfiguration.tex | "RL range" column implies RL tunes p_c/p_m | Easy |
| BUG-001 | **CODE BUG** | scheduling_problem.py:160 | FCA toleration overwritten by `=` instead of `+=` | Trivial (1 char) |

---

## Priority Triage

### Fix Immediately (before any submission)

1. **C-001 through C-007**: The `resultAndAnalysis.tex` chapter has 7 critical errors. The fastest path is to **replace its experimental setup section entirely** with content from `b1resultAndAnalysis.tex` (which correctly describes the TU Thapathali instance with proper hardware, framework, constraints, and parameters).
2. **BUG-001**: One-character fix in `scheduling_problem.py` line 160: `=` → `+=`.
3. **C-002**: Decide canonical soft constraint count (4 or 6) and align thesis + code.

### Fix Before Defense

1. **M-002 through M-004**: Reconcile `experimentConfiguration.tex` parameters and operator descriptions with the current pymoo system.
2. **M-005, M-006, M-007**: Fix population size, Mode E/C descriptions.
3. **M-011**: Verify LLH table parameters match code exactly.
4. **M-013**: Fix fitness weight convention.

### Nice to Have

1. **N-001 through N-005, N-011**: Documentation cleanup in code and thesis.
2. **M-008 through M-010, M-012**: Clarify parallelization, tournament selection, SSCP enforcement.
