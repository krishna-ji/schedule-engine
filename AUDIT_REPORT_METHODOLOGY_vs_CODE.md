# THESIS ↔ CODEBASE CONSISTENCY AUDIT REPORT  

**Date**: 2026-03-20  
**Scope**: Methodology chapter (Chapter 3) + Implementation chapter (Chapter 4) vs. full codebase  
**Branch**: `feat/rl-pymoo-hyperheuristic`  

---

## EXECUTIVE SUMMARY

| Metric                       | Count |
|------------------------------|-------|
| **Total claims checked**     | 87    |
| **Valid (confirmed)**        | 52    |
| **DIVERGENCE**               | 21    |
| **UNIMPLEMENTED_CLAIM**      | 3     |
| **UNDOCUMENTED_IMPLEMENTATION** | 6  |
| **NON_EXECUTABLE_DESCRIPTION** | 5   |

---

## DETAILED FINDINGS

### CRITICAL ISSUES (Blocking)

| ID | Type | Severity | Methodology/Thesis Claim | Code Reference | Issue |
|----|------|----------|--------------------------|----------------|-------|
| 1  | **DIVERGENCE** | **CRITICAL** | ICTD "Prevents multiple sessions of the same course from **overlapping**" (sec_constraint_taxonomy.tex) | `src/constraints/constraints.py` → `SiblingSameDay.evaluate()` | Code checks for **same-day scheduling**, NOT temporal overlap. The thesis definition describes a different semantic constraint. Overlapping is handled by CTE/FTE/SRE. |
| 2  | **DIVERGENCE** | **CRITICAL** | Implementation chapter describes legacy operators: HashMap-based position-independent crossover (p_c=0.70), constraint-guided mutation with retention probs (inst=0.70, room=0.60, time=0.50) (implDetails/evolutionaryOperators.tex) | `src/pipeline/pymoo_operators.py` → `EventBlockCrossover(prob=0.5)`, `EventLocalMutation(event_prob=0.05)` | The implementation chapter describes **legacy operators** (`src/ga/operators/`) that are NOT used by the Pymoo pipeline. The methodology chapter correctly describes the Pymoo operators, but the implementation chapter contradicts it by describing a completely different operator family. |
| 3  | **DIVERGENCE** | **CRITICAL** | Mode B: "Conservative Repair: 10% elite, 2 passes" applied every generation (sec_dataset_evaluation.tex) | `src/experiments/ga_experiment.py` → `MemeticExperiment`: elite_pct=0.05, repair_iters=5, repair_frequency=5; `runs/ga_02_memetic.py`: ELITE_PCT=0.15, REPAIR_ITERS=8, REPAIR_FREQUENCY=5 | Neither the code defaults (5%/5 passes/every 5 gen) nor the run script (15%/8 passes/every 5 gen) match the thesis claim (10%/2 passes/every gen). Three-way mismatch. |
| 4  | **DIVERGENCE** | **CRITICAL** | Thesis claims dataset: "~80-100 courses", "150 instructors", "37 groups", "70 rooms" (sec_dataset_evaluation.tex) | `data/Course.json`: 444 courses; `data/Groups.json`: 46 groups; `data/Instructors.json`: 189+ instructors; `data/Rooms.json`: 75 rooms | Massive data count discrepancy. Courses are 4-5x higher, instructors 26% higher, groups 25% higher. The "444 courses" is mentioned in the Problem Formulation section but contradicts the Dataset section. |
| 5  | **DIVERGENCE** | **CRITICAL** | RL Hyperparameters table: learning_rate=3×10⁻⁴, entropy_coef=0.01, batch_size=64 (sec_reinforcement_learning.tex, Table 6) | `runs/rl_06_train_ppo_titan_v4_sota.py`: LEARNING_RATE=5e-4, ENT_COEF=0.05, BATCH_SIZE=512 | The SOTA training run uses different hyperparameters than stated in the methodology. LR is 67% higher, entropy is 5× higher, batch is 8× larger. |

---

### HIGH-SEVERITY FINDINGS

| ID | Type | Severity | Methodology/Thesis Claim | Code Reference | Issue |
|----|------|----------|--------------------------|----------------|-------|
| 6  | **DIVERGENCE** | **HIGH** | Methodology: Production config "N=400, G=2000 generations" (sec_problem_formulation.tex, search space analysis) | `src/experiments/ga_experiment.py` → max defaults: pop_size=200 (Aggressive), ngen=300 (Adaptive). No config reaches N=400/G=2000. | Production parameters N=400/G=2000 are not present in any run config or experiment class default. These may have been aspirational or from a removed config. |
| 7  | **DIVERGENCE** | **HIGH** | Methodology: "Population Size (N) = 100" for RL env (sec_integrated_ga_rl.tex) | `runs/rl_06_train_ppo_titan_v4_sota.py`: TRAIN_POP_SIZE=40 | The SOTA RL training uses pop_size=40, not 100 as stated. The default in PymooHyperHeuristicEnv is 100, but no run script uses 100. |
| 8  | **DIVERGENCE** | **HIGH** | NSGA-II tikz figure (methodo_rl_ga_hybrid_framework.tex): tournament size k=3 | Code and thesis text both state k=2 (binary tournament) | The TikZ figure label says "Tournament Selection (k=3)" but both the thesis text and code use binary tournament (k=2). |
| 9  | **DIVERGENCE** | **HIGH** | RL-GA hybrid framework tikz figure: "Reward: r_t = ΔHV + Δfeas − λ·stag" | `src/rl/gym_env/pymoo_env.py` → Phase-transition delta reward | The figure shows a hypervolume-based reward formula, but the code implements phase-transition delta reward with hard/soft improvement deltas (Eq. 18 in thesis). |
| 10 | **DIVERGENCE** | **HIGH** | Implementation chapter: crossover p_c = 0.70 (implDetails/evolutionaryOperators.tex) | `src/pipeline/pymoo_operators.py` → EventBlockCrossover(prob=0.5); `src/ga/operators/crossover.py` → cx_prob=0.5 default | Even the legacy crossover operator defaults to cx_prob=0.5, not 0.70. The 0.70 value is only in AggressiveExperiment. |
| 11 | **DIVERGENCE** | **HIGH** | Implementation chapter: mutation retention probabilities: instructor=0.70, room=0.60, time=0.50 (implDetails/evolutionaryOperators.tex) | `src/ga/operators/mutation.py` → mutate_gene(): instructor keep=50%, room keep=50%, time keep=30% (approx) | Retention probabilities differ: thesis says 70/60/50, code has ~50/50/30. |
| 12 | **DIVERGENCE** | **HIGH** | Thesis constraint taxonomy: "4 soft constraints" (CSC, FSC, MIP, SSCP) | `src/constraints/constraints.py` → `SOFT_CONSTRAINT_CLASSES` contains 6 classes | Code implements 6 soft constraints but thesis only formally describes 4. While the thesis mentions SessionContinuity and BreakPlacementCompliance as "additional code-level" diagnostics, the code registers them in SOFT_CONSTRAINT_CLASSES alongside the main 4. |
| 13 | **DIVERGENCE** | **HIGH** | Thesis claims ICTD table description: "Prevents multiple sessions of the same course from overlapping" | `src/constraints/constraints.py` → `SiblingSameDay`: name="ICTD", checks same-day not overlap | Table 1 description says "overlapping" but constraint actually checks "same-day scheduling". This changes the interpretation: overlapping = exact time conflict, same-day = any sessions on the same calendar day. |
| 14 | **DIVERGENCE** | **HIGH** | Thesis sec_architecture_overview.tex: "five-component pipeline" with "(2) Two-Phase Repair Pipeline provides 6 LLH pipeline configurations" | Code: The 6 LLH configs are in `src/rl/actions/vectorized_ops.py`. But `src/ga/repair/` contains 15+ repair modules (basic, engine, hierarchy, igls, memetic, pipeline, selective, greedy, exhaustive, break_repair, group_clash_repair, heuristic_repair, selective_heuristic, cp/ subpackage, lns/ subpackage) | The extensive repair subsystem in code far exceeds what the thesis describes. CP repair, LNS repair, IGLS repair, and many repair modules are undocumented in the methodology. |
| 15 | **DIVERGENCE** | **HIGH** | Thesis experimentConfiguration.tex: Mode B IGLS repair target "Elite 20%" | Code `MemeticExperiment`: elite_pct=0.05 (5%) | Experiment config chapter says 20% elite repair target but code default is 5%. |

---

### MEDIUM-SEVERITY FINDINGS

| ID | Type | Severity | Methodology/Thesis Claim | Code Reference | Issue |
|----|------|----------|--------------------------|----------------|-------|
| 16 | **DIVERGENCE** | **MEDIUM** | Thesis dataset section: "approximately 80 to 100 courses" | `data/Course.json`: 444 entries; `sec_problem_formulation.tex`: "C = {c_1, ..., c_444}" | Internal thesis contradiction: Dataset section says 80-100 but Problem Formulation says 444. |
| 17 | **DIVERGENCE** | **MEDIUM** | Dataset section: "150 instructors" | Problem Formulation: "181 available instructors"; Code: 189+ instructors | Three different instructor counts across thesis and code. |
| 18 | **DIVERGENCE** | **MEDIUM** | Experiment config table in implementation: pop_size 50 (test) / 400 (production) | GAExperiment default: pop_size=100. No experiment class uses 50 or 400. | Commented-out NSGA-II parameter table (sec_nsga2.tex) says 10/400 for test/prod. Implementation chapter says 50/400. Neither matches defaults. |
| 19 | **DIVERGENCE** | **MEDIUM** | Thesis mentions "60% memory reduction" for contiguous representation | quantumTimeSystem.tex calculates 37.5% for 3-hour sessions, up to 50% for 5-hour | The "60%" claim in geneChromosomeAndPopulation.tex doesn't match the detailed calculations in quantumTimeSystem.tex (37.5%–50%). |
| 20 | **DIVERGENCE** | **MEDIUM** | Thesis: "gene encoding as 7-tuple" $\langle c, t, i, g, r, q_s, q_n \rangle$ | `src/domain/gene.py` → SessionGene attrs: course_id, course_type, instructor_id, group_ids, room_id, start_quanta, num_quanta | Match in substance but thesis uses shorthand (c, t, i, g, r, q_s, q_n) while code uses expanded names. The ordering differs: thesis has instructor before groups but code has instructor_id before group_ids. Minor but verified consistent. |
| 21 | **UNDOCUMENTED_IMPLEMENTATION** | **MEDIUM** | — | `src/ga/repair/cp/` (CP solver subpackage: solver.py, pipeline.py, partitioner.py, merger.py, frozen_selector.py) | CP (Constraint Programming) repair using OR-tools is implemented but not described in the methodology. Only briefly mentioned in CPHybridExperiment. |
| 22 | **UNDOCUMENTED_IMPLEMENTATION** | **MEDIUM** | — | `src/ga/repair/lns/` (Large Neighborhood Search: operator.py, repair.py, diagnostics.py) | LNS repair subsystem is implemented but not described in the methodology or implementation chapters. |
| 23 | **UNDOCUMENTED_IMPLEMENTATION** | **MEDIUM** | — | `src/ga/repair/igls.py` (Iterative Guided Local Search) | IGLS repair is referenced in experimentConfiguration.tex but not described in methodology. |
| 24 | **UNDOCUMENTED_IMPLEMENTATION** | **MEDIUM** | — | `src/domain/supergroup.py` (Supergroup, Cluster, UnionFind for cluster detection) | Supergroup/Cluster hierarchy with union-find for programme coupling detection is fully implemented but not described in thesis. |
| 25 | **UNDOCUMENTED_IMPLEMENTATION** | **MEDIUM** | — | `src/ga/core/usage_tracker.py` (UsageTracker for load-balanced initialization) | Load-balanced initialization using instructor/room/time usage tracking is not described in thesis. |
| 26 | **DIVERGENCE** | **MEDIUM** | Thesis sec_dataset_evaluation.tex: Modes A-E only | `runs/` contains rl_03_train_dqn.py, rl_04_train_ppo_baseline.py, rl_05_train_maskable_ppo.py | DQN training, vanilla PPO baseline, and basic MaskablePPO runs exist but are not described as formal experimental modes. They appear to be stepping stones to Mode E. |
| 27 | **DIVERGENCE** | **MEDIUM** | Implementation chapter experimentConfiguration.tex: Mode F "Heuristic Diagnostics" | No `runs/` script implements Mode F | Mode F is described in the experiment configuration but has no corresponding run script. |
| 28 | **DIVERGENCE** | **MEDIUM** | Thesis: MemeticExperiment pop_size=80, ngen=150, cx=0.6, mut=0.08 | `runs/ga_02_memetic.py`: POP_SIZE=120, NGEN=500, CX=0.4, MUT=0.10 | Run script overrides class defaults with significantly different values. |

---

### LOW-SEVERITY FINDINGS

| ID | Type | Severity | Methodology/Thesis Claim | Code Reference | Issue |
|----|------|----------|--------------------------|----------------|-------|
| 29 | **DIVERGENCE** | **LOW** | Thesis: "6 days" operating, Saturday closed | `src/io/time_system.py`: Sunday–Friday open, Saturday closed | Matches code actually. Sunday is day 0, Friday is day 5, Saturday is closed. This is correct for Nepal (Sunday–Friday work week). |
| 30 | **NON_EXECUTABLE_DESCRIPTION** | **LOW** | Thesis: commented-out parameter table (sec_nsga2.tex) with test/prod values | Code: table is LaTeX-commented with `%` | Commented-out NSGA-II parameter table in methodology contains values (10/400 pop, 30/2000 gen) that don't match any code config. Dead documentation. |
| 31 | **NON_EXECUTABLE_DESCRIPTION** | **LOW** | Thesis: experimentConfiguration.tex mentions Mode F | No implementation exists | Mode F described but unimplemented. Dead specification. |
| 32 | **UNIMPLEMENTED_CLAIM** | **LOW** | Thesis sec_architecture_overview.tex: Output Layer mentions "Excel" and "web interfaces" export formats | `src/io/export/` has PDF, JSON, CSV, PNG but no Excel or web exports | Excel and web interface exports are claimed but not implemented. |
| 33 | **UNIMPLEMENTED_CLAIM** | **LOW** | Thesis sec_architecture_overview.tex: "killswitch-controlled feature toggles" | No killswitch mechanism found in configs | Feature toggle/killswitch mechanism is described but not found in current configs. |
| 34 | **UNIMPLEMENTED_CLAIM** | **LOW** | Thesis sec_architecture_overview.tex: "Experiment Manifest for full traceability" | No manifest tracking system found | Experiment Manifest for run lineage tracking is described but not implemented as a dedicated system. |
| 35 | **NON_EXECUTABLE_DESCRIPTION** | **LOW** | Thesis mentions "32 multiprocessing workers" (experimentConfiguration.tex) | `src/utils/system_info.py`: default=8. No config sets 32. | 32 workers is stated but no config enforces it; default is 8. |
| 36 | **DIVERGENCE** | **LOW** | Thesis: Population initialization "~25% random, ~50% conflict-aware, ~25% greedy" | `src/ga/core/population_factory.py` → `create_population()` with strategy parameter | The hybrid strategy exists but exact percentages are not validated; they depend on the `strategy` parameter chosen at runtime. |
| 37 | **NON_EXECUTABLE_DESCRIPTION** | **LOW** | Thesis: RL hyperparameters "Optimization Epochs per Update = 10" and "Batch Size = 64" | `runs/rl_06_train_ppo_titan_v4_sota.py`: N_EPOCHS=10 ✓, BATCH_SIZE=512 ✗ | Epochs match but batch size diverges (64 vs 512). |

---

## FIGURE-SPECIFIC FINDINGS

| ID | Figure | Type | Severity | Issue |
|----|--------|------|----------|-------|
| F1 | `methodo_rl_ga_hybrid_framework.tex` (TikZ) | **DIVERGENCE** | **HIGH** | Tournament size labeled "k=3" but thesis text and code both use k=2 (binary tournament). |
| F2 | `methodo_rl_ga_hybrid_framework.tex` (TikZ) | **DIVERGENCE** | **HIGH** | Reward formula shown as "r_t = ΔHV + Δfeas − λ·stag" but code uses phase-transition delta reward (Eq. 18), not hypervolume-based. |
| F3 | `impl_mutation_decision_tree.drawio` | **DIVERGENCE** | **MEDIUM** | Retention probabilities in drawio: Instructor=70%, Room=50%. Code mutation.py: Instructor=50%, Room=50%(keep-if-suitable). Implementation chapter says 70/60/50. Three-way mismatch. |
| F4 | `impl_entire_process_visualized.drawio` | **DIVERGENCE** | **MEDIUM** | Shows "Course-group aware" crossover and "Constraint aware" mutation, implying legacy operators. But the actual Pymoo pipeline uses EventBlockCrossover and EventLocalMutation. |
| F5 | `impl_genetic_operator.drawio` | **DIVERGENCE** | **LOW** | Shows single-point style crossover diagram but actual EventBlockCrossover is Bernoulli block exchange (not single-point). |
| F6 | `methodo_nadir_point.drawio` | **MINOR** | **LOW** | Label for z* shows "Worst solution" in audit note but in actual drawio/PDF it correctly shows "Ideal point". No issue in rendered figure. |

---

## UNDOCUMENTED CODE COMPONENTS (Code → Missing Thesis Coverage)

| ID | Component | File Path | Significance |
|----|-----------|-----------|--------------|
| U1 | CP Repair (Constraint Programming solver using OR-Tools) | `src/ga/repair/cp/` (5 files) | Full CP-SAT repair pipeline exists but not in methodology |
| U2 | LNS Repair (Large Neighborhood Search) | `src/ga/repair/lns/` (3 files) | Destroy-and-repair LNS heuristic implemented but undocumented |
| U3 | IGLS (Iterative Guided Local Search) | `src/ga/repair/igls.py` | Referenced in experiment config but methodology incomplete |
| U4 | Supergroup/Cluster Detection | `src/domain/supergroup.py` | Programme-level group aggregation with union-find not in thesis |
| U5 | UsageTracker (Load-Balanced Init) | `src/ga/core/usage_tracker.py` | Smart initialization with load tracking not described |
| U6 | DQN Agent Training Path | `src/rl/agents/dqn_agent.py`, `runs/rl_03_train_dqn.py` | DQN competitor exists but not described as formal experiment |

---

## NON-EXECUTABLE PATHS

| ID | Description | Evidence |
|----|-------------|----------|
| N1 | Production config N=400, G=2000 | No experiment class or run script produces these values |
| N2 | Mode F "Heuristic Diagnostics" | Described in implementation chapter but no run script exists |
| N3 | 32 multiprocessing workers | Stated in experiment config but no code enforces this |
| N4 | Commented-out NSGA-II parameter table | LaTeX-commented with test=10pop/30gen, prod=400pop/2000gen — not in code |
| N5 | Excel and web interface exports | Claimed in architecture overview but not implemented |

---

## RECOMMENDATIONS

### Must-Fix (CRITICAL — thesis or code must be corrected)

1. **ICTD Definition** → Rewrite thesis to say "Prevents multiple sessions of the same course from being scheduled on the **same calendar day**" instead of "overlapping". Add clarification that temporal overlap is separately handled by CTE/FTE/SRE.

2. **Implementation Chapter Operators** → Either (a) clearly label the implementation chapter operators as "legacy" with a note that the Pymoo pipeline uses different operators, or (b) rewrite the implementation chapter to describe EventBlockCrossover and EventLocalMutation instead of the legacy operators.

3. **Mode B Parameters** → Update thesis to match actual run config: either change to match `ga_02_memetic.py` values (15%/8 passes/every 5 gen) or the `MemeticExperiment` defaults (5%/5 passes/every 5 gen). State which is canonical.

4. **Dataset Counts** → Reconcile dataset section (80-100 courses, 150 instructors) with problem formulation (444 courses, 181 instructors). Use correct counts throughout.

5. **RL Hyperparameters** → Update Table 6 (sec_reinforcement_learning.tex) to match `rl_06_train_ppo_titan_v4_sota.py`: LR=5e-4, ENT=0.05, BATCH=512.

### Should-Fix (HIGH — important for academic accuracy)

1. **Tournament size in TikZ figure** → Change "k=3" to "k=2" in `methodo_rl_ga_hybrid_framework.tex`.

2. **Reward formula in TikZ figure** → Replace "r_t = ΔHV + Δfeas − λ·stag" with the actual phase-transition delta formula in `methodo_rl_ga_hybrid_framework.tex`.

3. **Mutation retention probabilities** → Reconcile 70/60/50 (thesis) with 50/50/30 (code) in `implDetails/evolutionaryOperators.tex`.

4. **Soft constraint count** → Clarify that code has 6 registered soft constraints (4 formal + 2 diagnostic), or adjust SOFT_CONSTRAINT_CLASSES.

5. **Population size for RL** → Either change thesis to N=40 (reflecting actual SOTA training) or note that N=100 is the default but training uses N=40 for computational efficiency.

### Nice-to-Fix (MEDIUM — completeness)

1. Document CP repair, LNS repair, IGLS, Supergroup hierarchy in thesis or remove from codebase.
2. Remove Mode F from implementation chapter if unimplemented.
3. Remove claims about Excel/web exports, killswitches, and experiment manifests if unimplemented.
4. Remove or update the commented-out NSGA-II parameter table.
5. Reconcile memory savings figure (60% vs 37.5%) across chapters.

---

*Report generated by automated bidirectional consistency audit.*
