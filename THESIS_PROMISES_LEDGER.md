# THESIS PROMISES LEDGER

> **Phase 44 Audit — Zero-Hallucination Promise Extraction**
> Generated: 2026-02-28 | Auditor: Principal Academic Auditor
> Repository: `schedule-engine` branch `feat/rl-pymoo-hyperheuristic`

---

## Table of Contents

1. [Algorithm Commitments](#1-algorithm-commitments)
2. [Architectural Commitments](#2-architectural-commitments)
3. [Heuristic Commitments](#3-heuristic-commitments)
4. [Evaluation Metrics](#4-evaluation-metrics)
5. [Comparative Baselines & Ablation Modes](#5-comparative-baselines--ablation-modes)
6. [Specific Numerical Commitments](#6-specific-numerical-commitments)
7. [Critical Discrepancies](#7-critical-discrepancies)
8. [Deliverable Status Matrix](#8-deliverable-status-matrix)

---

## 1. Algorithm Commitments

### 1.1 Core Optimizer

- [ ] **NSGA-II as backbone multi-objective optimizer**
  > "To design and implement an automated timetabling system for Thapathali Campus using NSGA-II, a multi-objective genetic algorithm metaheuristic."
  - Source: `chap/intro.tex` ~L16
  - Status: **DELIVERED** — NSGA-II fully implemented via pymoo

- [ ] **Bi-objective formulation (f1: hard, f2: soft)**
  > "This study formulates the UCTP as a multi-objective optimization problem with two distinct and conflicting objectives, and separate evaluation of hard and soft constraints."
  - Source: `chap/systemArchitectureAndMethodology/sec_problem_formulation.tex` ~L5
  - Status: **DELIVERED**

- [ ] **Pareto-based selection with crowding distance**
  > "NSGA-II employs crowding distance—a density estimation metric that quantifies the proximity of solutions in objective space."
  - Source: `chap/systemArchitectureAndMethodology/sec_nsga2.tex` ~L43
  - Status: **DELIVERED**

- [ ] **Binary tournament selection**
  > "We employ binary tournament selection: for each parent slot, two individuals are sampled uniformly from the population."
  - Source: `chap/systemArchitectureAndMethodology/sec_nsga2.tex` ~L60
  - Status: **DELIVERED**

### 1.2 Reinforcement Learning Algorithms

- [ ] **PPO (Proximal Policy Optimization) — Primary RL algorithm**
  > "Proximal Policy Optimization (PPO) serves as the primary on-policy algorithm, chosen for its stability and ease of tuning."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L145
  - Status: **DELIVERED** — PPO implemented via Stable-Baselines3

- [ ] **DQN (Deep Q-Network) — Alternative RL algorithm**
  > "Deep Q-Network (DQN) is implemented as an alternative off-policy algorithm."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L179
  - Status: **PARTIALLY DELIVERED** — Code exists (`rl_02_train_dqn.py`) but no results reported

- [ ] **Dual-algorithm support (PPO + DQN)**
  > "This dual-algorithm support (PPO and DQN) with curriculum learning enables leveraging PPO's stability for continuous policy learning or DQN's sample efficiency for discrete action selection."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L261
  - Status: **PARTIALLY DELIVERED** — Both coded, only PPO has results

### 1.3 Curriculum Learning

- [ ] **3-stage curriculum learning with automatic progression**
  > "The system implements a multi-stage curriculum learning approach that progressively increases problem difficulty during training."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L215
  - Status: **NOT DELIVERED** — Listed as future work in `resultAndAnalysis.tex` L1243

- [ ] **Stage 1 (Easy): 10 courses, 200 episodes, 100 generations**
  > "Stage 1 (Easy): (10, 200, 100, 0.0, 3, 25, 5) — Focus: Learn basic heuristic selection."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L237
  - Status: **NOT DELIVERED**

- [ ] **Stage 2 (Medium): 20 courses, 300 episodes, 150 generations**
  > "Stage 2 (Medium): (20, 300, 150, 0.2, 3, 30, 5)"
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L242
  - Status: **NOT DELIVERED**

- [ ] **Stage 3 (Hard): 40 courses, 500 episodes, 200 generations**
  > "Stage 3 (Hard): (40, 500, 200, 0.4, 3, 50, 5)"
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L247
  - Status: **NOT DELIVERED**

- [ ] **Validation-driven curriculum progression**
  > "The curriculum manager implements automatic stage progression using a windowed validation policy."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L231
  - Status: **NOT DELIVERED**

### 1.4 Adaptive Operator Selection

- [ ] **UCB1 multi-armed bandit for operator selection (Mode D)**
  > "Mode D: Assesses the benefit of adaptive operator selection via multi-armed bandit learning (UCB1 algorithm)."
  - Source: `chap/resultAndAnalysis.tex` ~L236
  - Status: **PLACEHOLDER ONLY** — No real results

---

## 2. Architectural Commitments

### 2.1 System Architecture

- [ ] **6-layer modular architecture**
  > "The proposed hyperheuristic optimization system comprises six interconnected architectural layers."
  - Source: `chap/systemArchitectureAndMethodology/sec_architecture_overview.tex` ~L3
  - Status: **DELIVERED**

- [ ] **Bi-level hybrid GA-RL meta-optimization framework**
  > "The proposed system implements a Hybrid Evolutionary Architecture that fuses the global search capabilities of NSGA-II with the adaptive decision-making intelligence of Deep Reinforcement Learning."
  - Source: `chap/systemArchitectureAndMethodology/sec_integrated_ga_rl.tex` ~L3
  - Status: **DELIVERED**

- [ ] **12 implementation modules**
  > Includes: geneChromosomeAndPopulation, genotypeAndPhenotypeMapping, quantumTimeSystem, domainEntityRelations, dataValidationAndFeasibility, geneInitializationPipeline, evolutionaryOperators, nsga2Implementation, advancedImplementation, reinfLearning, algorithmicComplexity, experimentConfiguration
  - Source: `chap/implementationDetails.tex` L8-19
  - Status: **DELIVERED**

### 2.2 Neural Network Architecture

- [ ] **Actor-Critic shared feature extraction**
  > "The policy employs an actor-critic formulation with shared feature extraction."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L169
  - Status: **DELIVERED** — MlpPolicy in SB3

- [ ] **Network topology: 41 → 64 → 64**
  > "3-layer architecture (41→64→64), checkpointing mechanisms, and curriculum management."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L173
  - Status: **PARTIALLY DELIVERED** — Current capstone uses [64,64] but with 8 actions, not 41-dim state

### 2.3 State Space

- [ ] **41-dimensional normalized observation vector**
  > "The 41-dimensional state vector encompasses fitness quality, population diversity, search progress, constraint violations, and heuristic application history."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L9
  - Status: **DISCREPANCY** — Current `pymoo_env.py` uses a different state representation

- [ ] **State vector components: Fitness (5), Diversity (5), Progress (4), Violations (3+14), History (10)**
  > "41-dimensional representation enables the RL agent to understand both macro-level optimization progress and micro-level constraint satisfaction challenges."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L64
  - Status: **NEEDS VERIFICATION** against current code

### 2.4 Infrastructure

- [ ] **32-core CPU parallel fitness evaluation**
  > "Parallelism: 32 worker processes for constraint evaluation."
  - Source: `chap/resultAndAnalysis.tex` ~L292
  - Status: **DELIVERED** — multiprocessing implemented

- [ ] **GPU acceleration support (CUDA 12.1)**
  > "PyTorch 2.4.1+CUDA 12.1 serves as the neural network backend."
  - Source: `chap/requirements.tex` ~L31
  - Status: **CODE EXISTS** — GPU support available

- [ ] **1-3 hour production runtime target**
  > "Handle institutional-scale problem instances within reasonable computational time (1-3 hours for production runs)."
  - Source: `chap/requirements.tex` ~L15
  - Status: **EXCEEDED** — Mode B took 30 hours

---

## 3. Heuristic Commitments

### 3.1 Heuristic Count

- [ ] **20 heuristic operators (from slides & requirements)**
  > "Discrete selection from 20 heuristic operators"
  - Source: `SLIDE/method.tex` L456; `chap/requirements.tex` L29
  - Status: **DISCREPANCY** — Slides say 20, methodology says 19, current code has 8 vectorized actions

- [ ] **19 registered heuristic operators (from methodology)**
  > "Introduces 19 registered heuristic operators (construction, perturbation, improvement categories)."
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L134
  - Status: **DISCREPANCY** — Current capstone uses 8 vectorized operators

### 3.2 Heuristic Categories (from Slides)

- [ ] **Mutation Rate Control (5 operators)**: Low, Medium, High, Adaptive, Declining
  > Source: `SLIDE/method.tex` L501
  - Status: **NOT IN CURRENT CODE** — Replaced by vectorized tensor ops

- [ ] **Repair Heuristics (6 operators)**: Instructor conflict, Room capacity, Time slot, Qualification, Availability, Multi-constraint batch
  > Source: `SLIDE/method.tex` L502-508
  - Status: **REPLACED** — Now `spatial_resource_projection`, `faculty_temporal_projection`, etc.

- [ ] **Diversity Injection (3 operators)**: Random immigrant, Smart immigrant, Restart worst
  > Source: `SLIDE/method.tex` L510
  - Status: **NOT IN CURRENT CODE**

- [ ] **Local Search (4 operators)**: Hill climbing, Tabu search, Simulated annealing, Greedy descent
  > Source: `SLIDE/method.tex` L511
  - Status: **NOT IN CURRENT CODE**

- [ ] **Crossover Modulation (2 operators)**: Increase/decrease crossover probability
  > Source: `SLIDE/method.tex` L512
  - Status: **NOT IN CURRENT CODE**

### 3.3 Current Heuristic Implementation (8 Vectorized Actions)

The codebase has evolved to use 8 vectorized tensor-based operators (from `ACTION_NAMES`):

| ID | Current Action Name | Maps To Original Category |
|----|-------------------|--------------------------|
| 0 | `spatial_resource_projection` | Room capacity/suitability repair |
| 1 | `faculty_temporal_projection` | Instructor conflict repair |
| 2 | `cohort_temporal_projection` | Student group repair |
| 3 | `symmetric_subcohort_sync` | Multi-constraint batch repair |
| 4 | `universal_feasibility_projection` | Global feasibility repair |
| 5 | `stochastic_quanta_perturbation` | Time slot perturbation |
| 6 | `stochastic_spatial_perturbation` | Room/spatial perturbation |
| 7 | `meridian_compaction` | Schedule compaction |

### 3.4 Evolutionary Operators

- [ ] **Position-independent course-group aware crossover**
  > "Our approach implements position-independent crossover that preserves the fundamental (course, group) enrollment structure."
  - Source: `chap/systemArchitectureAndMethodology/sec_evolutionary_operators.tex` ~L9
  - Status: **DELIVERED**

- [ ] **Constraint-guided mutation**
  > "Constraint-guided mutation biases changes toward feasible regions using domain knowledge."
  - Source: `chap/systemArchitectureAndMethodology/sec_evolutionary_operators.tex` ~L57
  - Status: **DELIVERED**

### 3.5 Constraint Taxonomy

- [ ] **8 hard constraints implemented** (from methodology)
  > "The system implements 8 hard constraints."
  - Source: `chap/systemArchitectureAndMethodology/sec_constraint_taxonomy.tex` ~L15
  - Status: **DISCREPANCY** — `resultAndAnalysis.tex` L109 claims **12 hard constraints**

- [ ] **6 soft constraints implemented** (from methodology)
  > "The system implements 6 soft constraints."
  - Source: `chap/systemArchitectureAndMethodology/sec_constraint_taxonomy.tex` ~L114
  - Status: **DISCREPANCY** — `resultAndAnalysis.tex` L109 claims **8 soft constraints**

- [ ] **12 hard + 8 soft constraints** (from results chapter)
  > "12 hard constraints and 8 soft constraints"
  - Source: `chap/resultAndAnalysis.tex` ~L109
  - Status: **NEEDS RECONCILIATION** with methodology chapter

---

## 4. Evaluation Metrics

### 4.1 Multi-Objective Quality Indicators

- [ ] **Hypervolume (HV)** — Volume of objective space dominated by Pareto front
  > "Hypervolume (HV): Volume of objective space dominated by Pareto front — Target: Maximize"
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L18
  - Status: **DELIVERED** — WFG algorithm implemented

- [ ] **Inverted Generational Distance (IGD)**
  > "IGD: Inverted Generational Distance: proximity to reference front — Target: Minimize"
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L22
  - Status: **NOT REPORTED** — Code may exist but no results

- [ ] **Spacing (S)** — Uniformity of solution distribution
  > "Spacing (S): Uniformity of solution distribution along front — Target: Minimize"
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L20
  - Status: **NOT REPORTED**

- [ ] **Spread (Δ)** — Coverage extent across objective space
  > "Spread (Δ): Coverage extent across objective space — Target: Minimize"
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L24
  - Status: **NOT REPORTED**

- [ ] **Generational Distance (GD)**
  > "Performance Metrics: Compute hypervolume, inverted generational distance (IGD), and generational distance (GD)."
  - Source: `chap/requirements.tex` ~L11
  - Status: **NOT REPORTED**

- [ ] **Pareto Front Size**
  > "Pareto Front Size: |PF₁| — Target: Maximize"
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L26
  - Status: **DELIVERED** — Reported for Modes A & B

### 4.2 Constraint Metrics

- [ ] **Hard Constraint Violations (f₁)**
  > "Total count of mandatory constraint violations — Target: Zero (required)"
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L29
  - Status: **DELIVERED** — Reported for Modes A & B

- [ ] **Soft Constraint Violations (f₂)**
  > "Total measure of quality preference violations — Target: Minimize"
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L31
  - Status: **DELIVERED** — Reported for Modes A & B

- [ ] **Feasibility Rate** — % of Pareto solutions with zero hard violations
  > "Feasibility Rate: Percentage of Pareto solutions with zero hard constraint violations — Target: Maximize"
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L33
  - Status: **DELIVERED** — Both modes achieved 0%

- [ ] **Convergence Rate**
  > "Number of generations required before hypervolume improvement falls below threshold — Target: Minimize"
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L36
  - Status: **NOT REPORTED**

### 4.3 Diversity Metrics

- [ ] **Genotypic diversity** — Normalized Hamming distance
  > "Genotypic diversity measures chromosome-level variability through normalized Hamming distance."
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L86
  - Status: **DELIVERED** — Reported for both modes

- [ ] **Phenotypic diversity** — Objective-space spread
  > "Phenotypic diversity captures objective-space spread."
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L98
  - Status: **PARTIALLY DELIVERED**

### 4.4 Statistical Rigor

- [ ] **5 independent runs per mode with different seeds**
  > "We conduct 5 independent runs per mode with different random seeds (42, 123, 456, 789, 1024)."
  - Source: `chap/resultAndAnalysis.tex` ~L297
  - Status: **NOT DONE** — Only 1 run with seed 69

- [ ] **Wilcoxon signed-rank tests (α = 0.05)**
  > "Statistical significance is validated using Wilcoxon signed-rank tests."
  - Source: `chap/resultAndAnalysis.tex` ~L297
  - Status: **NOT DONE** — Requires 5+ runs

- [ ] **Cohen's d / Vargha-Delaney Â₁₂ effect sizes**
  > "Effect size analysis using Cohen's d metric complements significance testing."
  - Source: `chap/resultAndAnalysis.tex` ~L50
  - Status: **NOT DONE**

### 4.5 RL-Specific Metrics

- [ ] **Reward function: r_t = w_f·r_fitness + w_d·r_diversity - w_t·r_time**
  > "The reward function r_t guides the agent towards faster convergence."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L77
  - Status: **EVOLVED** — Current reward uses different formulation

- [ ] **Hypervolume-based reward for multi-objective mode**
  > "For multi-objective mode, we use the hypervolume indicator."
  - Source: `chap/systemArchitectureAndMethodology/sec_reinforcement_learning.tex` ~L89
  - Status: **NEEDS VERIFICATION** against current code

- [ ] **+25% hypervolume improvement from RL over baseline**
  > "Trained RL agent achieved +25% hypervolume improvement over baseline NSGA-II."
  - Source: `chap/resultAndAnalysis.tex` ~L1623
  - Status: **UNVERIFIED CLAIM** — No Mode E real data exists

---

## 5. Comparative Baselines & Ablation Modes

### 5.1 Six Experimental Modes (A–F)

- [ ] **Mode A: Pure NSGA-II (Baseline)**
  > "Standard multi-objective genetic algorithm with tournament selection, course-group aware crossover, and constraint-guided mutation. No local search or heuristic enhancements."
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L118
  - Status: **DELIVERED** — Real results: Best Hard=1439, Soft=1081, HV=1,172,007

- [ ] **Mode B: + Memetic Local Search (IGLS)**
  > "Adds IGLS repair operators applied to elite individuals."
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L120
  - Status: **DELIVERED** — Real results: Best Hard=347, Soft=699, HV=3,476,217

- [ ] **Mode C: + Round-Robin Heuristics (19 operators)**
  > "Introduces 19 registered heuristic operators applied in deterministic rotation."
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L134
  - Status: **XXX PLACEHOLDERS ONLY** — No real data

- [ ] **Mode D: + Adaptive Selection (UCB1)**
  > "Adds dynamic effectiveness tracking and priority reordering."
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L140
  - Status: **XXX PLACEHOLDERS ONLY** — No real data

- [ ] **Mode E: + RL-Guided Control (PPO)**
  > "Deploys the trained PPO agent that observes 41-dimensional state vectors and selects from 20 heuristic operators. Training spans 100K timesteps across 3 curriculum stages."
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L150
  - Status: **XXX PLACEHOLDERS ONLY** — No real data (capstone run is separate)

- [ ] **Mode F: Heuristic Profiling**
  > "Mode F (Heuristic-/Test): Individual heuristic profiling mode."
  - Source: `chap/resultAndAnalysis.tex` ~L227
  - Status: **MENTIONED ONLY** — No results

### 5.2 External Baselines

- [ ] **Manual Timetable comparison**
  > "Manual Timetable...serves as a real-world benchmark for evaluating practicality."
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L156
  - Status: **NOT DONE**

- [ ] **Non-adaptive Baseline GA comparison**
  > "A non-adaptive GA implementation with fixed crossover, mutation rates, and no hyperheuristics."
  - Source: `chap/systemArchitectureAndMethodology/sec_dataset_evaluation.tex` ~L158
  - Status: **Mode A serves this role**

---

## 6. Specific Numerical Commitments

### 6.1 Training Parameters

| Parameter | Promised Value | Source | Actual |
|-----------|---------------|--------|--------|
| PPO training timesteps | 100,000 | `sec_reinforcement_learning.tex` ~L99 | Capstone ran 151,552 |
| Population size (production) | 400 | `resultAndAnalysis.tex` ~L286 | Mode A used 200, Capstone used 120 |
| Generations (production) | 2,000 | `resultAndAnalysis.tex` ~L287 | Mode A ran 10,000 |
| Crossover rate (baseline) | 0.7 | `resultAndAnalysis.tex` ~L288 | Needs verification |
| Mutation rate (baseline) | 0.2 | `resultAndAnalysis.tex` ~L289 | Needs verification |
| PPO clip range | 0.2 | `sec_reinforcement_learning.tex` ~L99 | **MATCHES** capstone |
| PPO learning rate | 3×10⁻⁴ | `sec_reinforcement_learning.tex` ~L99 | **MATCHES** capstone |
| Network architecture | [64, 64] | `sec_reinforcement_learning.tex` ~L99 | **MATCHES** capstone |
| Discount factor (γ) | 0.99 | `sec_reinforcement_learning.tex` ~L175 | Needs verification |
| GAE lambda | 0.95 | `sec_reinforcement_learning.tex` ~L175 | Needs verification |
| Independent runs per mode | 5 | `resultAndAnalysis.tex` ~L297 | **Only 1 run** |
| Random seeds | 42, 123, 456, 789, 1024 | `resultAndAnalysis.tex` ~L297 | Used seed 69 |

### 6.2 Dataset Parameters

| Parameter | Promised Value | Source | Status |
|-----------|---------------|--------|--------|
| Courses | 444 | `b1resultAndAnalysis.tex` ~L14 | **VERIFIED** |
| Student Groups | 37 | `b1resultAndAnalysis.tex` ~L16 | **VERIFIED** |
| Instructors | 181 | `b1resultAndAnalysis.tex` ~L18 | **VERIFIED** |
| Rooms | 67 | `b1resultAndAnalysis.tex` ~L20 | **VERIFIED** |
| Time Quanta | 42 | `b1resultAndAnalysis.tex` ~L22 | **VERIFIED** |
| Programs | 9 | `b1resultAndAnalysis.tex` ~L24 | **VERIFIED** |

### 6.3 Performance Claims

| Claim | Source | Status |
|-------|--------|--------|
| 139× hypervolume speedup (WFG) | `b1resultAndAnalysis.tex` ~L264 | **VERIFIED** |
| 7.2× crossover speedup (hash map) | `b1resultAndAnalysis.tex` ~L258 | **NEEDS VERIFICATION** |
| 25× fitness eval speedup (32 cores) | `b1resultAndAnalysis.tex` ~L262 | **NEEDS VERIFICATION** |
| 50× diversity computation speedup | `b1resultAndAnalysis.tex` ~L266 | **NEEDS VERIFICATION** |
| 82% hard constraint reduction | `SLIDE/conclusion.tex` ~L7 | **DISCREPANCY** — Mode B achieved 75.9% |
| 71% soft constraint reduction | `SLIDE/conclusion.tex` ~L9 | **DISCREPANCY** — Mode B achieved 35.3% |
| 15s execution time | `SLIDE/conclusion.tex` ~L11 | **DISCREPANCY** — Mode B took 30 hours |

---

## 7. Critical Discrepancies

### 7.1 HIGH SEVERITY — Must Reconcile Before Defense

| # | Discrepancy | Location A | Location B | Resolution Needed |
|---|------------|-----------|-----------|-------------------|
| 1 | **Heuristic count: 19 vs 20 vs 8** | Methodology: 19 | Slides: 20, Code: 8 | Reconcile in text — explain evolution to vectorized ops |
| 2 | **Hard constraints: 8 vs 12** | Methodology: 8 | Results: 12 | Count actual implementation |
| 3 | **Soft constraints: 6 vs 8** | Methodology: 6 | Results: 8 | Count actual implementation |
| 4 | **Population size: 400 vs 200 vs 120** | Results: 400 | Mode A: 200, Capstone: 120 | Standardize or explain per-mode |
| 5 | **State vector: 41-dim vs actual** | Methodology: 41 | Code: different | Verify actual observation space |
| 6 | **Action space: 20 vs 5 vs 8** | Methodology: 20 | Results: 5 (a0-a4), Code: 8 | Reconcile action definitions |
| 7 | **Runs: 5 per mode vs 1** | Results: 5 | Actual: 1 | Either run more or downscale claim |
| 8 | **Seeds: 42,123,456,789,1024 vs 69** | Results chapter | b1results | Align actual seeds used |
| 9 | **Modes C/D/E/F: All XXX placeholders** | Results chapter | — | Fill or remove from scope |
| 10 | **Slide claims: 82%/71%/15s** | Slides | Mode B actual: 75.9%/35.3%/30h | Update slides or clarify context |

### 7.2 MEDIUM SEVERITY — Should Address

| # | Discrepancy | Notes |
|---|------------|-------|
| 11 | Curriculum learning promised but never implemented | Remove from methodology or implement |
| 12 | DQN described in detail but no results | Either produce DQN results or scope down |
| 13 | IGD, GD, Spacing, Spread, Convergence Rate all promised but unreported | Compute or remove from metrics table |
| 14 | `remainingTasks.tex` still says "Hybridize with RL" as future work | Contradicts claim that Mode E exists |
| 15 | Manual timetable comparison promised but not delivered | Either obtain or remove baseline |

### 7.3 LOW SEVERITY — Optional Improvements

| # | Item | Notes |
|---|------|-------|
| 16 | User Interface Design promised in `remainingTasks.tex` | Likely out of scope — remove or mark as future |
| 17 | Transfer learning mentioned as future direction | Not a hard commitment |
| 18 | LNS operators suggested for instructor availability bottleneck | Exploratory suggestion, not a commitment |

---

## 8. Deliverable Status Matrix

### DELIVERED (Use in thesis as-is)

| Deliverable | Evidence |
|-------------|----------|
| NSGA-II multi-objective optimizer | Full implementation, Mode A results |
| Memetic local search (IGLS) | Mode B results: 75.9% hard reduction |
| Position-independent crossover | Code + methodology description |
| Constraint-guided mutation | Code + methodology description |
| Hypervolume computation (WFG) | 139× speedup verified |
| Bi-objective formulation | f1=hard, f2=soft throughout |
| Pareto front analysis | Mode A=1 point, Mode B=11 points |
| Thapathali Campus dataset | 444 courses, 181 instructors, 67 rooms |
| PPO RL agent (capstone) | 151,552 timesteps, Efficacy Matrix generated |
| 8 vectorized heuristic operators | `spatial_resource_projection`, etc. |
| Training curve data | 3,093 episodes saved |
| Heuristic Efficacy Matrix | Per-action ΔHard/ΔSoft statistics |

### PARTIALLY DELIVERED (Needs completion or text adjustment)

| Deliverable | Gap |
|-------------|-----|
| Mode E (RL-Guided) | Capstone data exists but needs proper evaluation run |
| DQN implementation | Code exists, no results |
| 200-generation evaluation | Script ready, needs execution |
| Statistical tests | Only 1 seed, need 5 for Wilcoxon |

### NOT DELIVERED (Must address before defense)

| Deliverable | Priority |
|-------------|----------|
| Mode C (Round-Robin) results | HIGH |
| Mode D (Adaptive UCB) results | HIGH |
| Curriculum learning | HIGH — either implement or remove from text |
| IGD / GD / Spacing / Spread metrics | MEDIUM |
| 5 independent runs per mode | MEDIUM |
| Manual timetable baseline | LOW |
| User Interface | LOW — remove from scope |

---

## Summary Statistics

| Category | Promises Found | Delivered | Partial | Not Delivered |
|----------|---------------|-----------|---------|---------------|
| Algorithm | 18 | 6 | 4 | 8 |
| Architecture | 12 | 8 | 2 | 2 |
| Heuristic | 14 | 5 | 2 | 7 |
| Metric | 16 | 7 | 2 | 7 |
| Baseline | 8 | 2 | 2 | 4 |
| **TOTAL** | **68** | **28 (41%)** | **12 (18%)** | **28 (41%)** |

---

> **Bottom Line**: 41% of thesis promises are fully delivered, 18% are partially delivered, and 41% have gaps requiring either code execution, text rewriting, or scope changes before thesis defense.
