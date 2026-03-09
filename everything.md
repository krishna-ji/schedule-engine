# Everything: Comprehensive Ablation & Experimental Report

> **Project**: RL-Guided Hyper-Heuristic for University Course Timetabling  
> **Authors**: Krishna Acharya, Dinanath Padhya, Bipul Dahal  
> **Engine**: `schedule-engine` v1.0.0 · Python 3.12 · PyTorch 2.4.1+cu121 · Pymoo 0.6.1  
> **Problem Scale**: ~600 events, 42 time quanta, ~120 instructors, ~40 rooms, ~80 groups, ~1800 decision variables  

---

## Table of Contents

- [Everything: Comprehensive Ablation \& Experimental Report](#everything-comprehensive-ablation--experimental-report)
  - [Table of Contents](#table-of-contents)
  - [1. Problem Formulation](#1-problem-formulation)
  - [2. Constraint Taxonomy](#2-constraint-taxonomy)
    - [2.1 Hard Constraints ($\\mathcal{H}$) — 8 Total](#21-hard-constraints-mathcalh--8-total)
    - [2.2 Soft Constraints ($\\mathcal{S}$) — 4–6 Total](#22-soft-constraints-mathcals--46-total)
    - [2.3 Constraint Fitness Evaluation](#23-constraint-fitness-evaluation)
  - [3. Chromosome Encoding](#3-chromosome-encoding)
  - [4. Experiments Conducted — Full Inventory](#4-experiments-conducted--full-inventory)
    - [Overview Matrix](#overview-matrix)
  - [5. GA Ablation Studies (Modes A–E)](#5-ga-ablation-studies-modes-ae)
    - [5.1 Mode A: Baseline — Pure NSGA-II](#51-mode-a-baseline--pure-nsga-ii)
    - [5.2 Mode B: Memetic — NSGA-II + Elite Bitset Repair](#52-mode-b-memetic--nsga-ii--elite-bitset-repair)
    - [5.3 Mode C: Aggressive — 2× Offspring + Full-Pop Repair](#53-mode-c-aggressive--2-offspring--full-pop-repair)
    - [5.4 Mode D: Adaptive — Stagnation-Aware Escalation](#54-mode-d-adaptive--stagnation-aware-escalation)
    - [5.5 Mode E: CP Hybrid — NSGA-II + CP-SAT Deep Polish](#55-mode-e-cp-hybrid--nsga-ii--cp-sat-deep-polish)
  - [6. RL Ablation Studies](#6-rl-ablation-studies)
    - [6.1 RL Phase 1: PPO Baseline (Minimal)](#61-rl-phase-1-ppo-baseline-minimal)
    - [6.2 RL Phase 2: DQN Baseline (Minimal)](#62-rl-phase-2-dqn-baseline-minimal)
    - [6.3 RL Phase 54: Vectorized PPO Pipeline-LLH Training](#63-rl-phase-54-vectorized-ppo-pipeline-llh-training)
    - [6.4 RL Phase 55: Capstone Thesis Run (PPO 150k)](#64-rl-phase-55-capstone-thesis-run-ppo-150k)
    - [6.5 RL Phase 55b: DQN Competitor (150k)](#65-rl-phase-55b-dqn-competitor-150k)
    - [6.6 RL Phase 56: Static LLH Differentiation Analysis](#66-rl-phase-56-static-llh-differentiation-analysis)
    - [6.7 RL Phase 56: Maskable PPO with State-Conditioned Action Masking](#67-rl-phase-56-maskable-ppo-with-state-conditioned-action-masking)
    - [6.8 RL Phase 57: PPO on Validated 6-LLH Space](#68-rl-phase-57-ppo-on-validated-6-llh-space)
    - [6.9 RL Phase 57+: Titan V1 — MaskablePPO SOTA](#69-rl-phase-57-titan-v1--maskableppo-sota)
    - [6.10 RL Phase 52: Titan V2 — Meta-Heuristic Action Space (LNS + Kempe)](#610-rl-phase-52-titan-v2--meta-heuristic-action-space-lns--kempe)
    - [6.11 RL Phase 59: Titan V3 Overclock — Policy Collapse Fix](#611-rl-phase-59-titan-v3-overclock--policy-collapse-fix)
    - [6.12 RL Phase 61: Titan V3 Parallel — 24-Core SubprocVecEnv](#612-rl-phase-61-titan-v3-parallel--24-core-subprocvecenv)
    - [6.13 RL Phase 62: Titan V4 SOTA — PBRS + Constraint Curriculum](#613-rl-phase-62-titan-v4-sota--pbrs--constraint-curriculum)
  - [7. Cross-Cutting Ablation Studies](#7-cross-cutting-ablation-studies)
    - [7.1 Random vs. PPO vs. DQN Ablation](#71-random-vs-ppo-vs-dqn-ablation)
    - [7.2 Reward Shaping Comparison (Scalar vs. Hypervolume)](#72-reward-shaping-comparison-scalar-vs-hypervolume)
    - [7.3 Adaptive vs. Fixed GA Parameter Tuning](#73-adaptive-vs-fixed-ga-parameter-tuning)
    - [7.4 Learning Rate Sensitivity Sweep](#74-learning-rate-sensitivity-sweep)
    - [7.5 Multi-Agent Coordination](#75-multi-agent-coordination)
    - [7.6 Specialist Agent Selection](#76-specialist-agent-selection)
  - [8. Baseline Comparisons \& Evaluation Protocols](#8-baseline-comparisons--evaluation-protocols)
    - [8.1 Static Baseline Evaluation (6 LLHs × 3 Seeds)](#81-static-baseline-evaluation-6-llhs--3-seeds)
    - [8.2 200-Generation Comprehensive Baseline Comparison](#82-200-generation-comprehensive-baseline-comparison)
    - [8.3 Stochastic vs. Deterministic Policy Evaluation](#83-stochastic-vs-deterministic-policy-evaluation)
  - [9. Infrastructure \& Performance Studies](#9-infrastructure--performance-studies)
    - [9.1 Pymoo vs. DEAP Framework Comparison](#91-pymoo-vs-deap-framework-comparison)
    - [9.2 Vectorized Evaluator Benchmarks](#92-vectorized-evaluator-benchmarks)
    - [9.3 Bitset Repair Operator Benchmark](#93-bitset-repair-operator-benchmark)
    - [9.4 Numba JIT Injection (Phase 74)](#94-numba-jit-injection-phase-74)
    - [9.5 Operator-Level Unit Benchmark (Elite 8)](#95-operator-level-unit-benchmark-elite-8)
  - [10. Reward Engineering Evolution](#10-reward-engineering-evolution)
    - [Phase 1–2: Simple Scalar Reward](#phase-12-simple-scalar-reward)
    - [Phase 54: Strict Acceptance](#phase-54-strict-acceptance)
    - [Phase 55: Tolerance Annealing](#phase-55-tolerance-annealing)
    - [Phase 57: Phase-Transition Reward](#phase-57-phase-transition-reward)
    - [Phase 62 (SOTA): PBRS + Curriculum](#phase-62-sota-pbrs--curriculum)
  - [11. Key Findings \& Collateral Damage Report](#11-key-findings--collateral-damage-report)
    - [The Whack-A-Mole Problem (Phase 38 Audit)](#the-whack-a-mole-problem-phase-38-audit)
    - [Pymoo vs. DEAP Decision](#pymoo-vs-deap-decision)
    - [Policy Collapse Diagnosis (Phase 59)](#policy-collapse-diagnosis-phase-59)
    - [Feasibility Verdict](#feasibility-verdict)
  - [12. What to Include in the Thesis](#12-what-to-include-in-the-thesis)
    - [Recommended Thesis Structure (Ablation Coverage)](#recommended-thesis-structure-ablation-coverage)
      - [Chapter: Methodology — Algorithm Design](#chapter-methodology--algorithm-design)
      - [Chapter: GA Ablation Study (5 Modes)](#chapter-ga-ablation-study-5-modes)
      - [Chapter: RL Hyper-Heuristic](#chapter-rl-hyper-heuristic)
      - [Chapter: Scalability \& Engineering](#chapter-scalability--engineering)
      - [Chapter: Evaluation \& Baselines](#chapter-evaluation--baselines)
      - [Chapter: Failure Analysis \& Lessons](#chapter-failure-analysis--lessons)
    - [Summary Statistics for Thesis Tables](#summary-statistics-for-thesis-tables)
    - [Key Mathematical Contributions](#key-mathematical-contributions)

---

## 1. Problem Formulation

The university course timetabling problem (UCTP) is formulated as a **bi-objective constrained optimization**:

$$\min_{\sigma \in \Sigma} \mathbf{F}(\sigma) = \big(f_{\text{hard}}(\sigma),\; f_{\text{soft}}(\sigma)\big)$$

where $\sigma: \mathcal{E} \to \mathcal{I} \times \mathcal{R} \times \mathcal{T}$ is an assignment mapping each event $e \in \mathcal{E}$ to an instructor $i \in \mathcal{I}$, a room $r \in \mathcal{R}$, and a starting time quantum $t \in \mathcal{T}$.

**Objective functions:**

$$f_{\text{hard}}(\sigma) = \sum_{c \in \mathcal{H}} w_c \cdot \text{violations}(c, \sigma)$$

$$f_{\text{soft}}(\sigma) = \sum_{c \in \mathcal{S}} w_c \cdot \text{violations}(c, \sigma)$$

where $\mathcal{H}$ is the set of hard constraints (physical impossibilities — must reach zero) and $\mathcal{S}$ is the set of soft constraints (quality preferences — minimize).

**Problem instance scale:**

| Dimension | Count |
|-----------|-------|
| Events $|\mathcal{E}|$ | ~600 (549–790 depending on preprocessing) |
| Time quanta $|\mathcal{T}|$ | 42 (6 days × 7 quanta/day) |
| Instructors $|\mathcal{I}|$ | ~120–189 |
| Rooms $|\mathcal{R}|$ | ~40–75 |
| Student groups $|\mathcal{G}|$ | ~80–92 |
| Decision variables | ~1,800 ($3 \times |\mathcal{E}|$) |

---

## 2. Constraint Taxonomy

### 2.1 Hard Constraints ($\mathcal{H}$) — 8 Total

| # | Code | Full Name | Description | Mathematical Check |
|---|------|-----------|-------------|-------------------|
| 1 | **CTE** | Cohort Temporal Exclusivity | No student group is double-booked at the same timeslot | $\forall g, q: \text{count}(g, q) \leq 1$ |
| 2 | **FTE** | Faculty Temporal Exclusivity | No instructor is double-booked | $\forall i, q: \text{count}(i, q) \leq 1$ |
| 3 | **SRE** | Spatial Resource Exclusivity | No room is double-booked | $\forall r, q: \text{count}(r, q) \leq 1$ |
| 4 | **FPC** | Faculty Pedagogical Congruence | Instructor is qualified for the assigned course | $\forall e: \sigma_I(e) \in \text{qualified}(e)$ |
| 5 | **FFC** | Facility Feature Congruence | Room type matches course requirements | $\forall e: \text{type}(\sigma_R(e)) \supseteq \text{required}(e)$ |
| 6 | **FCA** | Faculty Chronological Availability | Part-time instructors only in available quanta | $\forall e: \sigma_T(e) \subseteq \text{avail}(\sigma_I(e))$ |
| 7 | **CQF** | Curriculum Quanta Fulfillment | Each (course, group) has exactly required quanta/week | $\forall (c, g): |\text{assigned}(c, g)| = \text{required}(c)$ |
| 8 | **ICTD** | Intra-Course Temporal Dispersion | Sibling sub-sessions not on the same day | $\forall \text{siblings}(e_1, e_2): \text{day}(e_1) \neq \text{day}(e_2)$ |

### 2.2 Soft Constraints ($\mathcal{S}$) — 4–6 Total

| # | Code | Full Name | Description |
|---|------|-----------|-------------|
| 1 | **CSC** | Cohort Schedule Contiguity | Minimize gaps in student daily schedules |
| 2 | **FSC** | Faculty Schedule Contiguity | Minimize gaps in instructor daily schedules |
| 3 | **MIP** | Meridian Interval Preservation | Ensure free quanta during lunch break window |
| 4 | **SSCP** | Symmetric Sub-Cohort Parallelism | Align paired-cohort practical sessions ($|Q_{\text{left}} \oplus Q_{\text{right}}|$) |

### 2.3 Constraint Fitness Evaluation

The vectorized pipeline uses count tensors for O(1) conflict checking:

$$\text{rc}[r, q],\; \text{ic}[i, q],\; \text{gc}[g, q] \in \mathbb{Z}_{\geq 0}$$

where $r \in [0, R)$, $i \in [0, I)$, $g \in [0, G)$, $q \in [0, T)$.

**Conflict detection**: Any count $> 1$ indicates a double-booking violation. Assignments to unavailable quanta (where availability mask is `False`) are penalized $\times 100$.

---

## 3. Chromosome Encoding

**Interleaved triple encoding** — each event $e$ is represented by 3 consecutive decision variables:

$$\mathbf{X} = [I_0, R_0, T_0,\; I_1, R_1, T_1,\; \ldots,\; I_{E{-}1}, R_{E{-}1}, T_{E{-}1}]$$

| Variable | Position | Domain | Description |
|----------|----------|--------|-------------|
| $I_e$ | $\mathbf{X}[3e + 0]$ | $\text{qualified\_instructors}(e)$ | Instructor index for event $e$ |
| $R_e$ | $\mathbf{X}[3e + 1]$ | $\text{suitable\_rooms}(e)$ | Room index for event $e$ |
| $T_e$ | $\mathbf{X}[3e + 2]$ | $\text{valid\_start\_quanta}(e)$ | Start quantum for event $e$ |

**Total decision variables**: $n_{\text{var}} = 3 \times |\mathcal{E}|$ ≈ 1,800

**Constraints per variable**: Event-dependent ragged domains (not all instructors/rooms/times are valid for every event).

---

## 4. Experiments Conducted — Full Inventory

### Overview Matrix

| ID | Category | Name | Algorithm | Key Innovation |
|----|----------|------|-----------|---------------|
| GA-01 | GA Ablation | Baseline | Pure NSGA-II | Control group — no repair |
| GA-02 | GA Ablation | Memetic | NSGA-II + Elite Repair | Bitset repair on top 15% |
| GA-03 | GA Ablation | Aggressive | NSGA-II + Full Repair | 2× offspring, 15% mutation |
| GA-04 | GA Ablation | Adaptive | NSGA-II + Stagnation Escalation | Mutation rate annealing |
| GA-05 | GA Ablation | CP Hybrid | NSGA-II + CP-SAT | Matheuristic deep polish |
| RL-01 | RL Baseline | PPO Minimal | PPO (5k steps) | Proof of concept |
| RL-02a | RL Baseline | DQN Minimal | DQN (5k steps) | Value-based comparison |
| RL-02b | RL Phase 54 | PPO Pipeline-LLH | PPO (50k) | 6-action pipeline configuration |
| RL-03 | RL Capstone | PPO 150k | PPO with tolerance annealing | Exploration/exploitation split |
| RL-03b | RL Capstone Fix | PPO 150k (Fixed) | Bug-fixed capstone | Model save ordering fix |
| RL-04a | RL Competitor | DQN 150k | DQN with tolerance annealing | PPO vs. DQN comparison |
| RL-04b | RL Analysis | Specialist Agents | State-based selection | Multi-agent specialization |
| RL-05 | RL Ablation | Reward Comparison | Scalar vs. Hypervolume | Reward function study |
| RL-06a | RL Analysis | Adaptive Params | Fixed vs. RL-adaptive GA | Parameter tuning study |
| RL-06b | RL Debug | Fast Training | MaskablePPO (10k) | Emergency speed script |
| RL-06c | RL Phase 56 | LLH Differentiation | Static analysis | Proves LLH viability |
| RL-06d | RL Debug | Single Env | MaskablePPO (20k) | Fixes vectorization bug |
| RL-06e | RL Phase 56 | Maskable PPO | MaskablePPO (50k) | State-conditioned masking |
| RL-07a | RL Ablation | Random/PPO/DQN | Multi-method × 5 trials | Statistical ablation |
| RL-07b | RL Phase 57 | PPO Phase 57 | PPO (2.5k) | On validated 6-LLH space |
| RL-07c | RL Phase 57+ | Titan V1 | MaskablePPO (100k) | Micro-memetic Elite 8 |
| RL-08a | RL Ablation | LR Sweep | PPO [1e-4, 3e-4, 1e-3] | Hyperparameter sensitivity |
| RL-08b | RL Phase 59 | Titan V3 Overclock | PPO (50k) | Policy collapse fix (20×) |
| RL-08c | RL Phase 61 | Titan V3 Parallel | MaskablePPO (50k) | 24-core SubprocVecEnv |
| RL-09a | RL Analysis | Multi-Agent | State-based coordination | Multi-agent dynamics |
| RL-09b | RL Phase 52 | Titan V2 Meta | MaskablePPO (100k) | LNS + Kempe Chain |
| RL-09c | RL Phase 62 | Titan V4 SOTA | MaskablePPO (100k) | PBRS + Curriculum |
| RL-10 | Verification | Component Check | N/A | Smoke test |
| EVAL-01 | Evaluation | Static Baselines | 6 LLHs × 3 seeds | Single-operator ceilings |
| EVAL-02 | Evaluation | 200-Gen Baselines | PPO/Random/RR/UCB1 × 200 | Comprehensive comparison |
| EVAL-03 | Evaluation | Stochastic vs. Det. | Policy distribution forensics | Deterministic collapse diagnosis |
| BENCH-01 | Infrastructure | Pymoo vs. DEAP | Framework comparison | Quality vs. speed tradeoff |
| BENCH-02 | Infrastructure | Vectorized Eval | NumPy batch evaluation | 5–20× speedup |
| BENCH-03 | Infrastructure | Bitset Repair | Numba JIT repair operator | 4.6× speedup |
| BENCH-04 | Infrastructure | Elite 8 Operators | Per-operator unit test | Operator validation |

---

## 5. GA Ablation Studies (Modes A–E)

### 5.1 Mode A: Baseline — Pure NSGA-II

**Script**: `runs/ga_01_baseline.py` → `BaselineExperiment`

**Methodology**: Standard NSGA-II (Deb et al., 2002) multi-objective evolutionary algorithm with **no repair operator**. This serves as the control group to isolate the contribution of local search / repair.

**Algorithm**:

1. **Initialization**: `RandomDomainSampling` — vectorized random valid assignments
2. **Selection**: NSGA-II binary tournament with crowding distance
3. **Crossover**: `EventBlockCrossover` — position-independent per-event triple swap (preserves course/group structure)
4. **Mutation**: `EventLocalMutation` — constraint-aware per-gene mutation (instructor: 70% qualified bias; room: capacity + feature check; time: conflict-aware slot selection)
5. **Survival**: NSGA-II non-dominated sorting + crowding distance

**Key Parameters**:

| Parameter | Value |
|-----------|-------|
| Population size | 100 |
| Generations | 200 |
| Crossover probability | 0.5 per event |
| Mutation probability | 0.05 per event |
| Seed | 42 |
| Repair | **None** |

**Mathematical Operators**:

$$\text{Crossover}: \text{For each event } e, \text{ swap } (I_e, R_e, T_e) \text{ from parent A or B with } p = 0.5$$

$$\text{Mutation}: \text{For each gene } g, \text{ with } p_m = 0.05: g' \sim \text{Uniform}(\mathcal{D}_g)$$

Subject to validity guards: `_is_swap_valid()` checks instructor qualification and room suitability before accepting crossover swaps.

**Purpose in Thesis**: Establishes the **lower bound** of what pure evolutionary search can achieve without any domain-specific repair knowledge. All other modes are compared against this baseline.

---

### 5.2 Mode B: Memetic — NSGA-II + Elite Bitset Repair

**Script**: `runs/ga_02_memetic.py` → `MemeticExperiment`

**Methodology**: Memetic Algorithm (MA) combining NSGA-II with a **bitset-based greedy repair operator** applied to the elite fraction of the population. This is the key contribution — hybridizing global search with domain-specific local search.

**Algorithm**:

1. Same NSGA-II base as Mode A
2. **Every `REPAIR_FREQUENCY` generations** (every 5th):
   - Select top `ELITE_PCT` (15%) individuals by hard penalty
   - For each elite individual, apply `BitsetSchedulingRepair` for `REPAIR_ITERS` (8) passes
   - Re-evaluate repaired individuals via `_reeval_modified()` (clears stale Pymoo fitness cache)
   - Repaired individuals replace originals in population
3. Repair is parallelized via `ProcessPoolExecutor` (each worker instantiates own `BitsetSchedulingRepair`)

**Bitset Repair Operator — 3-Stage Pipeline**:

**Stage 1: Domain Clamping** — $O(E)$

$$\forall e: (I_e, R_e, T_e) \leftarrow \text{clamp}(I_e, R_e, T_e) \text{ into } \mathcal{D}_e^{\text{inst}} \times \mathcal{D}_e^{\text{room}} \times \mathcal{D}_e^{\text{time}}$$

**Stage 2: Conflict Resolution** — Greedy Remove/Re-place

For each event $e$ with conflicts ($\text{count}(r_e, q) > 1$ or $\text{count}(i_e, q) > 1$):

- Remove $e$ from count maps
- Build cost matrix $\mathbf{C} \in \mathbb{R}^{|\mathcal{T}_e| \times |\mathcal{R}_e|}$ per instructor candidate $i$:

$$C[t, r] = \sum_{q=t}^{t+d_e-1} \left[\text{rc}[r, q] + \text{ic}[i, q] + \sum_{g \in \mathcal{G}_e} \text{gc}[g, q]\right] + 100 \cdot \mathbf{1}[\neg \text{avail}(i, q)]$$

- Place at $\arg\min C$ (greedy best-fit)
- Paired practicals: place simultaneously via joint cost matrix scouring both lecture + lab room types

**Stage 3: Group Deconfliction**

For each group $g$ with conflicts: remove all conflicting events, re-insert longest-first (greedy).

**Key Parameters**:

| Parameter | Value |
|-----------|-------|
| Population size | 120 |
| Generations | 50 |
| Crossover probability | 0.4 |
| Mutation probability | 0.10 |
| Elite fraction | 15% (18 individuals) |
| Repair passes per elite | 8 |
| Repair frequency | Every 5th generation |
| Parallelization | ProcessPoolExecutor |

**Amortized Complexity per Event**:

$$O\big(|\mathcal{I}_e| \cdot (|\mathcal{T}_e| \cdot d_e + |\mathcal{R}_e|)\big)$$

**HPC Optimizations**:

- Count arrays are `int16` (~18 KiB → fits L1 cache for reference instance)
- `_find_placement` builds full cost matrix via NumPy fancy indexing — no Python loop
- Numba JIT (`@njit(cache=True, nogil=True)`) on inner functions: `_numba_add`, `_numba_remove`, `_numba_count_conflicts`, `_numba_check_placement`, `_numba_build_counts`
- Bitset availability masks (`uint64`) for $O(1)$ population-count checks

**Purpose in Thesis**: Demonstrates the **impact of domain-specific repair** on evolutionary search quality. The 4.61× speedup from bitset repair (vs. original repair) and the quality improvement from elite local search are the primary contributions.

---

### 5.3 Mode C: Aggressive — 2× Offspring + Full-Pop Repair

**Script**: `runs/ga_03_aggressive.py` → `AggressiveExperiment`

**Methodology**: Trades compute budget for rapid constraint reduction by applying repair to **every individual** (not just elites) and generating 2× offspring per generation.

**Algorithm**:

1. NSGA-II with 2× offspring generation ($\lambda = 2 \times \mu$)
2. High mutation rate (15%) for aggressive exploration
3. Full-population repair (every individual, every generation)
4. Fewer generations (100) due to heavy per-generation compute

**Key Parameters**:

| Parameter | Value | vs. Baseline |
|-----------|-------|-------------|
| Population size | 200 | 2× |
| Generations | 100 | 0.5× |
| Offspring multiplier | 2.0 | 2× |
| Crossover probability | 0.7 | 1.4× |
| Mutation probability | 0.15 | 3× |
| Repair coverage | 100% population | vs. 15% elite |
| Repair passes | 3 | vs. 8 (fewer but broader) |

**Purpose in Thesis**: Ablates the **breadth vs. depth tradeoff** in local search. Mode B applies deep repair (8 passes) to few (15%), while Mode C applies shallow repair (3 passes) to all (100%). This isolates whether broad repair coverage or deep repair intensity is more effective.

---

### 5.4 Mode D: Adaptive — Stagnation-Aware Escalation

**Script**: `runs/ga_04_adaptive.py` → `AdaptiveExperiment`

**Methodology**: Self-adaptive parameter control inspired by reactive search optimization (Battiti & Tecchiolli, 1994). Starts conservative and escalates mutation + repair intensity when the search stagnates.

**Algorithm**:

1. **Normal phase**: Standard NSGA-II with low mutation (5%), no repair
2. **Stagnation detection**: If `best_hard` does not improve for `STAGNATION_WINDOW` (15) consecutive generations:
   - **Escalate**: mutation rate → 20%, activate elite repair (top 10%, 5 passes)
3. **De-escalation**: When improvement resumes, revert to normal phase

**State Machine**:

```
┌──────────────┐   15 gens no improvement   ┌──────────────┐
│   NORMAL     │ ────────────────────────→   │  ESCALATED   │
│  mut=0.05    │                             │  mut=0.20    │
│  repair=OFF  │   ←──────────────────────── │  repair=ON   │
└──────────────┘     improvement detected    │  elite=10%   │
                                             └──────────────┘
```

**Key Parameters**:

| Parameter | Normal | Escalated |
|-----------|--------|-----------|
| Mutation probability | 0.05 | 0.20 |
| Elite repair | OFF | ON (10%, 5 passes) |
| Generations | 300 (long adaptive run) | — |
| Stagnation window | 15 generations | — |

**Purpose in Thesis**: Tests whether **adaptive parameter control** can outperform static configurations. If Mode D matches or exceeds the best-performing static mode, it supports the thesis that parameter adaptation is valuable for this problem class.

---

### 5.5 Mode E: CP Hybrid — NSGA-II + CP-SAT Deep Polish

**Script**: `runs/ga_05_cp_hybrid.py` → `CPHybridExperiment`

**Methodology**: Matheuristic approach combining NSGA-II with periodic **exact constraint satisfaction** via Google OR-Tools CP-SAT solver. The GA explores globally; CP-SAT locally polishes the best solution to feasibility.

**Algorithm**:

1. Run NSGA-II normally
2. Every `CP_INTERVAL` (10) generations:
   - Extract the best individual (lowest hard penalty)
   - Convert chromosome → SessionGene representation
   - Formulate as a CP-SAT model with all hard constraints
   - Solve with `CP_TIMEOUT` (30 seconds) time limit
   - If improved: write repaired chromosome back into GA population
3. Continue NSGA-II with the improved population

**CP-SAT Mathematical Formulation**:

$$\text{Minimize } \sum_{c \in \mathcal{H}} \text{violations}(c)$$

Subject to:
$$\forall g \in \mathcal{G}, q \in \mathcal{T}: \sum_{e \in \mathcal{E}_g} x_{e,q} \leq 1 \quad \text{(CTE)}$$
$$\forall i \in \mathcal{I}, q \in \mathcal{T}: \sum_{e: \sigma_I(e) = i} x_{e,q} \leq 1 \quad \text{(FTE)}$$
$$\forall r \in \mathcal{R}, q \in \mathcal{T}: \sum_{e: \sigma_R(e) = r} x_{e,q} \leq 1 \quad \text{(SRE)}$$

**Key Parameters**:

| Parameter | Value |
|-----------|-------|
| Population size | 60 (smaller — CP is expensive) |
| Generations | 100 |
| CP-SAT interval | Every 10 generations |
| CP-SAT timeout | 30 seconds per invocation |
| Total CP calls | ~10 (100/10) over the run |

**Purpose in Thesis**: Demonstrates the **matheuristic paradigm** — can an exact solver complement evolutionary search? This ablates the value of deterministic satisfaction-based repair vs. the heuristic greedy repair of Modes B and C.

**Requirements**: `pip install ortools>=9.8`

---

## 6. RL Ablation Studies

### 6.1 RL Phase 1: PPO Baseline (Minimal)

**Script**: `runs/rl_01_train_ppo.py` → `RLTrainExperiment(agent_type="ppo")`

**Methodology**: Minimal proof-of-concept PPO agent training on the hyper-heuristic environment. Validates that the RL-GA interface works and PPO can produce a non-trivial policy.

**Algorithm**: Proximal Policy Optimization (Schulman et al., 2017)

**PPO Objective**:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1{-}\epsilon, 1{+}\epsilon) \hat{A}_t \right) \right]$$

where $r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$ and $\epsilon = 0.2$.

| Parameter | Value |
|-----------|-------|
| Timesteps | 5,000 |
| Population size | 20 |
| Max generations | 50 |
| Seed | 42 |

---

### 6.2 RL Phase 2: DQN Baseline (Minimal)

**Script**: `runs/rl_02_train_dqn.py` → `RLTrainExperiment(agent_type="dqn")`

**Methodology**: Identical setup to RL-01 but using Deep Q-Network (Mnih et al., 2015):

$$Q(s, a; \theta) \leftarrow Q(s, a; \theta) + \alpha \left[ r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right]$$

where $\theta^-$ is the target network, updated every 1,000 steps.

| Parameter | Value |
|-----------|-------|
| Timesteps | 5,000 |
| Algorithm | DQN |
| Target update interval | 1,000 |

**Purpose**: On-policy (PPO) vs. off-policy (DQN) comparison at minimal scale.

---

### 6.3 RL Phase 54: Vectorized PPO Pipeline-LLH Training

**Script**: `runs/rl_02_train_vectorized.py`

**Methodology**: End-to-end pipeline training PPO on the **6-action pipeline-configuration LLH space** (the redesigned action space where each action configures a different repair pipeline configuration rather than selecting individual heuristics).

**6 Low-Level Heuristics (LLH Actions)**:

| Action | Name | Elite % | Passes | Strategy |
|--------|------|---------|--------|----------|
| 0 | ConservativeRepair | 10% | 2 | Steady exploitation |
| 1 | AggressiveRepair | 25% | 3 | Aggressive exploration |
| 2 | MemeticEliteRepair | 15% | 4 | Memetic GA clone |
| 3 | SoftFocusRepair | 8% | 2 + compact | Soft objective focus |
| 4 | DestructiveConstructive | 20% | 2 + ruin 10% | Escape local optima |
| 5 | IntensifiedRepair | 20% | 3 | Balanced workhorse |

**Key Parameters**:

| Parameter | Value |
|-----------|-------|
| Algorithm | PPO (SB3) |
| Timesteps | 50,000 |
| Network architecture | MLP [64, 64] |
| Learning rate | 3e-4 |
| Clip range | 0.2 |
| Population size | 120 |
| Max generations per episode | 50 |
| Acceptance tolerance | 0.0 (strict) |

**Observation Space** — `Box(39,)` ∈ [0, 1]:

| Indices | Count | Features |
|---------|-------|----------|
| 0–4 | 5 | Fitness stats (min, max, mean, std, ptp on $F_{:,0}$) |
| 5–7 | 3 | Constraint violation stats (mean total, max total, frac feasible) |
| 8–12 | 5 | Diversity (pairwise distances via `scipy.pdist`) |
| 13–24 | 12 | Constraint breakdown (8 hard + 4 soft means) |
| 25–28 | 4 | Progress (gen/max_gen, stagnation, convergence rate, feasibility gain) |
| 29–38 | 10 | Heuristic history (last 10 action IDs) |

**Output**: Per-episode CSV, per-step CSV, 4 publication-ready PDF figures, baseline comparison table.

---

### 6.4 RL Phase 55: Capstone Thesis Run (PPO 150k)

**Script**: `runs/rl_03_capstone_thesis.py`

**Methodology**: Full-scale PPO training with **tolerance annealing** — the key insight that separates exploration (training) from exploitation (evaluation).

**Tolerance Annealing**:

During **training** ($T_{\text{accept}} = 10.0$): the environment accepts moves that increase hard penalties by up to 10 points, allowing the agent to cross fitness valleys:

$$\text{accept}(s \to s') \iff f_{\text{hard}}(s') \leq f_{\text{hard}}(s) + T_{\text{accept}}$$

During **evaluation** ($T_{\text{accept}} = 0.0$): strict mode — reject any hard penalty increase.

**Key Parameters**:

| Parameter | Training | Evaluation |
|-----------|----------|------------|
| Timesteps | 150,000 | — |
| Generations/episode | 50 | 200 |
| Tolerance | 10.0 | 0.0 |
| Population size | 120 | 120 |

**Output**: Model saved to `output/models/ppo_capstone_final.zip`. Heuristic Efficacy Matrix (per-action $\Delta_{\text{hard}}$, $\Delta_{\text{soft}}$).

---

### 6.5 RL Phase 55b: DQN Competitor (150k)

**Script**: `runs/rl_04_train_dqn.py`

**Methodology**: Full DQN training mirroring the PPO capstone to enable **PPO vs. DQN** comparison at scale.

**DQN-Specific Parameters**:

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 (lower than PPO's 3e-4) |
| Replay buffer size | 100,000 |
| Batch size | 32 |
| Target update interval | 1,000 steps |
| Exploration: $\epsilon$-fraction | 10% of training |
| Final $\epsilon$ | 0.05 |

**Purpose**: Compares on-policy actor-critic (PPO) vs. off-policy value-based (DQN) with identical training budgets and environment configurations.

---

### 6.6 RL Phase 56: Static LLH Differentiation Analysis

**Script**: `runs/rl_06_llh_differentiation.py`

**Methodology**: **Critical viability check** — before investing in RL training, prove that no single LLH dominates all others across all generations. If one LLH always wins, the RL experiment is pointless.

**Protocol**: Run each of the 6 LLHs statically (same action every generation) for 50 generations. At checkpoints (gen 5, 25, 50), compare all 6.

**Three Differentiation Questions**:

| # | Question | Method | Threshold |
|---|----------|--------|-----------|
| Q1 | Is there a generation where a non-Conservative LLH wins on hard? | $\min_a \text{hard}(a, g) \neq \text{hard}(0, g)$ for any $g$ | Any gen |
| Q2 | Do LLHs converge to different soft values? | Range $> 50$ or CV $> 0.05$ at gen 50 | Either |
| Q3 | Does any LLH escape a plateau others get stuck on? | 10+ consecutive gens without improvement in one, not others | Stagnation detection |

**Implication**: If all three questions are **Yes**, the RL hyper-heuristic has theoretical justification — different actions are optimal at different search stages.

---

### 6.7 RL Phase 56: Maskable PPO with State-Conditioned Action Masking

**Script**: `runs/rl_06_train_maskable_ppo.py`

**Methodology**: Implements **state-conditioned action masking** using `sb3-contrib.MaskablePPO`. When hard constraints are violated (`best_hard > 0`), soft-focused actions are blocked; when feasible, all actions are available.

**Action Masking Logic**:

$$\text{mask}(a) = \begin{cases} 1 & \text{if } a \notin \{3, 7\} \text{ (always available)} \\ 1 & \text{if } a \in \{3, 7\} \text{ and } f_{\text{hard}} = 0 \\ 0 & \text{if } a \in \{3, 7\} \text{ and } f_{\text{hard}} > 0 \end{cases}$$

Actions 3 (SymmetricSubcohortSync) and 7 (MeridianCompaction) are blocked when the schedule is infeasible, forcing the agent to focus on hard constraint repair first.

**Modified PPO Objective** (with invalid action masking):

$$\pi_\theta(a | s) = \frac{\exp(z_a) \cdot \text{mask}(a)}{\sum_{a'} \exp(z_{a'}) \cdot \text{mask}(a')}$$

**Key Parameters**:

| Parameter | Value |
|-----------|-------|
| Algorithm | MaskablePPO (sb3-contrib) |
| Timesteps | 50,000 |
| Acceptance tolerance | 5.0 |
| Network | MLP [64, 64] |

---

### 6.8 RL Phase 57: PPO on Validated 6-LLH Space

**Script**: `runs/rl_07_ppo_phase57.py`

**Methodology**: First PPO training on the **validated** 6-LLH action space (viability proven by Phase 56 differentiation analysis). Trains PPO, then evaluates deterministic policy against all 6 static baselines.

**Phase-Transition Reward** (Phase 57 redesign):

$$R_t = \begin{cases} 2\Delta_{\text{hard}} + 0.5 \frac{\Delta_{\text{soft}}}{\text{norm}_{\text{soft}}} & \text{if } \text{prev\_hard} < 100 \\ \frac{\Delta_{\text{hard}}}{\text{norm}_{\text{hard}}} + 0.1 \frac{\Delta_{\text{soft}}}{\text{norm}_{\text{soft}}} & \text{otherwise} \end{cases} + \text{feasibility\_bonus}$$

where $\Delta = \text{prev\_best} - \text{best}$ (positive = improvement), clipped to $[-10, 10]$.

**Pareto Dominance Check**:

$$\text{PPO dominates BL} \iff \left(h_{\text{PPO}} \leq h_{\text{BL}} \land s_{\text{PPO}} < s_{\text{BL}}\right) \lor \left(h_{\text{PPO}} < h_{\text{BL}} \land s_{\text{PPO}} \leq s_{\text{BL}}\right)$$

**Key Parameters**:

| Parameter | Value |
|-----------|-------|
| Timesteps | 2,500 (~104 episodes) |
| n_steps | 24 (1 full episode) |
| batch_size | 24 (full episode per mini-batch) |
| n_epochs | 10 |
| ent_coef | 0.01 |
| Network | MLP [64, 64] |

**Diagnostic**: Phase analysis — early (gen 0–8), mid (gen 9–16), late (gen 17–24) action mode frequency to detect phase-dependent policy behavior.

---

### 6.9 RL Phase 57+: Titan V1 — MaskablePPO SOTA

**Script**: `runs/rl_07_titan_maskable.py`

**Methodology**: Definitive 100k-step MaskablePPO with restored **Micro-Memetic Elite 8** action space (3 sophisticated meta-heuristics beyond simple repair).

**Elite 8 Action Space** — `Discrete(8)`:

| Action | Heuristic | Mathematical Operation |
|--------|-----------|----------------------|
| 0 | SpatialResourceProjection | Conflict-directed $k{=}5$ greedy room bursts |
| 1 | FacultyTemporalProjection | Instructor clash repair |
| 2 | CohortTemporalProjection | Group conflict resolution |
| 3 | SymmetricSubcohortSync | Paired practical alignment ($|Q_L \oplus Q_R|$ minimization) |
| 4 | UniversalFeasibilityProjection | Bounded depth-3 ejection chains |
| 5 | StochasticQuantaPerturbation | Random time-slot exploration |
| 6 | StochasticSpatialPerturbation | Random room exploration |
| 7 | MeridianCompactionHeuristic | Feasibility-gated soft optimizer (lunch break preservation) |

**Key Parameters**:

| Parameter | Value |
|-----------|-------|
| Algorithm | MaskablePPO |
| Timesteps | 100,000 |
| Population size | 120 (full) |
| Max generations | 50 |
| lr | 3e-4 |
| clip_range | 0.2 |
| n_steps | 2048 |
| batch_size | 64 |
| ent_coef | 0.01 |
| tolerance | 5.0 |

---

### 6.10 RL Phase 52: Titan V2 — Meta-Heuristic Action Space (LNS + Kempe)

**Script**: `runs/rl_09_titan_v2_meta.py`

**Methodology**: Replaces two operators in the Discrete(8) action space with sophisticated meta-heuristics: **Large Neighborhood Search (LNS)** and **Kempe Chain Interchange**.

**New Operators**:

**Action 2 — LNS Ruin & Recreate**:

- **Ruin**: Destroy top 5% worst events (by conflict count)
- **Recreate**: Greedy best-fit reinsertion ordered by domain restrictiveness (most constrained first)

$$\text{Ruin}(\sigma) = \sigma \setminus \{e : \text{conflicts}(e) \geq \text{quantile}_{0.95}(\text{conflicts})\}$$
$$\text{Recreate}(\sigma') = \text{GreedyInsert}(\sigma', \text{sort}(\text{ruined}, \text{key=restrictiveness}))$$

**Action 5 — Kempe Chain Interchange**:

- Construct bipartite time-slot sub-graphs
- Trace conflict-density-weighted cascades
- Swap events along Kempe chains to resolve temporal conflicts

$$\text{KempeChain}(q_1, q_2) = \{e : e \text{ forms alternating path between slots } q_1 \text{ and } q_2\}$$

**Key Parameters**: Same as Titan V1 but with LNS and Kempe replacing generic operators at positions 2 and 5.

---

### 6.11 RL Phase 59: Titan V3 Overclock — Policy Collapse Fix

**Script**: `runs/rl_08_titan_v3_overclock.py`

**Methodology**: Forensic diagnosis revealed Phase 57 PPO produced a **near-uniform policy** (almost equal probability for all actions) due to severe under-training: only 2,500 timesteps for a 39-D observation space. This script applies the fix: 20× training budget + aggressive hyperparameters.

**Diagnosis**:

$$\text{Policy entropy} = -\sum_a \pi(a|s) \log \pi(a|s) \approx \log(6) \implies \text{near-uniform}$$

**Prescription**:

- Training budget: 2,500 → 50,000 (20×)
- Learning rate: 3e-4 → **5e-4** (more aggressive gradient steps)
- Entropy coefficient: 0.01 → **0.05** (5× — prevent early policy collapse by keeping exploration high)
- Rollout buffer: 24 → **128** steps (multiple episodes per update for stable gradients)

**Key Parameters**:

| Parameter | Phase 57 | V3 Overclock | Change |
|-----------|----------|-------------|--------|
| Timesteps | 2,500 | 50,000 | **20×** |
| lr | 3e-4 | 5e-4 | **1.67×** |
| ent_coef | 0.01 | 0.05 | **5×** |
| n_steps | 24 | 128 | **5.3×** |
| batch_size | 24 | 128 | **5.3×** |

**Estimated wall-clock**: ~87 hours (~3.6 days) for single-core.

---

### 6.12 RL Phase 61: Titan V3 Parallel — 24-Core SubprocVecEnv

**Script**: `runs/rl_08_titan_v3_parallel.py`

**Methodology**: Exploits the 32-core / 128GB machine by running **24 parallel Pymoo environments** via `SubprocVecEnv`. Each subprocess independently loads the scheduling problem and runs its own GA population, feeding experience to a single shared MaskablePPO brain.

**Architecture**:

```
                    ┌──────────────┐
                    │  MaskablePPO │  (single shared network)
                    │  π(a|s; θ)   │
                    └──────┬───────┘
                           │ actions
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Env #0  │ │  Env #1  │ │ Env #23  │  ← 24 SubprocVecEnv workers
        │  seed=42 │ │  seed=43 │ │  seed=65 │
        │  NSGA-II │ │  NSGA-II │ │  NSGA-II │
        └──────────┘ └──────────┘ └──────────┘
```

**Rollout Math**:

$$\text{Buffer size per rollout} = 24 \times 128 = 3{,}072 \text{ steps}$$
$$\text{Total rollout updates} = \frac{50{,}000}{3{,}072} \approx 17$$
$$\text{Wall-clock estimate} = 17 \times 640\text{s} \approx 3.0 \text{ hours}$$

**Key Parameters**:

| Parameter | Value |
|-----------|-------|
| Algorithm | MaskablePPO |
| Timesteps | 50,000 |
| Parallel envs | 24 (SubprocVecEnv, spawn) |
| n_steps | 128 per env |
| batch_size | 512 (from 3,072 buffer) |
| lr | 5e-4 |
| ent_coef | 0.05 |
| Workers | `run_preflight=False` (skip redundant feasibility checks) |
| Seed diversity | seed + rank per worker |

---

### 6.13 RL Phase 62: Titan V4 SOTA — PBRS + Constraint Curriculum

**Script**: `runs/rl_09_titan_v4_sota.py`

**Methodology**: The most advanced configuration. Builds on Titan V3 Parallel (24-core SubprocVecEnv) and adds two major algorithmic upgrades:

1. **Potential-Based Reward Shaping (PBRS)** — Ng et al. (1999)
2. **Constraint Curriculum** — 3-phase progressive complexity

**PBRS Theory**: Guarantees policy invariance under potential shaping:

$$R_{\text{shaped}}(s, a, s') = R(s, a, s') + \gamma \Phi(s') - \Phi(s)$$

**Potential Function** — Bottleneck Density:

$$\Phi(s) = -\frac{\text{Var}(g_0, \ldots, g_7) + \text{Var}(\text{per-instructor conflicts}) + \text{Var}(\text{per-room conflicts})}{\text{max\_var}}$$

$\Phi \in [-1, 0]$: higher (closer to 0) = conflicts spread evenly across resources = easier to fix. Dynamic normalization tracks maximum variance observed.

**Intuition**: Even when $\Delta_{\text{hard}} \approx 0$ (convergence plateau), if the **distribution** of conflicts becomes more uniform, the potential increases, providing dense gradient signal.

**Three Tiers**:

- **Tier 0**: $O(8)$ — variance across 8 hard constraint G-row columns
- **Tier 1**: $O(E)$ — per-instructor conflict count variance from chromosome expansion
- **Tier 2**: $O(E)$ — per-room conflict count variance (enabled by `use_chromosome_potential=True`)

**Constraint Curriculum** — 3-Phase Schedule:

| Phase | Episodes | Active Constraints | Learning Focus |
|-------|----------|--------------------|----------------|
| 1 | 0 → 21 | SRE, FFC | Room conflict resolution |
| 2 | 21 → 63 | + FTE, FPC, FCA | Spatio-temporal resolution |
| 3 | 63 → ∞ | + CTE, CQF, ICTD | Full NP-hard complexity |

Approximate split: ~25% Phase 1 / ~49% Phase 2 / ~26% Phase 3.

**Smooth Phase Blending** (avoids reward distribution discontinuity):

$$\alpha = \frac{\text{episodes\_since\_transition}}{\text{BLEND\_WINDOW}} \quad (\text{BLEND\_WINDOW} = 5)$$

$$r_{\text{curriculum}} = \sum_{c \in \mathcal{A}(\text{phase})} \alpha \cdot w_c \cdot (\text{cv}_c^{\text{prev}} - \text{cv}_c^{\text{curr}})$$

**Total Shaped Reward**:

$$\boxed{R_{\text{total}} = R_{\text{base}} + \underbrace{\gamma \Phi(s') - \Phi(s)}_{\text{PBRS}} + \underbrace{\sum_{c \in \mathcal{A}} w_c \cdot \Delta_{\text{cv}_c}}_{\text{Curriculum}}}$$

Clipped to $[-15, +15]$.

**Key Design Decisions**:

- **Reward-weight modulation, not constraint masking**: The GA always evaluates all 8 hard constraints (required for NSGA-II fitness). Only the RL reward view changes.
- **CUDA acceleration**: PPO forward pass runs on GPU (RTX 4060), environment stepping on CPU.

**Key Parameters**:

| Parameter | Value |
|-----------|-------|
| Algorithm | MaskablePPO |
| Total timesteps | 100,000 (2× V3) |
| Parallel envs | 24 |
| n_steps | 128 per env |
| batch_size | 512 |
| lr | 5e-4 |
| ent_coef | 0.05 |
| $\gamma$ (PPO + PBRS) | 0.99 |
| Curriculum weight | 0.5 |
| Phase 1 end | Episode 21 |
| Phase 2 end | Episode 63 |
| Blend window | 5 episodes |

---

## 7. Cross-Cutting Ablation Studies

### 7.1 Random vs. PPO vs. DQN Ablation

**Script**: `runs/rl_07_ablation.py` → `RLAblationExperiment`

**Methodology**: Systematic ablation comparing three agent types with statistical validity through 5 independent trials.

| Method | Agent | Description |
|--------|-------|-------------|
| Random | Random action | Uniform $a \sim \text{Uniform}(\{0, \ldots, 5\})$ — control baseline |
| PPO | Proximal Policy Optimization | On-policy actor-critic |
| DQN | Deep Q-Network | Off-policy value-based |

**Protocol**: 3 methods × 5 trials × 3,000 timesteps × 50-gen evaluation

**Purpose**: Validates the fundamental hypothesis: *does learning a policy outperform random action selection?* And *which RL paradigm (on-policy vs. off-policy) is better suited for hyper-heuristic scheduling?*

---

### 7.2 Reward Shaping Comparison (Scalar vs. Hypervolume)

**Script**: `runs/rl_05_compare_rewards.py` → `RLRewardCompareExperiment`

**Two Reward Formulations**:

**Scalar Reward**:
$$R_{\text{scalar}} = w_1 \cdot \frac{\Delta f_{\text{hard}}}{\|f_{\text{hard}}\|} + w_2 \cdot \text{diversity\_bonus} - w_3 \cdot \text{time\_penalty}$$

**Hypervolume-Based Reward**:
$$R_{\text{HV}} = w_1 \cdot \Delta\text{HV}(\mathcal{P}, \mathbf{r}) + w_2 \cdot \text{diversity\_bonus} - w_3 \cdot \text{time\_penalty}$$

where $\text{HV}(\mathcal{P}, \mathbf{r}) = \text{vol}(\{y \in \mathbb{R}^2 : \exists p \in \mathcal{P}, p \prec y \prec \mathbf{r}\})$ is the hypervolume indicator.

**Purpose**: Tests whether a multi-objective quality signal (hypervolume) provides better gradient than scalarized fitness improvement.

---

### 7.3 Adaptive vs. Fixed GA Parameter Tuning

**Script**: `runs/rl_06_adaptive_params.py` → `RLAdaptiveParamsExperiment`

**Methodology**: Compares whether an RL agent can learn to dynamically tune GA parameters (crossover rate, mutation rate, population size) to outperform the best static configuration.

**Purpose**: Tests the meta-learning hypothesis — can RL be used not just to select heuristics but to control the underlying GA's hyperparameters?

---

### 7.4 Learning Rate Sensitivity Sweep

**Script**: `runs/rl_08_hyperparam_sweep.py` → `RLHyperparamSweepExperiment`

**Methodology**: Grid sweep across 3 learning rates to characterize PPO sensitivity in the scheduling domain.

| Config | Learning Rate | Expected Behavior |
|--------|--------------|-------------------|
| Low | 1e-4 | Stable but slow convergence |
| Medium (default) | 3e-4 | Balanced |
| High | 1e-3 | Fast but potentially unstable |

Each configuration trains for 3,000 timesteps with pop_size=20, max_gens=40.

---

### 7.5 Multi-Agent Coordination

**Script**: `runs/rl_09_multi_agent.py` → `RLMultiAgentExperiment`

**Methodology**: Analyzes coordination dynamics in a multi-agent scheduling setup where multiple agents simultaneously select heuristics using state-based coordination strategy.

**Key Parameters**: 10 episodes × 15 steps × pop_size=20.

**Purpose**: Tests whether cooperative multi-agent approaches improve over single-agent hyper-heuristic selection.

---

### 7.6 Specialist Agent Selection

**Script**: `runs/rl_04_train_specialist.py` → `RLSpecialistExperiment`

**Methodology**: State-based specialist agent selection — trains multiple specialist agents, each optimized for a different search state (e.g., high-conflict vs. near-feasible), and a meta-controller that routes to the appropriate specialist based on current state.

**Selection Logic**:
$$a = \text{specialist}(\text{state\_category}(s))$$

where `state_category` classifies the current observation into qualitative search phases.

---

## 8. Baseline Comparisons & Evaluation Protocols

### 8.1 Static Baseline Evaluation (6 LLHs × 3 Seeds)

**Script**: `runs/rl_03_static_baselines.py`

**Methodology**: For each action $a \in \{0, \ldots, 5\}$ and each seed $\in \{42, 123, 7\}$: reset the environment, apply action $a$ at every generation for 50 generations.

**Purpose**: Establishes the **single-operator performance ceiling** — the best any static strategy can achieve. The RL agent must outperform the best static baseline to justify its complexity.

| LLH | Action | Strategy | Elite % | Passes |
|-----|--------|----------|---------|--------|
| ConservativeRepair | 0 | `repair_batch(passes=3)` | 10% | 3 |
| AggressiveRepair | 1 | `repair_batch(passes=7)` | 25% | 7 |
| MemeticEliteRepair | 2 | 3 passes + 4 extra on worst 15% | 15% | 3+4 |
| SoftFocusRepair | 3 | 3 passes + time-compaction | 8% | 3 |
| DestructiveConstructive | 4 | ruin 10% + 5-pass rebuild | 20% | 5 |
| IntensifiedRepair | 5 | `repair_batch(passes=5)` | 20% | 5 |

**Output**: `output/rl_phase54/static_baselines.csv`

---

### 8.2 200-Generation Comprehensive Baseline Comparison

**Script**: `runs/eval_all_baselines.py`

**Methodology**: Runs 4 action-selection strategies through 200 generations and saves per-generation per-constraint trajectories.

**Strategies Compared**:

| Strategy | Algorithm | Mathematical Selection Rule |
|----------|-----------|---------------------------|
| **PPO** | Trained model | $a = \pi_\theta(s)$ (deterministic argmax) |
| **Random** | Uniform random | $a \sim \text{Uniform}(\{0, \ldots, 7\})$ |
| **Round-Robin** | Cyclic | $a = \text{gen} \mod 8$ |
| **UCB1** | Upper Confidence Bound | $a = \arg\max_a \left[Q(a) + \sqrt{2} \cdot \sqrt{\frac{\ln N}{N(a)}}\right]$ |

**UCB1 Multi-Armed Bandit** (Auer et al., 2002):

$$\text{UCB1}(a) = \underbrace{\bar{R}_a}_{\text{exploitation}} + \underbrace{c \sqrt{\frac{\ln N}{N_a}}}_{\text{exploration}}$$

where $\bar{R}_a$ = average reward for action $a$, $N$ = total selections, $N_a$ = selections of $a$, $c = \sqrt{2}$.

**Output**: 4 CSV files: `ppo_eval_200.csv`, `random_eval_200.csv`, `round_robin_eval_200.csv`, `ucb1_eval_200.csv`.

---

### 8.3 Stochastic vs. Deterministic Policy Evaluation

**Script**: `runs/eval_titan_v3_stochastic.py`

**Methodology**: Forensic discovery that deterministic (argmax) evaluation of a near-uniform policy collapses to a single action, creating misleading results. This script compares:

1. **Deterministic eval** (1 run): $a = \arg\max_a \pi_\theta(a|s)$
2. **Stochastic eval** (3 runs, different seeds): $a \sim \pi_\theta(\cdot|s)$
3. **6 static baselines** for reference

**Policy Distribution Forensics**:

At each step, extracts the full probability vector $\pi_\theta(a|s) = [\pi_0, \pi_1, \ldots, \pi_5]$ via `model.policy.get_distribution()`.

**State-Dependency Classification**:

$$\text{Var}(\pi) = \frac{1}{T} \sum_{t=1}^{T} \|\pi_t - \bar{\pi}\|^2$$

| Threshold | Classification |
|-----------|---------------|
| $\text{Var}(\pi) > 0.01$ | State-dependent (adaptive) |
| $0.001 < \text{Var}(\pi) \leq 0.01$ | Weakly state-dependent |
| $\text{Var}(\pi) \leq 0.001$ | Static (failed to learn) |

---

## 9. Infrastructure & Performance Studies

### 9.1 Pymoo vs. DEAP Framework Comparison

**Results**: `results/bench_compare/summary.json`

| Metric | Pymoo | DEAP | Winner |
|--------|-------|------|--------|
| Median hard penalty | **15** | 388 | Pymoo (26× better) |
| Median soft penalty | **1,306** | 1,319 | Pymoo |
| Time per generation | 29.7s | 0.65s | DEAP (45× faster) |
| Total runtime (avg) | 588s | 12.9s | DEAP |
| Feasible solutions | 0 | 0 | Neither |

**Verdict**: Pymoo produces **26× better hard constraint** scores despite being 45× slower. Quality compensates for runtime: **go with Pymoo**.

---

### 9.2 Vectorized Evaluator Benchmarks

**Phase A — Hard Evaluator** (`results/bench_phase_a.json`):

| Method | Time (s) | Speedup |
|--------|----------|---------|
| Batch (per-individual) | 0.128 | 1× |
| Vectorized (NumPy batch) | 0.032 | **4.06×** |

Uses `np.add.at` / `np.bincount` over entire population in single NumPy calls — no per-individual Python loops.

**Phase B — Soft Evaluator** (`results/bench_phase_b.json`):

| Method | Time (ms) | Speedup |
|--------|-----------|---------|
| OOP (per-individual) | 257.74 | 1× |
| Vectorized | 12.92 | **19.95×** |

**Population Scaling** (`results/bench_eval_vectorized.json`):

| Pop Size | Vectorized (ms/ind) | Speedup |
|----------|---------------------|---------|
| 50 | 0.192 | 5.07× |
| 100 | 0.150 | 6.53× |
| 200 | 0.162 | 5.92× |
| 400 | 0.167 | 5.76× |
| 800 | 0.170 | 5.63× |

---

### 9.3 Bitset Repair Operator Benchmark

**Results**: `results/bench_eval.json`

| Method | Mean (ms) | Speedup |
|--------|-----------|---------|
| Original repair | 1,667 | 1× |
| Bitset repair | 362 | **4.61×** |

**Key Innovation**: Count tensors (`rc[R,T]`, `ic[I,T]`, `gc[G,T]`) as `int16` arrays (~18 KiB) fit in L1 cache. NumPy fancy indexing for cost matrix construction.

---

### 9.4 Numba JIT Injection (Phase 74)

**5 JIT-compiled functions** with `@njit(cache=True, nogil=True)`:

| Function | Calls/sec | Latency |
|----------|-----------|---------|
| `_numba_count_conflicts` | 296,338 | 3.37 μs |
| `_numba_add` / `_numba_remove` | 493,510 | 2.03 μs |

**Ragged Array Solution**: `event_group_indices: list[list[int]]` (Numba-incompatible) → pre-padded `_egi_flat: np.int32[E, max_groups]` + `_egi_len: np.int32[E]`.

**Full Repair Benchmark** (10 calls):

| Metric | Value |
|--------|-------|
| Mean | 1,518 ms |
| Std | 313 ms |
| Min | 1,036 ms |
| Max | 2,142 ms |
| Throughput | 0.66 repairs/s |

**Cold start**: ~1.4s (JIT compilation), then instant from `.nbc`/`.nbi` disk cache.

---

### 9.5 Operator-Level Unit Benchmark (Elite 8)

**Script**: `runs/benchmark_heuristics.py`

Starting from a maximally broken population (random assignments, no repair), each operator applied once:

| Operator | $\Delta$Hard | $\Delta$Soft | Time (ms) | Verdict |
|----------|-------------|-------------|-----------|---------|
| SpatialResourceProjection | **-70.0** | +4.2 | 7 | ✅ PASS |
| FacultyTemporalProjection | **-49.2** | +11.3 | 8 | ✅ PASS |
| CohortTemporalProjection | -20.2 | +3.2 | 8 | ✅ PASS |
| SymmetricSubcohortSync | +105.3 | **-540.6** | 23 | ✅ PASS |
| UniversalFeasibilityProjection | +76.1 | **-537.2** | 66 | ✅ PASS |
| StochasticQuantaPerturbation | -0.9 | -1.5 | 1 | ✅ PASS |
| StochasticSpatialPerturbation | +0.2 | +0.0 | 1 | ✅ PASS |
| MeridianCompaction | +396.0 | **-151.8** | 4 | ✅ PASS |

**Key Insight**: Hard-focused operators (0–2) reduce hard but slightly increase soft. Soft-focused operators (3, 4, 7) massively reduce soft but severely increase hard. This **inherent conflict** is the Whack-A-Mole problem identified in the Collateral Damage Report.

---

## 10. Reward Engineering Evolution

The reward function evolved significantly across phases:

### Phase 1–2: Simple Scalar Reward

$$R = w_1 \cdot \Delta f + w_2 \cdot \text{diversity} - w_3 \cdot \text{time\_penalty}$$

### Phase 54: Strict Acceptance

$$R = \Delta f_{\text{hard}} \quad \text{with } T_{\text{accept}} = 0.0 \quad \text{(reject any degradation)}$$

### Phase 55: Tolerance Annealing

$$R = \Delta f_{\text{hard}} \quad \text{with } T_{\text{accept}} = 10.0 \text{ (train)} \to 0.0 \text{ (eval)}$$

### Phase 57: Phase-Transition Reward

$$R_t = \begin{cases} 2\Delta_{\text{hard}} + 0.5 \frac{\Delta_{\text{soft}}}{\text{norm}} & \text{if near-feasible} \\ \frac{\Delta_{\text{hard}}}{\text{norm}} + 0.1 \frac{\Delta_{\text{soft}}}{\text{norm}} & \text{otherwise} \end{cases}$$

### Phase 62 (SOTA): PBRS + Curriculum

$$R_{\text{total}} = R_{\text{base}} + \gamma\Phi(s') - \Phi(s) + \sum_{c \in \mathcal{A}} w_c \cdot \Delta_{\text{cv}_c}$$

---

## 11. Key Findings & Collateral Damage Report

### The Whack-A-Mole Problem (Phase 38 Audit)

**Root Cause**: Hard-focused operators destroy soft improvements and vice versa. In 100k training steps:

- Hard reduced: 1,482 → 1,214 ($\Delta = -268$)
- Soft **increased**: 161 → 642 ($\Delta = +481$)
- **42.7% of all steps** showed soft-penalty sign-flips (oscillation)
- MeridianCompaction alone adds **+396 hard** per application

**Agent Collapse**: The trained policy collapsed to a single action (SymmetricSubcohortSync for 48/49 eval steps) — the only operator that doesn't catastrophically damage the other objective.

**Recommendations Implemented**:

1. ✅ State-conditioned action masking (Phase 56) — blocks soft operators when infeasible
2. ✅ Phase-transition reward (Phase 57) — weights hard much more heavily above feasibility threshold
3. ✅ PBRS (Phase 62) — provides dense gradient signal during plateaus
4. ✅ Curriculum learning (Phase 62) — structured constraint introduction

### Pymoo vs. DEAP Decision

Quality wins: 26× better hard penalty justifies 45× slower runtime.

### Policy Collapse Diagnosis (Phase 59)

PPO with insufficient training budget produces near-uniform policy. Fix: 20× budget + 5× entropy coefficient.

### Feasibility Verdict

Problem confirmed **FEASIBLE**: 5/5 checks pass, 17.9% instructor utilization, all constraints satisfiable.

---

## 12. What to Include in the Thesis

### Recommended Thesis Structure (Ablation Coverage)

#### Chapter: Methodology — Algorithm Design

1. **Problem formulation** (§1): bi-objective UCTP with 8 hard + 4–6 soft constraints
2. **Chromosome encoding** (§3): interleaved triple $[I, R, T]$ per event
3. **NSGA-II base** (§5.1): selection, crossover, mutation operators with domain-specific guards
4. **Bitset repair operator** (§5.2): 3-stage pipeline with count tensors, cost matrix formulation
5. **Vectorized evaluation** (§9.2): NumPy batch evaluation with `add.at`/`bincount`

#### Chapter: GA Ablation Study (5 Modes)

| Mode | Key Variable Isolated | Expected Insight |
|------|----------------------|-----------------|
| A (Baseline) | No repair — evolutionary search only | Lower bound on GA quality |
| B (Memetic) | Elite repair (15%, 8 passes, every 5th gen) | Value of local search |
| C (Aggressive) | Full-pop repair (100%, 3 passes) + 2× offspring | Breadth vs. depth of repair |
| D (Adaptive) | Stagnation-aware parameter escalation | Value of adaptive control |
| E (CP Hybrid) | Periodic exact CP-SAT polish | Matheuristic improvement |

#### Chapter: RL Hyper-Heuristic

1. **Environment design**: 39-D observation space, 6-action LLH pipeline configuration
2. **PPO vs. DQN** (§6.1–6.2, §7.1): on-policy vs. off-policy comparison
3. **Reward engineering evolution** (§10): scalar → tolerance → phase-transition → PBRS+curriculum
4. **Action masking** (§6.7): state-conditioned blocking of soft operators during infeasibility
5. **Static differentiation analysis** (§6.6): proves no single LLH dominates — RL is justified
6. **PBRS theory** (§6.13): Ng et al. (1999) potential-based shaping with bottleneck density

#### Chapter: Scalability & Engineering

1. **Framework comparison** (§9.1): Pymoo vs. DEAP (26× quality vs. 45× speed tradeoff)
2. **Vectorization speedups** (§9.2): 4–20× evaluator acceleration
3. **Bitset repair** (§9.3): 4.61× repair speedup
4. **Numba JIT** (§9.4): C-level inner loop performance (2–3 μs/call)
5. **24-core parallelization** (§6.12): SubprocVecEnv for RL training

#### Chapter: Evaluation & Baselines

1. **Static baselines** (§8.1): per-LLH performance ceilings
2. **UCB1 multi-armed bandit** (§8.2): online adaptive baseline
3. **Stochastic vs. deterministic evaluation** (§8.3): policy distribution forensics
4. **Pareto dominance analysis**: PPO vs. all baselines

#### Chapter: Failure Analysis & Lessons

1. **Collateral Damage Report** (§11): Whack-A-Mole operator destruction
2. **Policy collapse** (§6.11): insufficient training budget diagnosis
3. **Agent action collapse**: single-action degeneration from reward deception

### Summary Statistics for Thesis Tables

| Experiment Category | Count |
|--------------------|-------|
| GA ablation modes | 5 (A–E) |
| RL training configurations | 13+ |
| Cross-cutting ablation studies | 6 |
| Baseline evaluation protocols | 3 |
| Infrastructure benchmarks | 5 |
| **Total distinct experiments** | **32+** |

### Key Mathematical Contributions

1. **Bitset repair cost matrix**: $C[t, r] = \sum_{q=t}^{t+d_e-1} [\text{rc}[r,q] + \text{ic}[i,q] + \sum_g \text{gc}[g,q]] + 100 \cdot \mathbf{1}[\neg\text{avail}]$
2. **PBRS potential**: $\Phi(s) = -\text{Var}(\text{per-resource conflicts}) / \text{max\_var}$
3. **Phase-transition reward**: bimodal amplification based on feasibility proximity
4. **Constraint curriculum**: 3-phase progressive reward weighting with linear blending
5. **UCB1 bandit baseline**: $\text{UCB1}(a) = \bar{R}_a + c\sqrt{\ln N / N_a}$

---

*Generated from codebase analysis of `schedule-engine` on 2026-03-10.*
