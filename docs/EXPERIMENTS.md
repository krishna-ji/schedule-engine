# Schedule Engine — Experiments

> Comprehensive guide to every experiment runner in the project.
> All scripts are in `runs/` and should be executed from the project root.

---

## Quick Start

```bash
# Activate the virtual environment
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

# Run any experiment
python -m runs.<script_name>
# Example:
python -m runs.ga_01_baseline
```

---

## GA Experiments

### GA 01 — Baseline: Pure NSGA-II

**Script:** `runs/ga_01_baseline.py`

Pure NSGA-II genetic algorithm with no repair or local search.
Serves as the control baseline for all other experiments.

| Parameter | Value |
|-----------|-------|
| `POP_SIZE` | 100 |
| `NGEN` | 200 |
| `CROSSOVER_PROB` | 0.5 |
| `MUTATION_PROB` | 0.05 |
| `SEED` | 42 |

**Output:** `output/ga_baseline/<timestamp>/`

---

### GA 02 — Memetic: NSGA-II + Elite Bitset Repair

**Script:** `runs/ga_02_memetic.py`

NSGA-II augmented with memetic local search — repairs the top 15% of
individuals every 5th generation using bitset-based repair (8 passes per elite).

| Parameter | Value |
|-----------|-------|
| `POP_SIZE` | 120 |
| `NGEN` | 50 |
| `CROSSOVER_PROB` | 0.4 |
| `MUTATION_PROB` | 0.10 |
| `ELITE_PCT` | 0.15 |
| `REPAIR_ITERS` | 8 |
| `REPAIR_FREQUENCY` | 5 |

**Output:** `output/ga_memetic/<timestamp>/`

---

### GA 03 — Aggressive: 2× Offspring, High Mutation, Full Repair

**Script:** `runs/ga_03_aggressive.py`

High-intensity GA variant — 2× offspring per generation, 15% mutation rate,
and full-population repair every generation. Trades compute for rapid
constraint reduction.

| Parameter | Value |
|-----------|-------|
| `POP_SIZE` | 200 |
| `NGEN` | 100 |
| `CROSSOVER_PROB` | 0.7 |
| `MUTATION_PROB` | 0.15 |
| `N_OFFSPRINGS_MULT` | 2.0 |
| `REPAIR_ITERS` | 3 |

**Output:** auto-generated timestamped directory

---

### GA 04 — Adaptive: Stagnation-Aware Mutation Escalation

**Script:** `runs/ga_04_adaptive.py`

Adaptive GA that monitors stagnation. When `best_hard` stalls for 15
generations, mutation rate escalates from 5% to 20% and elite repair
activates. De-escalates when improvement resumes.

| Parameter | Value |
|-----------|-------|
| `POP_SIZE` | 100 |
| `NGEN` | 300 |
| `CROSSOVER_PROB` | 0.5 |
| `MUTATION_PROB` | 0.05 (starting) |
| `STAGNATION_WINDOW` | 15 |
| `MUTATION_HI` | 0.20 |
| `ELITE_PCT` | 0.10 |
| `REPAIR_ITERS` | 5 |

**Output:** auto-generated timestamped directory

---

### GA 05 — CP Hybrid: NSGA-II + Periodic CP-SAT Polish

**Script:** `runs/ga_05_cp_hybrid.py`

Hybrid GA that invokes Google OR-Tools CP-SAT solver every 10 generations
for deep constraint satisfaction on the best individual. Requires `ortools>=9.8`.

| Parameter | Value |
|-----------|-------|
| `POP_SIZE` | 60 |
| `NGEN` | 100 |
| `CROSSOVER_PROB` | 0.5 |
| `MUTATION_PROB` | 0.05 |
| `CP_INTERVAL` | 10 |
| `CP_TIMEOUT` | 30.0 s |

**Output:** auto-generated timestamped directory

---

## RL Experiments

### RL 04 — PPO Baseline: Tolerance Exploration

**Script:** `runs/rl_04_train_ppo_baseline.py`

Two-phase training: (1) exploration with tolerance=10 for 150k timesteps,
(2) strict 200-gen evaluation with tolerance=0.

| Parameter | Value |
|-----------|-------|
| `TOTAL_TIMESTEPS` | 150,000 |
| `TRAINING_GENERATIONS` | 50 |
| `TRAINING_POP_SIZE` | 120 |
| `TRAINING_ACCEPTANCE_TOLERANCE` | 10.0 |
| `EVAL_GENERATIONS` | 200 |
| `EVAL_TOLERANCE` | 0.0 |
| `LEARNING_RATE` | 3e-4 |
| `CLIP_RANGE` | 0.2 |
| `NET_ARCH` | [64, 64] |

**Output:** `output/rl_capstone_fixed/<timestamp>/`, model at `output/models/ppo_capstone_final.zip`

---

### RL 01 — Static Baselines: Per-LLH Isolation

**Script:** `runs/rl_01_static_baselines.py`

Evaluates each of the 6 low-level heuristics (LLHs) in isolation as a
static policy for 50 generations, repeated across 3 seeds for robustness.

| Parameter | Value |
|-----------|-------|
| `POP_SIZE` | 120 |
| `MAX_GENERATIONS` | 50 |
| `SEEDS` | [42, 123, 7] |
| Actions | Conservative, Aggressive, Memetic, SoftFocus, Destructive, Intensified |

**Output:** `output/rl_phase54/static_baselines.csv`

---

### RL 03 — DQN Competitor

**Script:** `runs/rl_03_train_dqn.py`

Identical pipeline to the PPO capstone, but using Stable-Baselines3 DQN.

| Parameter | Value |
|-----------|-------|
| `TOTAL_TIMESTEPS` | 150,000 |
| `EVAL_GENERATIONS` | 200 |
| `LEARNING_RATE` | 1e-4 |
| `BUFFER_SIZE` | 100,000 |
| `BATCH_SIZE` | 32 |
| `TARGET_UPDATE_INTERVAL` | 1,000 |
| `EXPLORATION_FRACTION` | 0.1 |
| `EXPLORATION_FINAL_EPS` | 0.05 |

**Output:** `output/rl_dqn/<timestamp>/`, `output/baselines/dqn_eval_200.csv`

---

### RL 02 — LLH Differentiation Check

**Script:** `runs/rl_02_llh_differentiation.py`

Diagnostic script that verifies the 6 LLHs produce meaningfully different
optimisation trajectories. Outputs per-generation trajectory, checkpoint
comparison table (gen 5, 25, 50), and best-ever hard penalty per LLH.

| Parameter | Value |
|-----------|-------|
| `POP_SIZE` | 120 |
| `MAX_GEN` | 50 |
| `SEED` | 42 |
| `CHECKPOINTS` | [5, 25, 50] |

**Output:** console only (no file export)

---

### RL 05 — Maskable PPO: State-Conditioned Action Masking

**Script:** `runs/rl_05_train_maskable_ppo.py`

MaskablePPO (sb3-contrib) with state-conditioned action masking.
Blocks soft optimizers (Actions 3 & 7) when hard constraints are violated,
forcing feasibility repair before soft optimisation.

| Parameter | Value |
|-----------|-------|
| `POP_SIZE` | 50 |
| `MAX_GENERATIONS` | 25 |
| `TOTAL_TIMESTEPS` | 50,000 |
| `LEARNING_RATE` | 3e-4 |
| `CLIP_RANGE` | 0.2 |
| `TRAIN_TOLERANCE` | 5.0 |
| `NET_ARCH` | [64, 64] |

**Output:** `output/maskable_ppo/<timestamp>/` (model, training_curve.csv, action_mask_analysis.txt)

---

### RL 06 — Titan V4 SOTA: PBRS + Curriculum

**Script:** `runs/rl_06_train_ppo_titan_v4_sota.py`

State-of-the-art RL run using MaskablePPO on 24-core `SubprocVecEnv`.
Features Potential-Based Reward Shaping (PBRS) and a 3-phase constraint
curriculum (spatial → instructor → full NP-hard).

$$R_{\text{shaped}} = R_{\text{base}} + \gamma \Phi(s') - \Phi(s) + \sum_{c} w_c \cdot \Delta_{cv_c}$$

| Parameter | Value |
|-----------|-------|
| `NUM_CPU` | 24 |
| `TRAIN_POP_SIZE` | 40 |
| `TRAIN_MAX_GEN` | 50 |
| `TOTAL_TIMESTEPS` | 100,000 |
| `LEARNING_RATE` | 5e-4 |
| `N_STEPS` | 128 |
| `BATCH_SIZE` | 512 |
| `N_EPOCHS` | 10 |
| `ENT_COEF` | 0.05 |
| `PBRS_GAMMA` | 0.99 |
| `PHASE1_EPISODES` | 21 |
| `PHASE2_EPISODES` | 63 |
| `CURRICULUM_WEIGHT` | 0.5 |

**Output:** `output/rl_titan_v4_sota/<timestamp>/`, model at `output/models/ppo_titan_v4_sota.zip`

---

## Evaluation & Utility Scripts

### Eval All Baselines

**Script:** `runs/eval_all_baselines.py`

Runs 4 action-selection strategies (PPO, Random, Round-Robin, UCB1) over
200-gen strict rollouts. Saves per-generation CSVs for thesis comparison.

| Parameter | Value |
|-----------|-------|
| `POP_SIZE` | 120 |
| `EVAL_GENERATIONS` | 200 |
| `EVAL_TOLERANCE` | 0.0 |
| Model | `output/models/ppo_capstone_final.zip` |

**Output:** `output/baselines/` → `ppo_eval_200.csv`, `random_eval_200.csv`,
`round_robin_eval_200.csv`, `ucb1_eval_200.csv`

---

### Plot Master Thesis

**Script:** `runs/plot_master_thesis.py`

Generates a publication-quality 6-trajectory comparison plot (`best_hard` vs
generation) across PPO, DQN, UCB1, Round-Robin, Random, and best GA runs.

**Input:** Reads from `output/baselines/`, `output/ga_baseline/`, `output/ga_memetic/`

**Output:** `output/figures/master_trajectory_comparison.pdf` and `.png`

---

### Pre-Scheduling Audit

**Script:** `runs/pre_scheduling_audit.py`

Comprehensive pre-flight data validation covering 8 categories:

1. **Data Completeness** — missing fields, empty lists
2. **Pigeonhole Feasibility** — room/instructor/timeslot capacity
3. **Qualification Coverage** — instructor-course qualification gaps
4. **Lab/Room Features** — feature mismatch detection
5. **Availability Analysis** — instructor availability vs demand
6. **Cross-Reference Integrity** — referential consistency
7. **Capacity Analysis** — room capacity vs group sizes
8. **Schedule Density** — utilisation projections

**Input:** `data/` (Course.json, Groups.json, Instructors.json, Rooms.json)

**Output:** `output/` (text report) + rich console output

---

## Benchmark & Profiling Scripts

Located in `scripts/`:

| Script | Purpose |
|--------|---------|
| `bench_eval.py` | Benchmark OOP evaluator performance |
| `bench_eval_vectorized.py` | Benchmark vectorized evaluator |
| `build_reference_front.py` | Build Pareto reference front for IGD |
| `micro_bench.py` | Micro-benchmarks for critical paths |
| `profile_pipeline.py` | Profile full pipeline with cProfile |

---

## Experiment Output Structure

Each experiment creates a timestamped output directory:

```
output/<experiment_type>/<YYYYMMDD_HHMMSS>/
├── result.json              # Final objectives, parameters, timing
├── convergence_hard.png     # Hard penalty vs generation
├── convergence_soft.png     # Soft penalty vs generation
├── pareto_front.png         # Final Pareto front
├── diversity.png            # Population diversity
├── hypervolume.png          # HV indicator progression
├── violation_report.txt     # Per-constraint breakdown
├── timetable_*.pdf          # Per-group/instructor timetables
└── training_curve.csv       # (RL only) per-step metrics
```
