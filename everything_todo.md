# Thesis Results Generation — Master Plan

> **Goal**: Generate all thesis-worthy experimental results, excluding SE infrastructure ablations.  
> **Status**: SE (codebase) is COMPLETE. Now: run experiments → collect data → produce tables/plots.

---

## Scope Decision: What Goes In vs. Out

### ✅ IN THESIS (Methodology & Results)

| # | Experiment | Why It Matters | Script |
|---|-----------|---------------|--------|
| **GA-A** | Baseline NSGA-II (no repair) | Control group | `ga_01_baseline.py` |
| **GA-B** | Memetic (elite repair) | Key contribution — local search hybrid | `ga_02_memetic.py` |
| **GA-C** | Aggressive (full-pop repair) | Breadth vs. depth ablation | `ga_03_aggressive.py` |
| **GA-D** | Adaptive (stagnation escalation) | Adaptive parameter control | `ga_04_adaptive.py` |
| **GA-E** | CP Hybrid (NSGA-II + CP-SAT) | Matheuristic comparison | `ga_05_cp_hybrid.py` |
| **RL-ABL** | Random vs PPO vs DQN × 5 trials | Core RL ablation | `rl_07_ablation.py` |
| **RL-CAP** | PPO 150k capstone | Thesis primary RL result | `rl_03_capstone_FIXED.py` |
| **RL-DQN** | DQN 150k competitor | PPO vs DQN comparison | `rl_04_train_dqn.py` |
| **RL-MASK** | MaskablePPO with action masking | Action masking ablation | `rl_06_train_maskable_ppo.py` |
| **RL-SOTA** | Titan V4 (PBRS + Curriculum) | Final SOTA system | `rl_09_titan_v4_sota.py` |
| **BL-STATIC** | 6 LLHs × 3 seeds static baselines | Single-operator ceilings | `rl_03_static_baselines.py` |
| **BL-200** | PPO/Random/RR/UCB1 × 200 gen | Comprehensive comparison | `eval_all_baselines.py` |
| **RL-DIFF** | LLH differentiation analysis | Proves RL viability | `rl_06_llh_differentiation.py` |
| **RL-LR** | Learning rate sensitivity sweep | Hyperparameter sensitivity | `rl_08_hyperparam_sweep.py` |
| **RL-RWD** | Reward shaping comparison | Scalar vs. Hypervolume | `rl_05_compare_rewards.py` |

### ❌ OUT OF THESIS (SE / Engineering / Debug)

| Experiment | Reason to Exclude |
|-----------|-------------------|
| Pymoo vs DEAP benchmark | Framework decision — not methodology |
| Vectorized evaluator benchmark | Implementation optimization |
| Bitset repair speedup benchmark | Engineering speedup number |
| Numba JIT injection | Performance engineering |
| Bug-fix scripts (capstone_FIXED diff) | Debugging artifact |
| Debug scripts (fast_training, single_env) | Development aids |
| Policy collapse diagnosis (V3 overclock) | Engineering debugging |
| Titan V1/V2/V3 intermediate runs | Superseded by V4 SOTA |
| Multi-agent / specialist experiments | Incomplete, not enough results |
| RL-10 verification | Smoke test, not an experiment |
| Stochastic vs. Deterministic eval | Policy forensic — intermediate |

---

## Results Inventory: What EXISTS vs. What NEEDS Running

### 🟢 HAVE RESULTS (usable as-is)

| Experiment | Dir | Key Artifacts |
|-----------|-----|---------------|
| **GA-A** (Baseline) | `output/ga_baseline/20260224_120059/` | `convergence_history.csv`, `results.json`, PDFs, schedule |
| **RL-CAP** (PPO 150k) | `output/rl_capstone/20260301_001159/` | `training_curve.csv`, `evaluation_trajectory_200.csv`, 3 PDFs, model |
| **RL-DQN** (DQN 150k) | `output/rl_dqn/20260301_001224/` | `training_curve.csv`, `evaluation_trajectory_200.csv`, model |
| **BL-200** (4-strategy) | `output/baselines/` | `ppo_eval_200.csv`, `random_eval_200.csv`, `round_robin_eval_200.csv`, `ucb1_eval_200.csv` |
| **BL-STATIC** | `output/rl_phase54/static_baselines.csv` | Per-gen per-action CSV |

### 🟡 HAVE PARTIAL RESULTS (may need re-run or evaluation pass)

| Experiment | Dir | Status | What's Missing |
|-----------|-----|--------|---------------|
| **GA-B** (Memetic) | `output/ga_memetic/` | 41 run dirs, latest empty | Need clean re-run with full artifacts |
| **RL-MASK** (MaskablePPO) | `output/maskable_ppo/` | Unknown contents | Need to check artifacts |
| **RL-SOTA** (Titan V4) | `output/rl_titan_v4_sota/20260309_064026/` | **EMPTY** — training may still be running | Need completed training + evaluation |

### 🔴 NO RESULTS (must run from scratch)

| Experiment | Script | Est. Runtime | Priority |
|-----------|--------|-------------|----------|
| **GA-C** (Aggressive) | `ga_03_aggressive.py` | ~2–4 hours | HIGH |
| **GA-D** (Adaptive) | `ga_04_adaptive.py` | ~4–8 hours (300 gens) | HIGH |
| **GA-E** (CP Hybrid) | `ga_05_cp_hybrid.py` | ~2–6 hours (depends on CP-SAT) | HIGH |
| **RL-ABL** (3-method ablation) | `rl_07_ablation.py` | ~30–60 min (small scale) | HIGH |
| **RL-DIFF** (LLH differentiation) | `rl_06_llh_differentiation.py` | ~30–60 min | MEDIUM |
| **RL-LR** (LR sweep) | `rl_08_hyperparam_sweep.py` | ~20–40 min | MEDIUM |
| **RL-RWD** (Reward comparison) | `rl_05_compare_rewards.py` | ~20–30 min | MEDIUM |

---

## Execution Plan: 4 Phases

### Phase 1: GA Ablation Suite (Priority: CRITICAL)
>
> **Goal**: 5 GA modes with comparable results for the thesis table.
> **Output**: Per-mode `convergence_history.csv`, `results.json`, plots.

| Step | Task | Command | Status |
|------|------|---------|--------|
| 1.1 | ✅ GA-A Baseline — already done | — | 🟢 DONE |
| 1.2 | 🔄 GA-B Memetic — re-run cleanly | `python runs/ga_02_memetic.py` | 🔴 TODO |
| 1.3 | 🔴 GA-C Aggressive — first run | `python runs/ga_03_aggressive.py` | 🔴 TODO |
| 1.4 | 🔴 GA-D Adaptive — first run | `python runs/ga_04_adaptive.py` | 🔴 TODO |
| 1.5 | 🔴 GA-E CP Hybrid — first run | `python runs/ga_05_cp_hybrid.py` | 🔴 TODO |
| 1.6 | 📊 Collect GA comparison table | Extract final (hard, soft) from all 5 results.json | 🔴 TODO |
| 1.7 | 📈 Plot GA convergence overlay | 5 convergence curves on same axes | 🔴 TODO |

**Thesis Deliverable**: Table comparing 5 GA modes (best hard, best soft, time-to-feasibility, runtime, hypervolume) + convergence plot.

---

### Phase 2: RL Core Ablation (Priority: CRITICAL)
>
> **Goal**: Prove RL hyper-heuristic outperforms static baselines & compare PPO vs DQN.

| Step | Task | Command | Status |
|------|------|---------|--------|
| 2.1 | ✅ PPO 150k capstone — already done | — | 🟢 DONE |
| 2.2 | ✅ DQN 150k competitor — already done | — | 🟢 DONE |
| 2.3 | ✅ Static baselines (6 LLHs) — already done | — | 🟢 DONE |
| 2.4 | ✅ 200-gen 4-strategy comparison — already done | — | 🟢 DONE |
| 2.5 | 🔴 RL ablation (Random/PPO/DQN × 5) | `python runs/rl_07_ablation.py` | 🔴 TODO |
| 2.6 | 🔴 LLH differentiation proof | `python runs/rl_06_llh_differentiation.py` | 🔴 TODO |
| 2.7 | 📊 PPO vs DQN comparison table | Extract from capstone + DQN eval CSVs | 🔴 TODO |
| 2.8 | 📊 PPO vs all static baselines table | Extract from baselines/ CSVs | 🔴 TODO |
| 2.9 | 📈 Learning curve plot (PPO + DQN) | From training_curve.csv | 🔴 TODO |

**Thesis Deliverable**: PPO vs DQN table, PPO vs 6 static baselines table, ablation (Random/PPO/DQN) table with std, learning curves, LLH differentiation proof.

---

### Phase 3: RL Advanced Ablation (Priority: HIGH)
>
> **Goal**: Justify each RL design decision (masking, reward shaping, PBRS, curriculum).

| Step | Task | Command | Status |
|------|------|---------|--------|
| 3.1 | 🟡 Check if Maskable PPO has results | Check `output/maskable_ppo/` | 🟡 CHECK |
| 3.2 | 🔴 LR sensitivity sweep | `python runs/rl_08_hyperparam_sweep.py` | 🔴 TODO |
| 3.3 | 🔴 Reward shaping comparison | `python runs/rl_05_compare_rewards.py` | 🔴 TODO |
| 3.4 | 🟡 Titan V4 SOTA — wait/verify running | Check if training completed | 🟡 CHECK |
| 3.5 | 🔴 Titan V4 evaluation (if model exists) | `python runs/eval_titan_v3_stochastic.py` (adapted) | 🔴 TODO |
| 3.6 | 📊 Action masking ablation table | With masking vs without masking | 🔴 TODO |
| 3.7 | 📊 PBRS + Curriculum ablation table | V3 (no shaping) vs V4 (with shaping) | 🔴 TODO |

**Thesis Deliverable**: Masking effect table, LR sensitivity plot, reward shaping comparison, PBRS impact analysis.

---

### Phase 4: Thesis Figures & Tables (Priority: FINAL)
>
> **Goal**: Publication-ready output — every thesis table and figure.

| Step | Task | Output File | Status |
|------|------|-------------|--------|
| 4.1 | Table: GA 5-mode comparison | `figures/table_ga_modes.csv` | 🔴 TODO |
| 4.2 | Figure: GA convergence overlay (5 modes) | `figures/fig_ga_convergence.pdf` | 🔴 TODO |
| 4.3 | Table: PPO vs DQN (hard, soft, feasibility %) | `figures/table_ppo_vs_dqn.csv` | 🔴 TODO |
| 4.4 | Figure: RL learning curves | `figures/fig_rl_learning_curves.pdf` | 🔴 TODO |
| 4.5 | Table: PPO vs static baselines (6 LLHs + UCB1) | `figures/table_baselines.csv` | 🔴 TODO |
| 4.6 | Figure: Baseline comparison bar chart | `figures/fig_baseline_comparison.pdf` | 🔴 TODO |
| 4.7 | Table: Random/PPO/DQN ablation (mean ± std) | `figures/table_ablation.csv` | 🔴 TODO |
| 4.8 | Table: LR sensitivity results | `figures/table_lr_sweep.csv` | 🔴 TODO |
| 4.9 | Figure: Action distribution heatmap | `figures/fig_action_distribution.pdf` | 🔴 TODO |
| 4.10 | Table: Full constraint breakdown of best schedule | `figures/table_constraint_breakdown.csv` | 🔴 TODO |
| 4.11 | Figure: Pareto front (hard vs soft) best solutions | `figures/fig_pareto_front.pdf` | 🔴 TODO |
| 4.12 | Table: PBRS + Curriculum ablation | `figures/table_pbrs_curriculum.csv` | 🔴 TODO |
| 4.13 | Figure: Curriculum phase transitions | `figures/fig_curriculum_phases.pdf` | 🔴 TODO |

---

## Execution Order (Parallel Where Possible)

```
  ┌─────── Phase 1 (GA) ──────────────────────┐
  │                                            │
  │  1.2 GA-B ──┐                              │
  │  1.3 GA-C ──┤ can run in parallel if       │  Phase 2 (RL Core)
  │  1.4 GA-D ──┤ machine has resources        │  ┌─────────────────────┐
  │  1.5 GA-E ──┘                              │  │ 2.5 RL ablation     │
  │                                            │  │ 2.6 LLH diff        │
  │  1.6 Collect comparison ──────────┐        │  └──────┬──────────────┘
  │  1.7 Plot convergence             │        │         │
  └────────────────────────────────────┘        │         │
                                      │         │         │
                                      ▼         ▼         ▼
                               ┌─── Phase 3 (RL Advanced) ───┐
                               │ 3.2 LR sweep                │
                               │ 3.3 Reward comparison        │
                               │ 3.4 Check V4 status          │
                               │ 3.5 V4 evaluation            │
                               └──────────┬──────────────────┘
                                          │
                                          ▼
                               ┌─── Phase 4 (Tables/Plots) ──┐
                               │ 4.1–4.13 Generate all        │
                               │ thesis figures & tables       │
                               └──────────────────────────────┘
```

---

## Estimated Total Runtime

| Phase | Est. Wall-Clock | Can Parallelize? |
|-------|----------------|-----------------|
| Phase 1 (GA suite) | 8–16 hours total | Yes — run all 4 modes simultaneously |
| Phase 2 (RL core) | 1–2 hours | Yes — ablation + LLH diff |
| Phase 3 (RL advanced) | 2–4 hours | Partially |
| Phase 4 (plots/tables) | 30–60 min | Script-based, fast |
| **Total** | **~12–24 hours** (with parallelization) | |

---

## Thesis Chapter → Experiment Mapping

| Thesis Chapter | Experiments | Deliverables |
|---------------|------------|-------------|
| **Ch. 3: Methodology** | Problem formulation, encoding, NSGA-II, repair operator, RL env design | Math + architecture diagrams |
| **Ch. 4: GA Ablation** | GA-A/B/C/D/E | Table 4.1 (5-mode comparison), Fig 4.1 (convergence) |
| **Ch. 5: RL Hyper-Heuristic** | RL-DIFF, RL-CAP, RL-DQN, RL-ABL | Table 5.1 (PPO vs DQN), Table 5.2 (ablation), Fig 5.1–5.3 |
| **Ch. 6: Advanced RL** | RL-MASK, RL-SOTA, RL-LR, RL-RWD | Table 6.1 (masking), Table 6.2 (PBRS), Fig 6.1–6.2 |
| **Ch. 7: Baselines & Evaluation** | BL-STATIC, BL-200 | Table 7.1 (baselines), Fig 7.1 (bar chart) |
| **Ch. 8: Discussion** | Collateral damage findings, lessons learned | Qualitative analysis |

---

## Quick Commands Reference

```powershell
# Phase 1: GA modes (run in separate terminals)
python runs/ga_01_baseline.py       # already done
python runs/ga_02_memetic.py        # re-run for clean results
python runs/ga_03_aggressive.py     # first run
python runs/ga_04_adaptive.py       # first run  
python runs/ga_05_cp_hybrid.py      # first run (needs ortools)

# Phase 2: RL core
python runs/rl_07_ablation.py       # Random/PPO/DQN × 5 trials
python runs/rl_06_llh_differentiation.py  # LLH viability proof

# Phase 3: RL advanced
python runs/rl_08_hyperparam_sweep.py     # LR sensitivity
python runs/rl_05_compare_rewards.py      # Reward comparison
# V4 SOTA — check if already running, otherwise:
python runs/rl_09_titan_v4_sota.py

# Phase 4: Thesis plots (to be created)
python runs/plot_master_thesis.py    # or custom script TBD
```

---

## Checklist Summary

- [ ] **Phase 1.2**: GA-B Memetic clean re-run
- [ ] **Phase 1.3**: GA-C Aggressive first run
- [ ] **Phase 1.4**: GA-D Adaptive first run
- [ ] **Phase 1.5**: GA-E CP Hybrid first run
- [ ] **Phase 1.6**: GA comparison table extracted
- [ ] **Phase 1.7**: GA convergence overlay plot
- [ ] **Phase 2.5**: RL ablation (3 methods × 5 trials)
- [ ] **Phase 2.6**: LLH differentiation analysis
- [ ] **Phase 2.7**: PPO vs DQN comparison table
- [ ] **Phase 2.8**: PPO vs static baselines table
- [ ] **Phase 2.9**: RL learning curves plot
- [ ] **Phase 3.1**: Check Maskable PPO artifacts
- [ ] **Phase 3.2**: LR sensitivity sweep
- [ ] **Phase 3.3**: Reward shaping comparison
- [ ] **Phase 3.4**: V4 SOTA training status
- [ ] **Phase 3.5**: V4 SOTA evaluation
- [ ] **Phase 3.6**: Action masking ablation table
- [ ] **Phase 3.7**: PBRS + Curriculum ablation table
- [ ] **Phase 4.1–4.13**: All thesis figures & tables

**Total TODO items**: 22  
**Already done**: 5 (GA-A, PPO cap, DQN cap, static baselines, 200-gen baselines)  
**Remaining**: 17

---

*Plan created 2026-03-10. SE codebase is frozen — focus purely on experiment execution and result collection.*
