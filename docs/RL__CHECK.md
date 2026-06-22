# RL REALITY CHECK — Zero-Hallucination Audit

**Thesis**: *HYPERHEURISTIC OPTIMIZATION OF UNIVERSITY COURSE TIMETABLING PROBLEM USING RL AND GENETIC ALGORITHMS*

**Date**: 2026-02-25

**Auditor Scope**: `src/rl/` + `src/ga/heuristics/` + `src/experiments/rl_experiment.py` + Thesis proposal chapters

---

## Executive Summary

| Audit Area | Thesis Spec | Code Reality | Verdict |
|---|---|---|---|
| State Space | 41-D | **39-D** (12 constraint breakdown, not 14) | **GAP — 2 soft constraints missing** |
| Action Space | Discrete(20) | **Discrete(N) — dynamic** (26 heuristics registered, builds at runtime) | **GAP — mismatch vs. thesis 20-action taxonomy** |
| Reward Function | Eq. 1 with clip to [-1,1] | **Implemented and clipped** | **FUNCTIONAL — minor formula divergence** |
| PPO/DQN Agents | SB3, [64,64], clip 0.2 | **SB3 `MlpPolicy` (default [64,64])**, clip from config | **FUNCTIONAL — net_arch not explicitly set** |
| RL↔GA Bridge | 1 Gen = 1 RL Step (AOS) | **1 Step = 1 heuristic on best individual** | **FUNCTIONAL but semantically different** |

**Bottom line**: The RL subsystem is **structurally complete and runnable**. It is NOT a stub. Training, evaluation, and inference pipelines all exist and are wired end-to-end. However, the state/action specifications **do not exactly match** the numbers in the thesis (41-D / 20-action). Closing these gaps requires targeted dimensional fixes, not architectural rewrites.

---

## Audit 1: State Space (`src/rl/gym_env/state_encoder.py`)

### Current Truth

The `StateEncoder` class ([state_encoder.py](src/rl/gym_env/state_encoder.py)) builds an observation vector with the following structure:

| Index | Dims | Category | Source (line) |
|---|---|---|---|
| `s[0:5]` | **5** | Fitness | `best, avg, worst, std, range` — [L458-464](src/rl/gym_env/state_encoder.py#L458-L464) |
| `s[5:10]` | **5** | Diversity | `pop, geno, pheno, fitness_div, unique_ratio` — [L465-469](src/rl/gym_env/state_encoder.py#L465-L469) |
| `s[10:14]` | **4** | Progress | `generation, stagnation, convergence, improvement` — [L470-473](src/rl/gym_env/state_encoder.py#L470-L473) |
| `s[14:17]` | **3** | Violations | `hard, soft, violation_std` — [L474-476](src/rl/gym_env/state_encoder.py#L474-L476) |
| `s[17:29]` | **12** | Constraint Breakdown | 8 hard + 4 soft — [L479-486](src/rl/gym_env/state_encoder.py#L479-L486) |
| `s[29:39]` | **10** | Heuristic History | last 10 action IDs — [L489-491](src/rl/gym_env/state_encoder.py#L489-L491) |

**Total coded dimensions: 5 + 5 + 4 + 3 + 12 + 10 = 39**

The `observation_dim` property ([L588](src/rl/gym_env/state_encoder.py#L588)) confirms:
```python
base_features = 17
constraint_features = 12 if self.enable_constraint_breakdown else 0
return base_features + constraint_features + self.history_size  # 17 + 12 + 10 = 39
```

The `ScheduleEnv` ([schedule_env.py L109-111](src/rl/gym_env/schedule_env.py#L109-L111)) sets `observation_space = Box(low=0.0, high=1.0, shape=(obs_dim,))` where `obs_dim = self.state_encoder.observation_dim`.

### Thesis Gap

The thesis specifies **41 dimensions** with **14 constraint breakdown features** (8 hard + **6** soft).

The code has only **12 constraint breakdown features** (8 hard + **4** soft):

**Hard constraints coded** (8) — [L78-85](src/rl/gym_env/state_encoder.py#L78-L85):
| # | Code Name | Academic Name |
|---|---|---|
| 1 | `CTE` | Cohort Temporal Exclusivity |
| 2 | `FTE` | Faculty Temporal Exclusivity |
| 3 | `SRE` | Spatial Resource Exclusivity |
| 4 | `FPC` | Faculty Pedagogical Congruence |
| 5 | `FFC` | Facility Feature Congruence |
| 6 | `FCA` | Faculty Chronological Availability |
| 7 | `CQF` | Curriculum Quanta Fulfillment |
| 8 | `ICTD` | Intra-Course Temporal Dispersion |

**Soft constraints coded** (4) — [L88-93](src/rl/gym_env/state_encoder.py#L88-L93):
| # | Code Name | Academic Name |
|---|---|---|
| 1 | `CSC` | Cohort Schedule Contiguity |
| 2 | `FSC` | Faculty Schedule Contiguity |
| 3 | `MIP` | Meridian Interval Preservation |
| 4 | `SSCP` | Symmetric Sub-Cohort Parallelism |

**MISSING from thesis spec** (2 soft constraints needed to reach 6 soft / 14 total):
- The thesis table `\ref{tab:rl_state_vector}` claims `v_{SC_1...SC_6}` (6 soft constraints) but the code only implements 4.
- Candidates to add: `session_continuity` and possibly a `soft_weight_factor` aggregate — both exist in the config ([helpers.py L131-140](src/rl/helpers.py#L131-L140)) but are not encoded as state features.

### Verdict: **MATHEMATICALLY INCOMPLETE — 39-D instead of 41-D**

The encoder is NOT a stub. It performs real computation (diversity via `scipy.pdist`, constraint breakdown extraction from individual metadata, running history). The gap is precisely **2 missing soft constraint dimensions**.

### Fix Required

Add 2 more soft constraint names to `SOFT_CONSTRAINT_NAMES` and ensure the evaluator populates their per-individual breakdown. This changes:
- `SOFT_CONSTRAINT_NAMES`: 4 → 6 entries
- `constraint_features`: 12 → 14
- `observation_dim`: 39 → **41** (matches thesis)
- Normalization indices in `_normalize_observation()`: adjust range `[17:29]` → `[17:31]`, history `[29:]` → `[31:]`
- `ScheduleEnv.observation_space`: auto-updates via `obs_dim`

---

## Audit 2: Action Space & LLH Registry

### Current Truth

The `ActionMapper` class ([action_space.py](src/rl/gym_env/action_space.py)) builds the action space **dynamically** from the heuristic registry at runtime:

```python
# action_space.py L70-87
self.actions.append(ActionInfo(action_id=0, name="no-op", ...))  # Action 0 = no-op
for idx, h in enumerate(heuristics_sorted, start=1):              # Actions 1..N
    self.actions.append(ActionInfo(action_id=idx, ...))
```

The heuristic registry in [all_heuristics.py](src/ga/heuristics/all_heuristics.py) defines:

| Category | Count | Heuristics |
|---|---|---|
| Construction (3) | 3 | `largest_degree_first`, `most_constrained_first`, `earliest_deadline_first` |
| Perturbation (5) | 5 | `random_swap`, `temporal_shift`, `room_shuffle`, `instructor_reassign`, `multi_perturbation`* |
| Improvement (3) | 3 | `kempe_chain`, `ejection_chain`, `variable_depth_search` |
| Diversity (4) | 4 | `distance_preserving_crossover`, `crowding_mutation`, `niching_selection`, `adaptive_diversity_maintenance`* |
| Meta (4) | 4 | `variable_neighborhood_descent`, `iterated_local_search`, `adaptive_large_neighborhood`, `guided_local_search`* |
| Repair (7) | 7 | `igls_repair`, `greedy_repair`, `selective_repair`, `lns_repair`, `exhaustive_repair`*, `memetic_repair`*, `repair_break_placement` |
| **Total** | **26** | *items marked with \* are `enabled_by_default=False` |

With defaults: **26 total registered, ~20 enabled by default** (the 6 marked `*` are disabled).

The `_build_action_space()` only loads **enabled** heuristics when `use_config=True` ([L67-70](src/rl/gym_env/action_space.py#L67-L70)):
```python
if self.use_config:
    heuristics = get_enabled_heuristics().values()
```

So the runtime `Discrete(N)` depends on which heuristics the config enables. With defaults: 1 no-op + ~20 enabled = **Discrete(~21)**.

### Thesis Gap

The thesis proposes exactly **20 discrete actions** (Discrete(20)) organized as:
1. Mutation Rate Control (5)
2. Repair Heuristics (6)
3. Diversity Injection (3)
4. Local Search (4)
5. Crossover Modulation (2)

**Critical mismatch**: The thesis taxonomy and the code taxonomy are **different classification systems**.

| Thesis Category | Thesis Count | Code Equivalent | Code Count |
|---|---|---|---|
| Mutation Rate Control | 5 | **MISSING** — no rate-control actions exist | 0 |
| Repair Heuristics | 6 | `repair` category | 5 enabled |
| Diversity Injection | 3 | `diversity` category | 2 enabled |
| Local Search | 4 | `improvement` + `meta` categories | 3 + 3 = 6 enabled |
| Crossover Modulation | 2 | **MISSING** — no cxpb-control actions exist | 0 |
| Construction | (not in thesis) | `construction` category | 3 |
| Perturbation | (not in thesis) | `perturbation` category | 4 enabled |

**Key gaps**:
- ❌ **Mutation Rate Control actions** (thesis wants 5 discrete levels: 0.05, 0.15, 0.30, Adaptive, Declining) — **MISSING entirely**
- ❌ **Crossover Modulation actions** (thesis wants: increase/decrease cxpb) — **MISSING entirely**
- The existing heuristics are *operator applications*, not *parameter adjustments*

### BitsetSchedulingRepair Analysis

The `BitsetSchedulingRepair` ([repair_operator_bitset.py](src/pipeline/repair_operator_bitset.py)) is a monolithic 3D-tensor repair engine with three stages:
1. Domain clamping (instructor/room/time validity)
2. Conflict resolution (room/instructor/group double-booking via cost matrix)
3. Group deconfliction (cascading re-insertion)

**Can it be deconstructed into atomic LLHs?** Yes, with effort:

| Potential Atomic LLH | BitsetSchedulingRepair Source |
|---|---|
| `repair_domain_clamp` | Stage 1 — domain clamping scan |
| `repair_room_clash` | Stage 2 — room count > 1 resolution |
| `repair_instructor_clash` | Stage 2 — instructor count > 1 resolution |
| `repair_group_clash` | Stage 3 — group deconfliction |
| `repair_availability` | Stage 2 — availability mask violations |
| `repair_paired_practical` | `_find_paired_placement` — SSCP dual placement |

However, the current API (`repair(chromosome, rng) -> ndarray`) runs all stages as one atomic call. Deconstruction requires refactoring the internal pipeline into individually-callable stages.

### Verdict: **STRUCTURALLY DIVERGENT — Runtime ~21 actions, thesis requires exactly 20 with different taxonomy**

### Action Plan for 20-Action Registry

To match the thesis exactly, create a **fixed 20-action LLH registry** that maps to:

```
Action 0:  no-op
Action 1:  mutpb = 0.05 (Low)
Action 2:  mutpb = 0.15 (Medium)  
Action 3:  mutpb = 0.30 (High)
Action 4:  mutpb = adaptive
Action 5:  mutpb = declining
Action 6:  repair_instructor_clash
Action 7:  repair_room_capacity
Action 8:  repair_timeslot_realloc
Action 9:  repair_qualification
Action 10: repair_availability
Action 11: repair_batch_multi
Action 12: inject_random_immigrant
Action 13: inject_smart_immigrant
Action 14: restart_worst
Action 15: hill_climbing (kempe_chain)
Action 16: tabu_search (ejection_chain)
Action 17: simulated_annealing (VNS)
Action 18: greedy_descent (variable_depth_search)
Action 19: increase_cxpb / decrease_cxpb
```

**Alternatively** (recommended): Update the thesis table to match the code's existing categories and count. The code's categorization (construction/perturbation/improvement/diversity/meta/repair) is **arguably more principled** than the thesis's ad-hoc grouping. Either way, `n_actions` must be pinned to exactly 20 for thesis compliance.

---

## Audit 3: Reward Function & Agents

### Reward Calculator (`src/rl/gym_env/reward_calculator.py`)

**Current formula** ([L140-148](src/rl/gym_env/reward_calculator.py#L140-L148)):
```python
total_reward = (
    self.fitness_weight * fitness_reward
    + self.diversity_weight * diversity_bonus
    - self.time_weight * time_penalty
)
if self.normalize:
    total_reward = np.clip(total_reward, -1.0, 1.0)
```

**Thesis Eq. 1**: $r_t = w_f \cdot r_{\text{fit}} + w_d \cdot r_{\text{div}} - w_t \cdot r_{\text{time}}$, clipped to $[-1, 1]$.

| Component | Thesis Spec | Code Implementation | Match? |
|---|---|---|---|
| $r_{\text{fitness}}^{\text{scalar}}$ | $(f_{\text{best}}^{(t-1)} - f_{\text{best}}^{(t)}) / |f_{\text{best}}^{(t-1)}|$ | `_calculate_fitness_reward()` — same formula ([L163-175](src/rl/gym_env/reward_calculator.py#L163-L175)) | ✅ |
| $r_{\text{fitness}}^{\text{MO}}$ (hypervolume) | $\Delta HV / S_{HV}$ with $S_{HV}=1000$ | `_calculate_hypervolume_reward()` — uses `tanh(delta/1000)` ([L220-248](src/rl/gym_env/reward_calculator.py#L220-L248)) | ⚠️ `tanh` ≠ linear division |
| $r_{\text{diversity}}$ | $0.1 \cdot \Delta(\bar{D}_{\text{geno}} + \bar{D}_{\text{pheno}})/2$ | `_calculate_diversity_bonus()` — uses `delta * 0.1` for raw diversity delta ([L178-190](src/rl/gym_env/reward_calculator.py#L178-L190)) | ⚠️ Uses population diversity, not average of geno+pheno |
| $r_{\text{time}}$ | $-0.01 \cdot t/T_{\max}$ | `_calculate_time_penalty()` — uses `0.001 * generation` (linear, no $T_{\max}$ normalization) ([L192-198](src/rl/gym_env/reward_calculator.py#L192-L198)) | ⚠️ Different scaling |
| Clipping | $\text{clip}(r_t, -1, 1)$ | `np.clip(total_reward, -1.0, 1.0)` | ✅ |
| Default weights | $w_f=1.0, w_d=0.1, w_t=0.01$ | `fitness_weight=1.0, diversity_weight=0.1, time_weight=0.01` | ✅ |

**Verdict**: The reward structure **matches Eq. 1** in form and weights. Three minor deviations:
1. Hypervolume mode uses `tanh` normalization instead of linear division
2. Diversity bonus uses raw population diversity delta, not the average of genotype+phenotype
3. Time penalty lacks $T_{\max}$ normalization

These are implementation refinements, not structural gaps. The thesis can describe both modes.

### PPO Agent (`src/rl/agents/ppo_agent.py`)

| Parameter | Thesis Spec | Code Reality | Source |
|---|---|---|---|
| Framework | Stable-Baselines3 | `from stable_baselines3 import PPO` | [L8](src/rl/agents/ppo_agent.py#L8) ✅ |
| Policy | MlpPolicy | `policy="MlpPolicy"` | [L82](src/rl/agents/ppo_agent.py#L82) ✅ |
| Hidden Layers | [64, 64] | **SB3 default for MlpPolicy = [64, 64]** — NOT explicitly set via `policy_kwargs` | ⚠️ Implicit |
| Learning Rate | $3 \times 10^{-4}$ | `ppo_config.learning_rate` → default `0.0003` | [helpers.py L177](src/rl/helpers.py#L177) ✅ |
| Clip Range | 0.2 | `ppo_config.clip_range` → default `0.2` | [helpers.py L183](src/rl/helpers.py#L183) ✅ |
| GAE Lambda | 0.95 | `ppo_config.gae_lambda` → default `0.95` | [helpers.py L182](src/rl/helpers.py#L182) ✅ |
| Entropy Coef | 0.01 | `ppo_config.ent_coef` → default `0.01` | [helpers.py L184](src/rl/helpers.py#L184) ✅ |
| VF Coef | 0.5 | `ppo_config.vf_coef` — **not in helpers default, uses SB3 default (0.5)** | ⚠️ Implicit |
| Batch Size | 64 | `ppo_config.batch_size` → default `64` | [helpers.py L179](src/rl/helpers.py#L179) ✅ |
| Gamma | 0.99 | `ppo_config.gamma` → default `0.99` | [helpers.py L181](src/rl/helpers.py#L181) ✅ |
| N Steps | 512 | `ppo_config.n_steps` → default `512` | [helpers.py L178](src/rl/helpers.py#L178) ✅ |
| N Epochs | 10 | `ppo_config.n_epochs` → default `10` | [helpers.py L180](src/rl/helpers.py#L180) ✅ |

### DQN Agent (`src/rl/agents/dqn_agent.py`)

| Parameter | Thesis Spec | Code Reality | Source |
|---|---|---|---|
| Framework | Stable-Baselines3 | `from stable_baselines3 import DQN` | [L8](src/rl/agents/dqn_agent.py#L8) ✅ |
| Policy | MlpPolicy | `policy="MlpPolicy"` | [L65](src/rl/agents/dqn_agent.py#L65) ✅ |
| Hidden Layers | [64, 64] | **SB3 default = [64, 64]** — NOT explicitly set | ⚠️ Implicit |
| Learning Rate | $1 \times 10^{-4}$ | `dqn_config.learning_rate` → default `0.0001` | [helpers.py L189](src/rl/helpers.py#L189) ✅ |
| Buffer Size | 100,000 | `dqn_config.buffer_size` → default `100000` | [helpers.py L190](src/rl/helpers.py#L190) ✅ |
| Batch Size | 32 | `dqn_config.batch_size` → default `32` | [helpers.py L191](src/rl/helpers.py#L191) ✅ |
| Gamma | 0.99 | `dqn_config.gamma` → default `0.99` | [helpers.py L192](src/rl/helpers.py#L192) ✅ |
| Exploration Fraction | 0.1 | `dqn_config.exploration_fraction` → default `0.1` | [helpers.py L193](src/rl/helpers.py#L193) ✅ |
| Exploration Final Eps | 0.05 | `dqn_config.exploration_final_eps` → default `0.05` | [helpers.py L194](src/rl/helpers.py#L194) ✅ |

### Verdict: **FUNCTIONAL — Agents are correctly wired, hyperparameters match thesis**

The only "soft" gap: `net_arch=[64, 64]` is never explicitly passed via `policy_kwargs`. SB3's `MlpPolicy` default IS `[64, 64]`, so the thesis claim holds de facto. For robustness, `policy_kwargs=dict(net_arch=[64, 64])` should be explicitly set.

---

## Audit 4: The "Bridge" — RL ↔ GA Integration

### Current Truth

**There are two integration paths:**

#### Path A: `ScheduleEnv.step()` — Pure RL Training Loop

[schedule_env.py L205-400](src/rl/gym_env/schedule_env.py#L205-L400) implements a standard Gymnasium `step()`:

1. Agent calls `env.step(action)` with an action index
2. `ActionMapper.apply_action()` applies the selected heuristic to the **best individual** in the population
3. Modified individual replaces worst individual in population
4. Reward is calculated, observation is re-encoded
5. Step counter increments (`current_generation += 1`)

**Semantics**: 1 RL step ≈ 1 heuristic application on best individual. The env tracks `current_generation` and `current_step` equivalently — they increment together ([L413](src/rl/gym_env/schedule_env.py#L413)).

This is what runs during **SB3 training** (`agent.learn(timesteps=N)`).

#### Path B: `HybridController` — Production Deployment

[hybrid_controller.py](src/rl/hybrid/hybrid_controller.py) provides a `select_action(state)` method that:
- Loads a trained model via `RLInference`
- Supports 3 modes: `RL_PRIMARY`, `RL_FALLBACK`, `RL_ASSISTED`
- Falls back to random/greedy/round-robin/recent-best

This is designed for **inference during a real GA run**, NOT for training.

#### Path C: `rl_experiment.py` — Experiment Harness

[rl_experiment.py](src/experiments/rl_experiment.py) provides 6 experiment classes:
- `RLTrainExperiment`: Train PPO/DQN and evaluate
- `RLCurriculumExperiment`: 3-stage curriculum learning
- `RLSpecialistExperiment`: Multi-agent specialist selection
- `RLRewardCompareExperiment`: Scalar vs hypervolume reward comparison
- `RLAdaptiveParamsExperiment`: Fixed vs adaptive GA parameters
- `RLAblationExperiment`: Systematic ablation (random/PPO/DQN)

All experiments use real scheduling data, real populations, and real heuristic applications.

### Thesis Gap

The thesis describes a **bi-level architecture** where:
> "A timestep corresponds to a single GA generation where the agent selects one heuristic operator to apply to the population."

The code's `ScheduleEnv.step()` does **not run a full GA generation** per step. Instead:
- 1 RL step = 1 heuristic applied to best individual, then replace worst
- There is no NSGA-II selection/crossover/mutation cycle running per step
- The "generation" counter increments per step, but this is a **naming artifact** — it's really counting RL steps, not GA generations

**The bridge does NOT currently implement the AOS pattern described in the thesis**, where:
1. GA runs one full generation (selection → crossover → mutation → evaluation)
2. RL agent observes the resulting population state
3. RL agent selects which operator modification to apply before next generation
4. Repeat

Instead, the current system is a **standalone RL-driven search** that uses GA heuristics as actions but does NOT wrap a running GA loop.

### Is The Bridge Broken by Vectorization?

The `BitsetSchedulingRepair` and pymoo-based vectorized pipeline (`repair_operator_bitset.py`, `repair_operator_vectorized.py`) operate on **numpy chromosome arrays** (shape `(N, E, 3)` — population × events × [instructor, room, start_quanta]).

The RL environment operates on **DEAP Individual objects** (lists of `SessionGene` with `.fitness` metadata).

**These are two incompatible representations.** The RL environment cannot currently invoke `BitsetSchedulingRepair` because:
1. `BitsetSchedulingRepair.repair(chromosome)` expects a 1D numpy array, not a DEAP Individual
2. The RL `ActionMapper` calls heuristic functions via `function(individual, context)` — the bitset repairer has a different API
3. Population management differs: DEAP lists vs. numpy 2D arrays

The existing GA heuristics in `src/ga/heuristics/` (construction, perturbation, improvement, diversity, meta, original repair functions) **still work** with the RL environment because they operate on DEAP Individuals. Only the new bitset/vectorized pipeline is disconnected.

### Verdict: **FUNCTIONING but ARCHITECTURALLY DIVERGENT from thesis AOS model**

The RL system trains and evaluates successfully as a standalone heuristic selection agent. However:
1. It does NOT wrap a running GA loop (no AOS pattern)
2. It cannot access the new vectorized repair pipeline
3. The "generation" concept is misleading — it's counting RL steps

---

## Consolidated Action Plan

### Priority 1: Fix State Space (39-D → 41-D) — ~1 hour

1. **Add 2 soft constraint names** to `StateEncoder.SOFT_CONSTRAINT_NAMES`:
   - `"session_continuity"` — exists in config, measures temporal consistency
   - `"soft_weight_factor"` OR split an existing constraint — align with thesis table
2. **Update `_normalize_observation()`**: shift constraint range `[17:29]` → `[17:31]`, history `[29:]` → `[31:]`
3. **Ensure evaluator populates** `individual.constraint_breakdown` for the new constraints
4. **Verify**: `observation_dim` auto-returns 41, `observation_space` shape updates

### Priority 2: Pin Action Space to Discrete(20) — ~2-3 hours

**Option A (Recommended — Code-first)**: Select exactly 19 heuristics + 1 no-op from the existing 26-heuristic registry by pinning a static `action_id_map` in the config. This already has infrastructure via `_build_action_space_with_mapping()` ([action_space.py L100-146](src/rl/gym_env/action_space.py#L100-L146)).

**Option B (Thesis-first)**: Implement the 5 mutation-rate-control + 2 crossover-modulation actions as new "meta-parameter" heuristics that modify `config.ga.mutpb` / `config.ga.cxpb` rather than applying a search operator. This requires:
- New functions in `src/ga/heuristics/` that return modified GA parameters
- ActionMapper changes to handle parameter-modification vs. individual-modification actions

### Priority 3: Explicitly Set net_arch — ~5 minutes

In `ppo_agent.py` and `dqn_agent.py`, add:
```python
policy_kwargs=dict(net_arch=[64, 64])
```
This makes the thesis claim explicitly verifiable instead of relying on SB3 defaults.

### Priority 4: Implement True AOS Bridge — ~4-6 hours

To match the thesis bi-level architecture, create a new integration point:

```python
# src/rl/hybrid/aos_bridge.py
class AdaptiveOperatorSelection:
    """1 GA Generation = 1 RL Step wrapper."""
    
    def __init__(self, ga_scheduler, rl_agent, state_encoder):
        self.ga = ga_scheduler
        self.agent = rl_agent
        self.encoder = state_encoder
    
    def run_hybrid(self, max_generations):
        for gen in range(max_generations):
            # 1. Observe population state
            state = self.encoder.encode(self.ga.population, gen, ...)
            
            # 2. RL selects action
            action = self.agent.predict(state)
            
            # 3. Apply selected operator modification
            self._apply_meta_action(action)
            
            # 4. Run one full GA generation
            self.ga.run_one_generation()
            
            # 5. Calculate reward
            reward = self._calculate_reward()
```

This wraps the existing NSGA-II scheduler (`GAScheduler`) and calls `rl_agent.predict()` once per generation.

### Priority 5: Bridge Vectorized Pipeline — ~3-4 hours

To use `BitsetSchedulingRepair` as an RL action:
1. Add a **conversion layer** (DEAP Individual ↔ numpy chromosome)
2. Register bitset repair stages as atomic heuristic functions in `all_heuristics.py`
3. Each atomic LLH wraps one stage of `BitsetSchedulingRepair`:
   - `bitset_repair_domain_clamp(individual, context)` → converts to numpy, runs stage 1, converts back
   - `bitset_repair_room_clash(individual, context)` → stage 2 room conflicts
   - `bitset_repair_instructor_clash(individual, context)` → stage 2 instructor conflicts
   - `bitset_repair_group_deconflict(individual, context)` → stage 3

### Priority 6: Align Reward Formula — ~30 minutes

1. Fix `_calculate_time_penalty()` to use `generation / max_generations` instead of `0.001 * generation`
2. Fix `_calculate_diversity_bonus()` to average genotype + phenotype diversity per thesis spec
3. Hypervolume mode: document `tanh` normalization as an implementation refinement in thesis

---

## File Inventory

| File | Status | Lines | Role |
|---|---|---|---|
| `src/rl/gym_env/state_encoder.py` | **FUNCTIONAL** | 588 | 39-D state encoding (needs +2 dims) |
| `src/rl/gym_env/action_space.py` | **FUNCTIONAL** | 555 | Dynamic action mapping with timeout protection |
| `src/rl/gym_env/reward_calculator.py` | **FUNCTIONAL** | 376 | Multi-component reward with hypervolume option |
| `src/rl/gym_env/schedule_env.py` | **FUNCTIONAL** | 679 | Full Gymnasium env with profiling |
| `src/rl/gym_env/hypervolume.py` | **FUNCTIONAL** | — | Hypervolume indicator calculation |
| `src/rl/agents/ppo_agent.py` | **FUNCTIONAL** | 131 | SB3 PPO wrapper (thesis-compliant hyperparams) |
| `src/rl/agents/dqn_agent.py` | **FUNCTIONAL** | 119 | SB3 DQN wrapper (thesis-compliant hyperparams) |
| `src/rl/agents/random_agent.py` | **FUNCTIONAL** | 120 | Random baseline agent |
| `src/rl/agents/specialist_agents.py` | **FUNCTIONAL** | — | Multi-agent specialist system |
| `src/rl/hybrid/hybrid_controller.py` | **FUNCTIONAL** | 298 | Production hybrid RL+fallback controller |
| `src/rl/deployment/inference.py` | **FUNCTIONAL** | 312 | Fast inference engine (<10ms target) |
| `src/rl/training/trainer.py` | **FUNCTIONAL** | 797 | Full training pipeline with TB logging |
| `src/rl/training/callbacks.py` | **FUNCTIONAL** | — | SB3 training callbacks |
| `src/rl/training/curriculum.py` | **FUNCTIONAL** | — | 3-stage curriculum learning |
| `src/rl/helpers.py` | **FUNCTIONAL** | 418 | Notebook/experiment utilities |
| `src/experiments/rl_experiment.py` | **FUNCTIONAL** | 744 | 6 experiment runner classes |
| `src/ga/heuristics/all_heuristics.py` | **FUNCTIONAL** | 440 | 26 heuristics, 6 categories |

**Total RL codebase**: ~5,000+ lines of production code across 17+ files.

**This is NOT a placeholder system.** It is a complete, trainable, evaluable RL hyper-heuristic with real heuristic execution, real fitness evaluation, and real multi-agent support.
