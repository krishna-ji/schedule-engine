# Schedule Engine: Modes Documentation

This document provides a comprehensive overview of all optimization modes available in the Schedule Engine. Each mode represents a different approach to solving the university course timetabling problem using evolutionary algorithms and/or reinforcement learning.

---

## Table of Contents

1. [Overview](#overview)
2. [Mode Categories](#mode-categories)
3. [GA-Based Modes (A-E)](#ga-based-modes-a-e)
   - [Mode A: Baseline Pure NSGA-II](#mode-a-baseline-pure-nsga-ii)
   - [Mode B: Memetic NSGA-II](#mode-b-memetic-nsga-ii)
   - [Mode B1: Memetic + Budgeted Repair Operators](#mode-b1-memetic--budgeted-repair-operators)
   - [Mode B2: Memetic + Adaptive Targeting](#mode-b2-memetic--adaptive-targeting)
   - [Mode B3: Memetic + Two-Phase Strategy](#mode-b3-memetic--two-phase-strategy)
   - [Mode B4: Memetic + All Enhancements Combined](#mode-b4-memetic--all-enhancements-combined)
   - [Mode C: Round-Robin Heuristics](#mode-c-round-robin-heuristics)
   - [Mode D: Adaptive Heuristics](#mode-d-adaptive-heuristics)
   - [Mode E: RL-Guided NSGA-II](#mode-e-rl-guided-nsga-ii)
4. [RL Experiments (RL 01-10)](#rl-experiments-rl-01-10)
   - [RL 01: PPO Baseline](#rl-01-ppo-baseline)
   - [RL 02: DQN Baseline](#rl-02-dqn-baseline)
   - [RL 03: Curriculum Learning](#rl-03-curriculum-learning)
   - [RL 04: Specialist Agents](#rl-04-specialist-agents)
   - [RL 05: Reward Shaping](#rl-05-reward-shaping)
   - [RL 06: Adaptive Probabilities](#rl-06-adaptive-probabilities)
   - [RL 07: Full Ablation Study](#rl-07-full-ablation-study)
   - [RL 08: Hyperparameter Sensitivity](#rl-08-hyperparameter-sensitivity)
   - [RL 09: Multi-Agent Systems](#rl-09-multi-agent-systems)
   - [RL 10: Summary & Component Status](#rl-10-summary--component-status)
5. [Comparison Matrix](#comparison-matrix)
6. [Usage Guide](#usage-guide)
7. [Output Structure](#output-structure)

---

## Overview

The Schedule Engine uses a multi-objective optimization approach based on **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) to solve university course timetabling problems. The system minimizes two objectives:

1. **Hard Constraint Violations**: Room conflicts, instructor conflicts, group overlaps, qualification mismatches
2. **Soft Constraint Violations**: Preference violations, scheduling inefficiencies

Different modes enhance the base NSGA-II algorithm with various strategies:
- **Memetic algorithms** (local search after genetic operators)
- **Repair heuristics** (domain-specific fixes for violations)
- **Adaptive selection** (learning which heuristics work best)
- **Reinforcement learning** (intelligent action selection)

---

## Mode Categories

| Category | Modes | Description |
|----------|-------|-------------|
| **Baseline** | Mode A | Pure NSGA-II without enhancements |
| **Memetic** | Mode B, B1-B4 | NSGA-II + Local search/repair operators |
| **Heuristic Selection** | Mode C, D | Fixed or adaptive repair heuristic selection |
| **RL-Guided** | Mode E, RL 01-10 | Reinforcement learning for decision making |

---

## GA-Based Modes (A-E)

### Mode A: Baseline Pure NSGA-II

**File**: `runs/mode_a_baseline.py`

**Purpose**: Establish a baseline for comparison. This is pure NSGA-II with no enhancements.

**What It Does**:
- Creates random initial population of schedules
- Applies standard genetic operators (crossover, mutation)
- Uses NSGA-II selection (non-dominated sorting + crowding distance)
- No repair heuristics or local search
- Relies purely on evolutionary pressure to reduce violations

**Key Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `POP_SIZE` | 50 | Population size |
| `NGEN` | 1000 | Number of generations |
| `CXPB` | 0.9 | Crossover probability |
| `MUTPB` | 0.2 | Mutation probability |
| `FITNESS_WEIGHTS` | (-1.0, -1.0) | Minimize both objectives equally |

**Use Case**: 
- Benchmarking other modes
- Understanding raw GA performance without enhancements
- Baseline for academic comparisons

**Expected Output**:
- Higher violation counts than enhanced modes
- Slower convergence
- May not reach feasibility within given generations

---

### Mode B: Memetic NSGA-II

**File**: `runs/mode_b_memetic.py`

**Purpose**: Improve solution quality by applying local search (repair) after genetic operators.

**What It Does**:
1. Runs standard NSGA-II genetic operators (crossover, mutation)
2. With probability `LOCAL_SEARCH_PROB`, applies repair heuristics to offspring
3. Repair heuristics fix constraint violations:
   - `move_time`: Reschedule session to conflict-free time slot
   - `swap_room`: Change to a suitable room
   - `reassign_instructor`: Assign qualified, available instructor
4. Uses `RepairEngine` with configurable policies

**Key Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `POP_SIZE` | 50 | Population size |
| `NGEN` | 1000 | Number of generations |
| `LOCAL_SEARCH_PROB` | 0.2 | Probability of applying local search |
| `LOCAL_SEARCH_ITERATIONS` | 8 | Max repair iterations per individual |
| `REPAIR_POLICY` | "round_robin" | Cycles through repair operators |
| `REPAIR_BUDGET_MS` | 120.0 | Time budget for repair (milliseconds) |

**Algorithm Flow**:
```
for each generation:
    offspring = apply_crossover(population)
    offspring = apply_mutation(offspring)
    
    for each individual in offspring:
        if random() < LOCAL_SEARCH_PROB:
            individual = repair_engine.repair(individual)
    
    population = nsga2_selection(population + offspring)
```

**Use Case**:
- Faster convergence than pure NSGA-II
- Better final solution quality
- Standard memetic algorithm approach

---

### Mode B1: Memetic + Budgeted Repair Operators

**File**: `runs/mode_b1_repair_operators.py`

**Purpose**: Add budget constraints to repair operations and use lexicographic scoring.

**What It Does**:
1. Same as Mode B, but with enhanced repair engine
2. **Lexicographic scoring**: Prioritizes hard constraints over soft constraints
3. **Budget enforcement**: Limits repair time per individual
4. Uses domain-safe repair operators that maintain schedule validity

**Key Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `POP_SIZE` | 50 | Population size |
| `NGEN` | 200 | Fewer generations (relies on repair) |
| `REPAIR_PROB` | 0.3 | Higher repair probability |
| `REPAIR_BUDGET_MS` | 50.0 | Tighter time budget |
| `REPAIR_MAX_STEPS` | 5 | Maximum repair steps |

**Improvement over Mode B**:
- More controlled repair process
- Prevents excessive computation on unfixable individuals
- Prioritizes feasibility (hard constraints) over quality (soft constraints)

**Use Case**:
- When computation time is limited
- When feasibility is more important than optimality
- Production environments with strict time requirements

---

### Mode B2: Memetic + Adaptive Targeting

**File**: `runs/mode_b2_adaptive_targeting.py`

**Purpose**: Intelligently target repair operators based on constraint violation history.

**What It Does**:
1. Tracks constraint violations over generations
2. Identifies "stagnant" constraints (not improving)
3. Prioritizes repair operators that fix stagnant constraints
4. Uses constraint-to-operator mapping:

| Constraint | Repair Operators |
|------------|------------------|
| `student_group_exclusivity` | move_time |
| `instructor_exclusivity` | move_time |
| `room_exclusivity` | move_time |
| `instructor_time_availability` | move_time, reassign_instructor |
| `instructor_qualifications` | reassign_instructor |
| `room_suitability` | swap_room |

**Algorithm Flow**:
```
violation_history = {constraint_name: [violation_counts]}

for each generation:
    # ... standard NSGA-II + repair ...
    
    stagnant = identify_stagnant_constraints(violation_history)
    priority_operators = get_priority_operators(stagnant)
    
    # Prioritize operators that fix stagnant constraints
    repair_engine.set_priority(priority_operators)
```

**Key Functions**:
- `get_stagnant_constraints()`: Finds constraints not improving over lookback window
- `get_priority_operators()`: Maps stagnant constraints to fixing operators

**Use Case**:
- When certain constraints are persistently violated
- Adaptive optimization that responds to evolution dynamics
- Research into constraint-aware repair

---

### Mode B3: Memetic + Two-Phase Strategy

**File**: `runs/mode_b3_two_phase.py`

**Purpose**: Switch optimization strategy mid-run from exploration to exploitation.

**What It Does**:

| Phase | Generations | Repair Prob | Iterations | Rationale |
|-------|-------------|-------------|------------|-----------|
| **Phase 1** | 0-200 | 20% | 3 | Explore search space broadly |
| **Phase 2** | 201+ | 50% | 10 | Intensive local refinement |

**Strategy**:
1. **Phase 1 (Exploration)**: Low repair probability allows population diversity
2. **Phase 2 (Exploitation)**: High repair probability polishes best solutions

**Key Parameters**:
| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| `REPAIR_PROB` | 0.2 | 0.5 |
| `REPAIR_ITERATIONS` | 3 | 10 |
| `PHASE_SWITCH_GEN` | 200 | - |

**Algorithm Flow**:
```
for gen in range(NGEN):
    if gen <= PHASE_SWITCH_GEN:
        repair_prob = PHASE1_REPAIR_PROB
        repair_iterations = PHASE1_REPAIR_ITERATIONS
    else:
        repair_prob = PHASE2_REPAIR_PROB
        repair_iterations = PHASE2_REPAIR_ITERATIONS
    
    # ... NSGA-II with dynamic repair settings ...
```

**Use Case**:
- Long optimization runs
- When early diversity is important
- Avoiding premature convergence

---

### Mode B4: Memetic + All Enhancements Combined

**File**: `runs/mode_b4_combined.py`

**Purpose**: Combine all Mode B enhancements: B1 + B2 + B3.

**What It Does**:
1. **From B1**: Budgeted repair with lexicographic scoring
2. **From B2**: Adaptive targeting of stagnant constraints
3. **From B3**: Two-phase exploration → exploitation strategy

| Phase | Generations | Target | Iterations | Strategy |
|-------|-------------|--------|------------|----------|
| **Phase 1** | 0-199 | Worst 20% | 3 | Light exploration |
| **Phase 2** | 200+ | Worst 40% | 8 | Intensive exploitation |

**Key Features**:
- Targets worst-performing individuals for repair
- Adapts operator selection based on violation history
- Phases shift from exploration to exploitation

**Use Case**:
- Best overall memetic algorithm performance
- Production use when maximum quality is needed
- Research comparisons

---

### Mode C: Round-Robin Heuristics

**File**: `runs/mode_c_roundrobin.py`

**Purpose**: Simple deterministic cycling through repair heuristics.

**What It Does**:
- Applies repair heuristics in fixed round-robin order
- Each repair step uses the next heuristic in the sequence
- No learning or adaptation - purely deterministic

**Repair Cycle**:
```
Heuristics: [move_time, swap_room, reassign_instructor]

Step 1: move_time
Step 2: swap_room
Step 3: reassign_instructor
Step 4: move_time (cycle repeats)
...
```

**Key Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `REPAIR_PROB` | 0.45 | Moderate repair probability |
| `REPAIR_POLICY` | "round_robin" | Fixed cycling |
| `REPAIR_BUDGET_MS` | 120.0 | Time budget |

**Use Case**:
- Baseline for comparing adaptive methods
- Simple, predictable behavior
- When operator effectiveness is unknown

---

### Mode D: Adaptive Heuristics

**File**: `runs/mode_d_adaptive.py`

**Purpose**: Learn which heuristics work best and adapt selection probabilities.

**What It Does**:
1. Uses **epsilon-greedy** selection policy
2. Tracks success rate of each repair operator
3. Exploits best-performing operators (1-ε probability)
4. Explores random operators (ε probability)

**Epsilon-Greedy Algorithm**:
```
if random() < epsilon:
    operator = random_choice(all_operators)  # Explore
else:
    operator = best_performing_operator()     # Exploit

# After applying operator:
update_success_rate(operator, improvement)
```

**Key Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `REPAIR_POLICY` | "epsilon_greedy" | Adaptive selection |
| `REPAIR_EPSILON` | 0.1 | 10% exploration |
| `REPAIR_BUDGET_MS` | 120.0 | Time budget |

**Use Case**:
- Automatically discovers effective operators
- Adapts to problem characteristics
- Better than Mode C when operators have different effectiveness

---

### Mode E: RL-Guided NSGA-II

**File**: `runs/mode_e_rl_guided.py`

**Purpose**: Use Q-learning to intelligently select repair heuristics.

**What It Does**:
1. Uses `SimpleRLSelector` with Q-learning
2. Observes schedule state (violation counts, fitness)
3. Selects repair action based on learned Q-values
4. Updates Q-table based on improvement achieved

**Q-Learning Update**:
```
Q(state, action) += α * (reward + γ * max(Q(next_state, :)) - Q(state, action))
```

**Key Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `LEARNING_RATE` | 0.2 | Q-learning α |
| `EPSILON_START` | 1.0 | Initial exploration rate |
| `EPSILON_END` | 0.1 | Final exploration rate |
| `EPSILON_DECAY` | 0.995 | Decay rate per episode |

**State Representation**:
- Hard constraint violations (discretized)
- Soft constraint violations (discretized)
- Generation progress

**Actions**:
- 0: move_time
- 1: swap_room
- 2: reassign_instructor

**Use Case**:
- Learning optimal repair strategies
- Online adaptation during evolution
- Research into RL-guided optimization

---

## RL Experiments (RL 01-10)

These experiments focus specifically on training and evaluating reinforcement learning agents for schedule optimization.

### RL 01: PPO Baseline

**File**: `runs/rl_01_ppo_baseline.py`

**Purpose**: Train and evaluate PPO (Proximal Policy Optimization) agent.

**What It Does**:
1. Creates `ScheduleEnv` (Gymnasium environment)
2. Trains PPO agent using Stable-Baselines3
3. Evaluates trained agent on scheduling task

**Key Parameters**:
| Parameter | Value |
|-----------|-------|
| `TIMESTEPS` | 5000 |
| `POP_SIZE` | 20 |
| `MAX_GENERATIONS` | 50 |
| `MAX_STEPS` | 20 |

**Use Case**: Baseline RL performance with state-of-the-art algorithm

---

### RL 02: DQN Baseline

**File**: `runs/rl_02_dqn_baseline.py`

**Purpose**: Train and evaluate DQN (Deep Q-Network) agent for comparison with PPO.

**What It Does**:
- Same setup as RL 01 but uses DQN
- Compares value-based (DQN) vs policy-based (PPO) approaches

**Use Case**: Understanding which RL algorithm suits the scheduling problem

---

### RL 03: Curriculum Learning

**File**: `runs/rl_03_curriculum_learning.py`

**Purpose**: Train agents with progressively harder problems.

**What It Does**:
Trains in three stages:

| Stage | Difficulty | Max Generations | Max Steps | Timesteps |
|-------|------------|-----------------|-----------|-----------|
| Easy | Low | 30 | 10 | 3000 |
| Medium | Medium | 50 | 15 | 4000 |
| Hard | High | 80 | 20 | 5000 |

**Benefits**:
- Agent learns fundamentals before tackling hard problems
- Faster learning than training on hard problem directly
- More robust final policy

---

### RL 04: Specialist Agents

**File**: `runs/rl_04_specialist_agents.py`

**Purpose**: Demonstrate multi-agent coordination with specialist agents.

**What It Does**:
1. Creates `AgentCoordinator` with multiple specialist agents
2. Each agent specializes in different problem aspects
3. Coordinator selects appropriate agent based on state

**Agent Types**:
- **Exploration Agent**: Favors diverse actions
- **Exploitation Agent**: Focuses on best-known actions
- **Constraint-specific Agents**: Target specific violations

**Use Case**: Complex problems requiring different strategies for different situations

---

### RL 05: Reward Shaping

**File**: `runs/rl_05_reward_shaping.py`

**Purpose**: Compare different reward calculation methods.

**What It Does**:
Compares two reward approaches:

1. **Scalar Reward**: Simple weighted sum of violations
   ```
   reward = -(hard_violations + 0.1 * soft_violations)
   ```

2. **Hypervolume Reward**: Multi-objective quality measure
   ```
   reward = hypervolume(pareto_front) / reference_volume
   ```

**Use Case**: Understanding which reward signal leads to better learning

---

### RL 06: Adaptive Probabilities

**File**: `runs/rl_06_adaptive_probabilities.py`

**Purpose**: Compare fixed vs adaptive GA probabilities.

**What It Does**:
- **Fixed Config**: Constant crossover/mutation probabilities
- **Adaptive Config**: Probabilities adjusted based on population diversity

**Use Case**: Determining if adaptive probabilities improve convergence

---

### RL 07: Full Ablation Study

**File**: `runs/rl_07_full_ablation_study.py`

**Purpose**: Systematic comparison across RL methods.

**What It Does**:
Compares three approaches with multiple trials:
- Random action selection (baseline)
- PPO (policy gradient)
- DQN (value-based)

**Output**: Statistical analysis (mean, std) across trials

---

### RL 08: Hyperparameter Sensitivity

**File**: `runs/rl_08_hyperparameter_sensitivity.py`

**Purpose**: Analyze sensitivity to learning rate.

**What It Does**:
Tests PPO with different learning rates:
- 1e-4 (conservative)
- 3e-4 (default)
- 1e-3 (aggressive)

**Use Case**: Tuning RL hyperparameters for best performance

---

### RL 09: Multi-Agent Systems

**File**: `runs/rl_09_multi_agent_systems.py`

**Purpose**: Detailed analysis of multi-agent selection dynamics.

**What It Does**:
1. Runs multiple episodes with AgentCoordinator
2. Tracks which agent is selected at each step
3. Analyzes selection patterns vs problem state

**Output**: Selection frequency analysis, state-action correlations

---

### RL 10: Summary & Component Status

**File**: `runs/rl_10_summary.py`

**Purpose**: Verify all RL components are properly configured.

**What It Does**:
- Checks availability of all RL components
- Verifies configuration settings
- Reports component status (available/missing)

**Use Case**: Debugging RL setup issues

---

## Comparison Matrix

| Mode | Local Search | Adaptive | Two-Phase | RL | Complexity |
|------|--------------|----------|-----------|-------|------------|
| A | ❌ | ❌ | ❌ | ❌ | Low |
| B | ✅ | ❌ | ❌ | ❌ | Medium |
| B1 | ✅ (budgeted) | ❌ | ❌ | ❌ | Medium |
| B2 | ✅ | ✅ (constraint) | ❌ | ❌ | High |
| B3 | ✅ | ❌ | ✅ | ❌ | Medium |
| B4 | ✅ | ✅ | ✅ | ❌ | High |
| C | ✅ | ❌ (round-robin) | ❌ | ❌ | Low |
| D | ✅ | ✅ (ε-greedy) | ❌ | ❌ | Medium |
| E | ✅ | ✅ | ❌ | ✅ (Q-learning) | High |

---

## Usage Guide

### Running a Mode

```bash
# Run baseline
python runs/mode_a_baseline.py

# Run memetic
python runs/mode_b_memetic.py

# Run RL experiments
python runs/rl_01_ppo_baseline.py
```

### Comparing Results

All modes output to `output/<mode_name>/<timestamp>/`:
- `log_violations.log`: Detailed constraint violations
- `log_feasibility.log`: Feasibility analysis
- `metrics.json`: GA metrics (hypervolume, convergence)
- `best_schedule.json`: Best solution found

### Bulk Running

```bash
# Run modes A through D
python scripts/bulkrun_abcd.py
```

---

## Output Structure

```
output/
├── mode_a_baseline/
│   └── 20260205_114111/
│       ├── mode_a_baseline.log      # Detailed run log
│       ├── log_violations.log       # Constraint violation details
│       ├── log_feasibility.log      # Resource utilization analysis
│       ├── metrics.json             # GA performance metrics
│       ├── best_individual.json     # Best solution genes
│       └── decoded_schedule.json    # Human-readable schedule
├── mode_b_memetic/
│   └── ...
└── rl_01_ppo_baseline/
    └── ...
```

---

## Recommendations

| Scenario | Recommended Mode |
|----------|------------------|
| Quick baseline | Mode A |
| General use | Mode B or B1 |
| Persistent violations | Mode B2 |
| Long runs | Mode B3 |
| Maximum quality | Mode B4 |
| Simple repair | Mode C |
| Self-tuning repair | Mode D |
| Research / ML | Mode E, RL experiments |

---

## Further Reading

- [NSGA-II Paper](https://ieeexplore.ieee.org/document/996017)
- [Memetic Algorithms](https://en.wikipedia.org/wiki/Memetic_algorithm)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [DQN Paper](https://arxiv.org/abs/1312.5602)
