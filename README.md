<div align="center">

# Schedule Engine

### Hybrid Metaheuristic–Reinforcement Learning Framework for Multi-Objective University Course Timetabling

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Type Safety](https://img.shields.io/badge/mypy-strict%20mode-brightgreen)](https://mypy-lang.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![OR-Tools](https://img.shields.io/badge/OR--Tools-CP--SAT-red)](https://developers.google.com/optimization)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-PPO%20|%20DQN-orange)](https://stable-baselines3.readthedocs.io/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

> **TL;DR** — A research-grade optimization framework that hybridizes NSGA-II evolutionary search, deep reinforcement learning (PPO/DQN), multi-armed bandit operator selection, CP-SAT constraint propagation, and graph-theoretic chain neighborhoods to optimize NP-hard university course timetabling as a constrained multi-objective combinatorial optimization problem.

---

## Problem Statement

The **University Course Timetabling Problem (UCTP)** is a well-studied NP-hard combinatorial optimization problem. Given a set of courses $C$, instructors $I$, rooms $R$, student groups $G$, and discrete time quanta $Q$, the objective is to find an assignment:

$$f: C \times G \rightarrow I \times R \times Q$$

that satisfies a set of hard constraints $\mathcal{H}$ (feasibility) while simultaneously minimizing a vector of soft constraint penalties $\mathbf{s} = (s_1, s_2, \ldots, s_k)$ — framing it as a **constrained multi-objective optimization** problem:

$$\min_{\mathbf{x} \in \Omega} \; \mathbf{F}(\mathbf{x}) = \bigl(\sum_{h \in \mathcal{H}} v_h(\mathbf{x}),\; \sum_{s \in \mathcal{S}} w_s \cdot p_s(\mathbf{x})\bigr) \quad \text{s.t.} \; \mathbf{x} \in \Omega$$

This framework attacks the problem through a **layered metaheuristic architecture** — combining population-based evolutionary search with learned operator selection policies and exact decomposition repair.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SCHEDULE ENGINE FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌────────────────┐    ┌─────────────────────────┐  │
│  │  NSGA-II     │───▶│  Adaptive      │───▶│  Multi-Agent RL         │  │
│  │  Evolutionary│    │  Operator      │    │  Coordinator            │  │
│  │  Core        │    │  Selection     │    │  (PPO / DQN / MAB)      │  │
│  └──────┬───────┘    └────────┬───────┘    └────────────┬────────────┘  │
│         │                     │                          │               │
│         ▼                     ▼                          ▼               │
│  ┌──────────────┐    ┌────────────────┐    ┌─────────────────────────┐  │
│  │  Constraint  │    │  Repair        │    │  CP-SAT Hybrid          │  │
│  │  Evaluation  │    │  Pipeline      │    │  Decomposition          │  │
│  │  (Vectorized)│    │  (8 Operators) │    │  (OR-Tools)             │  │
│  └──────────────┘    └────────────────┘    └─────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Bitset-Accelerated Conflict Detection │ NumPy Batch Evaluation  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Algorithmic Components

### 1. Multi-Objective Evolutionary Optimization (NSGA-II)

The evolutionary backbone uses **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) via [pymoo](https://pymoo.org/) with custom genetic operators designed for the timetabling domain:

- **Structure-preserving crossover** — Position-independent recombination that swaps instructor/room/time attributes across individuals while preserving (course, group) gene topology. Validates instructor qualification and room feature compatibility at crossover time.
- **Constraint-aware mutation** — Per-gene mutation operators (`mutate_gene`, `mutate_time_quanta`) that respect domain invariants: contiguous session representation (`start_quanta` + `num_quanta`), day boundary constraints, instructor availability windows, and room feature requirements.
- **Lexicographic fitness** — Hard constraint violations dominate soft penalties in objective ordering, ensuring feasibility pressure before quality optimization.
- **Population diversity management** — Genotype, phenotype, and behavioral diversity metrics tracked per-generation with adaptive diversity maintenance.

### 2. Reinforcement Learning for Adaptive Operator Selection

The framework formulates **operator selection as a sequential decision problem**, training RL agents to learn which heuristic operator to apply at each evolutionary generation.

**Gymnasium Environment** (`ScheduleEnv`):

| Feature Group | Dimensionality | Description |
|---|---|---|
| Fitness landscape | 5 | Best, mean, worst, std, range of population fitness |
| Population diversity | 5 | Genotype, phenotype, behavioral, fitness diversity, unique ratio |
| Search dynamics | 4 | Generation progress, stagnation counter, convergence rate, improvement delta |
| Constraint decomposition | 12 | Per-constraint violation counts (8 hard + 4 soft) |
| Operator history | 10 | Recent action trajectory (temporal context) |
| **Total** | **36+** | Normalized observation vector ∈ [0, 1]³⁶ |

**Action space**: Discrete(20) — mapping to a portfolio of 19 heuristic operators (construction, perturbation, improvement, diversity preservation, targeted repair) plus a no-op action.

**Agents**:

- **PPO** (Proximal Policy Optimization) — On-policy actor-critic with clipped surrogate objective for stable operator selection learning.
- **DQN** (Deep Q-Network) — Off-policy value-based learning with experience replay and target network stabilization.
- **Specialist Agent Hierarchy** — Four domain-specific sub-agents (`RepairAgent`, `OptimizerAgent`, `ExplorerAgent`, `IntensifierAgent`) coordinated by:
  - State-based dispatch (search state → specialist mapping)
  - **UCB (Upper Confidence Bound)** bandit for meta-agent selection
  - Learned meta-policy for agent coordination

**Reward signal**: Composite reward balancing fitness improvement, diversity bonus, and computational cost penalty — bounded to $[-1, 1]$ for training stability.

### 3. Multi-Armed Bandit Operator Selection

For lightweight adaptive operator selection without deep RL overhead, the framework implements **bandit-based policies**:

- **ε-greedy** with adaptive decay schedule — Tracks empirical success rate (Δ hard violations, Δ soft penalties) per repair operator and selects greedily with probability $(1 - \varepsilon)$.
- Per-operator statistics tracking: application count, cumulative improvement, success rate — enabling online credit assignment for the operator portfolio.

### 4. CP-SAT Hybrid Decomposition Repair

A **two-phase exact repair pipeline** using Google OR-Tools CP-SAT solver for constraint-guaranteed feasibility restoration:

**Phase 1 — Global Bridge Resolution**: Identifies cross-cluster bridge genes (shared instructors, foundation courses spanning multiple programs) and solves them as a single CP model with `NoOverlap` interval constraints.

**Phase 2 — Parallel Cluster Decomposition**: Decomposes the remaining problem into independent cluster subproblems, solving each in parallel with bridge assignments frozen as hard constraints. Each subproblem models:

| Constraint | CP-SAT Encoding |
|---|---|
| Cohort Temporal Exclusivity (CTE) | `NoOverlap` per group family |
| Faculty Temporal Exclusivity (FTE) | Optional-interval `NoOverlap` per instructor |
| Spatial Resource Exclusivity (SRE) | Optional-interval `NoOverlap` per room |
| Faculty Pedagogical Congruence (FPC) | Domain restriction on instructor variable |
| Facility-Format Congruence (FFC) | Domain restriction on room variable |
| Faculty Chronological Availability (FCA) | Conditional start-quantum domain filtering |

### 5. Graph-Theoretic Local Search Neighborhoods

Beyond standard hill-climbing, the framework employs advanced neighborhood structures from the combinatorial optimization literature:

- **Kempe Chain Interchange** — Constructs a conflict graph between sessions, identifies maximal chains of dependent assignments, and performs simultaneous time-slot swaps along the chain — guaranteeing feasibility preservation during moves.
- **Ejection Chain Search** — Extends Kempe chains with cascading reassignment sequences: moving session $A$ ejects session $B$, whose reassignment ejects $C$, etc. — enabling complex multi-move transitions that escape local optima.
- **First-improvement / Steepest-descent** hill climbing with configurable evaluation budgets (10–200 neighbor evaluations per gene).

### 6. Constraint Model

The constraint system is formalized using academic nomenclature:

**Hard Constraints** $\mathcal{H}$ (feasibility — must be fully satisfied):

| Code | Constraint | Description |
|---|---|---|
| CTE | Cohort Temporal Exclusivity | No student group assigned to overlapping sessions |
| FTE | Faculty Temporal Exclusivity | No instructor teaching overlapping sessions |
| SRE | Spatial Resource Exclusivity | No room hosting overlapping sessions |
| FPC | Faculty Pedagogical Congruence | Instructor must be qualified for assigned course |
| FFC | Facility-Format Congruence | Room features must match course requirements |
| FCA | Faculty Chronological Availability | Part-time faculty only scheduled within availability windows |
| CQF | Curriculum Quanta Fulfillment | Each (course, group) pair receives exactly required weekly quanta |
| ICTD | Intra-Course Temporal Distribution | Sibling sessions of the same course not scheduled on the same day |

**Soft Constraints** $\mathcal{S}$ (quality — minimized with configurable weights $w_s$):

| Code | Constraint | Description |
|---|---|---|
| CSC | Cohort Schedule Compactness | Minimize idle gaps in student timetables |
| FSC | Faculty Schedule Compactness | Minimize idle gaps in instructor timetables |
| MIP | Mandatory Intermission Provision | Preserve lunch break windows for all cohorts |
| SSCP | Subcohort Congruence Penalty | Practical sessions aligned across paired cohort groups |

---

## Performance Engineering

- **Bitset-accelerated conflict detection** — `uint64` bitmask occupancy maps for $O(1)$ room/instructor/group conflict queries via bitwise AND operations.
- **Vectorized batch evaluation** — NumPy-based population-level constraint evaluation computing all 8 hard constraint columns simultaneously across the entire population.
- **CPU-parallelized repair** — Multi-process pipeline for independent repair operators and CP-SAT cluster decomposition.
- **GPU-accelerated RL** — PyTorch with CUDA 12.1 for neural network training and inference (PPO/DQN policy networks).

---

## MOEA Quality Indicators

The framework computes standard multi-objective quality metrics for rigorous algorithm comparison:

| Metric | Description |
|---|---|
| **Hypervolume (HV)** | Volume of objective space dominated by the Pareto front |
| **Inverted Generational Distance (IGD)** | Convergence to a reference Pareto front |
| **Additive Epsilon (ε⁺)** | Minimum translation to dominate reference front |
| **Spacing** | Uniformity of solution distribution along the front |
| **Feasibility Rate** | Proportion of feasible individuals in the population |
| **Convergence Rate** | Per-generation improvement trajectory |

---

## Experiment Pipeline

Progressive experiment modes from pure EA to full hybrid:

| Run Script | Mode | Algorithm |
|---|---|---|
| `ga_01_baseline` | A | Pure NSGA-II evolutionary search |
| `ga_02_memetic` | B | + Elite greedy local search (memetic) |
| `ga_03_aggressive` | C | + Aggressive repair with stagnation detection |
| `ga_04_adaptive` | D | + ε-greedy MAB adaptive operator selection |
| `ga_05_cp_hybrid` | E | + CP-SAT decomposition repair (OR-Tools) |
| `rl_01–rl_10` | RL | PPO/DQN training, curriculum learning, ablation, hyperparameter sweep, multi-agent coordination |

---

## Quick Start

```bash
# Install with uv (recommended)
uv sync --frozen

# Run baseline NSGA-II
uv run python runs/ga_01_baseline.py

# Run with adaptive MAB operator selection
uv run python runs/ga_04_adaptive.py

# Train PPO agent for operator selection
uv run python runs/rl_01_train_ppo.py

# Full test suite
uv run pytest tests/
```

---

## Repository Structure

```
schedule-engine/
├── src/
│   ├── config/          # Pydantic configuration models
│   ├── constraints/     # Hard/soft constraint formalization (CTE, FTE, SRE, ...)
│   ├── domain/          # Domain models (SessionGene, Course, Instructor, Room, Group)
│   ├── ga/
│   │   ├── operators/   # Crossover, mutation, selection operators
│   │   ├── repair/      # 8-operator repair pipeline + CP-SAT hybrid
│   │   └── metrics/     # Population diversity & convergence metrics
│   ├── rl/
│   │   ├── env/         # Gymnasium environment (36-dim obs, Discrete(20) action)
│   │   ├── agents/      # PPO, DQN, specialist hierarchy, multi-agent coordinator
│   │   └── rewards/     # Composite reward functions
│   ├── pipeline/        # Vectorized evaluators, bitset acceleration, batch API
│   ├── metrics/         # MOEA quality indicators (HV, IGD, ε⁺, spacing)
│   └── workflows/       # Experiment orchestration & progressive mode runner
├── runs/                # Executable experiment scripts (GA + RL)
├── tests/               # Comprehensive test suite
├── data/                # Problem instance data (JSON)
├── scripts/             # Profiling, benchmarking, diagnostics
└── docs/                # Architecture docs, vectorization plan, migration gates
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Evolutionary optimization | NSGA-II (pymoo), custom genetic operators |
| Reinforcement learning | Stable-Baselines3 (PPO, DQN), Gymnasium |
| Deep learning | PyTorch 2.4 + CUDA 12.1 |
| Exact solver | Google OR-Tools CP-SAT |
| Scientific computing | NumPy, SciPy, pandas |
| Visualization | matplotlib, seaborn, TensorBoard |
| Type safety | mypy (strict mode), Pydantic v2 |
| Package management | uv |

---

## License

MIT

---

<div align="center">
<sub>Built as a research exploration into hybrid metaheuristic–learning frameworks for constrained combinatorial optimization.</sub>
</div>
└── test/                # Unit tests
```

## Tech Stack

- **Python 3.12**
- **GA Core**: pymoo 0.6.1.3, NumPy 1.26.4
- **RL Stack**: PyTorch 2.4.1+CUDA12.1, Stable-Baselines3 2.3.2, Gymnasium 0.29.1
- **Config**: Pydantic 2.10.3
- **UI**: Rich 13.9.4, matplotlib, seaborn

## License

MIT
