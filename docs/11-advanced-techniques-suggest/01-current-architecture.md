# Current Architecture: Memetic NSGA-II with RL Hyper-Heuristic

**Status**:  Implemented  
**Type**: Architecture Analysis  
**Last Updated**: November 17, 2025

---

## Executive Summary

Yes, you **already have a Memetic NSGA-II with RL-guided local search**. Your system combines:
1. **NSGA-II** for multi-objective population evolution (global search)
2. **IGLS** for local refinement on elite solutions (local search)
3. **RL** for intelligent operator selection (learning component)

This is a **memetic algorithm** because it integrates evolutionary search with local refinement and learned heuristics.

---

## What Makes This a Memetic Algorithm?

### Definition: Memetic Algorithm
A memetic algorithm is an extension of evolutionary algorithms that incorporates:
- **Genetic transmission** (crossover/mutation of solutions)
- **Memetic transmission** (local refinement, learning, cultural evolution)

### Your Implementation

#### 1. Population-Based Search (NSGA-II)
```
Component: NSGA-II multi-objective genetic algorithm
Purpose: Global exploration of solution space
Mechanism: Selection, crossover, mutation
Location: src/core/ga_scheduler.py
```

**Key Features**:
- Two objectives: hard constraint violations, soft constraint penalties
- Pareto-based selection (non-dominated sorting)
- Crowding distance for diversity
- Elitism (best solutions preserved)

#### 2. Local Search (IGLS)
```
Component: Iterated Greedy Local Search
Purpose: Intensification and refinement
Mechanism: Conflict-focused repair
Location: src/lns/igls_repair.py
```

**Triggers**:
- Every N generations (periodic)
- On stagnation detection
- Applied to elite individuals (best in population)

**Process**:
1. Detect hard constraint violations
2. Extract conflicted sessions
3. Apply IGLS repair (greedy construction + local search)
4. Reintegrate if improved

#### 3. Learning Component (RL)
```
Component: PPO/DQN reinforcement learning agent
Purpose: Learn optimal heuristic selection policy
Mechanism: State → Action → Reward feedback
Location: src/rl/
```

**What RL Learns**:
- When to perturb vs improve vs diversify
- Which specific heuristic to apply
- Adaptive strategy based on population state

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    MEMETIC NSGA-II SYSTEM                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌─────────────────┐    ┌──────────────┐
│  NSGA-II      │    │  RL Hyper-      │    │  Local       │
│  (Global)     │◄───┤  Heuristic      │───►│  Search      │
│               │    │  (Learning)     │    │  (IGLS)      │
└───────────────┘    └─────────────────┘    └──────────────┘
        │                     │                     │
        │            ┌────────┴────────┐            │
        │            ▼                 ▼            │
        │    ┌──────────────┐  ┌──────────────┐    │
        │    │ Construction │  │ Perturbation │    │
        │    │  (3 ops)     │  │   (5 ops)    │    │
        │    └──────────────┘  └──────────────┘    │
        │            │                 │            │
        │    ┌──────────────┐  ┌──────────────┐    │
        │    │ Improvement  │  │  Diversity   │    │
        │    │   (3 ops)    │  │   (4 ops)    │    │
        │    └──────────────┘  └──────────────┘    │
        │                  │                        │
        │         ┌────────┴────────┐               │
        │         │   Meta (4 ops)  │               │
        │         └─────────────────┘               │
        │                                            │
        └──────────────── Population ────────────────┘
                      (200 individuals)
```

---

## Why You Said "Option 1: Memetic NSGA-II with RL Local Search"

Because your **current implementation** already has all three components:

### Option 1 Components ( All Present)
1.  **Memetic**: Combines GA + local search
2.  **NSGA-II**: Multi-objective Pareto optimization
3.  **RL**: Learns heuristic selection policy
4.  **Local Search**: IGLS repair on elite solutions

### Why Also "Option 2: RL-Guided Initialization"?

This is **future enhancement**. Currently:
- Initial population: 25% greedy, 50% smart, 25% random (hardcoded)
- Future: RL agent learns which construction heuristics to use for initialization

---

## Metaheuristic vs Hyperheuristic Clarification

### Metaheuristic (What NSGA-II Is)
**Definition**: A high-level problem-independent optimization strategy that guides other heuristics.

**Examples**:
- Genetic Algorithms (GA)
- NSGA-II
- Simulated Annealing
- Tabu Search
- Ant Colony Optimization

**Level**: Solution space search

### Hyperheuristic (What RL Agent Is)
**Definition**: A higher-level strategy that selects or generates heuristics.

**Your RL Agent**:
- **Input**: Population state (fitness, diversity, stagnation)
- **Output**: Which heuristic to apply
- **Learning**: Improves selection policy over time

**Level**: Heuristic space search

### Relationship in Your System

```
┌────────────────────────────────────────┐
│        Hyperheuristic (RL)             │  ← Learns WHICH heuristic
│  "What operator should I use now?"     │
└────────────────┬───────────────────────┘
                 │ selects
                 ▼
┌────────────────────────────────────────┐
│    Metaheuristic (NSGA-II)             │  ← Guides HOW to search
│  "How should I evolve population?"     │
└────────────────┬───────────────────────┘
                 │ applies to
                 ▼
┌────────────────────────────────────────┐
│      Heuristics (19 operators)         │  ← Modify solutions
│  "Swap sessions, shift times, ..."    │
└────────────────────────────────────────┘
```

---

## Round Robin vs RL: Both Are Hyperheuristics

### Round Robin Selector
```python
# Hyperheuristic: Simple selection strategy
heuristics = [h1, h2, h3, h4, h5]
current_index = 0

def select_heuristic():
    h = heuristics[current_index]
    current_index = (current_index + 1) % len(heuristics)
    return h
```

**Characteristics**:
- No learning
- Fixed cycle
- Ignores problem state
- Works on heuristic space

### RL Selector (Your System)
```python
# Hyperheuristic: Learning-based selection
def select_heuristic(state):
    # state = [fitness, diversity, stagnation, ...]
    action_probs = rl_agent.predict(state)
    heuristic_id = sample(action_probs)
    return heuristics[heuristic_id]
```

**Characteristics**:
- Learns from experience
- State-dependent selection
- Adapts to problem characteristics
- Works on heuristic space

**Both are hyperheuristics** because they operate on the heuristic space, not solution space.

---

## Where Do Operators Work?

### Population-Level Operators (Metaheuristic)
- **Selection** (tournament, NSGA-II): Choose parents from population
- **Elitism**: Preserve best solutions across generations
- **Diversity metrics**: Measure population-wide characteristics

**Level**: Population ↔ Population

### Individual-Level Operators (Heuristics)
- **Crossover**: Chromosome₁ + Chromosome₂ → Offspring
- **Mutation**: Chromosome → Modified Chromosome
- **Repair**: Chromosome → Repaired Chromosome
- **Local Search**: Chromosome → Improved Chromosome

**Level**: Individual(s) → Individual(s)

### Gene-Level Operators (Low-level)
- **Swap genes**: Gene[i] ↔ Gene[j]
- **Mutate gene**: Gene → {new time, new room, new instructor}
- **Shift gene**: Gene.quanta += offset

**Level**: Gene → Gene

---

## Why Not Use RL for Constraint Weights?

### Current Approach: Hardcoded Weights
```yaml
# configs/base.yaml
hard_constraints:
  no_group_overlap:
    enabled: true
    weight: 1.0
  no_instructor_overlap:
    enabled: true
    weight: 1.0
  instructor_availability:
    enabled: true
    weight: 1.0

soft_constraints:
  minimize_gaps:
    enabled: true
    weight: 0.5
  instructor_preference:
    enabled: true
    weight: 0.3
```

**Why This Works**:
- Hard constraints are **binary**: violate or not (all weight 1.0)
- Soft constraints reflect **domain knowledge**: what's more important
- Weights are problem-specific, not instance-specific

### RL for Constraint Weights: Pros & Cons

#### Potential Benefits
- Adapt weights to problem characteristics
- Learn trade-offs between objectives
- Handle dynamic importance

#### Why We Don't Do It (Yet)

**1. Hard Constraints Are Absolute**
```
Hard constraint violation = infeasible solution
No need to learn: weight = ∞ (or 1.0 in practice)
```

**2. Soft Constraints Are Domain-Specific**
```
"Students prefer no gaps" - educational domain knowledge
"Instructors prefer morning slots" - institutional policy
These don't change per problem instance
```

**3. RL Already Controls Strategy**
```
Instead of tuning weights, RL learns:
- When to prioritize feasibility vs quality
- Which constraints to repair first
- How much search effort to allocate
```

**4. Weight Tuning Is Expensive**
```
Learning 20+ constraint weights requires:
- Thousands of problem instances
- Clear reward signal for each weight
- Risk of overfitting to training set
```

### When RL Constraint Weights Make Sense

**Scenario 1: Instance-Specific Preferences**
```python
# Different universities have different priorities
if university == "A":
    prefer minimize_gaps
else:
    prefer instructor_preference
    
# RL could learn these patterns
```

**Scenario 2: Multi-Stakeholder Problems**
```python
# Balance competing objectives dynamically
weights = rl_agent.predict(stakeholder_feedback)
```

**Scenario 3: Online Learning**
```python
# Adjust weights based on user satisfaction
production_feedback → update_weights()
```

---

## Current Strengths

### 1. Multi-Objective Optimization 
- Pareto-based selection preserves trade-offs
- No need to collapse to single objective
- Natural handling of hard/soft constraint conflict

### 2. Diverse Heuristic Portfolio 
- 19 operators across 5 categories
- Construction, perturbation, improvement, diversity, meta
- Covers exploration, exploitation, and maintenance

### 3. Learning-Based Adaptation 
- RL learns operator effectiveness
- Curriculum learning (easy → medium → hard)
- State-dependent strategy

### 4. Local Refinement 
- IGLS repair for conflict resolution
- Applied to elite solutions
- Triggered by stagnation

### 5. Domain Knowledge Integration 
- Hardcoded constraint weights (educational expertise)
- Constraint-aware initialization
- Course-group relationship preservation

---

## Current Limitations

### 1. Scalar Reward Function
```python
# Current: Single scalar reward
reward = fitness_improvement + diversity_bonus - time_penalty

# Problem: Collapses multi-objective to single value
# Loses information about Pareto trade-offs
```

**Impact**: RL agent doesn't understand Pareto dominance

### 2. Single Monolithic Agent
```python
# Current: One agent for all scenarios
agent = PPO(policy, env, ...)

# Problem: Same agent handles:
# - Infeasible solutions (need repair)
# - Feasible solutions (need optimization)
# - Stagnated populations (need diversity)
```

**Impact**: Agent must learn conflicting strategies

### 3. Coarse State Representation
```python
# Current: Aggregated constraint violations
state = [total_hard_violations, total_soft_penalty, ...]

# Missing: Which constraints are violated
```

**Impact**: Agent can't target specific constraint types

### 4. Fixed Local Search Budget
```python
# Current: IGLS runs for fixed iterations
igls_iterations = 100  # Hardcoded

# Problem: Doesn't adapt to:
# - Solution quality (near-feasible vs far from feasible)
# - Remaining computational budget
# - Stagnation severity
```

**Impact**: Wastes computation or terminates too early

### 5. No Diversity Archive
```python
# Current: Only fitness-based selection
population = select_best_by_pareto(population)

# Missing: Behavioral diversity preservation
```

**Impact**: Premature convergence to local optima

### 6. Static Operator Probabilities
```python
# Current: Fixed crossover/mutation probabilities
crossover_prob = 0.7  # Hardcoded
mutation_prob = 0.2   # Hardcoded

# Problem: Optimal balance changes during search
# Early: More exploration (higher mutation)
# Late: More exploitation (higher crossover)
```

**Impact**: Suboptimal exploration/exploitation balance

---

## Comparison: Different Runtime Configurations

### Mode 1: Pure NSGA-II (Baseline)
```yaml
mode: "pure_ga"
repair: {enabled: false}
rl: {enabled: false}
local_search: {enabled: false}
```

**Operators**: Only crossover + mutation (DEAP defaults)

**Use Case**: Baseline for benchmarking

### Mode 2: NSGA-II + Repairs
```yaml
mode: "ga_with_repair"
repair: {enabled: true, mode: "selective"}
rl: {enabled: false}
```

**Operators**: Crossover + mutation + repair (after each operator)

**Use Case**: Constraint handling without learning

### Mode 3: NSGA-II + Heuristics (Round Robin)
```yaml
mode: "ga_with_heuristics"
heuristics: {selector: "round_robin"}
rl: {enabled: false}
```

**Operators**: 19 heuristics, cycled through in order

**Use Case**: Test heuristic portfolio without RL overhead

### Mode 4: NSGA-II + Local Search
```yaml
mode: "ga_with_local_search"
local_search: {enabled: true, interval: 100}
rl: {enabled: false}
```

**Operators**: Crossover + mutation + IGLS (periodic)

**Use Case**: Memetic algorithm without RL

### Mode 5: Memetic NSGA-II + RL (Current)
```yaml
mode: "memetic_rl"
rl: {enabled: true, agent_type: "ppo"}
local_search: {enabled: true, rl_controlled: false}
```

**Operators**: 19 RL-selected heuristics + IGLS (triggered)

**Use Case**: Full learning-based hyperheuristic

### Mode 6: Enhanced Memetic RL (Future)
```yaml
mode: "memetic_rl_enhanced"
rl:
  enabled: true
  agent_type: "ppo"
  reward_type: "multi_objective"  # NEW
  specialist_agents: true          # NEW
  adaptive_probabilities: true     # NEW
local_search:
  enabled: true
  rl_controlled: true               # NEW: RL controls budget
```

**Operators**: Everything + enhancements from docs 02-11

**Use Case**: Research-level advanced techniques

---

## Suggested Enhancements Summary

### Better (Practical Improvements)
1. **Multi-objective reward**: Hypervolume instead of scalar
2. **Specialist agents**: Separate agents for feasible/infeasible
3. **Constraint-specific state**: Per-constraint breakdown
4. **Archive-based diversity**: Novelty search with behavioral archive
5. **Memetic RL**: RL controls local search intensity
6. **Adaptive probabilities**: RL tunes crossover/mutation rates

### Best (Research-Level)
7. **Multi-agent RL**: Ensemble of specialist agents per Pareto rank
8. **Hierarchical RL**: Two-level selection (category → heuristic)
9. **Transfer learning**: Pre-train on synthetic problems
10. **Online learning**: Continual adaptation from production runs

---

## Next Steps

1. **Validate Current Architecture**: Run benchmarks comparing modes 1-5
2. **Choose Enhancement**: Start with highest ROI (likely #1 or #3)
3. **Incremental Implementation**: Add one enhancement at a time
4. **Measure Impact**: Compare before/after on standard test set
5. **Iterate**: Continue to next enhancement if beneficial

---

## References

### Internal Documentation
- [RL-GA Integration Framework](../03-architecture/rl-ga-integ-framework.md)
- [Phase 2 RL Implementation](../06-development/implementation-notes/PHASE_2_RL_COMPLETE.md)
- [Heuristics Toolbox](../04-algorithms/HEURISTICS_QUICKREF.md)

### Key Papers
- Memetic Algorithms: Moscato (1989) - Original memetic algorithm concept
- NSGA-II: Deb et al. (2002) - Multi-objective GA
- Hyper-heuristics: Burke et al. (2013) - Survey of hyperheuristic methods
- Deep RL for Combinatorial Optimization: Bello et al. (2016)

---

**Summary**: You have a sophisticated memetic algorithm combining NSGA-II, RL-guided heuristics, and local search. The enhancements in this documentation suite will make it even better, but your current architecture is already research-grade and production-ready.
