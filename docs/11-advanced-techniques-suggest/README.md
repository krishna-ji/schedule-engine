# Advanced Techniques for Memetic NSGA-II Optimization

**Status**: 📝 Comprehensive Guide  
**Last Updated**: November 17, 2025  
**Purpose**: Detailed technical documentation for enhancing the current Memetic NSGA-II + RL architecture

---

## Overview

This directory contains comprehensive documentation on advanced optimization techniques for the schedule-engine's Memetic NSGA-II with RL-guided heuristic selection. The documentation progresses from describing the current architecture to increasingly sophisticated enhancement strategies.

---

## What is a Memetic Algorithm?

A **memetic algorithm** combines:
1. **Population-based search** (Genetic Algorithm/NSGA-II) - global exploration
2. **Local search/refinement** (IGLS, hill climbing) - local exploitation
3. **Cultural evolution** (learning, knowledge transfer) - RL-guided adaptation

The term "memetic" refers to **memes** (units of cultural transmission), analogous to genes in biological evolution. In our context:
- **Genes** = Schedule structures evolved by NSGA-II
- **Memes** = Heuristic strategies learned by RL agents

---

## Current Architecture (What We Have)

✅ **Memetic NSGA-II with RL Hyper-Heuristic**
- NSGA-II multi-objective optimization (hard/soft constraints)
- 19 heuristic operators in toolbox (5 categories: construction, perturbation, improvement, diversity, meta)
- RL-guided operator selection (PPO/DQN agents)
- IGLS-based local search for conflict resolution
- Curriculum learning (3 stages: easy → medium → hard)
- Pareto-based multi-objective optimization
- Hardcoded constraint weights (domain knowledge)

This is **already a memetic algorithm** because it combines:
- Global search (NSGA-II population evolution)
- Local search (IGLS repair, Kempe chains, ejection chains)
- Learning (RL agent learns operator selection policy)

---

## Documentation Structure

### Part 1: Current State Analysis
**[01-current-architecture.md](./01-current-architecture.md)** - Detailed analysis of existing implementation
- What makes our system "memetic"
- RL as hyperheuristic (meta-level decision making)
- Current strengths and limitations

### Part 2: Suggested Enhancements (Better)
Practical improvements to the existing architecture:

**[02-multi-objective-reward.md](./02-multi-objective-reward.md)** - Pareto-aware RL rewards
- Why scalar rewards fail for multi-objective problems
- Hypervolume indicator and decomposition methods
- Implementation strategies

**[03-specialist-agents.md](./03-specialist-agents.md)** - Task-specific RL agents
- Separate agents for feasible vs infeasible regions
- Agent switching and coordination
- Training separate value functions

**[04-constraint-specific-state.md](./04-constraint-specific-state.md)** - Enhanced state representation
- Per-constraint violation breakdown
- Attention mechanisms for constraint focus
- Guided repair strategies

**[05-archive-based-diversity.md](./05-archive-based-diversity.md)** - Novelty search
- Archive maintenance for behavioral diversity
- Novelty metrics vs fitness-only selection
- Quality-diversity algorithms (MAP-Elites)

**[06-memetic-rl.md](./06-memetic-rl.md)** - RL-guided local search
- Why RL should control local search intensity
- Adaptive computational budgets
- Meta-learning search strategies

**[07-adaptive-probabilities.md](./07-adaptive-probabilities.md)** - RL controls crossover/mutation
- Beyond operator selection: tuning operator parameters
- Adaptive exploration/exploitation balance
- Self-adaptive mechanisms

### Part 3: Advanced Ideas (Best)
Research-level techniques requiring significant development:

**[08-multi-agent-rl.md](./08-multi-agent-rl.md)** - Specialist per Pareto rank
- Ensemble of agents for different solution qualities
- Competitive/cooperative multi-agent learning
- Portfolio-based selection

**[09-hierarchical-rl.md](./09-hierarchical-rl.md)** - Two-level decision making
- High-level: select heuristic category
- Low-level: select specific heuristic
- Temporal abstraction and options framework

**[10-transfer-learning.md](./10-transfer-learning.md)** - Pre-train on synthetic problems
- Domain randomization for robustness
- Progressive difficulty curriculum
- Meta-learning across problem instances

**[11-online-learning.md](./11-online-learning.md)** - Adapt from production runs
- Continual learning without catastrophic forgetting
- Experience replay from production schedules
- Safe policy updates

### Part 4: Implementation Guidance

**[12-implementation-roadmap.md](./12-implementation-roadmap.md)** - How to approach enhancements
- Prioritization framework
- Incremental development strategy
- Validation and benchmarking protocols
- Risk assessment

---

## Key Concepts Covered

### Metaheuristics vs Hyperheuristics
- **Metaheuristic**: High-level problem-independent strategy (e.g., NSGA-II, Simulated Annealing)
- **Hyperheuristic**: Strategy that selects/generates heuristics (e.g., RL agent choosing operators)

Our system uses:
- NSGA-II as the metaheuristic (population-based multi-objective search)
- RL as the hyperheuristic (learning which operators to apply when)

### Multi-Objective Optimization
- Pareto dominance and Pareto fronts
- Hypervolume indicator
- Crowding distance and diversity preservation
- Decomposition methods (MOEA/D)

### Reinforcement Learning
- Markov Decision Processes (MDPs)
- Policy gradient methods (PPO)
- Value-based methods (DQN)
- Multi-objective RL
- Hierarchical RL
- Multi-agent RL
- Transfer learning and meta-learning

### Local Search Integration
- Variable Neighborhood Search (VNS)
- Large Neighborhood Search (LNS)
- Iterated Local Search (ILS)
- Adaptive memory structures

---

## Mathematical Foundations

Each document includes relevant mathematical formulations:
- Reward functions and value functions
- Pareto dominance relations
- Hypervolume calculations
- Novelty metrics
- Hierarchical action spaces
- Transfer learning objectives

---

## How to Use This Documentation

### For Understanding Current Architecture
Start with [01-current-architecture.md](./01-current-architecture.md)

### For Planning Next Enhancements
1. Read enhancements 02-07 to understand practical improvements
2. Review [12-implementation-roadmap.md](./12-implementation-roadmap.md) for prioritization
3. Choose based on your immediate goals and available resources

### For Research Directions
Study advanced ideas 08-11 for cutting-edge techniques requiring deeper investigation

### For Implementation
Each document includes:
- **Why**: Motivation and expected benefits
- **What**: Technical description with mathematics
- **How**: Implementation strategies and pseudocode
- **When**: Appropriate use cases and timing
- **Validation**: How to measure success

---

## Comparison: Different Runtime Modes

The system can operate in multiple modes:

1. **Pure NSGA-II** (baseline)
   - Only crossover/mutation
   - No repair, no local search

2. **NSGA-II + Repairs** 
   - Constraint repair after operators
   - No RL guidance

3. **NSGA-II + Heuristics**
   - 19-operator toolbox
   - Round-robin or random selection

4. **NSGA-II + Local Search**
   - IGLS repair on stagnation
   - No RL guidance

5. **Memetic NSGA-II + RL** (current)
   - RL-guided operator selection
   - IGLS local search
   - Curriculum learning

6. **Future Enhancements**
   - + Multi-objective RL rewards
   - + Specialist agents
   - + Hierarchical selection
   - + Online learning

See [01-current-architecture.md](./01-current-architecture.md) for detailed comparison.

---

## Related Documentation

- **Architecture**: [docs/03-architecture/rl-ga-integ-framework.md](../03-architecture/rl-ga-integ-framework.md)
- **RL Implementation**: [docs/06-development/implementation-notes/PHASE_2_RL_COMPLETE.md](../06-development/implementation-notes/PHASE_2_RL_COMPLETE.md)
- **Heuristics**: [docs/04-algorithms/HEURISTICS_QUICKREF.md](../04-algorithms/HEURISTICS_QUICKREF.md)
- **Performance**: [docs/05-performance/](../05-performance/)

---

## Quick Navigation

| Enhancement | Difficulty | Impact | Priority |
|-------------|-----------|--------|----------|
| Multi-objective reward | Medium | High | 1 |
| Constraint-specific state | Low | High | 2 |
| Adaptive probabilities | Medium | Medium | 3 |
| Specialist agents | Medium | Medium | 4 |
| Memetic RL (local search) | High | High | 5 |
| Archive-based diversity | Medium | Medium | 6 |
| Hierarchical RL | High | High | 7 |
| Multi-agent RL | Very High | Medium | 8 |
| Transfer learning | Very High | Medium | 9 |
| Online learning | High | Medium | 10 |

---

**Note**: All enhancements are optional and can be implemented incrementally. The current architecture is already production-ready and effective.
