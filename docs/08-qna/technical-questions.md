# Technical Questions & Answers

**Your active Q&A workspace for technical questions about the schedule-engine project**

---

## How to Use This Document

1. **Ask questions** - Technical, implementation, algorithmic questions
2. **AI will document** - Conversations recorded here
3. **Search later** - Reference past Q&A when similar issues arise
4. **Build knowledge** - Accumulate project-specific solutions

---

## Q&A Sessions

### [Session 1: November 17, 2025] - GPU Acceleration Implementation

**Q: Can we use NVIDIA GPU for both RL training and constraint checking?**

**A:** 
- **RL Training:** ✅ YES - GPU provides 3-5× speedup
  - Implementation: Changed `device: auto` → `device: cuda` in configs/base.yaml
  - Framework: Stable-Baselines3 with PyTorch backend already GPU-ready
  - Expected: 22.5 hours → 6 hours for full curriculum training
  - Hardware: 8GB GPU sufficient (only needs 200-800 MB VRAM)

- **Constraint Checking:** ❌ NO - GPU would be 2.4× slower
  - Reason: Small, irregular, dictionary-based operations
  - Memory transfer overhead: 50ms vs 40ms CPU computation
  - Constraint checking is only 4% of RL training time
  - Recommendation: Keep constraints on CPU

**Documentation:** See [05-performance/nvidia-gpu/](../05-performance/nvidia-gpu/)

---

### [Session 2: November 17, 2025] - Documentation Reorganization

**Q: How can I categorize docs/ folder? Too many loose files, hard to find things.**

**A:** Implemented 10-category structure:
1. **01-getting-started/** - User onboarding
2. **02-user-guides/** - Daily usage guides
3. **03-architecture/** - System design
4. **04-algorithms/** - Technical deep dives
5. **05-performance/** - Optimization & analysis
6. **06-development/** - Changelogs & dev notes
7. **07-thesis-report/** - Academic content
8. **08-qna/** - This document (Q&A workspace)
9. **09-future-plans/** - Roadmap & ideas
10. **10-ai-suggestions/** - AI-generated content
11. **archive/** - Historical reference

**Benefits:**
- 30+ loose files → 2 files + 10 organized categories
- Clear purpose for each folder
- Easy navigation with README.md files
- Nothing lost (all archived, not deleted)

**Documentation:** See [REORGANIZATION_SUMMARY.md](../REORGANIZATION_SUMMARY.md)

---

## Template for New Q&A

```markdown
### [Session X: YYYY-MM-DD] - Topic Title

**Q: Your question here**

**A:** Detailed answer with:
- Key points
- Implementation details
- Code examples if relevant
- References to documentation

**Documentation:** Links to related docs
**Status:** Open/Resolved/Implemented
```

---

## Categories of Questions

### Technical Implementation
- How to implement specific features
- Code architecture questions
- Integration challenges
- Performance optimization

### Algorithms & Mathematics
- Algorithm complexity analysis
- Mathematical formulations
- Optimization strategies
- Constraint satisfaction approaches

### Architecture & Design
- System design decisions
- Component interactions
- Data flow questions
- Integration patterns

### Debugging & Troubleshooting
- Error investigation
- Performance issues
- Configuration problems
- Unexpected behavior

---

## Quick Links

- [Index](../INDEX.md)
- [Architecture](../03-architecture/)
- [Algorithms](../04-algorithms/)
- [Performance](../05-performance/)
- [Development](../06-development/)

---

**Ask anything!** AI agents will document conversations here for future reference.

---

## [Session 3: November 17, 2025] - Metaheuristics, Hyperheuristics, and NSGA-II + RL Architecture

### Q1: What is Metaheuristics vs Hyperheuristics in my project? Am I using both?

**A:** YES, you are using **BOTH** - your project is a **hyperheuristic system built on top of a metaheuristic algorithm**.

#### Metaheuristics (NSGA-II)
**Definition**: High-level problem-independent algorithmic frameworks for solving optimization problems.

**In Your Project:**
- **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) is your **base metaheuristic**
- **What it does**: Evolves a population of timetable solutions using genetic operators (selection, crossover, mutation)
- **Multi-objective**: Minimizes both hard constraints (must-satisfy) AND soft constraints (prefer-satisfy)
- **Location**: `src/core/ga_scheduler.py` - the GAScheduler class implements NSGA-II
- **Fitness weights**: `(-1.0, -1.0)` means **both objectives are equally important** in terms of *direction* (minimize both)

```python
# From src/core/ga_scheduler.py and src/ga/creator_registry.py
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
# Objective 1: Hard constraint violations (minimize)
# Objective 2: Soft constraint penalties (minimize)
```

#### Hyperheuristics (RL Controller + Heuristic Toolbox)
**Definition**: High-level strategies that **select or generate low-level heuristics** to apply during search. Instead of applying operators directly, a hyperheuristic **decides WHICH operator to use WHEN**.

**In Your Project:**
- **RL Agent** (PPO) is your **hyperheuristic controller**
- **19 Heuristic Operators** in the toolbox are your **low-level heuristics**
- **What it does**: The RL agent *learns* to select the best heuristic operator for the current population state
- **Location**: 
  - Hyperheuristic controller: `src/rl/hybrid/hybrid_controller.py`
  - RL integration: `src/core/ga_scheduler.py` → `_init_rl()` and `_apply_rl_operators()`
  - Heuristic toolbox: Phase 1.5 implementation (19 operators across 5 categories)

**Architecture:**
```
┌─────────────────────────────────────────────────┐
│         HYPERHEURISTIC LAYER (RL)              │
│  - Observes population state (21 features)     │
│  - Selects heuristic operator (action)         │
│  - Learns from fitness improvements (reward)   │
└────────────────┬────────────────────────────────┘
                 │ selects
                 ▼
┌─────────────────────────────────────────────────┐
│      HEURISTIC TOOLBOX (19 operators)          │
│  - Construction: largest_degree_first, ...     │
│  - Perturbation: temporal_shift, room_shuffle  │
│  - Improvement: kempe_chain, ejection_chain    │
│  - Diversity: crowding_mutation, niching       │
│  - Meta: variable_neighborhood_descent, ...    │
└────────────────┬────────────────────────────────┘
                 │ modifies
                 ▼
┌─────────────────────────────────────────────────┐
│       METAHEURISTIC LAYER (NSGA-II)            │
│  - Population evolution (selection)            │
│  - Genetic operators (crossover, mutation)     │
│  - Pareto-based sorting and elitism            │
└─────────────────────────────────────────────────┘
```

**Key Distinction:**
- **Metaheuristic (NSGA-II)**: *How* to search (population-based evolution)
- **Hyperheuristic (RL)**: *Which heuristics* to apply during search (adaptive operator selection)

---

### Q2: Is Round-Robin Heuristic Selector a Hyperheuristic?

**A:** YES, round-robin is a **simple hyperheuristic strategy** (but not a learning one).

**Hyperheuristic Strategies in Your Project:**

| Strategy | Type | Description | Location |
|----------|------|-------------|----------|
| **RL_PRIMARY** | Learning Hyperheuristic | RL agent selects heuristics based on learned policy | `hybrid_controller.py` |
| **ROUND_ROBIN** | Non-learning Hyperheuristic | Cycle through heuristics in fixed order | `hybrid_controller.py` (fallback) |
| **GREEDY** | Non-learning Hyperheuristic | Always select heuristic that improved most recently | `hybrid_controller.py` (fallback) |
| **RANDOM** | Non-learning Hyperheuristic | Randomly select heuristics | `hybrid_controller.py` (fallback) |

**Round-Robin Example:**
```python
# Generation 1: Apply temporal_shift
# Generation 2: Apply room_shuffle
# Generation 3: Apply kempe_chain
# Generation 4: Apply temporal_shift (cycle back)
# ...
```

**Key Point:** Round-robin is a **deterministic hyperheuristic** (no learning), while your RL agent is an **adaptive hyperheuristic** (learns which heuristics work best for different population states).

---

### Q3: Heuristic Operators - Do They Work on Population, Chromosome, or Gene?

**A:** **It depends on the operator category!** Here's the breakdown:

#### Operator Scope by Category

| Category | Scope | Input | Output | Examples |
|----------|-------|-------|--------|----------|
| **Construction** | Individual (Chromosome) | SchedulingContext | New Individual | `largest_degree_first`, `most_constrained_first` |
| **Perturbation** | Individual (Chromosome) | Individual + Context | Modified Individual | `temporal_shift`, `room_shuffle`, `instructor_reassign` |
| **Improvement** | Individual (Chromosome) | Individual + Context | Modified Individual | `kempe_chain`, `ejection_chain`, `variable_depth_search` |
| **Diversity** | Population OR Individual | Population/Individual | Modified Population/Individual | `crowding_mutation`, `niching_selection` |
| **Meta** | Individual (Chromosome) | Individual + Context | Modified Individual | `iterated_local_search`, `guided_local_search` |

#### Detailed Breakdown

**1. Gene-Level Operations** (SessionGene)
- **What**: Modify individual genes (course-group-instructor-room-time assignments)
- **Where**: Inside perturbation/improvement operators
- **Example**: `temporal_shift` shifts a gene's time quantum forward/backward

```python
# Gene structure (from src/ga/sessiongene.py)
SessionGene:
  - course_code
  - group_name
  - instructor_name
  - room_name
  - start_quantum
  - num_quanta
  - session_type
```

**2. Chromosome-Level Operations** (Individual)
- **What**: Modify entire individuals (list of SessionGenes)
- **Where**: Most heuristic operators
- **Example**: `kempe_chain` swaps time slots across multiple genes to resolve conflicts

```python
# Individual = list of SessionGenes
individual = [
    SessionGene(course='CS101', group='A', time=0, ...),
    SessionGene(course='CS101', group='A', time=1, ...),
    SessionGene(course='CS201', group='B', time=2, ...),
    # ... all sessions
]
```

**3. Population-Level Operations** (Population)
- **What**: Modify entire population (list of Individuals)
- **Where**: Diversity operators (niching, crowding)
- **Example**: `niching_selection` selects diverse individuals across the population

```python
# Population = list of Individuals
population = [individual1, individual2, ..., individual_N]
```

#### IGLS Repair System (All Levels)

**Phase 1.2 IGLS** (Intensive Global Local Search) operates at **ALL levels**:

| Tier | Scope | Description |
|------|-------|-------------|
| **Tier 1: Exhaustive** | Population → Individual → Gene | Steepest descent on top 10% of population |
| **Tier 2: Greedy Stagnation** | Population → Individual → Gene | Greedy search on top 40% when stagnant |
| **Tier 3: Selective Probabilistic** | Individual → Gene | Post-mutation/crossover repair (30% probability) |

**Location**: `src/ga/operators/intensive_local_search.py`

---

### Q4: Why Not Use RL for Constraint Weight Adjustment?

**A:** **EXCELLENT QUESTION!** This is a **valid alternative architecture** that you're currently **NOT using**. Here's why, and whether you should consider it:

#### Current Architecture: Hardcoded Weights

**What You Have:**
```yaml
# configs/base.yaml
hard_constraints:
  student_group_exclusivity:
    weight: 3.0  # ← HARDCODED
  instructor_exclusivity:
    weight: 3.0
  instructor_qualifications:
    weight: 3.0
  room_exclusivity:
    weight: 3.0
  # ...

soft_constraints:
  student_schedule_compactness:
    weight: 1.5  # ← HARDCODED
  instructor_schedule_compactness:
    weight: 1.0
  # ...
  soft_weight_factor: 0.01  # ← Global scaling factor
```

**Fitness Calculation:**
```python
# Hard penalties (all violations treated equally within category)
hard_penalty = sum(weight * violations for constraint, violations in hard_details.items())

# Soft penalties (scaled down by soft_weight_factor)
soft_penalty = soft_weight_factor * sum(weight * violations for constraint, violations in soft_details.items())

# Final fitness
fitness = (hard_penalty, soft_penalty)  # Both minimized with weights (-1.0, -1.0)
```

**Key Point:** Your constraint weights are **domain-designed constants**, NOT adaptive parameters.

---

#### Alternative Architecture: RL Constraint Weight Tuning

**What It Would Look Like:**

```python
# RL agent learns constraint weights
state = [current_fitness, stagnation_counter, diversity, ...]
action = rl_agent.predict(state)  # Outputs weight adjustments

# Apply weight adjustments
hard_constraints['instructor_exclusivity']['weight'] *= action[0]
soft_constraints['compactness']['weight'] *= action[1]
# ...

# Evaluate with new weights
fitness = evaluate_with_weights(individual, constraint_weights)
reward = improvement_in_fitness
```

**Is This Still a Hyperheuristic?**
- **Technically YES** - but it's a **parameter tuning hyperheuristic** rather than an **operator selection hyperheuristic**
- **Classification**: **Hyper-heuristic with online parameter control**
- **Example**: Adaptive Cuckoo Search (ACS) tunes step size parameters during search

---

#### Comparison: Operator Selection (Current) vs Weight Tuning (Alternative)

| Aspect | Operator Selection (Your Project) | Constraint Weight Tuning |
|--------|-----------------------------------|--------------------------|
| **RL Action Space** | 19 discrete actions (heuristic operators) | Continuous (weight multipliers) |
| **What RL Learns** | WHICH operator to apply WHEN | HOW MUCH to penalize each constraint |
| **Complexity** | High (19-dim action space) | Medium (N-dim continuous, N=constraints) |
| **Interpretability** | High (see which heuristics work) | Low (weights are abstract) |
| **Domain Knowledge** | Encoded in heuristic implementations | Encoded in initial weight ranges |
| **Search Strategy** | Structural changes to solutions | Fitness landscape reshaping |
| **Generalization** | Learns operator patterns | Learns constraint priorities |

---

#### Should You Use RL for Weight Tuning?

**Arguments FOR:**
1. **Automatic Tuning**: No need to manually design weights
2. **Adaptive Priorities**: Weights change based on search phase (early vs late)
3. **Instance-Specific**: Different problems may need different weights
4. **Synergy with NSGA-II**: Reshape fitness landscape to guide Pareto search

**Arguments AGAINST (Why You Probably Shouldn't):**
1. **Domain Knowledge Loss**: Your current weights encode **real-world importance** (e.g., "no instructor conflicts" is critical)
2. **Instability**: Changing weights mid-search can disrupt Pareto fronts
3. **Overfitting Risk**: RL might tune weights for training data but fail on new problems
4. **Interpretability**: Hard to explain why RL chose certain weights (black box)
5. **Computational Cost**: Evaluating new weights requires re-evaluating entire population
6. **NSGA-II Design**: Multi-objective algorithms are meant to handle trade-offs WITHOUT weight tuning

**Recommendation:** **STICK WITH YOUR CURRENT APPROACH** (operator selection) for these reasons:
- Your constraint weights reflect **domain expertise** (e.g., no instructor conflicts > schedule compactness)
- NSGA-II already handles multi-objective trade-offs via Pareto sorting
- Operator selection is more **interpretable and generalizable**
- Weight tuning risks **losing domain knowledge** encoded in constraints

**When Weight Tuning WOULD Make Sense:**
- Single-objective optimization (not NSGA-II)
- Unknown or ambiguous constraint priorities
- Need for instance-specific adaptation
- Sufficient training data to avoid overfitting

---

### Q5: NSGA-II Fitness Weights: `(-1.0, -1.0)` - Do They Matter?

**A:** **GREAT QUESTION!** The answer is nuanced:

#### The Weights DO NOT Directly Control Priority

**What `(-1.0, -1.0)` Means:**
```python
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
# weights[0] = -1.0 → MINIMIZE objective 1 (hard constraint violations)
# weights[1] = -1.0 → MINIMIZE objective 2 (soft constraint penalties)
```

**Key Point:** In NSGA-II, these weights **ONLY control direction** (minimize vs maximize), NOT relative importance!

#### How NSGA-II Handles Multi-Objective Trade-offs

**Pareto Dominance (Not Weighted Sum):**
```python
# Individual A dominates B if:
# 1. A is better in at least one objective
# 2. A is not worse in any objective

# Example:
A = (hard=10, soft=5)
B = (hard=15, soft=3)
# Result: NEITHER dominates (A better in hard, B better in soft)
# Both go on Pareto front (rank 1)

C = (hard=20, soft=10)
# Result: Both A and B dominate C (C goes to rank 2)
```

**Non-dominated Sorting:**
```python
# NSGA-II sorts population into Pareto fronts
Rank 1: [A, B]        # Non-dominated (Pareto front)
Rank 2: [C, D, E]     # Dominated by rank 1
Rank 3: [F, G]        # Dominated by rank 2
# ...

# Selection: Pick all of rank 1, then rank 2, etc. until population full
# If rank partially fills, use crowding distance (diversity)
```

**Crowding Distance (Diversity Metric):**
- When two solutions have same rank, prefer more isolated one
- Maintains diversity along Pareto front
- **NOT influenced by objective weights**

---

#### So What DO the Weights Control?

**1. Direction (Minimize vs Maximize):**
```python
weights = (-1.0, -1.0)  # Minimize both (your case)
weights = (1.0, -1.0)   # Maximize first, minimize second
weights = (1.0, 1.0)    # Maximize both
```

**2. Magnitude (Theoretically Irrelevant in Pure NSGA-II):**
```python
weights = (-1.0, -1.0)    # Same Pareto fronts as...
weights = (-100.0, -1.0)  # ...this (NSGA-II doesn't care about magnitude)
```

**WHY?** NSGA-II uses **dominance comparison**, not weighted sum:
```python
# NOT used in NSGA-II:
fitness_score = w1 * obj1 + w2 * obj2  # ← Weighted sum (NOT NSGA-II)

# ACTUALLY used in NSGA-II:
dominates = (obj1_a <= obj1_b) and (obj2_a <= obj2_b) and (at least one strict <)
```

---

#### How to Prioritize Hard Constraints Over Soft?

**You're ALREADY doing it correctly!** Here's how:

**1. Constraint Weight Design (Domain-Level):**
```yaml
# High weights for critical constraints
hard_constraints:
  instructor_exclusivity:
    weight: 3.0  # ← Critical (no instructor conflicts)
  
  course_completeness:
    weight: 2.0  # ← Important (all sessions scheduled)

# Lower weights for preferences
soft_constraints:
  compactness:
    weight: 1.5
  
  soft_weight_factor: 0.01  # ← Global scaling (soft << hard)
```

**2. Soft Weight Factor (Scaling):**
```python
# From constraint evaluation
hard_total = sum(weight * violations for constraint in hard_constraints)
soft_total = soft_weight_factor * sum(weight * violations for constraint in soft_constraints)

# Example values:
# hard_total = 15.0 (3 instructor conflicts × weight 3.0 + ...)
# soft_total = 0.5  (50 compactness violations × weight 1.0 × 0.01)

fitness = (hard_total, soft_total)  # (15.0, 0.5)
```

**3. Lexicographic Comparison (Implicit in Pareto Sorting):**
```python
# When comparing solutions, hard constraints dominate
A = (hard=10, soft=100)  # 10 hard violations, 100 soft
B = (hard=15, soft=1)    # 15 hard violations, 1 soft

# Result: A dominates B (better in hard, worse in soft, but hard matters more)
# Because soft_total is ALWAYS scaled to be << hard_total
```

**4. NSGA-II Selection Behavior:**
- Solutions with **hard=0** (feasible) will ALWAYS dominate solutions with **hard>0** (infeasible)
- Among feasible solutions, NSGA-II optimizes soft constraints
- Among infeasible solutions, NSGA-II minimizes hard violations first

**Example Pareto Fronts:**
```python
Rank 1: [
    (hard=0, soft=5),   # Feasible, low soft
    (hard=0, soft=10),  # Feasible, high soft
]

Rank 2: [
    (hard=2, soft=1),   # Infeasible (dominated by all rank 1)
]
```

---

#### Could You Change the Weights for Different Priorities?

**Short Answer:** **NO** (for NSGA-II) - weights only control direction, not priority.

**If You Want to Change Priorities:**

**Option 1: Adjust Constraint Weights (Recommended):**
```yaml
# Make instructor conflicts MORE important
hard_constraints:
  instructor_exclusivity:
    weight: 10.0  # ← Increase from 3.0

# Make compactness LESS important
soft_constraints:
  compactness:
    weight: 0.5   # ← Decrease from 1.5
```

**Option 2: Adjust Soft Weight Factor:**
```yaml
soft_constraints:
  soft_weight_factor: 0.001  # ← Decrease from 0.01 (soft even less important)
```

**Option 3: Switch to Weighted Sum (NOT Recommended):**
```python
# Single-objective (loses Pareto benefits)
fitness = -10.0 * hard_total + -0.001 * soft_total
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
```

---

#### Verification: Are You Conceptually Correct on NSGA-II?

**YES!** Your understanding is correct:

✅ **NSGA-II is a multi-objective algorithm** (not weighted sum)
✅ **Weights `(-1.0, -1.0)` control direction** (both minimize)
✅ **Magnitude of weights doesn't affect Pareto sorting**
✅ **Priority is controlled by constraint weights**, not DEAP fitness weights
✅ **Hard constraints naturally dominate** via scaling (soft_weight_factor)
✅ **NSGA-II finds Pareto-optimal trade-offs** between objectives

**Key Insight:** Your architecture is **well-designed**:
- Hard constraints have high absolute values (3.0, 2.5, 2.0)
- Soft constraints are globally scaled down (×0.01)
- NSGA-II naturally prioritizes hard over soft
- No need to change DEAP fitness weights

---

### Q6: Are You Conceptually Correct on NSGA-II + RL Integration?

**A:** **YES, MOSTLY CORRECT!** Let's verify both individually and integrated:

#### Individual Components

**✅ NSGA-II (Metaheuristic):**
- Multi-objective optimization (hard + soft constraints)
- Pareto-based selection (non-dominated sorting)
- Crowding distance for diversity
- Elitism preserves best solutions
- **Your implementation**: GAScheduler in `src/core/ga_scheduler.py`

**✅ RL (Hyperheuristic):**
- Learns to select heuristic operators
- State: 21 features (fitness, diversity, stagnation, etc.)
- Action: 19 heuristic operators
- Reward: Fitness improvement from heuristic application
- **Your implementation**: PPO agent with curriculum training

---

#### Integrated Architecture

**How They Work Together:**

```python
# Each GA generation:
for gen in range(num_generations):
    # 1. NSGA-II: Standard genetic operators
    offspring = select(population)
    offspring = crossover(offspring)
    offspring = mutate(offspring)
    
    # 2. RL: Adaptive heuristic application
    state = encode_state(population, gen)  # 21 features
    action = rl_agent.select_action(state)  # Pick heuristic
    offspring = apply_heuristic(action, offspring)  # Apply to population
    
    # 3. NSGA-II: Evaluate and select next generation
    evaluate(offspring)
    population = nsga2_select(population + offspring)
```

**Key Points:**
1. **RL augments NSGA-II** (doesn't replace it)
2. **RL is stateless across generations** (no memory beyond observation history)
3. **Heuristics modify individuals** (not population structure)
4. **NSGA-II handles selection** (RL only suggests modifications)

---

#### Potential Improvements

**How to Better Utilize NSGA-II + RL Synergy:**

**1. Multi-Objective Reward Function:**
```python
# Current reward: weighted sum
reward = fitness_improvement  # Scalar

# Better: Pareto-aware reward
if new_solution_dominates_old:
    reward = +1.0 + hypervolume_improvement
elif old_solution_dominates_new:
    reward = -1.0
else:
    reward = 0.0 + diversity_bonus
```

**2. Pareto Front Targeting:**
```python
# Apply different heuristics to different Pareto ranks
if solution in pareto_front_rank1:
    # Exploitation: refine already-good solutions
    action = rl_agent_exploit.select_action(state)
else:
    # Exploration: diversify poor solutions
    action = rl_agent_explore.select_action(state)
```

**3. Multi-Agent RL (Specialist Agents):**
```python
# Specialist agents for different solution regions
feasible_agent = PPO(...)    # Optimizes soft constraints (hard=0)
infeasible_agent = PPO(...)  # Reduces hard violations (hard>0)

if solution.hard_violations == 0:
    action = feasible_agent.select_action(state)
else:
    action = infeasible_agent.select_action(state)
```

**4. Hierarchical RL:**
```python
# High-level agent: Select heuristic category
category = high_level_agent.select_action(state)  # construction/perturbation/improvement

# Low-level agent: Select specific heuristic within category
heuristic = low_level_agents[category].select_action(state)
```

**5. Constraint-Aware State Representation:**
```python
# Current state: 21 global features
state = [hard_total, soft_total, diversity, ...]

# Better: Per-constraint breakdown
state = [
    instructor_conflicts,      # Specific violation counts
    room_conflicts,
    availability_violations,
    compactness_penalty,
    # ... all constraints
    diversity,
    stagnation,
    generation_progress
]
```

**6. Archive-Based Learning:**
```python
# Maintain archive of best solutions found
archive = ParetoArchive()

# RL reward includes archive novelty
reward = fitness_improvement + novelty_bonus(solution, archive)

# Use archive solutions as crossover partners (RL suggests which)
partner = rl_agent.select_archive_solution(archive, current_solution)
offspring = crossover(current_solution, partner)
```

**7. Curriculum Learning Enhancement:**
```python
# Current curriculum: 3 stages by problem size (easy/medium/hard)

# Better: Multi-dimensional curriculum
curriculum_stages = [
    {"hard_weight": 1.0, "soft_weight": 0.0},  # Focus on feasibility first
    {"hard_weight": 1.0, "soft_weight": 0.5},  # Balanced
    {"hard_weight": 1.0, "soft_weight": 1.0},  # Full optimization
]
```

**8. Adaptive Operator Probabilities (RL-Controlled):**
```python
# Instead of fixed crossover_prob, mutation_prob
# Let RL adjust them
state = encode_state(population)
adjustments = rl_agent.select_action(state)

crossover_prob = base_cxpb * adjustments[0]
mutation_prob = base_mutpb * adjustments[1]
```

---

#### Architecture Suggestions

**Option 1: Memetic NSGA-II with RL Local Search (Recommended)**
```python
# After NSGA-II selection, apply RL-selected local search to elite
elite = select_best(population, top_k=10)
for individual in elite:
    state = encode_individual(individual)
    local_search_heuristic = rl_agent.select_action(state)
    improved = apply_heuristic(local_search_heuristic, individual)
    if improved.fitness dominates individual.fitness:
        replace(individual, improved)
```

**Option 2: RL-Guided Initialization**
```python
# Use RL to generate initial population intelligently
population = []
for i in range(pop_size):
    individual = empty_individual()
    while not individual.is_complete():
        state = encode_partial_individual(individual)
        construction_heuristic = rl_agent.select_action(state)
        individual = apply_construction(construction_heuristic, individual)
    population.append(individual)
```

**Option 3: Bandit-Based Heuristic Selection**
```python
# Multi-Armed Bandit (simpler than full RL)
from src.rl.bandits import ThompsonSampling

bandit = ThompsonSampling(n_arms=19)  # 19 heuristics

for gen in range(num_generations):
    # Select heuristic based on historical performance
    heuristic_id = bandit.select_arm()
    
    # Apply heuristic
    improvement = apply_heuristic(heuristic_id, population)
    
    # Update bandit
    bandit.update(heuristic_id, reward=improvement)
```

---

#### Summary: Recommendations

**Current Architecture (Good):**
✅ NSGA-II + RL operator selection
✅ 19 heuristic operators in toolbox
✅ Curriculum learning (3 stages)
✅ Pareto-based multi-objective optimization
✅ Hardcoded constraint weights (domain knowledge)

**Suggested Enhancements (Better):**
1. **Multi-objective reward** (Pareto-aware, not scalar)
2. **Specialist agents** (feasible vs infeasible)
3. **Constraint-specific state** (per-constraint breakdown)
4. **Archive-based diversity** (novelty search)
5. **Memetic RL** (local search on elite)
6. **Adaptive probabilities** (RL controls crossover/mutation rates)

**Advanced Ideas (Best):**
7. **Multi-agent RL** (specialist for each Pareto rank)
8. **Hierarchical RL** (category selection → heuristic selection)
9. **Transfer learning** (pre-train on synthetic problems)
10. **Online learning** (adapt from production runs)

---

### Q7: Summary of Key Concepts

| Concept | Definition | Your Implementation | Level |
|---------|------------|---------------------|-------|
| **Metaheuristic** | High-level search framework | NSGA-II (GAScheduler) | Base algorithm |
| **Hyperheuristic** | Selects/generates heuristics | RL agent (PPO) + Toolbox | Control layer |
| **Multi-objective** | Optimizes multiple objectives | Hard + Soft constraints | NSGA-II feature |
| **Pareto dominance** | Comparison without weights | Non-dominated sorting | NSGA-II selection |
| **Constraint weights** | Domain importance | YAML config | Problem-specific |
| **DEAP fitness weights** | Direction (min/max) | `(-1.0, -1.0)` | Algorithm config |
| **RL action space** | Heuristic operators | 19 operators | Hyperheuristic toolbox |
| **RL state space** | Population features | 21 features | Observation |
| **RL reward** | Fitness improvement | Scalar (can improve) | Learning signal |

---

**Status**: All questions answered with architectural analysis and improvement suggestions.
**Documentation**: See `docs/06-development/implementation-notes/PHASE_1.5_SUMMARY.md` and `PHASE_2_RL_COMPLETE.md` for implementation details.
**Next Steps**: Run curriculum training, evaluate RL vs baselines, implement suggested enhancements if needed
