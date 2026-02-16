# Algorithm Reference: schedule-engine

> **Comprehensive Technical Documentation**  
> This document explains all optimization algorithms implemented in schedule-engine,
> their mathematical foundations, implementation details, and practical usage.

---

## Table of Contents

1. [Algorithm Hierarchy Overview](#algorithm-hierarchy-overview)
2. [Greedy Local Search (Hill Climbing)](#1-greedy-local-search-hill-climbing)
3. [Exhaustive Local Search (Steepest Descent)](#2-exhaustive-local-search-steepest-descent)
4. [Repair Operators](#3-repair-operators)
5. [Large Neighborhood Search (LNS)](#4-large-neighborhood-search-lns)
6. [Variable Neighborhood Descent (VND)](#5-variable-neighborhood-descent-vnd)
7. [Adaptive Large Neighborhood Search (ALNS)](#6-adaptive-large-neighborhood-search-alns)
8. [Iterated Local Search (ILS)](#7-iterated-local-search-ils)
9. [Guided Local Search (GLS)](#8-guided-local-search-gls)
10. [Construction Heuristics](#9-construction-heuristics)
11. [Improvement Heuristics](#10-improvement-heuristics-vnd-building-blocks)
12. [Algorithm Integration in GA Modes](#algorithm-integration-in-ga-modes)
13. [Performance Comparison](#performance-comparison)
14. [Suggestions & Recommendations](#suggestions--recommendations)

---

## Algorithm Hierarchy Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    META-HEURISTICS (Strategy Layer)                         │
│                    Orchestrate multiple operators                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │   ALNS   │  │   VND    │  │   ILS    │  │   GLS    │  │  LNS-    │      │
│  │ Adaptive │  │ Variable │  │ Iterated │  │ Guided   │  │  IGLS    │      │
│  │   LNS    │  │ Neighbor │  │  Local   │  │  Local   │  │          │      │
│  │          │  │ Descent  │  │  Search  │  │  Search  │  │          │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │             │             │             │             │             │
│       └─────────────┴──────┬──────┴─────────────┴─────────────┘             │
│                            │                                                 │
│                            ▼                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                    BUILDING BLOCKS (Operator Layer)                         │
│                    Individual search/repair moves                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐│
│  │   Greedy   │  │ Exhaustive │  │  Kempe     │  │    Repair Operators    ││
│  │   Local    │  │   Local    │  │  Chains    │  │ ┌──────────────────┐   ││
│  │  Search    │  │  Search    │  │            │  │ │ HC1-HC8 Fixers   │   ││
│  │            │  │            │  ├────────────┤  │ │ Priority-ordered │   ││
│  │ (First     │  │ (Best      │  │ Ejection   │  │ │ Constraint-spec. │   ││
│  │ Improve)   │  │ Improve)   │  │  Chains    │  │ └──────────────────┘   ││
│  └────────────┘  └────────────┘  ├────────────┤  └────────────────────────┘│
│                                  │ Variable   │                             │
│                                  │   Depth    │                             │
│                                  │  Search    │                             │
│                                  └────────────┘                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                   CONSTRUCTION HEURISTICS (Initialization)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐  │
│  │ Largest Degree     │  │ Most Constrained    │  │ Earliest Deadline   │  │
│  │     First          │  │      First          │  │      First          │  │
│  │ (Graph coloring)   │  │ (CSP MRV heuristic) │  │ (Scheduling order)  │  │
│  └────────────────────┘  └─────────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Greedy Local Search (Hill Climbing)

### Definition

**Greedy Local Search** (also known as **Hill Climbing** or **First Improvement**) is a simple 
optimization algorithm that iteratively makes small changes to a solution and immediately accepts 
the first change that improves fitness.

### Mathematical Formulation

Given:
- Current solution: $x$
- Neighborhood function: $N(x) = \{x' : x' \text{ is obtainable from } x \text{ by small change}\}$
- Objective function: $f(x)$ (minimization)

Algorithm:
$$
x_{t+1} = \text{first } x' \in N(x_t) \text{ such that } f(x') < f(x_t)
$$

### Implementation

**Location:** [src/schedule_engine/ga/operators/local_search.py](../src/schedule_engine/ga/operators/local_search.py#L41)

```python
def optimize_gene_greedy(
    gene: SessionGene,
    individual: list[SessionGene],
    gene_index: int,
    context: SchedulingContext,
    max_iterations: int = 10,
) -> tuple[SessionGene, int]:
    """
    Greedy local search: hill climbing with first improvement acceptance.
    
    Strategy:
        1. Evaluate current gene's constraint violations
        2. Generate random neighborhood samples
        3. Accept FIRST gene that improves fitness
        4. Repeat until no improvement or max_iterations reached
    """
```

### Algorithm Pseudocode

```
function GREEDY_LOCAL_SEARCH(gene, individual, max_iterations):
    current ← gene
    current_violations ← count_violations(current)
    
    for i = 1 to max_iterations:
        neighbors ← generate_neighborhood(current, max_samples=20)
        shuffle(neighbors)  # Random exploration order
        
        improved ← false
        for neighbor in neighbors:
            if violations(neighbor) < current_violations:
                current ← neighbor              # Accept FIRST improvement
                current_violations ← violations(neighbor)
                improved ← true
                break                           # Stop searching this iteration
        
        if not improved:
            break  # Local optimum reached
    
    return current
```

### Characteristics

| Property | Value |
|----------|-------|
| **Time Complexity** | $O(k \cdot n)$ where $k$ = iterations, $n$ = neighborhood size |
| **Evaluations per Gene** | 10-20 (fast) |
| **Guarantee** | Finds local optimum, NOT global |
| **Termination** | First improvement or no improvement in iteration |

### When to Use

 **Good for:**
- Stagnation repair (quick fix attempts)
- Large populations (must be fast)
- Early generations (exploration phase)

 **Bad for:**
- Final optimization (may miss better neighbors)
- Small populations (need thorough search)

---

## 2. Exhaustive Local Search (Steepest Descent)

### Definition

**Exhaustive Local Search** (also known as **Steepest Descent** or **Best Improvement**) evaluates
ALL neighbors before selecting the single best one. More thorough but computationally expensive.

### Mathematical Formulation

$$
x_{t+1} = \arg\min_{x' \in N(x_t)} f(x')
$$

Accept move only if $f(x_{t+1}) < f(x_t)$.

### Implementation

**Location:** [src/schedule_engine/ga/operators/local_search.py](../src/schedule_engine/ga/operators/local_search.py#L102)

```python
def optimize_gene_exhaustive(
    gene: SessionGene,
    individual: list[SessionGene],
    gene_index: int,
    context: SchedulingContext,
    max_neighborhood_size: int = 100,
) -> tuple[SessionGene, int]:
    """
    Steepest descent: exhaustive neighborhood search with best improvement.
    
    Strategy:
        1. Evaluate current gene's constraint violations
        2. Generate full neighborhood (up to max_neighborhood_size)
        3. Evaluate ALL neighbors
        4. Select BEST neighbor (steepest descent)
        5. Repeat until local optimum reached
    """
```

### Algorithm Pseudocode

```
function EXHAUSTIVE_LOCAL_SEARCH(gene, individual, max_neighborhood_size):
    current ← gene
    current_violations ← count_violations(current)
    no_improvement_streak ← 0
    
    while no_improvement_streak < 3:
        neighbors ← generate_neighborhood(current, max_neighborhood_size)
        
        best_neighbor ← null
        best_violations ← current_violations
        
        for neighbor in neighbors:                  # Evaluate ALL
            v ← violations(neighbor)
            if v < best_violations:
                best_neighbor ← neighbor
                best_violations ← v
        
        if best_neighbor ≠ null:
            current ← best_neighbor
            current_violations ← best_violations
            no_improvement_streak ← 0
        else:
            no_improvement_streak += 1              # Allow 3 tries
    
    return current
```

### Characteristics

| Property | Value |
|----------|-------|
| **Time Complexity** | $O(n^2)$ per iteration (all neighbors evaluated) |
| **Evaluations per Gene** | 50-200 (thorough) |
| **Guarantee** | Finds best local move, still local optimum |
| **Termination** | 3 consecutive iterations without improvement |

### When to Use

 **Good for:**
- Fixed generations (3, 25) for intensive optimization
- Elite individuals (worth the extra cost)
- Final polishing phase

 **Bad for:**
- Every generation (too slow)
- Early exploration (greedy is enough)

---

## 3. Repair Operators

### Definition

**Repair Operators** are targeted constraint-fixing procedures. Unlike general local search, each
repair operator knows exactly how to fix ONE specific constraint type.

### Implementation

**Location:** [src/schedule_engine/ga/repair/basic.py](../src/schedule_engine/ga/repair/basic.py)

### Repair Operator Registry

```
┌──────┬─────────────────────────────────┬──────────────────────────────────┬──────────┐
│ Code │ Constraint Name                 │ Repair Operator                  │ Priority │
├──────┼─────────────────────────────────┼──────────────────────────────────┼──────────┤
│ HC1  │ student_group_exclusivity       │ repair_group_overlaps            │    2     │
│ HC2  │ instructor_exclusivity          │ repair_instructor_conflicts      │    5     │
│ HC3  │ instructor_qualifications       │ repair_instructor_qualifications │    6     │
│ HC4  │ room_suitability                │ repair_room_type_mismatches      │    7     │
│ HC5  │ instructor_time_availability    │ repair_instructor_availability   │    1     │
│ HC6  │ room_time_availability          │ NO REPAIR (always available)     │    -     │
│ HC7  │ course_completeness             │ NO REPAIR (structural)           │    -     │
│ HC8  │ room_exclusivity                │ repair_room_overlap_reassign     │   3,4    │
│      │                                 │ repair_room_conflicts            │          │
├──────┼─────────────────────────────────┼──────────────────────────────────┼──────────┤
│ SC4  │ session_continuity              │ repair_session_clustering        │    8     │
└──────┴─────────────────────────────────┴──────────────────────────────────┴──────────┘
```

### Repair Strategies by Operator

#### `repair_instructor_availability` (Priority 1)
```
Problem:  Part-time instructor scheduled when unavailable
Solution: Shift session to time when instructor IS available

Algorithm:
1. Check if instructor is full-time (skip if yes)
2. For each quantum in session:
   - Check if quantum ∈ instructor.available_quanta
   - If not, find new start_quanta where ALL session quanta are available
3. Verify new time has no conflicts (room, group, instructor)
```

#### `repair_group_overlaps` (Priority 2)
```
Problem:  Two sessions with overlapping student groups at same time
Solution: Shift one session to a different time slot

Algorithm:
1. Build occupancy map: {group_id: {quanta: [gene_indices]}}
2. Find conflicts: quanta where count > 1
3. For each conflict:
   - Select gene to move (smaller session first)
   - Find empty time slot on same day
   - Fall back to any available day if needed
```

#### `repair_room_conflicts` (Priority 4)
```
Problem:  Two sessions assigned to same room at same time
Solution: Shift one session's time slot

Algorithm:
1. Build room occupancy: {room_id: {quanta: [gene_indices]}}
2. Identify conflicts
3. Try to shift session to different time (same room)
4. Prefer minimal shift distance (same day if possible)
```

#### `repair_room_overlap_reassign` (Priority 3)
```
Problem:  Two sessions need same room at same time  
Solution: Assign different room (keep time)

Algorithm:
1. Detect room conflicts
2. For conflicting session:
   - Find alternative room with matching type (lab/classroom)
   - Verify room capacity ≥ group size
   - Verify room is unoccupied at session time
```

#### `repair_instructor_qualifications` (Priority 6)
```
Problem:  Instructor not qualified to teach assigned course
Solution: Swap to a qualified instructor

Algorithm:
1. Check instructor_id ∈ course.qualified_instructors
2. If not:
   - Get list of qualified instructors
   - Filter by availability at session time
   - Prefer instructor with fewest existing sessions
```

#### `repair_room_type_mismatches` (Priority 7)
```
Problem:  Lab course assigned to classroom (or vice versa)
Solution: Reassign to appropriate room type

Algorithm:
1. Determine required room type from course.course_type
2. Find rooms matching type AND available at time
3. Select room with appropriate capacity
```

### Unified Repair Pipeline

```python
def repair_individual_unified(individual, context, max_iterations=2, selective=True):
    """
    Apply all repair operators in priority order.
    
    Args:
        selective: If True, only check genes with known violations (3-4x faster)
    """
    for iteration in range(max_iterations):
        for repair_op in sorted(REPAIR_REGISTRY, key=lambda x: x.priority):
            fixes = repair_op(individual, context)
            total_fixes += fixes
```

---

## 4. Large Neighborhood Search (LNS)

### Definition

**Large Neighborhood Search (LNS)** is a meta-heuristic that iteratively destroys and repairs 
large portions of a solution. Unlike local search that makes small moves, LNS makes **big jumps**
in the search space.

### Mathematical Formulation

Given current solution $x$:
1. **Destroy:** $x_d = \text{destroy}(x)$ — Remove subset of solution
2. **Repair:** $x_r = \text{repair}(x_d)$ — Reconstruct removed part
3. **Accept:** $x_{t+1} = x_r$ if $f(x_r) < f(x_t)$, else $x_{t+1} = x_t$

### Implementation

**Location:** [src/schedule_engine/ga/repair/lns/operator.py](../src/schedule_engine/ga/repair/lns/operator.py)

```python
def lns_igls_repair(
    individual: list[SessionGene],
    max_subproblem_size: int = 20,    # Max sessions to destroy
    min_subproblem_size: int = 4,     # Skip if too few conflicts
    expand_hops: int = 0,             # BFS expansion in conflict graph
    igls_max_iterations: int = 500,   # Repair iterations
    igls_time_limit: float = 5.0,     # Time limit per repair
) -> list[SessionGene]:
```

### Algorithm: LNS-IGLS

```
function LNS_IGLS_REPAIR(individual):
    # PHASE 1: DETECT
    conflicted_indices ← find_hard_conflict_sessions(individual)
    if |conflicted_indices| < min_subproblem_size:
        return individual  # Not worth LNS overhead
    
    # PHASE 2: EXPAND (optional)
    if expand_hops > 0:
        conflict_graph ← build_conflict_graph(individual)
        conflicted_indices ← bfs_expand(conflicted_indices, conflict_graph, expand_hops)
    
    # PHASE 3: LIMIT SIZE
    if |conflicted_indices| > max_subproblem_size:
        conflicted_indices ← select_worst_conflicts(conflicted_indices, max_subproblem_size)
    
    # PHASE 4: DESTROY
    subproblem ← [individual[i] for i in conflicted_indices]
    fixed_schedule ← remaining sessions (not in subproblem)
    
    # PHASE 5: REPAIR (using IGLS)
    repaired_subproblem ← igls_repair(subproblem, fixed_schedule, context)
    
    # PHASE 6: REINTEGRATE
    for i, idx in enumerate(conflicted_indices):
        individual[idx] ← repaired_subproblem[i]
    
    return individual
```

### Conflict Graph Expansion

```
Initial conflicts: {A, B, C}

Conflict Graph:
    A ─── D
    │     │
    B ─── E ─── F
    │
    C ─── G

After 1-hop expansion: {A, B, C, D, E, G}
After 2-hop expansion: {A, B, C, D, E, F, G}
```

**Why expand?** Sessions adjacent to conflicts often need to move to enable conflict resolution.

### Key Insight

Traditional local search makes **small changes** → easily trapped in local optima.

LNS makes **big jumps** → can escape local optima barriers by restructuring large portions.

### Characteristics

| Property | Value |
|----------|-------|
| **Subproblem Size** | 4-20 sessions (configurable) |
| **Repair Strategy** | IGLS (Iterated Guided Local Search) |
| **Time Limit** | 5 seconds per repair (configurable) |
| **When Applied** | After mutation/crossover when violations detected |

---

## 5. Variable Neighborhood Descent (VND)

### Definition

**Variable Neighborhood Descent (VND)** systematically explores **different types** of neighborhoods.
When stuck in one neighborhood structure, it switches to a different move type.

### Mathematical Formulation

Given neighborhood structures $N_1, N_2, \ldots, N_k$:

```
k ← 1
while k ≤ K:
    x' ← LocalSearch(x, N_k)
    if f(x') < f(x):
        x ← x'
        k ← 1        # RESTART from first neighborhood
    else:
        k ← k + 1    # Try next neighborhood
```

### Implementation

**Location:** [src/schedule_engine/ga/heuristics/meta.py](../src/schedule_engine/ga/heuristics/meta.py#L43)

```python
def variable_neighborhood_descent(
    individual: list[SessionGene],
    context: SchedulingContext,
    max_neighborhoods: int = 3,
    max_iterations: int = 5,
) -> int:
    """
    Neighborhoods:
    - N1: Kempe chain moves (graph coloring swaps)
    - N2: Ejection chain moves (cascading reassignments)  
    - N3: Variable depth search (multi-move lookahead)
    """
```

### Algorithm Pseudocode

```
function VND(individual, context):
    neighborhoods ← [kempe_chain, ejection_chain, variable_depth_search]
    total_improvements ← 0
    k ← 0  # Current neighborhood index
    
    while k < |neighborhoods|:
        operator ← neighborhoods[k]
        improvements ← operator(individual, context)
        
        if improvements > 0:
            total_improvements += improvements
            k ← 0                    # Restart from N1
        else:
            k ← k + 1                # Try next neighborhood
    
    return total_improvements
```

### Why It Works

Different neighborhoods "see" different paths:

```
Solution Space Landscape:

           ▲ Fitness
           │
   N2 path │    ╱╲      N1 stuck here
           │   ╱  ╲         ↓
           │  ╱    ╲       ●
           │ ╱      ╲     /│\
           │╱        ╲   / │ \
           └──────────●───●───●──→ Solution Space
                       Global
                       Optimum
                       (reached via N2)
```

If $N_1$ (Kempe chains) gets stuck, $N_2$ (Ejection chains) might have a move that escapes.

---

## 6. Adaptive Large Neighborhood Search (ALNS)

### Definition

**ALNS** extends LNS with **learning**: it tracks which destroy/repair operators work well and
uses successful operators more frequently.

### Mathematical Formulation

Operator selection via roulette wheel:
$$
P(\text{select operator } i) = \frac{\text{score}_i}{\sum_j \text{score}_j}
$$

Score update:
$$
\text{score}_i = \text{score}_i \cdot \gamma + r_i
$$

Where:
- $\gamma = 0.95$ (decay factor to prevent stagnation)
- $r_i = 3$ (major improvement), $1$ (minor improvement), $0$ (no improvement)

### Implementation

**Location:** [src/schedule_engine/ga/heuristics/meta.py](../src/schedule_engine/ga/heuristics/meta.py#L151)

```python
def adaptive_large_neighborhood(
    individual: list[SessionGene],
    context: SchedulingContext,
    num_iterations: int = 10,
    initial_destroy_rate: float = 0.3,
) -> int:
```

### Algorithm Pseudocode

```
function ALNS(individual, context):
    # Initialize operator scores
    scores ← {
        "temporal_destroy": 1.0,
        "room_destroy": 1.0,
        "random_destroy": 1.0
    }
    destroy_rate ← 0.3
    best ← copy(individual)
    best_fitness ← fitness(best)
    
    for iteration = 1 to num_iterations:
        # SELECT destroy operator (roulette wheel)
        operator ← roulette_select(scores)
        
        # DESTROY
        destroyed_indices ← destroy(individual, destroy_rate, operator)
        
        # REPAIR
        repair(individual, destroyed_indices, context)
        
        # EVALUATE & UPDATE
        current_fitness ← fitness(individual)
        
        if current_fitness < best_fitness:
            # MAJOR improvement
            scores[operator] += 3.0
            best ← copy(individual)
            best_fitness ← current_fitness
            destroy_rate ← max(0.1, destroy_rate × 0.9)  # Reduce destruction
            
        elif current_fitness < best_fitness × 1.05:
            # MINOR improvement
            scores[operator] += 1.0
            destroy_rate ← min(0.5, destroy_rate × 1.1)
            
        else:
            # NO improvement - revert
            individual ← copy(best)
            destroy_rate ← min(0.5, destroy_rate × 1.05)
        
        # DECAY all scores
        for op in scores:
            scores[op] ← scores[op] × 0.95
    
    return count_improvements
```

### Destroy Operators

| Operator | Strategy | Good For |
|----------|----------|----------|
| `temporal_destroy` | Remove sessions in specific time range | Time-blocked conflicts |
| `room_destroy` | Remove sessions using specific rooms | Room-centric conflicts |
| `random_destroy` | Remove random subset | General exploration |

### Adaptive Behavior

As iterations progress:
- Successful operators get higher scores → selected more often
- Score decay prevents over-exploitation of early winners
- Destroy rate adapts: smaller when improving, larger when stuck

---

## 7. Iterated Local Search (ILS)

### Definition

**ILS** alternates between **perturbation** (escape local optimum) and **local search** (improve).
Maintains best solution found across iterations.

### Mathematical Formulation

```
x* ← LocalSearch(x₀)
repeat:
    x' ← Perturbation(x*)
    x'' ← LocalSearch(x')
    x* ← AcceptanceCriterion(x*, x'')
until stopping_condition
```

### Implementation

**Location:** [src/schedule_engine/ga/heuristics/meta.py](../src/schedule_engine/ga/heuristics/meta.py#L92)

```python
def iterated_local_search(
    individual: list[SessionGene],
    context: SchedulingContext,
    num_iterations: int = 5,
    perturbation_strength: float = 0.3,
) -> int:
```

### Algorithm Pseudocode

```
function ILS(individual, context, num_iterations, perturbation_strength):
    best ← copy(individual)
    best_fitness ← fitness(best)
    improvements ← 0
    
    for iter = 1 to num_iterations:
        # LOCAL SEARCH phase (intensification)
        VND(individual, context, max_neighborhoods=2)
        
        current_fitness ← fitness(individual)
        if current_fitness < best_fitness:
            best ← copy(individual)
            best_fitness ← current_fitness
            improvements += 1
        
        # PERTURBATION phase (diversification)
        temporal_shift(individual, probability=perturbation_strength)
        room_shuffle(individual, probability=perturbation_strength × 0.5)
    
    # Restore best
    individual ← copy(best)
    return improvements
```

### Perturbation Operators

| Operator | Effect |
|----------|--------|
| `temporal_shift` | Randomly shift some sessions to different time slots |
| `room_shuffle` | Randomly reassign rooms (keeping times) |

### Key Insight

Local search alone gets trapped. Perturbation "kicks" the solution to a new region,
then local search finds a new (possibly better) local optimum.

---

## 8. Guided Local Search (GLS)

### Definition

**GLS** augments local search with **penalties** on solution features. When stuck, it adds
penalties to "bad" features, guiding search away from previously explored regions.

### Mathematical Formulation

Augmented objective:
$$
h(x) = f(x) + \lambda \sum_{i} p_i \cdot I_i(x)
$$

Where:
- $f(x)$ = original objective
- $\lambda$ = penalty factor
- $p_i$ = penalty for feature $i$
- $I_i(x) = 1$ if feature $i$ is in solution $x$

Penalty update (when stuck):
$$
p_i \leftarrow p_i + 1 \quad \text{for feature } i^* = \arg\max_i \frac{c_i \cdot I_i(x)}{1 + p_i}
$$

### Implementation

**Location:** [src/schedule_engine/ga/heuristics/meta.py](../src/schedule_engine/ga/heuristics/meta.py#L235)

```python
def guided_local_search(
    individual: list[SessionGene],
    context: SchedulingContext,
    num_iterations: int = 10,
    penalty_factor: float = 0.1,
) -> int:
```

### Features Tracked

| Feature Type | Key Format | Example |
|--------------|------------|---------|
| Time penalties | `(session_id, time_slot)` | `("CS101-1", 5)` |
| Room penalties | `(session_id, room_id)` | `("CS101-1", "Lab3")` |
| Instructor penalties | `(session_id, instructor_id)` | `("CS101-1", "INS-001")` |

### Key Insight

GLS makes frequently-used (but problematic) features increasingly expensive,
eventually forcing the search to try alternatives.

---

## 9. Construction Heuristics

### Definition

**Construction heuristics** build solutions **from scratch** by intelligently ordering
decisions based on domain knowledge.

### Implementation

**Location:** [src/schedule_engine/ga/heuristics/construction.py](../src/schedule_engine/ga/heuristics/construction.py)

### Available Strategies

#### 1. Largest Degree First

Based on **graph coloring**: schedule sessions with more conflicts first (they have fewer options).

```python
def largest_degree_first(context: SchedulingContext) -> list[SessionGene]:
    """
    Conflict degree = shared instructors + shared groups + room constraints
    
    Algorithm:
    1. Calculate conflict degree for each session
    2. Sort by degree (descending)
    3. For each session (high→low degree):
       - Find earliest valid time
       - Assign room and instructor
    """
```

#### 2. Most Constrained First (MRV)

Based on **Minimum Remaining Values** from CSP: schedule sessions with fewest valid options first.

```python
def most_constrained_first(context: SchedulingContext) -> list[SessionGene]:
    """
    Constraint level = instructor restrictions + room requirements + existing assignments
    
    Algorithm:
    1. Calculate constraint levels
    2. While sessions remain:
       - Pick session with fewest valid time slots
       - Find best assignment
       - Update remaining constraints
    """
```

#### 3. Earliest Deadline First

Schedule by urgency (not deadline-based in scheduling, but priority-based).

### When to Use

- **Initial population:** Produce better starting solutions than random
- **LNS repair phase:** Reconstruct destroyed portions intelligently
- **Warm starts:** Begin GA with feasible (or near-feasible) solutions

---

## 10. Improvement Heuristics (VND Building Blocks)

### Implementation

**Location:** [src/schedule_engine/ga/heuristics/improvement.py](../src/schedule_engine/ga/heuristics/improvement.py)

### Kempe Chain

**Origin:** Graph coloring algorithm for the four-color problem.

```python
def kempe_chain(individual, context, max_iterations=5):
    """
    Algorithm:
    1. Find conflicting pairs (share instructor/group but same time)
    2. Build chain: sessions connected by conflicts
    3. Swap times along the chain
    4. Accept if fewer conflicts
    
    Example:
        Before: A(t=1) conflicts with B(t=1)
        Chain:  A → C → D (all share resources)
        After:  A(t=2), C(t=1), D(t=2)  [swapped along chain]
    """
```

### Ejection Chain

**Extension of Kempe:** Allows cascading moves where one change "ejects" another session.

```python
def ejection_chain(individual, context, max_chain_length=5, max_iterations=3):
    """
    Algorithm:
    1. Move session A to new time → ejects session B
    2. Move session B to new time → ejects session C
    3. Continue until chain terminates
    4. Accept if overall improvement
    
    More powerful but more expensive than Kempe.
    """
```

### Variable Depth Search

**Multi-move lookahead:** Try sequences of moves before committing.

```python
def variable_depth_search(individual, context, max_depth=3, max_iterations=5):
    """
    Algorithm:
    1. Try move A, evaluate
    2. Try moves A+B, evaluate  
    3. Try moves A+B+C, evaluate
    4. Accept best sequence found
    5. Repeat
    
    Thorough but computationally intensive.
    """
```

---

## Algorithm Integration in GA Modes

### Mode A: Baseline (Pure GA)

```
┌────────────────────────────────────────────┐
│  Population → Selection → Crossover →      │
│  Mutation → Evaluation → Repeat            │
│                                            │
│  NO local search, NO repairs               │
│  (That's why Mode A rarely improves!)      │
└────────────────────────────────────────────┘
```

### Mode B: Memetic (GA + Local Search)

```
┌────────────────────────────────────────────┐
│  Standard GA cycle...                      │
│        ↓                                   │
│  After crossover/mutation:                 │
│    → Apply local_search to offspring       │
│    → Greedy repair (first improvement)     │
│        ↓                                   │
│  Continue to next generation               │
└────────────────────────────────────────────┘
```

### Mode C: Round-Robin Repair

```
┌────────────────────────────────────────────┐
│  Standard GA cycle...                      │
│        ↓                                   │
│  Every N generations:                      │
│    → Apply repair operators in fixed order │
│    → Priority: 1→2→3→4→5→6→7→8             │
│        ↓                                   │
│  Continue to next generation               │
└────────────────────────────────────────────┘
```

### Mode D: Adaptive (ALNS-style)

```
┌────────────────────────────────────────────┐
│  Standard GA cycle...                      │
│        ↓                                   │
│  Track repair operator success rates       │
│  Select operator via roulette wheel        │
│        ↓                                   │
│  Apply selected operator                   │
│  Update scores based on improvement        │
│        ↓                                   │
│  Continue to next generation               │
└────────────────────────────────────────────┘
```

### Mode E: RL-Guided (Q-Learning)

```
┌────────────────────────────────────────────┐
│  Standard GA cycle...                      │
│        ↓                                   │
│  State: Current violations, generation #   │
│  Action: Which repair operator to apply    │
│  Reward: Violation reduction achieved      │
│        ↓                                   │
│  Q-table guides operator selection         │
│        ↓                                   │
│  Continue to next generation               │
└────────────────────────────────────────────┘
```

---

## Performance Comparison

### Speed vs Thoroughness

| Algorithm | Evaluations | Time | Quality |
|-----------|-------------|------|---------|
| Greedy LS | 10-20/gene | Fast | Acceptable |
| Exhaustive LS | 50-200/gene | Slow | Better |
| Repair Operators | 1-5/gene | Very Fast | Constraint-specific |
| LNS-IGLS | 500/subproblem | Medium | High (escapes local optima) |
| VND | Variable | Medium | High |
| ALNS | Variable | Medium | Adaptive |

### When Each Shines

| Scenario | Best Algorithm |
|----------|----------------|
| Many small violations | Repair operators (targeted fix) |
| Stuck in local optimum | LNS (big jump) or ILS (perturbation) |
| Multiple conflict types | VND (try different neighborhoods) |
| Unknown best strategy | ALNS (learns what works) |
| Final polish | Exhaustive LS (thorough search) |

---

## Suggestions & Recommendations

### 1. **Enable VND for Stagnation Recovery**

Currently, Mode A doesn't use any improvement heuristics. Add VND when no improvement for N generations:

```python
if generations_without_improvement > 10:
    for individual in population[:elite_size]:
        variable_neighborhood_descent(individual, context)
```

### 2. **Use Construction Heuristics for Initial Population**

Random initialization produces low-quality solutions. Use `largest_degree_first`:

```python
# In population initialization
for i in range(population_size):
    if i < smart_population_ratio * population_size:
        population[i] = largest_degree_first(context)
    else:
        population[i] = random_individual(context)
```

### 3. **Implement Repair Triggering Based on Violation Type**

Instead of round-robin, match repair to detected violations:

```python
violations = detect_violations(individual)
for v in violations:
    repair_op = REPAIR_FOR_CONSTRAINT[v.constraint_code]
    repair_op(individual, context)
```

### 4. **Add LNS for Severe Cases**

When regular repairs fail, escalate to LNS:

```python
# After regular repair attempt
if count_hard_violations(individual) > threshold:
    lns_igls_repair(individual, context, max_subproblem_size=20)
```

### 5. **Consider Hybrid ALNS + VND**

Use ALNS for operator selection, VND for the actual improvement:

```python
operator = alns_select(operator_scores)
if operator == "vnd":
    improvements = variable_neighborhood_descent(individual, context)
elif operator == "lns":
    lns_igls_repair(individual, context)
elif operator == "kempe_only":
    improvements = kempe_chain(individual, context)
# Update ALNS scores based on improvement
```

### 6. **Implement Adaptive Destroy Rate for LNS**

Start with small destruction, increase if stuck:

```python
destroy_rate = initial_rate
for iteration in range(max_iterations):
    if iteration_improved:
        destroy_rate *= 0.9  # Less destruction
    else:
        destroy_rate *= 1.1  # More destruction
    destroy_rate = clamp(destroy_rate, 0.1, 0.6)
```

### 7. **Add VLNS (Variable Large Neighborhood Search)**

This is **NOT currently implemented** but would be valuable:

```python
def variable_large_neighborhood_search(individual, context):
    """
    VLNS = VND + LNS
    
    Vary BOTH:
    - Neighborhood TYPE (kempe, ejection, etc.)
    - Neighborhood SIZE (small local moves vs large LNS destruction)
    
    Strategy:
    1. Try small neighborhoods first (cheap)
    2. Escalate to larger neighborhoods if stuck
    3. Cycle back to small after improvement
    """
    neighborhood_sizes = [1, 5, 10, 20, 50]
    current_size_idx = 0
    
    while current_size_idx < len(neighborhood_sizes):
        size = neighborhood_sizes[current_size_idx]
        
        if size <= 5:
            # Local search moves
            improved = vnd_with_limit(individual, context, max_moves=size)
        else:
            # LNS-style destruction
            improved = lns_repair(individual, context, subproblem_size=size)
        
        if improved:
            current_size_idx = 0  # Restart from small
        else:
            current_size_idx += 1  # Try larger
```

### 8. **Profile and Tune Parameters**

Key parameters to tune:

| Parameter | Default | Tune Range | Impact |
|-----------|---------|------------|--------|
| `max_neighborhood_size` | 100 | 50-500 | More = better but slower |
| `igls_max_iterations` | 500 | 200-2000 | Repair thoroughness |
| `expand_hops` | 0 | 0-3 | LNS scope (0 = conflicts only) |
| `perturbation_strength` | 0.3 | 0.1-0.5 | ILS escape intensity |
| `score_decay` | 0.95 | 0.9-0.99 | ALNS adaptation speed |

### 9. **Add Parallel LNS for Large Problems**

Your codebase already has parallel infrastructure. Use it for LNS:

```python
# In lns/operator.py - already has this structure
with ProcessPoolExecutor(max_workers=get_cpu_count()) as executor:
    futures = [executor.submit(repair_subproblem, sp) for sp in subproblems]
    results = [f.result() for f in as_completed(futures)]
```

### 10. **Monitor and Log Algorithm Performance**

Add metrics to understand what's working:

```python
@dataclass
class AlgorithmMetrics:
    algorithm_name: str
    total_calls: int = 0
    total_improvements: int = 0
    avg_improvement: float = 0.0
    avg_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.total_improvements / max(1, self.total_calls)
```

---

## Summary

| Algorithm | Type | Speed | Escape Power | When to Use |
|-----------|------|-------|--------------|-------------|
| **Greedy LS** | Local |  | Low | Stagnation trigger, large pop |
| **Exhaustive LS** | Local |  | Low | Elite individuals, final polish |
| **Repairs** | Targeted |  | None | After crossover/mutation |
| **LNS-IGLS** | Destroy+Repair |  | High | Severe conflicts |
| **VND** | Multi-neighborhood |  | Medium | Systematic improvement |
| **ALNS** | Adaptive |  | High | Unknown best strategy |
| **ILS** | Perturb+Search |  | Medium | Diversification needed |
| **GLS** | Penalty-guided |  | Medium | Feature-rich solutions |

---

## Case Study: Initialization Strategy Analysis

### Current Gene Structure

```python
@dataclass
class SessionGene:
    # IMMUTABLE (structural identity - NEVER change)
    course_id: str           # Which course
    course_type: str         # "theory" or "practical"  
    group_ids: list[str]     # Which student groups (can be multiple for theory)
    num_quanta: int          # Duration (fixed by course.quanta_per_week)
    
    # MUTABLE (can change during evolution)
    instructor_id: str       # Who teaches
    room_id: str             # Where
    start_quanta: int        # When (start time only, duration is fixed)
```

### Mutable vs Immutable Design Rationale

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IMMUTABLE ATTRIBUTES                                  │
│                   (Fixed at initialization, never change)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  course_id       │ Defines WHAT is being taught                             │
│  course_type     │ theory/practical - determines room requirements          │
│  group_ids       │ Defines WHO attends (pedagogical constraint)             │
│  num_quanta      │ Duration = L+T or P from syllabus (cannot compress)      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WHY IMMUTABLE?                                                              │
│  ─────────────────                                                           │
│  • course_id + group_ids = enrollment relationship (from curriculum)         │
│  • num_quanta = credit hours (regulatory requirement)                        │
│  • Changing these would violate course_completeness (HC7)                    │
│  • GA structure: each (course, groups) pair = exactly one gene              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         MUTABLE ATTRIBUTES                                   │
│                   (Changed by mutation, crossover, repair)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  instructor_id   │ WHO teaches - must satisfy HC3 (qualifications)          │
│  room_id         │ WHERE session happens - must satisfy HC4 (suitability)   │
│  start_quanta    │ WHEN session starts - exclusivity constraints            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  WHY MUTABLE?                                                                │
│  ─────────────                                                               │
│  • These are the DECISIONS the scheduler makes                               │
│  • Multiple valid options exist (qualified instructors, suitable rooms)      │
│  • Changing them doesn't break curriculum structure                          │
│  • This is what optimization searches over                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Current Initialization Strategies

Your codebase has **3 initialization strategies**, each with different constraint-awareness levels:

#### 1. Pure Random (`generate_pure_random_population`)

```python
# Location: population.py - generate_pure_random_population()
def _create_pure_random_gene(...):
    # Random qualified instructor (constraint-aware for HC3)
    instructor_id = random.choice(qualified_instructors)
    
    # Random room (NOT constraint-aware - ignores suitability)
    room_id = random.choice(list(context.rooms.keys()))
    
    # Random time (NOT constraint-aware - ignores conflicts)
    start_quanta = random.randint(0, max_start)
```

**Constraint Awareness:**
| Constraint | Respected? | How |
|------------|------------|-----|
| HC3 (instructor qualification) |  Yes | Only picks from `qualified_instructors` |
| HC4 (room suitability) |  No | Random room |
| HC1 (group exclusivity) |  No | Random time |
| HC2 (instructor exclusivity) |  No | Random time |
| HC5 (instructor availability) |  No | Random time |
| HC8 (room exclusivity) |  No | Random time |

**Expected violations:** 600-800 hard constraint violations per individual

#### 2. Smart/Conflict-Aware (`generate_course_group_aware_population`)

```python
# Location: population.py - create_session_gene_with_conflict_avoidance()
def create_session_gene_with_conflict_avoidance(...):
    # 1. Try instructor-availability-aware assignment first
    qualified_with_availability = find_qualified_instructors_with_availability(
        course_id, context, num_quanta,
        exclude_quanta=used_by_instructors | used_by_groups
    )
    
    # 2. Pick from top 3 instructors with most flexibility
    if qualified_with_availability:
        top_instructors = qualified_with_availability[:3]
        chosen_instructor, available_starts = random.choice(top_instructors)
        start_q = random.choice(available_starts)  # Conflict-free time!
    
    # 3. Room selection respects course type
    suitable_rooms = find_suitable_rooms(course, session_type, context)
    room = random.choice(suitable_rooms)
```

**Constraint Awareness:**
| Constraint | Respected? | How |
|------------|------------|-----|
| HC3 (instructor qualification) |  Yes | `qualified_instructors` filter |
| HC4 (room suitability) |  Yes | `find_suitable_rooms()` |
| HC1 (group exclusivity) |  Yes | Tracks `group_schedule` |
| HC2 (instructor exclusivity) |  Yes | Tracks `instructor_schedule` |
| HC5 (instructor availability) |  Yes | `find_qualified_instructors_with_availability()` |
| HC8 (room exclusivity) |  Partial | Rooms not tracked across genes |

**Expected violations:** 100-200 hard constraint violations per individual

#### 3. Hybrid/Greedy (`generate_hybrid_population`)

Combines greedy construction (largest-degree-first) with random.

**Expected violations:** 50-150 hard constraint violations per individual

---

### Is the Initialization Optimal? Analysis

#### What's Already Optimal 

1. **Immutable/Mutable Separation**
   - Course structure is correctly preserved
   - num_quanta is never mutated (duration integrity)
   - Crossover swaps attributes, not genes

2. **Subsession Breaking**
   ```python
   # Theory 6 quanta → [2, 2, 2] (three 2-hour sessions)
   # Practical 30 quanta → [30] (single studio session)
   subsession_durations = get_subsession_durations(quanta_per_week, course_type)
   ```
   - Pedagogically sound: theory broken into digestible chunks
   - Practicals kept continuous (lab work requires continuity)

3. **Group Hierarchy Handling**
   - Theory: All sibling groups listed together `["BME1A", "BME1B"]`
   - Practical: Each subgroup separately `["BME1A"]`
   - Prevents double-scheduling of students

4. **Instructor Availability Awareness**
   - Part-time instructors correctly filtered by available quanta
   - Full-time assumed always available (correct per domain)

5. **Conflict Tracking**
   - `instructor_schedule`, `group_schedule` track quanta usage
   - Avoids known conflicts during initialization

---

#### What's NOT Optimal (Room for Improvement) ️

##### Issue 1: Room Exclusivity Not Tracked

```python
# Current: Tracks instructors and groups, but NOT ROOMS
used_quanta: set[int] = set()           # General pool
instructor_schedule: dict[str, set[int]]  # Per instructor
group_schedule: dict[str, set[int]]       # Per group
# MISSING: room_schedule: dict[str, set[int]]  # Per room!
```

**Impact:** Two sessions can be assigned to the same room at the same time during initialization.

**Fix (Easy):**
```python
# Add room tracking
room_schedule: dict[str, set[int]] = {}

# Before assigning room:
available_rooms = [
    r for r in suitable_rooms 
    if all(q not in room_schedule.get(r.room_id, set()) 
           for q in range(start_q, start_q + num_quanta))
]
room = random.choice(available_rooms) if available_rooms else random.choice(suitable_rooms)

# After assigning:
if room.room_id not in room_schedule:
    room_schedule[room.room_id] = set()
room_schedule[room.room_id].update(range(start_q, start_q + num_quanta))
```

##### Issue 2: Room Capacity Not Checked

```python
# Current: Checks room TYPE but not CAPACITY
suitable_rooms = find_suitable_rooms(course, session_type, context)
# Doesn't verify: room.capacity >= sum(group.size for g in group_ids)
```

**Impact:** Small room assigned to large group → soft constraint violation (or hard if overcrowded).

**Fix (Medium):**
```python
def find_suitable_rooms_with_capacity(course, group_ids, session_type, context):
    total_students = sum(context.groups[gid].size for gid in group_ids)
    suitable = find_suitable_rooms(course, session_type, context)
    return [r for r in suitable if r.capacity >= total_students]
```

##### Issue 3: Cross-Day Sessions Not Prevented for Short Courses

```python
# SessionGene.__post_init__ has smart day-boundary logic:
if num_quanta <= day_quanta and start_quanta + num_quanta > end_of_day:
    self.start_quanta = max(day_offset, end_of_day - self.num_quanta)
```

This is **already handled** in `SessionGene.__post_init__()`. No issue here.

##### Issue 4: Parallel Initialization Race Conditions

```python
# Pure random is sequential (no conflict tracking needed)
# Smart initialization is parallel BUT each worker has INDEPENDENT conflict tracking
```

**Impact:** When generating population in parallel, workers don't share conflict state.
Worker 1 might assign Room A at time 5, Worker 2 does the same → duplicated conflicts.

**Why it's OK:** Parallel initialization creates more conflicts, but:
- Diversity is higher (good for GA exploration)
- Repairs fix conflicts quickly
- Trade-off: speed vs initial quality

##### Issue 5: No Soft Constraint Awareness in Initialization

```python
# Current: Only hard constraints considered
# Not considered:
#   - SC1: instructor_load_balance (spread sessions across instructors)
#   - SC2: schedule_compactness (minimize gaps)
#   - SC3: time_preferences (morning/afternoon preferences)
```

**Impact:** Initial solutions have poor soft constraint scores.

**Why it might be OK:** 
- Hard constraints are priority (feasibility first)
- Soft constraints are optimized during evolution
- Adding soft awareness would slow initialization significantly

---

### Verdict: Is It Optimal?

| Aspect | Score | Notes |
|--------|-------|-------|
| **Structural Correctness** | 10/10 | Mutable/immutable perfectly separated |
| **Hard Constraint Awareness** | 8/10 | Missing room exclusivity tracking |
| **Soft Constraint Awareness** | 3/10 | Intentionally not considered |
| **Performance (Speed)** | 9/10 | Parallel, efficient |
| **Population Diversity** | 8/10 | Good with hybrid strategy |
| **Memory Efficiency** | 10/10 | Contiguous quanta representation |

**Overall: Your initialization is 85% optimal.**

The gene structure design is excellent. The main gap is **room exclusivity tracking** during 
smart initialization.

---

### Recommended Quick Fixes (Low Effort, High Impact)

#### Fix 1: Add Room Tracking (15 mins)

In `create_session_gene_with_conflict_avoidance()`:

```python
def create_session_gene_with_conflict_avoidance(
    ...
    room_schedule: ScheduleMap,  # ADD THIS PARAMETER
) -> SessionGene:
    ...
    # Filter rooms by time availability
    candidate_rooms = [
        r for r in suitable_rooms
        if not any(
            q in room_schedule.get(r.room_id, set())
            for q in assigned_quanta
        )
    ]
    room = random.choice(candidate_rooms) if candidate_rooms else random.choice(suitable_rooms)
    
    # Update tracking
    if room:
        if room.room_id not in room_schedule:
            room_schedule[room.room_id] = set()
        room_schedule[room.room_id].update(assigned_quanta)
```

**Expected improvement:** Reduce HC8 violations by 60-80%.

#### Fix 2: Capacity-Aware Room Selection (10 mins)

```python
def find_suitable_rooms(course, session_type, context, group_ids=None) -> list[Room]:
    suitable = [r for r in context.rooms.values() if room_type_matches(r, course)]
    
    if group_ids:
        total_students = sum(context.groups.get(g, Group(g, 0)).size for g in group_ids)
        suitable = [r for r in suitable if r.capacity >= total_students]
    
    return suitable if suitable else list(context.rooms.values())
```

**Expected improvement:** Reduce room-capacity soft constraint violations.

---

### Can You Squeeze More From Gene Structure? 

#### Current Memory Usage

```python
SessionGene = {
    course_id: str,        # ~20 bytes (pointer + string)
    course_type: str,      # ~15 bytes
    instructor_id: str,    # ~15 bytes
    group_ids: list[str],  # ~50 bytes (list + strings)
    room_id: str,          # ~15 bytes
    start_quanta: int,     # 28 bytes (Python int object)
    num_quanta: int,       # 28 bytes
}
# Total: ~170 bytes per gene
```

#### Optimization Options

##### Option A: Use `__slots__` (Already Done )
The `@dataclass(slots=True)` on Course shows you know about this.
SessionGene could benefit from explicit `__slots__`:

```python
@dataclass(slots=True)
class SessionGene:
    ...
# Saves ~40 bytes per gene (removes __dict__)
```

##### Option B: Pack IDs as Integers

```python
# Instead of string IDs:
course_id: str = "ARCH101"  # ~15 bytes
# Use integer indices:
course_idx: int = 42        # 28 bytes, but hashable/faster

# Create lookup tables at init:
course_id_to_idx = {c.course_id: i for i, c in enumerate(courses)}
instructor_id_to_idx = {i.instructor_id: j for j, i in enumerate(instructors)}
```

**Trade-off:** Faster comparison, smaller memory, but requires decoding for display.

##### Option C: Numpy Arrays for Population

```python
# Instead of: List[List[SessionGene]]
# Use: np.ndarray with dtype=[('course_idx', 'i4'), ('start', 'i4'), ...]

# Single individual as structured array:
gene_dtype = np.dtype([
    ('course_idx', 'i4'),
    ('type_idx', 'i1'),
    ('instructor_idx', 'i2'),
    ('room_idx', 'i2'),
    ('start_quanta', 'i2'),
    ('num_quanta', 'i1'),
    ('group_mask', 'u4'),  # Bitmask for up to 32 groups
])
# ~15 bytes per gene (vs ~170 bytes) = 10x reduction!
```

**Trade-off:** Major refactor, but 10x memory reduction and vectorized operations.

##### Option D: Frozen Immutables / Flyweight Pattern

```python
# Since (course_id, course_type, group_ids, num_quanta) never change,
# share them across genes via flyweight:

@dataclass(frozen=True)
class SessionIdentity:
    course_id: str
    course_type: str
    group_ids: tuple[str, ...]  # Tuple for hashability
    num_quanta: int

# Gene only stores reference to shared identity:
@dataclass
class SessionGene:
    identity: SessionIdentity  # Shared (readonly)
    instructor_id: str         # Mutable
    room_id: str               # Mutable
    start_quanta: int          # Mutable
```

**Trade-off:** Saves memory for large populations, adds indirection.

---

### Summary: What Should You Do?

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| **1** | Add room tracking in smart init | 15 min | -60% HC8 violations |
| **2** | Add `__slots__` to SessionGene | 5 min | -20% memory |
| **3** | Capacity-aware room selection | 10 min | Better soft scores |
| **4** | Integer IDs (if performance-critical) | 2 hrs | 2x faster evaluation |
| **5** | Numpy arrays (major refactor) | 1-2 days | 10x memory reduction |

**For your thesis timeline:** Focus on #1 and #2. They give best ROI.

---

## Case Study: Gene Tagging Analysis

### What is Gene Tagging?

**Gene Tagging** refers to adding metadata fields to genes that track state, history, 
or computed properties. Instead of recomputing information, tags cache it directly on 
the gene.

### Current State (No Tags)

Your current `SessionGene` is **tag-free** — it only stores core scheduling data:

```python
@dataclass
class SessionGene:
    # Core data only - no metadata
    course_id: str
    course_type: str
    instructor_id: str
    group_ids: list[str]
    room_id: str
    start_quanta: int
    num_quanta: int
```

**Current Workflow (Violation Detection):**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CURRENT: Recompute Every Time                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Generation 1:                                                               │
│    detect_violated_genes(individual) → builds 3 schedule maps → O(n)        │
│    repair(individual)                                                        │
│    evaluate(individual) → builds schedule AGAIN → O(n)                       │
│                                                                              │
│  Generation 2:                                                               │
│    detect_violated_genes(individual) → rebuilds maps → O(n)                  │
│    ... same for all 100 generations ...                                      │
│                                                                              │
│  PROBLEM: Same maps built hundreds of times per individual                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Potential Gene Tags

#### Category 1: Violation Status Tags

```python
@dataclass
class SessionGene:
    # ... existing fields ...
    
    # VIOLATION TAGS (cached detection results)
    has_group_conflict: bool = False
    has_room_conflict: bool = False
    has_instructor_conflict: bool = False
    has_qualification_violation: bool = False
    has_availability_violation: bool = False
    has_room_mismatch: bool = False
    
    # Computed property
    @property
    def is_violated(self) -> bool:
        return any([
            self.has_group_conflict,
            self.has_room_conflict,
            self.has_instructor_conflict,
            self.has_qualification_violation,
            self.has_availability_violation,
            self.has_room_mismatch,
        ])
    
    @property
    def violation_count(self) -> int:
        return sum([...])
```

**Benefits:**
- `O(1)` violation check instead of `O(n)` map building
- Selective repair can instantly find violated genes
- Fitness evaluation can sum cached flags

**Costs:**
- +6 bytes per gene (6 bools)
- Tags must be invalidated after ANY mutation/crossover
- Risk of stale tags if invalidation is forgotten

#### Category 2: History/Tracking Tags

```python
@dataclass
class SessionGene:
    # ... existing fields ...
    
    # HISTORY TAGS
    mutation_count: int = 0           # How many times mutated
    repair_attempts: int = 0          # How many repair attempts
    last_modified_gen: int = -1       # Last generation modified
    created_gen: int = 0              # Generation when created
    
    # LOCK TAGS (for Tabu Search integration)
    locked_until_gen: int = -1        # Forbid modification until gen N
    lock_reason: str = ""             # Why locked (e.g., "tabu", "elite")
```

**Benefits:**
- Tabu search: prevent cycling back to recently visited states
- Diagnostics: identify "stubborn" genes that resist repair
- Adaptive operators: treat old genes differently than new ones

**Costs:**
- +20-30 bytes per gene
- Need to update tags in ALL operators (mutation, crossover, repair)
- Complexity increase in codebase

#### Category 3: Priority/Scheduling Tags

```python
@dataclass
class SessionGene:
    # ... existing fields ...
    
    # PRIORITY TAGS
    scheduling_priority: int = 0      # Higher = schedule first
    flexibility_score: float = 1.0    # How many valid alternatives exist
    conflict_degree: int = 0          # How many other genes it conflicts with
```

**Benefits:**
- Construction heuristics can sort by priority
- ALNS can destroy high-conflict genes first
- Repair can target inflexible genes

**Costs:**
- Must recompute after structure changes
- Adds complexity to gene lifecycle

#### Category 4: Computed/Cached Properties

```python
@dataclass
class SessionGene:
    # ... existing fields ...
    
    # CACHED COMPUTATIONS
    _cached_quanta_list: list[int] | None = field(default=None, repr=False)
    _cached_day: str | None = field(default=None, repr=False)
    _cached_time_slot: str | None = field(default=None, repr=False)
    
    def get_quanta_list(self) -> list[int]:
        if self._cached_quanta_list is None:
            self._cached_quanta_list = list(range(self.start_quanta, self.end_quanta))
        return self._cached_quanta_list
    
    def invalidate_cache(self) -> None:
        self._cached_quanta_list = None
        self._cached_day = None
        self._cached_time_slot = None
```

**Benefits:**
- Avoid repeated `list(range(...))` calls
- Day/time string computed once, reused in display

**Costs:**
- Cache invalidation complexity
- Memory overhead for caches

---

### Full Analysis: Should You Add Tags?

#### Decision Matrix

| Tag Type | Benefit | Implementation Cost | Memory Cost | Maintenance Risk | Verdict |
|----------|---------|---------------------|-------------|------------------|---------|
| **Violation Flags** | High (O(1) detection) | Medium | +6 bytes | HIGH (stale tags) | ️ Maybe |
| **History Tags** | Medium (diagnostics) | Low | +20 bytes | Medium |  Skip |
| **Priority Tags** | Medium (ordering) | High | +12 bytes | Medium |  Skip |
| **Cached Props** | Low (minor speedup) | Medium | Variable | Medium |  Skip |

---

### Deep Dive: Violation Status Tags

This is the only tag type that could significantly benefit your requirement.

#### Current Detection Flow

```python
# File: detector.py
def detect_violated_genes(individual, context, strategy="hybrid"):
    # Build 3 schedule maps from scratch every time
    group_schedule = _build_group_schedule_map(individual)      # O(n*q)
    room_schedule = _build_room_schedule_map(individual)        # O(n*q)
    instructor_schedule = _build_instructor_schedule_map(individual)  # O(n*q)
    
    # Scan maps for conflicts
    for group_id, schedule in group_schedule.items():
        for quantum, gene_indices in schedule.items():
            if len(gene_indices) > 1:
                violations[idx].append("group_overlap")
    # ... repeat for room, instructor ...
```

**Cost per call:** O(n × q) where n=genes, q=avg quanta per gene

For your data: ~200 genes × 3 quanta avg = 600 operations per map × 3 maps = 1,800 operations

**Frequency:** Called in:
- `repair_individual_unified()` — every repair iteration
- `evaluate()` (indirectly via Timetable)
- `intensive_local_search()`
- `lns_igls_repair()`

**Total per generation:** ~5-10 times per individual × 50 individuals = 250-500 calls

#### With Violation Tags

```python
@dataclass
class SessionGene:
    # ... existing fields ...
    
    # Violation flags (updated by constraint evaluator)
    _v_group: bool = field(default=False, repr=False)
    _v_room: bool = field(default=False, repr=False)
    _v_instructor: bool = field(default=False, repr=False)
    _v_qualification: bool = field(default=False, repr=False)
    _v_availability: bool = field(default=False, repr=False)
    _v_suitability: bool = field(default=False, repr=False)
    _dirty: bool = field(default=True, repr=False)  # Needs re-evaluation
    
    @property
    def is_violated(self) -> bool:
        return self._v_group or self._v_room or self._v_instructor or \
               self._v_qualification or self._v_availability or self._v_suitability
    
    def mark_dirty(self) -> None:
        """Call after any mutation to invalidate cached flags."""
        self._dirty = True
```

**New Detection Flow:**

```python
def detect_violated_genes_tagged(individual, context):
    # Only rebuild maps if ANY gene is dirty
    if any(g._dirty for g in individual):
        _update_all_violation_tags(individual, context)
        for g in individual:
            g._dirty = False
    
    # Now just filter by flags — O(n), not O(n*q)
    return {i: _get_violations(g) for i, g in enumerate(individual) if g.is_violated}
```

**Benefit:** If no genes were modified, detection is O(n) bool checks, not O(n×q) map building.

#### The Critical Problem: Stale Tags

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DANGER ZONE: Stale Tag Scenarios                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Scenario 1: Forgot to mark dirty                                           │
│  ────────────────────────────────                                            │
│    gene.start_quanta = 10                                                    │
│    # OOPS: forgot gene.mark_dirty()                                          │
│    # Tag says "no conflict" but there IS one now                             │
│                                                                              │
│  Scenario 2: Cascading invalidation                                         │
│  ────────────────────────────────────                                        │
│    gene_A moves to time slot 5                                               │
│    gene_A.mark_dirty()  ✓                                                    │
│    # But gene_B was at slot 5 and now has FEWER conflicts!                   │
│    # gene_B's tag says "has_conflict" but it's now FALSE                     │
│    # WHO IS RESPONSIBLE FOR INVALIDATING gene_B?                             │
│                                                                              │
│  Scenario 3: Crossover complexity                                           │
│  ─────────────────────────────────                                           │
│    # Crossover swaps start_quanta between gene1 and gene2                   │
│    gene1.start_quanta, gene2.start_quanta = gene2.start_quanta, gene1...    │
│    gene1.mark_dirty()                                                        │
│    gene2.mark_dirty()                                                        │
│    # But what about OTHER genes that now conflict with gene1/gene2?         │
│    # Need to mark ALL genes dirty after ANY swap!                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**The Cascade Problem:** When gene A moves, it affects:
1. Gene A's own violation status
2. Genes that WERE conflicting with A (may now be clear)
3. Genes that NOW conflict with A (new conflicts)

**Solution options:**
1. **Conservative:** Mark ALL genes dirty after any modification → loses benefit
2. **Track neighbors:** Maintain conflict graph, invalidate neighbors → complex
3. **Generation-based:** Clear all tags at generation start → simpler, still beneficial

#### Realistic Implementation

```python
# In GAScheduler or population-level logic:

def start_generation(population):
    """Clear all violation caches at generation start."""
    for individual in population:
        for gene in individual:
            gene._dirty = True

def after_operator(individual, modified_indices):
    """Mark entire individual as needing re-evaluation."""
    # Conservative approach: any modification invalidates all tags
    for gene in individual:
        gene._dirty = True

def evaluate_with_caching(individual, context):
    """Evaluate with tag caching."""
    # Only one full evaluation per generation per individual
    if any(g._dirty for g in individual):
        _full_evaluation_and_tag_update(individual, context)
```

**Net benefit:** Instead of 5-10 map builds per generation per individual, 
just 1 build + 4-9 O(n) tag reads.

---

### Alternative: External Tag Storage (Recommended)

Instead of adding fields to `SessionGene`, store tags EXTERNALLY:

```python
@dataclass
class IndividualMetadata:
    """Metadata stored separately from genes (not pickled, not inherited)."""
    
    violation_map: dict[int, list[str]] = field(default_factory=dict)
    last_evaluated_gen: int = -1
    dirty: bool = True
    
    def get_violated_indices(self) -> set[int]:
        return set(self.violation_map.keys())
    
    def is_stale(self, current_gen: int) -> bool:
        return self.dirty or self.last_evaluated_gen != current_gen

# Store metadata per-individual (not on genes)
individual_metadata: dict[int, IndividualMetadata] = {}

def get_metadata(individual) -> IndividualMetadata:
    ind_id = id(individual)
    if ind_id not in individual_metadata:
        individual_metadata[ind_id] = IndividualMetadata()
    return individual_metadata[ind_id]
```

**Benefits:**
- SessionGene stays lightweight
- No serialization issues (metadata not pickled)
- Clear separation of concerns
- Easy to invalidate (just set `dirty=True`)

**This is what you already do!** Your `detect_violated_genes()` returns a dict, 
which IS external tag storage.

---

### What You Actually Need: Smarter Caching

Your code ALREADY has the right structure. The issue is **repeated computation**, 
not **missing tags**.

#### Current Inefficiency

```python
# In repair/basic.py - each repair operator rebuilds maps
def repair_group_overlaps(individual, context):
    # Builds group_schedule map
    overlaps = _find_group_overlaps(individual)  # O(n*q)
    ...

def repair_room_conflicts(individual, context):
    # Builds room_schedule map AGAIN (even though just built in previous repair)
    conflicts = _find_room_conflicts(individual)  # O(n*q)
    ...
```

#### The Fix: Schedule Index Caching (Not Gene Tags)

```python
class ScheduleIndex:
    """Cached schedule maps for conflict detection."""
    
    def __init__(self, individual: list[SessionGene]):
        self._individual = individual
        self._group_map = None
        self._room_map = None
        self._instructor_map = None
        self._dirty = True
    
    def invalidate(self):
        self._dirty = True
    
    def _rebuild_if_dirty(self):
        if self._dirty:
            self._group_map = _build_group_schedule_map(self._individual)
            self._room_map = _build_room_schedule_map(self._individual)
            self._instructor_map = _build_instructor_schedule_map(self._individual)
            self._dirty = False
    
    @property
    def group_conflicts(self) -> dict[int, list[str]]:
        self._rebuild_if_dirty()
        return _find_conflicts(self._group_map)

# Usage:
def repair_individual_with_cache(individual, context):
    index = ScheduleIndex(individual)
    
    # First repair uses index (builds maps once)
    group_violations = index.group_conflicts
    for idx in group_violations:
        _repair_gene(individual[idx], ...)
        index.invalidate()  # Maps need rebuild after modification
    
    # Second repair reuses maps if no modification happened
    room_violations = index.room_conflicts
    ...
```

---

### Verdict: Should You Tag Genes?

| Approach | Recommendation | Why |
|----------|----------------|-----|
| **Violation flags on gene** |  NO | Stale tag risk outweighs benefits |
| **History tags on gene** |  NO | Low value, maintenance burden |
| **Priority tags on gene** |  NO | Can compute on-demand |
| **External metadata dict** |  Already done | `detect_violated_genes()` returns this |
| **ScheduleIndex caching** |  YES | High impact, low risk |

---

### Recommended Implementation: ScheduleIndex

This is the highest-value change for your requirement.

```python
# New file: src/schedule_engine/ga/core/schedule_index.py

from collections import defaultdict
from dataclasses import dataclass, field
from schedule_engine.domain.gene import SessionGene

@dataclass
class ScheduleIndex:
    """
    Cached schedule maps for efficient conflict detection.
    
    Instead of rebuilding maps for each constraint check, build once
    and reuse until individual is modified.
    
    Usage:
        index = ScheduleIndex.from_individual(individual)
        
        # All these use cached maps (only built once):
        group_conflicts = index.find_group_conflicts()
        room_conflicts = index.find_room_conflicts()
        instructor_conflicts = index.find_instructor_conflicts()
        
        # After modification:
        individual[5].start_quanta = 10
        index.invalidate()  # Next access rebuilds
    """
    
    _individual: list[SessionGene]
    _group_map: dict[str, dict[int, list[int]]] = field(default_factory=dict, repr=False)
    _room_map: dict[str, dict[int, list[int]]] = field(default_factory=dict, repr=False)
    _instructor_map: dict[str, dict[int, list[int]]] = field(default_factory=dict, repr=False)
    _valid: bool = field(default=False, repr=False)
    
    @classmethod
    def from_individual(cls, individual: list[SessionGene]) -> "ScheduleIndex":
        return cls(_individual=individual)
    
    def invalidate(self) -> None:
        self._valid = False
    
    def _ensure_valid(self) -> None:
        if self._valid:
            return
        
        self._group_map = defaultdict(lambda: defaultdict(list))
        self._room_map = defaultdict(lambda: defaultdict(list))
        self._instructor_map = defaultdict(lambda: defaultdict(list))
        
        for idx, gene in enumerate(self._individual):
            for q in range(gene.start_quanta, gene.end_quanta):
                for gid in gene.group_ids:
                    self._group_map[gid][q].append(idx)
                self._room_map[gene.room_id][q].append(idx)
                self._instructor_map[gene.instructor_id][q].append(idx)
        
        self._valid = True
    
    def find_group_conflicts(self) -> dict[int, list[int]]:
        """Return {gene_idx: [conflicting_gene_indices]}."""
        self._ensure_valid()
        conflicts = defaultdict(list)
        for schedule in self._group_map.values():
            for gene_indices in schedule.values():
                if len(gene_indices) > 1:
                    for idx in gene_indices:
                        conflicts[idx].extend(i for i in gene_indices if i != idx)
        return dict(conflicts)
    
    def find_room_conflicts(self) -> dict[int, list[int]]:
        self._ensure_valid()
        conflicts = defaultdict(list)
        for schedule in self._room_map.values():
            for gene_indices in schedule.values():
                if len(gene_indices) > 1:
                    for idx in gene_indices:
                        conflicts[idx].extend(i for i in gene_indices if i != idx)
        return dict(conflicts)
    
    def find_instructor_conflicts(self) -> dict[int, list[int]]:
        self._ensure_valid()
        conflicts = defaultdict(list)
        for schedule in self._instructor_map.values():
            for gene_indices in schedule.values():
                if len(gene_indices) > 1:
                    for idx in gene_indices:
                        conflicts[idx].extend(i for i in gene_indices if i != idx)
        return dict(conflicts)
    
    def count_total_conflicts(self) -> int:
        """Total unique conflict pairs."""
        self._ensure_valid()
        pairs = set()
        for method in [self.find_group_conflicts, self.find_room_conflicts, self.find_instructor_conflicts]:
            for idx, conflicting in method().items():
                for other in conflicting:
                    pairs.add((min(idx, other), max(idx, other)))
        return len(pairs)
```

**Integration:**

```python
# In repair_individual_unified():
def repair_individual_unified(individual, context, max_iterations=2, selective=True):
    index = ScheduleIndex.from_individual(individual)
    
    for iteration in range(max_iterations):
        # Use cached index for detection
        group_conflicts = index.find_group_conflicts()
        
        if not group_conflicts:
            break
        
        for idx in group_conflicts:
            _repair_gene(individual[idx], ...)
            index.invalidate()  # Rebuild on next access
```

---

### Summary: Gene Tagging Verdict

| Question | Answer |
|----------|--------|
| **Should you add tags to SessionGene?** |  No - adds complexity, stale data risk |
| **Is external metadata useful?** |  Already implemented via `detect_violated_genes()` |
| **What SHOULD you do?** |  Add `ScheduleIndex` caching class |
| **Expected benefit** | 3-5x speedup in repair/evaluation |
| **Implementation effort** | ~2 hours |

**Bottom line:** Don't tag genes. Cache schedule maps externally via `ScheduleIndex`.

---

*Last updated: February 2026*  
*Generated for schedule-engine project*


