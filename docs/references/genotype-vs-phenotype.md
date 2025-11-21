# Genotype vs Phenotype in Schedule Engine

**Date**: November 21, 2025  
**Domain**: Genetic Algorithms for University Course Timetabling  
**Context**: NSGA-II multi-objective optimization with RL-guided hyper-heuristics

---

## Executive Summary

In evolutionary computation, the **genotype-phenotype distinction** is fundamental to understanding how solutions are represented (genotype) versus how they perform (phenotype). In this scheduling system:

- **Genotype** = Internal chromosome representation (`List[SessionGene]`)
- **Phenotype** = Observable solution quality (fitness values: hard violations, soft penalties)

This distinction enables two complementary diversity metrics that measure population health at different levels:

- **Genotypic diversity**: How different are the chromosome structures?
- **Phenotypic diversity**: How different are the solution qualities?

**Key insight**: Two chromosomes can be genotypically different (different timeslots/rooms) but phenotypically identical (same fitness), or vice versa. Understanding this relationship is critical for preventing premature convergence and maintaining search space exploration.

---

## Part 1: Conceptual Framework

### 1.1 The Genotype-Phenotype Mapping

```
┌─────────────────────────────────────────────────────────────┐
│                    GENOTYPE (Search Space)                   │
│                                                              │
│  Individual = List[SessionGene]                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Gene 1: ENME103 | Dr.A | BAE2 | R101 | [0,1,2]    │    │
│  │ Gene 2: PHYS201 | Dr.B | BAE4 | R202 | [8,9,10]   │    │
│  │ Gene 3: MATH101 | Dr.C | BAE6 | R301 | [16,17,18] │    │
│  │ ...                                                 │    │
│  └────────────────────────────────────────────────────┘    │
│                            ↓                                │
│                  (EVALUATION FUNCTION)                       │
│                   - Check 7 hard constraints                │
│                   - Check 4 soft constraints                │
│                            ↓                                │
└──────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│                  PHENOTYPE (Solution Space)                  │
│                                                              │
│  Fitness = (hard_violations, soft_penalties)                │
│  Example: (5, 120.5)                                        │
│    - 5 hard constraint violations                           │
│    - 120.5 soft penalty points                              │
│                                                              │
│  Observable Qualities:                                       │
│    ✓ Feasibility (hard violations = 0?)                     │
│    ✓ Quality (soft penalties)                               │
│    ✓ Usability (instructor preferences)                     │
│    ✓ Compactness (schedule spread)                          │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Biological Analogy

**Genetics (Biology)**:
- **Genotype**: DNA sequence (ATCG bases)
- **Phenotype**: Observable traits (eye color, height)
- **Mapping**: Genotype + Environment → Phenotype

**Evolutionary Algorithm (Our System)**:
- **Genotype**: Chromosome (`SessionGene` list)
- **Phenotype**: Fitness values (violation counts)
- **Mapping**: Genotype + Constraints → Phenotype

**Key parallel**: Same DNA (genotype) can produce different traits (phenotype) in different environments. Similarly, same chromosome can have different fitness in different constraint contexts.

---

## Part 2: Genotype in Schedule Engine

### 2.1 Data Structure

**File**: `src/ga/sessiongene.py`

```python
@dataclass
class SessionGene:
    """
    Single gene representing one scheduled session.
    This is the GENOTYPE - the genetic encoding of a solution.
    """
    course_id: str          # e.g., "ENME103"
    course_type: str        # "theory" or "practical"
    instructor_id: str      # e.g., "Dr. Smith"
    group_ids: List[str]    # e.g., ["BAE2", "BAE4"]
    room_id: str            # e.g., "R101"
    quanta: List[int]       # e.g., [0, 1, 2] (time slots)
```

**Individual (Chromosome)**:
```python
# Type alias
Individual = List[SessionGene]

# Example: 60-session timetable
individual = [
    SessionGene("ENME103", "theory", "Dr.A", ["BAE2"], "R101", [0,1,2]),
    SessionGene("PHYS201", "theory", "Dr.B", ["BAE4"], "R202", [8,9,10]),
    # ... 58 more genes
]
```

### 2.2 Genotypic Properties

**Properties defined by genotype**:
1. **Course assignment**: Which course is scheduled
2. **Instructor assignment**: Who teaches it
3. **Group assignment**: Which students attend
4. **Room assignment**: Where it happens
5. **Time assignment**: When it happens (quanta)

**Genotype space size**:
```
Total possible genotypes = (I × G × R × T)^N

Where:
  I = number of instructors per course (~2-3)
  G = number of group combinations (~10-20)
  R = number of rooms (~20-30)
  T = number of time slots (~44 quanta / day)
  N = number of sessions (~60)

Example: (3 × 15 × 25 × 44)^60 ≈ 10^250 possible chromosomes
```

**Genotypic search space** is **astronomical** - exhaustive search impossible, hence metaheuristics.

### 2.3 Genotypic Operations

**Genetic operators modify genotype directly**:

```python
# CROSSOVER: Exchange gene segments between parents
def crossover_course_group_aware(parent1, parent2):
    """Recombine genotypes at course boundaries."""
    # Create offspring by swapping course groups
    offspring1 = parent1[:split_point] + parent2[split_point:]
    offspring2 = parent2[:split_point] + parent1[split_point:]
    return offspring1, offspring2

# MUTATION: Randomly alter individual genes
def mutate_timeslot(individual):
    """Change time quantum of random gene."""
    gene = random.choice(individual)
    gene.quanta = [random_quantum()]  # Genotype modification
    return individual

def mutate_room(individual):
    """Change room of random gene."""
    gene = random.choice(individual)
    gene.room_id = random_room()  # Genotype modification
    return individual
```

**Key point**: Operators work on **genotype** (chromosome structure), but selection is based on **phenotype** (fitness values).

---

## Part 3: Phenotype in Schedule Engine

### 3.1 Fitness Function

**File**: `src/ga/evaluator/fitness.py`

```python
def evaluate(individual, courses, instructors, groups, rooms):
    """
    Map GENOTYPE to PHENOTYPE.
    
    Takes chromosome (List[SessionGene]) and evaluates it against
    constraints to produce fitness tuple (hard_violations, soft_penalties).
    
    This is the genotype-phenotype transformation function.
    """
    hard_violations = 0
    soft_penalties = 0.0
    
    # Hard constraints (MUST satisfy)
    hard_violations += check_room_capacity(individual, rooms)
    hard_violations += check_instructor_conflicts(individual)
    hard_violations += check_room_conflicts(individual)
    hard_violations += check_student_conflicts(individual, groups)
    hard_violations += check_room_exclusivity(individual)
    hard_violations += check_instructor_workload(individual)
    hard_violations += check_weekly_constraints(individual)
    
    # Soft constraints (PREFER to satisfy)
    soft_penalties += check_room_preferences(individual, courses, rooms)
    soft_penalties += check_time_preferences(individual, instructors)
    soft_penalties += check_compactness(individual)
    soft_penalties += check_isolated_lectures(individual)
    
    # Phenotype = fitness tuple
    return (-hard_violations, -soft_penalties)  # Negative for minimization
```

### 3.2 Phenotypic Properties

**Phenotype measures observable solution quality**:

1. **Feasibility**: `hard_violations == 0` (legally schedulable?)
2. **Quality**: `soft_penalties` (how good is it?)
3. **Usability**: Instructor preferences satisfied
4. **Compactness**: Schedule density (no gaps)
5. **Isolation**: No single-lecture days

**Phenotype space**:
```
Phenotype = ℝ² (two-dimensional real space)

Domain:
  hard_violations ∈ [0, ∞)    (non-negative integers)
  soft_penalties ∈ [0, ∞)     (non-negative reals)

Objective:
  Minimize both components (Pareto optimization)
```

### 3.3 Genotype-Phenotype Relationship

**Many-to-one mapping** (crucial concept):

```
Multiple genotypes → Same phenotype

Example:
┌─────────────────────────────────────────┐
│ Genotype 1:                             │
│   ENME103 | Dr.A | BAE2 | R101 | [0,1]  │  ───┐
│   PHYS201 | Dr.B | BAE4 | R202 | [8,9]  │     │
└─────────────────────────────────────────┘     ├──→ Fitness = (5, 120.5)
                                                  │
┌─────────────────────────────────────────┐     │
│ Genotype 2:                             │     │
│   ENME103 | Dr.A | BAE2 | R301 | [2,3]  │  ───┘
│   PHYS201 | Dr.B | BAE4 | R102 | [10,11]│
└─────────────────────────────────────────┘
  (Different rooms/times but SAME violations!)
```

**Why many-to-one?**
- Constraint satisfaction is **invariant** to certain genotype changes
- Example: Swapping rooms that are both acceptable → same violations
- Example: Moving to different timeslots with no conflicts → same fitness

**Implication for search**:
- **Genotypic diversity** can be HIGH while **phenotypic diversity** is LOW
- Population can explore many chromosomes that map to same fitness
- This is **neutral evolution** - genetic drift without fitness change
- Can enable **escaping local optima** via neutral networks

---

## Part 4: Genotypic Diversity

### 4.1 Definition

**Genotypic diversity** measures how different chromosomes are at the **genetic level** (structure), regardless of their fitness.

**File**: `src/rl/gym_env/state_encoder.py`

```python
def _calculate_genotype_diversity(self, population: List[Individual]) -> float:
    """
    Calculate genotype diversity (unique chromosome structures).

    Measures diversity at the GENETIC level by counting unique
    timeslot/room assignment pairs across the population.
    
    Returns:
        Float in [0, 1] where:
          0.0 = All chromosomes identical (clones)
          1.0 = Maximum structural diversity
    """
    if not population:
        return 0.0

    # Count unique timeslot/room assignments
    unique_assignments = set()
    for ind in population:
        for gene in ind:
            # Extract genotypic features (structure)
            quanta = tuple(sorted(getattr(gene, "quanta", [])))
            unique_assignments.add((quanta, gene.room_id))
    
    # Normalize by population size × chromosome length
    max_diversity = len(population) * len(population[0])
    return len(unique_assignments) / max(max_diversity, 1)
```

### 4.2 Calculation Method

**Step-by-step example**:

```python
# Population of 3 individuals, 2 genes each
pop = [
    [SessionGene("C1", "theory", "A", ["G1"], "R101", [0,1]),
     SessionGene("C2", "theory", "B", ["G2"], "R202", [8,9])],
    
    [SessionGene("C1", "theory", "A", ["G1"], "R101", [0,1]),  # Same as ind1
     SessionGene("C2", "theory", "B", ["G2"], "R303", [10,11])],  # Different room/time
    
    [SessionGene("C1", "theory", "A", ["G1"], "R404", [2,3]),  # Different room/time
     SessionGene("C2", "theory", "B", ["G2"], "R505", [12,13])]  # Different room/time
]

# Extract unique (quanta, room) pairs
unique_assignments = {
    ((0, 1), "R101"),   # Gene 1, Individual 1
    ((8, 9), "R202"),   # Gene 2, Individual 1
    ((0, 1), "R101"),   # Gene 1, Individual 2 (duplicate)
    ((10, 11), "R303"), # Gene 2, Individual 2
    ((2, 3), "R404"),   # Gene 1, Individual 3
    ((12, 13), "R505")  # Gene 2, Individual 3
}
# After deduplication: 5 unique assignments

# Normalize
max_diversity = 3 (individuals) × 2 (genes) = 6
genotype_diversity = 5 / 6 = 0.833
```

### 4.3 Interpretation

**High genotypic diversity (≥ 0.7)**:
- ✅ Population explores diverse regions of search space
- ✅ Many different chromosome structures present
- ✅ Good for early exploration (generations 0-500)
- ⚠️ May include many infeasible solutions

**Low genotypic diversity (≤ 0.3)**:
- ⚠️ Population converging to similar structures
- ⚠️ Risk of premature convergence (stuck in local optimum)
- ✅ OK if phenotypic diversity is HIGH (converging to Pareto front)
- ❌ BAD if both genotypic & phenotypic diversity are LOW (stagnation)

**Typical trajectory**:
```
Generation    Genotype Diversity    Phenotype Diversity
─────────────────────────────────────────────────────────
0-100         0.95 (very high)      0.85 (high)
100-500       0.75 (high)           0.60 (moderate)
500-1000      0.50 (moderate)       0.40 (low)
1000-2000     0.30 (low)            0.15 (very low)
```

### 4.4 Use Cases

**1. Detecting premature convergence**:
```python
if genotype_diversity < 0.2 and current_generation < 500:
    print("⚠️ WARNING: Premature convergence detected")
    apply_diversity_preserving_operators()
```

**2. Adaptive operator selection (RL agent)**:
```python
if genotype_diversity < threshold:
    # Favor exploration operators
    return [mutate_adaptive, mutate_swap_sessions, mutate_timeslot]
else:
    # Favor exploitation operators
    return [local_search_hill_climbing, repair_igls]
```

**3. Population restart**:
```python
if genotype_diversity < 0.1 and no_improvement_for > 100:
    # Population collapsed - reinitialize
    population = generate_new_population(elite=top_5_individuals)
```

---

## Part 5: Phenotypic Diversity

### 5.1 Definition

**Phenotypic diversity** measures how different solutions are at the **fitness level** (quality), regardless of their chromosome structure.

**File**: `src/rl/gym_env/state_encoder.py`

```python
def _calculate_phenotype_diversity(self, population: List[Individual]) -> float:
    """
    Calculate phenotype diversity (unique fitness outcomes).

    Measures diversity at the SOLUTION level by analyzing how different
    individuals are in terms of their evaluated fitness. Uses normalized
    pairwise distances in fitness space.
    
    Returns:
        Float in [0, 1] where:
          0.0 = All individuals have identical fitness
          1.0 = Maximum spread in fitness space (diverse Pareto front)
    """
    if len(population) < 2:
        return 0.0

    # Extract fitness vectors (hard, soft)
    fitness_array = np.array([ind.fitness.values for ind in population])
    
    # Calculate pairwise Euclidean distances in fitness space
    from scipy.spatial.distance import pdist
    distances = pdist(fitness_array, metric='euclidean')
    
    # Average distance normalized by population size
    if len(distances) == 0:
        return 0.0
    
    avg_distance = np.mean(distances)
    
    # Normalize by fitness range to get [0, 1] scale
    fitness_range = np.max(fitness_array) - np.min(fitness_array)
    if fitness_range < 1e-6:
        return 0.0  # All fitness identical
    
    return min(avg_distance / (fitness_range + 1e-6), 1.0)
```

### 5.2 Calculation Method

**Fitness space geometry**:

```
2D Fitness Space (hard violations, soft penalties)

      Soft Penalties (y-axis)
      ↑
  500 │  ×
      │      ×
  400 │          ×
      │              ×       ← High diversity: points spread out
  300 │                  ×
      │
  200 │    ○ ○ ○           ← Low diversity: points clustered
  100 │    ○ ○ ○
      │
    0 └─────────────────────────→ Hard Violations (x-axis)
      0   10   20   30   40   50
```

**Step-by-step example**:

```python
# Population of 4 individuals
pop = [
    Individual_1: fitness = (0, 100.0)    # Feasible, good quality
    Individual_2: fitness = (0, 105.0)    # Feasible, similar quality
    Individual_3: fitness = (5, 200.0)    # Infeasible, moderate quality
    Individual_4: fitness = (50, 500.0)   # Very infeasible, poor quality
]

# Fitness array (2D coordinates)
fitness_array = np.array([
    [0, 100.0],
    [0, 105.0],
    [5, 200.0],
    [50, 500.0]
])

# Pairwise distances (6 pairs for 4 individuals)
distances = [
    distance(1, 2) = sqrt((0-0)^2 + (100-105)^2) = 5.0
    distance(1, 3) = sqrt((0-5)^2 + (100-200)^2) = 100.12
    distance(1, 4) = sqrt((0-50)^2 + (100-500)^2) = 403.11
    distance(2, 3) = sqrt((0-5)^2 + (105-200)^2) = 95.13
    distance(2, 4) = sqrt((0-50)^2 + (105-500)^2) = 398.15
    distance(3, 4) = sqrt((5-50)^2 + (200-500)^2) = 303.35
]

# Average distance
avg_distance = mean(distances) = 217.47

# Fitness range (normalization factor)
fitness_range = max(fitness_array) - min(fitness_array)
             = max(50, 500) - min(0, 100)
             = 500 - 0 = 500

# Phenotype diversity
phenotype_diversity = avg_distance / fitness_range
                    = 217.47 / 500
                    = 0.435 (moderate diversity)
```

### 5.3 Interpretation

**High phenotypic diversity (≥ 0.6)**:
- ✅ Population covers wide range of fitness values
- ✅ Diverse Pareto front (many non-dominated solutions)
- ✅ Good for multi-objective optimization
- ✅ Indicates active search (not converged)

**Low phenotypic diversity (≤ 0.2)**:
- ✅ Population converging to similar fitness (good if near optimum)
- ⚠️ May indicate convergence to local optimum
- ⚠️ All individuals have similar quality
- ✅ Expected in late stages (generations 1500-2000)

**Relationship to Pareto front**:
```
High phenotypic diversity = Wide Pareto front coverage

      Soft Penalties
      ↑
      │  ×
      │      ×
      │          ×
      │              ×
      │                  ×  ← Well-spread Pareto front (HIGH diversity)
      │
      └─────────────────────────→ Hard Violations


      Soft Penalties
      ↑
      │
      │
      │
      │
      │  ○ ○ ○ ○ ○              ← Converged to single point (LOW diversity)
      │
      └─────────────────────────→ Hard Violations
```

### 5.4 Use Cases

**1. Detecting convergence**:
```python
if phenotype_diversity < 0.1:
    print("✓ Population converged to similar fitness")
    if best_fitness < threshold:
        print("✓ Convergence to good optimum")
    else:
        print("⚠️ Convergence to poor optimum (restart?)")
```

**2. NSGA-II selection pressure**:
```python
# Phenotypic diversity influences selection
if phenotype_diversity < 0.3:
    # Increase mutation rate to escape local optimum
    mutpb *= 1.5
else:
    # Normal mutation rate
    mutpb = default_mutpb
```

**3. RL reward shaping**:
```python
# Reward agent for maintaining diversity
reward = (
    -delta_hard * 10.0 +           # Primary objective
    -delta_soft * 1.0 +            # Secondary objective
    phenotype_diversity * 0.5      # Diversity bonus
)
```

---

## Part 6: Genotypic vs Phenotypic Diversity

### 6.1 Key Differences

| Aspect | **Genotypic Diversity** | **Phenotypic Diversity** |
|--------|-------------------------|--------------------------|
| **Measures** | Chromosome structure | Fitness values |
| **Space** | Search space (genotype) | Solution space (phenotype) |
| **Dimension** | High-dimensional (5N features) | 2D (hard, soft) |
| **Calculation** | Unique (quanta, room) pairs | Fitness distance |
| **Interpretation** | Structural variety | Quality variety |
| **Early GA** | High (0.8-0.95) | High (0.6-0.85) |
| **Late GA** | Low (0.2-0.4) | Low (0.1-0.3) |
| **Ideal** | Moderate (0.5-0.7) | Moderate (0.4-0.6) |

### 6.2 Four Scenarios

```
┌──────────────────────────┬──────────────────────────┐
│   High Genotypic         │   High Genotypic         │
│   High Phenotypic        │   Low Phenotypic         │
│                          │                          │
│  ✅ IDEAL EXPLORATION    │  ⚠️ NEUTRAL EVOLUTION     │
│  - Early generations     │  - Many structures       │
│  - Diverse structures    │  - Same fitness          │
│  - Diverse fitness       │  - Genetic drift         │
│  - Active search         │  - May escape local opt  │
├──────────────────────────┼──────────────────────────┤
│   Low Genotypic          │   Low Genotypic          │
│   High Phenotypic        │   Low Phenotypic         │
│                          │                          │
│  ✅ GOOD CONVERGENCE     │  ❌ PREMATURE CONVERGE   │
│  - Late generations      │  - Stuck in local opt    │
│  - Similar structures    │  - All clones            │
│  - Diverse Pareto front  │  - Same fitness          │
│  - Exploitation phase    │  - RESTART NEEDED        │
└──────────────────────────┴──────────────────────────┘
```

### 6.3 Detailed Scenario Analysis

**Scenario 1: High-High (Exploration)**
```python
genotype_diversity = 0.85
phenotype_diversity = 0.70

# Interpretation:
# - Population explores many different chromosome structures
# - These structures produce diverse fitness outcomes
# - Healthy exploration phase (typical: gen 0-500)

# Action:
# - Continue normal GA operations
# - No intervention needed
```

**Scenario 2: High-Low (Neutral Evolution)**
```python
genotype_diversity = 0.75
phenotype_diversity = 0.15

# Interpretation:
# - Many different chromosomes exist
# - BUT they all have similar fitness (many-to-one mapping)
# - Neutral networks: genetic drift without fitness change
# - Can enable escaping local optima (good!)
# - OR wasting computation on redundant structures (bad!)

# Action (depends on context):
if current_generation < 500:
    # Early: Neutral evolution is GOOD (exploration via drift)
    continue_normal_operations()
else:
    # Late: Wasteful - guide toward better regions
    apply_guided_mutation()
    increase_selection_pressure()
```

**Scenario 3: Low-High (Good Convergence)**
```python
genotype_diversity = 0.30
phenotype_diversity = 0.50

# Interpretation:
# - Chromosomes becoming structurally similar
# - BUT still covering diverse Pareto front
# - Exploitation phase: refining solutions in good region
# - IDEAL for multi-objective optimization

# Action:
# - Continue exploitation
# - Apply local search operators
# - Maintain Pareto front diversity (NSGA-II helps here)
```

**Scenario 4: Low-Low (Premature Convergence)**
```python
genotype_diversity = 0.10
phenotype_diversity = 0.05

# Interpretation:
# - All chromosomes are clones (or near-clones)
# - All have identical fitness
# - Population collapsed to single point
# - CRITICAL: Stuck in local optimum (if early gen)

# Action (depends on best fitness):
if best_hard_violations == 0 and best_soft_penalties < threshold:
    # Converged to GOOD optimum - SUCCESS!
    terminate_ga()
elif current_generation < 1000:
    # Converged EARLY to poor optimum - RESTART
    population = reinitialize_with_elite(top_5)
else:
    # Late convergence to poor optimum - try local search
    apply_intensive_local_search()
```

### 6.4 Correlation Analysis

**Typical correlation over time**:

```
Generation 0-500 (Exploration):
  Genotype ≈ 0.85, Phenotype ≈ 0.70
  Correlation: +0.8 (both high, strong positive)

Generation 500-1000 (Transition):
  Genotype ≈ 0.55, Phenotype ≈ 0.45
  Correlation: +0.6 (both decreasing together)

Generation 1000-1500 (Exploitation):
  Genotype ≈ 0.35, Phenotype ≈ 0.50
  Correlation: -0.2 (weak negative, phenotype higher)

Generation 1500-2000 (Convergence):
  Genotype ≈ 0.15, Phenotype ≈ 0.20
  Correlation: +0.9 (both low, converging to optimum)
```

**Expected relationship**:
- **Positive correlation (early)**: Both high → healthy exploration
- **Weak correlation (mid)**: Decoupling as search focuses
- **Positive correlation (late)**: Both low → convergence

**Anomaly detection**:
- **Negative correlation (early)**: Unusual - investigate constraint structure
- **High genotype, low phenotype (late)**: Wasted diversity - increase selection pressure

---

## Part 7: Implementation in Code

### 7.1 Diversity Calculation Pipeline

**File**: `src/rl/gym_env/state_encoder.py`

```python
class StateEncoder:
    """
    Encodes GA population state into RL observation vector.
    
    Includes both genotypic and phenotypic diversity metrics
    for comprehensive population health monitoring.
    """
    
    def encode(
        self,
        population: List[Individual],
        current_generation: int,
        generations_without_improvement: int,
    ) -> np.ndarray:
        """
        Encode population into 22D observation vector.
        
        Features include (index: description):
          [0]: best_fitness         - Best individual fitness
          [1]: avg_fitness          - Population average fitness
          [2]: worst_fitness        - Worst individual fitness
          [3]: fitness_std          - Fitness standard deviation
          [4]: fitness_range        - Max - Min fitness
          [5]: population_diversity - Overall diversity (legacy)
          [6]: genotype_diversity   - 🧬 Chromosome structure diversity
          [7]: phenotype_diversity  - 🎯 Fitness outcome diversity
          [8]: fitness_diversity    - Coefficient of variation
          [9]: unique_fitness_ratio - % unique fitness values
          [10]: current_generation  - Normalized generation counter
          [11]: gens_no_improve     - Stagnation counter
          [12]: convergence_rate    - Population uniformity
          [13]: improvement_rate    - Recent fitness improvement
          [14]: avg_hard_violations - Average hard constraint violations
          [15]: avg_soft_violations - Average soft penalties
          [16]: violation_std       - Violation standard deviation
          [17-21]: heuristic_history - Recent operator applications
        """
        # Extract raw features
        features = self._extract_features(
            population, current_generation, generations_without_improvement
        )
        
        # Normalize to [0, 1] or [-1, 1]
        normalized = self._normalize_features(features)
        
        # Add history
        obs = np.concatenate([normalized, features.recent_heuristic_ids])
        
        return obs.astype(np.float32)
    
    def _extract_features(
        self,
        population: List[Individual],
        current_generation: int,
        generations_without_improvement: int,
    ) -> StateFeatures:
        """Extract raw feature values from population."""
        if not population:
            return self._get_zero_features()

        # Fitness statistics
        hard_violations = np.array([ind.fitness.values[0] for ind in population])
        soft_violations = np.array([ind.fitness.values[1] for ind in population])
        fitness_values = hard_violations * 100 + soft_violations

        # Diversity metrics (THE KEY METRICS)
        population_diversity = self._calculate_diversity(population)
        genotype_diversity = self._calculate_genotype_diversity(population)  # 🧬
        phenotype_diversity = self._calculate_phenotype_diversity(population)  # 🎯
        fitness_diversity = np.std(fitness_values) / (np.mean(fitness_values) + 1e-6)
        unique_fitness_ratio = self._calculate_unique_fitness_ratio(population)

        # ... (other features)
        
        return StateFeatures(
            # ... (other fields)
            genotype_diversity=genotype_diversity,
            phenotype_diversity=phenotype_diversity,
            # ... (other fields)
        )
```

### 7.2 Integration with RL Agent

**File**: `src/rl/gym_env/schedule_env.py`

```python
class ScheduleEnv(gym.Env):
    """
    Gymnasium environment for RL-guided hyper-heuristics.
    
    The RL agent observes genotypic and phenotypic diversity
    to decide which heuristic operators to apply.
    """
    
    def reset(self):
        """Reset environment and return initial observation."""
        # Initialize GA scheduler
        self.scheduler = GAScheduler(self.config)
        self.scheduler.initialize_population()
        
        # Encode initial state (includes diversity metrics)
        obs = self.encoder.encode(
            self.scheduler.population,
            current_generation=0,
            generations_without_improvement=0
        )
        
        return obs, {}
    
    def step(self, action: int):
        """
        Apply heuristic operator and evolve one generation.
        
        The RL agent learns to select operators based on:
        - Current diversity (genotypic & phenotypic)
        - Convergence rate
        - Recent improvement
        
        Example decision logic (learned by PPO):
        - If genotype_diversity < 0.3: Apply mutation (increase diversity)
        - If phenotype_diversity < 0.2: Apply local search (exploit)
        - If both high: Apply crossover (continue exploration)
        """
        # Map action to operator
        operator_name = self.ACTION_MAP[action]
        
        # Apply operator to population
        self.scheduler.apply_operator(operator_name)
        
        # Evolve one generation (selection + reproduction)
        self.scheduler.evolve_one_generation()
        
        # Calculate reward (considers diversity maintenance)
        reward = self._calculate_reward()
        
        # Encode new state (updated diversity metrics)
        obs = self.encoder.encode(
            self.scheduler.population,
            current_generation=self.current_step,
            generations_without_improvement=self.gens_no_improve
        )
        
        # Check termination
        done = (
            self.current_step >= self.max_steps or
            self.scheduler.found_feasible_solution()
        )
        
        return obs, reward, done, False, {}
```

### 7.3 Reward Function Integration

**File**: `src/rl/rewards/base_reward.py`

```python
def _calculate_reward(self) -> float:
    """
    Multi-objective reward with diversity bonus.
    
    Encourages agent to:
    1. Reduce hard violations (primary)
    2. Reduce soft penalties (secondary)
    3. Maintain diversity (exploration bonus)
    """
    # Primary objectives
    delta_hard = self.prev_hard - self.current_hard
    delta_soft = self.prev_soft - self.current_soft
    
    hard_reward = delta_hard * 10.0   # α = 10
    soft_reward = delta_soft * 1.0    # β = 1
    
    # Diversity bonus (uses phenotypic diversity)
    phenotype_diversity = self.encoder._calculate_phenotype_diversity(
        self.scheduler.population
    )
    diversity_bonus = 0.5 if phenotype_diversity > 0.6 else 0.0  # γ = 0.5
    
    # Penalties
    feasibility_bonus = 50.0 if self.current_hard == 0 else 0.0
    time_penalty = -0.01 * self.current_step
    
    return (
        hard_reward +
        soft_reward +
        diversity_bonus +      # PHENOTYPIC diversity encouraged
        feasibility_bonus +
        time_penalty
    )
```

**Key insight**: Reward function uses **phenotypic diversity** to encourage maintaining solution quality variety, not just chromosome variety.

---

## Part 8: Practical Examples

### 8.1 Example 1: Early Generation (Exploration)

**Generation 50 / 2000**

```python
# Population snapshot
population = [
    # Individual 1: Genotype
    [SessionGene("ENME103", "theory", "Dr.A", ["BAE2"], "R101", [0,1,2]),
     SessionGene("PHYS201", "theory", "Dr.B", ["BAE4"], "R202", [8,9,10]),
     # ... 58 more genes
    ],
    # Fitness (phenotype)
    fitness = (25, 450.5)  # 25 hard violations, 450.5 soft penalties
    
    # Individual 2: Genotype (DIFFERENT structure)
    [SessionGene("ENME103", "theory", "Dr.C", ["BAE2"], "R303", [5,6,7]),
     SessionGene("PHYS201", "theory", "Dr.D", ["BAE6"], "R404", [15,16,17]),
     # ... 58 more genes
    ],
    # Fitness (phenotype)
    fitness = (30, 380.2)  # Different fitness (more diverse)
    
    # ... 98 more individuals with varied genotypes & phenotypes
]

# Diversity calculations
genotype_diversity = 0.87   # HIGH: Many unique (quanta, room) pairs
phenotype_diversity = 0.72  # HIGH: Fitness values spread out
unique_fitness_ratio = 0.95 # 95% of individuals have unique fitness

# Interpretation
print("✅ Healthy exploration phase")
print("   - High genotypic diversity: exploring many structures")
print("   - High phenotypic diversity: diverse solution qualities")
print("   - Ideal for early generations (gen 0-500)")

# RL agent decision (learned policy)
if genotype_diversity > 0.7 and phenotype_diversity > 0.6:
    # Continue exploration
    selected_operator = "crossover_course_group_aware"
else:
    # Shouldn't happen this early, but fallback to mutation
    selected_operator = "mutate_adaptive"
```

### 8.2 Example 2: Mid Generation (Neutral Evolution)

**Generation 800 / 2000**

```python
# Population snapshot (3 individuals shown for brevity)
population = [
    # Individual 1
    [SessionGene("ENME103", "theory", "Dr.A", ["BAE2"], "R101", [0,1,2]),
     SessionGene("PHYS201", "theory", "Dr.B", ["BAE4"], "R202", [8,9,10]),
     # ... 58 more
    ],
    fitness = (5, 120.5)  # Good fitness
    
    # Individual 2 (DIFFERENT genotype)
    [SessionGene("ENME103", "theory", "Dr.A", ["BAE2"], "R303", [2,3,4]),  # Different room/time
     SessionGene("PHYS201", "theory", "Dr.B", ["BAE4"], "R404", [10,11,12]),  # Different room/time
     # ... 58 more
    ],
    fitness = (5, 120.5)  # SAME fitness (neutral mutation!)
    
    # Individual 3 (DIFFERENT genotype again)
    [SessionGene("ENME103", "theory", "Dr.A", ["BAE2"], "R505", [5,6,7]),
     SessionGene("PHYS201", "theory", "Dr.B", ["BAE4"], "R606", [15,16,17]),
     # ... 58 more
    ],
    fitness = (5, 120.8)  # Almost same fitness
    
    # ... 97 more individuals (many with fitness ≈ (5, 120))
]

# Diversity calculations
genotype_diversity = 0.65   # MODERATE: Still some structural variety
phenotype_diversity = 0.18  # LOW: Most fitness values clustered around (5, 120)
unique_fitness_ratio = 0.35 # Only 35% unique fitness (lots of duplicates)

# Interpretation
print("⚠️ Neutral evolution detected")
print("   - Genotype diversity > Phenotype diversity (unusual)")
print("   - Many different chromosomes → same fitness")
print("   - Genetic drift on neutral networks")
print("   - May enable escaping local optimum (good!)")
print("   - OR wasting computation (bad!)")

# RL agent decision (learned policy)
if genotype_diversity > 0.5 and phenotype_diversity < 0.3:
    # High genotype, low phenotype → neutral evolution
    # Guide toward better fitness regions
    selected_operator = "local_search_hill_climbing"  # Exploit current region
else:
    selected_operator = "mutate_swap_sessions"  # Continue exploration
```

### 8.3 Example 3: Late Generation (Good Convergence)

**Generation 1800 / 2000**

```python
# Population snapshot (converging to Pareto front)
population = [
    # Individual 1: Best hard violations, moderate soft
    [SessionGene("ENME103", "theory", "Dr.A", ["BAE2"], "R101", [0,1,2]),
     # ... optimized structure
    ],
    fitness = (0, 150.0)  # Feasible!
    
    # Individual 2: Good balance
    [SessionGene("ENME103", "theory", "Dr.A", ["BAE2"], "R101", [0,1,2]),  # Similar structure
     # ... slightly different optimized structure
    ],
    fitness = (0, 145.2)  # Feasible, better soft
    
    # Individual 3: Best soft penalties, minimal hard
    [SessionGene("ENME103", "theory", "Dr.A", ["BAE2"], "R202", [2,3,4]),  # Slightly different
     # ... highly optimized structure
    ],
    fitness = (0, 132.5)  # Feasible, best soft so far!
    
    # ... 97 more individuals, all with fitness ≈ (0, 130-150)
]

# Diversity calculations
genotype_diversity = 0.25   # LOW: Converged to similar chromosome structures
phenotype_diversity = 0.45  # MODERATE: Pareto front has decent spread
unique_fitness_ratio = 0.80 # 80% unique fitness (good for Pareto front)

# Interpretation
print("✅ Good convergence to Pareto front")
print("   - Low genotype diversity: structures converging (expected)")
print("   - Moderate phenotype diversity: Pareto front coverage (good!)")
print("   - All solutions feasible (hard violations = 0)")
print("   - Exploiting good region of search space")

# RL agent decision (learned policy)
if genotype_diversity < 0.3 and phenotype_diversity > 0.3:
    # Low genotype, moderate phenotype → good Pareto convergence
    # Refine solutions with local search
    selected_operator = "repair_selective"  # Fine-tune soft penalties
else:
    selected_operator = "local_search_simulated_annealing"  # Escape local opt
```

### 8.4 Example 4: Premature Convergence (Bad)

**Generation 300 / 2000** (TOO EARLY!)

```python
# Population snapshot (collapsed to single point)
population = [
    # Individual 1
    [SessionGene("ENME103", "theory", "Dr.A", ["BAE2"], "R101", [0,1,2]),
     # ... specific structure
    ],
    fitness = (15, 250.0)  # Sub-optimal (not feasible)
    
    # Individual 2 (near-clone)
    [SessionGene("ENME103", "theory", "Dr.A", ["BAE2"], "R101", [0,1,2]),  # Same structure
     # ... almost identical
    ],
    fitness = (15, 250.1)  # Almost identical fitness
    
    # ... 98 more individuals, all clones with fitness ≈ (15, 250)
]

# Diversity calculations
genotype_diversity = 0.08   # VERY LOW: All chromosomes are clones
phenotype_diversity = 0.02  # VERY LOW: All fitness identical
unique_fitness_ratio = 0.10 # Only 10% unique fitness (90% duplicates!)

# Interpretation
print("❌ PREMATURE CONVERGENCE DETECTED")
print("   - Both diversities very low (< 0.1)")
print("   - Population collapsed to local optimum")
print("   - Still early in evolution (gen 300 / 2000)")
print("   - Fitness not optimal (15 violations)")
print("   - ACTION REQUIRED: Restart or diversity injection")

# RL agent decision (learned policy)
if genotype_diversity < 0.2 and phenotype_diversity < 0.2 and current_gen < 1000:
    # CRITICAL: Premature convergence
    # Apply drastic diversity-preserving operators
    selected_operator = "mutate_adaptive"  # Strong mutation
    
    # OR trigger population restart (external mechanism)
    if gens_no_improve > 50:
        trigger_population_restart(elite=top_10_individuals)
```

---

## Part 9: Relation to Research Literature

### 9.1 Genotype-Phenotype Distinction

**Seminal papers**:
- **Holland (1975)**: "Adaptation in Natural and Artificial Systems"
  - Introduced genotype-phenotype mapping in GAs
  - Schema theorem: genotype patterns propagate based on phenotype fitness

- **Goldberg (1989)**: "Genetic Algorithms in Search, Optimization, and Machine Learning"
  - Formalized building block hypothesis
  - Emphasized importance of genotype diversity for exploration

- **Burke et al. (2013)**: "Hyper-heuristics: A survey of the state of the art"
  - Genotypic operators (low-level heuristics) selected by high-level strategy
  - Our RL agent = high-level strategy, GA operators = low-level heuristics

### 9.2 Diversity Metrics

**Key references**:
- **Deb et al. (2002)**: "A fast and elitist multi-objective genetic algorithm: NSGA-II"
  - Crowding distance = phenotypic diversity measure
  - Maintains Pareto front spread (phenotype space)

- **Burke et al. (2010)**: "Exploring hyper-heuristic methodologies with genetic programming"
  - Diversity-preserving operators for hyper-heuristics
  - Balance exploration (genotype) vs exploitation (phenotype)

- **Toffolo and Benini (2003)**: "Genetic diversity as an objective in university timetabling"
  - Explicitly optimize diversity as constraint
  - Prevent premature convergence in timetabling domain

### 9.3 Neutral Evolution

**Neutral networks** (Kimura 1983):
- Many genotypes → same phenotype (fitness)
- Genetic drift without selection pressure
- Enables escaping local optima via "stepping stones"

**Application to GAs**:
- **Toussaint and Igel (2002)**: "Neutrality: A necessity for self-adaptation"
  - Neutral mutations = beneficial for long-term adaptation
  - High genotypic diversity + low phenotypic diversity = neutral networks

**In our system**:
- Scenario 2 (Example 8.2): High genotype, low phenotype diversity
- Indicates neutral evolution on room/time assignments
- Can help escape constraint-based local optima

---

## Part 10: Advanced Topics

### 10.1 Adaptive Diversity Control

**Dynamic diversity targets**:
```python
def get_diversity_target(current_generation: int, max_generations: int) -> float:
    """
    Calculate target diversity based on GA stage.
    
    Early: High diversity (exploration)
    Mid: Moderate diversity (transition)
    Late: Low diversity (exploitation)
    """
    progress = current_generation / max_generations
    
    if progress < 0.25:
        # Early: 0-500 generations
        return 0.80  # High genotype diversity target
    elif progress < 0.50:
        # Mid: 500-1000 generations
        return 0.60  # Moderate diversity
    elif progress < 0.75:
        # Late: 1000-1500 generations
        return 0.40  # Decreasing diversity
    else:
        # Final: 1500-2000 generations
        return 0.20  # Low diversity (converged)

# Use in RL reward function
diversity_target = get_diversity_target(current_gen, max_gens)
diversity_bonus = abs(genotype_diversity - diversity_target)  # Penalize deviation
reward = hard_reward + soft_reward - diversity_bonus
```

### 10.2 Multi-Level Diversity

**Hierarchical diversity metrics**:
1. **Gene-level**: Individual gene differences (finest granularity)
2. **Chromosome-level**: Individual differences (genotype diversity)
3. **Population-level**: Fitness spread (phenotype diversity)
4. **Pareto-front-level**: Non-dominated solution spread (archive diversity)

**Implementation**:
```python
def calculate_multi_level_diversity(population):
    """Calculate diversity at multiple levels."""
    
    # Level 1: Gene-level (average gene distance)
    gene_diversity = average_gene_distance(population)
    
    # Level 2: Chromosome-level (genotype)
    genotype_diversity = calculate_genotype_diversity(population)
    
    # Level 3: Population-level (phenotype)
    phenotype_diversity = calculate_phenotype_diversity(population)
    
    # Level 4: Pareto-front-level (non-dominated set)
    pareto_front = extract_pareto_front(population)
    pareto_diversity = calculate_crowding_distance(pareto_front)
    
    return {
        "gene": gene_diversity,
        "genotype": genotype_diversity,
        "phenotype": phenotype_diversity,
        "pareto": pareto_diversity
    }
```

### 10.3 Diversity-Fitness Tradeoff

**Pareto optimization**: Diversity vs Fitness

```python
# Multi-objective RL agent
# Objective 1: Minimize violations (fitness)
# Objective 2: Maximize diversity (exploration)

fitness_objective = -hard_violations * 10.0 - soft_penalties * 1.0
diversity_objective = phenotype_diversity * 5.0

# Weighted sum (learnable weights)
reward = (
    alpha * fitness_objective +
    beta * diversity_objective
)

# Alpha, beta learned by RL agent:
# Early: alpha=0.3, beta=0.7 (favor diversity)
# Late: alpha=0.9, beta=0.1 (favor fitness)
```

---

## Part 11: Debugging & Visualization

### 11.1 Logging Diversity Metrics

**File**: `src/core/ga_scheduler.py`

```python
def log_diversity_metrics(self, generation: int):
    """Log diversity metrics for debugging."""
    
    # Calculate metrics
    genotype_div = self.encoder._calculate_genotype_diversity(self.population)
    phenotype_div = self.encoder._calculate_phenotype_diversity(self.population)
    unique_ratio = self.encoder._calculate_unique_fitness_ratio(self.population)
    
    # Log to console
    console.print(
        f"[Gen {generation:4d}] "
        f"Genotype: {genotype_div:.3f} | "
        f"Phenotype: {phenotype_div:.3f} | "
        f"Unique: {unique_ratio:.3f}"
    )
    
    # Log to TensorBoard (for RL training)
    if self.tensorboard_writer:
        self.tensorboard_writer.add_scalar(
            "Diversity/Genotype", genotype_div, generation
        )
        self.tensorboard_writer.add_scalar(
            "Diversity/Phenotype", phenotype_div, generation
        )
        self.tensorboard_writer.add_scalar(
            "Diversity/UniqueRatio", unique_ratio, generation
        )
```

### 11.2 Visualization

**Plot diversity over time**:
```python
import matplotlib.pyplot as plt

def plot_diversity_evolution(log_file: str):
    """Plot genotypic and phenotypic diversity over generations."""
    
    # Load logged data
    data = pd.read_csv(log_file)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Both diversities
    ax1.plot(data["generation"], data["genotype_diversity"], 
             label="Genotypic Diversity", color="blue")
    ax1.plot(data["generation"], data["phenotype_diversity"], 
             label="Phenotypic Diversity", color="red")
    ax1.axhline(y=0.3, color="gray", linestyle="--", label="Warning Threshold")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Diversity")
    ax1.set_title("Diversity Evolution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Fitness evolution (phenotype)
    ax2.plot(data["generation"], data["best_fitness"], 
             label="Best Fitness", color="green")
    ax2.plot(data["generation"], data["avg_fitness"], 
             label="Avg Fitness", color="orange")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness")
    ax2.set_title("Fitness Evolution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("diversity_evolution.png", dpi=300)
    plt.show()
```

**Sample output**:
```
Diversity Evolution Over 2000 Generations

Genotypic Diversity (blue line):
  Starts at 0.95, decreases to 0.25
  Smooth descent (expected convergence)

Phenotypic Diversity (red line):
  Starts at 0.80, decreases to 0.20
  Follows genotypic diversity (healthy)

Warning Threshold (dashed gray):
  0.3 threshold crossed around gen 1200
  Indicates transition to exploitation phase

Fitness Evolution:
  Best fitness improves from 500 → 132
  Avg fitness converges to best around gen 1800
  Correlation with diversity decrease (good!)
```

---

## Part 12: Summary & Best Practices

### 12.1 Key Takeaways

1. **Genotype = Internal representation** (chromosome structure)
   - List of SessionGene objects
   - Defines search space
   - Modified by GA operators

2. **Phenotype = External manifestation** (fitness values)
   - Tuple of (hard_violations, soft_penalties)
   - Defines solution quality
   - Determines selection pressure

3. **Genotypic diversity = Structural variety**
   - Measures chromosome differences
   - High early, low late (expected)
   - Prevents cloning

4. **Phenotypic diversity = Quality variety**
   - Measures fitness spread
   - Important for Pareto front coverage
   - Indicates convergence state

5. **Many-to-one mapping**:
   - Multiple genotypes → same phenotype
   - Neutral evolution possible
   - Can help escape local optima

6. **Four scenarios**:
   - High-High: Healthy exploration ✅
   - High-Low: Neutral evolution ⚠️
   - Low-High: Good convergence ✅
   - Low-Low: Premature convergence (check if optimal) ❌/✅

### 12.2 Best Practices

**For GA Design**:
```python
# 1. Monitor both diversities
log_diversity_metrics(generation)

# 2. Set diversity thresholds
if genotype_diversity < 0.2 and generation < 500:
    apply_diversity_injection()

# 3. Use adaptive operators
if phenotype_diversity < 0.3:
    mutation_rate *= 1.5  # Increase exploration

# 4. Balance objectives
reward = fitness_improvement + diversity_bonus
```

**For RL Integration**:
```python
# 1. Include diversity in state
obs = [
    ...,
    genotype_diversity,  # Index 6
    phenotype_diversity, # Index 7
    unique_fitness_ratio, # Index 9
    ...
]

# 2. Reward diversity maintenance
diversity_bonus = 0.5 if phenotype_diversity > 0.6 else 0.0
reward += diversity_bonus

# 3. Learn adaptive policies
# Agent learns: "When diversity low, apply mutation"
#               "When diversity high, apply crossover"
```

**For Debugging**:
```python
# 1. Log every N generations
if generation % 10 == 0:
    log_diversity_metrics(generation)

# 2. Plot evolution curves
plot_diversity_evolution(log_file)

# 3. Detect anomalies
if genotype_diversity > 0.7 and phenotype_diversity < 0.2:
    console.print("⚠️ Neutral evolution detected")
```

### 12.3 Common Pitfalls

**Mistake 1**: Ignoring genotypic diversity
```python
# BAD: Only track phenotypic diversity
diversity = calculate_phenotype_diversity(population)

# GOOD: Track both
genotype_div = calculate_genotype_diversity(population)
phenotype_div = calculate_phenotype_diversity(population)
```

**Mistake 2**: Conflating diversity metrics
```python
# BAD: Using genotypic diversity for Pareto front spread
pareto_spread = calculate_genotype_diversity(pareto_front)  # Wrong!

# GOOD: Use phenotypic diversity for Pareto front
pareto_spread = calculate_phenotype_diversity(pareto_front)  # Correct
```

**Mistake 3**: No diversity-based intervention
```python
# BAD: Let population collapse without action
evolve_population()  # May converge prematurely

# GOOD: Monitor and intervene
if genotype_diversity < threshold:
    apply_diversity_operators()
```

---

## References

1. **Holland, J. H. (1975)**. "Adaptation in Natural and Artificial Systems."
2. **Goldberg, D. E. (1989)**. "Genetic Algorithms in Search, Optimization, and Machine Learning."
3. **Deb, K., et al. (2002)**. "A fast and elitist multi-objective genetic algorithm: NSGA-II." IEEE TEC.
4. **Burke, E. K., et al. (2013)**. "Hyper-heuristics: A survey of the state of the art." JORS.
5. **Kimura, M. (1983)**. "The Neutral Theory of Molecular Evolution."
6. **Toussaint, M., & Igel, C. (2002)**. "Neutrality: A necessity for self-adaptation." CEC.
7. **Toffolo, T. A. M., & Benini, E. (2003)**. "Genetic diversity as an objective in university timetabling." PATAT.

---

## Glossary

- **Genotype**: Internal chromosome representation (search space)
- **Phenotype**: Observable fitness values (solution space)
- **Genotypic diversity**: Structural variety in chromosomes
- **Phenotypic diversity**: Spread of fitness values
- **Neutral evolution**: Genetic drift without fitness change
- **Many-to-one mapping**: Multiple genotypes → same phenotype
- **Pareto front**: Set of non-dominated solutions (multi-objective)
- **Premature convergence**: Early collapse to local optimum
- **Exploration**: Searching new regions (high diversity)
- **Exploitation**: Refining known regions (low diversity)

---

**Document Version**: 1.0  
**Last Updated**: November 21, 2025  
**Author**: Schedule Engine Development Team  
**Related Docs**:
- `docs/04-algorithms/nsga2-implementation.md` - NSGA-II details
- `docs/reinf-learning/05-state-representation.md` - RL state encoding
- `src/metrics/diversity.py` - Diversity calculation code
- `src/rl/gym_env/state_encoder.py` - State encoder implementation
