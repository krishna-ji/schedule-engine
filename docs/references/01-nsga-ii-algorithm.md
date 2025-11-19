# NSGA-II Algorithm Reference

## Overview

**NSGA-II** (Non-dominated Sorting Genetic Algorithm II) is a multi-objective evolutionary algorithm that maintains a population of candidate solutions and evolves them toward the Pareto-optimal front.

**Paper:** Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

**DOI:** [10.1109/4235.996017](https://doi.org/10.1109/4235.996017)

## Key Concepts

### Multi-Objective Optimization

Schedule Engine minimizes two objectives simultaneously:

1. **Hard Constraint Violations** (must be 0)
2. **Soft Constraint Penalty** (minimize quality metric)

**Pareto Dominance:**
- Solution A dominates solution B if:
  - A is no worse than B in all objectives
  - A is strictly better than B in at least one objective

**Pareto Front:**
- Set of non-dominated solutions
- Trade-off curve between objectives
- Goal: Find diverse solutions on the front

### NSGA-II Characteristics

1. **Fast Non-dominated Sorting** - O(MN²) complexity
2. **Crowding Distance** - Maintains diversity
3. **Elitism** - Best solutions always preserved
4. **Tournament Selection** - Binary tournament

## Algorithm Flow

```
1. Initialize population (P₀) of size N
2. Evaluate fitness for all individuals
3. Sort by non-domination ranks
4. For generation t = 1 to max_generations:
   a. Create offspring (Q_t) via crossover & mutation
   b. Combine parent and offspring: R_t = P_t ∪ Q_t
   c. Fast non-dominated sort of R_t
   d. Select best N individuals for P_{t+1}:
      - Fill by Pareto fronts (F₁, F₂, ...)
      - If front doesn't fit, select by crowding distance
   e. Create offspring Q_{t+1} from P_{t+1}
5. Return final Pareto front
```

## Implementation in Schedule Engine

### Population Initialization

**File:** `src/ga/population.py`

**Strategies:**
- **Greedy (25%):** Constraint-aware construction
- **Smart (50%):** Heuristic-guided generation
- **Random (25%):** Pure random generation

**Code:**
```python
def generate_course_group_aware_population(
    pop_size: int,
    courses: list[Course],
    groups: list[Group],
    instructors: list[Instructor],
    rooms: list[Room],
    time_system: QuantumTimeSystem,
    config: Config,
    strategy: str = "hybrid"
) -> list[Individual]:
    """Generate initial population."""
    # Hybrid: 25% greedy, 50% smart, 25% random
    # ...
```

### Fitness Evaluation

**File:** `src/ga/evaluator/fitness.py`

**Function:**
```python
def evaluate(
    individual: Individual,
    context: SchedulingContext,
    config: Config
) -> tuple[float, float]:
    """
    Evaluate individual fitness.
    
    Returns:
        (-hard_violations, -soft_penalty)
        Both negative for minimization.
    """
    hard_penalties = 0
    soft_penalties = 0
    
    # Evaluate 14 hard constraints
    for constraint in hard_constraints:
        if constraint.enabled:
            violations = evaluate_constraint(individual, context, config)
            hard_penalties += violations * constraint.weight
    
    # Evaluate 4 soft constraints
    for constraint in soft_constraints:
        if constraint.enabled:
            penalty = evaluate_constraint(individual, context, config)
            soft_penalties += penalty * constraint.weight * config.soft_weight_factor
    
    return (-hard_penalties, -soft_penalties)
```

### Selection (NSGA-II)

**File:** `src/core/ga_scheduler.py`

**Method:** `tools.selNSGA2()`

**Parameters:**
- Population size: N
- Tournament size: 2 (binary tournament)
- Selection based on:
  1. Pareto rank (lower is better)
  2. Crowding distance (higher is better)

**Code:**
```python
# DEAP NSGA-II selection
offspring = toolbox.select(population, len(population))
```

### Crossover (Course-Group-Aware)

**File:** `src/ga/operators/crossover.py`

**Custom operator** preserving course-group relationships:

```python
def crossover_course_group_aware(
    ind1: Individual,
    ind2: Individual,
    context: SchedulingContext
) -> tuple[Individual, Individual]:
    """
    Course-group-aware crossover.
    
    Ensures sessions of same course stay together.
    """
    # Group sessions by course
    course_map1 = group_by_course(ind1)
    course_map2 = group_by_course(ind2)
    
    # Randomly select courses from each parent
    for course_id in courses:
        if random.random() < 0.5:
            # Take from parent 1
            child1_genes.extend(course_map1[course_id])
            child2_genes.extend(course_map2[course_id])
        else:
            # Take from parent 2
            child1_genes.extend(course_map2[course_id])
            child2_genes.extend(course_map1[course_id])
    
    return child1, child2
```

### Mutation

**File:** `src/ga/operators/mutation.py`

**Strategy:** Constraint-guided mutation

```python
def mutate_individual(
    individual: Individual,
    context: SchedulingContext,
    config: Config
) -> tuple[Individual]:
    """
    Mutate individual with constraint-aware strategy.
    """
    for gene in individual:
        if random.random() < gene_mut_prob:
            # Randomly mutate one aspect
            choice = random.choice(['time', 'room', 'instructor'])
            
            if choice == 'time':
                gene.time_quantum_start = random_valid_time()
            elif choice == 'room':
                gene.room_id = random_suitable_room()
            elif choice == 'instructor':
                gene.instructor_id = random_qualified_instructor()
    
    return (individual,)
```

### Elitism

**Configuration:**
```yaml
ga:
  elite_preservation: true
  elite_size: 0.1  # Top 10%
```

**Implementation:**
```python
# Preserve top 10% elite
if config.ga.elite_preservation:
    elite_count = int(config.ga.elite_size * len(population))
    elite = tools.selBest(population, elite_count)
    offspring[-elite_count:] = elite
```

## Parameters

### Recommended Settings

| Parameter | Test | Production | Description |
|-----------|------|------------|-------------|
| Population Size | 10 | 200 | Number of individuals |
| Generations | 30 | 2000 | Evolution iterations |
| Crossover Prob | 0.75 | 0.75 | Probability of crossover |
| Mutation Prob | 0.25 | 0.25 | Probability of mutation |
| Elite Size | 0.1 | 0.1 | Elite preservation (10%) |

### Configuration

**File:** `configs/base.yaml`

```yaml
ga:
  ngen: 2000                     # Generations (overridden by env)
  pop_size: 200                  # Population (overridden by env)
  cxpb: 0.75                     # Crossover probability
  mutpb: 0.25                    # Mutation probability
  elite_preservation: true       # Enable elitism
  elite_size: 0.1                # Elite percentage
  use_adaptive_probabilities: true  # Adaptive cxpb/mutpb
  use_constraint_guided_mutation: true  # Smart mutation
  population_strategy: hybrid    # Hybrid initialization
```

## Performance Characteristics

### Time Complexity

- **Population init:** O(N × K) where K = avg sessions per individual
- **Fitness evaluation:** O(N × C × K) where C = number of constraints
- **Non-dominated sorting:** O(M × N²) where M = number of objectives (2)
- **Crowding distance:** O(M × N log N)
- **Selection:** O(N)
- **Total per generation:** O(M × N² + N × C × K)

### Space Complexity

- **Population:** O(N × K) genes
- **Fronts:** O(N) individuals
- **Fitness cache:** O(N) fitness tuples

### Scalability

**Observed Performance (2000 generations, 200 population):**
- **Without GPU:** 24-48 hours
- **With GPU:** 1-2.5 hours (13-34x speedup)

**Bottleneck:** Constraint evaluation (solved with GPU batch evaluator)

## Extensions in Schedule Engine

### 1. Repair System (IGLS)

**Triggered when:**
- Stagnation detected (50 generations without improvement)
- Periodic trigger (every 100 generations)

**Effect:**
- Repairs best individual
- Reintegrates into population
- Helps escape local optima

### 2. Heuristic Toolbox

**19 specialized operators:**
- Applied in addition to standard crossover/mutation
- Targeted repairs for specific constraint violations
- Can be RL-guided or round-robin

### 3. GPU Acceleration

**Batch evaluation:**
- Evaluates population in parallel on GPU
- 10-50x speedup for constraint checks
- Automatic CPU fallback

### 4. Adaptive Probabilities

**Dynamic adjustment:**
- Crossover probability (cxpb) adjusted based on diversity
- Mutation probability (mutpb) adjusted based on convergence
- Controlled via RL agent or heuristics

## DEAP Library Usage

### Core Components

**Toolbox Setup:**
```python
from deap import base, creator, tools

# Create fitness (two-objective minimization)
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))

# Create individual type
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Initialize toolbox
toolbox = base.Toolbox()

# Register functions
toolbox.register("individual", init_individual, ...)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual, ...)
toolbox.register("mate", crossover_course_group_aware, ...)
toolbox.register("mutate", mutate_individual, ...)
toolbox.register("select", tools.selNSGA2)
```

**Evolution Loop:**
```python
# Initialize
population = toolbox.population(n=pop_size)
fitnesses = map(toolbox.evaluate, population)
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# Evolve
for gen in range(ngen):
    # Select offspring
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))
    
    # Apply crossover
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
    
    # Apply mutation
    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values
    
    # Evaluate invalid individuals
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    # Replace population
    population[:] = offspring
```

## Comparison with Standard GA

| Feature | Standard GA | NSGA-II | Schedule Engine |
|---------|-------------|---------|-----------------|
| Objectives | Single | Multiple | 2 (hard, soft) |
| Selection | Fitness-based | Pareto-based | NSGA-II + elitism |
| Diversity | Limited | Crowding distance | + Behavioral archive |
| Elitism | Optional | Built-in | Configurable (10%) |
| Convergence | Fast | Moderate | Fast (GPU + repair) |

## Tips for Tuning

### Increase Convergence Speed
- Increase population size (200 → 500)
- Increase elite size (10% → 20%)
- Enable repair system
- Use smart population initialization

### Improve Solution Quality
- Increase generations (2000 → 5000)
- Increase crossover probability (0.75 → 0.85)
- Decrease mutation probability (0.25 → 0.15)
- Enable heuristic toolbox

### Maintain Diversity
- Decrease elite size (10% → 5%)
- Increase mutation probability (0.25 → 0.35)
- Enable diversity operators
- Use archive-based diversity

## See Also

- [DEAP Documentation](https://deap.readthedocs.io/)
- [Original NSGA-II Paper](https://doi.org/10.1109/4235.996017)
- [Multi-Objective Optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization)
- [Pareto Efficiency](https://en.wikipedia.org/wiki/Pareto_efficiency)
