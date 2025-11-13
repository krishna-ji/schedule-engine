````instructions
---
applyTo: "src/ga/**/*.py"
---

# GA Core & Operators Instructions

## ⚠️ DEPRECATED - GA CODE REMOVED

**This engine now uses pure Google OR-Tools CP-SAT constraint programming.**

All genetic algorithm code has been removed from the codebase.

## What Was Removed

### Deleted Directories
- `src/ga/` - All GA operators (crossover, mutation, selection, repair)
- `src/core/` - GA scheduler and chromosome handling

### Deleted Components
- Population initialization (hybrid/smart/random strategies)
- NSGA-II multi-objective optimization
- Fitness evaluation functions (hard + soft constraints)
- Repair heuristics (registry, selective mode, adaptive repair)
- Diversity tracking and Pareto front calculation
- DEAP library dependency

### Deleted Features
- Soft constraints (gaps, preferences, clustering)
- Multi-objective fitness (hard violations + soft penalty)
- Evolution metrics (generation tracking, convergence detection)
- Crossover operators (course-group aware)
- Mutation operators (constraint-guided)

## Migration to CP-SAT

The new architecture uses declarative constraint modeling:

```python
# OLD: GA with fitness evaluation
def evaluate_fitness(individual):
    hard_viol = sum(check_constraint(gene) for gene in individual)
    soft_pen = sum(calc_preference(gene) for gene in individual)
    return (hard_viol, soft_pen)

# NEW: CP-SAT with declarative constraints
model = cp_model.CpModel()
model.Add(session_start[s1] + duration[s1] <= session_start[s2])  # No overlap
```

### Key Differences
| GA Approach | CP-SAT Approach |
|-------------|-----------------|
| Iterative improvement | Complete search |
| Soft + hard constraints | Hard constraints only |
| Probabilistic (local optima) | Deterministic (feasible solution) |
| 100+ generations | Single solve |
| Minutes to hours | Hours to days (but guaranteed feasible) |

## Removed Files Reference
- `src/ga/operators/crossover.py`
- `src/ga/operators/mutation.py`
- `src/ga/operators/repair_*.py`
- `src/ga/population.py`
- `src/ga/evaluator/fitness.py`
- `src/core/ga_scheduler.py`

For CP-SAT implementation, see `src/ortools/cp_scheduler_clean.py`.

````
