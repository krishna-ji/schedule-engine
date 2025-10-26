---
applyTo: "{src/core/**/*.py,src/ga/**/*.py}"
---

# GA Core & Operators Instructions

## Overview
Genetic algorithm implementation using DEAP. Core scheduling logic in `src/core/ga_scheduler.py`, operators in `src/ga/operators/`, population strategies in `src/ga/`.

## Key Components

### GAScheduler (`src/core/ga_scheduler.py`)
- Encapsulates DEAP toolbox, population, evolution loop
- Supports multiprocessing via worker pools
- Tracks metrics (hard/soft violations, diversity per generation)
- Methods: `setup_toolbox()`, `initialize_population()`, `evolve()`, `get_best_solution()`

### Chromosome Structure
- Individual = `list[SessionGene]`
- Fitness = `creator.FitnessMulti(weights=(-1.0, -0.01))` (minimize hard, minimize soft)
- Each gene represents one course-group session assignment

### Population Strategies
- **hybrid** (default): 25% greedy, 50% smart, 25% random
- **smart**: 100% constraint-aware initialization
- **random**: Baseline (not recommended)

## Genetic Operators

### Crossover (`src/ga/operators/crossover.py`)
- `crossover_course_group_aware()` - Preserves course-group relationships
- Requires course-group pairs to align across population
- Can trigger repair if `repair.apply_after_crossover=True`

### Mutation (`src/ga/operators/mutation.py`)
- `mutate_individual()` - Randomly changes time slot, instructor, or room
- Requires `context` dict: `{courses, groups, instructors, rooms, available_quanta}`
- Can trigger repair if `repair.apply_after_mutation=True`
- Constraint-guided mutation available via `use_constraint_guided_mutation`

### Repair System (`src/ga/operators/repair_*.py`)
- **Registry-based**: `repair_registry.py` manages heuristic functions
- **Selective mode**: Only repairs genes with detected violations (faster)
- **Full mode**: Repairs all genes (more thorough)
- **Adaptive**: Triggers on stagnation or periodic intervals

## Rules

### Creating New Operators
1. Accept `individual` (list[SessionGene]) as first parameter
2. Return modified `individual` (or tuple for toolbox.register)
3. Use `context` dict for GA dependencies (don't import data globally)
4. Mark as mutation/crossover in toolbox: `toolbox.register("mutate", ...)`

### SessionGene Constraints
- `quanta` must be sorted and unique (no duplicate slots)
- `course_id` must exist in context.courses
- `group_ids` must match course enrollment
- `instructor_id` must be qualified for course
- `room_id` must match course type (lab/theory)

### Multiprocessing Safety
- Worker initialization in `_worker_init()` loads data from JSON
- Set `os.environ["_GA_WORKER_PROCESS"] = "1"` to suppress worker output
- Use `pool.map()` for fitness evaluation, not custom processes

### Performance Considerations
- Selective repair 2-3x faster than full repair
- Hybrid population initialization improves convergence 15-25%
- Parallelization provides 3-6x speedup on multi-core systems

## Examples

### Adding a New Repair Heuristic
```python
# In src/ga/operators/repair_registry.py
@register_repair_heuristic(
    name="fix_room_conflicts",
    description="Resolves room double-booking",
    category="hard",
    priority=3
)
def repair_room_conflicts(gene: SessionGene, context: dict) -> SessionGene:
    # Implementation here
    return gene
```

### Custom Population Initializer
```python
# In src/ga/population.py
def generate_my_strategy(context, pop_size, toolbox):
    population = []
    for _ in range(pop_size):
        individual = toolbox.individual()
        # Custom initialization logic
        population.append(individual)
    return population
```

## Never Do
- ❌ Modify individual length during crossover (breaks alignment)
- ❌ Create genes for courses not enrolled by groups
- ❌ Access config before it's initialized
- ❌ Use `print()` in worker processes (use logger or suppress)
- ❌ Share mutable state across multiprocessing workers
