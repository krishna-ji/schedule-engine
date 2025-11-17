# Heuristic Toolbox Quick Reference

Complete reference for the Phase 1.5 Heuristic Toolbox with decorator-based registry and killswitch architecture.

---

## Overview

**Purpose**: Provide pluggable heuristic operators for schedule optimization, diversity maintenance, and meta-heuristic strategies. Designed to prepare for RL integration (Phase 2).

**Architecture**: Decorator-based registry (like constraints/repair operators) with category-based organization and config-driven killswitches.

---

## Structure

```
src/heuristics/
├── __init__.py              # Module exports
├── registry.py              # Decorator system & registry
├── construction.py          # Greedy construction (3 heuristics)
├── perturbation.py          # Solution perturbation (5 heuristics)
├── improvement.py           # Local search (3 heuristics)
├── diversity.py             # Diversity maintenance (4 heuristics)
└── meta.py                  # Meta-heuristic strategies (4 heuristics)
```

**Total**: 19 heuristics across 5 categories

---

## Heuristics by Category

### 1. Construction (3)
Build schedules greedily from scratch. Use for:
- Initial population generation (better than random)
- Warm-starting GA with feasible solutions

| Heuristic | Priority | Description |
|-----------|----------|-------------|
| `largest_degree_first` | 1 | Schedule most conflicting courses first (graph coloring) |
| `most_constrained_first` | 2 | Schedule sessions with fewest options first (MRV) |
| `earliest_deadline_first` | 3 | Prioritize courses by session frequency |

**Default**: All disabled (use for specialized initialization)

---

### 2. Perturbation (5)
Shake solutions to escape local optima. Use for:
- Diversification in GA
- Iterated local search

| Heuristic | Priority | Default | Description |
|-----------|----------|---------|-------------|
| `random_swap` | 1 | ✓ | Swap time/room between sessions |
| `temporal_shift` | 2 | ✓ | Move sessions by delta quanta |
| `room_shuffle` | 3 | ✓ | Reassign rooms to compatible sessions |
| `instructor_reassign` | 4 | ✓ | Change to qualified alternatives |
| `multi_perturbation` | 5 | ✗ | Chain multiple perturbations (aggressive) |

**Default**: First 4 enabled

---

### 3. Improvement (3)
Local search moves for refinement. Use for:
- Hill climbing
- Memetic algorithms (local search within GA)

| Heuristic | Priority | Default | Description |
|-----------|----------|---------|-------------|
| `kempe_chain` | 1 | ✓ | Graph coloring moves for conflict resolution |
| `ejection_chain` | 2 | ✓ | Cascading reassignments |
| `variable_depth_search` | 3 | ✗ | Multi-move lookahead (computationally expensive) |

**Default**: First 2 enabled

---

### 4. Diversity (4)
Maintain population diversity. Use for:
- Preventing premature convergence
- Niching and speciation

| Heuristic | Priority | Default | Description |
|-----------|----------|---------|-------------|
| `distance_preserving_crossover` | 1 | ✗ | Maintain phenotypic distance |
| `crowding_mutation` | 2 | ✗ | Favor less-explored regions |
| `niching_selection` | 3 | ✗ | Fitness sharing for diversity |
| `adaptive_diversity_maintenance` | 4 | ✗ | Dynamic diversity control |

**Default**: All disabled (advanced features)

---

### 5. Meta (4)
High-level search strategies. Use for:
- Hybrid algorithms
- Alternative to standard GA

| Heuristic | Priority | Default | Description |
|-----------|----------|---------|-------------|
| `variable_neighborhood_descent` | 1 | ✗ | Systematic neighborhood exploration |
| `iterated_local_search` | 2 | ✗ | Perturbation + local search cycles |
| `adaptive_large_neighborhood` | 3 | ✗ | Dynamic destroy-repair (ALNS) |
| `guided_local_search` | 4 | ✗ | Penalty-based guidance |

**Default**: All disabled (alternative strategies)

---

## Usage

### Import Heuristics

```python
from src.heuristics import (
    get_all_heuristics,
    get_enabled_heuristics,
    get_heuristic_by_name,
    list_all_heuristics,
    get_construction_heuristics,
    get_perturbation_heuristics,
    get_improvement_heuristics,
    get_diversity_heuristics,
    get_meta_heuristics,
)
```

### Get Heuristics

```python
# Get all heuristics
all_heuristics = get_all_heuristics()  # Returns Dict[str, HeuristicMetadata]

# Get by category
perturbation_ops = get_perturbation_heuristics()

# Get enabled only (from config)
enabled = get_enabled_heuristics()
enabled_perturbation = get_enabled_heuristics(category="perturbation")

# Get specific heuristic
heuristic = get_heuristic_by_name("temporal_shift")
```

### Apply Heuristics

```python
# Construction heuristic (returns new individual)
from src.heuristics.construction import largest_degree_first
individual = largest_degree_first(context)

# Perturbation heuristic (modifies in-place, returns count)
from src.heuristics.perturbation import temporal_shift
modifications = temporal_shift(individual, context, delta=3, probability=0.3)

# Improvement heuristic (modifies in-place, returns improvements)
from src.heuristics.improvement import kempe_chain
improvements = kempe_chain(individual, context, max_iterations=5)

# Meta-heuristic (orchestrates other heuristics)
from src.heuristics.meta import variable_neighborhood_descent
total_improvements = variable_neighborhood_descent(individual, context)
```

### Registry Access

```python
# Get heuristic metadata
heuristic = get_heuristic_by_name("temporal_shift")

if heuristic:
    print(f"Name: {heuristic.name}")
    print(f"Category: {heuristic.category.value}")
    print(f"Description: {heuristic.description}")
    print(f"Priority: {heuristic.priority}")
    print(f"Requires population: {heuristic.requires_population}")
    print(f"Modifies individual: {heuristic.modifies_individual}")
    
    # Call the function
    result = heuristic.function(individual, context, **params)
```

---

## Configuration

### Config Structure (`configs/base.yaml`)

```yaml
heuristics:
  # Construction heuristics
  construction:
    largest_degree_first:
      enabled: false  # Default: disabled
      priority: 1
    most_constrained_first:
      enabled: false
      priority: 2
    # ... etc

  # Perturbation heuristics
  perturbation:
    random_swap:
      enabled: true  # Default: enabled
      priority: 1
      swap_type: time  # Options: time, room, both
      num_swaps: 1
    temporal_shift:
      enabled: true
      priority: 2
      delta: null  # null = random
      probability: 0.3
    # ... etc

  # Improvement heuristics
  improvement:
    kempe_chain:
      enabled: true
      priority: 1
      max_iterations: 5
    # ... etc

  # Diversity heuristics
  diversity:
    distance_preserving_crossover:
      enabled: false
      priority: 1
    # ... etc

  # Meta-heuristics
  meta:
    variable_neighborhood_descent:
      enabled: false
      priority: 1
    # ... etc
```

### Killswitch Control

**Enable/disable by category:**
```yaml
heuristics:
  perturbation:
    random_swap:
      enabled: false  # Turn off this heuristic
```

**Override priority:**
```yaml
heuristics:
  improvement:
    kempe_chain:
      enabled: true
      priority: 10  # Lower priority (normally 1)
```

**Configure parameters:**
```yaml
heuristics:
  perturbation:
    temporal_shift:
      enabled: true
      delta: 5  # Fixed shift instead of random
      probability: 0.5  # Higher probability
```

---

## Registry System

### Decorator-Based Registration

```python
from src.heuristics.registry import perturbation_heuristic

@perturbation_heuristic(
    name="my_custom_heuristic",
    description="Custom perturbation operator",
    priority=10,
    enabled_by_default=False,
    requires_population=False,
    modifies_individual=True
)
def my_custom_heuristic(individual, context, **kwargs):
    """Custom heuristic implementation."""
    # ... implementation ...
    return modification_count
```

### Metadata Fields

- **name**: Unique identifier (matches config key)
- **description**: Human-readable explanation
- **category**: HeuristicCategory enum (construction/perturbation/etc)
- **priority**: Execution order (1 = highest)
- **enabled_by_default**: Default enable state
- **requires_population**: Whether needs population access
- **modifies_individual**: Whether modifies in-place

---

## RL Integration (Phase 2)

The heuristic toolbox is designed for RL integration:

### Action Space
```python
# RL agent selects heuristic to apply
from src.heuristics import get_all_heuristics

action_space = list(get_all_heuristics().keys())  # 19 actions
```

### Reward Shaping
```python
# Track improvement for RL reward
heuristic = get_heuristic_by_name("kempe_chain")
improvements = heuristic.function(individual, context)

reward = improvements  # Positive reward for improvements
```

### Statistics Tracking
```python
from src.heuristics.registry import get_heuristic_statistics_template

stats = get_heuristic_statistics_template()
# Returns: {
#     "total_applications": 0,
#     "total_improvements": 0,
#     "temporal_shift_applications": 0,
#     "temporal_shift_improvements": 0,
#     # ... for all 19 heuristics
# }
```

---

## Testing

### Registry Test
```bash
python test/test_heuristics_registry.py
```

Validates:
- All 19 heuristics registered
- Correct category counts (3, 5, 3, 4, 4)
- Metadata completeness

### Usage Examples
```bash
python test/heuristics_examples.py
```

Demonstrates:
- Registry access patterns
- Category filtering
- Config integration
- Metadata inspection

---

## Integration Points

### GA Scheduler
```python
# In src/core/ga_scheduler.py
from src.heuristics import get_enabled_heuristics

# Apply perturbation after mutation
enabled_perturbations = get_enabled_heuristics(category="perturbation")
for name, heuristic in enabled_perturbations.items():
    heuristic.function(individual, context)
```

### Memetic Algorithm
```python
# In memetic mode
from src.heuristics.improvement import kempe_chain

for individual in elite:
    improvements = kempe_chain(individual, context, max_iterations=3)
```

### Hybrid Initialization
```python
# In population generation
from src.heuristics.construction import largest_degree_first

# Mix of construction heuristics and random
for i in range(pop_size):
    if i < pop_size * 0.3:  # 30% construction
        individual = largest_degree_first(context)
    else:  # 70% random
        individual = random_individual(context)
```

---

## Next Steps

1. **Integrate into GA**: Add heuristic application hooks in GA scheduler
2. **Memetic Mode**: Apply improvement heuristics to elite individuals
3. **RL Environment**: Create Gym environment wrapping heuristics
4. **Performance Testing**: Benchmark each heuristic's impact
5. **Adaptive Selection**: Implement adaptive heuristic selection based on performance

---

## Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `registry.py` | Decorator system & registry | ~350 |
| `construction.py` | Construction heuristics | ~550 |
| `perturbation.py` | Perturbation heuristics | ~350 |
| `improvement.py` | Improvement heuristics | ~550 |
| `diversity.py` | Diversity heuristics | ~450 |
| `meta.py` | Meta-heuristics | ~550 |
| **Total** | | **~2800 lines** |

---

## Dependencies

- `src.ga.sessiongene.SessionGene` - Gene representation
- `src.core.types.SchedulingContext` - Context with entities
- `src.encoder.quantum_time_system.QuantumTimeSystem` - Time system
- `src.config` - Configuration access

---

## Notes

- All heuristics follow consistent interface patterns
- Perturbation/improvement heuristics invalidate fitness after modification
- Construction heuristics return new individuals (don't modify)
- Meta-heuristics orchestrate other heuristics
- Config system allows fine-grained control per heuristic
- Registry provides introspection for RL integration
