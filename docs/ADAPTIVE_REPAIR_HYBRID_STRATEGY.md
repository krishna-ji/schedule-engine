# Adaptive Repair: Hybrid Strategy

## Problem
Selective repair (10% genes) is fast but may miss constraint violations. Full repair (100% genes) is thorough but slow. Need intelligent switching between modes to balance speed and quality.

## Solution Approach
**Hybrid Strategy**: Combine stagnation detection + periodic triggers to dynamically switch between selective and full repair modes.

### Trigger Types (Priority Order)
1. **Intensive** (highest): Every `intensive_interval` (default 20) generations
   - Action: Full repair, `max_iterations=10`
   - Goal: Comprehensive search space exploration
   
2. **Stagnation**: No HC improvement for `window` (default 5) generations
   - Action: Full repair, `max_iterations=5`
   - Goal: Escape local optima when stuck
   
3. **Periodic**: Every `interval` (default 10) generations
   - Action: Full repair, `max_iterations=5`
   - Goal: Regular constraint satisfaction maintenance

## Key Components

### Configuration (`config/ga_params.REPAIR_HEURISTICS_CONFIG`)
```python
"adaptive_repair": {
    "stagnation_trigger": {
        "enabled": True/False,
        "window": 5,              # Gens without improvement
        "metric": "best_hc",      # Track hard violations
        "threshold": 0.0          # Min improvement required
    },
    "periodic_trigger": {
        "enabled": True/False,
        "interval": 10,           # Regular repair frequency
        "intensive_interval": 20  # Intensive repair frequency
    },
    "trigger_action": {
        "repair_mode": "full",    # or "selective"
        "max_iterations": 5
    },
    "intensive_action": {
        "repair_mode": "full",
        "max_iterations": 10      # More intensive
    }
}
```

### Stagnation Tracking (`GAScheduler.__init__`)
- `self.stagnation_counter`: Counts gens without improvement
- `self.last_best_hc`: Tracks previous best HC for comparison

### Trigger Logic (`GAScheduler._evolve_generation`)
1. Extract current best HC from population
2. Check improvement: `last_best_hc - current_best_hc`
3. Increment/reset `stagnation_counter` based on threshold
4. Detect trigger conditions (intensive → stagnation → periodic)
5. Update `repair_config["selective_mode"]` and `repair_config["max_iterations"]`
6. Log trigger event via `logger.console`

### Integration Points
All three repair call sites automatically use dynamic parameters:
- Post-crossover repair: `repair_config.get("selective_mode")` + `repair_config.get("max_iterations")`
- Post-mutation repair: Same
- Memetic mode repair: Uses `repair_config.get("memetic_iterations")`

## Expected Behavior

### Early Generations (0-19)
- Selective mode by default (fast)
- Gen 10: Periodic trigger → full repair (5 iterations)
- Gen 20: Intensive trigger → full repair (10 iterations)

### Mid-Search (20-80)
- Stagnation detection active
- If HC plateaus for 5 gens → full repair
- Periodic triggers continue every 10 gens
- Intensive triggers continue every 20 gens

### Late Search (80-100)
- Exploit good solutions (adaptive probabilities)
- Repair triggers help prevent constraint degradation
- Intensive triggers ensure quality maintenance

## Configuration Trade-offs

### Conservative (Fast, Less Thorough)
- `interval=20`, `intensive_interval=40`
- `trigger_action.max_iterations=3`
- `intensive_action.max_iterations=5`

### Aggressive (Slow, More Thorough)
- `interval=5`, `intensive_interval=10`
- `trigger_action.max_iterations=7`
- `intensive_action.max_iterations=15`

### Balanced (Default)
- `interval=10`, `intensive_interval=20`
- `trigger_action.max_iterations=5`
- `intensive_action.max_iterations=10`

## Implementation Files
- `config/ga_params.py`: Configuration block
- `src/core/ga_scheduler.py`: Trigger logic, stagnation tracking
- `src/ga/operators/violation_detector.py`: Detects violated genes
- `src/ga/operators/repair_selective.py`: Selective repair functions
- `src/ga/operators/repair.py`: Unified repair interface

## Rationale
- **Why stagnation detection?** Detects local optima, triggers intensive search
- **Why periodic triggers?** Ensures regular constraint maintenance regardless of convergence
- **Why intensive triggers?** Comprehensive exploration at longer intervals
- **Why configurable?** Different problem instances need different repair intensities
