# Repair Trigger System - Implementation Log

## [2025-10-27] Implemented Adaptive Repair Triggers

### Overview
Implemented three-tier adaptive repair system with different intensities based on evolutionary progress:
1. **Stagnation-triggered repair** (SOFT) - Quick fixes when evolution stalls
2. **Periodic repair** (SOFT) - Regular maintenance every 10 generations
3. **Intensive repair** (HARD) - Deep repair every 20 generations

### Files Modified
- `configs/common.yaml` - Added `adaptive_repair` configuration section
- `configs/dev.yaml` - Added dev-specific adaptive repair overrides
- `src/core/ga_scheduler.py` - Enhanced trigger logic and console output
- `test/test_adaptive_repair_config.py` - Created verification test

### Configuration Structure

```yaml
repair:
  adaptive_repair:
    stagnation_trigger:
      enabled: true
      window: 5  # Generations without improvement
      threshold: 0.0  # Minimum improvement threshold
    
    periodic_trigger:
      enabled: true
      interval: 10  # Regular soft repair
      intensive_interval: 20  # Hard repair
    
    trigger_action:  # For stagnation & periodic
      repair_mode: selective  # Fast, targeted
      max_iterations: 5  # Limited
    
    intensive_action:  # For intensive periodic
      repair_mode: full  # Thorough scan
      max_iterations: 100  # Can take time
```

### Repair Mode Characteristics

| Mode | Trigger | Selective | Max Iterations | Memetic | Use Case |
|------|---------|-----------|----------------|---------|----------|
| **SOFT (Stagnation)** | No improvement for N gens | ✓ | 5 | ✗ | Quick fix when stuck |
| **SOFT (Periodic)** | Every 10 gens | ✓ | 5 | ✗ | Regular maintenance |
| **HARD (Intensive)** | Every 20 gens | ✗ | 100 | ✓ | Deep constraint satisfaction |

### Implementation Details

#### Stagnation Detection
- Tracks best hard constraint (HC) value over rolling window
- Triggers when no improvement for `window` generations (default: 5 for dev, 5 for common)
- Applies SOFT repair: selective mode, 5 max iterations, no memetic

#### Periodic Repair
- Every 10 generations: SOFT repair (selective, 5 iterations)
- Every 20 generations: HARD repair (full scan, 100 iterations, memetic ON)
- Intensive overrides periodic (20 is multiple of 10, so only intensive runs)

#### Runtime Behavior
The code dynamically modifies `repair_config` dictionary during `_evolve_generation()`:
- Sets `selective_mode` based on trigger type
- Sets `max_iterations` based on intensity
- **Enables `memetic_mode`** only for intensive repair (every 20 gens)
- Resets to base config when no trigger is active

#### Console Output
Enhanced messages to clearly show which repair mode is active:
-  Red: Intensive repair (HARD mode)
- ⚠️ Yellow: Stagnation repair (SOFT mode)
-  Cyan: Periodic repair (SOFT mode)

### Testing
Run `python test/test_adaptive_repair_config.py` to verify configuration loads correctly.

### Performance Impact
- **Stagnation/Periodic SOFT**: Minimal overhead (selective mode, few iterations)
- **Intensive HARD**: Significant time investment every 20 gens, but prioritizes quality
- Expected: ~1-2s for soft repairs, 10-30s for intensive repairs (depends on pop size)

### Key Changes from Previous Behavior
1. **Disabled always-on memetic mode** in dev.yaml (was causing repair every generation)
2. **Memetic now triggers only at intensive intervals** (every 20 gens)
3. **Added explicit SOFT vs HARD distinction** in code and console output
4. **Intensive repair uses full scan** (not selective) for thorough checking

### Known Limitations
- Intensive repair can take significant time (100 max_iterations per individual)
- May cause generation time spikes every 20 generations
- Consider reducing `max_iterations` in production if time is critical

### Future Improvements
- Add timeout parameter for intensive repair
- Track repair effectiveness (violations before/after)
- Adaptive max_iterations based on violation density
