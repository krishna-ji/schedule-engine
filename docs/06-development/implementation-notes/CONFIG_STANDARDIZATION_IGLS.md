# Configuration Standardization - IGLS System

## [2025-10-27] Standardized test/dev/prod configs to use IGLS system

### Overview

All three environment configurations (test, dev, prod) now share the same IGLS (Intensive Global Local Search) repair system configuration, with only environment-specific GA parameters differing.

### Architecture

**Common Settings (configs/common.yaml)**
- All IGLS configuration (three-tier repair system)
- All constraint settings (hard/soft)
- Time configuration
- Feasibility checks
- IO paths, calendar display
- GA crossover/mutation probabilities (cxpb, mutpb)

**Environment-Specific (test/dev/prod.yaml)**
- Only 5 top-level keys: name, environment, description, ga, parallel
- GA: ngen, pop_size (execution scale)
- Parallel: use_multiprocessing, num_workers (debugging vs production)

### IGLS Three-Tier System (in common.yaml)

#### Tier 1: Exhaustive Search (Fixed Generations)
```yaml
exhaustive_search:
  enabled: true
  generations: [3, 25]  # Two-shot: early + late optimization
  population_coverage: 0.3  # Top 30%
  max_neighborhood_size: 80
  timeout_seconds: 120
```

#### Tier 2: Stagnation-Triggered Greedy Repair
```yaml
stagnation_repair:
  enabled: true
  patience: 5  # Trigger after 5 gens without improvement
  min_generation: 8
  population_coverage: 0.5  # Top 50%
  max_iterations: 10
  timeout_seconds: 60
  cooldown: 3
```

#### Tier 3: Selective Probabilistic Repair
```yaml
selective_repair:
  enabled: true
  apply_probability: 0.3  # 30% of offspring
  apply_after_mutation: true
  apply_after_crossover: false
```

### Environment-Specific Settings

| Environment | ngen | pop_size | Multiprocessing | Use Case |
|------------|------|----------|-----------------|----------|
| test       | 30   | 10       | OFF             | Fast smoke test, debug IGLS triggers |
| dev        | 200  | 50       | ON              | Development, iterative testing |
| prod       | 500  | 100      | ON              | Production quality runs |

### Benefits

1. **Single Source of Truth**: IGLS configuration maintained in one place
2. **Consistent Behavior**: All environments use same repair strategy
3. **Easy Tuning**: Change IGLS params once, applies everywhere
4. **Minimal Configs**: Environment files are < 15 lines each
5. **Clear Separation**: Algorithm settings vs execution scale

### Files Modified

- `configs/common.yaml` - Added complete IGLS configuration
- `configs/test.yaml` - Reduced to 5 keys (name, env, desc, ga, parallel)
- `configs/dev.yaml` - Reduced to 5 keys
- `configs/prod.yaml` - Reduced to 5 keys

### Migration Notes

**Before** (old configs):
- Each environment had full repair configuration
- Duplication of constraint weights, time settings
- Hard to maintain consistency
- dev/prod had legacy adaptive repair enabled

**After** (standardized):
- Common.yaml has all IGLS settings
- Environments only specify ngen, pop_size, parallel
- All use same three-tier IGLS system
- Legacy adaptive repair disabled (replaced by IGLS)

### Verification

Run `python scripts/verify_config_standardization.py` to validate:
- Config sizes are minimal
- IGLS features present in common.yaml
- Environment-specific settings correct

### Usage

```bash
# Fast test (30 gens, single-threaded)
python main.py --env test

# Medium dev run (200 gens, parallel)
python main.py --env dev

# Full production run (500 gens, parallel)
python main.py --env prod
```

All three use identical IGLS configuration, only execution scale differs.
