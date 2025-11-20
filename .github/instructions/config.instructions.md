---
applyTo: "configs/**/*.yaml"
---

# Configuration File Guidelines for Schedule Engine

## YAML Structure & Format

### Required Format
- **Indentation**: 2 spaces (NOT tabs)
- **Line length**: Max 100 characters
- **Comments**: Use `#` for inline explanations
- **Structure**: Hierarchical dictionary (key-value pairs)

### File Types

1. **Environment Configs** (`configs/test.yaml`, `configs/prod.yaml`)
   - Inherit from `base.yaml`
   - Only override what differs
   - Keep minimal (environment-specific settings only)

2. **Runtime Mode Configs** (`configs/{category}/{mode}.yaml`)
   - Full standalone configs
   - Document purpose in header comment
   - Include killswitch states explicitly

3. **Base Config** (`configs/base.yaml`)
   - All common settings
   - Shared across all environments
   - Comprehensive documentation

## Configuration Principles

### 1. Killswitch Pattern
Every major feature has a master killswitch:

```yaml
# CORRECT : Explicit killswitch
repair:
  enabled: true  # Master killswitch
  stagnation_repair:
    enabled: true
    patience: 8

# WRONG : Missing master switch
repair:
  stagnation_repair:
    enabled: true  # Can't work if parent is disabled!
```

### 2. Null for Auto-Detection
Use `null` for runtime auto-detection:

```yaml
# CORRECT : Let system detect CPU count
parallel:
  num_workers: null  # Auto-detect all available cores

# WRONG : Hardcoded value (not portable)
parallel:
  num_workers: 16  # Fails on systems with fewer cores
```

### 3. Environment Inheritance
Environment files only override differences:

```yaml
# base.yaml
ga:
  ngen: 2000
  pop_size: 200
  cxpb: 0.75

# test.yaml (only overrides)
ga:
  ngen: 30        # Override for quick testing
  pop_size: 10    # Override
  # cxpb inherited from base.yaml
```

### 4. Descriptive Comments
```yaml
# CORRECT : Explain purpose and values
repair:
  enabled: true
  stagnation_repair:
    patience: 8  # Generations without improvement before repair
    population_coverage: 0.4  # Repair top 40% of population

# WRONG : Missing context
repair:
  enabled: true
  stagnation_repair:
    patience: 8
    population_coverage: 0.4
```

## Common Configuration Sections

### GA Parameters
```yaml
ga:
  ngen: 2000                     # Total generations
  pop_size: 200                  # Population size
  cxpb: 0.75                     # Crossover probability
  mutpb: 0.25                    # Mutation probability
  elite_preservation: true       # Keep best solutions
  elite_size: 0.1               # Top 10% preserved
  use_adaptive_probabilities: false  # Fixed vs adaptive
  population_strategy: hybrid    # random, smart, or hybrid
```

### Performance & Metrics
```yaml
# Performance profiling
performance:
  enable_profiling: true         # Show phase-level timing breakdown
  show_per_generation: true      # Display after each generation
  show_summary_table: true       # Display summary at end

# Metrics optimization (pymoo-accelerated)
metrics:
  advanced_metrics_frequency: 10 # Calculate expensive metrics every 10 gens
  always_calculate_basic: true   # Always track hard/soft/diversity
  # Note: pymoo provides 139x speedup (50s → 0.36s per generation)
```

### Repair System
```yaml
repair:
  enabled: true                  # Master killswitch
  memetic_mode: false            # Elite local search
  
  stagnation_repair:
    enabled: true
    patience: 8                  # Gens without improvement
    population_coverage: 0.4     # Repair top 40%
    max_iterations: 15
    timeout_seconds: 120
    cooldown: 5                  # Gens before re-trigger
  
  selective_repair:
    enabled: true
    apply_probability: 0.4       # 40% chance per generation
    apply_after_mutation: true
    apply_after_crossover: true
```

### Heuristics (19 operators)
```yaml
heuristics:
  construction:
    largest_degree_first:
      enabled: true
      priority: 1                # Lower = higher priority
    most_constrained_first:
      enabled: true
      priority: 2
  
  perturbation:
    random_swap:
      enabled: true
      priority: 1
      swap_type: time            # Options: time, room, both
      num_swaps: 1
  
  improvement:
    kempe_chain:
      enabled: true
      priority: 1
      max_iterations: 5
```

### RL Integration
```yaml
rl:
  enabled: false                 # Master RL killswitch
  mode: disabled                 # disabled, guided, specialists, etc.
  model_path: models/rl_agents/best_model.zip
  
  state_encoder:
    constraint_specific: true    # Detailed constraint state
    history_window: 10           # Gens to track
  
  reward:
    fitness_weight: 1.0
    diversity_weight: 0.3
    violation_penalty: -10.0
```

## Validation Rules

### Required Before Commit
1. **Syntax check**: `python -c "import yaml; yaml.safe_load(open('config.yaml'))"`
2. **Schema validation**: `uv run verify-config`
3. **Test run**: `uv run exp1 --env test --config path/to/config.yaml`

### Common Errors to Avoid

 **Tabs instead of spaces**
```yaml
repair:
	enabled: true  # ERROR: tab character
```

 **Use 2 spaces**
```yaml
repair:
  enabled: true
```

 **Inconsistent indentation**
```yaml
ga:
  ngen: 2000
    pop_size: 200  # ERROR: wrong indentation
```

 **Consistent 2-space indent**
```yaml
ga:
  ngen: 2000
  pop_size: 200
```

 **Missing quotes for special strings**
```yaml
time:
  earliest_preferred_time: 08:00  # ERROR: treated as number
```

 **Quote time strings**
```yaml
time:
  earliest_preferred_time: "08:00"
```

## Configuration Testing

When creating new config:
```bash
# 1. Validate syntax
uv run verify-config --config configs/new-config.yaml

# 2. Quick test (30 gens)
uv run exp1 --env test --config configs/new-config.yaml

# 3. Verify experiment manifest
cat output/experiment_manifest.json
```

## Documentation Requirements

Each runtime mode config must have:
1. **Header comment** explaining purpose
2. **Killswitch states** explicitly set
3. **Parameter explanations** for non-obvious values
4. **Reference to docs** if complex (e.g., `See docs/02-user-guides/runtime-modes.md`)

Example:
```yaml
# ================
# RUNTIME MODE 3: NSGA-II + Repairs + Heuristics (No Local Search)
# ================
# Tests impact of 19 heuristics WITHOUT local search (LNS).
# This isolates the contribution of heuristics alone.
#
# Expected results:
# - Hard violations: 8-15 (good but not optimal)
# - Improvement vs baseline: 70-80%
#
# See: docs/02-user-guides/runtime-modes.md#mode-3

ga:
  ngen: 2000
  pop_size: 200
  # ... rest of config
```

## When to Create New Config

 **Do create** new config for:
- New experimental mode
- Thesis experiment configuration
- Special hardware setup (GPU, large runners)

 **Don't create** new config for:
- One-off parameter tweaks (use CLI override instead)
- Personal preferences (use environment configs)
- Temporary debugging (modify test.yaml)

## Config Hierarchy

Priority order (highest to lowest):
1. CLI arguments (`--ngen 100`)
2. Custom config file (`--config path/to/config.yaml`)
3. Environment config (`test.yaml`, `prod.yaml`)
4. Base config (`base.yaml`)

Example:
```bash
# Uses: CLI > custom > env > base
uv run exp1 --env prod --config my-config.yaml --ngen 500
# ngen=500 (CLI), other settings from my-config.yaml, then prod.yaml, then base.yaml
```

## Never Do

-  Use tabs instead of spaces
-  Hardcode system-specific values (use `null` for auto-detection)
-  Duplicate common settings in env files (put in base.yaml)
-  Add features without killswitches (`enabled: bool` field required)
-  Skip validation before committing (run `uv run verify-config`)
-  Commit configs without header comments explaining purpose
