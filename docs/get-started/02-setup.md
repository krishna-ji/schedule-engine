# Setup Guide

## Configuration Overview

Schedule Engine uses a hierarchical YAML-based configuration system with three layers:

1. **Base Configuration** (`configs/base.yaml`) - Common settings shared across all environments
2. **Environment Overrides** (`configs/test.yaml`, `configs/prod.yaml`) - Environment-specific parameters
3. **Runtime Mode Configs** (`configs/baseline/`, `configs/nsga/`, `configs/rl/`, etc.) - Experimental configurations

## Quick Setup

### 1. Verify Data Files

Ensure data files exist in `data/` directory:

```powershell
# Check required files
ls data/Course.json
ls data/Groups.json
ls data/Instructors.json
ls data/Rooms.json

# Validate data integrity
uv run check-data
```

**Expected output:**
```
✓ Course.json: 150 courses loaded
✓ Groups.json: 30 student groups loaded
✓ Instructors.json: 50 instructors loaded
✓ Rooms.json: 40 rooms loaded
✓ All data files valid
```

### 2. Review Base Configuration

```powershell
# Show current configuration
uv run show-config

# Show specific sections
uv run show-repair    # Repair system config
uv run show-soft      # Soft constraints only
uv run show-time      # Time system config
```

### 3. Choose Environment

**Test Environment** (smoke testing, quick iterations):
- 30 generations (~5-10 minutes)
- Small population (10 individuals)
- Use for rapid testing and validation

**Production Environment** (best quality, thesis runs):
- 2000 generations (~1-2.5 hours with GPU)
- Large population (200 individuals)
- Use for final experiments and benchmarking

## Configuration Files

### Base Configuration (`configs/base.yaml`)

Core settings shared across all environments:

```yaml
# Time system
time:
  quantum_minutes: 60  # 1-hour time slots
  earliest_preferred_time: "08:00"
  latest_preferred_time: "18:00"

# GA parameters (common)
ga:
  cxpb: 0.75              # Crossover probability
  mutpb: 0.25             # Mutation probability
  elite_preservation: true
  elite_size: 0.1
  population_strategy: hybrid  # 25% greedy, 50% smart, 25% random

# Parallel processing
parallel:
  use_multiprocessing: true
  num_workers: null  # Auto-detect CPU cores

# GPU acceleration
gpu:
  enabled: true
  batch_size: 100
  fallback_to_cpu: true
```

### Environment Overrides

**Test Environment** (`configs/test.yaml`):
```yaml
ga:
  ngen: 30          # Quick smoke test
  pop_size: 10

lns:
  trigger_interval: 10  # More frequent repairs for testing
```

**Production Environment** (`configs/prod.yaml`):
```yaml
ga:
  ngen: 2000        # Best quality
  pop_size: 200

lns:
  trigger_interval: 100  # Less frequent, more stable
```

### Runtime Mode Configs

10 progressive experimental modes (see [Runtime Modes Guide](../02-user-guides/runtime-modes.md)):

1. **Baseline** (`configs/baseline/1-pure-nsga.yaml`) - Pure NSGA-II
2. **Repairs** (`configs/nsga/2-nsga-repairs.yaml`) - + IGLS repairs
3. **Heuristics** (`configs/nsga/3-nsga-heuristics.yaml`) - + 19 heuristics
4. **Full** (`configs/nsga/4-nsga-full.yaml`) - Full GA (best non-RL)
5. **RL-Guided** (`configs/rl/5-rl-guided.yaml`) - RL-guided heuristic selection
6. **Round-Robin** (`configs/hybrid/6-roundrobin.yaml`) - Fixed rotation
7. **Specialists** (`configs/rl/7-rl-specialists.yaml`) - RL with specialists
8. **Archive** (`configs/rl/8-archive-diversity.yaml`) - Archive diversity
9. **Hierarchical** (`configs/rl/9-rl-hierarchical.yaml`) - Hierarchical RL
10. **Multi-Agent** (`configs/rl/10-rl-multiagent.yaml`) - Multi-agent RL

## Customizing Configuration

### Method 1: Override via Environment File

Create custom environment config (e.g., `configs/my-custom.yaml`):

```yaml
# Inherit from base
# Only specify what differs

ga:
  ngen: 500        # Custom generation count
  pop_size: 50

hard_constraints:
  instructor_exclusivity:
    weight: 5.0    # Increase penalty weight
```

Run with custom config:
```powershell
python main.py --config configs/my-custom.yaml
```

### Method 2: Programmatic Access

```python
from src.config import get_config, init_config

# Load default config
config = get_config()

# Load specific config
config = init_config("configs/my-custom.yaml")

# Access settings
print(config.ga.ngen)
print(config.hard_constraints.instructor_exclusivity.weight)
```

## Key Configuration Sections

### 1. Time System

Controls time slot discretization:

```yaml
time:
  quantum_minutes: 60             # Time slot duration (minutes)
  earliest_preferred_time: "08:00"
  latest_preferred_time: "18:00"
  midday_break_start: "12:00"
  midday_break_end: "13:00"
  max_sessions_per_day: 6
```

### 2. Genetic Algorithm

Core GA parameters:

```yaml
ga:
  ngen: 2000                     # Number of generations
  pop_size: 200                  # Population size
  cxpb: 0.75                     # Crossover probability
  mutpb: 0.25                    # Mutation probability
  elite_preservation: true       # Keep best solutions
  elite_size: 0.1                # 10% elitism
  population_strategy: hybrid    # Population initialization
```

### 3. Hard Constraints

Must-satisfy constraints (0 violations required):

```yaml
hard_constraints:
  student_group_exclusivity:
    enabled: true
    weight: 3.0    # Penalty multiplier
  instructor_exclusivity:
    enabled: true
    weight: 3.0
  instructor_qualifications:
    enabled: true
    weight: 3.0
```

### 4. Soft Constraints

Preference constraints (minimize violations):

```yaml
soft_constraints:
  avoid_early_sessions:
    enabled: true
    weight: 1.0
    soft_weight_factor: 1.0
  avoid_late_sessions:
    enabled: true
    weight: 1.0
```

### 5. Repair System (IGLS)

Local search repair configuration:

```yaml
repair:
  enabled: true
  max_iterations: 50
  neighborhood_size: 10
  stagnation_trigger: 50     # Generations without improvement
  exhaustive_initial_repair: true
```

### 6. Heuristic System

Heuristic operator configuration:

```yaml
heuristics:
  enabled: true
  parallel_execution: true
  operators:
    swap_rooms_violated:
      enabled: true
    shift_time_violated:
      enabled: true
    # ... 17 more operators
```

### 7. RL Integration

Reinforcement learning configuration:

```yaml
rl:
  enabled: false               # Killswitch (enable after training)
  model_path: "models/rl_agents/ppo_best.zip"
  inference_timeout_ms: 10
  fallback_strategy: "random"
  exploration_rate: 0.1
```

### 8. GPU Acceleration

GPU batch evaluation settings:

```yaml
gpu:
  enabled: true
  batch_size: 100              # Batch size for GPU evaluation
  fallback_to_cpu: true        # Graceful degradation
  device: "cuda:0"             # GPU device ID
```

## Validation

### Verify Configuration Syntax

```powershell
# Validate YAML syntax and config structure
uv run verify-config

# Verify runtime mode killswitches
python main.py --mode baseline --env test  # Validates killswitches
```

### Verify Data Compatibility

```powershell
# Check data integrity and feasibility
uv run check-data

# Run feasibility checks (constraint satisfaction analysis)
# Included automatically in first run
```

## Best Practices

### 1. Use Environment Variables

Set environment for easy switching:

```powershell
# Set environment
$env:ENVIRONMENT = "prod"
python main.py

# Or specify inline
python main.py --env prod
```

### 2. Version Control Configs

- Keep custom configs in separate files
- Don't modify `base.yaml` directly
- Use git branches for experimental configs

### 3. Document Custom Settings

Add comments to custom configs:

```yaml
ga:
  ngen: 500  # Reduced for faster iteration during development
  pop_size: 50  # Smaller population for memory constraints
```

### 4. Test Before Production

Always test custom configs with `--env test` first:

```powershell
# Test new config
python main.py --config configs/my-custom.yaml --env test

# If successful, run production
python main.py --config configs/my-custom.yaml --env prod
```

## Troubleshooting

### Issue: Config validation errors

**Solution:**
```powershell
# Check YAML syntax
uv run verify-config

# View full config (merged with environment)
uv run show-config
```

### Issue: Killswitch conflicts

Runtime modes validate killswitches automatically. If you see:
```
Error: rl.enabled must be true for rl-guided mode
```

**Solution:** Enable required killsitch in config or use compatible mode.

### Issue: Data file not found

**Solution:**
```powershell
# Verify data directory
ls data/

# Check io.data_dir in config
uv run show-config | Select-String "data_dir"
```

## Next Steps

- [First Run Guide](03-first-run.md) - Run your first experiment
- [UV Commands Reference](04-uv-commands.md) - All available commands
- [Runtime Modes Guide](../02-user-guides/runtime-modes.md) - Detailed mode documentation
- [Troubleshooting Guide](../troubleshooting/01-common-issues.md) - Common issues and solutions
