# Configuration Architecture Guide

**Last Updated**: November 2025  
**Status**: Active Design Document

## Overview

The schedule-engine uses a **layered configuration architecture** that enables:
- ✅ DRY principle (no duplication)
- ✅ Environment-specific scaling (test/prod)
- ✅ Runtime mode experimentation
- ✅ Custom configuration overrides

## Architecture Principles

### 1. Three-Layer Merge Strategy

```
┌─────────────────────────────────────────────────────┐
│ Layer 3: ENVIRONMENT (configs/{env}.yaml)           │
│ • Environment scaling (test=30 gens, prod=2000)     │
│ • ALWAYS applied last (final override)              │
└─────────────────────────────────────────────────────┘
                        ↑
┌─────────────────────────────────────────────────────┐
│ Layer 2: MODE (configs/{category}/{mode}.yaml)      │
│ • Runtime mode specific (baseline, rl-guided, etc.) │
│ • Optional - only when using --mode or --config     │
└─────────────────────────────────────────────────────┘
                        ↑
┌─────────────────────────────────────────────────────┐
│ Layer 1: BASE (configs/base.yaml)                   │
│ • Common settings for all configurations            │
│ • Default values                                    │
└─────────────────────────────────────────────────────┘
```

### 2. Merge Order (Critical!)

**Always**: `base.yaml → [mode.yaml] → env.yaml`

The environment layer ALWAYS applies last, ensuring proper test/prod scaling regardless of which priority path is used.

## Configuration Layers

### Layer 1: Base Configuration (`configs/base.yaml`)

**Purpose**: Common settings shared across ALL environments and modes

**Contains**:
- Time system configuration (quantum_minutes, preferred times)
- I/O paths (data_dir, output_dir)
- Parallel processing defaults
- Constraint weight defaults
- Feature flags (master switches)

**DO NOT** include in base.yaml:
- ❌ `ngen` or `pop_size` (environment-specific)
- ❌ Runtime mode constraints (mode-specific)
- ❌ Any values that differ between test/prod

### Layer 2: Mode Configuration (`configs/{category}/{mode}.yaml`)

**Purpose**: Runtime mode specific settings and constraints

**Structure**:
```
configs/
├── baseline/
│   └── 1-pure-nsga.yaml         # Mode 1
├── nsga/
│   ├── 2-nsga-repairs.yaml      # Mode 2
│   ├── 3-nsga-heuristics.yaml   # Mode 3
│   └── 4-nsga-full.yaml         # Mode 4
├── rl/
│   ├── 5-rl-guided.yaml         # Mode 5
│   └── 7-rl-specialists.yaml    # Mode 7
└── hybrid/
    └── 6-roundrobin.yaml        # Mode 6
```

**Contains**:
- Mode-specific algorithm settings
- Feature killswitches for that mode
- Operator probabilities
- Heuristic configurations

**MUST NOT** include (will be overridden by environment):
- ❌ `ga.ngen` - Use environment config
- ❌ `ga.pop_size` - Use environment config
- ❌ `parallel.num_workers` - Inherit from base

**Example** (`configs/hybrid/6-roundrobin.yaml`):
```yaml
# ✅ CORRECT: Mode-specific settings only
ga:
  # ngen and pop_size inherit from environment
  cxpb: 0.75
  mutpb: 0.25
  use_adaptive_probabilities: false  # Fixed round-robin

heuristics:
  construction:
    largest_degree_first:
      enabled: true
  # ... other heuristics
```

### Layer 3: Environment Configuration (`configs/{env}.yaml`)

**Purpose**: Environment-specific scaling and deployment settings

**Available Environments**:
- `test.yaml` - Smoke tests (30 gens, 10 pop, ~2-5 min)
- `prod.yaml` - Production runs (2000 gens, 200 pop, ~3-5 hours)

**Contains**:
- `ga.ngen` - Generation count scaling
- `ga.pop_size` - Population size scaling
- Repair/LNS trigger intervals (scaled proportionally)
- Timeout values

**Example** (`configs/test.yaml`):
```yaml
name: "Test - Smoke Test"
environment: test
description: "Fast smoke test - 30 gens, 10 pop, 5-10 min"

ga:
  ngen: 30
  pop_size: 10

repair:
  exhaustive_search:
    generations: [3, 25]  # Scaled from prod [3, 30, 150, ...]
    timeout_seconds: 120
```

## Configuration Loading Flow

### Priority Paths

#### Path 1: Runtime Mode (Recommended for Experiments)

```bash
uv run python main.py --mode baseline --env prod
```

**Flow**:
1. Set `ENVIRONMENT=prod` (in main.py, before config loading)
2. Load `configs/base.yaml`
3. Load `configs/baseline/1-pure-nsga.yaml` (mode config)
4. Load `configs/prod.yaml` (environment override)
5. Validate against RuntimeMode constraints
6. Return merged Config object

**Use Case**: Running experiments with proper mode tracking

#### Path 2: Explicit Config Path

```bash
uv run python main.py --config configs/hybrid/6-roundrobin.yaml --env test
```

**Flow**:
1. Set `ENVIRONMENT=test`
2. Load `configs/base.yaml`
3. Load `configs/hybrid/6-roundrobin.yaml` (custom config)
4. Load `configs/test.yaml` (environment override)
5. Return merged Config object

**Use Case**: Testing custom configurations, launcher shortcuts

#### Path 3: Environment Variable

```bash
export SCHEDULE_CONFIG=configs/custom/my-config.yaml
export ENVIRONMENT=prod
uv run python main.py
```

**Use Case**: CI/CD pipelines, automated workflows

#### Path 4: Default (Test Environment Only)

```bash
uv run python main.py
```

**Flow**: `base.yaml → test.yaml`

**Use Case**: Quick testing without mode specification

## CLI Command Patterns

### Direct Invocation (main.py)

```bash
# Baseline experiment (prod)
python main.py --mode baseline --env prod --experiment "exp1-baseline"

# RL-guided experiment (test)
python main.py --mode rl --env test

# Custom config (test)
python main.py --config configs/custom.yaml --env test

# Default (test environment)
python main.py
```

### Launcher Shortcuts (Recommended)

```bash
# Unified CLI with profile support
uv run nsga --test         # Mode 4 (full NSGA-II) - test profile
uv run nsga --prod         # Mode 4 (full NSGA-II) - prod profile

uv run heuristic-roundrobin --test   # Mode 6 - test profile
uv run heuristic-roundrobin --prod   # Mode 6 - prod profile

# Legacy shortcuts (backward compatible)
uv run baseline --test     # Mode 1
uv run repairs --test      # Mode 2
uv run full --prod         # Mode 4
```

## Environment Variable Management

### Critical Timing

**❌ WRONG** (Environment set too late):
```python
args = parser.parse_args()
config = load_config(runtime_mode)  # Reads ENVIRONMENT here!
os.environ["ENVIRONMENT"] = args.env  # Too late!
```

**✅ CORRECT** (Environment set early):
```python
args = parser.parse_args()
os.environ["ENVIRONMENT"] = args.env or "test"  # Set FIRST
config = load_config(runtime_mode)  # Now sees correct environment
```

### Where It's Set

1. **main.py** - From `--env` CLI argument (lines ~115-120)
2. **launcher.py** - Via `sys.argv` manipulation before importing main
3. **Environment entry points** - In `_create_env_main()` factory

## Experiment Management Integration

### ExperimentManager Synchronization

The `ExperimentManager` expects:
- `runtime_mode`: RuntimeMode enum
- `output_path`: Structured by mode
- `config_path`: Path to mode config file

**Synchronized Flow**:
```python
# 1. Load config with runtime mode
config = load_config(runtime_mode=RuntimeMode.RL_GUIDED)

# 2. Create output directory (mode-aware)
output_dir = manager.create_output_dir(RuntimeMode.RL_GUIDED, exp_name)

# 3. Register experiment run
run = manager.register_run(
    runtime_mode=RuntimeMode.RL_GUIDED,
    config_path=RuntimeMode.RL_GUIDED.config_path,
    output_path=output_dir,
    seed=69
)

# 4. Run experiment
result = run_standard_workflow(config=config, output_dir=output_dir)

# 5. Update experiment metadata
manager.update_run(run.run_id, duration=elapsed, best_hc=result.hc)
```

## Common Pitfalls & Solutions

### Pitfall 1: Mode Config Overrides Environment

**Problem**: Hardcoded `ngen: 2000` in mode config prevents test profile from working

**Solution**: Remove all environment-scalable values from mode configs
```yaml
# ❌ BAD
ga:
  ngen: 2000        # Prevents test profile (30 gens)
  pop_size: 200     # Prevents test profile (10 pop)

# ✅ GOOD
ga:
  # ngen and pop_size inherit from environment (test/prod)
  cxpb: 0.75        # Mode-specific only
  mutpb: 0.25
```

### Pitfall 2: Environment Set After Config Load

**Problem**: Config loader reads `ENVIRONMENT` env var before it's set

**Solution**: Set environment FIRST in main.py (before calling `load_config()`)

### Pitfall 3: Inconsistent Naming

**Problem**: "profile" (launcher) vs "environment" (main.py) vs "env" (CLI flag)

**Solution**: Use consistent terminology:
- **Environment**: test/prod (deployment target)
- **Profile**: Alias for environment in CLI
- **Mode**: Runtime mode (baseline, rl-guided, etc.)

### Pitfall 4: Missing Environment Override

**Problem**: Custom config doesn't scale with `--env` flag

**Solution**: Ensure config loader ALWAYS applies environment layer (fixed in recent update)

## Validation & Testing

### Config Layer Testing

```bash
# Test base.yaml loads
python -c "from src.config import load_config; c = load_config(); print(c.ga.ngen)"

# Test mode layer
export ENVIRONMENT=test
python -c "from src.config.runtime_mode import RuntimeMode; from src.config import load_config; c = load_config(runtime_mode=RuntimeMode.BASELINE); print(c.ga.ngen)"

# Test environment override
export ENVIRONMENT=prod
python -c "from src.config import load_config; c = load_config(); print(c.ga.ngen)"
```

### Expected Outputs

| Command | Expected ngen | Expected pop_size |
|---------|---------------|-------------------|
| `--mode baseline --env test` | 30 | 10 |
| `--mode baseline --env prod` | 2000 | 200 |
| `--config custom.yaml --env test` | 30 | 10 |
| `--config custom.yaml --env prod` | 2000 | 200 |

## Future Improvements

### Phase 1 (Immediate)
- [x] Fix environment layer always applying (completed Nov 2025)
- [x] Move environment setting before config load (completed Nov 2025)
- [ ] Add config validation warnings for environment-scalable values in mode configs
- [ ] Improve error messages for config merge conflicts

### Phase 2 (Short-term)
- [ ] Add `--dry-run` to show final merged config without running
- [ ] Create config diff tool (`compare_configs.py`)
- [ ] Add runtime config override via CLI (e.g., `--set ga.ngen=50`)

### Phase 3 (Long-term)
- [ ] Config schema validation with JSON Schema
- [ ] Config migration tool for version upgrades
- [ ] Web UI for config visualization and editing

## References

- **Config Models**: `src/config/models.py` - Pydantic validation schemas
- **Config Loader**: `src/config/loader.py` - Merge logic implementation
- **Runtime Modes**: `src/config/runtime_mode.py` - Mode definitions and constraints
- **Experiment Manager**: `src/workflows/experiment_manager.py` - Output organization
- **CLI Launcher**: `scripts/launcher.py` - Unified command interface
- **Main Entry**: `main.py` - Argument parsing and config initialization
