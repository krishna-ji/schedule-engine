---
applyTo: "configs/**/*.py"
---

# Configuration File Guidelines for Schedule Engine

## Python Dataclass System

### Architecture
- **Base**: `configs/base.py` → `BaseConfig` (all shared defaults)
- **Profiles**: `configs/profiles.py` → `TestConfig`/`ProdConfig` (scaling overrides)
- **Experiments**: `configs/experiments/*.py` → Experiment-specific configs
- **Inheritance**: `BaseConfig → TestConfig/ProdConfig → ExperimentConfig`

### File Types

1. **Base Config** (`configs/base.py`)
   - All shared defaults (GA params, constraints, killswitches)
   - Single source of truth for common settings
   - Comprehensive inline documentation via docstrings

2. **Profile Configs** (`configs/profiles.py`)
   - `TestConfig`: Smoke tests (30 gens, 10 pop, ~2-5 min)
   - `ProdConfig`: Production (2000 gens, 400 pop, ~1-3 hrs)
   - Only override scaling params (ngen, pop_size, workers)

3. **Experiment Configs** (`configs/experiments/{name}.py`)
   - Individual experiment definitions (A-F modes)
   - Killswitch states explicitly set
   - Multiple inheritance: `ExperimentBaseConfig + TestConfig/ProdConfig`
   - Example: `baseline.py`, `memetic.py`, `rl_guided.py`

## Configuration Principles

### 1. Killswitch Pattern
Every major feature has master killswitch in `BaseConfig`:

```python
# CORRECT: Explicit killswitch
@dataclass
class BaseConfig:
    repair_enabled: bool = False  # Master killswitch
    repair_patience: int = 8
    repair_coverage: float = 0.4

# Enable in experiment
@dataclass
class MemeticBaseConfig:
    repair_enabled: bool = True  # Override to enable
```

### 2. None for Auto-Detection
Use `None` for runtime auto-detection:

```python
# CORRECT: Let system detect CPU count
num_workers: int | None = None  # Auto-detect cores

# WRONG: Hardcoded (not portable)
num_workers: int = 16  # Fails on systems with fewer cores
```

### 3. DRY Inheritance
```python
# base.py - shared defaults
@dataclass
class BaseConfig:
    ngen: int = 100
    pop_size: int = 50
    repair_enabled: bool = False

# profiles.py - scaling only
@dataclass
class ProdConfig(BaseConfig):
    ngen: int = 2000
    pop_size: int = 400

# experiments/memetic.py - killswitch overrides
@dataclass
class MemeticBaseConfig:
    repair_enabled: bool = True  # Enable repair
    memetic_mode: bool = True

class MemeticProdConfig(MemeticBaseConfig, ProdConfig):
    pass  # Inherits both killswitches + production scaling
```

### 4. Type Safety
All configs are strictly typed:

```python
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class BaseConfig:
    ngen: int = 100              # Total generations
    pop_size: int = 50           # Population size
    cxpb: float = 0.70           # Crossover probability
    mutpb: float = 0.20          # Mutation probability
    elite_preservation: bool = True
    elite_size: float = 0.05     # Top 5% preserved

    # Type checker enforces correctness
    # mypy catches: ngen: str = "100"  # ERROR!
```

## Common Configuration Patterns

### GA Parameters
```python
@dataclass
class BaseConfig:
    ngen: int = 100
    pop_size: int = 50
    cxpb: float = 0.70
    mutpb: float = 0.20
    elite_preservation: bool = True
    elite_size: float = 0.05
    population_strategy: str = "random"  # random, smart, hybrid
```

### Performance & Metrics
```python
@dataclass
class BaseConfig:
    # Parallel processing
    use_multiprocessing: bool = True
    num_workers: int | None = None  # None = auto-detect

    # Metrics (pymoo-accelerated: 139x speedup)
    advanced_metrics_frequency: int = 10  # Every 10 gens
```

### Repair System
```python
@dataclass
class BaseConfig:
    repair_enabled: bool = False  # Master killswitch
    repair_patience: int = 8
    repair_coverage: float = 0.4
    repair_max_iterations: int = 15
    repair_timeout: int = 120
```

### Heuristics (19 operators)
```python
@dataclass
class BaseConfig:
    heuristics_master_enabled: bool = False  # Master killswitch
    heuristic_selection_mode: str = "fixed"  # fixed, adaptive, rl

    # Heuristic registry defined separately
    # See: src/heuristics/registry.py
```

### RL Integration
```python
@dataclass
class BaseConfig:
    rl_enabled: bool = False      # Master killswitch
    rl_model_path: str = "models/rl_agents/best_model.zip"
    rl_history_window: int = 10
```

## Validation & Testing

### Type Checking
```bash
# All configs type-checked with strict mypy
mypy configs/
```

### Test Configuration
```bash
# Test with smoke profile
uv run baseline --test

# Production run
uv run baseline --prod --name "my-experiment"
```

## Common Patterns to Avoid

❌ **Hardcoded system-specific values**
```python
num_workers: int = 16  # Fails on systems with fewer cores
```

✅ **Use auto-detection**
```python
num_workers: int | None = None  # Auto-detect
```

❌ **Missing type annotations**
```python
ngen = 100  # Type unclear
```

✅ **Explicit types**
```python
ngen: int = 100  # Clear and type-checked
```

## Configuration Access

```python
# In application code
from schedule_engine.config import get_config

config = get_config()  # Returns Pydantic Config model
ngen = config.ngen
repair_enabled = config.repair_enabled
```

## Experiment Registration

In `main.py`:
```python
from configs import experiment_a, experiment_b

EXPERIMENTS = {
    "a": ("Experiment A: Baseline", experiment_a, experiment_a_baseline),
    "b": ("Experiment B: Memetic", experiment_b, experiment_b_memetic),
}
```

In `pyproject.toml`:
```toml
[project.scripts]
baseline = "scripts.launcher:main_baseline"  # Mode A
memetic = "scripts.launcher:main_memetic"    # Mode B
```

## Documentation Requirements

Each experiment config must have:
1. **Module docstring** explaining purpose
2. **Killswitch states** explicitly set
3. **Metadata constants** (EXPERIMENT_ID, EXPERIMENT_NAME, etc.)

Example (`configs/experiments/baseline.py`):
```python
"""
Experiment A: Pure NSGA-II Baseline

Minimal NSGA-II with all enhancements disabled.
Serves as baseline for comparing other experiments.

Expected results:
- Hard violations: 15-25
- Runtime: ~2-5 min (test), ~1-3 hrs (prod)
"""

EXPERIMENT_ID = "a"
EXPERIMENT_NAME = "Experiment A: Pure NSGA-II"
KILLSWITCHES = {
    "repair_enabled": False,
    "heuristics_master_enabled": False,
    "rl_enabled": False,
}
```

## When to Create New Config

✅ **Do create** for:
- New experimental mode (baseline, memetic, etc.)
- Thesis experiment configuration
- Major feature testing

❌ **Don't create** for:
- One-off parameter tweaks (use profile overrides)
- Personal preferences (modify existing)
- Temporary debugging (use test profile)

## Configuration Hierarchy

Priority order (highest to lowest):
1. Runtime overrides (from launcher)
2. Experiment config (`ExperimentProdConfig`)
3. Profile config (`ProdConfig`/`TestConfig`)
4. Base config (`BaseConfig`)

Example inheritance:
```python
# BaseConfig: ngen=100, repair_enabled=False
# ProdConfig: ngen=2000 (override)
# MemeticProdConfig: repair_enabled=True (override)
# Result: ngen=2000, repair_enabled=True
```

## Best Practices

✅ **DO:**
- Use type annotations on all fields
- Set killswitches explicitly in experiments
- Document purpose in module docstring
- Test with `--test` profile first
- Run type checker: `mypy configs/`

❌ **DON'T:**
- Hardcode system-specific values (use `None`)
- Duplicate BaseConfig defaults
- Skip type annotations
- Add features without killswitches
- Commit without testing
