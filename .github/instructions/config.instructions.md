---
applyTo: "src/config/**/*.py"
---

# Configuration System Instructions

## Overview

YAML-based hierarchical configuration with domain separation and environment inheritance. Loaded via `src/config/loader.py`, validated by `src/config/models.py`.

## Key Files

- `src/config/models.py` - Pydantic models (Config, GAConfig, RepairConfig, etc.)
- `src/config/loader.py` - Hierarchical YAML loading with deep merge (supports legacy structure)
- `src/config/__init__.py` - Global config object + `init_config()`, `get_config()`

## Configuration Structure

### Hierarchical (Preferred)
```
configs/
├── common/          # Shared settings (time, I/O, parallel, feasibility)
│   ├── base.yaml
│   ├── prod.yaml
│   ├── test.yaml
│   └── med.yaml
├── ga/              # GA-specific (operators, constraints, repair, heuristics)
│   ├── base.yaml
│   ├── prod.yaml
│   ├── test.yaml
│   └── med.yaml
└── rl/              # RL-specific (agent, training, inference, evaluation)
    ├── base.yaml
    ├── prod.yaml
    ├── test.yaml
    └── med.yaml
```

### Legacy (Backward Compatible)
```
configs/
├── base.yaml        # All settings mixed (backward compatibility)
├── prod.yaml
└── test.yaml
```

## Configuration Loading Order

Hierarchical mode (when `configs/{common,ga,rl}/` exist):
1. `configs/common/base.yaml`
2. `configs/common/{environment}.yaml`
3. `configs/ga/base.yaml`
4. `configs/ga/{environment}.yaml`
5. `configs/rl/base.yaml`
6. `configs/rl/{environment}.yaml`

Legacy mode (fallback):
1. `configs/base.yaml`
2. `configs/{environment}.yaml`

**Note**: Training uses separate configs in `config-train/` with similar inheritance (base + profile overrides).

## Rules

### Accessing Configuration

```python
# Runtime access (anywhere in code)
from src.config import get_config
config = get_config()
ngen = config.ga.ngen
```

### Adding New Settings

1. Add field to appropriate Pydantic model in `src/config/models.py`
2. Include default value with type annotation
3. Add validation if needed (`@field_validator`)
4. **Hierarchical structure**: Add to appropriate domain base.yaml:
   - Common settings → `configs/common/base.yaml`
   - GA settings → `configs/ga/base.yaml`
   - RL settings → `configs/rl/base.yaml`
5. Override in specific env files if environment-specific:
   - `configs/{domain}/prod.yaml`
   - `configs/{domain}/test.yaml`
   - `configs/{domain}/med.yaml`
6. Update `Config.summary()` if user-facing

### YAML File Structure

- Use lowercase keys with underscores (e.g., `pop_size`, not `popSize`)
- Group related settings under sections (ga, repair, parallel, etc.)
- Include comments for non-obvious settings
- **Domain base files**: Contains ALL domain-specific defaults
- **Environment files**: Only override what differs per environment
- Keep test.yaml minimal (fast), prod.yaml comprehensive (quality)
- Use med.yaml for balanced development runs

### Domain Organization

**Common** (`configs/common/`):
- Time configuration (quantum system, scheduling windows)
- I/O paths (data_dir, output_dir)
- Calendar display settings
- Parallel processing configuration
- Feasibility check settings

**GA** (`configs/ga/`):
- GA parameters (population, generations, crossover, mutation)
- Hard and soft constraints
- Repair system (IGLS, LNS, stagnation, selective)
- Enhancements (hypermutation, population restart, etc.)
- Heuristic toolbox (construction, perturbation, improvement, diversity, meta)

**RL** (`configs/rl/`):
- RL integration (enabled, mode)
- Environment configuration
- Reward function weights
- Agent configuration (PPO/DQN hyperparameters)
- Training configuration (curriculum, checkpoints)
- Inference configuration
- Hybrid controller settings
- Evaluation and logging

### Validation Rules

- Use Pydantic Field constraints: `ge` (>=), `le` (<=), `gt` (>), `lt` (<)
- Population size must be even (NSGA-II requirement)
- Probabilities must be 0.0-1.0
- File paths can be relative or absolute
- Valid environments: "test", "prod"

### Never Do

- ❌ Import `config.ga_params` or `config.constraints` (removed in refactor)
- ❌ Hardcode configuration values in source files
- ❌ Access config before `init_config()` called (main.py does this)
- ❌ Modify config object at runtime (read-only after init)
- ❌ Duplicate common settings in environment files (put in base.yaml)

## Examples

### Adding a New GA Parameter

```python
# In src/config/models.py
class GAConfig(BaseModel):
    # ... existing fields ...
    tournament_size: int = Field(default=3, ge=2, le=10)
```

Then add to `configs/base.yaml`:
```yaml
ga:
  # ... existing ...
  tournament_size: 3
```

### Custom Validator

```python
@field_validator("num_workers")
@classmethod
def validate_workers(cls, v):
    if v is not None and v > os.cpu_count():
        raise ValueError(f"num_workers ({v}) exceeds CPU count ({os.cpu_count()})")
    return v
```

### GPU Configuration Example

```yaml
# configs/base.yaml
rl:
  device: cuda  # Enable GPU acceleration (auto, cuda, cpu)
  agent:
    type: ppo
    model_path: models/rl_agents/best_model.zip
```

**Important**: GPU acceleration is enabled by default. See `docs/04-algorithms/nvidia-gpu/QUICKSTART.md` for setup.
