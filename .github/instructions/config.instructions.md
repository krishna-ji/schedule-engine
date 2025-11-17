---
applyTo: "src/config/**/*.py"
---

# Configuration System Instructions

## Overview

YAML-based configuration with inheritance: `base.yaml` + environment overrides. Loaded via `src/config/loader.py`, validated by `src/config/models.py`.

## Key Files

- `src/config/models.py` - Pydantic models (Config, GAConfig, RepairConfig, etc.)
- `src/config/loader.py` - YAML loading with deep merge (base.yaml + env overrides)
- `src/config/__init__.py` - Global config object + `init_config()`, `get_config()`
- `configs/base.yaml` - Common settings (shared by all environments)
- `configs/{test,prod}.yaml` - Environment-specific overrides only

## Configuration Inheritance

1. Loader reads `configs/base.yaml` (all common settings)
2. Loader reads environment file (e.g., `configs/prod.yaml`)
3. Deep merge: environment values override base values
4. Result: complete configuration with minimal duplication

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
4. Add to `configs/base.yaml` (if common to all environments)
5. Override in specific env files (if environment-specific)
6. Update `Config.summary()` if user-facing

### YAML File Structure

- Use lowercase keys with underscores (e.g., `pop_size`, not `popSize`)
- Group related settings under sections (ga, repair, parallel, etc.)
- Include comments for non-obvious settings
- **base.yaml**: Contains ALL common settings (used by all environments)
- **Environment files**: Only override what differs (ngen, pop_size, parallel, repair triggers, etc.)
- Keep test.yaml minimal (fast), prod.yaml comprehensive

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

**Important**: GPU acceleration is enabled by default. See `docs/05-performance/nvidia-gpu/QUICKSTART.md` for setup.
