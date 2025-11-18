---
applyTo: "src/config/**/*.py"
---

# Configuration System Instructions

## Overview

YAML-based configuration with inheritance: `base.yaml` + environment overrides + runtime modes. Loaded via `src/config/loader.py`, validated by `src/config/models.py`.

## Key Files

- `src/config/models.py` - Pydantic models (Config, GAConfig, RepairConfig, etc.)
- `src/config/runtime_mode.py` - RuntimeMode enum, ExperimentConfig, validation
- `src/config/loader.py` - YAML loading with deep merge (base.yaml + env overrides + runtime modes)
- `src/config/__init__.py` - Global config object + `init_config()`, `get_config()`
- `configs/base.yaml` - Common settings (shared by all environments)
- `configs/{test,prod}.yaml` - Environment-specific overrides only
- `configs/{baseline,nsga,rl,hybrid}/*.yaml` - Runtime mode configurations with killswitches

## Configuration Inheritance

1. Loader reads `configs/base.yaml` (all common settings)
2. If runtime mode specified: Loader reads mode config (e.g., `configs/rl/5-rl-guided.yaml`)
3. Loader reads environment file (e.g., `configs/prod.yaml`)
4. Deep merge: mode values override base, env values override mode
5. Automatic validation: RuntimeMode.validate_config() checks killswitches
6. Result: complete configuration with minimal duplication

**Priority order**: env overrides → runtime mode → base.yaml (highest to lowest)

**Note**: Training uses separate configs in `config-train/` with similar inheritance (base + profile overrides).

## Runtime Modes

6 progressive modes for systematic experimentation:
- **Mode 1 (baseline)**: Pure NSGA-II, all killswitches OFF
- **Mode 2 (repairs)**: NSGA-II + IGLS repairs only
- **Mode 3 (heuristics)**: NSGA-II + repairs + 19 heuristic operators
- **Mode 4 (full)**: Full GA (best non-RL configuration)
- **Mode 5 (rl)**: RL-guided heuristic selection
- **Mode 6 (roundrobin)**: Fixed round-robin heuristic rotation

See `docs/02-user-guides/runtime-modes.md` for complete guide.

## Killswitch Pattern

Major features use master killswitches for easy toggling:

```yaml
# Master killswitch example
rl:
  enabled: false  # Master switch - when false, all RL features disabled
  mode: inference
  agent:
    type: ppo
    # ... other settings ...

repair:
  enabled: true   # Master switch for repair system
  mode: selective
  # ... other settings ...

enhancements:
  master_enabled: false  # Master switch for all enhancement features
  memetic_mode: false
  lns:
    enabled: false
  # ... other settings ...
```

**Killswitch validation** happens automatically in `RuntimeMode.validate_config()`.

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

### Runtime Mode Usage

```python
# Load config with runtime mode
from src.config.runtime_mode import RuntimeMode
from src.config.loader import load_config

config = load_config(runtime_mode=RuntimeMode.RL_GUIDED)
# Automatic validation ensures killswitches are set correctly

# CLI usage
python main.py --mode rl --env test
# or
uv run rl  # Uses test env by default
```

### Adding New Killswitches

When introducing experimental features:

1. **Add master switch to base.yaml**:
```yaml
my_feature:
  enabled: false  # Master killswitch
  setting1: value1
  setting2: value2
```

2. **Create runtime mode config** (if needed):
```yaml
# configs/experimental/my-feature.yaml
my_feature:
  enabled: true  # Override for this mode
  setting1: optimized_value
```

3. **Add validation to RuntimeMode**:
```python
# In src/config/runtime_mode.py
if self == RuntimeMode.MY_FEATURE:
    assert config_dict.get("my_feature", {}).get("enabled") is True
```

4. **Check killswitch in code**:
```python
from src.config import get_config

def my_function():
    config = get_config()
    if not config.my_feature.enabled:
        return  # Feature disabled, skip
    # ... feature implementation ...
```

### Never Do

- ❌ Import `config.ga_params` or `config.constraints` (removed in refactor)
- ❌ Hardcode configuration values in source files
- ❌ Access config before `init_config()` called (main.py does this)
- ❌ Modify config object at runtime (read-only after init)
- ❌ Duplicate common settings in environment files (put in base.yaml)
- ❌ Add features without master killswitches (always add `enabled: bool` field)
- ❌ Skip runtime mode validation (always use `RuntimeMode.validate_config()`)
- ❌ Hardcode mode-specific behavior (use config-driven approach)

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
