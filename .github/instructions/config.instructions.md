---[{

applyTo: "{config/**/*.py,configs/**/*.yaml}"	"resource": "/c:/Users/krishna/Desktop/schedule-engine/.github/instructions/config.instructions.md",

---	"owner": "prompts-diagnostics-provider",

	"severity": 8,

# Configuration System Instructions	"message": "The 'applyTo' attribute must be a string.",

	"startLineNumber": 3,

## Overview	"startColumn": 3,

YAML-based configuration with Pydantic validation. All runtime settings defined in `configs/{test,dev,prod}.yaml`, loaded via `config/loader.py`, validated by `config/models.py`.	"endLineNumber": 4,

	"endColumn": 24

## Key Files},{

- `config/models.py` - Pydantic models (Config, GAConfig, RepairConfig, etc.)	"resource": "/c:/Users/krishna/Desktop/schedule-engine/.github/instructions/ga-core.instructions.md",

- `config/loader.py` - YAML loading logic with defaults	"owner": "prompts-diagnostics-provider",

- `config/__init__.py` - Global config object + `init_config()`, `get_config()`	"severity": 8,

- `configs/{test,dev,prod}.yaml` - Environment-specific settings	"message": "The 'applyTo' attribute must be a string.",

	"startLineNumber": 3,

## Rules	"startColumn": 3,

	"endLineNumber": 4,

### Accessing Configuration	"endColumn": 21

```python}]r.py` - YAML loading logic with defaults

# Runtime access (anywhere in code)- `config/__init__.py` - Global config object + `init_config()`, `get_config()`

from config import get_config- `configs/{test,dev,prod}.yaml` - Environment-specific settings

config = get_config()

ngen = config.ga.ngen## Rules

```

### Accessing Configuration

### Adding New Settings```python

1. Add field to appropriate Pydantic model in `config/models.py`# Runtime access (anywhere in code)

2. Include default value with type annotationfrom config import get_config

3. Add validation if needed (`@field_validator`)config = get_config()

4. Update all YAML files in `configs/` with the new fieldngen = config.ga.ngen

5. Update `Config.summary()` if user-facing```



### YAML File Structure### Adding New Settings

- Use lowercase keys with underscores (e.g., `pop_size`, not `popSize`)1. Add field to appropriate Pydantic model in `config/models.py`

- Group related settings under sections (ga, repair, parallel, etc.)2. Include default value with type annotation

- Include comments for non-obvious settings3. Add validation if needed (`@field_validator`)

- Keep test.yaml minimal (fast), dev.yaml balanced, prod.yaml comprehensive4. Update all YAML files in `configs/` with the new field

5. Update `Config.summary()` if user-facing

### Validation Rules

- Use Pydantic Field constraints: `ge` (>=), `le` (<=), `gt` (>), `lt` (<)### YAML File Structure

- Population size must be even (NSGA-II requirement)- Use lowercase keys with underscores (e.g., `pop_size`, not `popSize`)

- Probabilities must be 0.0-1.0- Group related settings under sections (ga, repair, parallel, etc.)

- File paths can be relative or absolute- Include comments for non-obvious settings

- Keep test.yaml minimal (fast), dev.yaml balanced, prod.yaml comprehensive

### Never Do

- ❌ Import `config.ga_params` or `config.constraints` (removed in refactor)### Validation Rules

- ❌ Hardcode configuration values in source files- Use Pydantic Field constraints: `ge` (>=), `le` (<=), `gt` (>), `lt` (<)

- ❌ Access config before `init_config()` called (main.py does this)- Population size must be even (NSGA-II requirement)

- ❌ Modify config object at runtime (read-only after init)- Probabilities must be 0.0-1.0

- File paths can be relative or absolute

## Examples

### Never Do

### Adding a New GA Parameter- ❌ Import `config.ga_params` or `config.constraints` (removed in refactor)

```python- ❌ Hardcode configuration values in source files

# In config/models.py- ❌ Access config before `init_config()` called (main.py does this)

class GAConfig(BaseModel):- ❌ Modify config object at runtime (read-only after init)

    # ... existing fields ...

    tournament_size: int = Field(default=3, ge=2, le=10)## Examples

```

### Adding a New GA Parameter

### Custom Validator```python

```python# In config/models.py

@field_validator("num_workers")class GAConfig(BaseModel):

@classmethod    # ... existing fields ...

def validate_workers(cls, v):    tournament_size: int = Field(default=3, ge=2, le=10)

    if v is not None and v > os.cpu_count():```

        raise ValueError(f"num_workers ({v}) exceeds CPU count ({os.cpu_count()})")

    return v### Custom Validator

``````python

@field_validator("num_workers")
@classmethod
def validate_workers(cls, v):
    if v is not None and v > os.cpu_count():
        raise ValueError(f"num_workers ({v}) exceeds CPU count ({os.cpu_count()})")
    return v
```
