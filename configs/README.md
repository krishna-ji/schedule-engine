# Experiment Configuration System

**Status**: ✅ Active (Python-based, no YAML)

---

## Overview

Experiments are defined as Python modules that instantiate **blueprint classes** with explicit **killswitches**. Each experiment clearly documents which features are enabled/disabled.

## Architecture

```
src/config/presets/blueprints.py  # Reusable algorithm blueprints
configs/                           # Experiment-specific configs
├── experiment_a_baseline.py       # Experiment A: Pure NSGA-II
├── experiment_b_memetic.py        # Experiment B: + Memetic repairs
├── experiment_c_roundrobin.py     # Experiment C: + Round-robin heuristics
├── experiment_d_adaptive.py       # Experiment D: + Adaptive selection
├── experiment_e_rl_guided.py      # Experiment E: + RL guidance
└── archive/                       # Deprecated YAML files
```

## Usage

### Method 1: Direct Import (Recommended)

```python
from configs.experiment_a_baseline import get_config
from src.config.presets.profiles import Profile

# Get Experiment A config with TEST profile
config = get_config(Profile.TEST)

# Get Experiment E config with PROD profile
from configs.experiment_e_rl_guided import get_config as get_e_config
config = get_e_config(Profile.PROD)
```

### Method 2: Import Blueprint Instance

```python
from configs import experiment_a, experiment_e
from src.config.presets.profiles import Profile

# Use blueprint directly
config_a = experiment_a.build(Profile.TEST)
config_e = experiment_e.build(Profile.PROD)
```

### Method 3: Use Blueprint Classes Directly

```python
from src.config.presets.blueprints import RlGuidedBlueprint
from src.config.presets.profiles import Profile

# Instantiate blueprint
blueprint = RlGuidedBlueprint()
config = blueprint.build(Profile.PROD)
```

## Experiment Structure

Each experiment module (`experiment_X_name.py`) contains:

```python
# 1. Import blueprint class
from src.config.presets.blueprints import SomeBlueprint

# 2. Instantiate blueprint
experiment_x = SomeBlueprint()

# 3. Metadata
EXPERIMENT_ID = "X"
EXPERIMENT_NAME = "Descriptive Name"
EXPERIMENT_DESCRIPTION = "What this experiment tests"

# 4. Killswitches (explicit documentation)
KILLSWITCHES = {
    "repair.enabled": True/False,
    "rl.enabled": True/False,
    # ... other feature flags
}

# 5. Helper function
def get_config(profile: Profile = Profile.TEST):
    return experiment_x.build(profile)

# 6. Test code
if __name__ == "__main__":
    config = get_config()
    print(f"✓ {EXPERIMENT_NAME}")
    # ... verify killswitches
```

## Experiments

### Thesis Progressive Experiments (A-E)

| ID | Name | Blueprint | Description |
|----|------|-----------|-------------|
| A | Pure NSGA-II Baseline | `PureNsgaBlueprint` | Minimal NSGA-II (no repairs, no heuristics) |
| B | Memetic NSGA-II | `MemeticNsgaBlueprint` | + Memetic local search repairs |
| C | Round-Robin Heuristics | `RoundRobinHeuristicBlueprint` | + Fixed heuristic rotation |
| D | Adaptive Selection | `AdaptiveHeuristicBlueprint` | + Performance-based selection |
| E | RL-Guided | `RlGuidedBlueprint` | + RL-guided hyper-heuristic |

### Killswitch Summary

| Feature | A | B | C | D | E |
|---------|---|---|---|---|---|
| Repairs | ❌ | ✅ | ✅ | ✅ | ✅ |
| Memetic mode | ❌ | ✅ | ✅ | ✅ | ✅ |
| Heuristics | ❌ | ❌ | ✅ | ✅ | ✅ |
| Adaptive priority | ❌ | ❌ | ❌ | ✅ | ❌ |
| LNS | ❌ | ❌ | ❌ | ❌ | ✅ |
| RL | ❌ | ❌ | ❌ | ❌ | ✅ |
| Enhancements | ❌ | ❌ | ✅ | ✅ | ✅ |

## Profiles

- **TEST**: Smoke test (30 gens, 10 pop, ~2-5 min)
- **PROD**: Production (2000 gens, 400 pop, ~1-3 hours)
- **DEBUG**: Reserved for future use

## Blueprint Classes

Available in `src/config/presets/blueprints.py`:

- `PureNsgaBlueprint` - Minimal NSGA-II
- `MemeticNsgaBlueprint` - + Memetic repairs
- `RoundRobinHeuristicBlueprint` - + Round-robin heuristics
- `AdaptiveHeuristicBlueprint` - + Adaptive selection
- `RlGuidedBlueprint` - + RL guidance
- `FullStackNsgaBlueprint` - All features (no RL)
- `RlSpecialistBlueprint` - RL specialists
- `ArchiveDiversityBlueprint` - Archive diversity
- `HierarchicalRlBlueprint` - Hierarchical RL
- `MultiAgentRlBlueprint` - Multi-agent RL

## Creating Custom Experiments

### Option 1: New Experiment Module

```python
# configs/experiment_f_custom.py
from src.config.presets.blueprints import RlGuidedBlueprint
from src.config.presets.profiles import Profile

# Custom blueprint subclass
class CustomBlueprint(RlGuidedBlueprint):
    def additional_overrides(self, profile: Profile):
        return {
            "ga": {"cxpb": 0.9, "mutpb": 0.1},
            "rl": {"hybrid": {"rl_probability": 0.95}},
        }

experiment_f = CustomBlueprint()

EXPERIMENT_ID = "F"
EXPERIMENT_NAME = "My Custom Experiment"

def get_config(profile: Profile = Profile.TEST):
    return experiment_f.build(profile)
```

### Option 2: Direct Instantiation with Overrides

```python
from src.config.presets.blueprints import AdaptiveHeuristicBlueprint
from src.config.presets.profiles import Profile

blueprint = AdaptiveHeuristicBlueprint()
config = blueprint.build(Profile.PROD)

# Modify after build (if needed)
config.ga.cxpb = 0.85
```

## Testing Experiments

```bash
# Test individual experiment
uv run python configs/experiment_a_baseline.py
uv run python configs/experiment_e_rl_guided.py

# Import and test
uv run python -c "from configs import experiment_a; from src.config.presets.profiles import Profile; config = experiment_a.build(Profile.TEST); print(f'Repair: {config.repair.enabled}')"
```

## Migration from YAML

All YAML files have been archived. Use Python experiment modules instead.

**Old (deprecated)**:
```yaml
# configs/baseline/a-pure-nsga.yaml
repair:
  enabled: false
```

**New (required)**:
```python
# configs/experiment_a_baseline.py
from src.config.presets.blueprints import PureNsgaBlueprint

experiment_a = PureNsgaBlueprint()

KILLSWITCHES = {
    "repair.enabled": False,
}
```

## Benefits

✅ **Explicit Killswitches** - Clear documentation of enabled/disabled features  
✅ **Type Safety** - Pydantic validation at build time  
✅ **Testable** - Each experiment module is executable  
✅ **Reusable** - Blueprint classes can be shared across experiments  
✅ **Flexible** - Easy to extend, customize, override  
✅ **IDE Support** - Autocomplete, type hints, docstrings

---

## Overview

The configuration system uses **Python-native presets** with object-oriented blueprint inheritance. YAML files have been completely removed.

## Architecture

```
src/config/presets/        # Blueprint architecture
├── base.py                # ConfigBlueprint base class
├── blueprints.py          # 10 mode-specific blueprint classes
├── data.py                # BASE_DEFAULTS + PROFILE_OVERRIDES
├── profiles.py            # Profile enum (TEST/PROD/DEBUG)
├── registry.py            # RuntimeMode → Blueprint mapping
└── utils.py               # deep_merge + apply_dynamic_overrides

configs/                   # Pre-configured blueprint instances
├── baseline.py            # Mode A: Pure NSGA-II
├── memetic.py             # Mode B: + Memetic local search
├── roundrobin.py          # Mode C: + Round-robin heuristics
├── adaptive.py            # Mode D: + Adaptive selection
├── rl_guided.py           # Mode E: + RL-guided
└── archive/               # Deprecated YAML files (reference only)
```

## Usage

### Method 1: Runtime Mode Enum (Recommended)

```python
from src.config import load_config
from src.config.runtime_mode import RuntimeMode
from src.config.presets.profiles import Profile

# Load via RuntimeMode enum
config = load_config(RuntimeMode.RL_GUIDED, Profile.PROD)
```

### Method 2: Direct Blueprint Import

```python
from configs.rl_guided import rl_guided_blueprint
from src.config.presets.profiles import Profile

# Load via pre-configured blueprint
config = rl_guided_blueprint.build(Profile.PROD)
```

### Method 3: Custom Blueprint

```python
from src.config.presets.blueprints import ModeERlGuidedBlueprint
from src.config.presets.profiles import Profile

# Instantiate and configure
blueprint = ModeERlGuidedBlueprint()
config = blueprint.build(Profile.TEST)
```

## CLI Usage

```bash
# Run Mode A (Pure NSGA-II)
uv run python main.py --mode 1-pure-nsga --profile test

# Run Mode E (RL-Guided)
uv run python main.py --mode e-rl-guided --profile prod --experiment "thesis-rl-r01"

# List all modes
uv run python main.py --list-modes
```

## Profiles

- **TEST**: Smoke test (30 gens, 10 pop, ~2-5 min)
- **PROD**: Production (2000 gens, 400 pop, ~1-3 hours)
- **DEBUG**: Reserved for future use

## Blueprint Classes

### Progressive Thesis Modes (A-E)

| Class | Description |
|-------|-------------|
| `ModeAPureNsgaBlueprint` | Pure NSGA-II baseline |
| `ModeBNsgaMemeticBlueprint` | + Memetic local search |
| `ModeCRoundRobinBlueprint` | + Round-robin heuristics |
| `ModeDAdaptiveBlueprint` | + Adaptive selection |
| `ModeERlGuidedBlueprint` | + RL-guided hyper-heuristic |

### Numbered Feature Set (1-10)

| Class | Description |
|-------|-------------|
| `Mode1PureNsgaBlueprint` | Alias for Mode A |
| `Mode2NsgaRepairsBlueprint` | + Basic repairs |
| `Mode3NsgaHeuristicsBlueprint` | + Full heuristic toolbox |
| `Mode4NsgaFullBlueprint` | + Full stack (no RL) |
| `Mode5RlGuidedBlueprint` | Alias for Mode E |
| `Mode6RoundRobinBlueprint` | Alias for Mode C |
| `Mode7RlSpecialistsBlueprint` | + RL specialists |
| `Mode8ArchiveDiversityBlueprint` | + Archive diversity |
| `Mode9RlHierarchicalBlueprint` | + Hierarchical RL |
| `Mode10RlMultiAgentBlueprint` | + Multi-agent RL |

## Creating Custom Configs

### Option 1: Extend Blueprint Class

```python
from src.config.presets.blueprints import ModeERlGuidedBlueprint
from src.config.presets.profiles import Profile

class MyCustomBlueprint(ModeERlGuidedBlueprint):
    name = "My Custom Mode"

    def additional_overrides(self, profile: Profile):
        return {
            "ga": {"cxpb": 0.9, "mutpb": 0.1},
            "rl": {"hybrid": {"rl_probability": 0.9}},
        }

# Use it
blueprint = MyCustomBlueprint()
config = blueprint.build(Profile.PROD)
```

### Option 2: Create Config File

```python
# configs/my_custom.py
from src.config.presets.blueprints import ModeERlGuidedBlueprint

class MyCustomBlueprint(ModeERlGuidedBlueprint):
    name = "My Custom Mode"

    def base_overrides(self, profile):
        overrides = super().base_overrides(profile)
        overrides["ga"]["cxpb"] = 0.9
        return overrides

my_custom_blueprint = MyCustomBlueprint()
```

## Migration from YAML

All YAML files have been moved to `configs/archive/` for reference only. The system no longer reads YAML files.

**Old (deprecated)**:
```python
from src.config import get_config
config = get_config()  # ❌ No longer works
```

**New (required)**:
```python
from src.config import load_config
from src.config.runtime_mode import RuntimeMode
from src.config.presets.profiles import Profile

config = load_config(RuntimeMode.RL_GUIDED, Profile.PROD)  # ✅
```

## Benefits

✅ **Type Safety**: Pydantic validation catches errors at config build time  
✅ **Maintainability**: OOP inheritance eliminates copy-paste duplication  
✅ **Flexibility**: Dynamic overrides + profile system + runtime modes  
✅ **IDE Support**: Autocomplete, type hints, docstrings in blueprint classes  
✅ **No YAML Parsing**: Faster config loading, no YAML syntax errors

## Documentation

- **Implementation**: `docs/06-development/implementation-notes/PYTHON_CONFIG_MIGRATION.md`
- **User Guide**: `docs/02-user-guides/runtime-modes.md` (TODO: update)
- **Copilot Instructions**: `.github/copilot-instructions.md` (TODO: update)

---

## NEW: Dataclass Configs

See **docs/dataclass-config-guide.md** for Python dataclass-based configs with IDE support!
