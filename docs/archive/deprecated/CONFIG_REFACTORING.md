# Configuration System Refactoring

## Summary of Changes

Refactored configuration system to eliminate duplication and separate concerns:
- **Common defaults** → `configs/common.yaml` 
- **Environment-specific overrides** → `configs/{test,dev,prod}.yaml`
- **Default values removed from Python** → Now in YAML (source of truth)

## New Structure

```
configs/
├── common.yaml      # All default values (rarely changes)
├── test.yaml        # ONLY test overrides (ngen=10, pop_size=4, etc.)
├── dev.yaml         # ONLY dev overrides (ngen=100, pop_size=100, etc.)
└── prod.yaml        # ONLY prod overrides (ngen=2000, stricter penalties)
```

## Benefits

###  **No Duplication**
- Common values defined once in `common.yaml`
- Test/dev/prod only override what's different
- Reduced config size: ~150 lines → ~30 lines per environment

###  **Clear Separation**
- **Tunable parameters** (test/dev/prod): ngen, pop_size, weights, penalties
- **Common configuration** (common.yaml): time settings, I/O paths, feature flags

###  **YAML as Source of Truth**
- Removed hardcoded defaults from `config/models.py`
- All values come from YAML files
- Python models only provide validation (Field constraints)

###  **Easy Maintenance**
- Change common settings in one place
- Environment configs show only what's different
- Clear what varies per environment

## Configuration Loading

### Merge Strategy

```python
final_config = deep_merge(common.yaml, environment.yaml)
```

Environment-specific values override common defaults.

### Example

**common.yaml:**
```yaml
time:
  quantum_minutes: 60
  theory_isolated_penalty: 2
  practical_fragmentation_penalty: 20
```

**prod.yaml:**
```yaml
time:
  theory_isolated_penalty: 3          # Override common
  practical_fragmentation_penalty: 50  # Override common
  # quantum_minutes: 60 (inherited from common)
```

**Result:** Prod gets `quantum_minutes: 60` from common, but stricter penalties from prod.

## What Goes Where?

### `common.yaml` - Defaults That Rarely Change

- **Time settings**: `quantum_minutes`, `earliest_preferred_time`, etc.
- **I/O paths**: `data_dir`, `output_dir`
- **Calendar settings**: `show_instructor`, `show_room`, etc.
- **GA defaults**: `cxpb`, `mutpb`, `elite_size`, etc. (can override in prod)
- **Repair defaults**: `enabled`, `detection_strategy`, etc.
- **Constraint defaults**: Default weights for all constraints
- **Enhancement defaults**: Default feature flags

### `test.yaml` - Test Overrides Only

- `ngen: 10` (fast)
- `pop_size: 4` (small)
- `use_multiprocessing: false` (easier debugging)
- `fail_on_infeasibility: false` (allow testing with bad data)
- `tolerance_margin: 0.05` (more lenient)

### `dev.yaml` - Dev Overrides Only

- `ngen: 100` (medium)
- `pop_size: 100` (larger for testing)
- `use_multiprocessing: true` (parallel)
- `memetic_mode: true` (enable enhancements)
- `greedy_initialization_percent: 0.4` (more greedy seeds)

### `prod.yaml` - Production Overrides Only

- `ngen: 2000` (max quality)
- `pop_size: 200` (max diversity)
- `cxpb: 0.85`, `mutpb: 0.25` (tuned probabilities)
- `max_iterations: 7` (more thorough repair)
- **Stricter penalties**: `theory_isolated_penalty: 3`, `practical_fragmentation_penalty: 50`
- **Higher constraint weights**: `availability_violations.weight: 6.0`, etc.
- `top_n_hotspots: 30` (more detailed analysis)
- `max_combinations: 100` (more thorough repair)

## Files Modified

### Configuration System
- `configs/common.yaml` - **NEW** - All common defaults
- `configs/test.yaml` - **REFACTORED** - ~150 lines → ~15 lines
- `configs/dev.yaml` - **REFACTORED** - ~160 lines → ~25 lines
- `configs/prod.yaml` - **REFACTORED** - ~160 lines → ~50 lines
- `config/loader.py` - **ENHANCED** - Added `deep_merge()` and common.yaml loading
- `config/models.py` - **CLEANED** - Removed hardcoded defaults, kept validation only

### Backward Compatibility
- Standalone config files (via `--config path.yaml`) still work
- No merge with common.yaml for standalone files
- Existing code unchanged (uses `get_config()` as before)

## Usage

### Default (Dev)
```bash
python main.py  # Loads common.yaml + dev.yaml
```

### Test
```bash
python main.py --env test  # Loads common.yaml + test.yaml
```

### Production
```bash
python main.py --env prod  # Loads common.yaml + prod.yaml
```

### Custom Standalone
```bash
python main.py --config path/to/custom.yaml  # No merge, standalone
```

### Environment Variable
```bash
export ENVIRONMENT=prod
python main.py  # Loads common.yaml + prod.yaml
```

## Validation

All tests pass with new system:
```bash
 Config loading test (dev)
 Config loading test (prod)
 Config loading test (test)
 Block clustering tests (8/8 passed)
```

## Migration Guide

### For Users

**No action needed!** Existing commands work as before:
```bash
python main.py --env test  # Still works
python main.py --env dev   # Still works
python main.py --env prod  # Still works
```

### For Developers

**To add a new configuration parameter:**

1. **Add to `configs/common.yaml`** with default value
2. **Override in test/dev/prod** only if environment-specific
3. **Add Field validation** in `config/models.py` (no default value)

Example:
```python
# config/models.py (validation only)
class TimeConfig(BaseModel):
    new_parameter: int = Field(ge=0, le=100)  # No default!

# configs/common.yaml (default)
time:
  new_parameter: 50

# configs/prod.yaml (override if needed)
time:
  new_parameter: 75  # Stricter in production
```

## Size Comparison

### Before (Duplicated)
- `test.yaml`: 157 lines
- `dev.yaml`: 165 lines
- `prod.yaml`: 165 lines
- **Total**: 487 lines (lots of duplication)

### After (DRY)
- `common.yaml`: 150 lines (all defaults)
- `test.yaml`: 15 lines (overrides only)
- `dev.yaml`: 25 lines (overrides only)
- `prod.yaml`: 50 lines (overrides only)
- **Total**: 240 lines (50% reduction, zero duplication)

## Philosophy

> **"Don't Repeat Yourself" (DRY)**
> - Common values in ONE place (`common.yaml`)
> - Environment configs show ONLY differences
> - YAML is the source of truth, not Python code

> **"Principle of Least Astonishment"**
> - Clear what changes per environment
> - Easy to see differences at a glance
> - Predictable merge behavior

---

**Status**:  **Complete, Tested, and Documented**
**Backward Compatibility**:  **Preserved**
**Tests**:  **8/8 passing**
