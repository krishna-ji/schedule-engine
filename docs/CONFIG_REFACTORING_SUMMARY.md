# Configuration System Refactoring - Complete Summary

## Question

> "Why are you storing hard values in config/models.py? Keep that inside a common yaml file? What is better?"

## Answer

**You're absolutely right!** YAML should be the source of truth for configuration values, not Python code.

## What Was Wrong

### Before (❌ Bad Practice)

**Problem 1: Hardcoded Defaults in Python**
```python
# config/models.py
class TimeConfig(BaseModel):
    quantum_minutes: int = Field(default=60, ge=15, le=120)  # Hardcoded!
    earliest_preferred_time: str = "10:00"  # Hardcoded!
    theory_isolated_penalty: int = Field(default=2, ...)  # Hardcoded!
```

**Problem 2: Massive Duplication in YAML Files**
```yaml
# test.yaml (157 lines)
time:
  quantum_minutes: 60
  earliest_preferred_time: "10:00"
  # ... 50 more lines

# dev.yaml (165 lines) - SAME VALUES!
time:
  quantum_minutes: 60
  earliest_preferred_time: "10:00"
  # ... 50 more lines

# prod.yaml (165 lines) - SAME VALUES AGAIN!
time:
  quantum_minutes: 60
  earliest_preferred_time: "10:00"
  # ... 50 more lines
```

**Total**: 487 lines, ~80% duplication 😞

## What's Fixed Now

### After (✅ Best Practice)

**Solution 1: YAML is Source of Truth**
```python
# config/models.py - Validation ONLY, no defaults!
class TimeConfig(BaseModel):
    quantum_minutes: int = Field(ge=15, le=120)  # No default!
    earliest_preferred_time: str  # No default!
    theory_isolated_penalty: int = Field(ge=0, le=100)  # No default!
```

```yaml
# configs/common.yaml - Defaults here!
time:
  quantum_minutes: 60
  earliest_preferred_time: "10:00"
  theory_isolated_penalty: 2
```

**Solution 2: DRY with Inheritance**
```yaml
# common.yaml (150 lines) - All defaults
time:
  quantum_minutes: 60
  theory_isolated_penalty: 2
  practical_fragmentation_penalty: 20

# test.yaml (15 lines) - ONLY overrides
ga:
  ngen: 10
  pop_size: 4

# dev.yaml (25 lines) - ONLY overrides
ga:
  ngen: 100
  pop_size: 100

# prod.yaml (50 lines) - ONLY overrides
ga:
  ngen: 2000
  pop_size: 200
time:
  theory_isolated_penalty: 3  # Override common
  practical_fragmentation_penalty: 50  # Override common
```

**Total**: 240 lines, zero duplication 😊

## Architecture

### File Structure
```
configs/
├── common.yaml      # Source of truth for all defaults
├── test.yaml        # Minimal overrides for testing
├── dev.yaml         # Minimal overrides for development
└── prod.yaml        # Minimal overrides for production
```

### Loading Strategy
```python
def load_config(environment):
    common = load_yaml("configs/common.yaml")
    env_specific = load_yaml(f"configs/{environment}.yaml")
    
    # Deep merge: environment overrides common
    return deep_merge(common, env_specific)
```

### What Goes Where?

| Category | Goes In | Reason |
|----------|---------|--------|
| Time settings | `common.yaml` | Same for all environments |
| I/O paths | `common.yaml` | Same for all environments |
| Calendar settings | `common.yaml` | Same for all environments |
| Default constraint weights | `common.yaml` | Baseline for test/dev |
| Default penalties | `common.yaml` | Baseline for test/dev |
| **ngen, pop_size** | `test/dev/prod.yaml` | **Tuning parameters** |
| **Repair iterations** | `test/dev/prod.yaml` | **Environment-specific** |
| **Stricter penalties (prod)** | `prod.yaml` | **Quality vs speed tradeoff** |
| **Higher weights (prod)** | `prod.yaml` | **Production quality** |

## Benefits Achieved

### 1. ✅ No Duplication (DRY Principle)
- Common values: **ONE place** (`common.yaml`)
- Zero repetition across test/dev/prod
- Easy to change defaults

### 2. ✅ YAML as Source of Truth
- All values in YAML, not Python code
- Python only validates (Field constraints)
- Config files are self-documenting

### 3. ✅ Clear Intent
- Test/dev/prod show **only differences**
- Easy to see what varies per environment
- Clear what's being tuned

### 4. ✅ Size Reduction
- **487 lines → 240 lines** (50% reduction)
- Easier to read and maintain
- Less chance of copy-paste errors

### 5. ✅ Easy Maintenance
- Change common setting: **one edit** in `common.yaml`
- Add new parameter: Define once, override where needed
- No hunting through multiple files

## Comparison

### Duplication Example

**Before (❌):**
```yaml
# In test.yaml, dev.yaml, AND prod.yaml
time:
  quantum_minutes: 60
  earliest_preferred_time: "10:00"
  latest_preferred_time: "17:00"
  midday_break_start: "12:00"
  midday_break_end: "14:00"
  max_session_coalescence: 3
  preferred_block_size_min: 2
  preferred_block_size_max: 3
  # ... repeated 3 times!
```

**After (✅):**
```yaml
# In common.yaml ONCE
time:
  quantum_minutes: 60
  earliest_preferred_time: "10:00"
  # ... all defaults

# In prod.yaml ONLY overrides
time:
  theory_isolated_penalty: 3  # Only this changes
```

### Adding New Parameter

**Before (❌):**
1. Add to `config/models.py` with hardcoded default
2. Add to `test.yaml` (if different)
3. Add to `dev.yaml` (if different)
4. Add to `prod.yaml` (if different)
5. Hope you didn't miss anything!

**After (✅):**
1. Add to `config/models.py` (validation only, no default)
2. Add to `common.yaml` with default value
3. Override in prod.yaml if needed
4. Done!

## Validation

All systems tested and working:

```bash
✅ Test config: common.yaml + test.yaml
   ngen=10, pop_size=4, quantum_minutes=60 (from common)

✅ Dev config: common.yaml + dev.yaml
   ngen=100, pop_size=100, quantum_minutes=60 (from common)

✅ Prod config: common.yaml + prod.yaml
   ngen=2000, theory_penalty=3 (override), quantum_minutes=60 (from common)

✅ Block clustering tests: 8/8 passed

✅ Backward compatibility: Standalone configs still work
```

## Conclusion

### Why YAML > Python for Defaults?

1. **Separation of Concerns**: Configuration ≠ Code
2. **Easy to Edit**: YAML is user-friendly, no Python knowledge needed
3. **Version Control**: Easier to see config changes in git diffs
4. **Runtime Override**: Can change without recompiling/restarting
5. **Self-Documenting**: YAML structure shows relationships

### Why Common File > Duplication?

1. **DRY Principle**: "Don't Repeat Yourself"
2. **Single Source of Truth**: One place for defaults
3. **Consistency**: Impossible to have different "default" values
4. **Maintainability**: Change once, affects all environments
5. **Clarity**: Override files show only what's different

## Files Changed

- ✅ `configs/common.yaml` - **NEW** - All common defaults
- ✅ `configs/test.yaml` - **MINIMAL** - 15 lines (was 157)
- ✅ `configs/dev.yaml` - **MINIMAL** - 25 lines (was 165)
- ✅ `configs/prod.yaml` - **MINIMAL** - 50 lines (was 165)
- ✅ `config/loader.py` - Added deep_merge() and common loading
- ✅ `config/models.py` - Removed defaults, kept validation
- ✅ `docs/CONFIG_REFACTORING.md` - Architecture documentation
- ✅ `docs/code/ENHANCE.md` - Changelog entry

---

**Answer to your question**: ✅ **YAML is better! Refactoring complete.**

The configuration system now follows best practices:
- **YAML as source of truth** (not Python)
- **DRY principle** (common + overrides)
- **Clear separation** (defaults vs tuning parameters)
- **Easy maintenance** (change once, affect all)

**Status**: ✅ Complete, Tested, Documented, Best Practice Implemented
