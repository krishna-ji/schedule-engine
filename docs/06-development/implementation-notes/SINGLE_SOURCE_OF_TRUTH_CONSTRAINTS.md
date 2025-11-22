# Single Source of Truth - Constraint System Implementation

**Date**: November 22, 2025  
**Status**: ✅ Complete  
**Impact**: Eliminates duplication, ensures consistency across codebase

## Problem Statement

Previously, constraint names and metadata were **duplicated** across multiple files:
1. Function definitions in `src/constraints/hard.py` and `soft.py`
2. Manual hardcoded dict in `src/workflows/standard_run.py` (60+ lines)
3. Evaluator imports and usage scattered across codebase
4. No single source defining the hc1-hc8 → constraint name mapping

This caused:
- **Maintenance burden**: Adding a constraint required updating 3+ files
- **Risk of inconsistency**: Names could mismatch between definition and usage
- **Unclear ordering**: hc1-hc8 mapping was implicit and undocumented
- **Documentation drift**: No definitive reference for constraint codes

## Solution: Decorator-Based Registry

### Architecture

**Registry Location**: `src/constraints/registry.py`

All constraints are registered via decorators when the module is imported:

```python
from src.constraints.registry import hard_constraint

@hard_constraint(
    name="student_group_exclusivity",
    description="Ensures each student group can only be in one session at a time",
    default_weight=3.0,
    needs_courses=False
)
def student_group_exclusivity(sessions: List[CourseSession]) -> int:
    # Implementation
    return violations
```

### Key Components

1. **Decorator Functions** (`src/constraints/registry.py`)
   - `@hard_constraint(...)` - Registers hard constraint with metadata
   - `@soft_constraint(...)` - Registers soft constraint with metadata

2. **Global Registries**
   - `_HARD_CONSTRAINTS: Dict[str, ConstraintMetadata]`
   - `_SOFT_CONSTRAINTS: Dict[str, ConstraintMetadata]`

3. **Access Functions**
   - `get_all_hard_constraints()` - Get all registered hard constraints
   - `get_all_soft_constraints()` - Get all registered soft constraints
   - `get_enabled_hard_constraints()` - Get enabled constraints from config
   - `get_constraint_metadata(name)` - Get metadata for specific constraint

### Order Preservation (Critical for hc1-hc8 Mapping)

**Registration Order** (`src/constraints/hard.py`):
```python
@hard_constraint(name="student_group_exclusivity", ...)  # hc1
@hard_constraint(name="instructor_exclusivity", ...)     # hc2
@hard_constraint(name="instructor_qualifications", ...)  # hc3
@hard_constraint(name="instructor_time_availability", ...) # hc4
@hard_constraint(name="room_suitability", ...)           # hc5
@hard_constraint(name="room_exclusivity", ...)           # hc6
@hard_constraint(name="room_time_availability", ...)     # hc7
@hard_constraint(name="course_completeness", ...)        # hc8
```

**Order is deterministic** because:
1. Python 3.7+ preserves dict insertion order
2. Decorators execute in source code order
3. Registry dict `.keys()` returns items in insertion order

## Changes Made

### 1. Removed Duplication in `standard_run.py`

**Before** (60+ lines of hardcoded dict):
```python
hard_constraints_dict = {
    "student_group_exclusivity": {
        "enabled": config.hard_constraints.student_group_exclusivity.enabled,
        "weight": config.hard_constraints.student_group_exclusivity.weight,
    },
    "instructor_exclusivity": {
        "enabled": config.hard_constraints.instructor_exclusivity.enabled,
        "weight": config.hard_constraints.instructor_exclusivity.weight,
    },
    # ... 6 more constraints ...
}

hard_names = [name for name, cfg in hard_constraints_dict.items() if cfg["enabled"]]
```

**After** (8 lines using registry):
```python
from src.constraints.registry import get_all_hard_constraints

all_hard_constraints = get_all_hard_constraints()

hard_names = []
for name in all_hard_constraints.keys():
    constraint_cfg = getattr(config.hard_constraints, name, None)
    if constraint_cfg and constraint_cfg.enabled:
        hard_names.append(name)
```

### 2. Existing Registry Usage (Already Implemented)

These files **already use the registry** correctly:
- ✅ `src/ga/evaluator/fitness.py` - Uses `get_enabled_hard_constraints()`
- ✅ `src/ga/evaluator/detailed_fitness.py` - Uses `get_enabled_hard_constraints()`
- ✅ `src/lns/conflict_detection.py` - Uses `get_enabled_hard_constraints()`
- ✅ `scripts/benchmarking/bench_constraint_check.py` - Uses registry

### 3. Code Generation (`ga_scheduler.py`)

**Location**: `src/core/ga_scheduler.py` lines 315-322

```python
# Deterministic short codes (hc1, hc2, ...) for console output
self.hard_constraint_codes = {
    name: f"hc{i+1}" for i, name in enumerate(self.hard_constraint_names)
}
```

The `hard_constraint_names` list comes from `standard_run.py` which now uses the registry.

### 4. Reference Documentation

**Created**: `docs/03-architecture/CONSTRAINT_MAPPING_REFERENCE.md`

Definitive mapping table:
```
hc1 = student_group_exclusivity
hc2 = instructor_exclusivity
hc3 = instructor_qualifications
hc4 = instructor_time_availability
hc5 = room_suitability
hc6 = room_exclusivity
hc7 = room_time_availability
hc8 = course_completeness
```

## Benefits

### 1. Single Source of Truth
- Constraint names defined **once** in `src/constraints/hard.py` via decorator
- Metadata (description, weight, needs_courses) co-located with function
- No duplication, no risk of mismatch

### 2. Easy to Add Constraints
**Before** (3 files to update):
1. Add function in `hard.py`
2. Update hardcoded dict in `standard_run.py`
3. Update config schema

**After** (2 files to update):
1. Add function with decorator in `hard.py`
2. Update config schema

The evaluator automatically picks up new constraints via registry!

### 3. Type-Safe and Self-Documenting
```python
@hard_constraint(
    name="my_new_constraint",          # Visible in config
    description="Human-readable desc", # Auto-documentation
    default_weight=2.5,                # Default config value
    needs_courses=True                 # Signature requirement
)
def my_new_constraint(sessions, courses):
    return violations
```

### 4. Dynamic Config Generation
```python
from src.constraints.registry import generate_constraint_config_template

# Auto-generate config from registered constraints
config_template = generate_constraint_config_template()
```

Useful for:
- Validating config files have all required constraints
- Generating documentation
- Creating default configs

## Validation

### Test Cases

1. **Registry completeness**: All constraints in `hard.py` are registered
   ```python
   from src.constraints.registry import get_all_hard_constraints
   assert len(get_all_hard_constraints()) == 8  # All 8 hard constraints
   ```

2. **Order preservation**: Constraint order matches expected hc1-hc8 mapping
   ```python
   names = list(get_all_hard_constraints().keys())
   assert names[0] == "student_group_exclusivity"  # hc1
   assert names[7] == "course_completeness"        # hc8
   ```

3. **Evaluator consistency**: Enabled constraints match config
   ```python
   from src.constraints.registry import get_enabled_hard_constraints
   enabled = get_enabled_hard_constraints()
   # Returns only constraints enabled in config
   ```

### Integration Test

Run the scheduler and verify console output uses correct codes:
```bash
uv run nsga --test
# Output should show: hc1=X, hc2=Y, ..., hc8=Z
# Where order matches the registry
```

## Maintenance Guidelines

### Adding a New Constraint

1. **Define function with decorator** in `src/constraints/hard.py`:
   ```python
   @hard_constraint(
       name="my_new_constraint",
       description="What it enforces",
       default_weight=2.0,
       needs_courses=False
   )
   def my_new_constraint(sessions: List[CourseSession]) -> int:
       # Implementation
       return violations
   ```

2. **Add to config schema** in `src/config/model.py`:
   ```python
   class HardConstraintsConfig(BaseModel):
       # ... existing constraints ...
       my_new_constraint: ConstraintConfig = Field(
           default=ConstraintConfig(enabled=True, weight=2.0)
       )
   ```

3. **Add to config YAML** in `configs/base.yaml`:
   ```yaml
   hard_constraints:
     # ... existing constraints ...
     my_new_constraint:
       enabled: true
       weight: 2.0
   ```

4. **Done!** The constraint is now:
   - Auto-registered in the registry
   - Picked up by evaluators
   - Included in console output (as hc9)
   - Available for enable/disable in config

### Removing a Constraint

1. Remove decorator and function from `hard.py`
2. Remove from config schema (or mark deprecated)
3. Remove from config YAML

**No other files need updating!**

### Checking Constraint Order

```python
from src.constraints.registry import get_all_hard_constraints

for i, name in enumerate(get_all_hard_constraints().keys(), 1):
    print(f"hc{i} = {name}")
```

Output:
```
hc1 = student_group_exclusivity
hc2 = instructor_exclusivity
...
hc8 = course_completeness
```

## Related Documentation

- **Registry Implementation**: `src/constraints/registry.py`
- **Hard Constraints**: `src/constraints/hard.py`
- **Soft Constraints**: `src/constraints/soft.py`
- **Constraint Mapping**: `docs/03-architecture/CONSTRAINT_MAPPING_REFERENCE.md`
- **Workflow Integration**: `src/workflows/standard_run.py` lines 276-291

## Impact on Codebase

**Files Modified**:
- `src/workflows/standard_run.py` - Removed 60 lines of duplication, now uses registry

**Files Already Using Registry** (no changes needed):
- `src/ga/evaluator/fitness.py`
- `src/ga/evaluator/detailed_fitness.py`
- `src/lns/conflict_detection.py`
- `scripts/benchmarking/bench_constraint_check.py`

**Documentation Created**:
- `docs/03-architecture/CONSTRAINT_MAPPING_REFERENCE.md`
- `docs/06-development/implementation-notes/SINGLE_SOURCE_OF_TRUTH_CONSTRAINTS.md` (this file)

## Lessons Learned

1. **Decorator-based registration is powerful**: Auto-registration when module loads
2. **Python dict order is reliable**: Python 3.7+ guarantees insertion order
3. **Co-locate metadata with implementation**: Reduces maintenance burden
4. **Single source of truth prevents drift**: No duplicated constraint lists

## Future Enhancements

1. **Auto-generate config schema** from registry (eliminate manual config model updates)
2. **Runtime validation** that all registered constraints exist in config
3. **Constraint dependency graph** (e.g., course_completeness depends on group enrollments)
4. **Performance profiling** per constraint (which constraints are slowest?)
