# Repair Operator Refactoring - Clean Architecture

## Summary

Successfully refactored repair operators to use decorator-based registry pattern, matching the clean architecture of `src/constraints/registry.py`.

## Changes Made

### 1.  Removed Dead Code
- **Deleted**: `crossover_uniform()` function from `crossover.py` (deprecated, never used)
- **Removed**: Exports of `crossover_uniform` from `__init__.py`

### 2.  Created Decorator-Based Registry
- **New File**: `src/ga/operators/repair_wrappers.py` (334 lines)
- **Architecture**: Follows `src/constraints/registry.py` pattern exactly
- **Features**:
  - `@repair_operator` decorator for auto-registration
  - `RepairOperatorMetadata` dataclass
  - Config-based enable/disable
  - Priority ordering
  - Backward compatibility helpers

### 3.  Updated Repair Functions
- **Modified**: All 8 repair functions in `repair.py`
- **Added**: `@repair_operator` decorators to each function
- **Metadata**: Name, description, priority, modifies_length flags

Decorated functions:
```python
@repair_operator(name="repair_instructor_availability", priority=1, ...)
@repair_operator(name="repair_group_overlaps", priority=2, ...)
@repair_operator(name="repair_room_conflicts", priority=3, ...)
@repair_operator(name="repair_instructor_conflicts", priority=4, ...)
@repair_operator(name="repair_instructor_qualifications", priority=5, ...)
@repair_operator(name="repair_room_type_mismatches", priority=6, ...)
@repair_operator(name="repair_session_clustering", priority=7, ...)
@repair_operator(name="repair_incomplete_or_extra_sessions", priority=8, modifies_length=True)
```

### 4.  Backward Compatibility Layer
- **Modified**: `src/ga/operators/repair_registry.py`
- **Purpose**: Delegates to `repair_wrappers.py` but maintains old dict-based API
- **Status**: Marked as DEPRECATED but fully functional
- **Migration Path**: 
  ```python
  # Old (still works)
  from src.ga.operators.repair_registry import get_enabled_repair_heuristics
  
  # New (preferred)
  from src.ga.operators.repair_wrappers import get_enabled_repair_operators
  ```

### 5.  Updated Exports
- **Modified**: `src/ga/operators/__init__.py`
- **Added**: New wrapper functions to `__all__`
- **Maintained**: All existing imports (backward compatible)

## Architecture Benefits

### Before (Old Registry Pattern)
```python
# Hardcoded dict in repair_registry.py (95 lines)
{
    "repair_group_overlaps": {
        "function": repair_group_overlaps,
        "priority": 2,
        "description": "Fix group overlaps",
        "modifies_length": False,
    },
    # ... 7 more entries ...
}
```

### After (Decorator Pattern)
```python
# Self-registering with @repair_operator decorator
@repair_operator(
    name="repair_group_overlaps",
    description="Fix group schedule overlaps",
    priority=2,
    modifies_length=False
)
def repair_group_overlaps(individual, context):
    # implementation
    return fixes
```

**Benefits:**
1. **Single Source of Truth**: Metadata lives with function definition
2. **Auto-Registration**: No manual dict updates needed
3. **Type Safety**: Dataclass validation via `RepairOperatorMetadata`
4. **Consistency**: Same pattern as constraints registry
5. **Extensibility**: Easy to add new metadata fields
6. **Introspection**: Metadata stored on function (`func._repair_metadata`)

## Verification Results

```bash
✓ All operator imports successful
✓ Backward compat: True
✓ New wrappers: True
Old style: 8 operators | New style: 8 operators
Match: True (same operators in both registries)
```

## Migration Guide (For Future Code)

### Old Style (Still Works)
```python
from src.ga.operators.repair_registry import get_enabled_repair_heuristics

repairs = get_enabled_repair_heuristics()
for name, info in repairs.items():
    func = info["function"]
    priority = info["priority"]
    fixes = func(individual, context)
```

### New Style (Preferred)
```python
from src.ga.operators.repair_wrappers import get_enabled_repair_operators

repairs = get_enabled_repair_operators()
for name, metadata in repairs.items():
    func = metadata.function
    priority = metadata.priority
    fixes = func(individual, context)
```

### Adding New Repair Operator
```python
from src.ga.operators.repair_wrappers import repair_operator

@repair_operator(
    name="repair_new_constraint",
    description="Fix new constraint violations",
    priority=9,
    modifies_length=False,
    enabled_by_default=True
)
def repair_new_constraint(individual, context):
    fixes = 0
    # implementation
    return fixes
```

## Files Changed

| File | Lines Changed | Type |
|------|---------------|------|
| `crossover.py` | -35 lines | Deletion (dead code) |
| `__init__.py` | +13 lines | Update (exports) |
| `repair_wrappers.py` | +334 lines | New file (registry) |
| `repair.py` | +64 lines | Update (decorators) |
| `repair_registry.py` | -115 lines | Update (compat layer) |

**Total**: +261 lines (net after deletions)

## No Breaking Changes

- All existing code continues to work unchanged
- `repair_registry.py` delegates to new system transparently
- Old dict-based API preserved for backward compatibility
- Tests should pass without modification

## Next Steps (Phase 2)

With this clean architecture in place, Phase 2 (RL Environment) can now:

1. Create `src/rl/actions/repair_actions.py` with action wrappers
2. Use decorator pattern for action registration
3. Query repair operators via `get_all_repair_operators()`
4. Wrap selected operators as RL actions

Example Phase 2 code:
```python
from src.ga.operators.repair_wrappers import get_repair_operator_function

# RL Agent selects action by name
selected_action = "repair_group_overlaps"
repair_func = get_repair_operator_function(selected_action)

# Apply repair
fixes = repair_func(individual, context)
```

## Consistency with Constraints

Both constraint and repair systems now use identical architecture:

| Feature | Constraints | Repairs |
|---------|-------------|---------|
| Registry file | `constraints/registry.py` | `ga/operators/repair_wrappers.py` |
| Decorator | `@hard_constraint` / `@soft_constraint` | `@repair_operator` |
| Metadata class | `ConstraintMetadata` | `RepairOperatorMetadata` |
| Get all | `get_all_hard_constraints()` | `get_all_repair_operators()` |
| Get enabled | Config-filtered, priority-sorted | Config-filtered, priority-sorted |
| Auto-registration | ✓ | ✓ |
| Introspection | `func._constraint_metadata` | `func._repair_metadata` |

---

**Date**: 2025-11-15
**Commit**: `refactor(operators): remove dead code, add decorator-based repair registry`
