# Bug Fixes Changelog

This file tracks all bug fixes in chronological order.

## [2025-11-30] Baseline Experiment - Pure NSGA-II Fix

**Severity**: High  
**Impact**: Baseline experiment behavior (Mode A)

### Issue

Baseline experiment had constraint-guided mutation enabled, making it not truly "pure NSGA-II".

### Root Cause

`use_constraint_guided_mutation` field was:

1. Missing from BaseConfig
2. Not mapped in dataclass → Pydantic conversion
3. Not explicitly disabled in baseline configs

### Fix

- Added `use_constraint_guided_mutation: bool` to BaseConfig
- Mapped field in `_build_pydantic_dict()`
- Set to `False` in both baseline test and prod configs

### Details

See: `docs/06-development/bugfixes/baseline-pure-nsga-fix.md`

---

## [2025-11-30] Room Type Mismatch Fix

**Severity**: Critical  
**Impact**: All experiments (mutation, repair, room assignment)

### Issues Fixed

1. **Data**: 7 lecture rooms incorrectly labeled as "Practical" in `Rooms.json`
2. **Code**: Room selection logic in GA operators used wrong fields and literal matching

### Files Modified
- `data/Rooms.json` - Fixed room type labels (B202, B207, B304, B308, B310, F207, F207x)
- `src/ga/operators/mutation.py` - Fixed `find_suitable_rooms_for_course()` to use correct fields and flexible matching
- `src/ga/operators/repair.py` - Fixed `_find_compatible_room()` to handle room type compatibility
- `src/ga/operators/repair_selective.py` - Fixed `repair_room_type_mismatch_selective()` with proper matching

### Details

See: `docs/06-development/bugfixes/room-type-mismatch-fix.md`

---

## [2025-12-01] Typing/Lint Regression Cleanup

**Severity**: Medium  
**Impact**: Tooling reliability (mypy, Ruff, CLI helpers)

### Issue

Pre-commit tooling failed because of undefined variables, unused assignments, and stale notebooks being linted. RL hierarchical controller also referenced an undefined `PPO` symbol, and the heuristic configurator always reported success even when no change occurred.

### Fix

- Normalized config registry naming in `configs/dataclass_loader.py`
- Fixed Experiment E metadata script to reference instantiated configs
- Tightened `scripts/analysis/compare_experiments.py` plotting outputs and ensured directories exist before saving
- Removed unused assignments across benchmarking/diagnostics utilities and simplified conditional logic in `scripts/archive/replace_quanta_assignments.py`
- Improved `scripts/utilities/configure_heuristic.py` status reporting and `git_squash.py` error handling
- Added safe PPO typing in `src/rl/hierarchical/hierarchical_controller.py`
- Updated Ruff configuration to ignore data notebooks that are not executable code

### Notes

These fixes unblock Ruff + mypy pre-commit hooks and prevent false failures during quality gates.

---
