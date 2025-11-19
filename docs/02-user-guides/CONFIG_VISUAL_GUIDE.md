# Configuration System - Visual Guide

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Configuration Loading                         │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  User Command    │
│  python main.py  │
│  --env dev       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐       ┌─────────────────┐
│  config/loader   │──────▶│  common.yaml    │
│  .py             │       │  (All defaults) │
│  load_config()   │       └─────────────────┘
└────────┬─────────┘                 │
         │                           │ Load
         │                           ▼
         │                  ┌─────────────────┐
         │                  │  base_config    │
         │                  │  (dict)         │
         │                  └────────┬────────┘
         │                           │
         │  Load                     │ deep_merge()
         │  environment              │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  dev.yaml       │────────▶│  final_config   │
│  (Overrides)    │         │  (merged dict)  │
└─────────────────┘         └────────┬────────┘
                                     │
                                     │ Validate
                                     ▼
                            ┌─────────────────┐
                            │  config/models  │
                            │  Config object  │
                            │  (Pydantic)     │
                            └─────────────────┘
```

## File Hierarchy

```
configs/
│
├── common.yaml              ← Source of Truth (150 lines)
│   ├── time: {...}          ← All time settings
│   ├── io: {...}            ← All I/O paths
│   ├── ga: {...}            ← GA defaults
│   ├── repair: {...}        ← Repair defaults
│   ├── hard_constraints: {...}  ← Default weights
│   ├── soft_constraints: {...}  ← Default weights
│   └── enhancements: {...}  ← Feature flags
│
├── test.yaml (15 lines)     ← Minimal Overrides
│   ├── ga:
│   │   ├── ngen: 10        ✓ Override
│   │   └── pop_size: 4     ✓ Override
│   ├── parallel:
│   │   └── use_multiprocessing: false  ✓ Override
│   └── feasibility:
│       └── fail_on_infeasibility: false  ✓ Override
│
├── dev.yaml (25 lines)      ← Moderate Overrides
│   ├── ga:
│   │   ├── ngen: 100       ✓ Override
│   │   └── pop_size: 100   ✓ Override
│   ├── repair:
│   │   ├── memetic_mode: true  ✓ Override
│   │   └── elite_percentage: 0.2  ✓ Override
│   └── enhancements:
│       └── greedy_initialization_percent: 0.4  ✓ Override
│
└── prod.yaml (50 lines)     ← Comprehensive Overrides
    ├── ga:
    │   ├── ngen: 2000       ✓ Override
    │   ├── pop_size: 200    ✓ Override
    │   ├── cxpb: 0.85       ✓ Override
    │   └── mutpb: 0.25      ✓ Override
    ├── time:
    │   ├── theory_isolated_penalty: 3  ✓ Override (stricter)
    │   └── practical_fragmentation_penalty: 50  ✓ Override (stricter)
    ├── hard_constraints:
    │   ├── availability_violations.weight: 6.0  ✓ Override (higher)
    │   └── session_block_clustering_penalty.weight: 4.0  ✓ Override
    └── soft_constraints:
        ├── group_gaps_penalty.weight: 2.5  ✓ Override (higher)
        └── soft_weight_factor: 0.015  ✓ Override
```

## Merge Example

### Input Files

**common.yaml:**
```yaml
time:
  quantum_minutes: 60
  theory_isolated_penalty: 2
  practical_fragmentation_penalty: 20

ga:
  cxpb: 0.8
  mutpb: 0.3
```

**prod.yaml:**
```yaml
time:
  theory_isolated_penalty: 3         # Override
  practical_fragmentation_penalty: 50  # Override

ga:
  ngen: 2000    # New key
  pop_size: 200  # New key
```

### Merge Process

```python
def deep_merge(common, prod):
    result = common.copy()
    
    for key, value in prod.items():
        if key in result and isinstance(result[key], dict):
            # Recursive merge for nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override or add new key
            result[key] = value
    
    return result
```

### Result

```yaml
time:
  quantum_minutes: 60                   # From common
  theory_isolated_penalty: 3            # Overridden by prod
  practical_fragmentation_penalty: 50   # Overridden by prod

ga:
  cxpb: 0.8      # From common
  mutpb: 0.3     # From common
  ngen: 2000     # Added by prod
  pop_size: 200  # Added by prod
```

## Value Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Parameter: quantum_minutes                                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  common.yaml → 60                                           │
│      ↓                                                       │
│  test.yaml → (not defined) → inherit 60 from common ✓      │
│      ↓                                                       │
│  dev.yaml → (not defined) → inherit 60 from common ✓       │
│      ↓                                                       │
│  prod.yaml → (not defined) → inherit 60 from common ✓      │
│                                                              │
│  Result: All environments use 60                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Parameter: theory_isolated_penalty                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  common.yaml → 2 (default)                                  │
│      ↓                                                       │
│  test.yaml → (not defined) → inherit 2 from common ✓       │
│      ↓                                                       │
│  dev.yaml → (not defined) → inherit 2 from common ✓        │
│      ↓                                                       │
│  prod.yaml → 3 (override!) → USE 3 ✓                       │
│                                                              │
│  Result: test=2, dev=2, prod=3                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Parameter: ngen                                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  common.yaml → (not tunable, no default)                   │
│      ↓                                                       │
│  test.yaml → 10 ✓                                           │
│      ↓                                                       │
│  dev.yaml → 100 ✓                                           │
│      ↓                                                       │
│  prod.yaml → 2000 ✓                                         │
│                                                              │
│  Result: test=10, dev=100, prod=2000                       │
└─────────────────────────────────────────────────────────────┘
```

## Size Comparison

### Before (Duplication)
```
test.yaml:  ████████████████████████████████ 157 lines
dev.yaml:   ████████████████████████████████ 165 lines
prod.yaml:  ████████████████████████████████ 165 lines
            ─────────────────────────────────
Total:      ████████████████████████████████ 487 lines
            (~80% duplication)
```

### After (DRY)
```
common.yaml: ████████████████████████ 150 lines
test.yaml:   ██                      15 lines
dev.yaml:    ████                    25 lines
prod.yaml:   ████████                50 lines
             ─────────────────────────
Total:       ████████████████████████ 240 lines
             (zero duplication)
```

**Reduction**: 487 → 240 lines (50% smaller!)

## Decision Tree: Where to Put a New Parameter?

```
                    New Parameter
                         │
                         ▼
            ┌────────────────────────┐
            │ Same value for all     │
            │ environments?          │
            └────────┬───────────────┘
                     │
         ┌───────────┴──────────┐
         │                      │
        Yes                    No
         │                      │
         │                      ▼
         │          ┌───────────────────────┐
         │          │ Tuning parameter      │
         │          │ (ngen, pop_size)?     │
         │          └──────────┬────────────┘
         │                     │
         │         ┌───────────┴──────────┐
         │         │                      │
         │        Yes                    No
         │         │                      │
         ▼         ▼                      ▼
  ┌──────────┐  ┌──────────┐      ┌────────────┐
  │ common   │  │ test/dev │      │ prod.yaml  │
  │ .yaml    │  │ /prod    │      │ (override) │
  │          │  │ .yaml    │      │            │
  │ (default)│  │ (all 3)  │      │ (stricter) │
  └──────────┘  └──────────┘      └────────────┘

Examples:
├─ common.yaml: quantum_minutes, data_dir, earliest_preferred_time
├─ test/dev/prod: ngen, pop_size, max_iterations
└─ prod.yaml: theory_isolated_penalty (higher), constraint weights (higher)
```

## Best Practices

###  DO

- Define defaults in `common.yaml`
- Override in test/dev/prod only when necessary
- Keep environment configs minimal
- Use comments to explain overrides
- Group related parameters together

###  DON'T

- Duplicate values across multiple files
- Put tuning parameters in `common.yaml`
- Add hardcoded defaults in Python code
- Override values unnecessarily
- Mix configuration with code logic

---

**Visual Summary Complete!** 
Config system now follows industry best practices with clear separation of concerns and zero duplication.
