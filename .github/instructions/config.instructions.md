````instructions
---
applyTo: "**/*.py"
---

# Configuration System Instructions

## ⚠️ DEPRECATED - Pure CP-SAT Implementation

**The YAML-based configuration system has been removed.**

The engine now runs with hardcoded settings in `main.py`:
- Data directory: `"data"`
- Output directory: `"output"`  
- Time limit: `0` (unlimited)
- Parallel workers: `min(4, cpu_count())` (memory-safe)

## What Was Removed
- All YAML files in `configs/` (test.yaml, dev.yaml, prod.yaml)
- `config/models.py` (Pydantic validation models)
- `config/loader.py` (YAML loading logic)
- All GA configuration (population, generations, operators)
- All repair configuration (heuristics, selective mode)
- All feasibility configuration (checks, tolerance)
- Soft constraint weighting

## Current Settings
```python
# In main.py
DATA_DIR = "data"
OUTPUT_DIR = Path("output")
TIME_LIMIT = 0  # unlimited (set to seconds for limit)
WORKERS = min(4, cpu_count())  # memory-safe limit
```

To change settings, edit `main.py` directly. No configuration files needed.

## Why This Change?
Pure CP-SAT solver doesn't need the complexity of:
- Multi-objective optimization (no soft constraints)
- GA tuning parameters (no genetic algorithm)
- Complex configuration merging
- Environment-specific settings

Simpler is better for constraint programming.

````
