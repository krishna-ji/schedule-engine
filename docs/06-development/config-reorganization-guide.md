# Configuration Reorganization Guide

**Date**: November 18, 2025  
**Status**: Complete  
**Related**: Phase 1 & 2 Production Readiness

---

## Overview

The configuration system has been reorganized from a flat structure to a hierarchical, domain-specific structure for better maintainability and clarity.

## Migration Summary

### Before (Legacy Structure)
```
configs/
├── base.yaml (438 lines - all settings mixed)
├── prod.yaml
└── test.yaml
```

### After (Hierarchical Structure)
```
configs/
├── common/          # Shared settings
│   ├── base.yaml
│   ├── prod.yaml
│   ├── test.yaml
│   └── med.yaml
├── ga/              # GA-specific settings
│   ├── base.yaml
│   ├── prod.yaml
│   ├── test.yaml
│   └── med.yaml
├── rl/              # RL-specific settings
│   ├── base.yaml
│   ├── prod.yaml
│   ├── test.yaml
│   └── med.yaml
└── [legacy files kept for backward compatibility]
```

---

## Domain Breakdown

### Common Settings (`configs/common/`)
Settings shared across all components:
- **Time Configuration**: Quantum system, scheduling windows, break times
- **I/O Paths**: Data and output directories
- **Calendar Display**: Visualization settings
- **Parallel Processing**: Multiprocessing configuration
- **Feasibility Checks**: Constraint validation settings

### GA Settings (`configs/ga/`)
Genetic algorithm-specific settings:
- **GA Parameters**: Population size, generations, crossover/mutation rates
- **Hard Constraints**: Exclusivity, qualifications, availability, completeness
- **Soft Constraints**: Schedule compactness, lunch breaks, continuity
- **Repair System**: IGLS, LNS, stagnation repair, selective repair
- **Enhancements**: Hypermutation, population restart, violation heatmap
- **Heuristic Toolbox**: Construction, perturbation, improvement, diversity, meta-heuristics

### RL Settings (`configs/rl/`)
Reinforcement learning-specific settings:
- **RL Integration**: Enable/disable, mode selection
- **Environment**: Episode length, observation history, rendering
- **Reward Function**: Fitness, diversity, time weights
- **Agent Configuration**: PPO/DQN hyperparameters, device settings
- **Training**: Curriculum learning, checkpointing, TensorBoard logging
- **Inference**: Batch prediction, timeout, fallback strategies
- **Hybrid Controller**: Mode, fallback strategy, action masking
- **Evaluation**: Baseline strategies, metrics collection

---

## Configuration Loading

The config loader (`src/config/loader.py`) supports both structures with automatic detection:

### Hierarchical Loading (Preferred)
When `configs/{common,ga,rl}/` subdirectories exist:

```python
# Merges in order:
1. configs/common/base.yaml
2. configs/common/{environment}.yaml
3. configs/ga/base.yaml
4. configs/ga/{environment}.yaml
5. configs/rl/base.yaml
6. configs/rl/{environment}.yaml
```

### Legacy Loading (Backward Compatible)
When hierarchical structure doesn't exist:

```python
# Falls back to:
1. configs/base.yaml
2. configs/{environment}.yaml
```

### Priority Order
1. Explicit `--config` path (command-line argument)
2. `SCHEDULE_CONFIG` environment variable
3. Hierarchical structure (if exists)
4. Legacy structure (backward compatibility)
5. Built-in defaults

---

## Environment Variants

### Production (`prod`)
- **GA**: 1000 generations, 100 population, LNS enabled
- **RL**: 300K timesteps, inference mode, RL integration ready
- **Common**: Full multiprocessing enabled
- **Use case**: Best quality runs (24-48h runtime)

### Test (`test`)
- **GA**: 30 generations, 10 population, minimal repair
- **RL**: 10K timesteps, quick validation
- **Common**: Multiprocessing disabled (easier debugging)
- **Use case**: Smoke tests (5-10 min runtime)

### Medium (`med`)
- **GA**: 200 generations, 50 population, balanced repair
- **RL**: 100K timesteps, balanced training
- **Common**: Multiprocessing enabled
- **Use case**: Development runs (2-4h runtime)

---

## Usage Examples

### Using Hierarchical Configs
```bash
# Load test environment (hierarchical)
ENVIRONMENT=test python main.py

# Load production environment (hierarchical)
ENVIRONMENT=prod python main.py

# Or use convenience commands
uv run test   # Loads test environment
uv run prod   # Loads production environment
```

### Using Legacy Configs (Backward Compatible)
```bash
# Explicit path (legacy mode)
python main.py --config configs/base.yaml

# Environment variable (legacy mode)
SCHEDULE_CONFIG=configs/prod.yaml python main.py
```

### Verifying Configuration
```bash
# Show loaded configuration
python scripts/show_config.py

# Expected output includes:
# - Config source (hierarchical vs legacy)
# - Environment name
# - Key settings from each domain
```

---

## Migration Path

### For Existing Code
No code changes required! The loader automatically:
1. Detects hierarchical structure if present
2. Falls back to legacy structure if not
3. Maintains backward compatibility

### For New Features
Add domain-specific settings to appropriate subdirectory:
- Common settings → `configs/common/base.yaml`
- GA settings → `configs/ga/base.yaml`
- RL settings → `configs/rl/base.yaml`

### Environment Overrides
Override settings per environment:
- Production overrides → `configs/{domain}/prod.yaml`
- Test overrides → `configs/{domain}/test.yaml`
- Medium overrides → `configs/{domain}/med.yaml`

---

## Benefits

### Maintainability
- **Separation of Concerns**: GA, RL, and common settings clearly separated
- **Focused Changes**: Modify only relevant domain files
- **Easier Navigation**: Find settings by domain, not by searching large files

### Clarity
- **Domain Expertise**: GA experts edit GA configs, RL experts edit RL configs
- **Less Confusion**: No mixing of unrelated settings
- **Better Documentation**: Each domain documented independently

### Flexibility
- **Independent Evolution**: GA and RL configs can evolve independently
- **Environment-Specific**: Override only what differs per environment
- **Backward Compatible**: Legacy structure still works

---

## Testing

Tested configurations:
- ✅ Hierarchical loading with all environments (test/prod/med)
- ✅ Legacy loading with backward compatibility
- ✅ Explicit path loading
- ✅ Environment variable loading
- ✅ Deep merge correctness across domains

---

## Future Enhancements

Potential improvements:
- Schema validation for each domain
- Config diff tool to compare environments
- Config templates for common scenarios
- Auto-generation from Pydantic models

---

## Troubleshooting

### Config Not Loading
**Problem**: "No config files found" warning  
**Solution**: Ensure either hierarchical structure exists OR legacy base.yaml exists

### Unexpected Settings
**Problem**: Settings not taking effect  
**Solution**: Check merge order - later files override earlier ones

### Legacy vs Hierarchical
**Problem**: Unsure which structure is being used  
**Solution**: Check console output - loader prints which structure it used

---

## References

- **Implementation**: `src/config/loader.py`
- **Models**: `src/config/models.py`
- **Instructions**: `.github/instructions/config.instructions.md`
- **Show Config**: `scripts/show_config.py`

---

**Last Updated**: November 18, 2025  
**Author**: Krishna / Copilot Agent  
**Status**: Configuration reorganization complete and tested
