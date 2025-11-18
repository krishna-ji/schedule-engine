# Configuration Directory

**Last Updated**: November 18, 2025

---

## Structure

This directory contains hierarchical configuration files organized by domain:

```
configs/
├── common/          # Shared settings across all components
│   ├── base.yaml    # Common defaults
│   ├── prod.yaml    # Production overrides
│   ├── test.yaml    # Test/smoke test overrides
│   └── med.yaml     # Medium run overrides
├── ga/              # Genetic algorithm settings
│   ├── base.yaml    # GA defaults
│   ├── prod.yaml    # Production GA settings (1000 gens, 100 pop)
│   ├── test.yaml    # Test GA settings (30 gens, 10 pop)
│   └── med.yaml     # Medium GA settings (200 gens, 50 pop)
├── rl/              # Reinforcement learning settings
│   ├── base.yaml    # RL defaults
│   ├── prod.yaml    # Production RL settings (300K steps)
│   ├── test.yaml    # Test RL settings (10K steps)
│   └── med.yaml     # Medium RL settings (100K steps)
└── [Legacy files]   # Kept for backward compatibility
    ├── base.yaml    # (Legacy) All settings mixed
    ├── prod.yaml    # (Legacy) Environment override
    └── test.yaml    # (Legacy) Environment override
```

---

## Domain Breakdown

### Common Settings
**Location**: `configs/common/`

Settings shared across all components:
- **Time**: Quantum system, scheduling windows, break times
- **I/O**: Data and output directories  
- **Calendar**: Visualization settings
- **Parallel**: Multiprocessing configuration
- **Feasibility**: Constraint validation checks

### GA Settings
**Location**: `configs/ga/`

Genetic algorithm-specific settings:
- **GA Parameters**: Population size, generations, crossover/mutation rates
- **Constraints**: Hard (must-satisfy) and soft (prefer-satisfy) constraints
- **Repair**: IGLS, LNS, stagnation repair, selective repair
- **Enhancements**: Hypermutation, population restart, violation heatmap
- **Heuristics**: 19 operators across 5 categories (construction, perturbation, improvement, diversity, meta)

### RL Settings
**Location**: `configs/rl/`

Reinforcement learning-specific settings:
- **Integration**: Master killswitch, mode selection
- **Environment**: Episode length, observation history
- **Reward**: Fitness, diversity, time weights
- **Agent**: PPO/DQN hyperparameters, device (CPU/CUDA)
- **Training**: Curriculum learning, checkpointing, TensorBoard
- **Inference**: Timeout, fallback strategies
- **Hybrid**: RL-primary/fallback/assisted modes
- **Evaluation**: Baseline strategies, metrics

---

## Usage

### Load Hierarchical Config
```bash
# Load test environment (hierarchical)
ENVIRONMENT=test python main.py

# Load production environment (hierarchical)
ENVIRONMENT=prod python main.py

# Or use convenience commands
uv run test   # Test environment (30 gens, 10 pop)
uv run prod   # Production environment (1000 gens, 100 pop)
```

### Load Legacy Config (Backward Compatible)
```bash
# Explicit path (legacy mode)
python main.py --config configs/base.yaml

# Environment variable (legacy mode)
SCHEDULE_CONFIG=configs/prod.yaml python main.py
```

### Verify Configuration
```bash
# Show loaded configuration
python scripts/show_config.py
```

---

## Environment Profiles

### Production (`prod`)
**Runtime**: 24-48 hours  
**Hardware**: 16+ cores, 32+ GB RAM, GPU recommended

- **GA**: 1000 generations, 100 population
- **RL**: 300K timesteps, CUDA enabled, inference mode
- **Repair**: LNS enabled, exhaustive search at intervals
- **Quality**: Maximum (slow but best results)

### Test (`test`)
**Runtime**: 5-10 minutes  
**Hardware**: Any (single core)

- **GA**: 30 generations, 10 population
- **RL**: 10K timesteps, quick validation
- **Repair**: Minimal triggers, fast timeouts
- **Quality**: Minimal (smoke test only)

### Medium (`med`)
**Runtime**: 2-4 hours  
**Hardware**: 4+ cores, 8+ GB RAM

- **GA**: 200 generations, 50 population
- **RL**: 100K timesteps, balanced training
- **Repair**: Balanced triggers and timeouts
- **Quality**: Good (development use)

---

## Configuration Loading

The config loader automatically detects which structure to use:

1. **Hierarchical** (preferred): If `configs/{common,ga,rl}/` exist
   - Loads and merges: common → ga → rl (base then env overrides)
2. **Legacy** (fallback): If hierarchical doesn't exist
   - Loads and merges: base.yaml → {env}.yaml

### Priority Order
1. `--config` flag (explicit path)
2. `SCHEDULE_CONFIG` environment variable
3. Hierarchical structure (if exists)
4. Legacy structure (backward compatibility)
5. Built-in defaults

---

## Adding New Settings

1. **Determine domain**: Common, GA, or RL?
2. **Edit base file**: `configs/{domain}/base.yaml`
3. **Add default value**: With comments explaining purpose
4. **Override per environment**: Only if value differs
   - Production: `configs/{domain}/prod.yaml`
   - Test: `configs/{domain}/test.yaml`
   - Medium: `configs/{domain}/med.yaml`
5. **Update Pydantic model**: `src/config/models.py`
6. **Test loading**: `python scripts/show_config.py`

### Example: Add New GA Parameter
```yaml
# In configs/ga/base.yaml
ga:
  # ... existing ...
  tournament_size: 3  # Number of individuals in tournament selection

# Override for production (if needed)
# In configs/ga/prod.yaml
ga:
  tournament_size: 5  # Larger tournament for more selective pressure
```

---

## Benefits

### Maintainability
- **Separation of concerns**: GA, RL, common clearly separated
- **Focused changes**: Edit only relevant domain
- **Easier navigation**: Find settings by domain

### Clarity
- **Domain expertise**: GA experts edit GA configs, RL experts edit RL
- **Less confusion**: No mixing of unrelated settings
- **Better docs**: Each domain documented independently

### Flexibility
- **Independent evolution**: Domains evolve independently
- **Environment-specific**: Override only differences
- **Backward compatible**: Legacy structure still works

---

## Migration Notes

### Backward Compatibility
- Legacy `configs/base.yaml` still works
- Loader automatically detects structure
- No code changes required

### New Projects
- Use hierarchical structure from start
- Delete legacy files once migrated
- Follow domain separation guidelines

---

## References

- **Implementation**: `src/config/loader.py`
- **Models**: `src/config/models.py`
- **Documentation**: `docs/06-development/config-reorganization-guide.md`
- **Instructions**: `.github/instructions/config.instructions.md`

---

**Questions?** See `docs/06-development/config-reorganization-guide.md` for detailed guide.
