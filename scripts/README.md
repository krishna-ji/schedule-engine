# Scripts Directory

Utility scripts for development, testing, training, and validation.

---

##  Interactive Launcher

** The easiest way to run any command!**

```bash
uv run launcher    # Start interactive menu
uv run run         # Shorter alias
```

**Features:**
-  **Browse 50+ commands** in 10 organized categories
-  **Descriptions + runtime estimates** for each command
-  **Select by number** or type command name directly
-  **Live command output** with colored formatting
-  **Run multiple commands** in one session

**Categories:** Thesis Experiments | Quick Tests | Production Runs | Runtime Modes | RL Training | Analysis | Diagnostics | Benchmarking | Configuration | Development

**Full Guide:** [docs/QUICKREF_LAUNCHER.md](../docs/QUICKREF_LAUNCHER.md)

---

##  Quick Start

All scripts are also accessible via **UV shortcuts** for direct execution:

```bash
# Diagnostics
uv run diagnose-gpu           # Check GPU/CUDA setup
uv run check-data             # Validate input data quality

# Benchmarking  
uv run benchmark-gpu          # GPU vs CPU performance
uv run benchmark-lns          # LNS/CP-SAT comparison
uv run benchmark-constraints  # Constraint checking speed

# Configuration
uv run show-config            # Display all settings
uv run show-repair            # Repair system config
uv run show-soft              # Soft constraints only
uv run show-time              # Time system config

# Training & Models
uv run generate-validation    # Create validation dataset
uv run select-checkpoint      # Analyze checkpoints
uv run promote-model          # Deploy model to production

# Utilities
uv run tensorboard            # Start TensorBoard server
uv run git-squash             # Interactive commit squashing
```

##  Directory Structure

```
scripts/
├── cli.py             # UV entry point registry
├── benchmarking/      # Performance benchmarks
├── training/          # RL training workflows
├── validation/        # Data & config validation
├── diagnostics/       # System diagnostics
└── utilities/         # General utilities
```

##  Benchmarking

Performance measurement and comparison scripts.

| Script | Description |
|--------|-------------|
| `benchmark_gpu_training.py` | Benchmark RL training with/without GPU |
| `benchmark_lns_cp.py` | Compare LNS and CP-SAT performance |
| `bench_constraint_check.py` | Benchmark constraint evaluation speed |

**Usage:**
```bash
# Using UV shortcuts (recommended)
uv run benchmark-gpu
uv run benchmark-lns
uv run benchmark-constraints

# Or directly
python scripts/benchmarking/benchmark_gpu_training.py
```

##  Training

RL agent training infrastructure and model management.

| Script | Description |
|--------|-------------|
| `generate_validation_set.py` | Create validation dataset from solved schedules |
| `select_best_checkpoint.py` | Analyze checkpoints and select best model |
| `promote_model_to_prod.py` | Promote trained model to production |

**Usage:**
```bash
# Using UV shortcuts (recommended)
uv run generate-validation
uv run select-checkpoint --log-dir logs/tensorboard/train --metric episode_reward_mean
uv run promote-model --checkpoint models/rl_agents/checkpoints/best_model.zip

# Or directly
python scripts/training/generate_validation_set.py
python scripts/training/select_best_checkpoint.py --log-dir logs/tensorboard/train
python scripts/training/promote_model_to_prod.py --checkpoint best_model.zip
```

##  Validation

Data quality checks and configuration validation.

| Script | Description |
|--------|-------------|
| `check_data_quality.py` | Validate input JSON data for inconsistencies |
| `verify_config_standardization.py` | Ensure config files follow standards |
| `verify_enhancements.py` | Verify Phase 3 enhancements are implemented |

**Usage:**
```bash
# Using UV shortcuts (recommended)
uv run check-data
uv run verify-config
uv run verify-enhancements

# Or directly
python scripts/validation/check_data_quality.py
python scripts/validation/verify_config_standardization.py
python scripts/validation/verify_enhancements.py
```

##  Diagnostics

System health checks and integration tests.

| Script | Description |
|--------|-------------|
| `diagnose_gpu.py` | Check GPU availability and CUDA support |
| `test_dashboard_integration.py` | Test TensorBoard integration |

**Usage:**
```bash
# Using UV shortcuts (recommended)
uv run diagnose-gpu
uv run test-dashboard

# Or directly
python scripts/diagnostics/diagnose_gpu.py
python scripts/diagnostics/test_dashboard_integration.py
```

## Utilities

General-purpose helper scripts.

| Script | Description |
|--------|-------------|
| `show_config.py` | Display current configuration |
| `show_repair_config.py` | Display repair heuristics config |
| `show_soft_config.py` | Display soft constraint config |
| `show_time_config.py` | Display time quantum config |
| `refactor_csv_exports.py` | Convert old CSV exports to new format |
| `start_tensorboard.py` | Start TensorBoard server (cross-platform) |
| `git_squash.py` | Interactive git commit squashing (cross-platform) |

**Usage:**
```bash
# Using UV shortcuts (recommended)
uv run show-config           # All configuration
uv run show-repair           # Repair system config
uv run show-soft             # Soft constraints only
uv run show-time             # Time system config
uv run tensorboard           # Start TensorBoard
uv run git-squash            # Interactive commit squashing
uv run refactor-csv          # CSV export refactoring

# Or directly
python scripts/utilities/show_config.py
python scripts/utilities/start_tensorboard.py
python scripts/utilities/git_squash.py
```

##  Common Workflows

### Complete Training Workflow
```bash
# 1. Generate validation dataset
uv run generate-validation

# 2. Train agent (production config)
uv run train

# 3. Select best checkpoint
uv run select-checkpoint --log-dir logs/tensorboard/train

# 4. Promote to production
uv run promote-model --checkpoint best_model.zip
```

### Pre-Release Validation
```bash
# Check data quality
uv run check-data

# Verify configurations
uv run verify-config

# Verify Phase 3 enhancements
uv run verify-enhancements
```

### Performance Analysis
```bash
# GPU vs CPU comparison
uv run benchmark-gpu --timesteps 20000

# Constraint checking speed
uv run benchmark-constraints

# LNS/CP-SAT comparison
uv run benchmark-lns
```

### System Diagnostics
```bash
# GPU setup check
uv run diagnose-gpu

# TensorBoard integration
uv run test-dashboard
```

##  Adding New Scripts

When adding new scripts:

1. **Choose appropriate category** (benchmarking, training, validation, diagnostics, utilities)
2. **Add shebang**: `#!/usr/bin/env python3`
3. **Add docstring** explaining purpose and usage
4. **Create main() entry point** function
5. **Add to `scripts/cli.py`**:
   ```python
   def my_new_script():
       """Brief description."""
       from scripts.category.my_new_script import main
       main()
   ```
6. **Add UV shortcut to `pyproject.toml`**:
   ```toml
   my-new-script = "scripts.cli:my_new_script"
   ```
7. **Update this README** with script description and usage examples
8. **Update `__all__` in `scripts/cli.py`**

##  Related Documentation

- [Development Workflow](../docs/06-development/)
- [RL Training Guide](../docs/02-user-guides/rl-training.md)
- [Runtime Modes](../docs/02-user-guides/runtime-modes.md)
