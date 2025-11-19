# UV Shortcuts Reference

Complete list of all UV shortcuts for running schedule-engine commands.

##  Runtime Modes (Schedule Generation)

Run the genetic algorithm with different configurations:

```bash
# Environment variations
uv run test              # Quick smoke test (30 generations, ~5-10 min)
uv run prod              # Production run (2000 generations, ~24-48 hours)

# Progressive runtime modes (Baseline → Full GA)
uv run baseline          # Mode 1: Pure NSGA-II (no repairs, no heuristics)
uv run repairs           # Mode 2: NSGA-II + IGLS repairs
uv run heuristics        # Mode 3: NSGA-II + repairs + 19 heuristics
uv run full              # Mode 4: Full GA (best non-RL configuration)

# RL-enhanced modes
uv run rl                # Mode 5: RL-guided heuristic selection
uv run roundrobin        # Mode 6: Fixed round-robin rotation

# Advanced RL modes (Phase 3)
uv run specialists       # Mode 7: Specialist agents (repair, optimizer, explorer, intensifier)
uv run archive           # Mode 8: Archive-based diversity preservation
uv run hierarchical      # Mode 9: Hierarchical RL (high-level + low-level policies)
uv run multiagent        # Mode 10: Rank-based multi-agent cooperation
```

##  Training & Model Management

RL agent training and deployment:

```bash
# Training
uv run train                # Start RL training (default: curriculum learning)
uv run train-curriculum     # Explicit curriculum training mode

# Model management
uv run generate-validation  # Create validation dataset from solved schedules
uv run select-checkpoint    # Analyze training checkpoints, select best model
uv run promote-model        # Deploy validated model to production
```

##  Diagnostics & Validation

System health checks and data validation:

```bash
# System diagnostics
uv run diagnose-gpu         # Check GPU/CUDA setup (comprehensive report)
uv run test-dashboard       # Test TensorBoard integration

# Data & config validation
uv run check-data           # Validate input JSON data quality
uv run verify-config        # Verify configuration standardization
uv run verify-enhancements  # Verify Phase 3 enhancements implemented
```

##  Benchmarking & Performance

Performance measurement and optimization:

```bash
# Performance benchmarks
uv run benchmark-gpu        # GPU vs CPU training performance comparison
uv run benchmark-lns        # Large Neighborhood Search + CP-SAT benchmark
uv run benchmark-constraints # Constraint checking speed analysis
```

## ⚙️ Configuration Display

View current system configuration:

```bash
# Configuration viewers
uv run show-config          # Display all constraints (hard + soft)
uv run show-repair          # Display repair system configuration
uv run show-soft            # Display soft constraints only
uv run show-time            # Display time system configuration
```

## 🛠️ Development Utilities

Tools for development and debugging:

```bash
# Development tools
uv run tensorboard          # Start TensorBoard server (port 6006)
uv run git-squash           # Interactive git commit squashing
uv run refactor-csv         # CSV export refactoring utility
```

##  Common Workflows

### Quick Smoke Test
```bash
# Test with baseline mode (fastest, no enhancements)
uv run baseline --env test
```

### Full Production Run
```bash
# Best configuration for production schedules
uv run full --env prod
```

### GPU Setup & Training
```bash
# 1. Verify GPU is working
uv run diagnose-gpu

# 2. Benchmark GPU vs CPU
uv run benchmark-gpu --timesteps 20000

# 3. Generate validation data
uv run generate-validation

# 4. Start training
uv run train

# 5. Monitor training
uv run tensorboard

# 6. Select best checkpoint
uv run select-checkpoint --log-dir logs/tensorboard/train

# 7. Deploy to production
uv run promote-model --checkpoint best_model.zip
```

### Pre-Release Validation
```bash
# Complete validation suite
uv run check-data             # Data quality
uv run verify-config          # Config standardization
uv run verify-enhancements    # Phase 3 implementations
```

### Performance Analysis
```bash
# Run all benchmarks
uv run benchmark-gpu --timesteps 20000
uv run benchmark-constraints
uv run benchmark-lns
```

### Configuration Review
```bash
# Review all settings before production run
uv run show-config
uv run show-repair
uv run show-soft
uv run show-time
```

##  Runtime Mode Comparison

Quick reference for choosing the right mode:

| Mode | Name | Speed | Quality | Best For |
|------|------|-------|---------|----------|
| 1 | baseline |  |  | Quick tests, baseline comparison |
| 2 | repairs |  |  | Feasibility-focused schedules |
| 3 | heuristics |  |  | Balanced quality/speed |
| 4 | full |  |  | Best non-RL quality |
| 5 | rl |  |  | Adaptive heuristic selection |
| 6 | roundrobin |  |  | Deterministic diversity |
| 7 | specialists |  |  | Multi-agent cooperation |
| 8 | archive |  |  | Diversity preservation |
| 9 | hierarchical |  |  | Two-level optimization |
| 10 | multiagent |  |  | Maximum quality (slowest) |

##  Custom Arguments

Most commands accept additional arguments:

```bash
# Runtime modes with custom config
uv run full --env test --config custom.yaml

# Training with custom timesteps
uv run train --timesteps 500000

# Benchmarking with custom settings
uv run benchmark-gpu --timesteps 50000 --output results.json

# Checkpoint selection with custom metric
uv run select-checkpoint --log-dir logs/train --metric episode_reward_mean

# TensorBoard with custom port
# (Edit scripts/utilities/start_tensorboard.py to change port)
```

##  Related Documentation

- **Main README**: `README.md` - Project overview and installation
- **Runtime Modes Guide**: `docs/02-user-guides/runtime-modes.md` - Detailed mode explanations
- **Scripts README**: `scripts/README.md` - Script organization and usage
- **Training Guide**: `docs/02-user-guides/rl-training.md` - RL training workflow
- **Configuration Guide**: `docs/02-user-guides/configuration.md` - Config system

##  Troubleshooting

### Command not found
```bash
# Ensure UV is installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies
uv sync
```

### Import errors
```bash
# Re-sync dependencies after updates
uv sync

# Check Python version (requires 3.11+)
python --version
```

### GPU not detected
```bash
# Run diagnostics
uv run diagnose-gpu

# Follow recommendations in diagnostic output
```

### Training fails
```bash
# Check data quality first
uv run check-data

# Verify configuration
uv run verify-config

# Check GPU if using CUDA
uv run diagnose-gpu
```

---

**Last Updated**: November 19, 2025  
**Total Shortcuts**: 35+ commands across 6 categories
