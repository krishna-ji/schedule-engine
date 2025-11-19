# UV Commands Reference

Complete reference of all UV commands available in Schedule Engine.

## Quick Reference

### Main Entry Points

| Command | Description | Time | Use Case |
|---------|-------------|------|----------|
| `uv run test` | Smoke test (30 gens) | 5-10 min | Quick validation |
| `uv run prod` | Production run (2000 gens) | 1-2.5 hrs | Best quality |
| `uv run launcher` | Interactive menu | - | GUI selection |

### Thesis Experiments (Progressive)

| Command | Description | Mode | Time |
|---------|-------------|------|------|
| `uv run exp1` | Pure NSGA-II baseline | Mode 1 | 1-2 hrs |
| `uv run exp2` | + IGLS repairs | Mode 2 | 1-2 hrs |
| `uv run exp3` | + 19 heuristics (no LS) | Mode 3 | 1-2 hrs |
| `uv run exp4` | + Local search | Mode 4 | 1.5-2.5 hrs |
| `uv run exp5` | + RL-guided | Mode 5 | 1.5-2.5 hrs |

### Runtime Modes (All 10)

| Command | Mode | Description |
|---------|------|-------------|
| `uv run baseline` | 1 | Pure NSGA-II |
| `uv run repairs` | 2 | NSGA-II + IGLS repairs |
| `uv run heuristics` | 3 | + 19 heuristics |
| `uv run full` | 4 | Full GA (best non-RL) |
| `uv run rl` | 5 | RL-guided selection |
| `uv run roundrobin` | 6 | Fixed round-robin |
| `uv run specialists` | 7 | RL specialists |
| `uv run archive` | 8 | Archive diversity |
| `uv run hierarchical` | 9 | Hierarchical RL |
| `uv run multiagent` | 10 | Multi-agent RL |

## Detailed Command Reference

### Main Engine Commands

#### `uv run schedule-engine`
Main entry point with full control.

**Usage:**
```powershell
uv run schedule-engine [OPTIONS]
```

**Options:**
- `--env {test|prod}` - Environment selection
- `--mode {baseline|repairs|...}` - Runtime mode
- `--config PATH` - Custom config file
- `--experiment NAME` - Experiment tag
- `--list-modes` - Show all modes
- `--compare` - Compare mode features

**Examples:**
```powershell
# Basic usage
uv run schedule-engine --env test

# Specific mode
uv run schedule-engine --mode baseline --env prod

# Custom config
uv run schedule-engine --config configs/my-custom.yaml

# Tagged experiment
uv run schedule-engine --mode full --env prod --experiment "final-run-v3"
```

#### `uv run launcher`
Interactive menu system (recommended for beginners).

**Usage:**
```powershell
uv run launcher
```

**Menu Options:**
1. Quick Test Run (30 generations)
2. Production Run (2000 generations)
3. Thesis Experiments (Exp 1-5)
4. Runtime Modes (Modes 1-10)
5. RL Training & Management
6. Diagnostics & Validation
7. Results Analysis
8. Configuration Management
9. Exit

**Features:**
- Guided workflow
- Input validation
- Progress tracking
- Result summaries

---

### Environment Entry Points

#### `uv run test`
Quick smoke test for validation.

**Specifications:**
- **Generations:** 30
- **Population:** 10
- **Time:** 5-10 minutes
- **Use Case:** Bug testing, config validation, quick iteration

**Usage:**
```powershell
uv run test
```

#### `uv run prod`
Full production run for best quality.

**Specifications:**
- **Generations:** 2000
- **Population:** 200
- **Time:** 1-2.5 hours (with GPU)
- **Use Case:** Thesis experiments, final results, benchmarking

**Usage:**
```powershell
uv run prod
```

---

### Thesis Experiment Commands

#### `uv run exp1` / `uv run thesis-exp1-baseline`
**Experiment 1: Pure NSGA-II Baseline**

Establishes baseline performance without any enhancements.

**Features:**
- Pure genetic algorithm (crossover + mutation)
- No repairs, no heuristics, no RL
- Standard NSGA-II selection

**Config:** `configs/baseline/1-pure-nsga.yaml`

**Usage:**
```powershell
# Test run
uv run exp1 --env test

# Production run
uv run exp1 --env prod
```

#### `uv run exp2` / `uv run thesis-exp2-repairs`
**Experiment 2: + IGLS Repair System**

Adds iterative greedy local search repairs.

**Features:**
- Baseline + IGLS repair system
- Exhaustive initial repair
- Stagnation-triggered repairs

**Config:** `configs/nsga/2-nsga-repairs.yaml`

**Usage:**
```powershell
uv run exp2 --env prod
```

#### `uv run exp3` / `uv run thesis-exp3-heuristics`
**Experiment 3: + 19 Heuristic Operators**

Adds heuristic toolbox (no local search).

**Features:**
- Baseline + repairs + 19 heuristics
- Parallel heuristic execution
- No LNS local search

**Config:** `configs/nsga/3-nsga-heuristics.yaml`

**Usage:**
```powershell
uv run exp3 --env prod
```

#### `uv run exp4` / `uv run thesis-exp4-local-search`
**Experiment 4: + Local Search (LNS)**

Full GA with local search.

**Features:**
- All previous + LNS local search
- Best non-RL configuration
- Round-robin heuristic selection

**Config:** `configs/hybrid/6-roundrobin.yaml`

**Usage:**
```powershell
uv run exp4 --env prod
```

#### `uv run exp5` / `uv run thesis-exp5-rl`
**Experiment 5: + RL-Guided Selection**

Reinforcement learning guided heuristic selection.

**Features:**
- Full GA + RL agent
- Adaptive heuristic selection
- Requires trained model

**Config:** `configs/rl/5-rl-guided.yaml`

**Prerequisites:**
```powershell
# Train RL model first
uv run train-rl --timesteps 100000 --env prod

# Promote best checkpoint
uv run promote-model
```

**Usage:**
```powershell
uv run exp5 --env prod
```

---

### RL Training & Management

#### `uv run train-rl`
Train RL agent for heuristic selection.

**Usage:**
```powershell
uv run train-rl [OPTIONS]
```

**Options:**
- `--timesteps N` - Training timesteps (default: 100000)
- `--agent {ppo|dqn}` - Agent type (default: ppo)
- `--save-path PATH` - Model save location
- `--curriculum` - Use curriculum learning

**Examples:**
```powershell
# Basic training (100K timesteps, ~2 hours)
uv run train-rl --timesteps 100000

# Quick training (10K timesteps, ~15 min)
uv run train-rl --timesteps 10000

# Curriculum training (recommended)
uv run train-curriculum
```

#### `uv run train-curriculum`
Curriculum learning (easy → medium → hard).

**Stages:**
1. **Easy:** 10 courses, 200 episodes (~30s-2min)
2. **Medium:** 20 courses, 300 episodes (~1-5min)
3. **Hard:** 40+ courses, 500 episodes (~5-15min)

**Total time:** 60-120 minutes (GPU recommended)

**Usage:**
```powershell
uv run train-curriculum
```

#### `uv run select-checkpoint`
Select best checkpoint from training.

**Usage:**
```powershell
uv run select-checkpoint
```

**Features:**
- Evaluates all checkpoints
- Validation set metrics
- Recommends best model

#### `uv run promote-model`
Promote checkpoint to production.

**Usage:**
```powershell
uv run promote-model [CHECKPOINT_PATH]
```

**Actions:**
- Updates `configs/prod.yaml`
- Sets `rl.enabled: true`
- Registers in manifest

#### `uv run validate-rl`
Validate RL model performance.

**Usage:**
```powershell
uv run validate-rl
```

**Metrics:**
- Inference latency (<10ms target)
- Action distribution
- Reward statistics

---

### Diagnostics & System Checks

#### `uv run diagnose-system`
Comprehensive system diagnostics.

**Checks:**
- Python version
- Dependencies
- GPU availability
- CUDA version
- Memory availability
- CPU cores

**Usage:**
```powershell
uv run diagnose-system
```

#### `uv run diagnose-gpu`
GPU-specific diagnostics.

**Checks:**
- NVIDIA driver
- CUDA availability
- GPU memory
- PyTorch GPU support

**Usage:**
```powershell
uv run diagnose-gpu
```

#### `uv run check-data`
Validate input data integrity.

**Checks:**
- File existence
- JSON syntax
- Data structure
- Required fields
- Cross-references

**Usage:**
```powershell
uv run check-data
```

#### `uv run verify-config`
Validate configuration files.

**Checks:**
- YAML syntax
- Required fields
- Type validation
- Killswitch consistency

**Usage:**
```powershell
uv run verify-config
```

#### `uv run verify-enhancements`
Verify Phase 3 enhancements.

**Usage:**
```powershell
uv run verify-enhancements
```

---

### Results Analysis

#### `uv run compare-experiments`
Compare thesis experiment results.

**Features:**
- Side-by-side comparison
- Statistical tests
- Performance metrics
- Visual comparison

**Usage:**
```powershell
uv run compare-experiments
```

#### `uv run generate-thesis-plots`
Generate publication-quality plots.

**Plots:**
- Fitness evolution
- Convergence curves
- Pareto fronts
- Diversity metrics
- Box plots

**Usage:**
```powershell
uv run generate-thesis-plots
```

#### `uv run export-thesis-data`
Export metrics to CSV/LaTeX.

**Formats:**
- CSV (spreadsheet import)
- LaTeX tables (thesis)
- JSON (programmatic)

**Usage:**
```powershell
uv run export-thesis-data
```

#### `uv run analyze-convergence`
Convergence analysis.

**Metrics:**
- Convergence speed
- Stagnation detection
- Generation-to-best
- Improvement rate

**Usage:**
```powershell
uv run analyze-convergence
```

#### `uv run analyze-diversity`
Diversity metrics analysis.

**Metrics:**
- Genotypic diversity
- Phenotypic diversity
- Diversity maintenance
- Convergence vs diversity

**Usage:**
```powershell
uv run analyze-diversity
```

---

### Benchmarking

#### `uv run benchmark-gpu`
GPU performance benchmarking.

**Tests:**
- Batch evaluation speed
- CPU vs GPU comparison
- Scaling analysis

**Usage:**
```powershell
uv run benchmark-gpu
```

#### `uv run benchmark-lns`
LNS repair system benchmarking.

**Usage:**
```powershell
uv run benchmark-lns
```

#### `uv run benchmark-constraints`
Constraint evaluation benchmarking.

**Usage:**
```powershell
uv run benchmark-constraints
```

#### `uv run benchmark-all`
Run all benchmarks.

**Time:** ~30 minutes

**Usage:**
```powershell
uv run benchmark-all
```

---

### Configuration Utilities

#### `uv run show-config`
Display full configuration.

**Usage:**
```powershell
uv run show-config [--env ENV]
```

#### `uv run show-repair`
Show repair system config.

**Usage:**
```powershell
uv run show-repair
```

#### `uv run show-soft`
Show soft constraints only.

**Usage:**
```powershell
uv run show-soft
```

#### `uv run show-time`
Show time system config.

**Usage:**
```powershell
uv run show-time
```

#### `uv run list-experiments`
List all available experiments.

**Usage:**
```powershell
uv run list-experiments
```

---

### Development Utilities

#### `uv run tensorboard`
Start TensorBoard server.

**Usage:**
```powershell
uv run tensorboard
```

Opens: `http://localhost:6006`

#### `uv run git-squash`
Interactive commit squashing.

**Usage:**
```powershell
uv run git-squash
```

#### `uv run clean-output`
Clean old output files.

**Usage:**
```powershell
uv run clean-output [--days N]
```

---

## UV Package Management

### Core UV Commands

#### Install Dependencies
```powershell
# Install from lock file (recommended)
uv sync --frozen

# Install with updates
uv sync

# Install dev dependencies
uv sync --group dev
```

#### Add/Remove Packages
```powershell
# Add package
uv add package-name

# Add dev package
uv add --dev pytest

# Remove package
uv remove package-name
```

#### Update Dependencies
```powershell
# Update all
uv sync --upgrade

# Update specific package
uv add package-name --upgrade
```

#### Virtual Environment
```powershell
# Create venv
uv venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source .venv/bin/activate
```

---

## Python Direct Invocation

If UV not available, use Python directly:

```powershell
# Environment entry points
python main.py --env test
python main.py --env prod

# Runtime modes
python main.py --mode baseline --env test
python main.py --mode full --env prod

# Custom config
python main.py --config configs/my-custom.yaml
```

---

## Chaining Commands

### Run Full Thesis Pipeline

```powershell
# Run all experiments sequentially
uv run exp1 --env prod
uv run exp2 --env prod
uv run exp3 --env prod
uv run exp4 --env prod
uv run exp5 --env prod

# Compare and generate plots
uv run compare-experiments
uv run generate-thesis-plots
uv run export-thesis-data
```

### Train and Deploy RL

```powershell
# Full RL pipeline
uv run train-curriculum
uv run select-checkpoint
uv run promote-model
uv run validate-rl

# Run RL experiment
uv run exp5 --env prod
```

---

## Tips & Best Practices

### 1. Always Test First
```powershell
# Test with quick run before production
uv run <command> --env test
# If successful, run production
uv run <command> --env prod
```

### 2. Use Experiment Tags
```powershell
uv run exp1 --env prod --experiment "thesis-final-v3"
```

### 3. Monitor Progress
```powershell
# Open TensorBoard for RL training
uv run tensorboard &

# In another terminal
uv run train-rl
```

### 4. Check System Before Long Runs
```powershell
# Diagnose system
uv run diagnose-system

# Check data
uv run check-data

# Verify config
uv run verify-config
```

### 5. Save Output Logs
```powershell
# Redirect output to file
uv run prod 2>&1 | Tee-Object -FilePath "logs/run-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
```

---

## Troubleshooting

### Command Not Found
```powershell
# Ensure UV is installed
uv --version

# Reinstall UV
irm https://astral.sh/uv/install.ps1 | iex
```

### Permission Errors
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Import Errors
```powershell
# Reinstall dependencies
uv sync --frozen
```

---

## See Also

- [First Run Guide](03-first-run.md) - Step-by-step first run
- [Setup Guide](02-setup.md) - Configuration details
- [Runtime Modes Guide](../02-user-guides/runtime-modes.md) - Mode documentation
- [Troubleshooting](../troubleshooting/01-common-issues.md) - Common issues
