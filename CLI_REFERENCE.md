# CLI Reference - Schedule Engine

## Convention

**Main Commands (0-9)**: Numbers for primary experiments (GA, RL)  
**Helper Commands (a-z)**: Letters for utilities (diagnose, clean)  
**Profiles**: `--test` (smoke), `--med` (medium), `--prod` (full production)

## Profile Hierarchy (DRY Principle)

```
base.yaml (common settings)
  ↓
test.yaml (30 gens, 10 pop) - inherits base + overrides for smoke test
  ↓
med.yaml (200 gens, 100 pop) - inherits test + overrides for medium
  ↓
prod.yaml (2000 gens, 500 pop) - inherits med + overrides for production
```

---

## Main Commands (0-9)

### NSGA-II Experiments

#### Command 0: Pure NSGA-II Baseline

```bash
# Smoke test (30 gens, ~2 min)
uv run nsga --test

# Medium (200 gens, ~30 min)
uv run nsga --med

# Production (2000 gens, ~3-5 hours)
uv run nsga --prod

# Custom name
uv run nsga --prod --name "my-experiment"
```

**What it does:**
- Runs pure NSGA-II genetic algorithm
- Uses runtime mode: baseline (mode 1)
- No repairs, no heuristics, no RL
- Perfect for baseline comparison

---

### RL Training

#### Command 5: RL Training

```bash
# Smoke test (10K steps, ~5-10 min)
uv run train-rl --test

# Medium (50K steps, ~30-45 min)
uv run train-rl --med

# Production (100K steps, ~1-2 hours)
uv run train-rl --prod

# With curriculum learning (auto-enabled in config)
uv run train-rl --prod --curriculum
```

**What it does:**
- Trains PPO agent to select heuristics
- Uses curriculum learning (easy → medium → hard)
- Saves model to `models/rl_agents/`
- Logs to TensorBoard (`logs/tensorboard/`)

**Profile Mapping:**
- `--test`: 10,000 timesteps (3 curriculum stages)
- `--med`: 50,000 timesteps
- `--prod`: 100,000 timesteps

---

## Helper Commands (a-z)

### Utilities

### Command a: Diagnose System

```bash
uv run diagnose
```

**Checks:**
- GPU availability (CUDA)
- Python environment
- Dependencies
- Data integrity
- Config validation

---

### Command b: Clean Output

```bash
uv run clean
```

**Removes:**
- Old experiment outputs
- Temporary files
- Keeps manifest.json

---

### Command c: List Experiments

```bash
uv run list-experiments
```

**Shows:**
- All completed experiments
- Timestamps and configurations
- Best fitness scores

---

## Quick Reference

### NSGA-II Experiments

| Command | Profile | Generations | Pop Size | Time | Use Case |
|---------|---------|-------------|----------|------|----------|
| `nsga --test` | test | 30 | 10 | ~2 min | Smoke test locally |
| `nsga --med` | med | 200 | 100 | ~30 min | Medium validation |
| `nsga --prod` | prod | 2000 | 500 | ~3-5 hrs | Full production (VM) |

### RL Training

| Command | Profile | Timesteps | Time | Use Case |
|---------|---------|-----------|------|----------|
| `train-rl --test` | test | 10K | ~5-10 min | Verify RL works |
| `train-rl --med` | med | 50K | ~30-45 min | Medium training |
| `train-rl --prod` | prod | 100K | ~1-2 hrs | Full training |

### Utilities

| Command | Description | Time |
|---------|-------------|------|
| `diagnose` | System check | ~1 min |
| `clean` | Clean outputs | instant |
| `list-experiments` | Show history | instant |

---

## Advanced Usage

### Custom Config

```bash
# Use custom config file
uv run nsga --prod --config path/to/custom.yaml

# Override runtime mode
uv run nsga --test --mode full
```

### Combining Flags

```bash
# Named experiment with custom config
uv run nsga --prod --name "thesis-baseline" --config configs/custom.yaml

# RL training with specific settings
uv run train-rl --med --curriculum --name "curriculum-test"
```

---

## Legacy Commands (Backward Compatibility)

These still work but use new unified commands instead:

```bash
# Old way
uv run test-nsga
uv run prod-nsga

# New way (recommended)
uv run nsga --test
uv run nsga --prod
```

---

## Configuration Files

### GA Profiles
- `configs/test.yaml` - Quick smoke test
- `configs/med.yaml` - Medium validation
- `configs/prod.yaml` - Full production

### RL Profiles
- `configs/training/test.yaml` - Quick RL test
- `configs/training/med.yaml` - Medium RL training
- `configs/training/prod.yaml` - Full RL training

All configs use **DRY hierarchy** - each inherits from previous level.

---

## Next Steps

1. **Verify setup**: `uv run diagnose`
2. **Smoke test GA**: `uv run nsga --test` (~2 min)
3. **Smoke test RL**: `uv run train-rl --test` (~5-10 min)
4. **Run full experiments**: Use `--prod` on VM

---

## Troubleshooting

**Command not found?**
```bash
# Reinstall CLI commands
uv sync
```

**Profile not working?**
```bash
# Check available profiles
ls configs/*.yaml
ls configs/training/*.yaml
```

**Need help?**
```bash
uv run nsga --help
uv run train-rl --help
```
