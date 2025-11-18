# Runtime Modes User Guide

## Overview

The Schedule Engine supports **6 runtime modes** for systematic benchmarking and research experiments. Each mode represents a different configuration of the GA scheduler with specific features enabled/disabled via **killswitches**.

This modular architecture enables:
- **Easy comparison studies** (baseline vs enhanced methods)
- **Reproducible experiments** (controlled feature sets)
- **Production flexibility** (select best mode for deployment)
- **Research workflows** (test individual components)

---

## Architecture

### Configuration Structure

```
configs/
├── base.yaml                  # Common settings (inherited by all)
├── test.yaml                  # Environment: smoke test
├── prod.yaml                  # Environment: best quality
├── baseline/
│   └── 1-pure-nsga.yaml      # Mode 1: Pure NSGA-II
├── nsga/
│   ├── 2-nsga-repairs.yaml   # Mode 2: + Repairs
│   ├── 3-nsga-heuristics.yaml # Mode 3: + Heuristics
│   └── 4-nsga-full.yaml      # Mode 4: + Local Search
├── rl/
│   └── 5-rl-guided.yaml      # Mode 5: RL-guided
└── hybrid/
    └── 6-roundrobin.yaml     # Mode 6: Round-robin
```

### Output Organization

Experiment outputs are automatically organized by runtime mode:

```
output/
├── experiment_manifest.json   # Tracks all runs
├── baseline/
│   └── pure-nsga/
│       ├── evaluation_20251118_140530/
│       └── evaluation_20251118_153200_exp1/
├── nsga/
│   ├── nsga-repairs/
│   ├── nsga-heuristics/
│   └── nsga-full/
├── rl/
│   └── rl-guided/
└── hybrid/
    └── roundrobin/
```

---

## Runtime Modes

### Mode 1: Pure NSGA-II (Baseline) 🔴

**File:** `configs/baseline/1-pure-nsga.yaml`

**Description:** Minimal NSGA-II with no repairs, no heuristics, no enhancements. Use as baseline for comparing all other modes.

**Features:**
- ✅ Pure NSGA-II genetic algorithm
- ✅ Random population initialization
- ❌ No repairs (IGLS disabled)
- ❌ No heuristics (Phase 1.5 disabled)
- ❌ No local search (LNS-IGLS disabled)
- ❌ No RL guidance
- ❌ No enhancements

**Usage:**
```bash
# CLI
python main.py --mode baseline

# Or
python main.py --mode 1-pure-nsga
```

**Use Cases:**
- Research baseline for benchmarking
- Understanding pure GA performance
- Isolating effects of enhancements

---

### Mode 2: NSGA-II + Repairs 🟡

**File:** `configs/nsga/2-nsga-repairs.yaml`

**Description:** NSGA-II with IGLS repair system but no advanced heuristics. Tests effectiveness of repair system alone.

**Features:**
- ✅ NSGA-II genetic algorithm
- ✅ Hybrid population initialization (25% greedy, 50% smart, 25% random)
- ✅ IGLS repair system (stagnation + selective)
- ❌ No Phase 1.5 heuristics
- ❌ No local search
- ❌ No RL guidance
- ❌ No enhancements

**Usage:**
```bash
python main.py --mode nsga-repairs
# or
python main.py --mode repairs
```

**Use Cases:**
- Testing repair system effectiveness
- Comparing repair vs no-repair
- Baseline for heuristic evaluation

---

### Mode 3: NSGA-II + Repairs + Heuristics 🟢

**File:** `configs/nsga/3-nsga-heuristics.yaml`

**Description:** NSGA-II with repairs + Phase 1.5 heuristic toolbox (19 operators). Tests effectiveness of heuristic operators.

**Features:**
- ✅ NSGA-II genetic algorithm
- ✅ Hybrid population initialization
- ✅ IGLS repair system
- ✅ **Phase 1.5 heuristics** (19 operators):
  - Construction: Largest-degree-first, Most-constrained-first, Earliest-deadline-first
  - Perturbation: Random swap, Temporal shift, Room shuffle, Instructor reassign
  - Improvement: Kempe chain, Ejection chain, Variable-depth search
  - Diversity: Distance-preserving crossover, Crowding mutation, Niching, Adaptive diversity
  - Meta: VND, ILS, ALNS, Guided local search
- ✅ Constraint-guided mutation
- ❌ No memetic local search
- ❌ No RL guidance

**Usage:**
```bash
python main.py --mode nsga-heuristics
# or
python main.py --mode heuristics
```

**Use Cases:**
- Evaluating Phase 1.5 heuristic toolbox
- Comparing heuristics vs basic GA
- Understanding operator contributions

---

### Mode 4: NSGA-II + Full (Best GA) 🔵

**File:** `configs/nsga/4-nsga-full.yaml`

**Description:** Full NSGA-II with repairs, heuristics, and LNS-IGLS local search. This is the "best GA" configuration without RL.

**Features:**
- ✅ NSGA-II genetic algorithm
- ✅ Hybrid population initialization
- ✅ IGLS repair system with **memetic mode**
- ✅ Phase 1.5 heuristics (all 19 operators)
- ✅ **LNS-IGLS local search**
- ✅ **Adaptive probabilities**
- ✅ **All enhancements enabled**:
  - Hypermutation
  - Constraint priorities
  - Population restart
  - Violation heatmap
  - Multi-neighborhood search
- ❌ No RL guidance

**Usage:**
```bash
python main.py --mode nsga-full
# or
python main.py --mode full
```

**Use Cases:**
- Best non-RL configuration
- Production baseline
- Benchmark for RL comparison

---

### Mode 5: RL-Guided 🤖

**File:** `configs/rl/5-rl-guided.yaml`

**Description:** Full NSGA-II with RL agent controlling heuristic selection. RL guides both repair strategies and local search budget.

**Features:**
- ✅ NSGA-II genetic algorithm
- ✅ IGLS repair system with memetic mode
- ✅ Phase 1.5 heuristics (RL selects from these)
- ✅ LNS-IGLS local search
- ✅ All enhancements enabled
- ✅ **RL agent** (PPO) in inference mode
  - Trained model: `models/rl_agents/best_model.zip`
  - State space: 39D (constraint-specific)
  - Reward: Multi-objective with hypervolume
  - Action: Heuristic selection (19 options)
- ✅ GPU acceleration (CUDA enabled)

**Usage:**
```bash
python main.py --mode rl-guided
# or
python main.py --mode rl
```

**Requirements:**
- Trained RL model at `models/rl_agents/best_model.zip`
- GPU recommended (10x faster inference)

**Use Cases:**
- Testing RL-guided search
- Production deployment (if trained)
- Research: RL vs fixed strategies

---

### Mode 6: Round-Robin 🔄

**File:** `configs/hybrid/6-roundrobin.yaml`

**Description:** Full NSGA-II with round-robin heuristic selection (no RL). Cycles through all enabled heuristics in fixed order.

**Features:**
- ✅ NSGA-II genetic algorithm
- ✅ IGLS repair system with memetic mode
- ✅ Phase 1.5 heuristics (applied in round-robin by priority)
- ✅ LNS-IGLS local search
- ✅ All enhancements enabled
- ✅ **Fixed round-robin scheduling**
  - No adaptive probabilities
  - Deterministic heuristic rotation
  - Priority-based ordering
- ❌ No RL guidance

**Usage:**
```bash
python main.py --mode roundrobin
# or
python main.py --mode rr
```

**Use Cases:**
- Baseline for RL comparison
- Deterministic heuristic application
- Understanding fixed strategies

---

## CLI Usage

### Basic Commands

```bash
# Run with specific mode
python main.py --mode baseline
python main.py --mode nsga-full
python main.py --mode rl-guided

# Combine with environment
python main.py --mode baseline --env test    # Smoke test
python main.py --mode nsga-full --env prod   # Best quality

# Add experiment name
python main.py --mode rl-guided --experiment "rl-test-run-1"

# List all modes
python main.py --list-modes

# Compare all runs
python main.py --compare
```

### UV Shortcuts

Update `pyproject.toml` to add mode-specific shortcuts:

```toml
[project.scripts]
baseline = "main:main_baseline"
repairs = "main:main_repairs"
heuristics = "main:main_heuristics"
full = "main:main_full"
rl = "main:main_rl"
roundrobin = "main:main_roundrobin"
```

Then run:

```bash
uv run baseline
uv run full
uv run rl
```

---

## Experiment Workflow

### 1. Run Baseline

```bash
python main.py --mode baseline --env prod --experiment "baseline-v1"
```

**Output:** `output/baseline/pure-nsga/evaluation_20251118_140530_baseline-v1/`

### 2. Run Enhanced Modes

```bash
python main.py --mode nsga-repairs --env prod --experiment "repairs-v1"
python main.py --mode nsga-heuristics --env prod --experiment "heuristics-v1"
python main.py --mode nsga-full --env prod --experiment "full-v1"
```

### 3. Compare Results

```bash
python main.py --compare
```

**Output:**

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Mode                       ┃ Runs ┃ Best Hard ┃ Best Soft ┃ Avg Duration ┃ Latest     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Pure NSGA-II (Baseline)    │    3 │      12.0 │     45.23 │       3600.0s│ 2025-11-18 │
│ NSGA-II + Repairs          │    2 │       8.0 │     32.15 │       4200.5s│ 2025-11-18 │
│ NSGA-II + Repairs + ...    │    2 │       4.0 │     18.67 │       5100.2s│ 2025-11-18 │
│ NSGA-II + Full (...)       │    1 │       0.0 │      7.42 │       6800.0s│ 2025-11-18 │
│ RL-Guided Heuristic ...    │    0 │         - │         - │            - │ -          │
│ Round-Robin Heuristic ...  │    1 │       2.0 │     12.34 │       5400.0s│ 2025-11-18 │
└────────────────────────────┴──────┴───────────┴───────────┴──────────────┴────────────┘
```

### 4. Export for Analysis

```python
from src.workflows.experiment_manager import ExperimentManager

manager = ExperimentManager()
manager.export_comparison_csv("output/comparison.csv")
```

---

## Killswitch Reference

### Core Killswitches

| Feature | Baseline | Repairs | Heuristics | Full | RL | RoundRobin |
|---------|----------|---------|------------|------|-------|------------|
| `repair.enabled` | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `heuristics.*.enabled` | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| `repair.memetic_mode` | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `lns.enabled` | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `enhancements.master_enabled` | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
| `ga.use_adaptive_probabilities` | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| `rl.enabled` | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |

### Validation

Configs are automatically validated on load:

```python
# Example: Baseline mode validation
if runtime_mode == RuntimeMode.BASELINE:
    if config.repair.enabled:
        raise ValueError("Baseline must have repair.enabled=false")
    if config.rl.enabled:
        raise ValueError("Baseline must have rl.enabled=false")
```

---

## Experiment Manager API

### Python Interface

```python
from src.workflows.experiment_manager import ExperimentManager
from src.config.runtime_mode import RuntimeMode

# Initialize
manager = ExperimentManager(base_output_dir="output")

# Create output directory
output_dir = manager.create_output_dir(
    runtime_mode=RuntimeMode.NSGA_FULL,
    experiment_name="full-test-1"
)
# → output/nsga/nsga-full/evaluation_20251118_140530_full-test-1/

# Register run
run = manager.register_run(
    runtime_mode=RuntimeMode.NSGA_FULL,
    config_path=Path("configs/nsga/4-nsga-full.yaml"),
    output_path=output_dir,
    experiment_name="full-test-1",
    seed=69
)

# Update with results
manager.update_run_results(
    run=run,
    duration_seconds=6800.5,
    generations=2000,
    population_size=200,
    best_hard_violations=0.0,
    best_soft_penalty=7.42
)

# Query runs
runs = manager.get_runs_by_mode(RuntimeMode.NSGA_FULL)
latest = manager.get_latest_run(RuntimeMode.NSGA_FULL)

# Compare modes
table = manager.compare_modes(modes=[RuntimeMode.BASELINE, RuntimeMode.NSGA_FULL])
console.print(table)

# Export CSV
manager.export_comparison_csv(Path("output/comparison.csv"))

# Clean old runs
manager.clean_old_runs(keep_last_n=10)
```

---

## Recommended Experiments

### Experiment 1: Feature Ablation Study

Test each component's contribution:

```bash
# 1. Baseline (no features)
python main.py --mode baseline --env prod --experiment "ablation-baseline"

# 2. + Repairs
python main.py --mode nsga-repairs --env prod --experiment "ablation-repairs"

# 3. + Heuristics
python main.py --mode nsga-heuristics --env prod --experiment "ablation-heuristics"

# 4. + Local search
python main.py --mode nsga-full --env prod --experiment "ablation-full"

# Compare
python main.py --compare
```

### Experiment 2: RL vs Fixed Strategies

Compare RL-guided vs deterministic approaches:

```bash
# 1. RL-guided
python main.py --mode rl-guided --env prod --experiment "rl-vs-fixed-1"

# 2. Round-robin
python main.py --mode roundrobin --env prod --experiment "rl-vs-fixed-2"

# 3. Full (adaptive)
python main.py --mode nsga-full --env prod --experiment "rl-vs-fixed-3"
```

### Experiment 3: Scalability Study

Test modes at different problem sizes:

```bash
# Small (10 courses)
python main.py --mode baseline --experiment "scale-small-baseline"
python main.py --mode nsga-full --experiment "scale-small-full"

# Medium (20 courses)
python main.py --mode baseline --experiment "scale-med-baseline"
python main.py --mode nsga-full --experiment "scale-med-full"

# Large (40 courses)
python main.py --mode baseline --experiment "scale-large-baseline"
python main.py --mode nsga-full --experiment "scale-large-full"
```

---

## Troubleshooting

### Config Validation Errors

**Problem:** Config violates mode constraints.

**Solution:** Check killswitches match mode expectations.

```bash
# Error example:
[!ERR] Config validation failed for mode 1-pure-nsga: Baseline mode must have repair.enabled=false

# Fix: Ensure configs/{category}/{mode}.yaml has correct killswitches
```

### Missing RL Model

**Problem:** RL mode can't find trained model.

**Solution:** Train RL agent first or use fallback mode.

```bash
# Train RL agent
python src/rl/training/train_script.py --timesteps 100000

# Or use fallback
# Edit configs/rl/5-rl-guided.yaml:
rl:
  hybrid:
    mode: rl_fallback  # Falls back to greedy if model missing
```

### Output Directory Conflicts

**Problem:** Multiple runs overwrite each other.

**Solution:** Use unique experiment names or timestamps.

```bash
# Good
python main.py --mode baseline --experiment "run-1"
python main.py --mode baseline --experiment "run-2"

# Bad (overwrites)
python main.py --mode baseline  # No experiment name
```

---

## Next Steps

- **Run baseline experiments** to establish performance floor
- **Test individual enhancements** (repairs, heuristics, local search)
- **Train RL agent** for RL-guided mode
- **Compare all modes** systematically
- **Select best mode** for production deployment

For implementation details, see:
- `docs/03-architecture/` - System architecture
- `docs/04-algorithms/` - Algorithm documentation
- `docs/06-development/implementation-notes/` - Implementation notes
