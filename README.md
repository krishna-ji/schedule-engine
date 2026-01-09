# Schedule Engine

[![Type Safety](https://img.shields.io/badge/mypy-strict%20mode-brightgreen)](https://mypy-lang.org/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

University course scheduling engine that combines NSGA-II genetic algorithms with reinforcement learning, CPU-parallelized fitness evaluation, and rich orchestration tooling. The system targets large university timetabling problems with hard/soft constraints and staged runtime modes for experiments and production runs.

## Highlights

- Progressive runtime modes (baseline NSGA-II through multi-agent RL) with killswitch-controlled features.
- CPU multiprocessing for parallel fitness evaluation, operators, and local search phases.
- GPU acceleration reserved for RL training and neural network inference (where it excels).
- **Python dataclass configuration** with DRY inheritance (`BaseConfig → TestConfig/ProdConfig → ExperimentConfig`).
- **100% strict mypy typing** on all pure Python packages (diversity, heuristics, workflows, utils, config, entities, constraints, metrics).
- Extensive CLI and UV script shortcuts for experiments, diagnostics, benchmarking, and RL training.
- Rich documentation in `docs/` covering algorithms, architecture, and thesis guides.

## Quick Start

```bash
uv sync --frozen          # Install dependencies
uv run launcher           # Interactive experiment launcher
uv run baseline --test    # Quick smoke test (Mode A: Pure NSGA-II)
uv run memetic --test     # Mode B: + Memetic local search
```

## Progressive Mode Experiments (A→F)

**Systematic ablation study** with increasing complexity:

| Command | Mode | Description | Killswitches |
|---------|------|-------------|--------------|
| `uv run baseline --test/prod` | **A** | Pure NSGA-II | All OFF |
| `uv run memetic --test/prod` | **B** | + Memetic local search | `repair_enabled=True`, `memetic_mode=True` |
| `uv run roundrobin --test/prod` | **C** | + Round-robin heuristics | `heuristics_master_enabled=True` |
| `uv run adaptive --test/prod` | **D** | + Adaptive selection | `heuristic_selection_mode="adaptive"` |
| `uv run rl --test/prod` | **E** | + RL-guided control | `rl_enabled=True` (requires trained model) |
| `uv run heuristic-testing --test/prod` | **F** | Individual heuristic tests | Auto-named output folders |

**Profiles:** `--test` (30 gens, 10 pop, ~2-5 min) • `--prod` (2000 gens, 400 pop, ~1-3 hours)

## RL Training & Inference

```bash
# Train RL agents
uv run train-rl --test      # Smoke test (10K steps, ~5-10 min)
uv run train-rl --prod      # Full training (100K steps, ~1-2 hrs)

# Run inference with latest trained model
uv run rl-inference --test  # Auto-detect latest model (smoke)
uv run rl-inference --prod  # Auto-detect latest model (production)
uv run rl-inference --list-only  # List all available models
```

## Helper Commands

```bash
# Utilities
uv run diagnose             # System/GPU diagnostics
uv run clean                # Clean output directory
uv run list-experiments     # Show experiment history
uv run stats                # Manifest statistics
uv run archive              # Archive incomplete runs
## Repository Layout

- `src/` – Core engine (config, GA, constraints, RL, workflows, utils)
- `configs/` – Python dataclasses (`base.py`, `profiles.py`, `experiments/*.py`)
- `scripts/` – CLI launcher, utilities, diagnostics, training
- `test/` – Unit and integration tests
- `docs/` – User guides, architecture, experiment playbooks, thesis material
- `output/` – Experiment results, logs, manifests

## Code Quality

```bash
# Format & Lint
black src/ test/             # Auto-format (PEP 8, line length 88)
ruff check src/ test/        # Fast linter

# Type Checking (strict mode)
mypy src/                    # 100% coverage on pure Python packages

# Testing
pytest test/unit/            # Unit tests
pytest --cov=src test/       # With coverage
```

## Configuration System

**Python dataclass hierarchy** with DRY inheritance:

```python
# configs/base.py - Shared defaults
@dataclass
class BaseConfig:
    ngen: int = 100
    pop_size: int = 50
    repair_enabled: bool = False

# configs/profiles.py - Scaling overrides
@dataclass
class ProdConfig(BaseConfig):
    ngen: int = 2000
    pop_size: int = 400

# configs/experiments/memetic.py - Killswitch states
@dataclass
class MemeticBaseConfig:
    repair_enabled: bool = True
    memetic_mode: bool = True

class MemeticProdConfig(MemeticBaseConfig, ProdConfig):
```

**Access in code:**

```pythonin code:**
```python
from src.config import get_config
config = get_config()  # Returns Pydantic Config model
```

**Type safety:** All configs are strictly typed with mypy 1.13.0 strict mode.

## Repository Layout

- `src/` – Core engine (config, GA, constraints, RL, workflows, utils)
- `configs/` – Python dataclasses (`base.py`, `profiles.py`, `experiments/*.py`)
- `scripts/` – CLI launcher, utilities, diagnostics, training
- `test/` – Unit and integration tests
- `docs/` – User guides, architecture, experiment playbooks, thesis material
- `output/` – Experiment results, logs, manifests

## Code Quality

```bash
# Format & Lint
black src/ test/             # Auto-format (PEP 8, line length 88)
ruff check src/ test/        # Fast linter

# Type Checking (strict mode)
mypy src/                    # 100% coverage on pure Python packages

# Testing
pytest test/unit/            # Unit tests
pytest --cov=src test/       # With coverage
```

## Documentation

Start with `docs/INDEX.md` for full navigation, or see:

- `docs/02-user-guides/runtime-modes.md` – Progressive mode reference (A→F)
- `docs/06-development/implementation-notes/PHASE_3_ADVANCED_RL.md` – GA/RL enhancements
- `docs/45-resource-unused-problem/THESIS_EXPERIMENTS_GUIDE.md` – Thesis experiment guide
- `.github/copilot-instructions.md` – AI coding agent guide with architecture overview

## Contributing

1. **Code Style**: Follow `.github/copilot-instructions.md` standards
2. **Format**: Run `black src/ test/` before committing
3. **Lint**: Run `ruff check src/ test/` and fix issues
4. **Type Check**: Ensure `mypy src/` passes (strict mode required)
5. **Test**: Run `pytest test/unit/` to verify changes
6. **Docstrings**: Use Google-style docstrings (no separate .md files for code)

## Tech Stack

- **Python 3.12** (strict mypy typing)
- **GA Core**: DEAP 1.4.1, NumPy 1.26.4, pymoo 0.6.1.3
- **RL Stack**: PyTorch 2.4.1+CUDA12.1, Stable-Baselines3 2.3.2, Gymnasium 0.29.1
- **Config**: Pydantic 2.10.3, Python dataclasses
- **UI**: Rich 13.9.4, matplotlib, seaborn
- **Performance**: CPU multiprocessing (32 cores), pymoo-accelerated metrics (139x speedup)

## License

MIT
