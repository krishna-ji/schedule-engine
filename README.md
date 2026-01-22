# Schedule Engine

[![Type Safety](https://img.shields.io/badge/mypy-strict%20mode-brightgreen)](https://mypy-lang.org/)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

University course scheduling engine that combines NSGA-II genetic algorithms with reinforcement learning, CPU-parallelized fitness evaluation, and rich orchestration tooling. The system targets large university timetabling problems with hard/soft constraints.

## Highlights

- **Notebook-first workflow**: Configure and run experiments directly in Jupyter notebooks.
- Progressive experiment modes (baseline NSGA-II through RL-guided selection).
- CPU multiprocessing for parallel fitness evaluation, operators, and local search phases.
- GPU acceleration reserved for RL training and neural network inference.
- **Mypy with targeted strictness** for core modules.
- DRY helper module (`schedule_engine.notebooks`) for clean, reusable notebook code.

## Quick Start

```bash
uv sync --frozen          # Install dependencies
```

Then open a notebook in `notebooks/`:

| Notebook | Description |
|----------|-------------|
| `mode_a_baseline.ipynb` | **Pure NSGA-II** - baseline without enhancements |
| `mode_b_memetic.ipynb` | **+ Memetic search** - with IGLS repair |
| `mode_c_roundrobin.ipynb` | **+ Round-robin** - cycling heuristics |
| `mode_d_adaptive.ipynb` | **+ Adaptive** - performance-based selection |
| `mode_e_rl_guided.ipynb` | **+ RL-guided** - Q-learning heuristic selection |

Each notebook has inline configuration - just tweak the parameters and run!

## Notebook Helper Module

All notebooks use shared helpers from `schedule_engine.notebooks`:

```python
from schedule_engine.notebooks import (
    load_data,           # Load course/instructor/room/group data
    run_nsga2,           # Run NSGA-II evolution
    export_full_results, # Export schedule.json, calendar.pdf, plots
    EvolutionConfig,     # Configure evolution parameters
)

# Load data
data = load_data("data")

# Configure & run
config = EvolutionConfig(ngen=100, pop_size=50, cxpb=0.7, mutpb=0.2)
result = run_nsga2(data, config)

# Export results
export_full_results(result.best_individual, data, "output/my_run")
```

## CLI Utilities

```bash
uv run run              # Show help & available notebooks
uv run diagnose         # System/GPU diagnostics  
uv run clean            # Clean output directory
uv run list-experiments # Show experiment history
uv run stats            # Manifest statistics
uv run lint             # Code linting
uv run typecheck        # Type checking
```

## Code Quality

```bash
black src/schedule_engine/ test/         # Auto-format
ruff check src/schedule_engine/ test/    # Lint
mypy src/schedule_engine/                # Type check
pytest test/                             # Run tests
```
## Repository Layout

```
schedule-engine/
├── notebooks/           # Jupyter notebooks (primary workflow)
├── src/
│   ├── schedule_engine/ # Core package
│   │   ├── notebooks/   # DRY helper module for notebooks
│   │   ├── config/      # Pydantic config models
│   │   ├── constraints/ # Hard/soft constraint functions
│   │   ├── domain/      # Domain models
│   │   ├── ga/          # Operators, population
│   │   ├── heuristics/  # Heuristic operators
│   │   ├── io/          # Loading, decoding, export
│   │   ├── metrics/     # Analysis metrics
│   │   ├── rl/          # RL environment & agents
│   │   ├── utils/       # Shared utilities
│   │   └── workflows/   # Orchestration
├── scripts/             # CLI utilities
├── data/                # Input JSON files
├── output/              # Experiment results
└── test/                # Unit tests
```

## Tech Stack

- **Python 3.12**
- **GA Core**: DEAP 1.4.1, NumPy 1.26.4, pymoo 0.6.1.3
- **RL Stack**: PyTorch 2.4.1+CUDA12.1, Stable-Baselines3 2.3.2, Gymnasium 0.29.1
- **Config**: Pydantic 2.10.3
- **UI**: Rich 13.9.4, matplotlib, seaborn

## License

MIT
