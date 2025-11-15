# Schedule Engine - Agent Guide

## Project Summary

University course scheduling engine using NSGA-II genetic algorithm with constraint-based optimization. Written in Python with DEAP, rich terminal UI, and YAML configuration.

## Tech Stack

- **Language**: Python 3.11+
- **Core Libraries**: DEAP (genetic algorithms), Pydantic (validation), Rich (terminal UI)
- **Config**: YAML-based with base.yaml + environment overrides
- **Package Manager**: UV (uv.lock, pyproject.toml)

## Repository Structure

```
schedule-engine/
├── main.py              # CLI entry point with env-specific entry functions
├── src/config/          # Pydantic models & loader
├── configs/             # base.yaml + test/prod.yaml
├── src/
│   ├── core/            # GA scheduler & types
│   ├── ga/              # GA operators, population, repair
│   ├── constraints/     # Hard & soft constraints
│   ├── encoder/         # JSON → entities + time system
│   ├── decoder/         # Individual → CourseSession
│   ├── entities/        # Domain models
│   ├── exporter/        # PDF/JSON/plots generation
│   ├── validation/      # Input & feasibility checks
│   ├── workflows/       # Orchestration logic
│   └── utils/           # Helpers & logging
├── data/                # Input JSON files
├── test/                # Test files (ALL tests go here)
└── docs/                # Documentation
```

## Configuration System

**New simplified structure:**
- `configs/base.yaml` - All common settings (shared)
- `configs/test.yaml` - Smoke test overrides (30 gens, 10 pop)
- `configs/prod.yaml` - Best quality overrides (2000 gens, 200 pop)

Environment configs inherit from base.yaml via deep merge in `src/config/loader.py`.

**Access config:** `from src.config import get_config; config = get_config()`

## Running the Engine

```bash
# UV commands (recommended)
uv run test      # Smoke test (30 gens, ~5-10 min)
uv run prod      # Best quality (2000 gens, ~24-48 hours)

# Or Python directly
python main.py --env test
python main.py --env prod
python main.py --config path/to/custom.yaml
```

## Architecture

- **Entry Point**: `main.py` with `main()` + environment-specific entry functions (`main_prod()`, `main_test()`)
- **Workflow**: `src/workflows/standard_run.py` orchestrates: load → validate → feasibility → GA → decode → report
- **GA Core**: `src/core/ga_scheduler.py` - GAScheduler class with DEAP toolbox, population init, evolution
- **Multiprocessing**: Enabled via `parallel.use_multiprocessing` in YAML

## Key Components

- **Chromosomes**: `list[SessionGene]`, fitness `(-hard, -soft)` with weights `(-1.0, -0.01)`
- **Population Strategies**: hybrid (25% greedy, 50% smart, 25% random) / smart / random
- **Operators**: `crossover_course_group_aware()`, `mutate_individual()`
- **Repair**: IGLS system with exhaustive search, stagnation repair, selective repair
- **Constraints**: Hard (must-satisfy) and soft (prefer-satisfy) in `src/constraints/`
- **Time System**: `QuantumTimeSystem` converts wall-clock ↔ discrete quanta (default 60 min)
- **Validation**: Input validation + feasibility checking before GA
- **Exports**: JSON, PDF calendar, plots to `output/evaluation_<timestamp>/`

## General Coding Standards

- **Python Style**: PEP 8 compliant
- **Imports**: Standard lib → third-party → local (sorted alphabetically)
- **Type Hints**: Use where beneficial
- **Error Handling**: Informative error messages
- **Logging**: Rich console for user-facing, logger for debugging
- **Docstrings**: Required for all modules, classes, functions (NO separate .md files for code docs)
- **Config Access**: `from src.config import get_config` (never import old `config.ga_params`)

## Documentation Policy

### 1. Code Documentation
**Use Python docstrings only** - no separate .md files for code.

### 2. Minor Changes → `docs/code/`
Bugfixes, small enhancements, refactoring:
- Add timestamped entry to `docs/code/{BUGFIX,ENHANCE,REFACTOR}.md`
- Format: `## [YYYY-MM-DD] Brief description` + file list
- No detailed explanations needed

### 3. Major Changes → `docs/for_report/`
New algorithms, architectural changes:
- Create new file in `docs/for_report/`
- Thesis-ready prose with placement comment
- Structure: Problem → Solution → Implementation → Results

## Commit Message Format

Format: `<type>(<scope>): <summary>`

Types: `feat`, `fix`, `refactor`, `test`, `doc`, `data`, `method`, `analysis`

Use imperative mood, keep under 72 characters.

Example: `feat(config): simplify to base.yaml + 3 environments`

## Path-Specific Instructions

Detailed module-specific instructions in `.github/instructions/`:

- `config.instructions.md` - Configuration system
- `ga-core.instructions.md` - GA operators & scheduler
- `constraints.instructions.md` - Constraint functions
- `data-flow.instructions.md` - Encoder/decoder/entities
- `validation.instructions.md` - Input & feasibility validation
- `export.instructions.md` - Report generation & plotting
- `workflows.instructions.md` - Orchestration logic
- `tests.instructions.md` - Testing guidelines
