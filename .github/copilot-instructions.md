# Schedule Engine - Copilot Agent Guide

## 🎯 Project Overview

**Type**: University course scheduling optimization system  
**Domain**: Educational timetabling, multi-objective optimization  
**Language**: Python 3.12+  
**Framework**: NSGA-II genetic algorithm with reinforcement learning  
**Key Libraries**: DEAP (GA), PyTorch (RL/GPU), Stable-Baselines3 (RL), Pydantic (config), Rich (UI)  
**Package Manager**: UV (modern, fast Python package installer)  
**Architecture**: Modular constraint-based optimization with 10 progressive runtime modes

## 🏗️ Build & Test Commands

### Quick Setup
```bash
# Install dependencies (UV package manager)
uv sync --frozen

# Run quick smoke test (30 generations, ~5 min)
uv run exp1 --env test

# Run production experiment (2000 generations, ~2 hours with GPU)
uv run exp1 --env prod
```

### Testing
```bash
# Run all unit tests
pytest test/unit/

# Run specific test file
pytest test/unit/test_config_loader.py

# Run with coverage
pytest --cov=src --cov-report=html test/unit/
```

### Code Quality
```bash
# Format code (Black)
black src/ test/

# Lint (Ruff)
ruff check src/ test/

# Type checking (MyPy)
mypy src/
```

### Validation Commands
```bash
# Verify configuration
uv run verify-config

# Check input data integrity
uv run check-data

# Diagnose GPU/system status
uv run diagnose-system

# List available experiments
uv run list-experiments
```

## Phase Roadmap & Status

- **Phase 1.5 – Heuristic Toolbox**: ✅ Complete (19 operators + registry, documented in `docs/06-development/implementation-notes/PHASE_1.5_SUMMARY.md`).
- **Phase 2.1 – Gymnasium Environment**: ✅ Complete (env, reward, action mapper). See `docs/06-development/implementation-notes/PHASE_2.1_SUMMARY.md`.
- **Phase 2.2-2.4 – RL Training/Deployment/Integration**: ✅ Code complete (`docs/06-development/implementation-notes/PHASE_2_RL_COMPLETE.md`). Pending execution tasks: curriculum training runs, checkpoint selection, promotion, RL-enabled GA benchmarking, and documentation updates.
- **Phase 3 – Advanced RL / Evaluation**: 🚧 Planned (multi-agent, transfer learning, evaluation suite) per `Todo.md` and `docs/10-ai-suggestions/rlphase2.2-2.4_guide_manual.md`.
- **GPU Acceleration**: ✅ Deployed (CUDA enabled in configs/base.yaml, see `docs/04-algorithms/nvidia-gpu/`).

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

**Modular structure with runtime modes:**
- `configs/base.yaml` - All common settings (shared)
- `configs/test.yaml` - Smoke test overrides (30 gens, 10 pop)
- `configs/prod.yaml` - Best quality overrides (2000 gens, 200 pop)
- `configs/baseline/1-pure-nsga.yaml` - Mode 1: Pure NSGA-II (all killswitches OFF)
- `configs/nsga/2-nsga-repairs.yaml` - Mode 2: NSGA-II + IGLS repairs
- `configs/nsga/3-nsga-heuristics.yaml` - Mode 3: NSGA-II + repairs + 19 heuristics
- `configs/nsga/4-nsga-full.yaml` - Mode 4: Full GA (best non-RL)
- `configs/rl/5-rl-guided.yaml` - Mode 5: RL-guided heuristic selection
- `configs/hybrid/6-roundrobin.yaml` - Mode 6: Fixed round-robin rotation
- `configs/rl/7-rl-specialists.yaml` - Mode 7: RL with specialist agents
- `configs/rl/8-archive-diversity.yaml` - Mode 8: Archive-based diversity
- `configs/rl/9-rl-hierarchical.yaml` - Mode 9: Hierarchical RL (two-level)
- `configs/rl/10-rl-multiagent.yaml` - Mode 10: Rank-based multi-agent RL

Environment configs inherit from base.yaml via deep merge in `src/config/loader.py`.
Runtime mode configs support automatic killswitch validation.

**Access config:** `from src.config import get_config; config = get_config()`

## Running the Engine

```bash
# UV commands (recommended)
uv run test      # Smoke test (30 gens, ~5-10 min)
uv run prod      # Best quality (2000 gens, ~24-48 hours)

# Runtime mode shortcuts (UV)
uv run baseline      # Mode 1: Pure NSGA-II baseline
uv run repairs       # Mode 2: NSGA-II + IGLS repairs
uv run heuristics    # Mode 3: NSGA-II + repairs + 19 heuristics
uv run full          # Mode 4: Full GA (best non-RL)
uv run rl            # Mode 5: RL-guided heuristic selection
uv run roundrobin    # Mode 6: Fixed round-robin rotation
uv run specialists   # Mode 7: RL with specialist agents
uv run archive       # Mode 8: Archive-based diversity
uv run hierarchical  # Mode 9: Hierarchical RL (two-level)
uv run multiagent    # Mode 10: Rank-based multi-agent RL

# Or Python directly
python main.py --env test
python main.py --env prod
python main.py --mode baseline --env test
python main.py --list-modes
python main.py --compare
python main.py --config path/to/custom.yaml
```

## Architecture

- **Entry Point**: `main.py` with `main()` + environment-specific entry functions (`main_prod()`, `main_test()`)
- **Runtime Modes**: 10 progressive modes (baseline → repairs → heuristics → full → RL-guided → round-robin → specialists → archive → hierarchical → multiagent) via `--mode` flag
- **Experiment Management**: `src/workflows/experiment_manager.py` tracks runs in `manifest.json` with `ExperimentManager` class
- **Workflow**: `src/workflows/standard_run.py` orchestrates: load → validate → feasibility → GA → decode → report
- **GA Core**: `src/core/ga_scheduler.py` - GAScheduler class with DEAP toolbox, population init, evolution
- **Multiprocessing**: Enabled via `parallel.use_multiprocessing` in YAML
- **Advanced RL**: 8 enhancements (constraint-specific state, multi-objective rewards, adaptive probabilities, specialist agents, archive diversity, memetic RL, hierarchical RL, rank-based multi-agent)

## Active Workstream (January 2025)

1. **Phase 3 Implementation**: ✅ Complete (8 advanced RL/GA enhancements, see `docs/06-development/implementation-notes/PHASE_3_ADVANCED_RL.md`)
2. **GPU Acceleration**: ✅ Deployed (CUDA enabled, documentation complete in `docs/04-algorithms/nvidia-gpu/`)
3. **Documentation Reorganization**: ✅ Complete (10-category structure, see `docs/INDEX.md`)
4. **Next Steps (Testing & Training):**
   - Install pymoo: `uv add pymoo`
   - Test constraint evaluator with existing data
   - Train specialist agents (4 agents, 100K steps each)
   - Train hierarchical policies (1 high-level + 5 low-level)
   - Train rank-based agents (4 agents, 100K steps each)
   - Benchmark all 10 runtime modes
   - Document empirical results in `docs/06-development/implementation-notes/PHASE_3_RESULTS.md`

Always log notable runs in `output/` and reference them inside documentation or onboarding guides.

## Key Components

- **Chromosomes**: `list[SessionGene]`, fitness `(-hard, -soft)` with weights `(-1.0, -0.01)`
- **Population Strategies**: hybrid (25% greedy, 50% smart, 25% random) / smart / random
- **Operators**: `crossover_course_group_aware()`, `mutate_individual()`
- **Repair**: IGLS system with exhaustive search, stagnation repair, selective repair
- **Constraints**: Hard (must-satisfy) and soft (prefer-satisfy) in `src/constraints/`
- **Time System**: `QuantumTimeSystem` converts wall-clock ↔ discrete quanta (default 60 min)
- **Validation**: Input validation + feasibility checking before GA
- **Exports**: JSON, PDF calendar, plots to `output/evaluation_<timestamp>/`

## Key References for Agents

- `docs/INDEX.md` – master navigation for all documentation (start here!).
- `docs/02-user-guides/runtime-modes.md` – comprehensive guide to 10 runtime modes and experiment management.
- `docs/06-development/implementation-notes/PHASE_3_ADVANCED_RL.md` – complete summary of 8 advanced RL/GA enhancements (constraint-specific state, multi-objective rewards, adaptive probabilities, specialist agents, archive diversity, memetic RL, hierarchical RL, rank-based multi-agent).
- `docs/QUICKREF_RUNTIME_MODES.md` – quick reference for runtime mode CLI usage.
- `RUNTIME_MODES_SUMMARY.md` – complete overview of runtime mode architecture and implementation.
- `docs/06-development/implementation-notes/PHASE_2_RL_COMPLETE.md` – authoritative summary of RL implementation (files, tasks, next steps).
- `docs/06-development/implementation-notes/PHASE_1.5_SUMMARY.md` & `PHASE_2.1_SUMMARY.md` – prior phase retrospectives.
- `docs/04-algorithms/nvidia-gpu/` – GPU acceleration guides and deployment documentation.
- `Todo.md` – master backlog (Phase 2+ and optional Phase 3).
- `docs/08-qna/technical-questions.md` – active Q&A workspace for technical discussions.
- `.github/instructions/*.instructions.md` – path-specific rules (config, GA, RL, constraints, etc.).

## 📋 Coding Standards & Best Practices

### Python Style Guidelines
- **PEP 8 Compliant**: Line length 88 (Black default)
- **Import Order**: Standard lib → third-party → local (sorted alphabetically)
- **Type Hints**: Required for function signatures, use `from __future__ import annotations`
- **Error Handling**: Raise informative exceptions with context
- **Logging**: 
  - Use `from src.utils.console_service import get_console` for user-facing output (Rich)
  - Use `logging.getLogger(__name__)` for debugging
- **Docstrings**: Google-style docstrings for all public APIs
  - **NO separate .md files for code documentation** - docstrings only!
- **Config Access**: 
  - Always use `from src.config import get_config; config = get_config()`
  - Never import deprecated `config.ga_params`

### Code Organization
```python
# Standard lib imports
import logging
from pathlib import Path
from typing import List, Dict, Optional

# Third-party imports
import numpy as np
from deap import base, tools
from rich.console import Console

# Local imports
from src.config import get_config
from src.entities.course import Course
from src.ga.sessiongene import SessionGene
```

### Naming Conventions
- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private**: Prefix with `_` (e.g., `_internal_helper()`)

### Error Handling Pattern
```python
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    console.print(f"[red]Error:[/red] {e}")
    raise
```

### Required Before Each Commit
- Run `black src/ test/` - Auto-format code
- Run `ruff check src/ test/` - Lint for issues
- Run `pytest test/unit/` - Ensure tests pass
- Verify config syntax if changed: `uv run verify-config`

## Documentation Policy

### 1. Code Documentation
**Use Python docstrings only** - no separate .md files for code.

### 2. Minor Changes → `docs/06-development/changelog/`
Bugfixes, small enhancements, refactoring:
- Add timestamped entry to `docs/06-development/changelog/{bugfixes,enhancements}.md`
- Format: `## [YYYY-MM-DD] Brief description` + file list
- No detailed explanations needed

### 3. Bug Fixes (Code-Related) → `docs/06-development/bugfixes/`
**NEW**: Detailed bug fix documentation with code examples:
- Create separate file: `docs/06-development/bugfixes/issue-name.md`
- Include: Root cause, solution, verification tests, impact
- Full code snippets and technical details
- Example: `hypervolume-calculation-fix.md`

### 4. Major Changes → `docs/06-development/implementation-notes/`
Significant implementations, phase completions:
- Create new file in `docs/06-development/implementation-notes/`
- Comprehensive summary with status, files, examples
- Structure: Overview → Tasks → Files → Usage → Next Steps

### 4a. Experimental Features → Modular Configs + Killswitches
When adding major new features requiring experimentation:
- **Create modular config folder**: `configs/{category}/{mode-name}.yaml`
- **Add RuntimeMode enum entry**: Update `src/config/runtime_mode.py`
- **Implement killswitches**: Master switch in `base.yaml` (e.g., `rl.enabled`, `repair.enabled`)
- **Document mode in user guide**: Add to `docs/02-user-guides/runtime-modes.md`
- **Add UV shortcut**: Register in `pyproject.toml` `[project.scripts]`
- **Use ExperimentManager**: Track runs via `src/workflows/experiment_manager.py`
- **Killswitch validation**: Automatic via `RuntimeMode.validate_config()`
- **Example**: See RL integration (configs/rl/5-rl-guided.yaml, rl.enabled killswitch)

### 5. Thesis Content → `docs/07-thesis-report/`
Academic documentation, publication-ready content:
- Thesis-ready prose with placement comment
- Structure: Problem → Solution → Implementation → Results

### 6. Questions & Discussions → `docs/08-qna/`
**Non-code technical Q&A only** - architecture, algorithms, design decisions:
- **Use `docs/08-qna/technical-questions.md`** for high-level questions:
  - Architecture discussions (metaheuristics vs hyperheuristics)
  - Algorithm concepts (NSGA-II, RL integration)
  - Design decisions (why certain approaches were chosen)
  - Conceptual clarifications (fitness weights, constraint priorities)
- **DO NOT document code bugs/fixes here** - use `docs/06-development/bugfixes/` instead
- Track sessions chronologically with timestamps
- Include context, questions, answers, and decisions
- Keep answers conceptual, not implementation-focused

### 7. Algorithm & Performance Documentation → `docs/04-algorithms/`
Detailed algorithm descriptions, mathematical formulations, and performance analysis:
- NSGA-II, repair heuristics, RL algorithms
- Mathematical notation and pseudocode
- Performance characteristics and complexity analysis
- GPU acceleration guides (nvidia-gpu/)
- Time complexity analysis and optimization strategies
- Benchmark guides and profiling results

### 8. Architecture Documentation → `docs/03-architecture/`
System design, component interactions, data flow:
- High-level architecture diagrams
- Component relationships
- Design patterns and principles

## 📝 Commit Message Format

**Format**: `<type>(<scope>): <summary>`

**Types**:
- `feat`: New feature (e.g., `feat(rl): add hierarchical RL policy`)
- `fix`: Bug fix (e.g., `fix(ga): correct fitness weight calculation`)
- `refactor`: Code restructure (e.g., `refactor(config): simplify loader`)
- `perf`: Performance improvement (e.g., `perf(gpu): add batch evaluator`)
- `test`: Add/update tests (e.g., `test(constraints): add room capacity tests`)
- `docs`: Documentation only (e.g., `docs(readme): update quickstart`)
- `style`: Code style/formatting (e.g., `style(ga): apply black formatting`)
- `chore`: Maintenance (e.g., `chore(deps): update deap to 1.4.1`)

**Guidelines**:
- Use imperative mood ("add" not "added")
- Keep summary under 72 characters
- Include scope (module/file affected)
- Add body for complex changes (after blank line)

**Examples**:
```
feat(gpu): integrate GPU batch evaluator for 10-50x speedup

- Add GPUConstraintEvaluator class in ga/evaluator/
- Update ga_scheduler to use GPU for batches >50
- Fallback to CPU for small batches or GPU unavailable
```

```
fix(constraints): correct instructor exclusivity calculation

Room exclusivity was counting same-room conflicts incorrectly.
Fixed by checking quantum overlap properly.

Fixes #123
```

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
- `rl.instructions.md` - RL environment, training, deployment, and promotion workflow
