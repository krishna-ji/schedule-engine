# Schedule Engine - Copilot Agent Guide

##  Project Overview

**Type**: Constraint-satisfaction problem (CSP) solver for educational timetabling via multi-objective evolutionary algorithms  
**Domain**: University course scheduling (NP-hard combinatorial optimization)  
**Language**: Python 3.12 (pinned via `requires-python = "==3.12.*"`)  
**Metaheuristic**: NSGA-II (Non-dominated Sorting Genetic Algorithm II) with PPO/DQN reinforcement learning hyper-heuristic layer  
**Dependencies**: DEAP 1.4.1 (NSGA-II toolbox), PyTorch 2.4.1 + CUDA 12.1 (RL training/inference only), Stable-Baselines3 2.3.2 (PPO/DQN agents), Pydantic 2.10.3 (config validation), Rich 13.9.4 (TUI)  
**Package Manager**: UV (modern, fast Python package installer)  
**Architecture**: Modular constraint-based optimization with 10 progressive runtime modes  
**Performance**: CPU multiprocessing for GA (32 parallel workers), GPU reserved for RL neural networks

## ️ Build & Test Commands

### Quick Setup
```bash
# Install dependencies (UV package manager)
uv sync --frozen

# Verify setup
uv run diagnose
```

### Unified CLI Launcher (Recommended)
```bash
# Progressive Mode Experiments (A→E: Increasing Complexity)
uv run baseline --test      # Mode A: Pure NSGA-II (~2-5 min)
uv run memetic --test       # Mode B: + Memetic local search (~3-7 min)
uv run roundrobin --test    # Mode C: + Round-robin heuristics (~5-10 min)
uv run adaptive --test      # Mode D: + Adaptive selection (~7-15 min)
uv run rl --test            # Mode E: + RL-guided (requires trained model, ~10-20 min)

# RL Training
uv run train-rl --test      # Smoke test (10K steps, ~5-10 min)
uv run train-rl --prod      # Production (100K steps, ~1-2 hours)

# Helper Commands
uv run diagnose             # Check GPU/system status
uv run clean                # Clean output directory
uv run list-experiments     # Show experiment history
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

### Helper Commands
```bash
# Diagnose GPU/system status
uv run diagnose

# Clean old outputs
uv run clean

# List experiment history
uv run list-experiments

# Legacy commands (still supported)
uv run verify-config
uv run check-data
```

## Phase Roadmap & Status

- **Phase 1.5 – Heuristic Toolbox**:  Complete (19 operators + registry, documented in `docs/06-development/implementation-notes/PHASE_1.5_SUMMARY.md`).
- **Phase 2.1 – Gymnasium Environment**:  Complete (env, reward, action mapper). See `docs/06-development/implementation-notes/PHASE_2.1_SUMMARY.md`.
- **Phase 2.2-2.4 – RL Training/Deployment/Integration**:  Code complete (`docs/06-development/implementation-notes/PHASE_2_RL_COMPLETE.md`). Pending execution tasks: curriculum training runs, checkpoint selection, promotion, RL-enabled GA benchmarking, and documentation updates.
- **Phase 3 – Advanced RL / Evaluation**:  Planned (multi-agent, transfer learning, evaluation suite) per `Todo.md` and `docs/10-ai-suggestions/rlphase2.2-2.4_guide_manual.md`.
- **GPU Acceleration**:  Removed from GA loop (CPU multiprocessing only). GPU reserved for RL training/inference where it excels (neural networks).

## Tech Stack

- **Language**: Python 3.12 (pinned via `requires-python = "==3.12.*"`)
- **Type Safety**: mypy 1.13.0 strict mode (100% coverage for all pure Python packages)
- **Core Libraries**: 
  - DEAP 1.4.1 (genetic algorithms)
  - PyTorch 2.4.1 + CUDA 12.1 (RL training/inference only - NOT used for GA fitness evaluation)
  - Stable-Baselines3 2.3.2 (RL agents)
  - Pydantic 2.10.3 (validation)
  - Rich 13.9.4 (terminal UI)
  - NumPy 1.26.4 (scientific computing)
  - Gymnasium 0.29.1 (RL environment)
  - pymoo 0.6.1.3 (optimized multi-objective metrics)
- **Config**: YAML-based with base.yaml + environment overrides + runtime modes
- **Package Manager**: UV (uv.lock, pyproject.toml)
- **Performance**: CPU multiprocessing (32 cores), parallel operators, concurrent validation, pymoo-accelerated metrics

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
- `configs/baseline/a-pure-nsga.yaml` - Mode A: Pure NSGA-II (all killswitches OFF)
- `configs/nsga/b-nsga-memetic.yaml` - Mode B: NSGA-II + memetic local search
- `configs/hybrid/c-roundrobin.yaml` - Mode C: NSGA-II + round-robin heuristics
- `configs/hybrid/d-adaptive.yaml` - Mode D: NSGA-II + adaptive heuristic selection
- `configs/rl/e-rl-guided.yaml` - Mode E: RL-guided control (full deployment)

Environment configs inherit from base.yaml via deep merge in `src/config/loader.py`.
Runtime mode configs support automatic killswitch validation.

**Access config:** `from src.config import get_config; config = get_config()`

## Running the Engine

### Unified CLI (Recommended)
```bash
# Progressive Mode Experiments (A→E: Increasing Complexity)
uv run baseline --test      # Mode A: Pure NSGA-II (~2-5 min)
uv run memetic --test       # Mode B: + Memetic local search (~3-7 min)
uv run roundrobin --test    # Mode C: + Round-robin heuristics (~5-10 min)
uv run adaptive --test      # Mode D: + Adaptive selection (~7-15 min)
uv run rl --test            # Mode E: + RL-guided (requires trained model, ~10-20 min)

# RL Training
uv run train-rl --test      # RL agent training smoke test (~5-10 min)
uv run train-rl --prod      # RL agent training production (~1-2 hours)

# Helper Commands
uv run diagnose             # Check GPU/system/config
uv run clean                # Remove old outputs
uv run list-experiments     # Show experiment history
uv run stats                # Show manifest statistics
uv run archive              # Archive incomplete runs

# Production runs with custom names
uv run baseline --prod --name "thesis-baseline-r01"
uv run memetic --prod --name "thesis-memetic-r01"
```

### Profile Hierarchy (DRY Principle)
```
base.yaml (common settings)
  ↓
test.yaml (30 gens, 10 pop) - smoke test
  ↓
prod.yaml (2000 gens, 500 pop) - full production
```

## CLI Convention (November 2025)

**Philosophy**: Clean, unified CLI with profile-based experiments.

**Command Structure**:
- **Main Launcher**: `nsga` (unified NSGA-II launcher)
- **Progressive Modes (A→E)**: Systematic ablation study
  - Mode A: `baseline` - Pure NSGA-II
  - Mode B: `memetic` - + Memetic local search
  - Mode C: `roundrobin` - + Round-robin heuristics
  - Mode D: `adaptive` - + Adaptive selection
  - Mode E: `rl` - + RL-guided (full deployment)
- **RL Training**: `train-rl`
- **Helper Commands (a-z)**: Utilities (`diagnose`, `clean`, `list-experiments`, `stats`, `archive`)
- **Profiles**: `--test` (smoke), `--prod` (full)

**Key Files**:
- `scripts/launcher.py` - Unified CLI launcher with profile routing
- `pyproject.toml` - Script definitions in `[project.scripts]`

**Quick Examples**:
```bash
# Progressive mode experiments (local development)
uv run baseline --test   # 2-5 min
uv run memetic --test    # 3-7 min
uv run roundrobin --test # 5-10 min

# Production runs (VM deployment)
uv run baseline --prod --name "thesis-baseline-r01"  # 1-3 hours
uv run memetic --prod --name "thesis-memetic-r01"    # 2-4 hours
```

**DRY Principle**: Configs inherit hierarchically (base → test → prod).

## Architecture

- **Entry Point**: `scripts/launcher.py` (unified CLI) → `main.py` (GA) or `src/rl/training/train_script.py` (RL)
- **Runtime Modes**: 5 progressive modes (A→E: increasing complexity) via alphabetic commands
  - Mode A: `baseline` - Pure NSGA-II (configs/baseline/a-pure-nsga.yaml)
  - Mode B: `memetic` - + Memetic local search (configs/nsga/b-nsga-memetic.yaml)
  - Mode C: `roundrobin` - + Round-robin heuristics (configs/hybrid/c-roundrobin.yaml)
  - Mode D: `adaptive` - + Adaptive selection (configs/hybrid/d-adaptive.yaml)
  - Mode E: `rl` - + RL-guided control (configs/rl/e-rl-guided.yaml)
- **Experiment Management**: `src/workflows/experiment_manager.py` tracks runs in `manifest.json` with `ExperimentManager` class
- **Workflow**: `src/workflows/standard_run.py` orchestrates: load → validate → feasibility → GA → decode → report
- **GA Core**: `src/core/ga_scheduler.py` - GAScheduler class with DEAP toolbox, population init, evolution
- **Parallel Processing**: 
  - CPU multiprocessing for fitness evaluation (`parallel.use_multiprocessing`)
  - Parallel crossover/mutation operators (3-5x speedup)
  - Concurrent feasibility checks (3-5x speedup)
- **Metrics Optimization**: pymoo-accelerated multi-objective metrics (hypervolume, IGD, GD) - 139x speedup (50s → 0.36s per generation)
- **Advanced RL**: 8 enhancements (constraint-specific state, multi-objective rewards, adaptive probabilities, specialist agents, archive diversity, memetic RL, hierarchical RL, rank-based multi-agent)

## Active Workstream (November 2025)

1. **Phase 3 Implementation**:  Complete (8 advanced RL/GA enhancements, see `docs/06-development/implementation-notes/PHASE_3_ADVANCED_RL.md`)
2. **GPU Acceleration**:  Removed from GA loop (CPU multiprocessing only)
   - GPU not beneficial for timetabling constraints (complex Python logic, small problem size)
   - CPU multiprocessing provides better parallelization (32 cores)
   - GPU reserved for RL neural network training/inference (where it excels)
   - **Removed files**: GPU evaluator integration from `ga_scheduler.py`
3. **Metrics Optimization**:  Complete (pymoo integration)
   - Hypervolume: 139x faster using WFG algorithm (Cython backend)
   - IGD/GD: Vectorized implementations
   - Configurable frequency: `metrics.advanced_metrics_frequency`
   - **Impact**: 27.8 hours → 12 minutes for 2000 generations
4. **Type Safety**:  Complete (comprehensive strict mypy typing)
   - **All pure Python packages**: 100% typed (diversity/, lns/, heuristics/, workflows/, utils/, config/, entities/, encoder/, decoder/, constraints/, metrics/, exporter/, validation/)
   - **147 errors fixed**: Systematic fixes across workflows (53), heuristics (49), lns (28), diversity (17)
   - **56+ files**: Already passing strict mypy from previous work
   - **33 type: ignore**: Only legitimate library limitations (yaml, numpy, DEAP, RL frameworks)
   - **Achievement**: All custom-written pure Python code fully typed with mypy 1.13.0 strict mode
5. **Thesis Experiments**:  Ready (5 progressive experiments)
   - Mode A: Pure NSGA-II baseline (`uv run baseline`)
   - Mode B: + Memetic local search (`uv run memetic`)
   - Mode C: + Round-robin heuristics (`uv run roundrobin`)
   - Mode D: + Adaptive selection (`uv run adaptive`)
   - Mode E: + RL-guided control (`uv run rl`)
   - **Guide**: `docs/45-resource-unused-problem/THESIS_EXPERIMENTS_GUIDE.md`
6. **Documentation Reorganization**:  Complete (10-category structure, see `docs/INDEX.md`)
7. **Next Steps (Execution):**
   - Run all 5 thesis experiments (6-10 hours total)
   - Analyze results and generate comparison plots
   - Train RL agents if needed (`uv run train prod`)
   - Document empirical results in thesis report

Always log notable runs in `output/` and reference them inside documentation or onboarding guides.

## Key Components

- **Chromosome Encoding**: Direct representation `list[SessionGene]` where each gene = (course_id, group_ids, instructor_id, room_id, quanta_slots)
- **Fitness Function**: Lexicographic multi-objective `(-hard_violations, -soft_penalty)` with weights `(-1.0, -0.01)` for NSGA-II Pareto dominance
- **Population Initialization**: Hybrid strategy (25% greedy heuristic, 50% constraint-guided smart, 25% random uniform)
- **Genetic Operators**: Course-group-aware crossover (preserves enrollment relationships), constraint-guided mutation (biased toward feasible regions)
- **Repair Heuristics**: IGLS (Iterative Greedy Local Search) with 19 registered operators (construction/perturbation/improvement categories)
- **Constraint Taxonomy**: 12 hard constraints (instructor conflicts, room capacity, qualification mismatches) + 8 soft constraints (schedule gaps, clustering, preferred times)
- **Temporal Discretization**: `QuantumTimeSystem` converts continuous time → integer quanta (default 60-minute slots, 7 days × 10 hours = 70 total quanta)
- **Pre-GA Validation Pipeline**: JSON schema validation → entity relationship checks → pigeonhole feasibility analysis (instructor workload, room capacity, group availability)
- **Output Artifacts**: JSON (decoded schedule), PDF (visual calendar via ReportLab), PNG (Pareto front, constraint trends, diversity metrics)

## Key References for Agents

- **`.github/instructions/cli.instructions.md`** – **START HERE**: Quick reference for unified CLI launcher system with progressive modes (A→E).
- **`.github/instructions/README.md`** – Overview of path-specific AI agent instructions and experimentation best practices.
- `docs/INDEX.md` – master navigation for all documentation.
- `PHASE_3_COMPLETION_SUMMARY.md` – **LATEST**: Complete Phase 3 implementation (27 files, 8 enhancements, GPU acceleration).
- `docs/45-resource-unused-problem/THESIS_EXPERIMENTS_GUIDE.md` – **THESIS**: 5 progressive experiments (A→E) with commands and expected results.
- `docs/02-user-guides/runtime-modes.md` – comprehensive guide to runtime modes and experiment management.
- `scripts/launcher.py` – unified CLI launcher implementation with profile support.
- `docs/04-algorithms/nvidia-gpu/` – GPU acceleration guides and deployment documentation.
- `src/ga/evaluator/gpu_batch_evaluator.py` – GPU batch evaluator implementation (10-50x speedup).
- `Todo.md` – master backlog (currently focused on RL training and benchmarking).
- `docs/08-qna/technical-questions.md` – active Q&A workspace for technical discussions.
- `.github/instructions/*.instructions.md` – path-specific rules (config, GA, RL, constraints, tests, etc.).

##  Coding Standards & Best Practices

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
- Run `mypy src/` - Type check (strict mode, must pass)
- Run `pytest test/unit/` - Ensure tests pass
- Verify config syntax if changed: `uv run verify-config`

### Type Checking Guidelines
- **Strict mode required**: All new code must pass `mypy --strict`
- **Pure Python packages**: 100% type coverage (no type: ignore except for library limitations)
- **Legitimate type: ignore cases**:
  - yaml module (no type stubs available)
  - numpy return types (floating[Any] → float, ndarray → list conversions)
  - DEAP Individual type (uses Any internally)
  - RL framework optional components
  - Circular import forward references
  - Dynamic attribute assignment on function objects
  - Private library attributes (e.g., Pool._processes)
- **Type annotation requirements**:
  - All function parameters and return types
  - All class attributes and instance variables
  - Complex data structures (dict, list with specific element types)
  - Optional parameters as `T | None = None` (PEP 604 union syntax)
- **Common patterns**:
  - Use `from __future__ import annotations` for forward references
  - Wrap numpy operations: `float(np.mean(...))`, `int(np.argmax(...))`
  - Assert after None checks: `assert x is not None` (helps type checker)
  - Import proper types: Use `SessionGene` not `CourseSession` in appropriate contexts

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

##  Commit Message Format

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

- `cli.instructions.md` - CLI launcher system & command conventions
- `config.instructions.md` - Configuration system
- `ga-core.instructions.md` - GA operators & scheduler
- `constraints.instructions.md` - Constraint functions
- `data-flow.instructions.md` - Encoder/decoder/entities
- `validation.instructions.md` - Input & feasibility validation
- `export.instructions.md` - Report generation & plotting
- `workflows.instructions.md` - Orchestration logic
- `tests.instructions.md` - Testing guidelines
- `rl.instructions.md` - RL environment, training, deployment, and promotion workflow
