# Schedule Engine - AI Coding Agent Guide

## Project Overview

**Type**: CSP solver for educational timetabling via multi-objective evolutionary algorithms  
**Domain**: University course scheduling (NP-hard combinatorial optimization)  
**Language**: Python 3.12 (strict mypy typing enforced)  
**Metaheuristic**: NSGA-II + RL hyper-heuristic (PPO/DQN agents)  
**Stack**: DEAP 1.4.1, PyTorch 2.4.1+CUDA, Stable-Baselines3 2.3.2, Pydantic 2.10.3, Rich 13.9.4  
**Package Manager**: UV  
**Config System**: Python dataclasses (`configs/*.py`) with DRY inheritance (base → profiles → experiments)  
**Architecture**: 6 progressive experimental modes (A-F: baseline → RL-guided)  
**Performance**: CPU multiprocessing (32 workers), pymoo-accelerated metrics (139x speedup)

## Quick Start

```bash
# Install dependencies
uv sync --frozen

# Progressive experiments (A→F: increasing complexity)
uv run baseline --test       # A: Pure NSGA-II (~2-5 min)
uv run memetic --test        # B: + Memetic search (~3-7 min)
uv run roundrobin --test     # C: + Round-robin (~5-10 min)
uv run adaptive --test       # D: + Adaptive (~7-15 min)
uv run rl --test             # E: + RL-guided (~10-20 min)
uv run heuristic-testing --test  # F: Individual heuristic testing

# RL training
uv run train-rl --test       # Smoke test (10K steps, ~5 min)
uv run train-rl --prod       # Full training (100K steps, ~1-2 hrs)

# Utilities
uv run diagnose              # System/GPU diagnostics
uv run clean                 # Clean outputs
uv run list-experiments      # Show history
uv run stats                 # Manifest statistics
```

## Code Quality

```bash
black src/ test/             # Format
ruff check src/ test/        # Lint
mypy src/                    # Type check (strict mode)
pytest test/unit/            # Test suite
```

## Development Status

- **Phase 1.5**: ✅ Heuristic toolbox (19 operators + registry)
- **Phase 2.1**: ✅ Gymnasium environment (state/reward/actions)
- **Phase 2.2-2.4**: ✅ RL training pipeline (PPO/DQN agents)
- **Phase 3**: ✅ Advanced RL (8 enhancements: multi-agent, hierarchical, memetic)
- **Type Safety**: ✅ Strict mypy (100% typed pure Python packages)
- **GPU**: ❌ Removed from GA (CPU multiprocessing only); reserved for RL neural networks
- **Metrics**: ✅ pymoo acceleration (139x speedup: 50s → 0.36s/gen)

## Tech Stack

- **Python 3.12**: Strict mypy typing (`requires-python = "==3.12.*"`)
- **GA Core**: DEAP 1.4.1, NumPy 1.26.4, pymoo 0.6.1.3 (metrics)
- **RL Stack**: PyTorch 2.4.1+CUDA12.1, Stable-Baselines3 2.3.2, Gymnasium 0.29.1
- **Config**: Pydantic 2.10.3 (validation), Python dataclasses (DRY inheritance)
- **UI**: Rich 13.9.4 (terminal), matplotlib/seaborn (plots)
- **Performance**: CPU multiprocessing (32 cores), parallel operators, pymoo acceleration

## Repository Structure

```
schedule-engine/
├── main.py              # GA entry point (calls experiments)
├── configs/             # Python dataclass configs
│   ├── base.py          # BaseConfig (shared defaults)
│   ├── profiles.py      # TestConfig, ProdConfig (scaling)
│   ├── experiments/     # Experiment configs (A-F modes)
│   └── archive/         # Old YAML configs (deprecated)
├── src/
│   ├── config/          # Pydantic models + global accessor
│   ├── core/            # GAScheduler, DEAP toolbox
│   ├── ga/              # Operators, population, repair
│   ├── constraints/     # Hard/soft constraint functions
│   ├── encoder/         # JSON → entities + time system
│   ├── decoder/         # Individual → CourseSession
│   ├── entities/        # Domain models (Course, Instructor, etc.)
│   ├── exporter/        # PDF/JSON/plots
│   ├── validation/      # Input + feasibility checks
│   ├── workflows/       # Orchestration (standard_run, experiment_manager)
│   ├── heuristics/      # Repair operators (19 registered)
│   ├── rl/              # RL environment, agents, training
│   └── utils/           # Console, logging, helpers
├── scripts/             # CLI launcher + utilities
├── data/                # Input JSON (Courses, Instructors, Rooms, Groups)
├── test/                # pytest unit tests
└── docs/                # Documentation (INDEX.md for navigation)
```

## Configuration System

**Modular Python dataclass hierarchy** (DRY principle):
- `configs/base.py` → `BaseConfig` - All shared defaults (GA params, constraints, killswitches)
- `configs/profiles.py` → `TestConfig`/`ProdConfig` - Scaling overrides (ngen, pop_size)
- `configs/experiments/*.py` → Experiment-specific configs (A-F modes with killswitch states)

**Inheritance chain**: `BaseConfig → TestConfig/ProdConfig → ExperimentConfig`

**Config access**:
```python
from src.config import get_config
config = get_config()  # Returns Pydantic Config model
```

**Experiment registration** (in `main.py`):
```python
from configs import experiment_a, experiment_b  # Dataclass instances
EXPERIMENTS = {
    "a": ("Experiment A: Pure NSGA-II", experiment_a, experiment_a_baseline),
    "b": ("Experiment B: Memetic", experiment_b, experiment_b_memetic),
}
```

## Running the Engine

### Unified CLI (Recommended)
```bash
# Progressive Mode Experiments (A→F: Increasing Complexity)
uv run baseline --test      # A: Pure NSGA-II (~2-5 min)
uv run memetic --test       # B: + Memetic search (~3-7 min)
uv run roundrobin --test    # C: + Round-robin (~5-10 min)
uv run adaptive --test      # D: + Adaptive (~7-15 min)
uv run rl --test            # E: + RL-guided (~10-20 min)
uv run heuristic-testing --test  # F: Heuristic tests

# RL Training
uv run train-rl --test      # Smoke (10K steps, ~5-10 min)
uv run train-rl --prod      # Full (100K steps, ~1-2 hrs)

# Helper Commands
uv run diagnose             # System/GPU diagnostics
uv run clean                # Clean outputs
uv run list-experiments     # Experiment history
uv run stats                # Manifest statistics
uv run archive              # Archive incomplete runs

# Production runs (custom names)
uv run baseline --prod --name "thesis-baseline-r01"
uv run memetic --prod --name "thesis-memetic-r01"
```

### Profile Hierarchy (DRY Principle)
```
base.py (BaseConfig: shared defaults)
  ↓
profiles.py (TestConfig: 30 gens, 10 pop | ProdConfig: 2000 gens, 400 pop)
  ↓
experiments/*.py (Experiment-specific killswitches + overrides)
```

## CLI Convention (November 2025)

**Philosophy**: Clean, unified CLI with profile-based experiments.

**Command Structure**:
- **Main Launcher**: `nsga` (unified NSGA-II launcher)
- **Progressive Modes (A→F)**: Systematic ablation study
  - Mode A: `baseline` - Pure NSGA-II
  - Mode B: `memetic` - + Memetic local search
  - Mode C: `roundrobin` - + Round-robin heuristics
  - Mode D: `adaptive` - + Adaptive selection
  - Mode E: `rl` - + RL-guided (full deployment)
  - Mode F: `heuristic-testing` - Individual heuristic tests
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

**DRY Principle**: Configs inherit hierarchically (base → test/prod → experiments).

## Architecture

- **Entry Point**: `scripts/launcher.py` (unified CLI) → `main.py` (GA) or `src/rl/training/train_script.py` (RL)
- **Runtime Modes**: 6 progressive modes (A→F: increasing complexity) via alphabetic commands
  - Mode A: `baseline` - Pure NSGA-II (configs/experiments/baseline.py)
  - Mode B: `memetic` - + Memetic local search (configs/experiments/memetic.py)
  - Mode C: `roundrobin` - + Round-robin heuristics (configs/experiments/roundrobin.py)
  - Mode D: `adaptive` - + Adaptive selection (configs/experiments/adaptive.py)
  - Mode E: `rl` - + RL-guided control (configs/experiments/rl_guided.py)
  - Mode F: `heuristic-testing` - Individual heuristic tests (configs/experiments/heuristic_testing.py)
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

1. **Phase 3 Implementation**: ✅ Complete (8 advanced RL/GA enhancements, see `docs/06-development/implementation-notes/PHASE_3_ADVANCED_RL.md`)
2. **GPU Acceleration**: ❌ Removed from GA loop (CPU multiprocessing only)
   - GPU not beneficial for timetabling constraints (complex Python logic, small problem size)
   - CPU multiprocessing provides better parallelization (32 cores)
   - GPU reserved for RL neural network training/inference (where it excels)
   - **Removed files**: GPU evaluator integration from `ga_scheduler.py`
3. **Metrics Optimization**: ✅ Complete (pymoo integration)
   - Hypervolume: 139x faster using WFG algorithm (Cython backend)
   - IGD/GD: Vectorized implementations
   - Configurable frequency: `metrics.advanced_metrics_frequency`
   - **Impact**: 27.8 hours → 12 minutes for 2000 generations
4. **Type Safety**: ✅ Complete (comprehensive strict mypy typing)
   - **All pure Python packages**: 100% typed (diversity/, lns/, heuristics/, workflows/, utils/, config/, entities/, encoder/, decoder/, constraints/, metrics/, exporter/, validation/)
   - **147 errors fixed**: Systematic fixes across workflows (53), heuristics (49), lns (28), diversity (17)
   - **56+ files**: Already passing strict mypy from previous work
   - **33 type: ignore**: Only legitimate library limitations (yaml, numpy, DEAP, RL frameworks)
   - **Achievement**: All custom-written pure Python code fully typed with mypy 1.13.0 strict mode
5. **Thesis Experiments**: ✅ Ready (5 progressive experiments)
   - Mode A: Pure NSGA-II baseline (`uv run baseline`)
   - Mode B: + Memetic local search (`uv run memetic`)
   - Mode C: + Round-robin heuristics (`uv run roundrobin`)
   - Mode D: + Adaptive selection (`uv run adaptive`)
   - Mode E: + RL-guided control (`uv run rl`)
   - **Guide**: `docs/45-resource-unused-problem/THESIS_EXPERIMENTS_GUIDE.md`
6. **Documentation Reorganization**: ✅ Complete (10-category structure, see `docs/INDEX.md`)
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

### ⚠️ CRITICAL: Strict Type Safety Requirement
**ALL code generated MUST be strictly typed from the start:**
- ✅ Every function parameter must have a type annotation
- ✅ Every function must have a return type annotation (including `-> None`)
- ✅ All class attributes must be typed
- ✅ All instance variables must be typed
- ✅ Complex structures (list, dict) must specify element types: `list[str]`, `dict[str, int]`
- ✅ Use PEP 604 union syntax: `str | None` instead of `Optional[str]`
- ✅ Add `from __future__ import annotations` at the top of every file
- ⛔ NEVER generate untyped code - even for quick fixes or small changes
- ⛔ Code without proper types will fail mypy --strict and will be rejected

**This is a non-negotiable requirement.** All code must pass `mypy --strict` before commit.

### Python Style Guidelines
- **PEP 8 Compliant**: Line length 88 (Black default)
- **Import Order**: Standard lib → third-party → local (sorted alphabetically)
- **Type Hints**: **MANDATORY** for all function signatures, use `from __future__ import annotations`
- **Error Handling**: Raise informative exceptions with context
- **Logging**:
  - Use `from src.utils.console_service import get_console` for user-facing output (Rich)
  - Use `logging.getLogger(__name__)` for debugging
- **Docstrings**: Google-style docstrings for all public APIs
  - **NO separate .md files for code documentation** - docstrings only!
- **Config Access**:
  - Always use `from src.config import get_config; config = get_config()`
  - Never import deprecated `config.ga_params`

### Code Organization & Type Annotations Template
**Every new file must follow this template with strict typing:**
```python
"""Module docstring describing the file's purpose.

This module should have a comprehensive docstring.
"""

from __future__ import annotations  # MANDATORY - enables forward references

# Standard lib imports
import logging
from pathlib import Path

# Third-party imports
import numpy as np
from deap import base, tools
from rich.console import Console

# Local imports
from src.config import get_config
from src.entities.course import Course
from src.ga.sessiongene import SessionGene

# Logger
logger = logging.getLogger(__name__)


def example_function(
    param1: str,
    param2: int,
    param3: list[Course],
    param4: dict[str, int] | None = None,
) -> tuple[bool, str]:
    """Example of properly typed function.

    Args:
        param1: Description of param1
        param2: Description of param2
        param3: List of Course objects
        param4: Optional dictionary mapping strings to integers

    Returns:
        Tuple of (success status, message)
    """
    # Implementation
    return True, "Success"


class ExampleClass:
    """Example of properly typed class."""

    # Class attributes with types
    class_attr: str = "default"

    def __init__(
        self,
        name: str,
        count: int,
        items: list[str] | None = None,
    ) -> None:
        """Initialize with typed parameters.

        Args:
            name: Name of the instance
            count: Count value
            items: Optional list of items
        """
        # Instance variables with types
        self.name: str = name
        self.count: int = count
        self.items: list[str] = items if items is not None else []

    def process(self, data: dict[str, int]) -> bool:
        """Process data and return success status.

        Args:
            data: Dictionary to process

        Returns:
            True if successful, False otherwise
        """
        # Implementation
        return True
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
- **Create modular config**: `configs/experiments/{feature}.py` with dataclass
- **Implement killswitches**: Master switch in `BaseConfig` (e.g., `rl_enabled`, `repair_enabled`)
- **Inherit from profiles**: Use `TestConfig`/`ProdConfig` for scaling
- **Register experiment**: Add to `main.py` EXPERIMENTS dict
- **Add CLI shortcut**: Register in `pyproject.toml` `[project.scripts]`
- **Experiment tracking**: Uses `ExperimentManager` via `src/workflows/experiment_manager.py`
- **Example**: See RL integration (configs/experiments/rl_guided.py, rl_enabled killswitch)

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
