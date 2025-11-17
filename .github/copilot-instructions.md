# Schedule Engine - Agent Guide

## Project Summary

University course scheduling engine using NSGA-II genetic algorithm with constraint-based optimization. Written in Python with DEAP, rich terminal UI, and YAML configuration.

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

## Active Workstream (November 2025)

1. **GPU Acceleration**: ✅ Deployed (CUDA enabled, documentation complete in `docs/04-algorithms/nvidia-gpu/`)
2. **Documentation Reorganization**: ✅ Complete (10-category structure, see `docs/INDEX.md`)
3. **Run PPO curriculum training (100K–300K steps)** with GPU acceleration and capture TensorBoard logs.
4. **Generate/refresh validation sets** (`scripts/generate_validation_set.py`).
5. **Select & promote best checkpoint** using `scripts/select_best_checkpoint.py` + `scripts/promote_model_to_prod.py`.
6. **Enable RL in configs/prod.yaml**, run `uv run prod`, and compare RL vs non-RL GA baselines.
7. **Update docs** (especially `docs/06-development/implementation-notes/PHASE_2_RL_COMPLETE.md`) with empirical results once runs finish.

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
- `docs/06-development/implementation-notes/PHASE_2_RL_COMPLETE.md` – authoritative summary of RL implementation (files, tasks, next steps).
- `docs/06-development/implementation-notes/PHASE_1.5_SUMMARY.md` & `PHASE_2.1_SUMMARY.md` – prior phase retrospectives.
- `docs/04-algorithms/nvidia-gpu/` – GPU acceleration guides and deployment documentation.
- `Todo.md` – master backlog (Phase 2+ and optional Phase 3).
- `docs/08-qna/technical-questions.md` – active Q&A workspace for technical discussions.
- `.github/instructions/*.instructions.md` – path-specific rules (config, GA, RL, constraints, etc.).

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
- `rl.instructions.md` - RL environment, training, deployment, and promotion workflow
