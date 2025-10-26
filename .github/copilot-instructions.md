# Schedule Engine - Repository Overview# Schedule Engine Agent Guide



## Project Summary## Commit Message Format

University course scheduling engine using NSGA-II genetic algorithm with constraint-based optimization. Written in Python with DEAP, rich terminal UI, and YAML configuration.When generating or suggesting commit messages for this repository:

- Follow this format: `<type>(<scope>): <summary>`

## Tech Stack- Use imperative mood ("add", "fix", "update") and keep summary under 72 characters.

- **Language**: Python 3.8+- Choose types from: `feat`, `fix`, `data`, `method`, `analysis`, `result`, `doc`, `test`, `refactor`, `setup`, `release`.

- **Core Libraries**: DEAP (genetic algorithms), Pydantic (validation), Rich (terminal UI)- Include a short explanation of the change's purpose or scientific motivation if relevant.

- **Config**: YAML-based with Pydantic models- Reference experiment numbers, datasets, or manuscript sections when applicable.

- **Testing**: Manual testing with test configurations- Avoid generic summaries like "update" or "minor fix".



## Repository Structure## Project Guidelines

```- create all test files inside test/

schedule-engine/

├── main.py              # CLI entry point (--env test|dev|prod)### Architecture (Refactored Modular Structure)

├── config/              # Pydantic models & loaders- **Entry Point** `main.py` is now a lightweight CLI (< 100 lines) that parses `--env` or `--config` arguments, loads YAML configuration via `config/loader.py`, and delegates to `src/workflows/standard_run.py`

├── configs/             # YAML configuration files- **Configuration System** All settings (GA params, constraints, parallelization, repair, feasibility) are defined in YAML files under `configs/{test,dev,prod}.yaml`. Pydantic models in `config/models.py` provide type-safe validation. Access config anywhere via `from config import get_config`

├── src/- **Workflow Orchestration** `src/workflows/standard_run.py` contains `run_standard_workflow()` which orchestrates: data loading → validation → feasibility checks → GA execution → decoding → report generation. This replaces the old monolithic `main.py`

│   ├── core/            # GA scheduler & types- **GA Execution** `src/core/ga_scheduler.py` contains `GAScheduler` class that encapsulates DEAP toolbox setup, population initialization (hybrid/smart/random strategies), evolution loop, and metrics tracking. Configuration passed via `GAConfig` dataclass

│   ├── ga/              # GA operators, population, repair- **Multiprocessing Support** Parallel fitness evaluation enabled via `parallel.use_multiprocessing` in YAML. Worker processes initialized in `_worker_init()` with serialized context (Windows-safe spawn method)

│   ├── constraints/     # Hard & soft constraints

│   ├── encoder/         # JSON → entities + time system### Data Flow & Core Components

│   ├── decoder/         # Individual → CourseSession- **Input Loading** JSON files in `data/*.json` → `src/encoder/input_encoder.py` → entities (`src/entities/*`) + `QuantumTimeSystem` quanta → `SchedulingContext` dataclass

│   ├── entities/        # Domain models- **GA Chromosomes** Individuals are `list[SessionGene]` with `creator.FitnessMulti(weights=(-1.0, -0.01))`. Population initialization supports three strategies (configured via `ga.population_strategy`): "hybrid" (25% greedy, 50% smart, 25% random), "smart" (100% constraint-aware), "random" (baseline)

│   ├── exporter/        # PDF/JSON/plots generation- **Genetic Operators** Crossover: `crossover_course_group_aware()` preserves course-group relationships. Mutation: `mutate_individual()` requires `context` dict with `courses`, `groups`, `instructors`, `rooms`, `available_quanta`. Both operators can trigger repair heuristics if `repair.apply_after_crossover/mutation` is enabled

│   ├── validation/      # Input & feasibility checks- **Constraint Evaluation** Hard constraints in `src/constraints/hard.py`, soft in `src/constraints/soft.py`. Both return integer penalties. Constraints are enabled/weighted via YAML config under `hard_constraints` and `soft_constraints` sections. Evaluators in `src/ga/evaluator/{fitness,detailed_fitness}.py` aggregate constraint penalties

│   ├── workflows/       # Orchestration logic- **Repair System** Registry-based repair heuristics in `src/ga/operators/repair_registry.py`. Selective mode (configured via `repair.selective_mode`) uses violation detection (`src/ga/operators/violation_detector.py`) to target only problematic genes. Adaptive repair can trigger based on stagnation or periodic intervals

│   └── utils/           # Helpers & logging- **Feasibility Checks** Pre-GA validation in `src/validation/feasibility_checker.py` detects unsolvable problems (instructor bottlenecks, room capacity issues, pigeonhole violations). Configured via `feasibility` section in YAML. Can fail-fast or proceed with warnings

├── data/                # Input JSON files

├── test/                # Test files (ALL tests go here)### Time System

└── docs/                # Documentation (see below)- **Quantum Conversion** Always convert between wall-clock and quanta using `QuantumTimeSystem`. Operating quanta come from `get_all_operating_quanta()`. `SessionGene.quanta` must be unique and sorted to avoid duplicate-slot penalties

```- **Quantum Size** Configurable via `time.quantum_minutes` (default 60). Time preferences (`earliest_preferred_time`, `latest_preferred_time`, `midday_break_start/end`) also in YAML config



## Running the Engine### Exports & Reports

```bash- **Output Structure** Results saved to `output/evaluation_<timestamp>/`: `schedule.json`, `ScheduleCalendar.pdf`, `logger.txt`, `feasibility_report.txt`, `violation_report.txt`, plus plots under `plots/` subdirectory

python main.py --env test   # Fast (10 gens, 4 pop)- **Report Generation** Orchestrated by `src/workflows/reporting.py` which calls plot modules in `src/exporter/{plothard,plotsoft,plotdiversity,plotpareto,plot_detailed_constraints}.py`

python main.py --env dev    # Medium (100 gens, 20 pop)- **Calendar Display** Visual settings in `config/calendar_config.py` (unchanged). Color palette in `config/color_palette.py`

python main.py --env prod   # Full quality (200+ gens, 50+ pop)

python main.py --config path/to/custom.yaml### Running the Engine

```- **Commands** `python main.py --env test` (fast, 10 gens), `python main.py --env dev` (medium, 100 gens), `python main.py --env prod` (full quality, 200+ gens), or `python main.py --config path/to/custom.yaml`

- **Dependencies** Listed in `requirements.txt`: DEAP, Matplotlib, Pydantic, PyYAML, Rich (for terminal UI)

## Documentation System- **Testing Configs** Use `configs/test.yaml` for quick smoke tests (small population, few generations)



### Code Documentation### Validation & Warnings

- **Use Python docstrings only** for all modules, classes, functions- **Input Validation** `src/validation/input_validator.py` checks for missing references, invalid data. Enabled via `validate=True` in `run_standard_workflow()`

- **Never create separate .md files to document code**- **Feasibility Analysis** Run before GA (unless `feasibility.enable_checks=False`). Reports appear in console and saved to output directory

- **Missing Data Defaults** When availability is absent: groups default to operating quanta, instructors/rooms become fully available. Warnings printed during data loading

### Change Documentation (Bifurcated)- **Population Integrity** Optional validation via `ga.validate_population_integrity` (checks course-group pair alignment during crossover)

1. **Minor Changes** → `docs/code/{BUGFIX,ENHANCE,REFACTOR}.md`

   - Format: `## [YYYY-MM-DD] Brief description` + file list### Style & Conventions

   - For: bugfixes, minor refactoring, config changes- **Terminal UI** Rich library for colored console output, progress bars, panels. Use `console.print()` for formatted messages

   - **Sunday-First Ordering** Schedules display Sunday as first day of week (matches `QuantumTimeSystem.DAY_NAMES`)

2. **Major Changes** → `docs/for_report/new-file.md`- **Docstrings** All modules, classes, functions must have docstrings. No separate .md files for code documentation

   - Thesis-ready prose, begin with `<!-- Suggested thesis placement: ... -->`- **Config Access** Import `from config import get_config` (runtime) or `from config.models import Config` (type hints). Never import old `config.ga_params` or `config.constraints` modules (removed in refactor)

   - For: new algorithms, architectural changes, novel techniques

## Documentation Policy (Bifurcated System)

## Commit Message Format

Format: `<type>(<scope>): <summary>`### 1. Code Documentation: Docstrings Only

- **Types**: `feat`, `fix`, `refactor`, `test`, `doc`, `data`, `method`, `analysis`- **All code must be documented using Python docstrings** (functions, classes, modules).

- Use imperative mood, keep under 72 characters- Docstrings are the single source of truth for what code does.

- Example: `feat(repair): add selective violation detection`- **Never create separate .md files to document code**—code documents itself.



## General Coding Standards### 2. Minor Changes: Changelog in `docs/code/`

- **Python Style**: PEP 8 compliant- For **routine bugfixes, small enhancements, or refactoring** that don't alter core architecture:

- **Imports**: Standard lib → third-party → local (sorted alphabetically within groups)  - Add a **single timestamped entry** to the appropriate changelog:

- **Type Hints**: Use where beneficial for clarity    - `docs/code/BUGFIX.md` - Bug fixes

- **Error Handling**: Informative error messages with context    - `docs/code/ENHANCE.md` - Minor enhancements

- **Logging**: Use Rich console for user-facing output, logger for debugging    - `docs/code/REFACTOR.md` - Code refactoring

- **Config Access**: `from config import get_config` (never import removed `config.ga_params`)    - Create additional changelogs as needed (e.g., `PERF.md`, `DATA.md`)

  - **Format**: `## [YYYY-MM-DD] Brief description` + list of affected files

## Path-Specific Instructions  - **No detailed explanations**—just timestamp, description, files

Detailed instructions for different modules are in `.github/instructions/`:  - **Example**:

- `config.instructions.md` - Configuration system    ```markdown

- `ga-core.instructions.md` - GA operators & scheduler    ## [2025-10-26] Fixed incorrect penalty in group gap constraint

- `constraints.instructions.md` - Constraint functions    - `src/constraints/soft.py`

- `data-flow.instructions.md` - Encoder/decoder/entities    ```

- `validation.instructions.md` - Input & feasibility validation

- `export.instructions.md` - Report generation & plotting### 3. Major Changes: Thesis Reports in `docs/for_report/`

- `workflows.instructions.md` - Orchestration logic- For **significant architectural, algorithmic, or core logic changes** (e.g., new chromosome structure, repair strategy, constraint category):

- `tests.instructions.md` - Testing guidelines  - Create a **new file** in `docs/for_report/` (e.g., `adaptive_repair_mechanism.md`)

  - Write in **formal, thesis-ready prose**—content must be publication-quality
  - **Must begin with a comment block** suggesting thesis placement
  - **Structure**: Problem → Solution → Implementation → Results/Trade-offs
  - **Focus on WHY and WHAT**, not HOW (code does that)
  - **Example start**:
    ```markdown
    <!-- Suggested thesis placement: Chapter 3 - Algorithmic Design, Section 3.4 -->
    
    ## Adaptive Repair Mechanism
    
    To address the challenge of constraint satisfaction in highly constrained
    scheduling problems, the engine implements an adaptive repair mechanism...
    ```

### 4. What Requires Documentation?

#### ✅ **Must Document** (Thesis Report in `docs/for_report/`)
- New algorithms or heuristics
- Architectural changes (e.g., chromosome structure)
- New constraint categories or evaluation strategies
- Novel optimization techniques
- Changes to GA operators (selection, crossover, mutation logic)
- Performance-critical design decisions

#### 📝 **Changelog Only** (Entry in `docs/code/`)
- Bugfixes
- Minor refactoring (variable renames, code cleanup)
- Small performance tweaks
- Configuration changes
- Data format adjustments
- UI/output formatting changes

#### [!ERR] **No Documentation Needed**
- Typo fixes
- Comment updates
- Whitespace changes
- Trivial variable renames

### 5. Documentation File Naming
- **Thesis Reports**: Use descriptive, topic-based names (e.g., `hybrid_population_initialization.md`, `nsga2_selection_mechanism.md`)
- **Changelogs**: Use category-based names (e.g., `BUGFIX.md`, `ENHANCE.md`, `REFACTOR.md`)

### 6. When in Doubt
- **Ask**: "Would this go in my thesis?"
  - **Yes** → Create thesis report in `docs/for_report/`
  - **No** → Add changelog entry in `docs/code/`
- **Ask**: "Did I change the algorithm or just fix a bug?"
  - **Algorithm** → Thesis report
  - **Bug/Tweak** → Changelog
