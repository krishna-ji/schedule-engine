# Schedule Engine Agent Guide# Schedule Engine - Repository Overview# Schedule Engine Agent Guide



## Project Summary



University course scheduling engine using NSGA-II genetic algorithm with constraint-based optimization. Written in Python with DEAP, rich terminal UI, and YAML configuration.## Project Summary## Commit Message Format



## Tech StackUniversity course scheduling engine using NSGA-II genetic algorithm with constraint-based optimization. Written in Python with DEAP, rich terminal UI, and YAML configuration.When generating or suggesting commit messages for this repository:



- **Language**: Python 3.8+- Follow this format: `<type>(<scope>): <summary>`

- **Core Libraries**: DEAP (genetic algorithms), Pydantic (validation), Rich (terminal UI)

- **Config**: YAML-based with Pydantic models## Tech Stack- Use imperative mood ("add", "fix", "update") and keep summary under 72 characters.

- **Testing**: Manual testing with test configurations

- **Language**: Python 3.8+- Choose types from: `feat`, `fix`, `data`, `method`, `analysis`, `result`, `doc`, `test`, `refactor`, `setup`, `release`.

## Repository Structure

- **Core Libraries**: DEAP (genetic algorithms), Pydantic (validation), Rich (terminal UI)- Include a short explanation of the change's purpose or scientific motivation if relevant.

```

schedule-engine/- **Config**: YAML-based with Pydantic models- Reference experiment numbers, datasets, or manuscript sections when applicable.

├── main.py              # CLI entry point (--env test|dev|prod)

├── config/              # Pydantic models & loaders- **Testing**: Manual testing with test configurations- Avoid generic summaries like "update" or "minor fix".

├── configs/             # YAML configuration files

├── src/

│   ├── core/            # GA scheduler & types

│   ├── ga/              # GA operators, population, repair## Repository Structure## Project Guidelines

│   ├── constraints/     # Hard & soft constraints

│   ├── encoder/         # JSON → entities + time system```- create all test files inside test/

│   ├── decoder/         # Individual → CourseSession

│   ├── entities/        # Domain modelsschedule-engine/

│   ├── exporter/        # PDF/JSON/plots generation

│   ├── validation/      # Input & feasibility checks├── main.py              # CLI entry point (--env test|dev|prod)### Architecture (Refactored Modular Structure)

│   ├── workflows/       # Orchestration logic

│   └── utils/           # Helpers & logging├── config/              # Pydantic models & loaders- **Entry Point** `main.py` is now a lightweight CLI (< 100 lines) that parses `--env` or `--config` arguments, loads YAML configuration via `config/loader.py`, and delegates to `src/workflows/standard_run.py`

├── data/                # Input JSON files

├── test/                # Test files (ALL tests go here)├── configs/             # YAML configuration files- **Configuration System** All settings (GA params, constraints, parallelization, repair, feasibility) are defined in YAML files under `configs/{test,dev,prod}.yaml`. Pydantic models in `config/models.py` provide type-safe validation. Access config anywhere via `from config import get_config`

└── docs/                # Documentation

```├── src/- **Workflow Orchestration** `src/workflows/standard_run.py` contains `run_standard_workflow()` which orchestrates: data loading → validation → feasibility checks → GA execution → decoding → report generation. This replaces the old monolithic `main.py`



## Project Guidelines│   ├── core/            # GA scheduler & types- **GA Execution** `src/core/ga_scheduler.py` contains `GAScheduler` class that encapsulates DEAP toolbox setup, population initialization (hybrid/smart/random strategies), evolution loop, and metrics tracking. Configuration passed via `GAConfig` dataclass



- Create all test files inside `test/`│   ├── ga/              # GA operators, population, repair- **Multiprocessing Support** Parallel fitness evaluation enabled via `parallel.use_multiprocessing` in YAML. Worker processes initialized in `_worker_init()` with serialized context (Windows-safe spawn method)



### Architecture (Refactored Modular Structure)│   ├── constraints/     # Hard & soft constraints



- **Entry Point** `main.py` is now a lightweight CLI (< 100 lines) that parses `--env` or `--config` arguments, loads YAML configuration via `config/loader.py`, and delegates to `src/workflows/standard_run.py`│   ├── encoder/         # JSON → entities + time system### Data Flow & Core Components

- **Configuration System** All settings (GA params, constraints, parallelization, repair, feasibility) are defined in YAML files under `configs/{test,dev,prod}.yaml`. Pydantic models in `config/models.py` provide type-safe validation. Access config anywhere via `from config import get_config`

- **Workflow Orchestration** `src/workflows/standard_run.py` contains `run_standard_workflow()` which orchestrates: data loading → validation → feasibility checks → GA execution → decoding → report generation. This replaces the old monolithic `main.py`│   ├── decoder/         # Individual → CourseSession- **Input Loading** JSON files in `data/*.json` → `src/encoder/input_encoder.py` → entities (`src/entities/*`) + `QuantumTimeSystem` quanta → `SchedulingContext` dataclass

- **GA Execution** `src/core/ga_scheduler.py` contains `GAScheduler` class that encapsulates DEAP toolbox setup, population initialization (hybrid/smart/random strategies), evolution loop, and metrics tracking. Configuration passed via `GAConfig` dataclass

- **Multiprocessing Support** Parallel fitness evaluation enabled via `parallel.use_multiprocessing` in YAML. Worker processes initialized in `_worker_init()` with serialized context (Windows-safe spawn method)│   ├── entities/        # Domain models- **GA Chromosomes** Individuals are `list[SessionGene]` with `creator.FitnessMulti(weights=(-1.0, -0.01))`. Population initialization supports three strategies (configured via `ga.population_strategy`): "hybrid" (25% greedy, 50% smart, 25% random), "smart" (100% constraint-aware), "random" (baseline)



### Data Flow & Core Components│   ├── exporter/        # PDF/JSON/plots generation- **Genetic Operators** Crossover: `crossover_course_group_aware()` preserves course-group relationships. Mutation: `mutate_individual()` requires `context` dict with `courses`, `groups`, `instructors`, `rooms`, `available_quanta`. Both operators can trigger repair heuristics if `repair.apply_after_crossover/mutation` is enabled



- **Input Loading** JSON files in `data/*.json` → `src/encoder/input_encoder.py` → entities (`src/entities/*`) + `QuantumTimeSystem` quanta → `SchedulingContext` dataclass│   ├── validation/      # Input & feasibility checks- **Constraint Evaluation** Hard constraints in `src/constraints/hard.py`, soft in `src/constraints/soft.py`. Both return integer penalties. Constraints are enabled/weighted via YAML config under `hard_constraints` and `soft_constraints` sections. Evaluators in `src/ga/evaluator/{fitness,detailed_fitness}.py` aggregate constraint penalties

- **GA Chromosomes** Individuals are `list[SessionGene]` with `creator.FitnessMulti(weights=(-1.0, -0.01))`. Population initialization supports three strategies (configured via `ga.population_strategy`): "hybrid" (25% greedy, 50% smart, 25% random), "smart" (100% constraint-aware), "random" (baseline)

- **Genetic Operators** Crossover: `crossover_course_group_aware()` preserves course-group relationships. Mutation: `mutate_individual()` requires `context` dict with `courses`, `groups`, `instructors`, `rooms`, `available_quanta`. Both operators can trigger repair heuristics if `repair.apply_after_crossover/mutation` is enabled│   ├── workflows/       # Orchestration logic- **Repair System** Registry-based repair heuristics in `src/ga/operators/repair_registry.py`. Selective mode (configured via `repair.selective_mode`) uses violation detection (`src/ga/operators/violation_detector.py`) to target only problematic genes. Adaptive repair can trigger based on stagnation or periodic intervals

- **Constraint Evaluation** Hard constraints in `src/constraints/hard.py`, soft in `src/constraints/soft.py`. Both return integer penalties. Constraints are enabled/weighted via YAML config under `hard_constraints` and `soft_constraints` sections. Evaluators in `src/ga/evaluator/{fitness,detailed_fitness}.py` aggregate constraint penalties

- **Repair System** Registry-based repair heuristics in `src/ga/operators/repair_registry.py`. Selective mode (configured via `repair.selective_mode`) uses violation detection (`src/ga/operators/violation_detector.py`) to target only problematic genes. Adaptive repair can trigger based on stagnation or periodic intervals│   └── utils/           # Helpers & logging- **Feasibility Checks** Pre-GA validation in `src/validation/feasibility_checker.py` detects unsolvable problems (instructor bottlenecks, room capacity issues, pigeonhole violations). Configured via `feasibility` section in YAML. Can fail-fast or proceed with warnings

- **Feasibility Checks** Pre-GA validation in `src/validation/feasibility_checker.py` detects unsolvable problems (instructor bottlenecks, room capacity issues, pigeonhole violations). Configured via `feasibility` section in YAML. Can fail-fast or proceed with warnings

├── data/                # Input JSON files

### Time System

├── test/                # Test files (ALL tests go here)### Time System

- **Quantum Conversion** Always convert between wall-clock and quanta using `QuantumTimeSystem`. Operating quanta come from `get_all_operating_quanta()`. `SessionGene.quanta` must be unique and sorted to avoid duplicate-slot penalties

- **Quantum Size** Configurable via `time.quantum_minutes` (default 60). Time preferences (`earliest_preferred_time`, `latest_preferred_time`, `midday_break_start/end`) also in YAML config└── docs/                # Documentation (see below)- **Quantum Conversion** Always convert between wall-clock and quanta using `QuantumTimeSystem`. Operating quanta come from `get_all_operating_quanta()`. `SessionGene.quanta` must be unique and sorted to avoid duplicate-slot penalties



### Exports & Reports```- **Quantum Size** Configurable via `time.quantum_minutes` (default 60). Time preferences (`earliest_preferred_time`, `latest_preferred_time`, `midday_break_start/end`) also in YAML config



- **Output Structure** Results saved to `output/evaluation_<timestamp>/`: `schedule.json`, `ScheduleCalendar.pdf`, `logger.txt`, `feasibility_report.txt`, `violation_report.txt`, plus plots under `plots/` subdirectory

- **Report Generation** Orchestrated by `src/workflows/reporting.py` which calls plot modules in `src/exporter/{plothard,plotsoft,plotdiversity,plotpareto,plot_detailed_constraints}.py`

- **Calendar Display** Visual settings in `config/calendar_config.py` (unchanged). Color palette in `config/color_palette.py`## Running the Engine### Exports & Reports



### Running the Engine```bash- **Output Structure** Results saved to `output/evaluation_<timestamp>/`: `schedule.json`, `ScheduleCalendar.pdf`, `logger.txt`, `feasibility_report.txt`, `violation_report.txt`, plus plots under `plots/` subdirectory



- **Commands** `python main.py --env test` (fast, 10 gens), `python main.py --env dev` (medium, 100 gens), `python main.py --env prod` (full quality, 200+ gens), or `python main.py --config path/to/custom.yaml`python main.py --env test   # Fast (10 gens, 4 pop)- **Report Generation** Orchestrated by `src/workflows/reporting.py` which calls plot modules in `src/exporter/{plothard,plotsoft,plotdiversity,plotpareto,plot_detailed_constraints}.py`

- **Dependencies** Listed in `requirements.txt`: DEAP, Matplotlib, Pydantic, PyYAML, Rich (for terminal UI)

- **Testing Configs** Use `configs/test.yaml` for quick smoke tests (small population, few generations)python main.py --env dev    # Medium (100 gens, 20 pop)- **Calendar Display** Visual settings in `config/calendar_config.py` (unchanged). Color palette in `config/color_palette.py`



### Validation & Warningspython main.py --env prod   # Full quality (200+ gens, 50+ pop)



- **Input Validation** `src/validation/input_validator.py` checks for missing references, invalid data. Enabled via `validate=True` in `run_standard_workflow()`python main.py --config path/to/custom.yaml### Running the Engine

- **Feasibility Analysis** Run before GA (unless `feasibility.enable_checks=False`). Reports appear in console and saved to output directory

- **Missing Data Defaults** When availability is absent: groups default to operating quanta, instructors/rooms become fully available. Warnings printed during data loading```- **Commands** `python main.py --env test` (fast, 10 gens), `python main.py --env dev` (medium, 100 gens), `python main.py --env prod` (full quality, 200+ gens), or `python main.py --config path/to/custom.yaml`

- **Population Integrity** Optional validation via `ga.validate_population_integrity` (checks course-group pair alignment during crossover)

- **Dependencies** Listed in `requirements.txt`: DEAP, Matplotlib, Pydantic, PyYAML, Rich (for terminal UI)

### Style & Conventions

## Documentation System- **Testing Configs** Use `configs/test.yaml` for quick smoke tests (small population, few generations)

- **Terminal UI** Rich library for colored console output, progress bars, panels. Use `console.print()` for formatted messages

- **Sunday-First Ordering** Schedules display Sunday as first day of week (matches `QuantumTimeSystem.DAY_NAMES`)

- **Docstrings** All modules, classes, functions must have docstrings. No separate .md files for code documentation

- **Config Access** Import `from config import get_config` (runtime) or `from config.models import Config` (type hints). Never import old `config.ga_params` or `config.constraints` modules (removed in refactor)### Code Documentation### Validation & Warnings



## General Coding Standards- **Use Python docstrings only** for all modules, classes, functions- **Input Validation** `src/validation/input_validator.py` checks for missing references, invalid data. Enabled via `validate=True` in `run_standard_workflow()`



- **Python Style**: PEP 8 compliant- **Never create separate .md files to document code**- **Feasibility Analysis** Run before GA (unless `feasibility.enable_checks=False`). Reports appear in console and saved to output directory

- **Imports**: Standard lib → third-party → local (sorted alphabetically within groups)

- **Type Hints**: Use where beneficial for clarity- **Missing Data Defaults** When availability is absent: groups default to operating quanta, instructors/rooms become fully available. Warnings printed during data loading

- **Error Handling**: Informative error messages with context

- **Logging**: Use Rich console for user-facing output, logger for debugging### Change Documentation (Bifurcated)- **Population Integrity** Optional validation via `ga.validate_population_integrity` (checks course-group pair alignment during crossover)

- **Config Access**: `from config import get_config` (never import removed `config.ga_params`)

1. **Minor Changes** → `docs/code/{BUGFIX,ENHANCE,REFACTOR}.md`

## File Organization & Root Directory Policy

   - Format: `## [YYYY-MM-DD] Brief description` + file list### Style & Conventions

**CRITICAL: Keep root directory clean!**

   - For: bugfixes, minor refactoring, config changes- **Terminal UI** Rich library for colored console output, progress bars, panels. Use `console.print()` for formatted messages

### ✅ Allowed in Root Directory:

- `main.py`, `requirements.txt`, `README.md`   - **Sunday-First Ordering** Schedules display Sunday as first day of week (matches `QuantumTimeSystem.DAY_NAMES`)

- Setup scripts (`setup-venv.ps1`, `setup-venv.sh`)

- Config files (`environment.yml`, `.gitignore`, `.editorconfig`, etc.)2. **Major Changes** → `docs/for_report/new-file.md`- **Docstrings** All modules, classes, functions must have docstrings. No separate .md files for code documentation

- Core project files (`pyproject.toml`, `setup.py`, etc.)

   - Thesis-ready prose, begin with `<!-- Suggested thesis placement: ... -->`- **Config Access** Import `from config import get_config` (runtime) or `from config.models import Config` (type hints). Never import old `config.ga_params` or `config.constraints` modules (removed in refactor)

### ❌ Never Create in Root Directory:

- Documentation files (except `README.md`)   - For: new algorithms, architectural changes, novel techniques

- User guides

- Setup guides## Documentation Policy (Bifurcated System)

- Quick references

- Any `.md` files except `README.md`## Commit Message Format



### 📁 Documentation Location Rules:Format: `<type>(<scope>): <summary>`### 1. Code Documentation: Docstrings Only

- **All documentation goes in `docs/` folder**:

  - User guides → `docs/`- **Types**: `feat`, `fix`, `refactor`, `test`, `doc`, `data`, `method`, `analysis`- **All code must be documented using Python docstrings** (functions, classes, modules).

  - Setup guides → `docs/`

  - Quick references → `docs/`- Use imperative mood, keep under 72 characters- Docstrings are the single source of truth for what code does.

  - Code changelogs → `docs/code/`

  - Thesis reports → `docs/for_report/`- Example: `feat(repair): add selective violation detection`- **Never create separate .md files to document code**—code documents itself.

- **Exception**: `README.md` is the ONLY markdown file allowed in root



## Documentation Policy (Bifurcated System)

## General Coding Standards### 2. Minor Changes: Changelog in `docs/code/`

### 1. Code Documentation: Docstrings Only

- **Python Style**: PEP 8 compliant- For **routine bugfixes, small enhancements, or refactoring** that don't alter core architecture:

- **All code must be documented using Python docstrings** (functions, classes, modules).

- Docstrings are the single source of truth for what code does.- **Imports**: Standard lib → third-party → local (sorted alphabetically within groups)  - Add a **single timestamped entry** to the appropriate changelog:

- **Never create separate .md files to document code**—code documents itself.

- **Type Hints**: Use where beneficial for clarity    - `docs/code/BUGFIX.md` - Bug fixes

### 2. Minor Changes: Changelog in `docs/code/`

- **Error Handling**: Informative error messages with context    - `docs/code/ENHANCE.md` - Minor enhancements

For **routine bugfixes, small enhancements, or refactoring** that don't alter core architecture:

- Add a **single timestamped entry** to the appropriate changelog:- **Logging**: Use Rich console for user-facing output, logger for debugging    - `docs/code/REFACTOR.md` - Code refactoring

  - `docs/code/BUGFIX.md` - Bug fixes

  - `docs/code/ENHANCE.md` - Minor enhancements- **Config Access**: `from config import get_config` (never import removed `config.ga_params`)    - Create additional changelogs as needed (e.g., `PERF.md`, `DATA.md`)

  - `docs/code/REFACTOR.md` - Code refactoring

  - Create additional changelogs as needed (e.g., `PERF.md`, `DATA.md`)  - **Format**: `## [YYYY-MM-DD] Brief description` + list of affected files

- **Format**: `## [YYYY-MM-DD] Brief description` + list of affected files

- **No detailed explanations**—just timestamp, description, files## Path-Specific Instructions  - **No detailed explanations**—just timestamp, description, files

- **Example**:

  ```markdownDetailed instructions for different modules are in `.github/instructions/`:  - **Example**:

  ## [2025-10-26] Fixed incorrect penalty in group gap constraint

  - `src/constraints/soft.py`- `config.instructions.md` - Configuration system    ```markdown

  ```

- `ga-core.instructions.md` - GA operators & scheduler    ## [2025-10-26] Fixed incorrect penalty in group gap constraint

### 3. Major Changes: Thesis Reports in `docs/for_report/`

- `constraints.instructions.md` - Constraint functions    - `src/constraints/soft.py`

For **significant architectural, algorithmic, or core logic changes** (e.g., new chromosome structure, repair strategy, constraint category):

- Create a **new file** in `docs/for_report/` (e.g., `adaptive_repair_mechanism.md`)- `data-flow.instructions.md` - Encoder/decoder/entities    ```

- Write in **formal, thesis-ready prose**—content must be publication-quality

- **Must begin with a comment block** suggesting thesis placement- `validation.instructions.md` - Input & feasibility validation

- **Structure**: Problem → Solution → Implementation → Results/Trade-offs

- **Focus on WHY and WHAT**, not HOW (code does that)- `export.instructions.md` - Report generation & plotting### 3. Major Changes: Thesis Reports in `docs/for_report/`

- **Example start**:

  ```markdown- `workflows.instructions.md` - Orchestration logic- For **significant architectural, algorithmic, or core logic changes** (e.g., new chromosome structure, repair strategy, constraint category):

  <!-- Suggested thesis placement: Chapter 3 - Algorithmic Design, Section 3.4 -->

  - `tests.instructions.md` - Testing guidelines  - Create a **new file** in `docs/for_report/` (e.g., `adaptive_repair_mechanism.md`)

  ## Adaptive Repair Mechanism

    - Write in **formal, thesis-ready prose**—content must be publication-quality

  To address the challenge of constraint satisfaction in highly constrained  - **Must begin with a comment block** suggesting thesis placement

  scheduling problems, the engine implements an adaptive repair mechanism...  - **Structure**: Problem → Solution → Implementation → Results/Trade-offs

  ```  - **Focus on WHY and WHAT**, not HOW (code does that)

  - **Example start**:

### 4. What Requires Documentation?    ```markdown

    <!-- Suggested thesis placement: Chapter 3 - Algorithmic Design, Section 3.4 -->

#### ✅ **Must Document** (Thesis Report in `docs/for_report/`)    

- New algorithms or heuristics    ## Adaptive Repair Mechanism

- Architectural changes (e.g., chromosome structure)    

- New constraint categories or evaluation strategies    To address the challenge of constraint satisfaction in highly constrained

- Novel optimization techniques    scheduling problems, the engine implements an adaptive repair mechanism...

- Changes to GA operators (selection, crossover, mutation logic)    ```

- Performance-critical design decisions

### 4. What Requires Documentation?

#### 📝 **Changelog Only** (Entry in `docs/code/`)

- Bugfixes#### ✅ **Must Document** (Thesis Report in `docs/for_report/`)

- Minor refactoring (variable renames, code cleanup)- New algorithms or heuristics

- Small performance tweaks- Architectural changes (e.g., chromosome structure)

- Configuration changes- New constraint categories or evaluation strategies

- Data format adjustments- Novel optimization techniques

- UI/output formatting changes- Changes to GA operators (selection, crossover, mutation logic)

- Performance-critical design decisions

#### ❌ **No Documentation Needed**

- Typo fixes#### 📝 **Changelog Only** (Entry in `docs/code/`)

- Comment updates- Bugfixes

- Whitespace changes- Minor refactoring (variable renames, code cleanup)

- Trivial variable renames- Small performance tweaks

- Configuration changes

### 5. Documentation File Naming- Data format adjustments

- **Thesis Reports**: Use descriptive, topic-based names (e.g., `hybrid_population_initialization.md`, `nsga2_selection_mechanism.md`)- UI/output formatting changes

- **Changelogs**: Use category-based names (e.g., `BUGFIX.md`, `ENHANCE.md`, `REFACTOR.md`)

#### [!ERR] **No Documentation Needed**

### 6. When in Doubt- Typo fixes

- **Ask**: "Would this go in my thesis?"- Comment updates

  - **Yes** → Create thesis report in `docs/for_report/`- Whitespace changes

  - **No** → Add changelog entry in `docs/code/`- Trivial variable renames

- **Ask**: "Did I change the algorithm or just fix a bug?"

  - **Algorithm** → Thesis report### 5. Documentation File Naming

  - **Bug/Tweak** → Changelog- **Thesis Reports**: Use descriptive, topic-based names (e.g., `hybrid_population_initialization.md`, `nsga2_selection_mechanism.md`)

- **Changelogs**: Use category-based names (e.g., `BUGFIX.md`, `ENHANCE.md`, `REFACTOR.md`)

## Commit Message Format

### 6. When in Doubt

When generating or suggesting commit messages for this repository:- **Ask**: "Would this go in my thesis?"

  - **Yes** → Create thesis report in `docs/for_report/`

- Follow this format: `<type>(<scope>): <summary>`  - **No** → Add changelog entry in `docs/code/`

- Use imperative mood ("add", "fix", "update") and keep summary under 72 characters.- **Ask**: "Did I change the algorithm or just fix a bug?"

- Choose types from: `feat`, `fix`, `data`, `method`, `analysis`, `result`, `doc`, `test`, `refactor`, `setup`, `release`.  - **Algorithm** → Thesis report

- Include a short explanation of the change's purpose or scientific motivation if relevant.  - **Bug/Tweak** → Changelog

- Reference experiment numbers, datasets, or manuscript sections when applicable.
- Avoid generic summaries like "update" or "minor fix".
- Example: `feat(repair): add selective violation detection`

## Path-Specific Instructions

Detailed instructions for different modules are in `.github/instructions/`:

- `config.instructions.md` - Configuration system
- `ga-core.instructions.md` - GA operators & scheduler
- `constraints.instructions.md` - Constraint functions
- `data-flow.instructions.md` - Encoder/decoder/entities
- `validation.instructions.md` - Input & feasibility validation
- `export.instructions.md` - Report generation & plotting
- `workflows.instructions.md` - Orchestration logic
- `tests.instructions.md` - Testing guidelines
