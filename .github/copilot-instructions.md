# Schedule Engine Agent Guide

## Commit Message Format
When generating or suggesting commit messages for this repository:
- Follow this format: `<type>(<scope>): <summary>`
- Use imperative mood ("add", "fix", "update") and keep summary under 72 characters.
- Choose types from: `feat`, `fix`, `data`, `method`, `analysis`, `result`, `doc`, `test`, `refactor`, `setup`, `release`.
- Include a short explanation of the change's purpose or scientific motivation if relevant.
- Reference experiment numbers, datasets, or manuscript sections when applicable.
- Avoid generic summaries like "update" or "minor fix".

## Project Guidelines
- create all test files inside test/


- **Architecture** `main.py` drives the GA run: load JSON via `src/encoder/input_encoder.py`, seed RNG, register GA toolbox, execute NSGA-II for `config/ga_params.NGEN`, decode winners with `src/decoder/individual_decoder.py`, then export using `src/exporter/exporter.py`.
- **Data Flow** Inputs in `data/*.json` become entities (`src/entities/*`) and availability quanta (`QuantumTimeSystem`). GA chromosomes (`src/ga/sessiongene.py`) pass through seeding (`src/ga/population.py`), evaluation (`src/ga/evaluator/{fitness,detailed_fitness}.py` plus `src/constraints/{hard,soft}.py`), and decoding to `CourseSession` records.
- **Time System** Always convert between wall-clock and quanta with `QuantumTimeSystem`; operating quanta come from `get_all_operating_quanta()` and `SessionGene.quanta` must stay unique and sorted to avoid duplicate-slot penalties.
- **Seeding & Mutation** Keep individuals as `list[SessionGene]` with `creator.FitnessMulti(weights=(-1.0, -0.01))`. Seeding relies on `generate_course_group_aware_population` (respects course–group relationships). Mutations in `src/ga/operators/mutation.py` expect a `context` dict containing `courses`, `groups`, `instructors`, `rooms`, `available_quanta`.
- **Constraints** Hard rules live in `src/constraints/hard.py`, soft penalties in `src/constraints/soft.py`; both consume decoded `CourseSession` lists. New constraints should be pure functions returning integers and registered in the evaluator modules.
- **Evaluation Metrics** `average_pairwise_diversity` in `src/metrics/diversity.py` assumes chromosomes remain aligned; update this if you change gene ordering or length.
- **Exports & Reports** `export_everything` writes JSON plus plots into `output/evaluation_<timestamp>/`. Plot modules under `src/exporter/` require the same directory structure. Calendar appearance is controlled via `config/calendar_config.py`.
- **Workflows** Run the solver with `python main.py`. Dependencies: DEAP, Matplotlib (install with `pip install -r requirements.txt` when available, otherwise `pip install deap matplotlib`). With no automated tests, shorten `config/ga_params.NGEN` for quick smoke runs.
- **Warnings & Defaults** When availability is absent, groups default to operating quanta, instructors/rooms become fully available. Existing warning prints highlight missing enrollments/qualifications; keep them when extending validation.
- **Style Notes** Match existing ASCII-only style, Sunday-first ordering for schedules, and reuse shared constants (e.g., `QuantumTimeSystem.DAY_NAMES`) when adding new reports.

## Documentation Policy (Bifurcated System)

### 1. Code Documentation: Docstrings Only
- **All code must be documented using Python docstrings** (functions, classes, modules).
- Docstrings are the single source of truth for what code does.
- **Never create separate .md files to document code**—code documents itself.

### 2. Minor Changes: Changelog in `docs/code/`
- For **routine bugfixes, small enhancements, or refactoring** that don't alter core architecture:
  - Add a **single timestamped entry** to the appropriate changelog:
    - `docs/code/BUGFIX.md` - Bug fixes
    - `docs/code/ENHANCE.md` - Minor enhancements
    - `docs/code/REFACTOR.md` - Code refactoring
    - Create additional changelogs as needed (e.g., `PERF.md`, `DATA.md`)
  - **Format**: `## [YYYY-MM-DD] Brief description` + list of affected files
  - **No detailed explanations**—just timestamp, description, files
  - **Example**:
    ```markdown
    ## [2025-10-26] Fixed incorrect penalty in group gap constraint
    - `src/constraints/soft.py`
    ```

### 3. Major Changes: Thesis Reports in `docs/for_report/`
- For **significant architectural, algorithmic, or core logic changes** (e.g., new chromosome structure, repair strategy, constraint category):
  - Create a **new file** in `docs/for_report/` (e.g., `adaptive_repair_mechanism.md`)
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
