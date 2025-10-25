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
- **Documentation** Keep project docs inside `docs/`. **Be VERY concise**: focus on concepts, architecture, and design decisions ONLY—NO code snippets (only if absolutely crucial can use code snippets), NO detailed implementations, NO verbose explanations. Structure: problem → solution approach → key components. Use bullet points. Max 1-2 short paragraphs per major topic. Reference module names for details (e.g., "see `constraints/hard.py`") rather than explaining code. Split theory from implementation. Avoid filler words. Document WHY (rationale, tradeoffs), not HOW (code does that).
- **Style Notes** Match existing ASCII-only style, Sunday-first ordering for schedules, and reuse shared constants (e.g., `QuantumTimeSystem.DAY_NAMES`) when adding new reports.
