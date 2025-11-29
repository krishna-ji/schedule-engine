# Path-Specific AI Agent Instructions

This directory contains high-entropy, constraint-specific instructions for AI-assisted development across the schedule-engine codebase. Each instruction file targets specific modules using glob patterns, optimized for agentic AI coding tools (GitHub Copilot, Cursor, Cody, etc.).

## Instruction Files

| File | Applies To | Description |
|------|------------|-------------|
| `config.instructions.md` | `config/**/*.{py,yaml}` | Configuration system (Pydantic models, YAML configs, runtime modes) |
| `ga-core.instructions.md` | `src/{core,ga}/**/*.py` | GA scheduler, operators, population strategies |
| `constraints.instructions.md` | `src/constraints/**/*.py` | Hard & soft constraint functions |
| `data-flow.instructions.md` | `src/{encoder,decoder,entities}/**/*.py` | Data transformation pipeline |
| `validation.instructions.md` | `src/validation/**/*.py` | Input validation & feasibility checking |
| `export.instructions.md` | `src/exporter/**/*.py` | Report generation & plotting |
| `workflows.instructions.md` | `src/workflows/**/*.py` | Workflow orchestration & experiment management |
| `tests.instructions.md` | `test/**/*.py` | Testing guidelines |
| `rl.instructions.md` | `src/rl/**/*.py, scripts/**/rl*.py` | RL environment, training, deployment |

## How It Works

GitHub Copilot automatically loads the appropriate instruction file based on the file you're editing. For example:

- Editing `config/models.py` → Copilot reads `config.instructions.md`
- Editing `src/ga/operators/mutation.py` → Copilot reads `ga-core.instructions.md`
- Editing `test/test_constraints.py` → Copilot reads `tests.instructions.md`

## Design Principles for AI Agents

 **High-Entropy Terminology**: Use precise domain vocabulary (CSP, NSGA-II, phenotype, genotype) over generic terms (algorithm, solution, optimization)
 **Explicit Constraints**: Specify invariants, preconditions, postconditions for every operation
 **Structured Context**: JSON-compatible formats, typed schemas, unambiguous specifications
 **Actionable Directives**: Imperative commands ("Preserve course-group relationships") not vague guidance ("Be careful with groups")
 **Minimal Ambiguity**: Eliminate filler words, pronouns without clear antecedents, vague quantifiers ("some", "often")
 **Type Safety**: All pure Python code must pass mypy strict mode; type: ignore only for legitimate library limitations

## Benefits of This Architecture

 **Token Efficiency**: Path-specific loading reduces context bloat (10-50% token savings per session)
 **Deterministic Outputs**: High-entropy instructions reduce AI hallucinations via precise terminology
 **Maintenance Isolation**: Module-specific updates don't cascade to unrelated components
 **Multi-Agent Compatibility**: Works across Copilot, Cursor, Cody, Tabnine, Amazon CodeWhisperer

## Adding New Instructions

To add instructions for a new module:

1. Create `.github/instructions/mymodule.instructions.md`
2. Add front matter with `applyTo` glob pattern:

   ```yaml
   ---
   applyTo: "src/mymodule/**/*.py"
   ---
   ```

3. Write module-specific guidelines
4. Update this README

## Main Instructions

The repository-wide instructions are in `.github/copilot-instructions.md` (parent directory). This file contains:

- Project overview
- Tech stack
- Repository structure
- General coding standards
- Documentation policy
- Runtime mode architecture
- Experimentation guidelines

Path-specific instructions **supplement** (not replace) the main instructions.

## Type Checking Best Practices

All pure Python packages must pass strict mypy type checking:

### Verified Packages (100% Coverage)

-  `src/utils/` - 10 files, comprehensive utilities
-  `src/config/` - 4 files, Pydantic models and loader
-  `src/constants.py`, `src/exceptions.py` - Core definitions
-  `src/entities/` - 6 files, domain models
-  `src/encoder/` - 2 files, JSON input processing
-  `src/decoder/` - 1 file, schedule decoding
-  `src/core/types.py` - Type definitions
-  `src/constraints/` - 5 files, hard/soft constraints
-  `src/metrics/` - 6 files, performance metrics
-  `src/exporter/` - 13 files, all plotting and export functions
-  `src/validation/` - 2 files, input validation and feasibility
-  `src/diversity/` - 4 files, diversity metrics and archive
-  `src/lns/` - 5 files, large neighborhood search
-  `src/heuristics/` - 16 files, repair heuristics and meta-strategies
-  `src/workflows/` - 4 files, orchestration and experiment management

### Excluded Packages (Library Dependencies)

- `src/ga/` - DEAP Individual type uses Any
- `src/rl/` - PyTorch/Stable-Baselines3 optional components
- `src/lns/cp_solver.py` - OR-Tools integration (external solver)

### Type: Ignore Usage (33 Total)

Only for legitimate library limitations:

- yaml module (3) - No type stubs available
- numpy assignments (3) - floating[Any] incompatible with float
- QuantumTimeSystem forward refs (3) - Circular import resolution
- Decorator metadata (2) - Dynamic function attributes
- Pydantic internals (2) - Model validation internals
- RL optional components (3) - Conditional imports
- ga/population.py (13) - DEAP Individual type limitations
- validation (2) - Backward compatibility patterns
- feasibility_checker (1) - numpy operations
- Pool._processes (1) - Private multiprocessing attribute
- archive.py numpy types (2) - ndarray → list conversions

### Common Type Fixes

1. **Optional parameters**: Use `param: T | None = None` (not implicit None)
2. **numpy wraps**: `float(np.mean(...))`, `int(np.argmax(...))`
3. **Dict keys**: Use tuple types for complex keys: `dict[tuple[str, int], float]`
4. **Assertions**: Add after None checks to satisfy type checker
5. **Imports**: Import correct types (SessionGene vs CourseSession)
6. **Conditional unpacking**: Use isinstance checks before tuple unpacking
7. **Defaultdict types**: Specify key and value types explicitly
8. **Path vs str**: Convert Path to str for function calls expecting strings

### Verification Commands

```bash
# Check all pure Python packages
uv run mypy src/diversity/ src/lns/ src/heuristics/ src/workflows/

# Check specific package
uv run mypy src/utils/

# Full check (includes DEAP/RL - expect some errors)
uv run mypy src/
```

## Experimentation Best Practices

When adding major experimental features:

### 1. Modular Config Structure

- Create category folder: `configs/{category}/`
- Add mode config: `configs/{category}/{n}-{name}.yaml`
- Use descriptive names (e.g., `5-rl-guided.yaml`)

### 2. Killswitch Implementation

- Add master switch to `configs/base.yaml`: `feature.enabled: false`
- Override in mode configs: `feature.enabled: true`
- Check in code: `if not config.feature.enabled: return`
- Add validation: `RuntimeMode.validate_config()`

### 3. Runtime Mode Registration

- Add enum entry: `src/config/runtime_mode.py`
- Register UV shortcut: `pyproject.toml` `[project.scripts]`
- Add CLI entry point: `scripts/launcher.py` (e.g., `def main_myfeature():`)
- Document in user guide: `docs/02-user-guides/runtime-modes.md`

### 4. Experiment Tracking

- Use `ExperimentManager` for all production runs
- Register with meaningful names: `prod-{mode}-r{run_number}`
- Add tags: `["production", "ablation", "gpu"]`
- Include notes: Document experiment purpose
- Commit `manifest.json` for reproducibility

### 5. Systematic Comparison

- Run all modes with same data
- Use consistent environments (test/prod)
- Generate comparison table: `python main.py --compare`
- Export to CSV: `manager.export_comparison_csv()`
- Document results in thesis/implementation notes

### Example: Adding RL Feature

1. **Config structure**:
   - `configs/base.yaml` → `rl.enabled: false`
   - `configs/rl/e-rl-guided.yaml` → `rl.enabled: true`

2. **Runtime mode**:
   - `src/config/runtime_mode.py` → `RL_GUIDED = auto()`
   - Validation: Check `rl.enabled == True` for RL mode

3. **UV shortcut**:
   - `pyproject.toml` → `rl = "scripts.launcher:main_rl"`
   - `scripts/launcher.py` → `def main_rl(): ...`

4. **Experiment tracking**:

   ```python
   manager = ExperimentManager()
   run = manager.register_run(
       experiment_name="prod-rl-r01",
       runtime_mode=RuntimeMode.RL_GUIDED,
       tags=["rl", "production"]
   )
   ```

5. **Documentation**:
   - User guide: `docs/02-user-guides/runtime-modes.md`
   - Implementation notes: `docs/06-development/implementation-notes/`
   - Quick reference: `docs/QUICKREF_RUNTIME_MODES.md`
