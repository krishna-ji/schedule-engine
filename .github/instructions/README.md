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
