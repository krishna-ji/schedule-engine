# Path-Specific Copilot Instructions

This directory contains detailed instructions for different parts of the schedule-engine codebase. Each file applies to specific file patterns using GitHub Copilot's path-specific instructions feature.

## Instruction Files

| File | Applies To | Description |
|------|------------|-------------|
| `config.instructions.md` | `config/**/*.{py,yaml}` | Configuration system (Pydantic models, YAML configs) |
| `ga-core.instructions.md` | `src/{core,ga}/**/*.py` | GA scheduler, operators, population strategies |
| `constraints.instructions.md` | `src/constraints/**/*.py` | Hard & soft constraint functions |
| `data-flow.instructions.md` | `src/{encoder,decoder,entities}/**/*.py` | Data transformation pipeline |
| `validation.instructions.md` | `src/validation/**/*.py` | Input validation & feasibility checking |
| `export.instructions.md` | `src/exporter/**/*.py` | Report generation & plotting |
| `workflows.instructions.md` | `src/workflows/**/*.py` | Workflow orchestration |
| `tests.instructions.md` | `test/**/*.py` | Testing guidelines |

## How It Works

GitHub Copilot automatically loads the appropriate instruction file based on the file you're editing. For example:

- Editing `config/models.py` → Copilot reads `config.instructions.md`
- Editing `src/ga/operators/mutation.py` → Copilot reads `ga-core.instructions.md`
- Editing `test/test_constraints.py` → Copilot reads `tests.instructions.md`

## Benefits of This Structure

✅ **Focused Context**: Each instruction file contains only relevant information for that module
✅ **Reduced Token Usage**: Copilot doesn't load unrelated instructions
✅ **Easier Maintenance**: Update instructions for specific modules without affecting others
✅ **Better Suggestions**: More targeted guidance = more accurate code suggestions

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

Path-specific instructions **supplement** (not replace) the main instructions.
