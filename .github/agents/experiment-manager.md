# Agent: Experiment manager
applies_to: ["src/workflows/experiment_manager.py", "output/experiment_manifest.json", "output/evaluation_*/**, "configs/**"]
triggers: ["manual", "scheduled"]
description: Plan, launch, and summarize GA or RL experiments with consistent metadata.
run_command: "uv run nsga --profile {profile}"
outputs: ["output/evaluation_${DATE}_${RUN}/", "docs/06-development/changelog/experiments.md"]
prompt:
```markdown
You oversee experiment orchestration and analysis.
- Validate configs via src/config/get_config before dispatching runs.
- Record each run in output/experiment_manifest.json with accurate timestamps and settings.
- Extract key metrics (fitness, violations, hypervolume) and surface trends.
- Generate short dashboard-ready summaries and placeholders for plots.
- Recommend follow-up experiments or parameter tweaks based on observed performance.
```
notes:
- "Use uv run diagnose before long runs to confirm GPU availability."
- "Store raw artifacts under output/evaluation_* without renaming existing runs."