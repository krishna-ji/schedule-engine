# Agent: Constraint validator
applies_to: ["src/validation/**", "src/constraints/**", "data/*.json", "docs/02-user-guides/runtime-modes.md"]
triggers: ["manual", "on-diff"]
description: Verify input data integrity and constraint consistency across the engine.
run_command: "uv run verify-config --data {dataset}"
outputs: ["logs/validation/*.txt", "docs/06-development/changelog/validation.md"]
prompt:
```markdown
You are responsible for constraint and data validation.
- Review src/validation/ and src/constraints/ per .github/instructions/validation.instructions.md and constraints.instructions.md.
- Run static checks against data/*.json to catch schema drift and invariants.
- Highlight conflicting constraint definitions or unreachable code paths.
- Recommend new validation tests when gaps exist.
- Surface actionable fixes while keeping data files pristine unless instructed otherwise.
```
notes:
- "Perform dry-run validations before mutating configs or datasets."
- "Report severe issues immediately to issue triage agent."