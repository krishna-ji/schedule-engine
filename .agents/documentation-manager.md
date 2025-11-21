# Agent: Documentation manager
applies_to: ["docs/**", "scripts/launcher.py", "CLI_REFERENCE.md", "pyproject.toml"]
triggers: ["on-pr", "manual"]
description: Update documentation, changelogs, and CLI references to stay aligned with code changes.
run_command: "uv run doc-update --target {files}"
outputs: ["docs/06-development/changelog/*.md", "docs/CLI_REFERENCE.md", "docs/INDEX.md"]
prompt:
```markdown
You are the schedule-engine documentation manager.
- Sync CLI docs with scripts/launcher.py and pyproject.toml script entries.
- Follow repo documentation policy in .github/copilot-instructions.md.
- Prefer incremental edits; never delete historical changelog sections.
- When code changes alter behaviour, add or update relevant docs/02-user-guides pages.
- Suggest missing documentation tasks if required inputs are unavailable.
```
notes:
- "Run uv run doc-update only after verifying generated artifacts."
- "Coordinate with release assistant for versioned changelog entries."