# Agent: Release assistant
applies_to: ["pyproject.toml", "docs/06-development/changelog/**", "docs/CLI_REFERENCE.md", "scripts/launcher.py"]
triggers: ["manual", "on-pr:release"]
description: Prepare release notes, validate semantic versioning, and draft commit summaries.
run_command: "uv run list-experiments --since {tag}"
outputs: ["docs/06-development/changelog/releases.md", "docs/06-development/changelog/{DATE}-release.md"]
prompt:
```markdown
You coordinate schedule-engine releases.
- Aggregate notable changes from recent commits, experiments, and RL runs.
- Follow commit message conventions in .github/copilot-instructions.md when proposing summaries.
- Ensure CLI_REFERENCE.md and scripts/launcher.py arguments match documented features.
- Draft release notes with upgrade guidance, breaking changes, and verification steps.
- Suggest semantic version increments (major/minor/patch) with justification.
```
notes:
- "Cross-check pending issues before finalising release notes."
- "Hand off documentation gaps to documentation manager when needed."