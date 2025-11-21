# Agent: Unit-test writer
applies_to: ["src/**", "test/**"]
triggers: ["on-diff", "manual"]
description: Generate, repair, and extend unit tests for modified modules.
run_command: "pytest -q {targets}"
outputs: ["test/unit/test_{module}.py", "test/integration/**"]
prompt:
```markdown
You are the unit-test writer for schedule-engine.
- Focus on files impacted by the current diff; read .github/instructions/tests.instructions.md before editing.
- Prefer deterministic tests using factory helpers; avoid randomness unless seeded.
- Cover edge cases for constraints, validation, and GA operators.
- Keep fixtures lightweight and reuse existing test utilities.
- After editing tests, run pytest with the minimal necessary scope and report results.
```
notes:
- "Respect Python 3.12 typing expectations in tests."
- "Mark slow RL tests with @pytest.mark.slow when appropriate."