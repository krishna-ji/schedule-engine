# Development Documentation Process

Follow this workflow to keep historical context synchronized with code changes.

## 1. Decide Documentation Scope

| Change Type | Location |
| --- | --- |
| Minor bugfix/refactor | `docs/06-development/changelog/` (bugfixes or enhancements) |
| Major implementation | `docs/06-development/implementation-notes/` |
| Bug root-cause write-up | `docs/06-development/bugfixes/issue-name.md` |
| Experiment results | `docs/development/experiment-log.md` |
| Thesis-quality prose | `docs/07-thesis-report/` |

## 2. Changelog Entry Template

```
## [YYYY-MM-DD] Concise summary
- Files touched: src/..., configs/...
- Motivation: short reason
- Impact: user/developer visible effect
```

Example:
```
## [2025-11-20] Fix RL action mapper regression
- Files: src/rl/gym_env/action_mapper.py, test/rl/test_action_mapper.py
- Motivation: Action IDs 5-7 misaligned after new heuristics
- Impact: RL runs no longer fail with KeyError
```

## 3. Implementation Notes Structure

1. Overview & motivation
2. Tasks completed (bullet list)
3. Files & modules touched
4. How to enable/try feature
5. Next steps or open questions

## 4. Experiment Logging

Create/append `docs/development/experiment-log.md` with:
- Experiment name & timestamp
- Config hash + runtime mode
- Key metrics (best fitness, hard/soft violations)
- Hardware details
- Observations (e.g., "RL agent converged after gen 120")

## 5. Review Checklist

- [ ] Docs directory updated when new features land.
- [ ] Cross-links added to `docs/00-INDEX.md`.
- [ ] File paths referenced with inline code formatting (e.g., `src/ga/...`).
- [ ] Mermaid diagrams included for new flows where helpful.

Consistent documentation keeps onboarding smooth and ensures thesis reporting stays accurate.
