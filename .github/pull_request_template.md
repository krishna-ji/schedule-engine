## Pull Request Description

### Type of Change
<!-- Check relevant boxes with [x] -->
- [ ] `feat`: New feature (non-breaking change adding functionality)
- [ ] `fix`: Bug fix (non-breaking change fixing an issue)
- [ ] `refactor`: Code restructure (no functional changes)
- [ ] `perf`: Performance improvement
- [ ] `test`: Add/update tests
- [ ] `docs`: Documentation only
- [ ] `chore`: Maintenance (deps, config, tooling)

### Scope
<!-- Affected module(s) -->
**Module(s)**: `src/<module>/`

### Summary
<!-- Brief description (1-3 sentences) -->


### Motivation & Context
<!-- Why is this change needed? What problem does it solve? -->
<!-- Link related issues: Fixes #123, Resolves #456 -->


### Implementation Details
<!-- Key changes, algorithm modifications, design decisions -->


### Testing
<!-- How was this tested? -->
- [ ] Unit tests added/updated (`test/unit/`)
- [ ] Smoke test passed (`uv run nsga --test`)
- [ ] Integration test passed
- [ ] Manual testing performed

**Test commands run**:
```bash
# Example:
pytest test/unit/test_my_feature.py -v
uv run nsga --test --name "pr-validation"
```

### Checklist
<!-- Confirm all requirements before submitting -->
- [ ] Code follows PEP 8 style guidelines
- [ ] Ran `black src/ test/` (auto-format)
- [ ] Ran `ruff check src/ test/` (lint)
- [ ] Ran `pytest test/unit/` (tests pass)
- [ ] Added docstrings for new functions/classes
- [ ] Updated relevant documentation in `docs/`
- [ ] Config changes validated (`uv run verify-config`)
- [ ] Commit messages follow `<type>(<scope>): <summary>` format

### Breaking Changes
<!-- Does this PR break backward compatibility? -->
- [ ] Yes (describe migration path below)
- [x] No

<!-- If yes, describe how users should update their code/configs -->


### Screenshots/Outputs
<!-- If applicable, add screenshots or command outputs -->


### Performance Impact
<!-- Does this change affect runtime performance? -->
- [ ] Significant improvement (>10% speedup)
- [ ] Minor improvement (<10% speedup)
- [ ] No measurable impact
- [ ] Slight degradation (justified by feature value)

<!-- If performance changed, include benchmark results -->


### Deployment Notes
<!-- Special instructions for deploying this change -->
<!-- Example: requires config update, database migration, etc. -->


### Related Documentation
<!-- Link to related docs, issues, or PRs -->
- Implementation notes: `docs/06-development/implementation-notes/`
- User guide updates: `docs/02-user-guides/`
- Related issues: #
- Related PRs: #

---
**Reviewer Checklist**
- [ ] Code quality acceptable
- [ ] Tests adequate
- [ ] Documentation updated
- [ ] No security concerns
- [ ] Performance impact acceptable
