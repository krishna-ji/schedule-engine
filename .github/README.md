# GitHub Repository Configuration

## Purpose
This directory contains GitHub-specific configuration files for automated workflows, security scanning, code quality enforcement, and AI-assisted development.

## Structure

```
.github/
├── workflows/              # GitHub Actions CI/CD pipelines
│   ├── ci.yml             # Test, lint, format check, smoke test
│   ├── security.yml       # Dependency scan, CodeQL, secret detection
│   └── release.yml        # Build, tag, publish to PyPI
├── instructions/          # Path-specific AI agent instructions
│   ├── README.md          # AI agent guidance overview
│   ├── cli.instructions.md
│   ├── config.instructions.md
│   ├── constraints.instructions.md
│   ├── data-flow.instructions.md
│   ├── export.instructions.md
│   ├── ga-core.instructions.md
│   ├── rl.instructions.md
│   ├── tests.instructions.md
│   ├── validation.instructions.md
│   └── workflows.instructions.md
├── ISSUE_TEMPLATE/        # Structured issue forms
│   ├── bug_report.yml     # Bug report template
│   ├── feature_request.yml # Feature request template
│   └── config.yml         # Issue template config
├── CODEOWNERS             # Auto-assign PR reviewers
├── dependabot.yml         # Automated dependency updates
├── pull_request_template.md # PR checklist and structure
└── copilot-instructions.md # Repository-wide AI instructions

```

## Workflows

### ci.yml - Continuous Integration
**Trigger**: Push to main/dev-* branches, PRs to main/dev-*
**Jobs**:
- **test**: Run pytest unit tests with coverage (Codecov integration)
- **lint**: Black format check, Ruff linting, MyPy type checking
- **config-validation**: Validate YAML configs and Pydantic schemas
- **smoke-test**: Run NSGA-II 30-gen test to verify end-to-end functionality

**Duration**: ~5-10 minutes
**Required to pass**: Yes (branch protection recommended)

### security.yml - Security Scanning
**Trigger**: Push to main/dev-*, PRs to main, weekly Monday 6 AM UTC
**Jobs**:
- **dependency-scan**: Safety check for vulnerable dependencies
- **codeql-analysis**: GitHub CodeQL static analysis for Python
- **secret-scan**: TruffleHog secret detection in commit history

**Duration**: ~3-5 minutes
**Required to pass**: No (informational)

### release.yml - Automated Release
**Trigger**: Git tags matching `v*.*.*` (e.g., `v1.2.3`)
**Jobs**:
- **build**: Build Python package distribution (wheel + sdist)
- **github-release**: Create GitHub release with artifacts
- **publish-pypi**: Publish to PyPI (requires PYPI_API_TOKEN secret)

**Duration**: ~2-3 minutes
**Manual steps**: Create and push git tag

## AI Agent Instructions

### High-Entropy Design Principles
1. **Domain Vocabulary**: Use CSP, NSGA-II, Pareto dominance, phenotype/genotype over generic terms
2. **Explicit Invariants**: Specify preconditions, postconditions, and preservation constraints
3. **Typed Schemas**: Reference Pydantic models, YAML schemas, JSON structures
4. **Actionable Commands**: Imperative directives ("Preserve X", "Validate Y") not suggestions
5. **Minimal Ambiguity**: Eliminate pronouns, vague quantifiers, filler words

### Path-Specific Loading
- `config.instructions.md` → `configs/**/*.yaml`, `src/config/**/*.py`
- `ga-core.instructions.md` → `src/core/**/*.py`, `src/ga/**/*.py`
- `constraints.instructions.md` → `src/constraints/**/*.py`
- `rl.instructions.md` → `src/rl/**/*.py`, `scripts/**/rl*.py`
- ... (see `instructions/README.md` for full mapping)

### Multi-Agent Compatibility
Tested with:
-  GitHub Copilot
-  Cursor AI
-  Cody (Sourcegraph)
-  Tabnine
-  Amazon CodeWhisperer

## Issue Templates

### bug_report.yml
**Sections**:
- Bug description with reproduction steps
- Expected vs actual behavior
- Error logs and stack traces
- Environment (OS, Python version, GPU)
- Configuration and data characteristics
- Runtime mode selection
- Reproducibility assessment

### feature_request.yml
**Sections**:
- Problem statement and motivation
- Proposed solution with alternatives
- Category (GA, constraints, RL, etc.)
- Priority/impact assessment
- Use cases and implementation sketch
- Breaking change analysis
- Documentation and testing strategy

## CODEOWNERS

Defines code review ownership:
- **Global**: `@krishna-ji` reviews all changes
- **Configs**: Extra scrutiny for `configs/**/*.yaml`, `src/config/**/*.py`
- **Core GA**: Critical path review for `src/core/**`, `src/ga/**`
- **RL Modules**: Experimental feature review for `src/rl/**`
- **Documentation**: Quality control for `docs/**`, `*.md`

## Dependabot

**Update frequency**:
- Python dependencies: Weekly (Monday)
- GitHub Actions: Monthly

**Configuration**:
- Max 5 open PRs per ecosystem
- Ignores major version updates for DEAP, PyTorch, Stable-Baselines3
- Auto-labels: `dependencies`, `python`, `ci`
- Commit prefix: `chore(deps):` or `chore(ci):`

## Pull Request Template

**Required checklist**:
- [ ] Code follows PEP 8 style
- [ ] Ran Black + Ruff + pytest
- [ ] Added docstrings
- [ ] Updated documentation
- [ ] Config validation passed
- [ ] Commit messages formatted correctly

**Sections**:
- Type of change (feat, fix, refactor, etc.)
- Scope (affected modules)
- Summary and motivation
- Implementation details
- Testing strategy
- Performance impact
- Deployment notes

## Repository-Wide Instructions

`copilot-instructions.md` contains:
- Project overview with metaheuristic taxonomy
- Tech stack (DEAP, PyTorch, Stable-Baselines3)
- Repository structure
- Coding standards (PEP 8, type hints, docstrings)
- Documentation policy (7 categories)
- Commit message format
- Path-specific instruction index

## Best Practices

### For Contributors
1. **Read instructions first**: Check `copilot-instructions.md` + relevant path-specific file
2. **Use issue templates**: Bug reports and feature requests require structured input
3. **Run pre-commit checks**: Black, Ruff, pytest, config validation
4. **Follow commit format**: `<type>(<scope>): <summary>` (e.g., `feat(rl): add PPO agent`)
5. **Test with smoke test**: `uv run nsga --test` before pushing

### For Maintainers
1. **Update instructions when changing architecture**: Keep AI context accurate
2. **Review Dependabot PRs weekly**: Merge security updates promptly
3. **Monitor CI failures**: Fix failing tests immediately
4. **Update CODEOWNERS**: Add reviewers for new critical modules
5. **Validate workflows locally**: Use `act` tool to test GitHub Actions

## Maintenance Schedule

| Task | Frequency | Owner |
|------|-----------|-------|
| Review Dependabot PRs | Weekly | @krishna-ji |
| Update AI instructions | Per major feature | @krishna-ji |
| Audit security scans | Weekly | @krishna-ji |
| Update workflow versions | Quarterly | @krishna-ji |
| Review issue templates | Bi-annually | @krishna-ji |

## References

- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Dependabot Config](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file)
- [Issue Templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests)
- [CODEOWNERS](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)
- [GitHub Copilot Instructions](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot)
