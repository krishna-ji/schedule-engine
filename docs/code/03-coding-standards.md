# Coding Standards & Best Practices

The conventions below keep 15k+ lines of Python readable and safe to evolve.

## 1. Style & Formatting

- **Formatter:** Run `black src/ test/` before commits (line length = 88).
- **Linter:** `ruff check src/ test/` (treat warnings as errors; config lives in `pyproject.toml`).
- **Imports:** Use `isort` order implicitly enforced by Ruff (`stdlib → third-party → local`).
- **Spacing:** Prefer single blank line between logical sections; never rely on tabs.

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.entities.course import Course
```

## 2. Typing & Contracts

- **Future annotations** at top of every module: `from __future__ import annotations`.
- **Function signatures** must include type hints, even for internal helpers.
- **Protocols/TypedDicts** for complex payloads (e.g., RL state vectors) to ensure IDE support.
- **Pydantic models** for config/data schemas; avoid `dict[str, Any]` except in serialization layers.

```python
def evaluate_population(population: list[Individual], ctx: SchedulingContext) -> list[Fitness]:
    ...
```

## 3. Logging & Console Output

- Use `logging.getLogger(__name__)` for debug/trace statements that should land in files.
- Use `get_console()` from `src/utils/console_service.py` for user-facing Rich output.
- Include context in log messages (`logger.warning("GPU fallback: %s", reason)`).
- Never log raw student data; redact IDs in error paths.

## 4. Error Handling

- Catch only the exceptions you can recover from.
- Re-raise with context when propagating (`raise ConstraintError("...") from exc`).
- For CLI fatal errors, log + print + exit with non-zero code using `sys.exit(1)`.
- Use custom exceptions in `src/exceptions.py` (e.g., `ConfigValidationError`, `DataIntegrityError`).

## 5. Configuration Access

- Call `get_config()` once per module; store result in module-level constant.
- Never mutate config objects; treat as read-only dataclasses.
- Expose any feature toggles via config rather than `Environment` variables.

```python
CONFIG = get_config()

if not CONFIG.rl.enabled:
    return
```

## 6. Randomness & Reproducibility

- Source RNGs from `random.Random(seed)` instances passed through call stacks.
- For NumPy/PyTorch, set seeds via config-provided value when initializing components.
- Record seeds in manifests and logs for reproducibility.

## 7. Dependency Management

- Add runtime dependencies to `pyproject.toml`; run `uv sync --frozen` to refresh lockfile.
- Avoid optional imports sprinkled through code; gate heavy dependencies via config but keep imports at top to fail fast.
- Pin CUDA toolkit versions explicitly (PyTorch 2.4.1+cu121 already pinned).

## 8. Testing Guidelines

- **Directory parity:** tests mirror `src/` layout.
- **Fixtures:** prefer `pytest` fixtures stored in `test/conftest.py` or feature-specific modules.
- **Property tests:** use Hypothesis where helpful (e.g., constraint invariants).
- **GPU tests:** mark with `@pytest.mark.gpu` and skip automatically when CUDA missing.
- **Snapshot tests:** store under `test/data/` with descriptive filenames.

```bash
pytest test/unit/test_ga_scheduler.py -k "not gpu"
```

## 9. Documentation & Comments

- Use Google-style docstrings for public APIs; keep inline comments scarce and purposeful.
- Update `docs/` whenever adding modules or configs; include cross-links in `docs/00-INDEX.md`.
- Include diagrams (Mermaid) for non-trivial flows.

## 10. Performance Considerations

- Avoid Python-level loops inside GPU evaluators; rely on vectorized PyTorch ops.
- Prefer generators when iterating over large populations to keep memory down.
- Batch file I/O (e.g., exporting plots) to reduce disk churn.
- Profile with `scripts/diagnostics/profile_ga.py` before micro-optimizing.

## 11. Review Checklist

Before opening a PR:
1. `black src/ test/`
2. `ruff check src/ test/`
3. `pytest test/unit/`
4. `uv run verify-config`
5. Update docs/tests where behavior changed
6. Append manifest sample if new runtime mode introduced

Consistent adherence to these standards ensures contributors can reason about the codebase quickly and ship reliable scheduling improvements.
