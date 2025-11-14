# Pre-commit Hooks - Quick Reference

## ✅ Installation Complete

Pre-commit hooks are now installed and will run automatically before each commit.

---

## 🚀 Usage

### Automatic (Recommended)
Hooks run automatically when you commit:
```bash
git add .
git commit -m "Your commit message"
# Pre-commit hooks run automatically here
```

### Manual Checks

```bash
# Check all files
uv run pre-commit run --all-files

# Check only staged files
uv run pre-commit run

# Check specific hook
uv run pre-commit run ruff --all-files
```

---

## 🔧 What Gets Checked

### Ruff Linter
- ✓ Code style violations (PEP 8)
- ✓ Unused imports
- ✓ Unused variables
- ✓ Bare exceptions
- ✓ Common bugs

### Ruff Formatter
- ✓ Auto-formats code (Black-compatible)
- ✓ Consistent indentation
- ✓ Line length (88 chars)

### Built-in Hooks
- ✓ Trailing whitespace removal
- ✓ End-of-file newline fixing
- ✓ YAML/JSON/TOML syntax validation
- ✓ Large file detection (>1MB)
- ✓ Merge conflict detection

### YAML Linter
- ✓ Validates config files in `configs/`
- ✓ Line length checks
- ✓ Syntax validation

---

## 📋 Current Issues Found

From the last scan:

1. **Unused Variables** (4 files):
   - `src/core/ga_scheduler.py:1394` - `start_time`
   - `src/encoder/quantum_time_system.py:190` - `quanta_count`
   - `src/exporter/plot_convergence.py:75` - `generations`
   - `src/exporter/plot_detailed_constraints.py:53,221` - `min_value`

2. **Bare Exception**:
   - `setup-uv:200` - Should use specific exception type

3. **Notebook Issues**:
   - `data/archive/krishna/1.ipynb` - Duplicate function definition

---

## 🛠️ Fixing Issues

### Skip a Hook Temporarily
```bash
# Skip all hooks for one commit
git commit -m "Message" --no-verify

# Skip specific hook
SKIP=ruff git commit -m "Message"
```

### Update Hook Versions
```bash
uv run pre-commit autoupdate
```

### Clear Cache
```bash
uv run pre-commit clean
```

---

## ⚙️ Configuration

Edit `.pre-commit-config.yaml` to:
- Add/remove hooks
- Change excluded paths
- Adjust linter settings
- Update hook versions

**Current excludes:**
- `output/` - Generated files
- `notebook/` - Experimental notebooks
- `test/` - Test files (linting disabled)
- `__pycache__/` - Python cache
- `.venv/` - Virtual environment

---

## 🎯 Best Practices

### Before Committing
1. Run tests: `pytest test/unit/`
2. Check code: `uv run pre-commit run --all-files`
3. Fix any issues
4. Commit changes

### CI/CD Integration (Future)
```yaml
# .github/workflows/ci.yml
- name: Run pre-commit
  run: |
    pip install pre-commit
    pre-commit run --all-files
```

---

## 🐛 Troubleshooting

### "pre-commit: command not found"
```bash
# Always use: uv run pre-commit
uv run pre-commit install
```

### Hooks fail with Python version error
Check `.pre-commit-config.yaml`:
```yaml
default_language_version:
  python: python3.13  # Match your Python version
```

### Hooks are too slow
```bash
# Run only fast hooks
SKIP=ruff-format git commit -m "Message"
```

### Want to disable temporarily
```bash
# Uninstall hooks (can reinstall later)
uv run pre-commit uninstall

# Reinstall when ready
uv run pre-commit install
```

---

## 📚 Resources

- Pre-commit docs: https://pre-commit.com/
- Ruff docs: https://docs.astral.sh/ruff/
- Hook list: https://pre-commit.com/hooks.html

---

**Note**: Pre-commit hooks help maintain code quality but don't replace proper testing and code review!
