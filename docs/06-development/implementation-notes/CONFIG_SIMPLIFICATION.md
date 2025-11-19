## [2025-11-14] Documentation cleanup and config system update

### Files Changed
- `.github/copilot-instructions.md` - Removed massive duplication (was 3x repeated), updated to reflect new config system
- `.github/instructions/config.instructions.md` - Recreated cleanly, updated for base.yaml inheritance
- `.github/instructions/README.md` - Path-specific instructions guide
- `docs/code/CONFIG_SIMPLIFICATION.md` - This file
- `docs/CONFIG_QUICKSTART.md` - Quick reference guide

### Issues Fixed
1. **copilot-instructions.md** - Was ~3000 lines with 3x duplication, now ~150 lines clean
2. **config.instructions.md** - Had corrupted frontmatter and duplication, now clean with proper applyTo
3. **Updated references** - Changed all `dev` → `notprod`, documented base.yaml + inheritance
4. **UV commands** - Documented entry point functions in main.py (no separate wrapper files needed)

### Configuration System Summary
- **Before**: 7 standalone config files with duplication
- **After**: 1 base.yaml + 3 environment overrides (test/notprod/prod)
- **Commands**: `uv run prod/notprod/test` work via entry points in main.py
- **Inheritance**: Deep merge of base.yaml + environment file via loader.py

---

# Configuration Simplification Summary (Original)

## Changes Made

### 1. New Configuration Structure

Simplified from 7 config files to just 4:

**Before:**
- `common.yaml` - shared settings
- `test.yaml` - smoke test (standalone)
- `dev.yaml` - development (standalone)
- `prod.yaml` - production (standalone)
- `prod_safe.yaml` - VM-optimized prod
- `prod_test.yaml` - scale test
- `igls_test.yaml` - IGLS testing

**After:**
- `base.yaml` - all common settings (shared by all environments)
- `test.yaml` - smoke test (5-10 min, 30 gens, 10 pop)
- `notprod.yaml` - medium quality (4-8 hours, 400 gens, 80 pop)
- `prod.yaml` - best quality, no constraints (24-48 hours, 2000 gens, 200 pop)

### 2. Configuration Inheritance

All environment configs now inherit from `base.yaml` and only override what differs:

```yaml
# base.yaml contains all common settings
# prod.yaml only contains:
name: "Production - Best Quality (No Constraints)"
environment: prod
description: "Maximum quality production run (2000 gens, 200 pop, 24-48h runtime)"

ga:
  ngen: 2000
  pop_size: 200

parallel:
  use_multiprocessing: true
  num_workers: null

repair:
  exhaustive_search:
    generations: [3, 30, 100, 200, 350, 500, 800, 1200, 1600, 1900]
    # ... other repair overrides
```

### 3. UV Run Commands

Added simple `uv run` commands for each environment:

```bash
uv run prod      # Best quality (2000 gens, 200 pop, 24-48h)
uv run notprod   # Medium quality (400 gens, 80 pop, 4-8h)
uv run test      # Smoke test (30 gens, 10 pop, 5-10 min)
```

Implemented via entry point functions in `main.py`:
- `main_prod()` - sets `ENVIRONMENT=prod` and calls `main()`
- `main_notprod()` - sets `ENVIRONMENT=notprod` and calls `main()`
- `main_test()` - sets `ENVIRONMENT=test` and calls `main()`

These are registered in `pyproject.toml` as `[project.scripts]` entries.

**No separate wrapper files needed!** Everything is in `main.py`.

### 4. Updated Code

**src/config/loader.py:**
- Added `_deep_merge()` function for config inheritance
- Updated `load_config()` to merge `base.yaml` with environment configs
- Changed default from `dev` to `notprod`

**src/config/models.py:**
- Changed valid environments from `["test", "dev", "prod"]` to `["test", "notprod", "prod"]`
- Changed default environment from `"dev"` to `"notprod"`

**main.py:**
- Updated `--env` choices to `["test", "notprod", "prod"]`
- Simplified help text to match new environment names

**pyproject.toml:**
- Added project scripts for `prod`, `notprod`, and `test` commands

### 5. Configuration Details

| Config | Generations | Population | Parallel | Runtime | Use Case |
|--------|-------------|------------|----------|---------|----------|
| test | 30 | 10 | No | 5-10 min | Smoke testing |
| notprod | 400 | 80 | Yes | 4-8 hours | Standard production |
| prod | 2000 | 200 | Yes | 24-48 hours | Best quality, no constraints |

### 6. Benefits

1. **Simpler structure** - Only 4 config files instead of 7
2. **Less duplication** - Common settings in one place (base.yaml)
3. **Easy to maintain** - Changes to common settings propagate automatically
4. **Clear naming** - `notprod` clearly indicates "not full production" vs ambiguous "dev"
5. **Easy to run** - Simple `uv run prod/notprod/test` commands
6. **No constraints on prod** - Best possible settings for maximum quality

### 7. Usage Examples

```bash
# Quick smoke test (5-10 minutes)
uv run test

# Standard production run (4-8 hours)
uv run notprod

# Maximum quality run (24-48 hours, requires good hardware)
uv run prod

# Or use explicit config path
python main.py --config configs/prod.yaml

# Or use environment flag
python main.py --env prod
```

### 8. Verification

All configurations verified to load correctly:
-  prod: ngen=2000, pop=200, parallel=True
-  notprod: ngen=400, pop=80, parallel=True
-  test: ngen=30, pop=10, parallel=False

All inherit from base.yaml successfully with proper deep merging.
