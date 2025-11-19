# Standalone Configuration System

**Date**: 2025-10-27  
**Change**: Eliminated `common.yaml` inheritance - all configs are now standalone

## What Changed

### Before (Inheritance System)
- `common.yaml` contained base settings
- `test.yaml`, `dev.yaml`, `prod.yaml` only had ngen/pop_size/parallel overrides
- Config loader used `deep_merge()` to combine base + environment configs
- **Problems**:
  - Worker processes loaded wrong configs (`dev` instead of `test`)
  - Validation errors when using `--config` flag (no merge)
  - Confusing two-mode loading (standalone vs merged)

### After (Standalone System)
- **NO `common.yaml`** (renamed to `.OLD_DEPRECATED`)
- Each config file contains **ALL settings** (complete/standalone)
- Config loader is simple - just loads the YAML file directly
- **Benefits**:
  -  No merge confusion
  -  Worker processes load correct configs
  -  Same behavior whether using `--env` or `--config`
  -  Easy to understand and modify

## Config Files

### `configs/test.yaml` (Standalone)
- **Purpose**: Fast smoke test (5-10 min)
- **Settings**: 30 gens, 10 pop, no multiprocessing, 2 IGLS triggers [3, 25]

### `configs/dev.yaml` (Standalone)
- **Purpose**: Medium development runs (2-4 hours)
- **Settings**: 200 gens, 50 pop, multiprocessing ON, 4 IGLS triggers [3, 30, 100, 180]

### `configs/prod.yaml` (Standalone)
- **Purpose**: Full production quality (12-24 hours)
- **Settings**: 500 gens, 100 pop, multiprocessing ON, 6 IGLS triggers [3, 30, 100, 200, 350, 480]

### `configs/prod_test.yaml` (Standalone)
- **Purpose**: Scale test before production (30-60 min)
- **Settings**: 30 gens, 100 pop, multiprocessing ON, 2 IGLS triggers [3, 25]

## How to Use

### Option 1: Environment Flag (Recommended)
```bash
python main.py --env test    # Uses configs/test.yaml
python main.py --env dev     # Uses configs/dev.yaml
python main.py --env prod    # Uses configs/prod.yaml
```

### Option 2: Direct Config Path
```bash
python main.py --config configs/test.yaml
python main.py --config configs/prod_test.yaml
```

### Option 3: Environment Variable
```powershell
$env:ENVIRONMENT='test'; python main.py
$env:SCHEDULE_CONFIG='configs/prod_test.yaml'; python main.py
```

## Config Loader Priority

```
1. --config flag            → Load specified file
2. SCHEDULE_CONFIG env var  → Load specified file
3. ENVIRONMENT env var      → Load configs/{ENVIRONMENT}.yaml
4. Default                  → Load configs/dev.yaml
5. Fallback                 → Built-in defaults
```

## Making Changes

To modify settings for a specific environment:

1. **Open the config file** (e.g., `configs/test.yaml`)
2. **Edit the values** directly (no need to check common.yaml)
3. **Save and run** - changes take effect immediately

Example: Change test population to 20
```yaml
# configs/test.yaml
ga:
  ngen: 30
  pop_size: 20  # Changed from 10
  cxpb: 0.75
  # ... rest of settings
```

## Removed Files

- `configs/common.yaml` → Renamed to `configs/common.yaml.OLD_DEPRECATED`
- Loader's `deep_merge()` function → Removed
- Loader's merge logic → Removed

## Migration Notes

If you created custom configs that relied on `common.yaml` inheritance:
1. Copy ALL settings from `common.yaml.OLD_DEPRECATED` into your custom config
2. Override the specific values you need (ngen, pop_size, etc.)
3. Your config is now standalone and will work with any loading method
