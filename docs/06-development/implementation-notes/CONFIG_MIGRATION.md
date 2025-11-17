## [2025-10-26] Full migration to YAML-based configuration system

**Files Created:**
- `config/models.py` - Pydantic models for type-safe config
- `config/loader.py` - Config loader with environment detection
- `configs/test.yaml` - Quick test config (10 gen, 4 pop)
- `configs/dev.yaml` - Development config (50 gen, 8 pop)
- `configs/prod.yaml` - Production config (100 gen, 50 pop)

**Files Modified:**
- `main.py` - Added CLI args (--env, --config), loads YAML config
- `src/workflows/standard_run.py` - Accepts config object, removed old imports
- `config/ga_params.py` - Backward compatibility shim (redirects to YAML)
- `config/__init__.py` - Added init_config() for global config

**Files Backed Up:**
- `config/ga_params.py.old` → `config/ga_params_orig.py.bak`
- `config/constraints.py` → `config/constraints.py.old` 
- `config/feasibility_config.py` → `config/feasibility_config.py.old`
- `config/time_config.py` → `config/time_config.py.old`

**Breaking Changes:**
- Config now loaded from YAML files instead of Python files
- Use `python main.py --env test|dev|prod` to select config
- Old imports still work via compatibility shims

**Benefits:**
✓ Comments in config files (YAML supports comments)
✓ Easy test/dev/prod switching
✓ No Python knowledge needed to edit configs
✓ Type validation via Pydantic
✓ Reproducible experiments (version control YAML)
