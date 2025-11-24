# Configuration Architecture - Summary of Fixes

**Date**: November 24, 2025  
**Issue**: Config management and experiment handling not properly synchronized  
**Status**: ✅ Resolved

## Problems Identified

### 1. Environment Variable Timing Issue
**Problem**: `ENVIRONMENT` was set AFTER config loading, causing test/prod scaling to fail
```python
# ❌ WRONG ORDER
config = load_config(runtime_mode)  # Reads ENVIRONMENT here
os.environ["ENVIRONMENT"] = args.env  # Too late!
```

**Solution**: Set environment FIRST in main.py (line ~115-120)
```python
# ✅ CORRECT ORDER
os.environ["ENVIRONMENT"] = args.env or "test"  # Set first
config = load_config(runtime_mode)  # Now sees correct value
```

### 2. Mode Configs Overriding Environment
**Problem**: Hardcoded `ngen: 2000` in round-robin config prevented test profile (30 gens)

**Solution**: 
1. Removed `ngen` and `pop_size` from mode configs (inherit from environment)
2. Updated config loader to ALWAYS apply environment layer last
3. Added clear comments in configs about inheritance

### 3. Config Priority Confusion
**Problem**: Multiple config paths (--mode, --config, SCHEDULE_CONFIG, default) without clear hierarchy

**Solution**: Documented clear priority system with consistent environment override:
```
Priority 1: --mode baseline → base + mode + env
Priority 2: --config path  → base + custom + env
Priority 3: SCHEDULE_CONFIG → base + $config + env
Priority 4: default        → base + env
```

### 4. Inconsistent Naming
**Problem**: "profile" (launcher) vs "environment" (main.py) vs "env" (CLI)

**Solution**: Clarified terminology:
- **Environment**: test/prod (YAML files)
- **Profile**: CLI alias for environment (--test, --prod)
- **Mode**: Runtime mode (baseline, rl-guided, etc.)

### 5. Missing Synchronization with ExperimentManager
**Problem**: ExperimentManager expected runtime_mode but config could be loaded without it

**Solution**: Made flow explicit in documentation and ensured consistent patterns

## Files Modified

### 1. `main.py`
**Changes**:
- Moved `ENVIRONMENT` setting before config loading (line ~115)
- Added default environment fallback: `os.environ["ENVIRONMENT"] = args.env or "test"`
- Added descriptive logging for config loading path
- Improved comments explaining merge order

### 2. `src/config/loader.py`
**Changes**:
- Enhanced docstring with complete architecture explanation
- Added clear section on merge order and priority paths
- Fixed Priority 2 (explicit config path) to ALWAYS apply environment override
- Improved logging messages showing merge chain

### 3. `configs/hybrid/6-roundrobin.yaml`
**Changes**:
- Removed hardcoded `ngen: 2000` and `pop_size: 200`
- Added comment: `# ngen and pop_size inherit from environment (test/prod)`
- Kept only mode-specific settings (cxpb, mutpb, heuristic configs)

### 4. `src/ga/hybrid_population.py`
**Changes**:
- Fixed `greedy_count = max(1, ...)` to allow 0 when `greedy_percent = 0.0`
- Changed to: `greedy_count = int(n * greedy_percent)`
- Allows construction heuristics to be completely disabled

### 5. `docs/02-user-guides/config-architecture.md` (NEW)
**Created**: Comprehensive guide covering:
- Three-layer merge strategy
- Priority paths and flow diagrams
- Common pitfalls and solutions
- Validation and testing procedures
- Future improvements roadmap

## Verification Results

### Test 1: Round-Robin Test Profile
```bash
$ uv run heuristic-roundrobin --test
Loading: configs/hybrid/6-roundrobin.yaml + test.yaml
genetic algorithm: 30 gen x 10 pop | cx=0.75 mut=0.25
parallel mode: 32 workers
Hybrid initialization: 0 greedy, 8 smart, 2 random
✅ PASS - Correct test profile (30 gens, 10 pop)
```

### Test 2: Baseline Mode Test Profile
```bash
$ python main.py --mode baseline --env test
Loading: 1-pure-nsga + test.yaml
genetic algorithm: 30 gen x 10 pop | cx=0.75 mut=0.25
parallel mode: 32 workers
✅ PASS - Correct test profile
```

### Test 3: Environment Variable Fallback
```bash
$ python main.py --mode baseline  # No --env specified
Loading: 1-pure-nsga + test.yaml  # Defaults to test
genetic algorithm: 30 gen x 10 pop
✅ PASS - Correct default environment
```

## Architecture Improvements

### Before (Fragile)
```
┌──────────────────────────────────────┐
│ Multiple config paths                │
│ • Environment set too late           │
│ • Mode configs override environment  │
│ • No clear merge hierarchy           │
│ • Inconsistent terminology           │
└──────────────────────────────────────┘
```

### After (Robust)
```
┌──────────────────────────────────────┐
│ Clear Three-Layer Architecture       │
│ ✅ Environment set early (main.py)   │
│ ✅ Environment ALWAYS applies last   │
│ ✅ Documented priority system        │
│ ✅ Consistent naming convention      │
│ ✅ Mode configs inherit scaling      │
└──────────────────────────────────────┘
```

## Key Principles Established

1. **DRY Configuration**: No duplication - test/prod scaling in one place
2. **Clear Hierarchy**: base → mode → environment (always)
3. **Environment First**: Set `ENVIRONMENT` before loading config
4. **No Hardcoding**: Mode configs never hardcode scalable values
5. **Explicit Over Implicit**: Clear logging shows merge chain

## Usage Patterns

### Recommended: Launcher Shortcuts
```bash
# Test profiles (fast smoke tests)
uv run nsga --test
uv run heuristic-roundrobin --test

# Production profiles (full runs)
uv run nsga --prod
uv run heuristic-roundrobin --prod
```

### Advanced: Direct main.py
```bash
# Runtime mode with environment
python main.py --mode baseline --env prod

# Custom config with environment
python main.py --config custom.yaml --env test

# Environment variable approach
export ENVIRONMENT=prod
python main.py --mode baseline
```

## Impact

### Before Fixes
- ❌ Test profiles didn't work (always ran 2000 gens)
- ❌ Confusion about which config takes precedence
- ❌ Environment set too late (no effect)
- ❌ Multiprocessing not documented properly

### After Fixes
- ✅ Test profiles work correctly (30 gens, 10 pop)
- ✅ Clear priority system documented
- ✅ Environment set early and consistently
- ✅ Multiprocessing enabled in all profiles
- ✅ Comprehensive architecture guide created

## Future Maintenance

### Adding New Runtime Modes
1. Create config in `configs/{category}/{N-name}.yaml`
2. Define RuntimeMode enum entry with path
3. **DO NOT** hardcode `ngen` or `pop_size`
4. Use comments to document inheritance
5. Test with both `--test` and `--prod` profiles

### Adding New Environments
1. Create `configs/{env}.yaml`
2. Define scaling parameters (ngen, pop_size, timeouts)
3. Update documentation
4. Test with multiple runtime modes

### Testing Config Changes
```bash
# Quick validation
python -c "from src.config import load_config; import os; os.environ['ENVIRONMENT']='test'; c = load_config(); print(f'ngen={c.ga.ngen}, pop={c.ga.pop_size}')"

# Expected: ngen=30, pop=10 for test
# Expected: ngen=2000, pop=200 for prod
```

## Related Documentation

- **User Guide**: `docs/02-user-guides/config-architecture.md` - Complete architecture reference
- **Runtime Modes**: `docs/02-user-guides/runtime-modes.md` - Available experiment modes
- **CLI Reference**: `CLI_REFERENCE.md` - Command-line interface guide
- **Config Models**: `src/config/models.py` - Pydantic schemas
- **Config Loader**: `src/config/loader.py` - Merge logic implementation

## Conclusion

The configuration architecture is now **properly synchronized** with clear:
- ✅ Layered merge strategy (base → mode → environment)
- ✅ Environment timing (set before config load)
- ✅ Inheritance rules (mode configs don't override scaling)
- ✅ Priority system (--mode, --config, env var, default)
- ✅ Comprehensive documentation

The system follows DRY principles while maintaining flexibility for both quick testing and production experiments.
