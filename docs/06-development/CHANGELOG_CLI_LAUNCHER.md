# Changelog - CLI Launcher System

## [2025-11-21] Unified CLI Launcher with Profile Support

**Type**: Enhancement  
**Impact**: User Experience, Developer Workflow  
**Status**: Complete

### Overview
Created unified CLI launcher system with clean command conventions and profile-based configuration inheritance (test/med/prod).

### Files Added
- `scripts/launcher.py` - Unified CLI entry point with profile routing
- `CLI_REFERENCE.md` - User-facing CLI documentation
- `.github/instructions/cli.instructions.md` - Developer instructions for CLI system

### Files Modified
- `pyproject.toml` - Updated [project.scripts] with new command structure
- `.github/copilot-instructions.md` - Added CLI convention section and references

### Command Convention

**Main Commands (0-9)**: Primary experiments
- `uv run nsga --test/med/prod` - NSGA-II experiments
- `uv run train-rl --test/med/prod` - RL training

**Helper Commands (a-z)**: Utilities
- `uv run diagnose` - System diagnostics
- `uv run clean` - Clean outputs
- `uv run list-experiments` - Show history

### Profile Hierarchy (DRY Principle)
```
base.yaml → test.yaml → med.yaml → prod.yaml
```

Each profile inherits from previous, adding only necessary overrides.

### Backward Compatibility
Legacy commands still work:
- `uv run baseline` → `uv run nsga --test`
- `uv run exp1` → `uv run nsga --test`

### Benefits
1. **Consistent UX**: Same command structure for all experiments
2. **Clear Naming**: Numbers for experiments, letters for utilities
3. **DRY Configs**: No duplication via inheritance
4. **Quick Testing**: `--test` profile for 2-minute smoke tests
5. **Production Ready**: `--prod` profile for full runs on VM

### Usage Examples
```bash
# Local development (smoke tests)
uv run nsga --test       # 2 min
uv run train-rl --test   # 5-10 min

# Medium validation
uv run nsga --med        # 30 min

# Production deployment
uv run nsga --prod       # 3-5 hours
uv run train-rl --prod   # 1-2 hours
```

### Next Steps
- Test launcher with all profiles
- Run RL smoke test
- Update thesis experiments guide
