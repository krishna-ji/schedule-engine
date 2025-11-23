# CLI Consolidation Summary (Nov 21, 2025)

## Problem
Had 3 CLI files with overlapping functionality:
- `scripts/launcher.py` (NEW) - Unified CLI with profiles
- `scripts/cli.py` (OLD) - Entry point registry  
- `scripts/interactive_launcher.py` (OLD) - Interactive TUI menu

## Solution

###  Consolidated into ONE file: `launcher.py`

**Merged functionality:**
1. Profile-based commands (`--test/med/prod`)
2. Helper commands (diagnose, clean, list)
3. Interactive TUI menu (merged from interactive_launcher.py)

###  Kept: `cli.py`

**Purpose**: Entry point registry for utility scripts
- Not redundant - provides organized access to all utility scripts
- Used by launcher.py for `diagnose_gpu()` etc.
- Clean separation: launcher = UX, cli = registry

###  Deleted: `interactive_launcher.py`

**Reason**: Functionality merged into launcher.py

## New Command Structure

### Main Commands (direct)
```bash
uv run nsga --test/med/prod      # NSGA-II experiments
uv run train-rl --test/med/prod  # RL training
```

### Helper Commands
```bash
uv run diagnose           # Quick system check (calls diagnose_gpu via cli.py)
uv run clean              # Clean output dir
uv run list-experiments   # Show history
```

### Interactive Launcher (TUI)
```bash
uv run launcher   # Interactive menu
uv run run        # Same (alias)
```

### Utility Scripts (via cli.py)
```bash
uv run diagnose-gpu       # Full GPU diagnostics
uv run benchmark-gpu      # GPU benchmarking
uv run verify-config      # Config validation
uv run check-data         # Data integrity
```

## File Purposes

| File | Purpose | Keep? |
|------|---------|-------|
| `launcher.py` | Unified CLI + interactive menu |  YES |
| `cli.py` | Entry point registry for utilities |  YES |
| `interactive_launcher.py` | Old interactive menu |  DELETED |

## Benefits

1. **Single source of truth**: One launcher file
2. **Clean separation**: UX (launcher.py) vs Registry (cli.py)  
3. **No duplication**: Interactive menu integrated
4. **Backward compatible**: All old commands still work
5. **DRY**: Profile hierarchy (test < med < prod)

## Testing

```bash
 uv sync                    # Reload scripts
 uv run launcher            # Interactive menu works
 uv run nsga --test         # Profile commands work
 uv run diagnose            # Helper commands work
 uv run diagnose-gpu        # Utility scripts work via cli.py
```

## Summary

**Before**: 3 overlapping files, confusion  
**After**: 1 unified launcher + 1 utility registry, clarity

All commands working, backward compatible, cleaner codebase.
