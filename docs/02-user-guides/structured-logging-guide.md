# Structured Logging System - User Guide

## Overview

The Schedule Engine now features a unified, structured logging system that provides:

- **Clean console output**: Compact, one-line messages with Rich formatting
- **Comprehensive file logs**: Everything written to timestamped log files
- **Context awareness**: Automatic prefixing with `[env=X gen=Y step=Z]`
- **Reduced redundancy**: No more repeated "generation=9" in every message
- **Visual hierarchy**: Clear success/error indicators and color-coded output

## Quick Start

### Basic Usage

```python
from src.utils.structured_logger import StructuredLogger, setup_logging

# Initialize logging (call once at application start)
setup_logging(
    log_file=Path("logs/my_run.log"),  # Optional: auto-generated if None
    console_level="INFO",  # Console verbosity
    file_level="DEBUG",    # File verbosity (more detailed)
)

# Get logger instance
logger = StructuredLogger.get_logger(__name__)

# Log messages
logger.info("Training started", total_steps=10000)
logger.debug("Detailed debug info")  # Only in file, not console
logger.warning("Something needs attention", code=42)
logger.error("An error occurred", exception="ValueError")
```

### Context Management

Set context once and it persists across all subsequent messages:

```python
# Set context (automatically prefixes all messages)
logger.set_context(env_rank=0, generation=5, step=10)

logger.info("Action selected")  # Output: [env=0 gen=5 step=10] Action selected
logger.info("Action completed") # Output: [env=0 gen=5 step=10] Action completed

# Update context
logger.set_context(generation=6, step=11)
logger.info("Next step")  # Output: [env=0 gen=6 step=11] Next step

# Clear context
logger.clear_context()
logger.info("Message without context")  # Output: Message without context
```

### Specialized Logging Methods

#### Action Logging
```python
logger.action("heuristic_1", success=True, duration_ms=5.2)
# Output: ✓ Action heuristic_1 duration_ms=5.2

logger.action("heuristic_2", success=False, error="timeout")
# Output: ✗ Action heuristic_2 error=timeout
```

#### Step Summary (Compact One-Line Format)
```python
logger.step_summary(
    action="most_constrained_first",
    reward=0.85,
    success=True,
    best_fitness=64046.6,
    diversity=3257.03,
    stagnation=0,
    duration_ms=2132.62,
    improvement=1198714.6,  # Optional
)
# Output: ✓ most_constrained_first r=0.850 best=64046.6 div=3257.0 stag=0 +1198714.6 2132.6ms
```

#### Success Messages
```python
logger.success("Training completed", total_steps=1000)
# Output: ✓ Training completed total_steps=1000
```

## Output Examples

### Before (Old System)
```
INFO     [ENV 0] Reset called (total steps so far: 0)
INFO     [env=0 gen=0 step=0] Environment reset complete (seed=None, pop=10, duration=4.49 ms)
INFO     [env=0 gen=0 step=0] Selected action=adaptive_large_neighborhood (11); generation=0, prev_fitness=(12516.00, 11161.20)
ERROR    Action unknown failed: 'str' object cannot be interpreted as an integer
INFO     [env=0 gen=0 step=0] Action adaptive_large_neighborhood success=False
INFO     [env=0 gen=0 step=0] Step summary: action=adaptive_large_neighborhood reward=0.0000 success=False best=1262761.20 diversity=67.0604 stagnation=0 duration=7.36 ms IMPROVED by inf
                    apply_adaptive_large_neighborhood=3ms | calculate_diversity=82µs | calculate_reward=40µs
```

### After (New System)
```
INFO     Environment reset total_steps_so_far=0
DEBUG    Reset complete seed=None population=10 duration_ms=4.49
DEBUG    Applying action action=adaptive_large_neighborhood action_id=11 prev_fitness=(12516.00, 11161.20)
ERROR    Action unknown failed: 'str' object cannot be interpreted as an integer
INFO     ✗ Action adaptive_large_neighborhood
INFO     ✗ adaptive_large_neighborhood r=0.000 best=1262761.2 div=67.1 stag=0  7.4ms
```

**Key improvements:**
- ✅ Context `[env=0 gen=0 step=0]` shown automatically (no repetition)
- ✅ One compact line per key event
- ✅ Visual indicators (✓/✗) for success/failure
- ✅ Consistent metric abbreviations (r=reward, div=diversity, stag=stagnation)
- ✅ DEBUG messages only in file (not flooding console)

## Configuration

### Console vs File Logging

- **Console**: Clean, minimal output at INFO level
  - Shows only important events (actions, summaries, errors)
  - Uses Rich formatting (colors, symbols)
  - Perfect for monitoring progress

- **File**: Comprehensive, detailed output at DEBUG level
  - Captures everything including debug messages
  - Structured format with timestamps
  - Ideal for post-run analysis

### Verbosity Levels

```python
# Production: Clean console, detailed file
setup_logging(console_level="INFO", file_level="DEBUG")

# Development: Verbose console for debugging
setup_logging(console_level="DEBUG", file_level="DEBUG")

# Quiet: Errors only on console
setup_logging(console_level="ERROR", file_level="DEBUG")
```

## Integration with Existing Code

The structured logging system has been integrated into:

- ✅ `src/rl/gym_env/schedule_env.py` - RL environment logging
- ✅ `src/rl/training/train_script.py` - Training script initialization
- ✅ `src/utils/system_info.py` - System diagnostics (single-line output)

### Migration Guide

If you have existing code using the old logging system:

**Old:**
```python
import logging
logger = logging.getLogger(__name__)

logger.info(f"[env={env} gen={gen}] Action {action} success={success}")
```

**New:**
```python
from src.utils.structured_logger import StructuredLogger
logger = StructuredLogger.get_logger(__name__)

logger.set_context(env_rank=env, generation=gen)
logger.action(action, success=success)
```

## File Locations

- **Training logs**: `logs/training/train_YYYYMMDD_HHMMSS.log`
- **Test logs**: `logs/test_structured_logging.log`
- **Custom logs**: Specified via `setup_logging(log_file=...)`

## Testing

Run the test script to verify logging functionality:

```bash
python test_structured_logging.py
```

This will:
1. Create `logs/test_structured_logging.log`
2. Demonstrate all logging features
3. Show console vs file output differences

## Benefits

### For Users
- **Readable logs**: Easy to scan and understand progress
- **Less noise**: Important events stand out
- **File retention**: Everything saved for later analysis

### For Developers
- **Consistent format**: Standardized across codebase
- **Context management**: Automatic prefixing reduces boilerplate
- **Type-safe**: Strong typing with Pydantic-style validation

### For Debugging
- **Complete history**: All DEBUG messages in files
- **Structured format**: Easy to parse programmatically
- **Timestamps**: Precise timing information

## Advanced Features

### Custom Context Fields

```python
logger.set_context(
    env_rank=0,
    generation=5,
    experiment_id="exp001",
    model_version="v2.1"
)
```

### Profiling Integration

```python
# Performance metrics automatically included in step summaries
logger.step_summary(
    action="heuristic_1",
    reward=0.5,
    success=True,
    best_fitness=1000.0,
    diversity=50.0,
    stagnation=3,
    duration_ms=250.5,
    # Additional metrics
    memory_mb=512.3,
    gpu_util=85.2,
)
```

### Exception Logging

```python
try:
    risky_operation()
except Exception as e:
    logger.error("Operation failed", exception=str(e), exc_info=True)
    # exc_info=True includes full traceback in file
```

## Best Practices

1. **Set context once**: At the start of a loop or function
2. **Use specialized methods**: `action()`, `step_summary()`, `success()` over generic `info()`
3. **Keep console clean**: Use `logger.debug()` for verbose details
4. **Structure kwargs**: Pass metrics as keyword arguments, not formatted strings
5. **Clear context**: When context changes (e.g., new episode)

## Troubleshooting

### Logs not appearing in file
- Check file path is correct
- Ensure `setup_logging()` was called
- Verify logger name hierarchy (should start with `schedule_engine`)

### Too much console output
- Increase console level: `console_level="WARNING"`
- Use `logger.debug()` instead of `logger.info()` for verbose messages

### Missing context in logs
- Call `logger.set_context()` before logging
- Verify context isn't cleared prematurely

## Future Enhancements

- [ ] Log rotation (automatic file management)
- [ ] Remote logging (send logs to monitoring service)
- [ ] Structured JSON output (for programmatic analysis)
- [ ] Performance dashboard (real-time log visualization)
