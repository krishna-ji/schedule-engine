# Structured Logging - Quick Reference

## Setup (Once at App Start)

```python
from src.utils.structured_logger import setup_logging

setup_logging(
    log_file=Path("logs/my_run.log"),  # Auto-generated if None
    console_level="INFO",               # Console: INFO, WARNING, ERROR
    file_level="DEBUG",                 # File: DEBUG (everything)
)
```

## Get Logger

```python
from src.utils.structured_logger import StructuredLogger

logger = StructuredLogger.get_logger(__name__)
```

## Context Management

```python
# Set once, applies to all subsequent messages
logger.set_context(env_rank=0, generation=5, step=10)

logger.info("Message")  
# → [env=0 gen=5 step=10] Message

# Update context
logger.set_context(generation=6, step=11)

# Clear context
logger.clear_context()
```

## Basic Logging

```python
logger.debug("Detailed debug info", variable=value)     # File only
logger.info("Key event", metric=123)                    # Console + file
logger.warning("Attention needed", code=42)             # Console + file
logger.error("Error occurred", exception="ValueError")  # Console + file
```

## Specialized Methods

```python
# Action logging (with visual indicator)
logger.action("heuristic_1", success=True, duration_ms=5.2)
# → ✓ Action heuristic_1 duration_ms=5.2

# Step summary (compact one-line format)
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
# → ✓ most_constrained_first r=0.850 best=64046.6 div=3257.0 stag=0 +1198714.6 2132.6ms

# Success logging
logger.success("Training completed", total_steps=1000)
# → ✓ Training completed total_steps=1000
```

## Output Examples

### Console (Rich-formatted, INFO level)
```
INFO     Environment reset total_steps_so_far=0
INFO     ✓ most_constrained_first r=0.850 best=64046.6 div=3257.0 stag=0 +1198714.6 2132.6ms
INFO     ✗ random_swap r=-0.100 best=64046.6 div=3257.0 stag=5  15.3ms
```

### File (Structured, DEBUG level)
```
[02:24:44.339] [INFO    ] [env=0 gen=5 step=10] Environment reset total_steps_so_far=0
[02:24:44.342] [DEBUG   ] [env=0 gen=5 step=10] Detailed debug info variable=value
[02:24:44.468] [INFO    ] [env=0 gen=10 step=5] ✓ most_constrained_first r=0.850 best=64046.6 div=3257.0 stag=0 +1198714.6 2132.6ms
```

## Symbols

- ✓ = Success
- ✗ = Failure
- Metrics: r=reward, div=diversity, stag=stagnation

## Best Practices

1. **Set context once** at start of loop/function
2. **Use specialized methods** (`action`, `step_summary`, `success`)
3. **DEBUG for details**, **INFO for key events**
4. **Pass kwargs** instead of formatting strings
5. **Clear context** when moving to new episode/run

## File Locations

- Training: `logs/training/train_YYYYMMDD_HHMMSS.log`
- Custom: Specify via `setup_logging(log_file=...)`

## Testing

```bash
python test_structured_logging.py
```

## Full Documentation

See `docs/02-user-guides/structured-logging-guide.md`
