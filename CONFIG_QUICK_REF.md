# Configuration Quick Reference

## TL;DR

```bash
# Test profile (30 gens, 2-5 min)
uv run nsga --test

# Production profile (2000 gens, 3-5 hours)
uv run nsga --prod

# Custom config with test profile
python main.py --config my-config.yaml --env test
```

## Merge Order (Always)

```
base.yaml → [mode.yaml] → env.yaml
   ↓            ↓              ↓
 common     experiment     test/prod
 settings    specific       scaling
```

## What Goes Where

### `configs/base.yaml`
✅ Time system, I/O paths, parallel settings, constraint weights  
❌ ngen, pop_size (environment-specific)

### `configs/{category}/{mode}.yaml`
✅ Algorithm settings, killswitches, heuristic configs  
❌ ngen, pop_size (inherit from environment)

### `configs/{env}.yaml` (test/prod)
✅ ngen, pop_size, timeout scaling  
❌ Algorithm specifics (use mode configs)

## Common Commands

```bash
# Runtime mode + environment
python main.py --mode baseline --env test
python main.py --mode baseline --env prod

# Custom config + environment
python main.py --config path/to/config.yaml --env test

# Launcher shortcuts (recommended)
uv run baseline --test          # Mode 1
uv run repairs --test           # Mode 2
uv run heuristics --test        # Mode 3
uv run nsga --test              # Mode 4 (full NSGA-II)
uv run heuristic-roundrobin --test  # Mode 6
```

## Adding New Mode Config

```yaml
# configs/my-category/N-my-mode.yaml

# ✅ DO: Mode-specific settings
ga:
  cxpb: 0.75
  mutpb: 0.25
  use_adaptive_probabilities: false

# ✅ DO: Feature killswitches
repair:
  enabled: true
  
# ❌ DON'T: Environment-scalable values
# ga:
#   ngen: 2000      # WRONG - breaks test profile
#   pop_size: 200   # WRONG - breaks test profile
```

## Debugging Config Issues

```bash
# Check final merged config
python -c "
from src.config import load_config
import os
os.environ['ENVIRONMENT'] = 'test'
c = load_config()
print(f'ngen={c.ga.ngen}, pop={c.ga.pop_size}')
"

# Expected outputs:
# test: ngen=30, pop=10
# prod: ngen=2000, pop=200
```

## Critical Rules

1. **Set ENVIRONMENT before load_config()** - Timing matters!
2. **Mode configs inherit scaling** - Don't hardcode ngen/pop_size
3. **Environment layer always last** - Final override
4. **Use launcher shortcuts** - Easier than raw main.py

## Full Documentation

See `docs/02-user-guides/config-architecture.md` for complete guide.
