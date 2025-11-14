# Quick Start - Simplified Config System

## Running the Engine

### Fast Smoke Test (5-10 minutes)
```bash
uv run test
```

### Standard Production (4-8 hours)
```bash
uv run notprod
```

### Maximum Quality (24-48 hours, requires 16+ cores, 32+ GB RAM)
```bash
uv run prod
```

## Configuration Files

All configs are in `configs/`:

- `base.yaml` - Common settings (shared by all)
- `test.yaml` - Overrides for smoke test (30 gens, 10 pop)
- `notprod.yaml` - Overrides for standard production (400 gens, 80 pop)
- `prod.yaml` - Overrides for maximum quality (2000 gens, 200 pop)

## Editing Configurations

### To change common settings (e.g., constraint weights):
Edit `configs/base.yaml`

### To change environment-specific settings (e.g., generation count):
Edit the specific environment file (`test.yaml`, `notprod.yaml`, or `prod.yaml`)

### How inheritance works:
1. Load all settings from `base.yaml`
2. Override with environment-specific settings
3. Result: merged configuration

Example:
```yaml
# base.yaml has:
ga:
  cxpb: 0.75
  mutpb: 0.25
  ngen: 100  # default (ignored)

# prod.yaml has:
ga:
  ngen: 2000
  pop_size: 200

# Result: cxpb=0.75, mutpb=0.25, ngen=2000, pop_size=200
```

## Alternative Run Methods

```bash
# Using Python directly
python main.py --env prod
python main.py --env notprod
python main.py --env test

# Using custom config file
python main.py --config path/to/custom.yaml

# Using wrapper scripts
python run_prod.py
python run_notprod.py
python run_test.py
```

## Configuration Comparison

| Setting | Test | NotProd | Prod |
|---------|------|---------|------|
| **Generations** | 30 | 400 | 2000 |
| **Population** | 10 | 80 | 200 |
| **Parallel** | No | Yes | Yes |
| **Runtime** | 5-10 min | 4-8 hrs | 24-48 hrs |
| **Exhaustive triggers** | 2 | 5 | 10 |
| **Hardware** | Any | 4-8 cores | 16+ cores |
| **RAM needed** | 4 GB | 8-16 GB | 32+ GB |
