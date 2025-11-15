# Quick Start - Simplified Config System

## Running the Engine

### Fast Smoke Test (5-10 minutes)
```bash
uv run test
```

### Maximum Quality Production (24-48 hours, requires 16+ cores, 32+ GB RAM)
```bash
uv run prod
```

## Configuration Files

All configs are in `configs/`:

- `base.yaml` - Common settings (shared by all)
- `test.yaml` - Overrides for smoke test (30 gens, 10 pop)
- `prod.yaml` - Overrides for maximum quality (2000 gens, 200 pop)

## Editing Configurations

### To change common settings (e.g., constraint weights):
Edit `configs/base.yaml`

### To change environment-specific settings (e.g., generation count):
Edit the specific environment file (`test.yaml` or `prod.yaml`)

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
python main.py --env test

# Using custom config file
python main.py --config path/to/custom.yaml
```

## Configuration Comparison

| Setting | Test | Prod |
|---------|------|------|
| **Generations** | 30 | 2000 |
| **Population** | 10 | 200 |
| **Parallel** | No | Yes |
| **Runtime** | 5-10 min | 24-48 hrs |
| **Exhaustive triggers** | 2 | 10 |
| **Hardware** | Any | 16+ cores |
| **RAM needed** | 4 GB | 32+ GB |
