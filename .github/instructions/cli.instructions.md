# CLI Launcher Instructions

**Applies to**: `scripts/launcher.py`, `CLI_REFERENCE.md`, `pyproject.toml` [project.scripts]

## Overview

The unified CLI launcher system provides a clean, consistent interface for running experiments with profile-based configuration inheritance (DRY principle).

## Command Convention

### Main Commands (0-9)
**Purpose**: Primary experimental workflows (GA, RL training)  
**Naming**: Numbers for easy memorization  
**Examples**:
- `nsga` - NSGA-II genetic algorithm (unified launcher)
- `train-rl` - RL agent training

### Progressive Mode Experiments (A→E)
**Purpose**: Systematic ablation study with increasing complexity  
**Convention**: Alphabetic progression (A=baseline → E=full RL)  
**Modes**:
- Mode A: `baseline` - Pure NSGA-II (no repairs, no heuristics)
- Mode B: `memetic` - + Memetic local search
- Mode C: `roundrobin` - + Round-robin heuristics
- Mode D: `adaptive` - + Adaptive heuristic selection
- Mode E: `rl` - + RL-guided control (full deployment)

### Helper Commands (a-z)
**Purpose**: Utilities and diagnostics  
**Naming**: Descriptive lowercase letters  
**Examples**:
- `diagnose` - System/GPU/config diagnostics
- `clean` - Clean output directory
- `list-experiments` - Show experiment history

## Profile System

**Two-tier hierarchy** (DRY principle):
```
BaseConfig (shared defaults)
  ↓
--test (TestConfig: 30 gens, 10 pop, ~2-10 min)
--prod (ProdConfig: 2000 gens, 400 pop, ~1-5 hours)
```

**Implementation**:
- Profiles passed as CLI flags: `--test`, `--prod`
- Configs inherit via Python dataclass system
- Launcher maps profile → experiment config instantiation

## File Structure

### scripts/launcher.py
**Purpose**: Unified CLI entry point with profile routing

**Structure**:
```python
def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with profile support."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['test', 'prod'])
    parser.add_argument('--test', action='store_const', const='test')
    parser.add_argument('--prod', action='store_const', const='prod')
    return parser

def main_baseline():
    """Mode A: Pure NSGA-II."""
    args = parse_args()
    profile = _resolve_profile(args.profile)
    # Route to main.py with experiment_a

def main_train_rl():
    """RL training launcher."""
    args = parse_args()
    profile = _resolve_profile(args.profile)
    # Route to train_script.py
```

**Key Responsibilities**:
1. Parse CLI arguments (profile, custom flags)
2. Resolve profile (test/prod)
3. Instantiate appropriate experiment config
4. Route to main.py or train_script.py
5. Provide consistent help text

### pyproject.toml [project.scripts]
**Purpose**: UV script registration

**Convention**:
```toml
[project.scripts]
# Main launcher (unified NSGA-II)
nsga = "scripts.launcher:main_nsga"
train-rl = "scripts.launcher:main_train_rl"

# Progressive Mode Experiments (A→E)
baseline = "scripts.launcher:main_baseline"      # Mode A: Pure NSGA-II
memetic = "scripts.launcher:main_memetic"        # Mode B: + Memetic
roundrobin = "scripts.launcher:main_roundrobin"  # Mode C: + Round-robin
adaptive = "scripts.launcher:main_adaptive"      # Mode D: + Adaptive
rl = "scripts.launcher:main_rl"                  # Mode E: + RL-guided

# Helper commands (a-z)
diagnose = "scripts.launcher:main_diagnose"
clean = "scripts.launcher:main_clean"
list-experiments = "scripts.launcher:main_list"  # Updated from main_list_experiments
stats = "scripts.launcher:main_stats"
archive = "scripts.launcher:main_archive"
```

**Rules**:
- Use kebab-case for multi-word commands (`train-rl`, not `train_rl`)
- Group by function (main vs helpers)
- Keep legacy aliases for backward compatibility
- Comment heavily for maintainability

### CLI_REFERENCE.md
**Purpose**: User-facing documentation

**Structure**:
1. Convention overview
2. Profile hierarchy diagram
3. Main commands with examples
4. Helper commands with examples
5. Quick reference tables
6. Advanced usage patterns
7. Troubleshooting

**Rules**:
- Show actual command examples with expected output
- Include timing estimates for each profile
- Use consistent formatting (code blocks with bash syntax)
- Add "What it does" explanations
- Link to detailed docs for complex topics

## Usage Patterns

### Standard Workflow
```bash
# 1. Diagnose setup
uv run diagnose

# 2. Run progressive modes
uv run baseline --test     # Mode A: Pure NSGA-II
uv run memetic --test      # Mode B: + Memetic local search
uv run roundrobin --test   # Mode C: + Round-robin heuristics
uv run adaptive --test     # Mode D: + Adaptive selection
uv run rl --test           # Mode E: + RL-guided (requires trained model)

# 3. Production runs
uv run baseline --prod --name "thesis-baseline-r01"
uv run memetic --prod --name "thesis-memetic-r01"
```

### Custom Arguments
```bash
# Override experiment name
uv run baseline --prod --name "custom-run-01"

# Combine flags
uv run memetic --test --name "quick-test"
```

### Profile Resolution
```python
from configs.profiles import Profile

def _resolve_profile(value: str | None) -> Profile:
    """Validate profile value and fallback to default."""
    try:
        return Profile.from_string(value or "test")
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(1)
```

## Adding New Commands

### Step 1: Add Experiment Config
```python
# configs/experiments/new_mode.py
@dataclass
class NewModeBaseConfig:
    some_enabled: bool = True

class NewModeTestConfig(NewModeBaseConfig, TestConfig):
    pass

class NewModeProdConfig(NewModeBaseConfig, ProdConfig):
    pass
```

### Step 2: Register in main.py
```python
from configs import experiment_new

EXPERIMENTS = {
    "new": ("New Mode", experiment_new, experiment_new_module),
}
```

### Step 3: Add Launcher Function
```python
# scripts/launcher.py
def main_new_mode():
    """New mode description."""
    parser = create_parser()
    args = parser.parse_args()
    profile = _resolve_profile(args.profile)
    # Route to main with mode="new"
```

### Step 4: Register in pyproject.toml
```toml
[project.scripts]
new-mode = "scripts.launcher:main_new_mode"
```

### Step 5: Test
```bash
uv sync  # Reload scripts
uv run new-mode --test
```

## Error Handling

### Profile Validation
```python
from configs.profiles import Profile

try:
    profile = Profile.from_string(value)
except ValueError:
    console.print(f"[red]Unknown profile: {value}[/red]")
    console.print("Available: test, prod")
    sys.exit(1)
```

### Graceful Defaults
```python
profile = _resolve_profile(args.profile)  # Defaults to test
console.print(f"[yellow]Profile: {profile.value}[/yellow]")
```

## Testing Guidelines

### Manual Testing
```bash
# Test each profile
uv run nsga --test
uv run nsga --prod

# Test each command
uv run train-rl --test
uv run diagnose
uv run clean
```

### Automated Testing
```python
# test/unit/test_launcher.py
def test_profile_mapping():
    assert PROFILE_MAP['test']['config'] == 'configs/test.yaml'
    assert PROFILE_MAP['prod']['env'] == 'prod'

def test_argument_parsing():
    args = parse_args(['--profile', 'test', '--name', 'test-run'])
    assert args.profile == 'test'
    assert args.name == 'test-run'
```

## Best Practices

1. **Consistent Help Text**: All commands should have clear `--help` output
2. **Profile First**: Always check profile before other arguments
3. **Fail Fast**: Validate inputs early, exit with clear error messages
4. **Forward Arguments**: Pass unknown arguments to underlying scripts
5. **Logging**: Use Rich console for user-facing messages
6. **DRY**: Don't duplicate config values, inherit from base
7. **Backward Compatibility**: Keep legacy commands as aliases
8. **Documentation**: Update CLI_REFERENCE.md with every change

## Common Pitfalls

 **Don't**:
- Hardcode config paths in multiple places
- Forget to update pyproject.toml after adding commands
- Mix profile logic with business logic
- Use different naming conventions for similar commands

 **Do**:
- Use PROFILE_MAP for centralized config
- Run `uv sync` after script changes
- Keep launcher.py as thin routing layer
- Follow number/letter convention strictly

## Examples from Codebase

### Good: Clean Profile Routing
```python
def main_nsga():
    """NSGA-II launcher with profile support."""
    parser = create_parser()
    args = parser.parse_args()
    profile = args.profile or 'test'

    # Map profile to environment
    env_name = PROFILE_MAP[profile]['env']

    # Route to main.py
    from main import main
    sys.argv = ['main.py', '--env', env_name]
    main()
```

### Good: Descriptive Help
```python
parser.add_argument(
    '--profile',
    choices=['test', 'prod'],
    help='Experiment profile: test (2 min) or prod (3-5 hrs)'
)
```

### Bad: Hardcoded Paths
```python
#  Don't do this
if profile == 'test':
    config = 'configs/test.yaml'
elif profile == 'prod':
    config = 'configs/prod.yaml'
```

### Good: Centralized Mapping
```python
#  Do this
config = PROFILE_MAP[profile]['config']
```

## References

- Main implementation: `scripts/launcher.py`
- User documentation: `CLI_REFERENCE.md`
- Script registration: `pyproject.toml` [project.scripts]
- Legacy commands: `main.py` (backward compatible entry points)
