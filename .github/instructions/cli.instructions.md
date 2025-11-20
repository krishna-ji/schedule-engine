# CLI Launcher Instructions

**Applies to**: `scripts/launcher.py`, `CLI_REFERENCE.md`, `pyproject.toml` [project.scripts]

## Overview

The unified CLI launcher system provides a clean, consistent interface for running experiments with profile-based configuration inheritance (DRY principle).

## Command Convention

### Main Commands (0-9)
**Purpose**: Primary experimental workflows (GA, RL training)  
**Naming**: Numbers for easy memorization  
**Examples**:
- Command 0: `nsga` - NSGA-II genetic algorithm
- Command 5: `train-rl` - RL agent training

### Helper Commands (a-z)
**Purpose**: Utilities and diagnostics  
**Naming**: Descriptive lowercase letters  
**Examples**:
- `diagnose` - System/GPU/config diagnostics
- `clean` - Clean output directory
- `list-experiments` - Show experiment history

## Profile System

**Three-tier hierarchy** (DRY principle):
```
base.yaml (shared config)
  ↓
--test (smoke test: 30 gens, 10K steps, ~2-10 min)
  ↓
--med (medium: 200 gens, 50K steps, ~30-45 min)
  ↓
--prod (production: 2000 gens, 100K steps, ~1-5 hours)
```

**Implementation**:
- Profiles passed as CLI flags: `--test`, `--med`, `--prod`
- Config files inherit from lower tiers
- Launcher maps profile → environment name → config file

## File Structure

### scripts/launcher.py
**Purpose**: Unified CLI entry point with profile routing

**Structure**:
```python
def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with profile support."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['test', 'med', 'prod'])
    return parser

def main_nsga():
    """NSGA-II launcher."""
    args = parse_args()
    profile = args.profile or 'test'
    # Route to main.py with environment override

def main_train_rl():
    """RL training launcher."""
    args = parse_args()
    profile = args.profile or 'test'
    # Route to train_script.py with config override
```

**Key Responsibilities**:
1. Parse CLI arguments (profile, custom flags)
2. Map profile → config file path
3. Route to appropriate main script (main.py or train_script.py)
4. Forward additional arguments
5. Provide consistent help text

### pyproject.toml [project.scripts]
**Purpose**: UV script registration

**Convention**:
```toml
[project.scripts]
# Main commands (0-9)
nsga = "scripts.launcher:main_nsga"
train-rl = "scripts.launcher:main_train_rl"

# Helper commands (a-z)
diagnose = "scripts.launcher:main_diagnose"
clean = "scripts.launcher:main_clean"
list-experiments = "scripts.launcher:main_list_experiments"

# Backward compatibility (optional)
baseline = "scripts.launcher:main_nsga"  # Alias
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

# 2. Smoke test locally
uv run nsga --test

# 3. Medium validation
uv run nsga --med

# 4. Production on VM
uv run nsga --prod --name "thesis-baseline-r01"
```

### Custom Arguments
```bash
# Override config
uv run nsga --prod --config path/to/custom.yaml

# Named experiment
uv run train-rl --med --name "curriculum-test"

# Combine flags
uv run nsga --test --mode full --name "quick-test"
```

### Profile Mapping
```python
PROFILE_MAP = {
    'test': {
        'env': 'test',
        'config': 'configs/test.yaml',
        'desc': 'Smoke test (30 gens, ~2 min)'
    },
    'med': {
        'env': 'med',
        'config': 'configs/med.yaml',
        'desc': 'Medium run (200 gens, ~30 min)'
    },
    'prod': {
        'env': 'prod',
        'config': 'configs/prod.yaml',
        'desc': 'Production (2000 gens, ~3-5 hours)'
    }
}
```

## Adding New Commands

### Step 1: Add Function to launcher.py
```python
def main_new_command():
    """New command description."""
    parser = create_parser()
    parser.add_argument('--custom-flag', help='Custom flag')
    args = parser.parse_args()
    
    profile = args.profile or 'test'
    # Implementation here
```

### Step 2: Register in pyproject.toml
```toml
[project.scripts]
new-command = "scripts.launcher:main_new_command"
```

### Step 3: Update CLI_REFERENCE.md
Add section with:
- Command syntax
- Profile support
- Examples
- Expected output

### Step 4: Test
```bash
uv sync  # Reload scripts
uv run new-command --help
uv run new-command --test
```

## Error Handling

### Profile Validation
```python
if profile not in PROFILE_MAP:
    console.print(f"[red]Unknown profile: {profile}[/red]")
    console.print("Available: test, med, prod")
    sys.exit(1)
```

### Config File Existence
```python
config_path = Path(PROFILE_MAP[profile]['config'])
if not config_path.exists():
    console.print(f"[red]Config not found: {config_path}[/red]")
    sys.exit(1)
```

### Graceful Fallback
```python
profile = args.profile or 'test'  # Default to test
console.print(f"[yellow]Using profile: {profile}[/yellow]")
```

## Testing Guidelines

### Manual Testing
```bash
# Test each profile
uv run nsga --test
uv run nsga --med
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

❌ **Don't**:
- Hardcode config paths in multiple places
- Forget to update pyproject.toml after adding commands
- Mix profile logic with business logic
- Use different naming conventions for similar commands

✅ **Do**:
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
    choices=['test', 'med', 'prod'],
    help='Experiment profile: test (2 min), med (30 min), prod (3-5 hrs)'
)
```

### Bad: Hardcoded Paths
```python
# ❌ Don't do this
if profile == 'test':
    config = 'configs/test.yaml'
elif profile == 'prod':
    config = 'configs/prod.yaml'
```

### Good: Centralized Mapping
```python
# ✅ Do this
config = PROFILE_MAP[profile]['config']
```

## References

- Main implementation: `scripts/launcher.py`
- User documentation: `CLI_REFERENCE.md`
- Script registration: `pyproject.toml` [project.scripts]
- Legacy commands: `main.py` (backward compatible entry points)
