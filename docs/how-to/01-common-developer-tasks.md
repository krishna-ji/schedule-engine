# Common Developer Tasks

## Quick Reference

| Task | Command | Time |
|------|---------|------|
| Run smoke test | `uv run test` | 5-10 min |
| Run production | `uv run prod` | 1-2.5 hrs |
| Format code | `black src/ test/` | 10 sec |
| Lint code | `ruff check src/ test/` | 5 sec |
| Run tests | `pytest test/` | 30 sec |
| Check coverage | `pytest --cov=src test/` | 1 min |
| Validate config | `uv run verify-config` | 2 sec |
| Check data | `uv run check-data` | 5 sec |
| Train RL agent | `uv run train-rl` | 2 hrs |

## Development Workflow

### 1. Setting Up Development Environment

**Initial Setup:**
```powershell
# Clone repository
git clone https://github.com/krishna-ji/schedule-engine.git
cd schedule-engine

# Install dependencies
uv sync --frozen --group dev

# Verify installation
uv run diagnose-system
```

**IDE Setup (VS Code):**
```powershell
# Install extensions
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension charliermarsh.ruff

# Open project
code .
```

**Configure Python Interpreter:**
1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose `.venv/Scripts/python.exe`

### 2. Making Code Changes

**Standard Workflow:**
```powershell
# 1. Create feature branch
git checkout -b feature/my-new-feature

# 2. Make changes (edit files)

# 3. Format code
black src/ test/

# 4. Lint code
ruff check src/ test/ --fix

# 5. Run tests
pytest test/

# 6. Check coverage
pytest --cov=src --cov-report=html test/

# 7. Commit changes
git add .
git commit -m "feat(module): add new feature"

# 8. Push to remote
git push origin feature/my-new-feature
```

### 3. Adding a New Constraint

**Steps:**

1. **Create constraint file:**
```powershell
# Create file: src/constraints/hard_my_new_constraint.py
```

2. **Implement constraint function:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ga.sessiongene import Individual
    from src.core.types import SchedulingContext
    from src.config.models import Config


def evaluate_my_new_constraint(
    individual: "Individual",
    context: "SchedulingContext",
    config: "Config"
) -> int:
    """
    Evaluate my new constraint.
    
    Args:
        individual: Chromosome to evaluate
        context: Scheduling problem data
        config: Configuration object
    
    Returns:
        Number of violations (0 = satisfied)
    """
    violations = 0
    
    # Implementation logic
    for gene in individual:
        if violates_constraint(gene):
            violations += 1
    
    return violations
```

3. **Register in fitness evaluator:**
```python
# Edit: src/ga/evaluator/fitness.py

from src.constraints.hard_my_new_constraint import evaluate_my_new_constraint

# Add to evaluation
if config.hard_constraints.my_new_constraint.enabled:
    violations = evaluate_my_new_constraint(individual, context, config)
    hard_penalties += violations * config.hard_constraints.my_new_constraint.weight
```

4. **Add to configuration:**
```yaml
# Edit: configs/base.yaml

hard_constraints:
  my_new_constraint:
    enabled: true
    weight: 3.0
```

5. **Add to config model:**
```python
# Edit: src/config/models.py

class HardConstraintsConfig(BaseModel):
    # ... existing constraints ...
    my_new_constraint: ConstraintConfig = Field(
        default=ConstraintConfig(enabled=True, weight=3.0)
    )
```

6. **Write tests:**
```python
# Create: test/unit/test_my_new_constraint.py

import pytest
from src.constraints.hard_my_new_constraint import evaluate_my_new_constraint

def test_my_new_constraint_satisfied():
    """Test constraint when satisfied."""
    individual = create_valid_individual()
    context = create_test_context()
    config = create_test_config()
    
    violations = evaluate_my_new_constraint(individual, context, config)
    assert violations == 0

def test_my_new_constraint_violated():
    """Test constraint when violated."""
    individual = create_invalid_individual()
    context = create_test_context()
    config = create_test_config()
    
    violations = evaluate_my_new_constraint(individual, context, config)
    assert violations > 0
```

7. **Run tests:**
```powershell
pytest test/unit/test_my_new_constraint.py -v
```

### 4. Adding a New Heuristic Operator

**Steps:**

1. **Choose category:**
- Construction: Build schedules from scratch
- Perturbation: Modify existing schedules
- Repair: Fix specific violations
- Optimization: Improve quality
- Diversity: Maintain variety

2. **Create operator file:**
```powershell
# Example: src/heuristics/perturbation/swap_my_operator.py
```

3. **Implement operator:**
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ga.sessiongene import Individual
    from src.core.types import SchedulingContext
    from src.config.models import Config


def swap_my_operator(
    individual: "Individual",
    context: "SchedulingContext",
    config: "Config"
) -> "Individual":
    """
    Apply my custom swap operator.
    
    Args:
        individual: Input chromosome
        context: Scheduling problem data
        config: Configuration object
    
    Returns:
        Modified chromosome (new copy)
    """
    import copy
    import random
    
    modified = copy.deepcopy(individual)
    
    # Implementation logic
    if len(modified) >= 2:
        i, j = random.sample(range(len(modified)), 2)
        modified[i], modified[j] = modified[j], modified[i]
    
    return modified
```

4. **Register in heuristics registry:**
```python
# Edit: src/heuristics/registry.py

from src.heuristics.perturbation.swap_my_operator import swap_my_operator

HEURISTIC_REGISTRY = {
    # ... existing operators ...
    "swap_my_operator": HeuristicMetadata(
        id="swap_my_operator",
        name="Swap My Operator",
        category="perturbation",
        function=swap_my_operator,
        enabled=True,
        avg_time_ms=50.0,
        success_rate=0.7,
        description="Custom swap operator"
    )
}
```

5. **Add to configuration:**
```yaml
# Edit: configs/base.yaml

heuristics:
  enabled: true
  operators:
    swap_my_operator:
      enabled: true
```

6. **Write tests:**
```python
# Create: test/unit/test_swap_my_operator.py

import pytest
from src.heuristics.perturbation.swap_my_operator import swap_my_operator

def test_swap_my_operator():
    """Test custom swap operator."""
    individual = create_test_individual()
    context = create_test_context()
    config = create_test_config()
    
    result = swap_my_operator(individual, context, config)
    
    # Verify operator worked
    assert result != individual  # Should be different
    assert len(result) == len(individual)  # Same length
```

7. **Test operator:**
```powershell
pytest test/unit/test_swap_my_operator.py -v

# Test in GA run
uv run heuristics --env test
```

### 5. Training an RL Agent

**Standard Training:**
```powershell
# 1. Generate validation sets
python scripts/validation/generate_validation_set.py

# 2. Train with curriculum learning
uv run train-curriculum

# 3. Monitor training (in separate terminal)
uv run tensorboard

# 4. Open browser: http://localhost:6006

# 5. Select best checkpoint
uv run select-checkpoint

# 6. Promote to production
uv run promote-model

# 7. Validate model
uv run validate-rl

# 8. Run RL experiment
uv run exp5 --env prod
```

**Quick Training (for testing):**
```powershell
# Train for 10K timesteps (~15 min)
uv run train-rl --timesteps 10000

# Test RL agent
python main.py --mode rl-guided --env test
```

**Custom Training Script:**
```python
# scripts/train_custom.py

from src.rl.training.trainer import RLTrainer
from src.rl.gym_env.schedule_env import ScheduleEnv

# Create environment
env = ScheduleEnv(
    context=scheduling_context,
    config=config,
    max_generations=100
)

# Create trainer
trainer = RLTrainer(
    env=env,
    agent_type="ppo",
    total_timesteps=100000,
    save_path="models/rl_agents/custom_ppo.zip"
)

# Train
trainer.train()

# Evaluate
results = trainer.evaluate(n_eval_episodes=10)
print(f"Mean reward: {results['mean_reward']}")
```

### 6. Running Experiments

**Single Experiment:**
```powershell
# Run with experiment tag
uv run exp1 --env prod --experiment "thesis-final-v3"

# Results saved to:
# output/baseline/evaluation_20251120_123456_thesis-final-v3/
```

**Batch Experiments:**
```powershell
# Run all thesis experiments
$experiments = @("exp1", "exp2", "exp3", "exp4", "exp5")
foreach ($exp in $experiments) {
    Write-Host "Running $exp..."
    uv run $exp --env prod --experiment "thesis-batch-$(Get-Date -Format 'yyyyMMdd')"
}

# Compare results
uv run compare-experiments
```

**Custom Experiment Script:**
```python
# scripts/run_custom_experiment.py

from src.workflows.standard_run import run_standard_workflow
from src.workflows.experiment_manager import ExperimentManager
from src.config import init_config

# Load custom config
config = init_config("configs/my-experiment.yaml")

# Initialize experiment manager
manager = ExperimentManager()

# Run experiment
result = run_standard_workflow(
    pop_size=config.ga.pop_size,
    generations=config.ga.ngen,
    config=config
)

# Log results
print(f"Best fitness: {result['best_individual'].fitness.values}")
```

### 7. Benchmarking Performance

**GPU Benchmarking:**
```powershell
# Run GPU benchmark
uv run benchmark-gpu

# Compare CPU vs GPU
python scripts/benchmarking/compare_cpu_gpu.py
```

**Heuristic Benchmarking:**
```powershell
# Benchmark all heuristics
python scripts/benchmarking/benchmark_heuristics.py

# Results saved to: output/heuristic_benchmarks.json
```

**Custom Benchmark:**
```python
# scripts/benchmark_custom.py

import time
from src.ga.evaluator.fitness import evaluate

# Benchmark evaluation speed
start = time.time()
for _ in range(1000):
    fitness = evaluate(individual, context, config)
duration = time.time() - start

print(f"1000 evaluations: {duration:.2f}s")
print(f"Per evaluation: {duration/1000*1000:.2f}ms")
```

### 8. Debugging Issues

**Enable Debug Logging:**
```python
# Add to main.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Use Rich Console for Debugging:**
```python
from src.utils.console_service import get_console

console = get_console()
console.print("[yellow]Debug:[/yellow] Variable value:", variable)
console.print(individual)  # Pretty-print complex objects
```

**Breakpoint Debugging:**
```python
# Add breakpoint
breakpoint()  # Python 3.7+

# Or use pdb
import pdb; pdb.set_trace()
```

**VS Code Debugging:**
1. Set breakpoint (click left of line number)
2. `F5` → "Python File"
3. Step through code with `F10` (step over), `F11` (step into)

### 9. Writing Tests

**Unit Test Template:**
```python
# test/unit/test_my_module.py

import pytest
from src.my_module import my_function

@pytest.fixture
def sample_data():
    """Fixture for reusable test data."""
    return create_test_data()

def test_my_function_basic(sample_data):
    """Test basic functionality."""
    result = my_function(sample_data)
    assert result is not None

def test_my_function_edge_case():
    """Test edge case."""
    result = my_function([])
    assert result == expected_value

def test_my_function_raises_error():
    """Test error handling."""
    with pytest.raises(ValueError):
        my_function(invalid_input)
```

**Run Tests:**
```powershell
# Run all tests
pytest test/

# Run specific test file
pytest test/unit/test_my_module.py

# Run specific test
pytest test/unit/test_my_module.py::test_my_function_basic

# Run with coverage
pytest --cov=src --cov-report=html test/

# View coverage report
start htmlcov/index.html  # Windows
```

### 10. Code Quality Checks

**Pre-commit Checklist:**
```powershell
# 1. Format code
black src/ test/

# 2. Lint code
ruff check src/ test/ --fix

# 3. Type check
mypy src/

# 4. Run tests
pytest test/

# 5. Check coverage
pytest --cov=src test/

# 6. Validate config (if changed)
uv run verify-config

# 7. Check data (if changed)
uv run check-data
```

**Automated Pre-commit Hook:**
```powershell
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
# (Add black, ruff, mypy, pytest)

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### 11. Creating Documentation

**Add Docstrings:**
```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief one-line summary.
    
    Longer description with more details about what
    the function does and why.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When input is invalid
    
    Example:
        >>> result = my_function(42, "test")
        >>> print(result)
        True
    """
    # Implementation
```

**Generate API Documentation:**
```powershell
# Install sphinx
pip install sphinx sphinx-rtd-theme

# Generate docs
cd docs/
sphinx-build -b html . _build/html

# View docs
start _build/html/index.html
```

### 12. Deploying Changes

**Merge to Main Branch:**
```powershell
# 1. Ensure all tests pass
pytest test/

# 2. Update from main
git checkout dev-krishna
git pull origin dev-krishna

# 3. Merge feature branch
git merge feature/my-feature

# 4. Push to remote
git push origin dev-krishna

# 5. Create pull request (GitHub)
# Navigate to repository → Pull Requests → New PR
```

**Tag Release:**
```powershell
# Create tag
git tag -a v2.0.0 -m "Release v2.0.0: RL integration complete"

# Push tag
git push origin v2.0.0
```

## Common Issues & Solutions

### Issue: Import Errors

**Solution:**
```powershell
# Reinstall dependencies
uv sync --frozen

# Verify Python path
python -c "import sys; print(sys.path)"

# Check if in project root
pwd  # Should be .../schedule-engine
```

### Issue: Test Failures

**Solution:**
```powershell
# Run specific failing test
pytest test/unit/test_module.py::test_function -v

# Run with print output
pytest test/unit/test_module.py -s

# Debug with pdb
pytest test/unit/test_module.py --pdb
```

### Issue: GPU Not Available

**Solution:**
```powershell
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Diagnose GPU
uv run diagnose-gpu

# Reinstall PyTorch with CUDA
uv sync --frozen --reinstall-package torch
```

### Issue: Configuration Errors

**Solution:**
```powershell
# Validate config
uv run verify-config

# Show merged config
uv run show-config

# Check for syntax errors
python -c "import yaml; yaml.safe_load(open('configs/base.yaml'))"
```

## See Also

- [Code Structure](01-code-structure.md) - File organization
- [First Run Guide](../get-started/03-first-run.md) - Getting started
- [UV Commands Reference](../get-started/04-uv-commands.md) - All commands
- [Troubleshooting](../troubleshooting/01-common-issues.md) - Common issues
