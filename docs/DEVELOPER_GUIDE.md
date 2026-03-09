# Schedule Engine — Developer Guide

> Setup, conventions, testing, and contribution guide.

---

## Table of Contents

- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Running Experiments](#running-experiments)
- [Testing](#testing)
- [Code Conventions](#code-conventions)
- [Adding a New Constraint](#adding-a-new-constraint)
- [Adding a New Repair Operator](#adding-a-new-repair-operator)
- [Adding a New RL Action](#adding-a-new-rl-action)
- [Adding a New GA Experiment Mode](#adding-a-new-ga-experiment-mode)
- [Data Format](#data-format)
- [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Prerequisites

- **Python 3.12** (required — numba and ortools don't support 3.14 yet)
- **CUDA 12.1** (optional, for GPU-accelerated PyTorch)
- **24+ CPU cores** recommended for Titan V4 SOTA experiments

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd main-sch-engine

# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -e ".[dev]"
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pymoo | 0.6.1.3 | NSGA-II multi-objective optimisation |
| stable-baselines3 | 2.3.2 | RL agent training (PPO, DQN) |
| sb3-contrib | 2.3.0 | MaskablePPO for action masking |
| torch | 2.4.1+cu121 | Neural network backend |
| numba | 0.64.0 | JIT compilation for repair loops |
| ortools | ≥9.8 | CP-SAT constraint solver |
| gymnasium | 0.29.1 | RL environment standard |
| numpy | 1.26.4 | Numerical computing |
| scipy | 1.11.4 | Scientific computing |
| pydantic | 2.10.3 | Data validation |
| rich | — | Console formatting |
| matplotlib | — | Plotting |

---

## Project Structure

```
main-sch-engine/
├── src/              # Source code (all importable modules)
│   ├── config/       # Config singleton
│   ├── domain/       # Data models
│   ├── constraints/  # 8 hard + 6 soft constraints
│   ├── io/           # Data loading, export, plotting
│   ├── ga/           # GA operators, repair, metrics
│   ├── pipeline/     # Pymoo integration
│   ├── rl/           # RL environment, agents, training
│   ├── experiments/  # Experiment base classes
│   └── utils/        # Logging, console
├── runs/             # Experiment entry points
├── scripts/          # Benchmarks, profiling, diagnostics
├── tests/            # pytest test suite (690+ tests)
├── data/             # Input JSON data files
├── output/           # Experiment output (git-ignored)
├── configs/          # Dataclass config loader
├── docs/             # This documentation
├── pyproject.toml    # Project metadata + tool config
└── solve.py          # Standalone solver entry point
```

### Import Convention

All source code uses absolute imports from the `src` package:

```python
from src.domain import Course, SessionGene, SchedulingContext
from src.constraints import Evaluator, build_constraints
from src.io import DataStore
from src.ga.repair import RepairPipeline
from src.pipeline import SchedulingProblem
from src.rl.gym_env import PymooHyperHeuristicEnv
```

---

## Running Experiments

```bash
# GA experiments
python -m runs.ga_01_baseline
python -m runs.ga_02_memetic
python -m runs.ga_03_aggressive
python -m runs.ga_04_adaptive
python -m runs.ga_05_cp_hybrid

# RL experiments
python -m runs.rl_03_capstone_FIXED
python -m runs.rl_04_train_dqn
python -m runs.rl_06_train_maskable_ppo
python -m runs.rl_09_titan_v4_sota

# Evaluation
python -m runs.eval_all_baselines
python -m runs.plot_master_thesis

# Pre-flight audit
python -m runs.pre_scheduling_audit
```

---

## Testing

### Running Tests

```bash
# Full test suite
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_constraints.py -v

# Specific test
python -m pytest tests/test_constraints.py::TestHardConstraints::test_cte -v

# With coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Test Organisation

| File | Tests | What it covers |
|------|-------|----------------|
| `test_constraints.py` | Hard/soft constraint evaluation |
| `test_evaluator.py` | Evaluator fitness/breakdown |
| `test_domain.py` | Domain model dataclasses |
| `test_timetable.py` | Timetable indexing/conflicts |
| `test_population.py` | Population factory |
| `test_repair_*.py` | Repair operators |
| `test_pipeline_*.py` | Pymoo pipeline |
| `test_rl_*.py` | RL environment/agents |
| `test_algorithm_correctness.py` | End-to-end algorithm validation |
| `test_integration_audit.py` | Import/lint/structure audits |

### Known Failures

Two tests in `test_soft_eval_vectorized.py` fail due to a pre-existing numerical
mismatch (~45% relative difference) between the vectorized and OOP soft evaluators.
These are not blocking.

---

## Code Conventions

### Docstring Style: Google

All docstrings follow **Google style**:

```python
def repair_individual(individual: list[SessionGene], context: SchedulingContext) -> dict:
    """Repair all constraint violations in an individual.

    Applies registered repair operators in priority order. Each operator
    targets a specific hard constraint.

    Args:
        individual: List of session genes to repair (mutated in place).
        context: Scheduling problem context.

    Returns:
        Dictionary of repair statistics keyed by operator name.

    Raises:
        ValueError: If individual is empty.
    """
```

### Type Annotations

All public functions and methods use type annotations:

```python
def calculate_hypervolume(
    front: np.ndarray,
    ref_point: np.ndarray | None = None,
) -> float:
```

### Naming Conventions

| Element | Convention | Example |
|---------|-----------|---------|
| Modules | `snake_case` | `population_factory.py` |
| Classes | `PascalCase` | `RepairPipeline` |
| Functions | `snake_case` | `repair_individual()` |
| Constants | `UPPER_SNAKE` | `HARD_CONSTRAINT_CLASSES` |
| Type aliases | `PascalCase` | `Individual` |
| Private | `_leading_underscore` | `_build_indexes()` |

### Linting

The project uses **ruff** for linting:

```bash
# Check
ruff check src/ tests/ runs/ scripts/

# Fix auto-fixable issues
ruff check --fix src/ tests/ runs/ scripts/

# Format
ruff format src/ tests/ runs/ scripts/
```

Ruff configuration is in `pyproject.toml` under `[tool.ruff]`.

---

## Adding a New Constraint

### 1. Create the constraint class

```python
# src/constraints/hard/new_constraint.py  (or soft/)
"""New constraint — description."""

from src.domain.timetable import Timetable


class NewConstraint:
    """Enforce some new rule.

    Args:
        weight: Penalty multiplier.
    """

    name = "NEW"
    kind = "hard"  # or "soft"

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def evaluate(self, tt: Timetable) -> float:
        """Evaluate the constraint violation.

        Args:
            tt: The timetable to evaluate.

        Returns:
            Weighted violation count (0.0 = no violations).
        """
        violations = 0
        # ... constraint logic ...
        return violations * self.weight
```

### 2. Register in the constraints package

Add to `src/constraints/__init__.py`:

```python
from src.constraints.hard.new_constraint import NewConstraint
# Add to HARD_CONSTRAINT_CLASSES or SOFT_CONSTRAINT_CLASSES
```

### 3. Add tests

Create tests in `tests/test_constraints.py` covering:

- Zero violations on a valid timetable
- Non-zero violations on an invalid timetable
- Correct weight scaling

---

## Adding a New Repair Operator

### 1. Create the operator function

```python
# In src/ga/repair/basic.py (or a new file)
from src.ga.repair.wrappers import repair_operator


@repair_operator(
    name="new_repair",
    description="Fix new constraint violations",
    priority=8,
)
def repair_new_constraint(individual, context):
    """Repair violations of the new constraint.

    Args:
        individual: Individual to repair (mutated in place).
        context: Scheduling context.

    Returns:
        Number of repairs applied.
    """
    repairs = 0
    # ... repair logic ...
    return repairs
```

The `@repair_operator` decorator auto-registers the operator in the registry.

### 2. Add to the repair pipeline

If the operator should run by default, add it to the priority chain
in `src/ga/repair/basic.py::repair_individual()`.

---

## Adding a New RL Action

### 1. Define a new action class

```python
# In src/rl/actions/vectorized_ops.py
class NewAction(_AtomicRepairBase):
    """New repair strategy for the RL agent.

    Args:
        problem: The pymoo scheduling problem.
        config: Post-generation repair configuration.
    """

    def __init__(self, problem):
        super().__init__(
            problem,
            config=PostGenConfig(
                elite_fraction=0.12,
                passes=3,
                stochastic_alternate=True,
            ),
        )
```

### 2. Register in the action space

Add to `VECTORIZED_ACTION_SPACE` in `src/rl/actions/__init__.py`.

### 3. Update the environment

Update `PymooHyperHeuristicEnv` action space size in
`src/rl/gym_env/pymoo_env.py`.

---

## Adding a New GA Experiment Mode

### 1. Create a new experiment class

```python
# In src/experiments/ga_experiment.py
class NewModeExperiment(GAExperiment):
    """GA Mode XX — Description of the new mode.

    Args:
        kwargs: Passed to GAExperiment.
    """

    def __init__(self, **kwargs):
        super().__init__(
            mode="new_mode",
            pop_size=100,
            ngen=200,
            **kwargs,
        )
```

### 2. Create a runner script

```python
# runs/ga_xx_new_mode.py
"""GA Mode XX — New Mode: Description."""

from src.experiments import NewModeExperiment

if __name__ == "__main__":
    exp = NewModeExperiment(seed=42, data_dir="data", output_dir="output/ga_new")
    exp.run()
```

### 3. Export from the experiments package

Add to `src/experiments/__init__.py`:

```python
from src.experiments.ga_experiment import NewModeExperiment
```

---

## Data Format

### Input Data (`data/`)

**Course.json:**

```json
[
  {
    "course_id": "CS101",
    "name": "Introduction to Computing",
    "quanta_per_week": 4,
    "required_room_features": "theory",
    "course_type": "theory",
    "L": 3, "T": 1, "P": 0,
    "department": "CS",
    "semester": "1"
  }
]
```

**Groups.json:**

```json
[
  {
    "group_id": "CS-A",
    "name": "CS Section A",
    "student_count": 45,
    "enrolled_courses": ["CS101", "MA101"]
  }
]
```

**Instructors.json:**

```json
[
  {
    "instructor_id": "INS001",
    "name": "Dr. Smith",
    "qualified_courses": ["CS101", "CS201"],
    "is_full_time": true,
    "max_hours_per_week": 20
  }
]
```

**Rooms.json:**

```json
[
  {
    "room_id": "R101",
    "name": "Lecture Hall 1",
    "capacity": 60,
    "room_features": "theory"
  }
]
```

---

## Troubleshooting

### Common Issues

**`ModuleNotFoundError: No module named 'src'`**

- Ensure you're running from the project root directory
- Ensure the package is installed: `pip install -e .`

**`RuntimeError: Config not initialized`**

- Each entry point must call `init_config(Config(...))` before using `get_config()`
- Check that your experiment class calls `super().__init__()`

**Numba compilation errors**

- First run compiles JIT functions (~30s); subsequent runs are cached
- Clear cache if corrupted: delete `__pycache__` directories

**ortools import failure**

- Requires Python ≤3.12 (not compatible with 3.14 yet)
- Install: `pip install ortools>=9.8`

**CUDA out of memory**

- Reduce `n_envs` in `RLTrainer`
- Set `device="cpu"` for debugging

**Ruff lint errors**

- Run `ruff check --fix` for auto-fixable issues
- Configuration is in `pyproject.toml` under `[tool.ruff]`
