---
applyTo: "test/**/*.py"
---

# Testing Guidelines for Schedule Engine

## ️ PRIORITY: All tests go in `test/` directory (NOT in `src/`)

## Test Structure & Organization

```
test/
├── unit/                # Unit tests for individual components
│   ├── test_config_loader.py
│   ├── test_constraints.py
│   ├── test_ga_operators.py
│   └── ...
├── integration/         # Integration tests for workflows
│   ├── test_standard_workflow.py
│   └── ...
├── fixtures/            # Shared test data and fixtures
│   ├── sample_courses.json
│   └── ...
└── conftest.py          # Shared pytest fixtures
```

## Testing Framework

We use **pytest** with these key conventions:

### File Naming
- Test files: `test_<module_name>.py`
- Test functions: `test_<function_description>()`
- Test classes: `Test<ComponentName>`

### Required Test Practices

#### 1. Use pytest fixtures for test data
```python
import pytest
from src.entities.course import Course

@pytest.fixture
def sample_course():
    """Reusable test course."""
    return Course(
        course_id="CS101",
        title="Intro to Programming",
        credits=3,
        lecture_hours=4,
    )

def test_course_initialization(sample_course):
    assert sample_course.course_id == "CS101"
    assert sample_course.credits == 3
```

#### 2. Use descriptive test names
```python
# Good 
def test_fitness_evaluation_with_no_violations_returns_zero():
    ...

# Bad 
def test_fitness():
    ...
```

#### 3. Follow AAA pattern (Arrange-Act-Assert)
```python
def test_constraint_evaluation():
    # Arrange - Setup test data
    courses = [sample_course1, sample_course2]
    schedule = create_test_schedule(courses)
    
    # Act - Execute function being tested
    violations = evaluate_constraints(schedule)
    
    # Assert - Verify expected outcome
    assert violations == 0
```

#### 4. Test edge cases explicitly
```python
@pytest.mark.parametrize("input,expected", [
    ([], 0),           # Empty case
    ([1], 1),          # Single element
    ([1, 2, 3], 6),    # Normal case
    ([0] * 1000, 0),   # Large input
])
def test_sum_with_edge_cases(input, expected):
    assert sum(input) == expected
```

#### 5. Mock external dependencies
```python
from unittest.mock import Mock, patch

@patch('src.ga.evaluator.fitness.get_config')
def test_fitness_with_mocked_config(mock_config):
    mock_config.return_value.ga.cxpb = 0.75
    # Test code using config
```

## Test Coverage Requirements

- **Minimum coverage**: 70% overall
- **Critical modules** (constraints, ga/operators, config): 90%+
- Run coverage: `pytest --cov=src --cov-report=html test/`

## Testing Specific Components

### GA Operators (crossover, mutation)
- Test chromosome structure preservation
- Test that offspring differ from parents
- Test boundary cases (empty, single-element)

### Constraints
- Test with valid schedules (should return 0 violations)
- Test with known violations (verify detection)
- Test with edge cases (empty rooms, no instructors)

### Config System
- Test YAML parsing
- Test inheritance (test.yaml overrides base.yaml)
- Test validation (invalid values raise errors)

### RL Components
- Mock Gymnasium environment interactions
- Test state encoding correctness
- Test action mapping consistency

## Performance Tests

Mark slow tests with `@pytest.mark.slow`:
```python
@pytest.mark.slow
def test_full_ga_evolution():
    # Run 100 generations (takes ~30 seconds)
    ...
```

Run fast tests only: `pytest -m "not slow"`

## Integration Tests

Integration tests should:
- Use realistic test data (from `test/fixtures/`)
- Test complete workflows (load → validate → GA → export)
- Verify file outputs exist and are valid
- Clean up generated files after test

```python
def test_complete_scheduling_workflow(tmp_path):
    # tmp_path is pytest fixture for temporary directory
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "output"
    
    # Run workflow
    result = run_scheduling_workflow(config_path, output_dir)
    
    # Verify outputs
    assert (output_dir / "schedule.json").exists()
    assert (output_dir / "calendar.pdf").exists()
    assert result.hard_violations == 0
```

## When to Write Tests

- **Before fixing a bug**: Write failing test first, then fix (TDD)
- **When adding features**: Write tests alongside feature code
- **When refactoring**: Ensure tests pass before and after

## Test Maintenance

- Keep tests fast (mock slow operations)
- Keep tests independent (no shared state)
- Update tests when requirements change
- Remove tests for removed features

## Required Before Each Commit

- [ ] Run tests: `pytest test/unit/`
- [ ] Check coverage: `pytest --cov=src test/`
- [ ] Format code: `black src/ test/`
- [ ] Lint code: `ruff check src/ test/`

## Common Pitfalls to Avoid

 **Don't test implementation details** (private methods)  
 **Don't write flaky tests** (dependent on timing/randomness)  
 **Don't skip test cleanup** (use fixtures with `yield`)  
 **Don't test library code** (e.g., don't test DEAP internals)

 **Do test public APIs**  
 **Do test business logic**  
 **Do test error handling**  
 **Do test integration points**

## Never Do

-  Put test files outside `test/` directory
-  Commit failing tests
-  Skip testing after major changes
-  Use production data in tests (create minimal test data)
-  Hardcode paths (use relative paths from test/ directory)
-  Leave debug print statements in committed tests
-  Write tests without docstrings
