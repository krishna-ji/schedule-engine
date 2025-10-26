---
applyTo: "test/**/*.py"
---

# Testing Instructions

## Overview
All test files must be placed in the `test/` directory. Currently manual testing with small configurations, future automated testing with pytest.

## Test Organization
```
test/
├── test_constraints.py      # Constraint function tests
├── test_ga_operators.py     # Crossover/mutation tests
├── test_data_loading.py     # Encoder/decoder tests
├── test_feasibility.py      # Validation tests
└── fixtures/                # Test data files
    ├── small_course.json
    ├── small_groups.json
    └── ...
```

## Writing Tests

### Manual Testing (Current)
```python
# test/manual_test_constraints.py
"""
Manual test for constraint functions.
Run: python test/manual_test_constraints.py
"""

from src.constraints.hard import no_group_overlap
from src.entities.decoded_session import CourseSession
# ... imports ...

def test_no_overlap_basic():
    """Test basic case: no conflicts."""
    sessions = [
        CourseSession(course=..., groups=["G1"], quanta=[1, 2]),
        CourseSession(course=..., groups=["G2"], quanta=[1, 2]),
    ]
    context = create_test_context()
    
    penalty = no_group_overlap(sessions, context)
    assert penalty == 0, f"Expected 0, got {penalty}"
    print("✓ test_no_overlap_basic passed")

def test_overlap_detected():
    """Test overlap detection."""
    sessions = [
        CourseSession(course=..., groups=["G1"], quanta=[1, 2]),
        CourseSession(course=..., groups=["G1"], quanta=[2, 3]),  # Overlap at quantum 2
    ]
    context = create_test_context()
    
    penalty = no_group_overlap(sessions, context)
    assert penalty > 0, f"Expected penalty > 0, got {penalty}"
    print("✓ test_overlap_detected passed")

if __name__ == "__main__":
    test_no_overlap_basic()
    test_overlap_detected()
    print("\n✓ All tests passed!")
```

### Automated Testing (Future - pytest)
```python
# test/test_constraints.py
"""
Automated constraint tests.
Run: pytest test/test_constraints.py
"""

import pytest
from src.constraints.hard import no_group_overlap
# ... imports ...

@pytest.fixture
def test_context():
    """Create test context for all tests."""
    return create_test_context()

def test_no_overlap_basic(test_context):
    sessions = [...]  # Test data
    assert no_group_overlap(sessions, test_context) == 0

def test_overlap_detected(test_context):
    sessions = [...]  # Test data with overlap
    assert no_group_overlap(sessions, test_context) > 0

@pytest.mark.parametrize("overlap_count,expected_penalty", [
    (0, 0),
    (1, 1),
    (5, 5),
])
def test_overlap_scaling(test_context, overlap_count, expected_penalty):
    sessions = create_overlap_sessions(overlap_count)
    assert no_group_overlap(sessions, test_context) == expected_penalty
```

## Test Data Creation

### Minimal Test Context
```python
def create_test_context():
    """Create minimal context for testing."""
    qts = QuantumTimeSystem()
    
    courses = {
        "CS101": Course(
            code="CS101",
            name="Test Course",
            L=3, T=0, P=0,
            type="theory"
        )
    }
    
    groups = {
        "G1": Group(
            id="G1",
            enrolled_courses=["CS101"],
            available_quanta=list(range(72))
        )
    }
    
    instructors = {
        "I1": Instructor(
            id="I1",
            name="Test Instructor",
            qualifications=["CS101"],
            available_quanta=list(range(72))
        )
    }
    
    rooms = {
        "R1": Room(
            id="R1",
            capacity=50,
            type="theory",
            available_quanta=list(range(72))
        )
    }
    
    return SchedulingContext(
        courses=courses,
        groups=groups,
        instructors=instructors,
        rooms=rooms,
        available_quanta=qts.get_all_operating_quanta(),
        qts=qts
    )
```

### Test Fixture Files
```json
// test/fixtures/small_course.json
[
  {
    "courseCode": "TEST101",
    "courseName": "Test Course",
    "L": 2,
    "T": 0,
    "P": 0,
    "courseType": "theory"
  }
]
```

## Testing Strategies

### Unit Tests
- Test individual functions in isolation
- Use minimal test data
- Mock dependencies if needed
- Fast execution (< 1s per test)

### Integration Tests
- Test component interactions (encoder → GA → decoder)
- Use realistic test data
- Slower execution acceptable (< 10s per test)

### Smoke Tests
- Quick sanity checks with `configs/test.yaml`
- Run full pipeline with 10 generations, 4 population
- Verify no crashes, output files created

### Regression Tests
- Save known-good outputs
- Compare new runs against saved outputs
- Detect unexpected changes

## Testing Checklist

### Before Committing
- [ ] Run smoke test: `python main.py --env test`
- [ ] Check for new warnings/errors
- [ ] Verify output files generated
- [ ] Review logger.txt for issues

### Adding New Features
- [ ] Write tests for new code
- [ ] Test edge cases (empty inputs, max values, etc.)
- [ ] Test error handling
- [ ] Update test documentation

### Constraint Testing
- [ ] Test with 0 violations (should return 0)
- [ ] Test with known violation count
- [ ] Test with edge cases (empty schedule, single session)
- [ ] Test with realistic data

### Operator Testing
- [ ] Test crossover preserves gene count
- [ ] Test mutation stays within bounds
- [ ] Test repair reduces violations
- [ ] Test with different population sizes

## Test Configuration

### Use Test Config
```yaml
# configs/test.yaml - Optimized for fast testing
ga:
  ngen: 10              # Fast
  pop_size: 4           # Minimal
  
feasibility:
  enable_checks: false  # Skip for speed
  
parallel:
  use_multiprocessing: false  # Easier debugging
```

## Debugging Tests

### Print Debugging
```python
def test_my_function():
    result = my_function()
    print(f"DEBUG: result = {result}")  # Temporary debug
    assert result == expected
```

### Rich Console in Tests
```python
from rich.console import Console
console = Console()

def test_my_function():
    console.print("[cyan]Testing my_function...[/cyan]")
    # ... test code ...
    console.print("[green]✓ Test passed[/green]")
```

### Logging in Tests
```python
import logging
logging.basicConfig(level=logging.DEBUG)

def test_my_function():
    logger = logging.getLogger(__name__)
    logger.debug("Starting test...")
    # ... test code ...
```

## Performance Testing

### Timing Tests
```python
import time

def test_ga_speed():
    start = time.time()
    run_standard_workflow(pop_size=4, generations=10)
    elapsed = time.time() - start
    
    assert elapsed < 30, f"Test took {elapsed}s, expected < 30s"
    print(f"✓ Speed test passed ({elapsed:.1f}s)")
```

### Memory Testing
```python
import tracemalloc

def test_memory_usage():
    tracemalloc.start()
    run_standard_workflow(...)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert peak < 500 * 1024 * 1024, f"Peak memory {peak/1024/1024:.1f}MB"
    print(f"✓ Memory test passed (peak: {peak/1024/1024:.1f}MB)")
```

## Never Do
- ❌ Put test files outside `test/` directory
- ❌ Commit failing tests
- ❌ Skip testing after major changes
- ❌ Use production data in tests (create minimal test data)
- ❌ Hardcode paths (use relative paths from test/ directory)
- ❌ Leave debug print statements in committed tests
- ❌ Write tests without docstrings
