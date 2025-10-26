# Quick Action Items - Schedule Engine Improvements

**TL;DR:** Prioritized list of improvements you can implement right away.

---

## 🔴 Critical Priority (Do This Week)

### 1. Set Up Basic Testing Infrastructure (2-3 hours)

**Why:** Zero test coverage is the biggest risk for regressions.

**What to do:**
```bash
# 1. Install pytest
pip install pytest pytest-cov

# 2. Create tests directory structure
mkdir -p tests/unit tests/integration tests/fixtures
touch tests/__init__.py tests/conftest.py

# 3. Create first test file
# See example below
```

**Example Test (`tests/unit/test_constraints_hard.py`):**
```python
import pytest
from src.constraints.hard import no_group_overlap
from src.entities.decoded_session import CourseSession

def test_no_overlap_different_groups():
    """Test that different groups can be scheduled at same time"""
    sessions = [
        CourseSession(
            course_id="CS101",
            course_type="theory",
            group_ids=["G1"],
            session_quanta=[1, 2],
            instructor_id="I1",
            room_id="R1"
        ),
        CourseSession(
            course_id="CS102",
            course_type="theory",
            group_ids=["G2"],
            session_quanta=[1, 2],
            instructor_id="I2",
            room_id="R2"
        ),
    ]
    assert no_group_overlap(sessions) == 0

def test_detects_overlap_same_group():
    """Test that overlap is detected for same group"""
    sessions = [
        CourseSession(
            course_id="CS101",
            course_type="theory",
            group_ids=["G1"],
            session_quanta=[1, 2],
            instructor_id="I1",
            room_id="R1"
        ),
        CourseSession(
            course_id="CS102",
            course_type="theory",
            group_ids=["G1"],
            session_quanta=[2, 3],  # Overlap at quantum 2
            instructor_id="I2",
            room_id="R2"
        ),
    ]
    assert no_group_overlap(sessions) == 1
```

**Run tests:**
```bash
pytest tests/ -v
```

### 2. Add Pre-commit Hooks (30 minutes)

**Why:** Catch code quality issues before they're committed.

**What to do:**
```bash
# 1. Install pre-commit
pip install pre-commit black ruff

# 2. Create .pre-commit-config.yaml
```

**Create `.pre-commit-config.yaml`:**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.1.0
    hooks:
      - id: black
        language_version: python3.12

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.2.0
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

**Install hooks:**
```bash
pre-commit install
pre-commit run --all-files  # Test on existing files
```

---

## 🟡 High Priority (Do This Month)

### 3. Add Caching to SchedulingContext (1-2 hours)

**Why:** Repeated lookups for qualified instructors are slowing down mutations.

**Expected Performance Gain:** 20-30% faster GA execution

**What to do:**

**Edit `src/core/types.py`:**
```python
from dataclasses import dataclass, field
from typing import Dict, List
# ... other imports

@dataclass
class SchedulingContext:
    courses: Dict[tuple, Course]
    groups: Dict[str, Group]
    instructors: Dict[str, Instructor]
    rooms: Dict[str, Room]
    available_quanta: List[int]
    qts: "QuantumTimeSystem"
    
    # ADD: Cache fields
    _qualified_instructors_cache: Dict[tuple, List[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    _suitable_rooms_cache: Dict[tuple, List[str]] = field(
        default_factory=dict, init=False, repr=False
    )
    
    def get_qualified_instructors(self, course_key: tuple) -> List[str]:
        """Get qualified instructors for a course (cached)"""
        if course_key not in self._qualified_instructors_cache:
            self._qualified_instructors_cache[course_key] = [
                inst_id
                for inst_id, inst in self.instructors.items()
                if course_key in getattr(inst, "qualified_courses", [])
            ]
        return self._qualified_instructors_cache[course_key]
    
    def get_suitable_rooms(
        self, course_id: str, group_id: str, course_type: str
    ) -> List[str]:
        """Get suitable rooms for a course-group pair (cached)"""
        cache_key = (course_id, group_id, course_type)
        if cache_key not in self._suitable_rooms_cache:
            # Implementation from mutation.py's find_suitable_rooms_for_course
            self._suitable_rooms_cache[cache_key] = self._compute_suitable_rooms(
                course_id, group_id, course_type
            )
        return self._suitable_rooms_cache[cache_key]
    
    def _compute_suitable_rooms(
        self, course_id: str, group_id: str, course_type: str
    ) -> List[str]:
        """Internal method to compute suitable rooms"""
        # Move logic from mutation.py here
        suitable = []
        group = self.groups.get(group_id)
        if not group:
            return list(self.rooms.keys())
        
        for room_id, room in self.rooms.items():
            # Check type match
            if course_type == "lab" and room.room_type != "lab":
                continue
            # Check capacity
            if room.capacity < group.size:
                continue
            suitable.append(room_id)
        
        return suitable if suitable else list(self.rooms.keys())
```

**Update `src/ga/operators/mutation.py` to use cache:**
```python
# OLD:
qualified_instructors = [
    inst_id
    for inst_id, inst in context.instructors.items()
    if course_key in getattr(inst, "qualified_courses", [])
]

# NEW:
qualified_instructors = context.get_qualified_instructors(course_key)
```

### 4. Add `__slots__` to SessionGene (10 minutes)

**Why:** Reduce memory usage by 30-40% for SessionGene objects.

**Expected Memory Savings:** ~20MB for population of 100

**What to do:**

**Edit `src/ga/sessiongene.py`:**
```python
from dataclasses import dataclass
from typing import List

@dataclass
class SessionGene:
    __slots__ = [
        'course_id',
        'course_type',
        'instructor_id',
        'group_ids',
        'room_id',
        'quanta'
    ]
    
    course_id: str
    course_type: str
    instructor_id: str
    group_ids: List[str]
    room_id: str
    quanta: List[int]
```

**Note:** This is a breaking change if external code accesses SessionGene attributes dynamically. Test thoroughly.

### 5. Split Large Files (2-3 hours)

**Why:** `repair.py` has 2400+ lines, making it hard to maintain.

**What to do:**

Create directory structure:
```bash
mkdir -p src/ga/operators/repair
```

**Create files:**
```
src/ga/operators/repair/
├── __init__.py           # Export main repair function
├── base.py               # Base classes and interfaces
├── instructor.py         # repair_instructor_availability
├── group.py              # repair_group_overlaps
├── room.py               # repair_room_conflicts
├── qualification.py      # repair_instructor_qualification
└── orchestrator.py       # repair_individual_unified
```

**Move functions from `repair.py` to appropriate modules, then update imports.**

---

## 🟢 Medium Priority (Do Next Month)

### 6. Add Type Hints with mypy (1-2 hours)

**What to do:**

**Create `pyproject.toml` (if doesn't exist):**
```toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Start lenient
ignore_missing_imports = true

# Gradually increase strictness:
# disallow_untyped_defs = true
# disallow_any_unimported = true
```

**Run mypy:**
```bash
pip install mypy
mypy src/ --config-file pyproject.toml
```

**Fix type errors incrementally.**

### 7. Add GitHub Actions CI (30 minutes)

**What to do:**

**Create `.github/workflows/ci.yml`:**
```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]

    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov black ruff mypy
      
      - name: Lint with ruff
        run: ruff check src/
      
      - name: Format check with black
        run: black --check src/
      
      - name: Type check with mypy
        run: mypy src/ --config-file pyproject.toml || true
      
      - name: Test with pytest
        run: pytest tests/ --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        if: matrix.python-version == '3.12'
```

---

## 🔵 Low Priority (Nice to Have)

### 8. Add API Documentation with Sphinx (2-3 hours)

```bash
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints
cd docs
sphinx-quickstart
# Answer prompts

# Edit docs/conf.py:
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

# Generate docs
sphinx-apidoc -o docs/api src/
sphinx-build -b html docs docs/_build
```

### 9. Create requirements-dev.txt

```txt
# Development dependencies
pytest>=8.0.0
pytest-cov>=4.1.0
pytest-xdist>=3.5.0
black>=24.1.0
ruff>=0.2.0
mypy>=1.8.0
pre-commit>=3.6.0
sphinx>=7.2.0
sphinx-rtd-theme>=2.0.0
```

### 10. Use Python 3.12+ Features

**Replace type hints:**
```python
# OLD
from typing import List, Dict, Tuple, Optional

def func(items: List[int]) -> Optional[Dict[str, str]]:
    pass

# NEW (Python 3.12+)
def func(items: list[int]) -> dict[str, str] | None:
    pass
```

**Use match statements:**
```python
# OLD
if strategy == "hybrid":
    return generate_hybrid()
elif strategy == "smart":
    return generate_smart()
else:
    raise ValueError()

# NEW
match strategy:
    case "hybrid":
        return generate_hybrid()
    case "smart":
        return generate_smart()
    case _:
        raise ValueError()
```

---

## Summary Checklist

### This Week
- [ ] Set up pytest testing (3 hours)
- [ ] Add pre-commit hooks (30 min)
- [ ] Write 10-15 basic tests (2 hours)

### This Month
- [ ] Add caching to SchedulingContext (2 hours)
- [ ] Add `__slots__` to SessionGene (10 min)
- [ ] Split repair.py into modules (3 hours)
- [ ] Set up GitHub Actions CI (30 min)
- [ ] Add type hints and run mypy (2 hours)

### Next Month
- [ ] Increase test coverage to 50%+ (ongoing)
- [ ] Add API documentation (3 hours)
- [ ] Profile and optimize hot paths (2-3 hours)
- [ ] Modernize to Python 3.12+ syntax (1-2 hours)

---

## Expected Results

After implementing **Critical + High Priority** items:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Test Coverage | 0% | 40%+ | ✅ Huge |
| GA Performance | Baseline | +20-30% | ✅ Significant |
| Memory Usage | Baseline | -30% | ✅ Good |
| Code Quality | Good | Excellent | ✅ Better |
| Maintainability | Medium | High | ✅ Easier refactoring |

**Total Time Investment:** ~15-20 hours for critical + high priority items

**ROI:** High - Better code quality, faster execution, reduced technical debt

---

**Next Steps:**
1. Review this document with your team
2. Prioritize based on your immediate needs
3. Start with testing infrastructure (most critical)
4. Implement incrementally, test each change
5. Update this checklist as you complete items

**Questions?** Refer to `comprehensive_improvement_report.md` for detailed explanations.
