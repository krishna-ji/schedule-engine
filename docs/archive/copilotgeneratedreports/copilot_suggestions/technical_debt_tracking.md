# Technical Debt & Code Smells Report

**Project:** Schedule Engine  
**Date:** October 26, 2025  
**Analysis Type:** Static Code Analysis + Manual Review

---

## Overview

This document catalogs specific instances of technical debt, code smells, and areas requiring attention in the Schedule Engine codebase.

**Debt Severity Scale:**
-  **Critical** - Blocks future development or causes bugs
-  **High** - Significant impact on maintainability
-  **Medium** - Minor impact, can wait
-  **Low** - Nice to have, cosmetic

---

## 1. Testing Debt

### 1.1 Zero Automated Test Coverage  CRITICAL

**Location:** Entire codebase  
**Severity:**  Critical  
**Impact:** High risk of regressions, difficult to refactor confidently

**Evidence:**
- No `tests/` directory with pytest infrastructure
- Only manual test scripts in `test/` directory (excluded by `.gitignore`)
- No CI/CD testing pipeline

**Example Risk:**
```python
# src/constraints/hard.py - No tests for critical logic
def no_group_overlap(sessions: List[CourseSession]) -> int:
    # Complex logic with no automated tests
    # Any change risks breaking constraint evaluation
```

**Recommendation:**
Create test suite with minimum 70% coverage for core modules (constraints, GA operators, encoding/decoding).

**Estimated Fix Time:** 20-30 hours for comprehensive suite

---

## 2. Complexity Debt

### 2.1 Large File - `repair.py` (2409 lines)  HIGH

**Location:** `src/ga/operators/repair.py`  
**Severity:**  High  
**Impact:** Hard to navigate, difficult to test individual repair strategies

**Evidence:**
```bash
$ wc -l src/ga/operators/repair.py
2409 src/ga/operators/repair.py
```

**Code Smell:** God Object - Single file handles 8+ different repair strategies

**Affected Functions:**
- `repair_instructor_availability()` - Lines 57-144
- `repair_group_overlaps()` - Lines 147-280
- `repair_room_conflicts()` - Lines 283-420
- `repair_instructor_conflicts()` - Lines 423-560
- `repair_instructor_qualification()` - Lines 563-690
- `repair_room_type_mismatch()` - Lines 693-800
- `repair_session_clustering()` - Lines 803-950
- `repair_incomplete_extra_sessions()` - Lines 953-1150
- `repair_individual_unified()` - Lines 1153-1400

**Recommendation:**
Split into modular structure:
```
src/ga/operators/repair/
    ├── instructor.py      # Instructor-related repairs
    ├── group.py           # Group overlap repairs
    ├── room.py            # Room conflict repairs
    ├── qualification.py   # Qualification repairs
    └── orchestrator.py    # Main repair_individual_unified()
```

**Estimated Fix Time:** 3-4 hours

### 2.2 Large File - `ga_scheduler.py` (1350 lines)  HIGH

**Location:** `src/core/ga_scheduler.py`  
**Severity:**  High  
**Impact:** Mixing concerns (worker init, GA logic, metrics, UI)

**Evidence:**
- Worker initialization function (lines 48-150)
- GAScheduler class (lines 152-1350)
- Metrics tracking embedded in evolution loop
- Rich UI formatting mixed with GA logic

**Recommendation:**
Extract into separate modules:
```
src/core/
    ├── ga_scheduler.py        # Core GAScheduler class (< 500 lines)
    ├── worker_init.py         # Worker initialization logic
    └── evolution_tracker.py   # Metrics and progress tracking
```

**Estimated Fix Time:** 2-3 hours

### 2.3 High Cyclomatic Complexity in `mutate_time_quanta()`  MEDIUM

**Location:** `src/ga/operators/mutation.py:77-150`  
**Severity:**  Medium  
**Cyclomatic Complexity:** ~15

**Evidence:**
```python
def mutate_time_quanta(gene: SessionGene, course, context) -> List[int]:
    num_quanta = len(gene.quanta)
    
    # Multiple nested if-else branches
    if random.random() < 0.3:
        return gene.quanta
    
    if course and hasattr(course, 'prefer_coalesced') and course.prefer_coalesced:
        if random.random() < 0.6:
            # Coalesced block logic
            # ... 20 lines of nested ifs
        else:
            # Scattered logic
            # ... 15 lines of nested ifs
    else:
        # Default logic
        # ... more nested branches
```

**Code Smell:** Long Method with complex control flow

**Recommendation:**
Extract strategies:
```python
class QuantaMutationStrategy(ABC):
    @abstractmethod
    def mutate(self, gene, course, context) -> List[int]:
        pass

class PreserveCurrentStrategy(QuantaMutationStrategy):
    def mutate(self, gene, course, context):
        return gene.quanta

class CoalescedBlockStrategy(QuantaMutationStrategy):
    def mutate(self, gene, course, context):
        # Simplified logic
        pass

class ScatteredQuantaStrategy(QuantaMutationStrategy):
    def mutate(self, gene, course, context):
        # Simplified logic
        pass

# Usage:
strategies = [
    (0.3, PreserveCurrentStrategy()),
    (0.6, CoalescedBlockStrategy()),
    (1.0, ScatteredQuantaStrategy()),
]

def mutate_time_quanta(gene, course, context):
    rand = random.random()
    cumulative = 0
    for threshold, strategy in strategies:
        cumulative += threshold
        if rand < cumulative:
            return strategy.mutate(gene, course, context)
```

**Estimated Fix Time:** 1-2 hours

---

## 3. Performance Debt

### 3.1 Repeated Qualified Instructor Lookups  HIGH

**Location:** `src/ga/operators/mutation.py:29-35`  
**Severity:**  High  
**Impact:** Repeated O(n) lookups during every mutation

**Evidence:**
```python
# Called for EVERY gene mutation (thousands of times per generation)
qualified_instructors = [
    inst_id
    for inst_id, inst in context.instructors.items()
    if course_key in getattr(inst, "qualified_courses", [])
]
```

**Performance Impact:**
- For 100 instructors, 200 courses: 20,000 list comprehensions per generation
- With 100 generations: 2,000,000 repeated computations
- Estimated wasted time: 30-40% of mutation time

**Recommendation:**
Add caching layer to `SchedulingContext`:
```python
@dataclass
class SchedulingContext:
    # ... existing fields
    _qualified_instructors_cache: Dict[tuple, List[str]] = field(default_factory=dict)
    
    def get_qualified_instructors(self, course_key: tuple) -> List[str]:
        if course_key not in self._qualified_instructors_cache:
            self._qualified_instructors_cache[course_key] = [
                inst_id for inst_id, inst in self.instructors.items()
                if course_key in inst.qualified_courses
            ]
        return self._qualified_instructors_cache[course_key]
```

**Expected Performance Gain:** 20-30% faster mutations

**Estimated Fix Time:** 1 hour

### 3.2 Repeated Room Suitability Calculations  MEDIUM

**Location:** `src/ga/operators/mutation.py:50-58`  
**Severity:**  Medium  
**Impact:** Repeated O(m) lookups for room suitability

**Similar Issue:** Same pattern as qualified instructors

**Recommendation:** Add `get_suitable_rooms()` cache to `SchedulingContext`

**Expected Performance Gain:** 10-15% faster mutations

**Estimated Fix Time:** 1 hour

### 3.3 Decoding Individual Multiple Times  MEDIUM

**Location:** `src/ga/evaluator/fitness.py:35`  
**Severity:**  Medium  
**Impact:** Decode same individual for hard + soft + detailed constraints

**Evidence:**
```python
# Called 3+ times for same individual in different contexts
sessions = decode_individual(individual, courses, instructors, groups, rooms)
```

**Recommendation:**
Cache decoded sessions per individual:
```python
# Add to Individual class or use lru_cache
from functools import lru_cache

@lru_cache(maxsize=128)
def decode_individual_cached(individual_tuple, ...):
    return decode_individual(list(individual_tuple), ...)
```

**Expected Performance Gain:** 5-10% faster fitness evaluation

**Estimated Fix Time:** 1-2 hours

---

## 4. Memory Debt

### 4.1 Missing `__slots__` in High-Frequency Classes  MEDIUM

**Location:** `src/ga/sessiongene.py`  
**Severity:**  Medium  
**Impact:** Excessive memory usage for large populations

**Evidence:**
```python
@dataclass
class SessionGene:
    # No __slots__ defined
    course_id: str
    course_type: str
    instructor_id: str
    group_ids: List[str]
    room_id: str
    quanta: List[int]
```

**Memory Impact:**
- Python objects without `__slots__` use ~48-56 bytes overhead per instance
- With `__slots__`: ~16-24 bytes overhead
- For population of 100 with 500 genes each: ~1.6MB wasted

**Recommendation:**
```python
@dataclass
class SessionGene:
    __slots__ = ['course_id', 'course_type', 'instructor_id', 
                 'group_ids', 'room_id', 'quanta']
    course_id: str
    # ... rest of fields
```

**Expected Memory Savings:** 30-40% for SessionGene objects

**Estimated Fix Time:** 10 minutes

---

## 5. Code Organization Debt

### 5.1 Inconsistent Error Handling  MEDIUM

**Location:** Throughout `src/encoder/input_encoder.py`  
**Severity:**  Medium  
**Impact:** Silent failures, unclear error messages

**Evidence:**
```python
# Some functions have error handling:
def load_courses(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        console.print(f"[red]File not found: {json_path}[/red]")
        return {}

# Others don't:
def load_groups(json_path, qts):
    with open(json_path, 'r') as f:  # No error handling
        raw_data = json.load(f)
```

**Recommendation:**
Standardize error handling:
```python
class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass

def load_with_validation(json_path: str, validator: Callable):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        validator(data)
        return data
    except FileNotFoundError:
        raise DataLoadError(f"File not found: {json_path}")
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON in {json_path}: {e}")
    except Exception as e:
        raise DataLoadError(f"Error loading {json_path}: {e}")
```

**Estimated Fix Time:** 2 hours

### 5.2 Magic Numbers in Code  LOW

**Location:** Multiple files  
**Severity:**  Low  
**Impact:** Unclear meaning of hardcoded values

**Examples:**
```python
# src/ga/operators/mutation.py
if random.random() < 0.7:  # What does 0.7 mean?
    new_instructor = gene.instructor_id

if random.random() < 0.3:  # What does 0.3 mean?
    return gene.quanta

# src/ga/population.py
smart_percentage = 0.5  # Why 50%?
greedy_percentage = 0.25  # Why 25%?
```

**Recommendation:**
Define constants:
```python
# src/ga/constants.py
INSTRUCTOR_PRESERVATION_PROBABILITY = 0.7
TIME_PRESERVATION_PROBABILITY = 0.3
SMART_POPULATION_PERCENTAGE = 0.5
GREEDY_POPULATION_PERCENTAGE = 0.25
```

**Estimated Fix Time:** 1 hour

### 5.3 Code Duplication in Constraint Functions  MEDIUM

**Location:** `src/constraints/hard.py`  
**Severity:**  Medium  
**Impact:** Copy-paste pattern repeated 4+ times

**Evidence:**
```python
# Pattern repeated in multiple functions:
def no_group_overlap(sessions):
    conflicts = 0
    resource_time_map = {}
    for session in sessions:
        for resource_id in session.group_ids:
            for q in session.session_quanta:
                key = (resource_id, q)
                if key in resource_time_map:
                    conflicts += 1
                else:
                    resource_time_map[key] = session.course_id
    return conflicts

def no_instructor_conflict(sessions):
    conflicts = 0
    resource_time_map = {}  # Same pattern!
    for session in sessions:
        for resource_id in [session.instructor_id]:  # Slightly different
            for q in session.session_quanta:
                # ... exact same logic
```

**Recommendation:**
Extract common pattern (see Section 2.3 in comprehensive report)

**Estimated Fix Time:** 1 hour

---

## 6. Documentation Debt

### 6.1 Missing API Documentation  MEDIUM

**Location:** Entire codebase  
**Severity:**  Medium  
**Impact:** Hard for new developers to understand API

**Evidence:**
- No Sphinx documentation
- No API reference guide
- Docstrings exist but not compiled into docs

**Recommendation:**
Set up Sphinx with autodoc:
```bash
pip install sphinx sphinx-rtd-theme
cd docs
sphinx-quickstart
sphinx-apidoc -o docs/api src/
sphinx-build -b html docs docs/_build
```

**Estimated Fix Time:** 2-3 hours

### 6.2 Missing Architecture Decision Records  LOW

**Location:** `docs/`  
**Severity:**  Low  
**Impact:** Unclear why certain design decisions were made

**Recommendation:**
Create `docs/adr/` directory with ADR templates

**Estimated Fix Time:** 1 hour per ADR

---

## 7. Maintenance Debt

### 7.1 No Type Checking with mypy  HIGH

**Location:** Entire codebase  
**Severity:**  High  
**Impact:** Type errors not caught until runtime

**Evidence:**
- No `mypy` configuration
- No CI type checking
- Type hints exist but not validated

**Recommendation:**
Add mypy to CI:
```toml
# pyproject.toml
[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false  # Start lenient
```

**Estimated Fix Time:** 2-3 hours (initial setup + fixing errors)

### 7.2 Manual Dependency Management  MEDIUM

**Location:** `requirements.txt`  
**Severity:**  Medium  
**Impact:** No lock file, inconsistent environments

**Recommendation:**
Use `pip-tools` or migrate to `poetry`:
```bash
pip install pip-tools
pip-compile requirements.txt > requirements.lock
```

Or use modern `pyproject.toml` with poetry/hatch

**Estimated Fix Time:** 1 hour

---

## Summary Statistics

### Debt by Severity

| Severity | Count | Estimated Total Fix Time |
|----------|-------|--------------------------|
|  Critical | 1 | 20-30 hours |
|  High | 5 | 12-15 hours |
|  Medium | 8 | 10-12 hours |
|  Low | 3 | 3-4 hours |
| **TOTAL** | **17** | **45-61 hours** |

### Debt by Category

| Category | Issues | Priority |
|----------|--------|----------|
| Testing | 1 |  Critical |
| Complexity | 3 |  High |
| Performance | 3 |  High |
| Memory | 1 |  Medium |
| Organization | 3 |  Medium |
| Documentation | 2 |  Medium |
| Maintenance | 2 |  High |

---

## Prioritized Action Plan

### Week 1-2: Critical Items
1.  Set up pytest testing infrastructure (1 item, 20-30h)
2.  Write tests for constraints and GA operators

### Week 3-4: High Priority Items
1.  Add caching to SchedulingContext (2h)
2.  Split large files (repair.py, ga_scheduler.py) (5-7h)
3.  Add mypy type checking (3h)
4.  Extract mutation strategies (2h)

### Week 5-6: Medium Priority Items
1.  Add `__slots__` to SessionGene (10min)
2.  Standardize error handling (2h)
3.  Extract common constraint patterns (1h)
4.  Set up Sphinx documentation (3h)

### Week 7-8: Low Priority Items
1.  Define constants for magic numbers (1h)
2.  Create ADR documentation (ongoing)
3.  Modernize dependency management (1h)

---

## Tracking Progress

Use this checklist to track technical debt resolution:

- [ ] Testing infrastructure set up
- [ ] Test coverage > 50%
- [ ] Large files split into modules
- [ ] Caching added to hot paths
- [ ] `__slots__` added to high-frequency classes
- [ ] Type checking with mypy enabled
- [ ] Error handling standardized
- [ ] API documentation generated
- [ ] Pre-commit hooks configured
- [ ] CI/CD pipeline running
- [ ] Code complexity reduced (all files < 800 lines)
- [ ] Magic numbers replaced with constants
- [ ] Common patterns extracted to utilities
- [ ] Memory profiling completed
- [ ] Performance benchmarks established

---

**Last Updated:** October 26, 2025  
**Next Review:** After implementing critical items  
**Owner:** Development Team
