# Optimization Strategies for Constraint Checking

**Document Version:** 1.0  
**Date:** November 17, 2025  
**Status:** Design Proposals

---

## Executive Summary

This document proposes concrete optimization strategies to reduce constraint evaluation complexity from **O(S × Q + N × D × Q_d × log(Q_d))** to **O(modified_sessions)** for incremental updates.

**Key Strategies:**
1. **Delta Evaluation** - Only re-evaluate changed sessions (5-10× speedup)
2. **Constraint Caching** - Cache individual constraint results (2-3× speedup)
3. **Data Structure Improvements** - Use better indexing structures (10-20% speedup)
4. **Parallel Evaluation** - Evaluate independent constraints in parallel (2-4× speedup)

**Combined Potential:** 10-30× speedup for typical GA/RL operations

---

## 1. Delta Evaluation (Incremental Checking)

### 1.1 Core Concept

**Problem:** Current evaluation re-checks ALL sessions even if only 1 gene mutated.

**Solution:** Track which sessions changed and only re-evaluate affected constraints.

### 1.2 Change Tracking

```python
from dataclasses import dataclass
from typing import Set, List, FrozenSet
from src.ga.sessiongene import SessionGene

@dataclass
class IndividualDelta:
    """Tracks changes between two individuals for incremental evaluation."""
    
    added_genes: List[SessionGene]
    removed_genes: List[SessionGene]
    modified_genes: List[tuple[SessionGene, SessionGene]]  # (old, new)
    
    # Affected resources (for fast lookup)
    affected_instructors: Set[str]
    affected_groups: Set[str]
    affected_rooms: Set[str]
    affected_quanta: Set[int]
    affected_courses: Set[tuple]  # (course_id, course_type)
    
    @property
    def is_empty(self) -> bool:
        """Check if there are no changes."""
        return not (self.added_genes or self.removed_genes or self.modified_genes)
    
    @property
    def total_changes(self) -> int:
        """Count total number of changes."""
        return len(self.added_genes) + len(self.removed_genes) + len(self.modified_genes)


def compute_delta(
    old_individual: List[SessionGene],
    new_individual: List[SessionGene]
) -> IndividualDelta:
    """
    Compute changes between two individuals.
    
    Complexity: O(S) with hash-based comparison
    """
    # Convert to sets for fast comparison
    old_set = {_gene_key(g): g for g in old_individual}
    new_set = {_gene_key(g): g for g in new_individual}
    
    # Find differences
    old_keys = set(old_set.keys())
    new_keys = set(new_set.keys())
    
    added_keys = new_keys - old_keys
    removed_keys = old_keys - new_keys
    common_keys = old_keys & new_keys
    
    added_genes = [new_set[k] for k in added_keys]
    removed_genes = [old_set[k] for k in removed_keys]
    modified_genes = []
    
    # Check for modifications in common genes
    for key in common_keys:
        old_gene = old_set[key]
        new_gene = new_set[key]
        if not _genes_equal(old_gene, new_gene):
            modified_genes.append((old_gene, new_gene))
    
    # Collect affected resources
    all_changed_genes = added_genes + removed_genes + [g for _, g in modified_genes]
    
    affected_instructors = {g.instructor_id for g in all_changed_genes}
    affected_groups = {gid for g in all_changed_genes for gid in g.group_ids}
    affected_rooms = {g.room_id for g in all_changed_genes}
    affected_quanta = {q for g in all_changed_genes for q in g.quanta}
    affected_courses = {(g.course_id, g.course_type) for g in all_changed_genes}
    
    return IndividualDelta(
        added_genes=added_genes,
        removed_genes=removed_genes,
        modified_genes=modified_genes,
        affected_instructors=affected_instructors,
        affected_groups=affected_groups,
        affected_rooms=affected_rooms,
        affected_quanta=affected_quanta,
        affected_courses=affected_courses,
    )


def _gene_key(gene: SessionGene) -> tuple:
    """Create unique key for gene (for fast comparison)."""
    return (
        gene.course_id,
        gene.course_type,
        frozenset(gene.group_ids),
    )


def _genes_equal(g1: SessionGene, g2: SessionGene) -> bool:
    """Check if two genes are identical."""
    return (
        g1.course_id == g2.course_id
        and g1.course_type == g2.course_type
        and g1.instructor_id == g2.instructor_id
        and g1.room_id == g2.room_id
        and set(g1.group_ids) == set(g2.group_ids)
        and set(g1.quanta) == set(g2.quanta)
    )
```

### 1.3 Delta-Aware Evaluation

```python
from typing import Optional

def evaluate_with_delta(
    individual: List[SessionGene],
    context: SchedulingContext,
    previous_individual: Optional[List[SessionGene]] = None,
    previous_fitness: Optional[tuple[int, int]] = None,
) -> tuple[int, int]:
    """
    Evaluate individual with optional delta optimization.
    
    If previous_individual and previous_fitness are provided,
    only re-evaluate constraints affected by changes.
    
    Complexity:
        - Full evaluation: O(S × Q + N × D × Q_d)
        - Delta evaluation: O(modified_sessions + affected_resources)
    """
    # Full evaluation if no delta info
    if previous_individual is None or previous_fitness is None:
        return evaluate(individual, context.courses, context.instructors, 
                       context.groups, context.rooms)
    
    # Compute delta
    delta = compute_delta(previous_individual, individual)
    
    # If too many changes, full evaluation is faster
    if delta.total_changes > len(individual) * 0.3:  # >30% changed
        return evaluate(individual, context.courses, context.instructors,
                       context.groups, context.rooms)
    
    # Delta evaluation: start with previous fitness
    prev_hard, prev_soft = previous_fitness
    
    # Decode only changed sessions
    sessions = decode_individual(individual, context.courses, 
                                context.instructors, context.groups, 
                                context.rooms)
    
    # Re-evaluate only affected constraints
    hard_delta = _evaluate_hard_delta(sessions, delta, context)
    soft_delta = _evaluate_soft_delta(sessions, delta, context)
    
    return (prev_hard + hard_delta, prev_soft + soft_delta)


def _evaluate_hard_delta(
    sessions: List[CourseSession],
    delta: IndividualDelta,
    context: SchedulingContext
) -> int:
    """
    Re-evaluate only hard constraints affected by changes.
    
    Strategy:
    1. For exclusivity constraints: check only affected resources
    2. For global constraints: re-evaluate fully (but fast O(S))
    """
    penalty_delta = 0
    
    # Instructor exclusivity: check only affected instructors
    if delta.affected_instructors:
        penalty_delta += _check_instructor_exclusivity_delta(
            sessions, delta.affected_instructors
        )
    
    # Room exclusivity: check only affected rooms
    if delta.affected_rooms:
        penalty_delta += _check_room_exclusivity_delta(
            sessions, delta.affected_rooms
        )
    
    # Group exclusivity: check only affected groups
    if delta.affected_groups:
        penalty_delta += _check_group_exclusivity_delta(
            sessions, delta.affected_groups
        )
    
    # Qualifications: check only affected courses
    if delta.affected_courses:
        penalty_delta += _check_qualifications_delta(
            sessions, delta.affected_courses, context.courses
        )
    
    # Availability constraints: check only affected resources
    # ... similar pattern
    
    return penalty_delta


def _check_instructor_exclusivity_delta(
    sessions: List[CourseSession],
    affected_instructors: Set[str]
) -> int:
    """
    Check instructor exclusivity only for affected instructors.
    
    Complexity: O(affected_sessions × Q) instead of O(S × Q)
    """
    conflicts = 0
    instructor_time_map = {}
    
    # Filter to only affected instructors' sessions
    affected_sessions = [
        s for s in sessions 
        if s.instructor_id in affected_instructors
    ]
    
    for session in affected_sessions:
        iid = session.instructor_id
        for q in session.session_quanta:
            key = (iid, q)
            if key in instructor_time_map:
                conflicts += 1
            else:
                instructor_time_map[key] = session.course_id
    
    return conflicts


# Similar pattern for other constraints...
```

### 1.4 Impact Analysis

**Typical Mutation:** Changes 1-3 genes out of 150 (1-2%)

| Constraint Type | Full Eval | Delta Eval | Speedup |
|----------------|-----------|------------|---------|
| Exclusivity (instructor) | O(S × Q) | O(affected_S × Q) | **50-100×** |
| Exclusivity (room) | O(S × Q) | O(affected_S × Q) | **50-100×** |
| Exclusivity (group) | O(S × Q) | O(affected_S × Q) | **50-100×** |
| Qualifications | O(S) | O(affected_S) | **50-100×** |
| Compactness | O(G × D × Q_d) | O(affected_G × D × Q_d) | **10-20×** |
| **Overall** | **40ms** | **2-5ms** | **8-20×** |

**When to Use:**
- ✅ Mutation operators (typically 1-5% genes changed)
- ✅ RL single-action steps (1 gene modified)
- ❌ Crossover (typically 50% genes changed) - use full eval
- ❌ Initial population - no previous fitness

---

## 2. Constraint Caching

### 2.1 Session-Level Caching

```python
from functools import lru_cache
from typing import Hashable

@dataclass(frozen=True)
class SessionKey:
    """Immutable key for session caching."""
    course_id: str
    course_type: str
    instructor_id: str
    room_id: str
    group_ids: FrozenSet[str]
    quanta: FrozenSet[int]
    
    @staticmethod
    def from_gene(gene: SessionGene) -> 'SessionKey':
        return SessionKey(
            course_id=gene.course_id,
            course_type=gene.course_type,
            instructor_id=gene.instructor_id,
            room_id=gene.room_id,
            group_ids=frozenset(gene.group_ids),
            quanta=frozenset(gene.quanta),
        )


class CachedConstraintChecker:
    """Cache constraint results at session level."""
    
    def __init__(self, context: SchedulingContext):
        self.context = context
        self._qualification_cache = {}
        self._suitability_cache = {}
        self._availability_cache = {}
    
    def check_qualification(self, session_key: SessionKey) -> bool:
        """Check if cached, otherwise evaluate."""
        if session_key not in self._qualification_cache:
            # Evaluate and cache
            course_key = (session_key.course_id, session_key.course_type)
            course = self.context.courses[course_key]
            qualified = course.qualified_instructor_ids
            result = session_key.instructor_id in qualified
            self._qualification_cache[session_key] = result
        
        return self._qualification_cache[session_key]
    
    def clear_cache(self, affected_keys: Set[SessionKey] = None):
        """Clear cache (optionally only for affected keys)."""
        if affected_keys is None:
            self._qualification_cache.clear()
            self._suitability_cache.clear()
            self._availability_cache.clear()
        else:
            for key in affected_keys:
                self._qualification_cache.pop(key, None)
                self._suitability_cache.pop(key, None)
                self._availability_cache.pop(key, None)
```

### 2.2 Population-Level Caching

```python
class PopulationConstraintCache:
    """Cache constraint results across population."""
    
    def __init__(self, max_cache_size: int = 1000):
        self.max_cache_size = max_cache_size
        self._fitness_cache: Dict[FrozenSet[SessionKey], tuple[int, int]] = {}
    
    def get_fitness(self, individual: List[SessionGene]) -> Optional[tuple[int, int]]:
        """Get cached fitness if available."""
        key = self._individual_key(individual)
        return self._fitness_cache.get(key)
    
    def set_fitness(self, individual: List[SessionGene], fitness: tuple[int, int]):
        """Cache fitness result."""
        if len(self._fitness_cache) >= self.max_cache_size:
            # Evict random entry (simple LRU alternative)
            self._fitness_cache.pop(next(iter(self._fitness_cache)))
        
        key = self._individual_key(individual)
        self._fitness_cache[key] = fitness
    
    @staticmethod
    def _individual_key(individual: List[SessionGene]) -> FrozenSet[SessionKey]:
        """Create hashable key for individual."""
        return frozenset(SessionKey.from_gene(g) for g in individual)
```

**Impact:** 2-3× speedup for duplicate evaluations (common in elitism, steady-state GA)

---

## 3. Data Structure Improvements

### 3.1 Convert Lists to Sets (Membership Testing)

**Current Problem:** `qualified_instructor_ids` is a list → O(I) membership test

**Solution:**

```python
# In src/entities/course.py
@dataclass
class Course:
    qualified_instructor_ids: Set[str]  # Was: List[str]
    enrolled_group_ids: Set[str]        # Was: List[str]
    
    # Convert in encoder
    def __post_init__(self):
        if isinstance(self.qualified_instructor_ids, list):
            self.qualified_instructor_ids = set(self.qualified_instructor_ids)
        if isinstance(self.enrolled_group_ids, list):
            self.enrolled_group_ids = set(self.enrolled_group_ids)
```

**Impact:** O(S × I) → O(S) for qualifications (1-2ms improvement)

### 3.2 Pre-Build Resource Indices

```python
@dataclass
class SchedulingIndex:
    """Pre-built indices for fast constraint checking."""
    
    # Instructor → sessions mapping
    instructor_sessions: Dict[str, List[CourseSession]]
    
    # Room → sessions mapping
    room_sessions: Dict[str, List[CourseSession]]
    
    # Group → sessions mapping
    group_sessions: Dict[str, List[CourseSession]]
    
    # Course → sessions mapping
    course_sessions: Dict[tuple, List[CourseSession]]
    
    # Time → sessions mapping (for conflict detection)
    quantum_sessions: Dict[int, List[CourseSession]]
    
    @staticmethod
    def build(sessions: List[CourseSession]) -> 'SchedulingIndex':
        """Build indices from sessions (O(S × Q) one-time cost)."""
        instructor_sessions = defaultdict(list)
        room_sessions = defaultdict(list)
        group_sessions = defaultdict(list)
        course_sessions = defaultdict(list)
        quantum_sessions = defaultdict(list)
        
        for session in sessions:
            instructor_sessions[session.instructor_id].append(session)
            room_sessions[session.room.room_id].append(session)
            
            for gid in session.group_ids:
                group_sessions[gid].append(session)
            
            course_key = (session.course_id, session.course_type)
            course_sessions[course_key].append(session)
            
            for q in session.session_quanta:
                quantum_sessions[q].append(session)
        
        return SchedulingIndex(
            instructor_sessions=dict(instructor_sessions),
            room_sessions=dict(room_sessions),
            group_sessions=dict(group_sessions),
            course_sessions=dict(course_sessions),
            quantum_sessions=dict(quantum_sessions),
        )


def instructor_exclusivity_indexed(index: SchedulingIndex) -> int:
    """
    Check instructor exclusivity using pre-built index.
    
    Complexity: O(I × avg_sessions × Q) instead of O(S × Q)
    But only faster if avg_sessions << S
    """
    conflicts = 0
    
    for instructor_id, sessions in index.instructor_sessions.items():
        time_map = {}
        for session in sessions:
            for q in session.session_quanta:
                if q in time_map:
                    conflicts += 1
                else:
                    time_map[q] = session.course_id
    
    return conflicts
```

**Impact:** Minimal for current implementation (hash maps already used effectively)

### 3.3 Interval Trees for Temporal Conflicts

```python
from intervaltree import IntervalTree, Interval

class TemporalIndex:
    """Use interval trees for fast overlap detection."""
    
    def __init__(self):
        # Separate tree per resource per day
        self.instructor_trees: Dict[str, Dict[int, IntervalTree]] = defaultdict(
            lambda: defaultdict(IntervalTree)
        )
        self.room_trees: Dict[str, Dict[int, IntervalTree]] = defaultdict(
            lambda: defaultdict(IntervalTree)
        )
    
    def add_session(self, session: CourseSession):
        """Add session to interval trees."""
        if not session.session_quanta:
            return
        
        # Group quanta by day
        quanta_by_day = defaultdict(list)
        for q in session.session_quanta:
            day = q // QUANTA_PER_DAY
            within_day = q % QUANTA_PER_DAY
            quanta_by_day[day].append(within_day)
        
        # Add intervals
        for day, quanta in quanta_by_day.items():
            if not quanta:
                continue
            start, end = min(quanta), max(quanta) + 1
            
            # Add to instructor tree
            self.instructor_trees[session.instructor_id][day].add(
                Interval(start, end, session)
            )
            
            # Add to room tree
            self.room_trees[session.room.room_id][day].add(
                Interval(start, end, session)
            )
    
    def check_conflicts(self) -> int:
        """Check all conflicts using interval trees."""
        conflicts = 0
        
        # Check instructor conflicts
        for instructor_id, day_trees in self.instructor_trees.items():
            for day, tree in day_trees.items():
                # Find overlapping intervals
                overlaps = tree.overlap_size()  # Custom method
                if overlaps > 0:
                    conflicts += overlaps
        
        return conflicts
```

**Impact:** O(S × Q) → O(S × log(S)) for conflict detection (10-20% improvement for large S)

**Tradeoff:** Added complexity, external dependency (intervaltree package)

---

## 4. Parallel Constraint Evaluation

### 4.1 Independent Constraint Parallelization

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

def evaluate_parallel(
    individual: List[SessionGene],
    context: SchedulingContext,
    num_workers: int = 4
) -> tuple[int, int]:
    """
    Evaluate constraints in parallel.
    
    Independent constraints can be evaluated simultaneously.
    """
    sessions = decode_individual(individual, context.courses,
                                context.instructors, context.groups,
                                context.rooms)
    
    # Group independent constraints
    hard_constraint_funcs = [
        (name, info["function"], info["weight"])
        for name, info in get_enabled_hard_constraints().items()
    ]
    
    soft_constraint_funcs = [
        (name, info["function"], info["weight"])
        for name, info in get_enabled_soft_constraints().items()
    ]
    
    # Evaluate in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit hard constraints
        hard_futures = {
            executor.submit(_eval_constraint, func, sessions, context, name): 
            (name, weight)
            for name, func, weight in hard_constraint_funcs
        }
        
        # Submit soft constraints
        soft_futures = {
            executor.submit(_eval_constraint, func, sessions, context, name):
            (name, weight)
            for name, func, weight in soft_constraint_funcs
        }
        
        # Collect results
        hard_penalty = 0
        for future in as_completed(hard_futures):
            name, weight = hard_futures[future]
            penalty = future.result()
            hard_penalty += weight * penalty
        
        soft_penalty = 0
        for future in as_completed(soft_futures):
            name, weight = soft_futures[future]
            penalty = future.result()
            soft_penalty += weight * penalty
    
    return (hard_penalty, soft_penalty)


def _eval_constraint(
    constraint_func: Callable,
    sessions: List[CourseSession],
    context: SchedulingContext,
    name: str
) -> int:
    """Evaluate single constraint (worker function)."""
    from src.constraints.registry import constraint_needs_courses
    
    if constraint_needs_courses(name):
        return constraint_func(sessions, context.courses)
    else:
        return constraint_func(sessions)
```

**Impact:** 2-4× speedup with 4 workers (diminishing returns due to overhead)

**Tradeoff:** Process creation overhead (~10-20ms), best for large evaluations

---

## 5. Compiled Extensions (Cython)

### 5.1 Cython Implementation Example

```python
# constraints_fast.pyx (Cython)
from libc.stdlib cimport malloc, free
from cpython cimport array
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int c_instructor_exclusivity(
    int[:] instructor_ids,
    int[:, :] quanta_arrays,
    int n_sessions,
    int max_quanta_per_session
):
    """
    Cython-optimized instructor exclusivity check.
    
    Uses C arrays and tight loops for 2-3× speedup.
    """
    cdef int conflicts = 0
    cdef int i, j, q
    cdef int iid
    
    # Use hash map (dict) for conflict detection
    time_map = {}
    
    for i in range(n_sessions):
        iid = instructor_ids[i]
        for j in range(max_quanta_per_session):
            q = quanta_arrays[i, j]
            if q == -1:  # Sentinel for end of quanta
                break
            
            key = (iid, q)
            if key in time_map:
                conflicts += 1
            else:
                time_map[key] = i
    
    return conflicts


# Python wrapper
def instructor_exclusivity_fast(sessions):
    """Python interface to Cython function."""
    # Convert to C arrays
    instructor_ids = array.array('i', [hash(s.instructor_id) for s in sessions])
    # ... prepare quanta_arrays ...
    
    return c_instructor_exclusivity(
        instructor_ids, quanta_arrays, len(sessions), max_quanta
    )
```

**Impact:** 2-3× speedup for hot constraint functions

**Tradeoff:** Compilation complexity, platform-specific builds

---

## 6. Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
1. ✅ Convert qualification lists to sets
2. ✅ Remove unnecessary sorting in soft constraints
3. ✅ Add fitness caching for duplicates

**Expected Impact:** 15-20% speedup

### Phase 2: Delta Evaluation (1 week)
1. Implement `compute_delta()` function
2. Add delta-aware constraint functions
3. Integrate with GA mutation operators
4. Add unit tests for correctness

**Expected Impact:** 5-10× speedup for mutations

### Phase 3: Advanced Optimizations (2-3 weeks)
1. Implement parallel evaluation
2. Add interval trees for temporal indexing
3. Explore Cython compilation

**Expected Impact:** Additional 2-4× speedup

---

## 7. Risk Assessment

| Optimization | Complexity | Risk | Reward |
|-------------|-----------|------|---------|
| Lists → Sets | Low | Very Low | Low (1-2ms) |
| Remove Sorting | Low | Very Low | Medium (5-10ms) |
| Fitness Caching | Medium | Low | Medium (2-3×) |
| Delta Evaluation | High | Medium | High (5-10×) |
| Parallel Eval | Medium | Medium | Medium (2-4×) |
| Cython | High | High | Medium (2-3×) |

**Recommendation:** Start with Phase 1 (quick wins), then implement Phase 2 (delta evaluation) for maximum impact.

---

## 8. Code Sketches for Key Optimizations

See implementation examples in:
- Section 1.2: Change tracking (`compute_delta`)
- Section 1.3: Delta-aware evaluation
- Section 2.1: Session-level caching
- Section 3.2: Pre-built indices
- Section 4.1: Parallel evaluation

**Next Steps:**
1. Review and approve optimization strategy
2. Implement Phase 1 optimizations
3. Benchmark before/after (see `03_BENCHMARK_GUIDE.md`)
4. Validate correctness with unit tests

---

## References

- Complexity Analysis: `01_COMPLEXITY_ANALYSIS.md`
- Benchmark Guide: `03_BENCHMARK_GUIDE.md`
- Implementation: `04_DELTA_EVALUATION_IMPL.md` (to be created)
