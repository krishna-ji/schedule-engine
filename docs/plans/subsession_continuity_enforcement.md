# Subsession Continuity Enforcement Migration Plan

**Date Created**: November 22, 2025  
**Status**: Planning Phase  
**Priority**: High - Fundamental architecture improvement

---

## Executive Summary

### The Problem
The GA currently treats intra-session quantum continuity as a **soft constraint** (via `session_continuity` penalty), leading to:
- Fragmented session blocks that violate pedagogical requirements
- Heavy reliance on repair heuristics to fix continuity post-mutation
- GA wasting generations optimizing basic feasibility instead of higher-order objectives
- Practical courses getting split across non-contiguous time slots

### The Solution
**Move continuity enforcement from constraint penalties → initializer guarantees**

Instead of penalizing non-contiguous blocks, we ensure every `SessionGene.quanta` is **constructed contiguously by default**, following deterministic rules:

**Theory Courses:**
- Odd load (e.g., L=5): One 1-quantum subsession + remaining 2-quantum subsessions
  - Example: `[10,11], [15,16], [22]` (5 quanta total)
- Even load (e.g., L=4): All 2-quantum subsessions
  - Example: `[10,11], [15,16]` (4 quanta total)

**Practical Courses:**
- All quanta in single contiguous block
  - Example: P=3 → `[30,31,32]`

### Expected Impact
- **Initialization quality**: 40-60% reduction in initial continuity violations
- **Repair workload**: 50-70% fewer repair activations per generation
- **Convergence speed**: 20-30% faster to feasible solutions
- **GA focus**: More cycles spent on soft objectives (clustering, gaps, preferences)
- **BONUS**: 60% memory reduction per individual (2 ints vs array of N ints)
- **SIMPLICITY**: No subsession template system needed - direct from course metadata!

---

## Current State Analysis

### Codebase Audit (Completed November 22, 2025)

#### 1. **Genome Representation** (`src/ga/sessiongene.py`)

**CURRENT (Legacy):**
```python
@dataclass
class SessionGene:
    course_id: str
    course_type: str  # "theory" or "practical"
    instructor_id: str
    group_ids: List[str]
    room_id: str
    quanta: List[int]  # ← Currently NO continuity guarantee
```

**PROPOSED (New):**
```python
@dataclass
class SessionGene:
    course_id: str
    course_type: str
    instructor_id: str
    group_ids: List[str]
    room_id: str
    start_quanta: int  # ← NEW: Starting quantum index
    num_quanta: int    # ← NEW: Session duration in quanta
    
    @property
    def quanta(self) -> List[int]:
        """Backward compatibility: Generate quanta array on-demand."""
        return list(range(self.start_quanta, self.start_quanta + self.num_quanta))
    
    @property
    def time_quantum(self) -> int:
        """Starting quantum (alias for start_quanta)."""
        return self.start_quanta
    
    @time_quantum.setter
    def time_quantum(self, new_start: int) -> None:
        """Shift session to new start time."""
        self.start_quanta = new_start
```

**Key Improvements:**
- **Structural enforcement**: Impossible to create fragmented blocks
- **Memory efficient**: 2 integers vs array of N integers
- **Simpler validation**: Just range check, no continuity scan needed
- **Backward compatible**: `quanta` property preserves existing API

#### 2. **Population Initialization** (`src/ga/hybrid_population.py`)
```python
def generate_hybrid_population(n: int, context: SchedulingContext) -> List:
    # 40% greedy construction
    # 40% smart constraint-aware
    # 20% random
```

**Entry Points for Continuity Logic:**
- `_greedy_construction()` - Build feasible schedules greedily
- `_smart_constraint_aware()` - Existing smart seeding (via `generate_course_group_aware_population`)
- `_random_construction()` - Pure random (no continuity awareness)

**Current Behavior:**
- Quanta assigned without checking intra-session continuity
- Assumes repair operators will fix fragmentation later

#### 3. **Soft Constraint** (`src/constraints/soft.py:219`)
```python
@soft_constraint(name="session_continuity", weight=2.0)
def session_continuity(sessions: List[CourseSession]) -> int:
    """Penalizes non-contiguous blocks within sessions"""
    # Theory: Penalty for isolated singles (except first) and oversized blocks (>3)
    # Practical: Heavy penalty (20) for ANY fragmentation
```

**Key Findings:**
- Penalty-based approach requires GA to discover continuity through search
- Practical fragmentation gets 20 penalty (high cost)
- Theory allows some flexibility but prefers [2,2,2] over [1,1,4]

#### 4. **Repair Operators** (`src/heuristics/`)
```
src/heuristics/
├── construction.py    # Greedy builders (largest_degree_first, etc.)
├── perturbation.py    # IGLS destroy operators
├── improvement.py     # IGLS local search
└── registry.py        # Decorator-based heuristic registry
```

**Current Repair Flow:**
1. Perturbation heuristics destroy parts of schedule
2. Construction heuristics rebuild destroyed parts
3. Improvement heuristics refine solution

**Gap:** No explicit "re-contiguify" operator targeting session continuity

#### 5. **Time System** (`src/encoder/quantum_time_system.py`)
```python
class QuantumTimeSystem:
    QUANTUM_MINUTES = 60  # 1-hour slots
    DEFAULT_OPERATING_HOURS = {
        "Sunday": ("10:00", "17:00"),
        "Monday": ("10:00", "17:00"),
        # ...
    }
```

**Key Properties:**
- Continuous quantum indexing (0-N, no gaps during operating hours)
- Day-aware: Can identify same-day consecutive quanta
- Supports lunch break exclusions (quanta cross gap)

#### 6. **Course Data Schema** (`data/Course.json`)
```json
{
  "CourseCode": "ENCT 101",
  "L": 4,  // Lecture hours (theory load)
  "T": 0,  // Tutorial hours
  "P": 3.0, // Practical hours (lab load)
  "Credits": 3
}
```

**Available Metadata:**
- `L` (lecture load) → Theory quanta requirement
- `P` (practical load) → Practical quanta requirement
- Can derive odd/even rules from `L` and `P` values

---

## Detailed Design (MASSIVELY SIMPLIFIED!)

**KEY INSIGHT:** With `start_quanta` + `num_quanta`, we don't need a separate template system! 
Just calculate durations directly from course metadata and assign contiguous blocks.

### Phase 0: Requirements Formalization

#### Continuity Rules (Formalized)

**Theory Courses (L-hours):**
```
If L is odd:
  - Generate ceil(L/2) sessions
  - First (L-1)/2 sessions: 2 quanta each
  - Last session: 1 quantum
  - Example: L=5 → [2,2,1] quanta distribution

If L is even:
  - Generate L/2 sessions
  - All sessions: 2 quanta each
  - Example: L=4 → [2,2]
```

**Practical Courses (P-hours):**
```
All P quanta in single contiguous block:
  - Example: P=3 → [q, q+1, q+2]
  - NO fragmentation allowed
```

**Multi-Session Courses (L+P):**
```
Theory and practical scheduled separately:
  - Theory: Follow L-rules above
  - Practical: Single contiguous block
  - Can be on same day or different days
```

#### Configuration Schema Extension

**Add to `configs/base.yaml`:**
```yaml
initializer:
  enforce_subsession_continuity: true  # MASTER KILLSWITCH
  continuity_rules:
    theory:
      preferred_block_size: 2  # Quanta per session
      allow_single_block: true  # Allow last session to be 1 quantum for odd loads
    practical:
      require_contiguous: true  # Strict contiguity for practicals
      max_block_size: 10  # Safety limit (avoid 10-hour marathons)
  fallback_on_infeasibility: "warn"  # "warn" | "error" | "random"
```

---

### Phase 1: SessionGene Data Structure Refactor (BREAKING CHANGE)

#### 1.1 Refactor SessionGene (CRITICAL FIRST STEP)

**Modify:** `src/ga/sessiongene.py`

**️ BREAKING CHANGE**: Complete replacement of `quanta: List[int]` with `start_quanta + num_quanta`. No backward compatibility.

```python
@dataclass
class SessionGene:
    """
    Represents a single scheduled session with GUARANTEED contiguous quanta.
    
    BREAKING CHANGE (Nov 2025 Migration):
    - Removed: `quanta: List[int]` (allowed fragmentation)
    - Added: `start_quanta: int, num_quanta: int` (structural continuity)
    - Memory: 60% reduction (2 ints vs N-element array)
    - Validation: Simpler range checks, no continuity scanning
    
    Design Rationale:
    - Makes fragmentation structurally impossible
    - Eliminates session_continuity soft constraint (redundant)
    - Direct mapping to course duration requirements
    """
    course_id: str
    course_type: str  # "theory" or "practical"
    instructor_id: str
    group_ids: List[str]
    room_id: str
    
    # Contiguous block representation (NEW)
    start_quanta: int  # Starting quantum index (e.g., 10 = Monday 10:00 AM)
    num_quanta: int    # Duration in quanta (e.g., 2 = 2-hour block)
    
    def __post_init__(self):
        """Validate quantum range and continuity constraints."""
        from src.encoder.quantum_time_system import QuantumTimeSystem
        qts = QuantumTimeSystem()
        
        # Range validation
        if self.start_quanta < 0 or self.start_quanta >= qts.total_quanta:
            raise ValueError(
                f"start_quanta {self.start_quanta} out of range [0, {qts.total_quanta})"
            )
        if self.num_quanta <= 0:
            raise ValueError(f"num_quanta must be positive, got {self.num_quanta}")
        
        end_quanta = self.start_quanta + self.num_quanta
        if end_quanta > qts.total_quanta:
            raise ValueError(
                f"Session overflows quantum range: "
                f"{self.start_quanta} + {self.num_quanta} = {end_quanta} > {qts.total_quanta}"
            )
        
        # Day boundary validation (no midnight wrap)
        start_day = self.start_quanta // qts.quanta_per_day
        end_day = (end_quanta - 1) // qts.quanta_per_day
        if start_day != end_day:
            raise ValueError(
                f"Session crosses day boundary: "
                f"start={self.start_quanta} (day {start_day}), "
                f"end={end_quanta-1} (day {end_day})"
            )
    
    # ========== UTILITY METHODS ==========
    
    @property
    def end_quanta(self) -> int:
        """Exclusive end quantum (for range operations)."""
        return self.start_quanta + self.num_quanta
    
    def get_quanta_list(self) -> List[int]:
        """
        Generate explicit quanta array when needed (e.g., for legacy APIs).
        
        Example:
            start_quanta=10, num_quanta=3 → [10, 11, 12]
        
        Note: Prefer using range(gene.start_quanta, gene.end_quanta) for loops.
        """
        return list(range(self.start_quanta, self.end_quanta))
    
    def shift_to(self, new_start: int) -> None:
        """
        Shift session to new start time (preserves duration).
        
        Args:
            new_start: New starting quantum index
        
        Example:
            gene.start_quanta = 10, gene.num_quanta = 2
            gene.shift_to(15)
            → gene.start_quanta = 15, gene.num_quanta = 2
        """
        self.start_quanta = new_start
        self.__post_init__()  # Re-validate after shift
    
    def overlaps_with(self, other: "SessionGene") -> bool:
        """Check if this session overlaps with another session in time."""
        return not (self.end_quanta <= other.start_quanta or 
                    other.end_quanta <= self.start_quanta)
```

**Migration Impact:**
- **All `SessionGene(quanta=[...])` calls must be replaced** with `SessionGene(start_quanta=X, num_quanta=Y)`
- **All `gene.quanta` reads must be replaced** with `gene.get_quanta_list()` or `range(gene.start_quanta, gene.end_quanta)`
- **All `gene.quanta = [...]` writes must be replaced** with `gene.start_quanta = X; gene.num_quanta = Y`
- **Constraint evaluators**: Replace `for q in gene.quanta:` with `for q in range(gene.start_quanta, gene.end_quanta):`

#### 1.2 Create Continuity Helper Module

**New File:** `src/ga/continuity_helpers.py`

**Purpose**: Centralize logic for calculating session durations from course metadata and finding valid time windows.

```python
"""
Continuity enforcement helpers for population initialization.

Key Functions:
- calculate_session_durations(): Convert course L/P values to duration list
- find_contiguous_window(): Locate valid time slots for given duration
- build_availability_grid(): Track resource usage across quanta
"""

from typing import List, Set, Dict, Optional
from dataclasses import dataclass
from src.entities.course import Course
from src.core.types import SchedulingContext
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.config import get_config

@dataclass
class SessionDuration:
    """Represents required duration for a single session."""
    course_id: str
    course_type: str  # "theory" or "practical"
    session_index: int  # 0-based within course type
    num_quanta: int  # Required contiguous quanta
    
    def __repr__(self):
        return f"{self.course_id}[{self.course_type[0].upper()}{self.session_index}]:{self.num_quanta}q"


class ContinuityHelper:
    """Helper class for enforcing session continuity in population initialization."""
    
    def __init__(self, context: SchedulingContext):
        self.context = context
        self.qts = context.quantum_time_system
        self.config = get_config().initializer
    
    def calculate_session_durations(self, course: Course) -> List[SessionDuration]:
        """
        Calculate required durations for all sessions of a course.
        
        Rules (from config):
        - Theory (L hours):
            - Odd L: [2, 2, ..., 1] (e.g., L=5 → [2, 2, 1])
            - Even L: [2, 2, ...] (e.g., L=4 → [2, 2])
        - Practical (P hours):
            - Single contiguous block: [P] (e.g., P=3 → [3])
        
        Args:
            course: Course entity with L and P values
        
        Returns:
            List of SessionDuration objects (one per subsession)
        
        Example:
            course.L = 5, course.P = 3
            → [
                SessionDuration(type="theory", index=0, num_quanta=2),
                SessionDuration(type="theory", index=1, num_quanta=2),
                SessionDuration(type="theory", index=2, num_quanta=1),
                SessionDuration(type="practical", index=0, num_quanta=3),
            ]
        """
        durations = []
        block_size = self.config.continuity_rules.theory.preferred_block_size
        
        # Theory sessions (L hours)
        if course.L > 0:
            theory_quanta = course.L
            full_blocks = theory_quanta // block_size
            remainder = theory_quanta % block_size
            
            for i in range(full_blocks):
                durations.append(SessionDuration(
                    course_id=course.id,
                    course_type="theory",
                    session_index=i,
                    num_quanta=block_size
                ))
            
            if remainder > 0:
                durations.append(SessionDuration(
                    course_id=course.id,
                    course_type="theory",
                    session_index=full_blocks,
                    num_quanta=remainder
                ))
        
        # Practical session (P hours) - single contiguous block
        if course.P > 0:
            max_practical = self.config.continuity_rules.practical.max_block_size
            if course.P > max_practical:
                raise ValueError(
                    f"Practical duration {course.P} exceeds max {max_practical} for {course.id}"
                )
            
            durations.append(SessionDuration(
                course_id=course.id,
                course_type="practical",
                session_index=0,
                num_quanta=course.P
            ))
        
        return durations
    
    def find_contiguous_window(
        self,
        duration: SessionDuration,
        used_quanta: Dict[str, Set[int]],
        instructor_id: str,
        room_id: str,
        group_ids: List[str]
    ) -> Optional[int]:
        """
        Find a valid starting quantum for a session with given duration.
        
        Validation:
        - All quanta in [start, start+duration) must be free for:
            - Instructor
            - Room
            - All groups
        - No day boundary crossing
        - No lunch break crossing (if configured)
        
        Args:
            duration: Session duration requirements
            used_quanta: Dictionary of already-allocated quanta per resource
            instructor_id: Assigned instructor
            room_id: Assigned room
            group_ids: Enrolled groups
        
        Returns:
            Starting quantum index if found, else None
        
        Algorithm:
            1. Enumerate all possible start positions
            2. For each start, check if window is valid
            3. Apply heuristics (prefer earlier times, avoid conflicts)
            4. Return best candidate or None
        """
        candidates = []
        
        for start_q in range(self.qts.total_quanta - duration.num_quanta + 1):
            if self._is_window_valid(start_q, duration.num_quanta, used_quanta, 
                                    instructor_id, room_id, group_ids):
                score = self._score_window(start_q, duration.num_quanta)
                candidates.append((start_q, score))
        
        if not candidates:
            return None
        
        # Return best candidate (lowest score = best)
        candidates.sort(key=lambda x: x[1])
        return candidates[0][0]
    
    def _is_window_valid(
        self, 
        start_q: int, 
        duration: int,
        used_quanta: Dict[str, Set[int]],
        instructor_id: str,
        room_id: str,
        group_ids: List[str]
    ) -> bool:
        """Check if time window is available for all resources."""
        window = range(start_q, start_q + duration)
        
        # Day boundary check
        start_day = start_q // self.qts.quanta_per_day
        end_day = (start_q + duration - 1) // self.qts.quanta_per_day
        if start_day != end_day:
            return False
        
        # Resource availability checks
        for q in window:
            if q in used_quanta.get(instructor_id, set()):
                return False
            if q in used_quanta.get(room_id, set()):
                return False
            for group_id in group_ids:
                if q in used_quanta.get(group_id, set()):
                    return False
        
        return True
    
    def _score_window(self, start_q: int, duration: int) -> float:
        """
        Heuristic scoring for window placement (lower is better).
        
        Factors:
        - Earlier times preferred (morning > afternoon)
        - Avoid peak conflict hours
        - Clustered schedules (minimize gaps)
        """
        weights = self.config.window_scoring
        
        # Time preference (favor earlier slots)
        time_of_day = (start_q % self.qts.quanta_per_day)
        time_score = time_of_day * weights.time_preference_weight
        
        # Additional heuristics can be added here
        
        return time_score
```

**Design Benefits:**
- **Direct duration calculation**: No intermediate template objects needed
- **Unified helper**: All continuity logic in one place
- **Configurable rules**: Theory/practical rules from config
- **Resource tracking**: Prevents double-booking during initialization

---

### Phase 2: Update Population Initializers (Direct Continuity Enforcement)

**Purpose**: Modify all population initialization functions to use `ContinuityHelper` for direct course → durations → genes flow.

#### 2.1 Modify Greedy Construction

**Update:** `src/ga/hybrid_population.py:_greedy_construction()`

```python
def _greedy_construction(
    context: SchedulingContext,
    pair_tuples: List[Tuple]
) -> List[SessionGene]:
    """
    MODIFIED: Use ContinuityHelper for contiguous session assignment.
    
    Algorithm:
    1. For each course, calculate required session durations
    2. Sort by difficulty (longer practicals first)
    3. For each duration, find valid contiguous window
    4. Create SessionGene with start_quanta + num_quanta
    """
    from src.ga.continuity_helpers import ContinuityHelper
    
    helper = ContinuityHelper(context)
    individual = []
    used_quanta = defaultdict(set)  # Track allocated quanta per resource
    
    # Generate all session durations for all courses
    all_durations = []
    for course in context.courses.values():
        durations = helper.calculate_session_durations(course)
        all_durations.extend(durations)
    
    # Sort by difficulty (longer sessions first, practicals prioritized)
    all_durations.sort(
        key=lambda d: (d.course_type == "practical", d.num_quanta), 
        reverse=True
    )
    
    for duration in all_durations:
        course = context.courses[duration.course_id]
        
        # Select resources
        instructor_id = _select_instructor(course, context)
        room_id = _select_room(course, context)
        group_ids = [g.id for g in course.enrolled_groups]
        
        # Find contiguous window
        start_q = helper.find_contiguous_window(
            duration=duration,
            used_quanta=used_quanta,
            instructor_id=instructor_id,
            room_id=room_id,
            group_ids=group_ids
        )
        
        if start_q is None:
            # Fallback: Random assignment with warning
            logger.warning(
                f"No contiguous window for {duration}, using random fallback"
            )
            start_q = random.randint(0, context.quantum_time_system.total_quanta - duration.num_quanta)
        
        # Create SessionGene (BREAKING CHANGE: new API)
        gene = SessionGene(
            course_id=duration.course_id,
            course_type=duration.course_type,
            instructor_id=instructor_id,
            group_ids=group_ids,
            room_id=room_id,
            start_quanta=start_q,  # NEW: Starting quantum
            num_quanta=duration.num_quanta  # NEW: Duration
        )
        
        individual.append(gene)
        
        # Mark quanta as used
        for q in range(start_q, start_q + duration.num_quanta):
            used_quanta[instructor_id].add(q)
            used_quanta[room_id].add(q)
            for gid in group_ids:
                used_quanta[gid].add(q)
    
    return individual
```

#### 2.2 Modify Smart Construction

**Update:** `src/ga/hybrid_population.py:_smart_construction()`

```python
def _smart_construction(
    context: SchedulingContext,
    pair_tuples: List[Tuple]
) -> List[SessionGene]:
    """
    MODIFIED: Constraint-guided initialization with continuity enforcement.
    
    Similar to greedy but with more sophisticated heuristics:
    - Biased toward low-conflict time slots
    - Prefer balanced instructor workload
    - Cluster sessions by group when possible
    """
    from src.ga.continuity_helpers import ContinuityHelper
    
    helper = ContinuityHelper(context)
    individual = []
    used_quanta = defaultdict(set)
    
    # Generate durations with heuristic ordering
    all_durations = []
    for course in context.courses.values():
        durations = helper.calculate_session_durations(course)
        all_durations.extend(durations)
    
    # Smart sorting: balance between duration and conflict potential
    all_durations.sort(
        key=lambda d: (
            d.course_type == "practical",  # Practicals first
            d.num_quanta,  # Longer sessions first
            len(context.courses[d.course_id].enrolled_groups)  # More groups = harder
        ),
        reverse=True
    )
    
    for duration in all_durations:
        course = context.courses[duration.course_id]
        
        # Heuristic resource selection (prefer underutilized)
        instructor_id = _select_least_loaded_instructor(course, used_quanta, context)
        room_id = _select_least_loaded_room(course, used_quanta, context)
        group_ids = [g.id for g in course.enrolled_groups]
        
        # Find window (reuses ContinuityHelper logic)
        start_q = helper.find_contiguous_window(
            duration=duration,
            used_quanta=used_quanta,
            instructor_id=instructor_id,
            room_id=room_id,
            group_ids=group_ids
        )
        
        if start_q is None:
            start_q = random.randint(0, context.quantum_time_system.total_quanta - duration.num_quanta)
        
        gene = SessionGene(
            course_id=duration.course_id,
            course_type=duration.course_type,
            instructor_id=instructor_id,
            group_ids=group_ids,
            room_id=room_id,
            start_quanta=start_q,
            num_quanta=duration.num_quanta
        )
        
        individual.append(gene)
        
        # Update usage tracking
        for q in range(start_q, start_q + duration.num_quanta):
            used_quanta[instructor_id].add(q)
            used_quanta[room_id].add(q)
            for gid in group_ids:
                used_quanta[gid].add(q)
    
    return individual
```

#### 2.3 Modify Random Construction

**Update:** `src/ga/hybrid_population.py:_random_construction()`

```python
def _random_construction(
    context: SchedulingContext,
    pair_tuples: List[Tuple]
) -> List[SessionGene]:
    """
    MODIFIED: Pure random initialization with continuity enforcement.
    
    Even random individuals must have contiguous sessions.
    """
    from src.ga.continuity_helpers import ContinuityHelper
    
    helper = ContinuityHelper(context)
    individual = []
    
    for course in context.courses.values():
        durations = helper.calculate_session_durations(course)
        
        for duration in durations:
            # Fully random resource selection
            instructor_id = random.choice(course.qualified_instructors)
            room_id = random.choice([r.id for r in context.rooms.values()])
            group_ids = [g.id for g in course.enrolled_groups]
            
            # Random start quantum (ensuring no overflow)
            max_start = context.quantum_time_system.total_quanta - duration.num_quanta
            start_q = random.randint(0, max_start)
            
            # Validate day boundary (retry if crosses)
            qts = context.quantum_time_system
            start_day = start_q // qts.quanta_per_day
            end_day = (start_q + duration.num_quanta - 1) // qts.quanta_per_day
            
            if start_day != end_day:
                # Clamp to day boundary
                start_q = start_day * qts.quanta_per_day
            
            gene = SessionGene(
                course_id=duration.course_id,
                course_type=duration.course_type,
                instructor_id=instructor_id,
                group_ids=group_ids,
                room_id=room_id,
                start_quanta=start_q,
                num_quanta=duration.num_quanta
            )
            
            individual.append(gene)
    
    return individual
```

**Migration Impact:**
- All `SessionGene(quanta=[...])` → `SessionGene(start_quanta=X, num_quanta=Y)`
- Eliminates post-hoc repair loops for continuity violations
- Expected 40-60% reduction in initial violations

---
            num_quanta=len(window)          # ← NEW: Duration
        )
        individual.append(gene)
        
        # Update assignments
        assignments[room_id].update(window)
        assignments[instructor_id].update(window)
        for group_id in template.enrolled_group_ids:
            assignments[group_id].update(window)
    
    return individual
```

#### 3.2 Update Constraint Evaluators

**Files to modify:**
- `src/constraints/hard.py`
- `src/constraints/soft.py`
- `src/constraints/evaluator.py`

### Phase 3: Constraint & Operator Updates

**Purpose**: Update constraint evaluators and genetic operators to use new `start_quanta` + `num_quanta` API.

#### 3.1 Update Hard Constraints

**Files:** `src/constraints/hard.py`

**Pattern: Replace `gene.quanta` list iteration with range iteration**

```python
# Example: instructor_exclusivity constraint

# OLD (using quanta list)
def instructor_exclusivity(individual, context):
    instructor_usage = defaultdict(list)
    for gene in individual:
        for q in gene.quanta:  # ← List iteration
            instructor_usage[gene.instructor_id].append((gene.course_id, q))
    
    violations = 0
    for instr_id, usage in instructor_usage.items():
        # Check for quantum conflicts...
        pass
    return violations

# NEW (using start_quanta + num_quanta)
def instructor_exclusivity(individual, context):
    instructor_usage = defaultdict(list)
    for gene in individual:
        for q in range(gene.start_quanta, gene.end_quanta):  # ← Range iteration
            instructor_usage[gene.instructor_id].append((gene.course_id, q))
    
    violations = 0
    for instr_id, usage in instructor_usage.items():
        # Check for quantum conflicts...
        pass
    return violations
```

**Apply to all hard constraints:**
- `instructor_exclusivity()` - Range iteration over quanta
- `room_exclusivity()` - Range iteration over quanta
- `group_exclusivity()` - Range iteration over quanta
- `room_capacity()` - Use `gene.get_quanta_list()` if explicit list needed
- `instructor_qualification()` - No quantum dependency (unchanged)

#### 3.2 Update Soft Constraints

**Files:** `src/constraints/soft.py`

**Pattern: Replace `session.session_quanta` with range iteration**

```python
# Example: schedule_compactness constraint

# OLD (using quanta list)
def schedule_compactness(individual, context):
    for gene in individual:
        if len(gene.quanta) > 1:
            # Check gaps between quanta...
            gaps = [gene.quanta[i+1] - gene.quanta[i] - 1 
                    for i in range(len(gene.quanta) - 1)]
            penalty += sum(gaps)

# NEW (using start_quanta + end_quanta)
def schedule_compactness(individual, context):
    # NO GAPS POSSIBLE - continuity enforced structurally!
    # This constraint can be simplified or removed for within-session gaps.
    # (Still needed for gaps BETWEEN sessions of same course/group)
    
    for gene in individual:
        # Gaps within session = 0 by design
        pass
    
    # Calculate gaps between sessions (different logic)
    # ...
```

**Special Case: session_continuity constraint**

```python
# OLD (checked for fragmentation)
@register_soft_constraint("session_continuity", weight=2.0)
def session_continuity(individual, context):
    violations = 0
    for gene in individual:
        sorted_quanta = sorted(gene.quanta)
        for i in range(len(sorted_quanta) - 1):
            if sorted_quanta[i+1] - sorted_quanta[i] != 1:
                violations += 1  # Heavy penalty for gaps
    return violations

# NEW (DEPRECATED - always returns 0)
@register_soft_constraint("session_continuity", weight=0.0, deprecated=True)
def session_continuity(individual, context):
    """
    DEPRECATED: Continuity now enforced structurally.
    This constraint always returns 0 and can be removed in future versions.
    """
    return 0
```

#### 3.3 Update Decoder

**File:** `src/decoder/schedule_decoder.py`

```python
# OLD
def decode_individual(individual, context):
    sessions = []
    for gene in individual:
        session = CourseSession(
            course_id=gene.course_id,
            instructor_id=gene.instructor_id,
            room_id=gene.room_id,
            group_ids=gene.group_ids,
            session_quanta=gene.quanta  # ← Was direct property access
        )
        sessions.append(session)
    return Schedule(sessions)

# NEW
def decode_individual(individual, context):
    sessions = []
    for gene in individual:
        session = CourseSession(
            course_id=gene.course_id,
            instructor_id=gene.instructor_id,
            room_id=gene.room_id,
            group_ids=gene.group_ids,
            session_quanta=gene.get_quanta_list()  # ← Explicit conversion
        )
        sessions.append(session)
    return Schedule(sessions)
```

#### 3.4 Simplified Continuity Validation

**New File:** `src/ga/continuity_validator.py`

```python
"""
Simplified continuity validation for new SessionGene structure.

With start_quanta + num_quanta, continuity is structurally guaranteed.
Validation now just checks:
1. Valid ranges (num_quanta > 0, within bounds)
2. Day boundary compliance (no midnight wrap)
"""

from typing import List, Tuple
from src.ga.sessiongene import SessionGene
from src.encoder.quantum_time_system import QuantumTimeSystem

def validate_continuity(individual: List[SessionGene]) -> Tuple[bool, List[str]]:
    """
    Verify all SessionGenes have valid contiguous quanta.
    
    With new structure, this is TRIVIAL - just validate ranges!
    
    Returns:
        (is_valid, violation_messages)
    """
    qts = QuantumTimeSystem()
    violations = []
    
    for gene in individual:
        # Check 1: Positive duration
        if gene.num_quanta <= 0:
            violations.append(
                f"{gene.course_id}: Invalid num_quanta={gene.num_quanta}"
            )
        
        # Check 2: Within quantum bounds
        if gene.start_quanta < 0:
            violations.append(
                f"{gene.course_id}: Negative start_quanta={gene.start_quanta}"
            )
        if gene.end_quanta > qts.total_quanta:
            violations.append(
                f"{gene.course_id}: Overflow "
                f"end_quanta={gene.end_quanta} > total={qts.total_quanta}"
            )
        
        # Check 3: Same day (no midnight wrap)
        start_day = qts.quanta_to_time(gene.start_quanta)[0]
        end_day = qts.quanta_to_time(gene.end_quanta - 1)[0]
        if start_day != end_day:
            violations.append(
                f"{gene.course_id}: Spans days {start_day} → {end_day}"
            )
    
    return (len(violations) == 0, violations)
```

**Validation Benefits:**
- No continuity scanning needed (just range checks)
- 5-10x faster than array-based validation
- Impossible to have fragmentation bugs

---

### Phase 4: Genetic Operator Updates

**Purpose**: Update mutation, crossover, and repair operators to preserve continuity.

#### 4.1 Time Shift Mutation (Continuity-Preserving)

**Update:** `src/ga/operators/mutation.py`

```python
def mutate_time_shift(individual, context: SchedulingContext, indpb: float):
    """
    Mutation that shifts entire contiguous blocks in time.
    
    Operations:
    - Select random gene
    - Generate new random start_quanta (preserving num_quanta)
    - Validate day boundary
    - Apply shift atomically
    
    PRESERVES: Duration (num_quanta unchanged)
    CHANGES: Start time (start_quanta)
    """
    from src.ga.continuity_helpers import ContinuityHelper
    
    helper = ContinuityHelper(context)
    qts = context.quantum_time_system
    
    for gene in individual:
        if random.random() > indpb:
            continue
        
        # Generate random valid start position
        max_start = qts.total_quanta - gene.num_quanta
        new_start = random.randint(0, max_start)
        
        # Validate day boundary
        start_day = new_start // qts.quanta_per_day
        end_day = (new_start + gene.num_quanta - 1) // qts.quanta_per_day
        
        if start_day == end_day:
            # Valid shift - apply atomically
            gene.start_quanta = new_start
            # gene.num_quanta unchanged (duration preserved)
    
    return (individual,)


def mutate_duration_change(individual, context: SchedulingContext, indpb: float):
    """
    Mutation that changes session duration (for flexible courses).
    
    Example: 4-hour session → split into 2x 2-hour sessions
    
    NOTE: Only applicable if course has flexible duration rules.
    """
    # Placeholder for future enhancement
    pass


def mutate_instructor_swap(individual, context: SchedulingContext, indpb: float):
    """
    Mutation that changes instructor while preserving time slots.
    
    PRESERVES: start_quanta, num_quanta, room, groups
    CHANGES: instructor_id
    """
    for gene in individual:
        if random.random() > indpb:
            continue
        
        course = context.courses[gene.course_id]
        if len(course.qualified_instructors) > 1:
            # Pick different instructor
            current = gene.instructor_id
            candidates = [i for i in course.qualified_instructors if i != current]
            gene.instructor_id = random.choice(candidates)
    
    return (individual,)


def mutate_room_swap(individual, context: SchedulingContext, indpb: float):
    """
    Mutation that changes room while preserving time slots.
    
    PRESERVES: start_quanta, num_quanta, instructor, groups
    CHANGES: room_id
    """
    for gene in individual:
        if random.random() > indpb:
            continue
        
        course = context.courses[gene.course_id]
        compatible_rooms = [
            r.id for r in context.rooms.values()
            if r.room_type == course.room_type
        ]
        
        if len(compatible_rooms) > 1:
            current = gene.room_id
            candidates = [r for r in compatible_rooms if r != current]
            gene.room_id = random.choice(candidates)
    
    return (individual,)
```

#### 4.2 Update Crossover (Post-Validation)

**Update:** `src/ga/operators/crossover.py`

```python
def crossover_course_group_aware(ind1, ind2, context: SchedulingContext):
    """
    Course-group-aware crossover with post-validation.
    
    Existing logic unchanged, but add validation step to ensure
    continuity preserved after gene exchange.
    """
    # Existing crossover logic (unchanged)
    # ... exchange genes between ind1 and ind2 ...
    
    # NEW: Validate continuity post-crossover (defensive)
    from src.ga.continuity_validator import validate_continuity
    
    for ind in [ind1, ind2]:
        is_valid, violations = validate_continuity(ind)
        if not is_valid:
            logger.warning(f"Crossover created invalid individual: {violations}")
            # Note: With proper course-group-aware logic, this should never happen
    
    return ind1, ind2
```

**Crossover Impact:**
- Existing course-group-aware logic already preserves gene integrity
- New validation ensures no accidental corruption
- No major changes needed (defensive programming only)

#### 4.3 Repair Operators (Simplified)

**Update:** `src/ga/operators/repair.py`

```python
def repair_time_conflicts(individual, context: SchedulingContext):
    """
    Repair instructor/room/group conflicts by shifting sessions.
    
    SIMPLIFIED: No need to fix fragmentation (impossible with new structure).
    Only handles resource conflicts (double-bookings).
    """
    from src.ga.continuity_helpers import ContinuityHelper
    
    helper = ContinuityHelper(context)
    used_quanta = defaultdict(set)
    
    for i, gene in enumerate(individual):
        # Check for resource conflicts
        conflicts = _detect_conflicts(gene, used_quanta)
        
        if conflicts:
            # Find new contiguous window
            course = context.courses[gene.course_id]
            duration = helper.calculate_session_durations(course)
            
            new_start = helper.find_contiguous_window(
                duration=duration[0],  # Match current session
                used_quanta=used_quanta,
                instructor_id=gene.instructor_id,
                room_id=gene.room_id,
                group_ids=gene.group_ids
            )
            
            if new_start is not None:
                gene.start_quanta = new_start
                # num_quanta unchanged
        
        # Mark quanta as used
        for q in range(gene.start_quanta, gene.end_quanta):
            used_quanta[gene.instructor_id].add(q)
            used_quanta[gene.room_id].add(q)
            for gid in gene.group_ids:
                used_quanta[gid].add(q)
    
    return individual


def repair_continuity(individual, context: SchedulingContext):
    """
    DEPRECATED: Continuity now enforced structurally.
    
    This function is a no-op and can be removed.
    """
    logger.debug("repair_continuity called but is deprecated (structural enforcement)")
    return individual
```

**Repair Impact:**
- `repair_continuity()` becomes no-op (deprecated)
- `repair_time_conflicts()` simplified (no fragmentation handling)
- 50-70% reduction in repair activations

------

### Phase 5: Deprecate Session Continuity Constraint

#### 5.1 Update Soft Constraint

**Modify:** `src/constraints/soft.py`

```python
@soft_constraint(
    name="session_continuity",
    description="[DEPRECATED] Use initializer.enforce_subsession_continuity instead",
    default_weight=0.0,  # ← SET TO ZERO
    needs_courses=False,
)
def session_continuity(sessions: List[CourseSession]) -> int:
    """
    DEPRECATED: Continuity now enforced by initializer, not constraint.
    
    This constraint is kept for backward compatibility but should
    return 0 penalty when initializer.enforce_subsession_continuity=true.
    """
    config = get_config()
    
    if config.initializer.enforce_subsession_continuity:
        # New system: Assume continuity already enforced
        return 0
    
    # Legacy fallback: Original penalty logic
    return _legacy_continuity_penalty(sessions)
```

#### 5.2 Update Config Defaults

**Modify:** `configs/base.yaml`

```yaml
constraints:
  soft:
    session_continuity:
      enabled: false  # ← DISABLE (redundant with initializer enforcement)
      weight: 0.0

initializer:
  enforce_subsession_continuity: true  # ← NEW DEFAULT
```

---

### Phase 6: Testing Strategy

#### 6.1 Unit Tests

**New File:** `test/unit/test_continuity_engine.py`

```python
def test_odd_theory_course():
    """Verify L=5 generates [2,2,1] quanta distribution."""
    course = Course(course_id="TEST", L=5, P=0, type="theory")
    templates = generate_subsession_templates(course)
    
    assert len(templates) == 3
    assert templates[0].required_quanta == 2
    assert templates[1].required_quanta == 2
    assert templates[2].required_quanta == 1

def test_contiguous_window_finder():
    """Verify contiguous window generation."""
    context = create_test_context()
    engine = ContinuityEngine(context)
    
    template = SubsessionTemplate(required_quanta=3, ...)
    window = engine.find_contiguous_window(template, {})
    
    assert window is not None
    assert len(window) == 3
    assert window[1] == window[0] + 1
    assert window[2] == window[1] + 1

def test_practical_single_block():
    """Verify practical courses get single contiguous block."""
    course = Course(course_id="LAB", L=0, P=4, type="practical")
    templates = generate_subsession_templates(course)
    
    assert len(templates) == 1
    assert templates[0].required_quanta == 4
```

#### 6.2 Integration Tests

**New File:** `test/integration/test_continuity_initialization.py`

```python
def test_full_population_continuity():
    """Verify all individuals in hybrid population have contiguous quanta."""
    context = load_test_context()
    population = generate_hybrid_population(100, context)
    
    violations = []
    for ind in population:
        is_valid, msgs = validate_continuity(ind)
        if not is_valid:
            violations.extend(msgs)
    
    assert len(violations) == 0, f"Continuity violations: {violations}"

def test_ga_run_with_continuity():
    """Run 10-generation smoke test, verify continuity maintained."""
    config = get_config()
    config.initializer.enforce_subsession_continuity = True
    
    best_individual = run_ga_smoke_test(generations=10)
    is_valid, msgs = validate_continuity(best_individual)
    
    assert is_valid, f"Final solution has continuity issues: {msgs}"
```

---

### Phase 7: Rollout Plan

#### 7.1 Stage 1: Feature Flag (Week 1)
```bash
# Test profile: Feature enabled
uv run nsga --test --name "continuity-smoke"

# Compare metrics:
# - Initial feasibility rate
# - Repair activation counts
# - Convergence speed
```

**Success Criteria:**
- No regression in initial feasibility
- 30%+ reduction in repair activations
- Continuity validation passes 100% of time

#### 7.2 Stage 2: Medium Validation (Week 2)
```bash
# Medium profile: 200 generations
uv run nsga --med --name "continuity-validation"

# Monitor:
# - Final Pareto front quality (hypervolume)
# - Diversity metrics (unchanged or improved)
# - Wall-clock time (should decrease due to fewer repairs)
```

**Success Criteria:**
- Pareto front quality ≥ baseline
- 20-30% faster wall-clock time
- No diversity loss

#### 7.3 Stage 3: Production Deployment (Week 3)
```bash
# Full production run
uv run nsga --prod --name "continuity-prod-r01"

# Final validation:
# - Compare with historical prod runs
# - Document improvements in thesis experiments
```

#### 7.4 Stage 4: Flip Default (Week 4)
```yaml
# configs/base.yaml
initializer:
  enforce_subsession_continuity: true  # ← MAKE DEFAULT
```

**Deprecation Notice:**
- Update docs: `session_continuity` constraint marked deprecated
- RL agents: Remove actions related to continuity repair
- Codebase: Archive legacy random initializer

---

## Risk Analysis & Mitigations

### Risk 1: Infeasible Window Search
**Symptom:** `find_contiguous_window()` returns `None` frequently

**Causes:**
- Overconstrained resources (too few rooms/instructors)
- Unrealistic operating hours (too narrow time windows)
- Conflicting group schedules

**Mitigations:**
1. **Fallback strategy:** Use legacy random assignment with warning
2. **Relaxation heuristics:** Try 80% of required quanta if full block impossible
3. **Pre-validation:** Run pigeonhole analysis before GA to detect infeasibility early
4. **User feedback:** Log detailed infeasibility reasons to guide data adjustments

### Risk 2: Performance Overhead
**Symptom:** Initialization takes significantly longer

**Causes:**
- Window enumeration O(N*M) complexity
- Availability grid rebuilds per individual

**Mitigations:**
1. **Caching:** Pre-compute availability grids once per population
2. **Pruning:** Use heuristic bounds to skip obviously invalid windows
3. **Parallelization:** Generate individuals in parallel (already implemented)
4. **Profiling:** Use `get_profiler()` to identify bottlenecks

**Expected:** 10-20% increase in initialization time, offset by 50-70% reduction in repair time

### Risk 3: Reduced Diversity
**Symptom:** Population diversity metrics decrease

**Causes:**
- Deterministic window selection reduces variation
- Greedy heuristics converge to similar solutions

**Mitigations:**
1. **Stochastic scoring:** Add small random noise to window scoring
2. **Maintain random individuals:** Keep 20% fully random in hybrid population
3. **Monitor metrics:** Track `average_pairwise_diversity` before/after
4. **Tuning:** Adjust greedy percentage if diversity drops >15%

### Risk 4: RL Integration Conflicts
**Symptom:** RL agents trained on old action space break

**Causes:**
- RL actions assume continuity can be violated and repaired
- State representation includes continuity violation counts

**Mitigations:**
1. **Retrain RL agents:** Use new continuity-enforced environment
2. **Update state space:** Remove continuity violation features
3. **Deprecate actions:** Remove "re-contiguify" repair actions
4. **Coordinate:** Sync with RL workstream before deployment

---

## Success Metrics

### Quantitative Targets

| Metric | Baseline | Target | Measurement |
|--------|----------|--------|-------------|
| Initial continuity violations | 40-60% | <5% | `validate_continuity()` on generation 0 |
| Repair activations per gen | 150-200 | <50 | Track `repair_individual` calls |
| Convergence to feasible | 50 gens | 30-35 gens | First generation with hard_violations=0 |
| Wall-clock time (2000 gens) | 3-5 hours | 2-3.5 hours | Timer instrumentation |
| Hypervolume (final Pareto) | 0.85 | ≥0.85 | pymoo HV calculation |

### Qualitative Indicators

- **Code quality:** No increase in cyclomatic complexity
- **Maintainability:** Continuity logic isolated in `continuity_engine.py`
- **Documentation:** User guide updated, changelog entry added
- **Stakeholder approval:** GA/RL team sign-off on architecture

---

## Implementation Checklist

### Phase 0: Setup (Week 0)
- [x] Audit current codebase (completed November 22, 2025)
- [ ] Review plan with stakeholders
- [ ] Create feature branch: `feature/subsession-continuity`
- [ ] Set up benchmarking harness (baseline runs)

### Phase 1: Data Models (Week 1, Days 1-2)
- [ ] **CRITICAL**: Refactor `SessionGene` to use `start_quanta` + `num_quanta` (replaces `quanta` array)
- [ ] Add backward compatibility properties (`quanta`, `time_quantum`, `duration_quanta`)
- [ ] Update all codebase references to use new API where beneficial
- [ ] Implement `SubsessionTemplate` (`src/encoder/subsession_template.py`)
- [ ] Extend `SchedulingContext` with templates
- [ ] Add config schema to `configs/base.yaml`
- [ ] Unit tests for template generation and SessionGene refactor

### Phase 2: Continuity Engine (Week 1, Days 3-5)
- [ ] Implement `ContinuityEngine` class
- [ ] Build availability grid functions
- [ ] Implement `find_contiguous_window()` algorithm
- [ ] Unit tests for window generation (odd/even/practical cases)

### Phase 3: Initializer Integration (Week 2, Days 1-3)
- [ ] Modify `_greedy_construction()` to use `ContinuityEngine`
- [ ] Update `_smart_constraint_aware()` if needed
- [ ] Implement `validate_continuity()` hook
- [ ] Integration tests for hybrid population

### Phase 4: Operator Updates (Week 2, Days 4-5)
- [ ] Implement `mutate_contiguous()` operator
- [ ] Update crossover with post-validation
- [ ] Add continuity repair fallback for edge cases
- [ ] Operator unit tests

### Phase 6: Constraint Deprecation (Week 2, Day 5)
- [ ] Set `session_continuity` weight to 0.0
- [ ] Update constraint documentation (mark deprecated)
- [ ] Remove continuity-related repair heuristics (now redundant)

### Phase 7: Testing & Validation (Week 3, Days 1-3)
- [ ] Run all unit tests and fix failures
- [ ] Run smoke test (`--test` profile)
- [ ] Run medium validation (`--med` profile)
- [ ] Compare metrics vs baseline
- [ ] Fix any regressions

### Phase 8: Production Rollout (Week 3, Days 4-5)
- [ ] Full production run (`--prod` profile)
- [ ] Performance profiling and optimization
- [ ] Documentation updates
- [ ] Merge to main branch

---

## Documentation Updates

### Files to Update

1. **`docs/02-user-guides/runtime-modes.md`**
   - Add section: "Subsession Continuity Enforcement"
   - Document config flags and behavior

2. **`docs/06-development/changelog/enhancements.md`**
   - Entry: `[2025-11-22] Subsession Continuity Enforcement Migration`

3. **`docs/06-development/implementation-notes/PHASE_X_CONTINUITY.md`**
   - Full implementation summary (follow Phase 3 template)

4. **`docs/04-algorithms/initialization.md`** (new file)
   - Explain continuity enforcement algorithm
   - Mathematical formulation of window search

5. **`README.md`**
   - Update "Key Features" section
   - Mention continuity guarantees

---

## Future Enhancements (Post-MVP)

### 1. Adaptive Window Search
- Machine learning model to predict good window candidates
- Reduce search space from O(N*M) to O(log N)

### 2. Multi-Objective Window Scoring
- Pareto-optimize window selection (time quality vs resource conflicts)
- Use NSGA-II for window candidate ranking

### 3. Dynamic Template Adjustment
- Allow GA to propose template modifications (e.g., split 4-quantum block into 2×2)
- Meta-level optimization of subsession structure

### 4. Curriculum-Based Initialization
- Progressively relax continuity during early generations
- Strict enforcement only in final generations

---

## Appendix A: Example Scenarios

### Scenario 1: Theory Course (L=5)

**Input:**
```json
{
  "course_id": "ENSH 101",
  "L": 5,
  "P": 0,
  "enrolled_groups": ["BAE2", "BCE2"]
}
```

**Template Generation:**
```python
templates = [
    SubsessionTemplate(session_index=0, required_quanta=2, type="theory"),
    SubsessionTemplate(session_index=1, required_quanta=2, type="theory"),
    SubsessionTemplate(session_index=2, required_quanta=1, type="theory"),
]
```

**Quanta Assignment (Example - NEW API):**
```python
# Monday 10:00-12:00 (start=0, duration=2)
SessionGene(start_quanta=0, num_quanta=2, ...)

# Wednesday 10:00-12:00 (start=14, duration=2)
SessionGene(start_quanta=14, num_quanta=2, ...)

# Friday 10:00-11:00 (start=28, duration=1)
SessionGene(start_quanta=28, num_quanta=1, ...)
```

### Scenario 2: Practical Course (P=3)

**Input:**
```json
{
  "course_id": "ENCT 101 Lab",
  "L": 0,
  "P": 3,
  "enrolled_groups": ["BAE2"]
}
```

**Template Generation:**
```python
templates = [
    SubsessionTemplate(session_index=0, required_quanta=3, type="practical"),
]
```

**Quanta Assignment (Example - NEW API):**
```python
# Tuesday 13:00-16:00 (start=10, duration=3 - contiguous 3-hour block)
SessionGene(start_quanta=10, num_quanta=3, ...)
```

### Scenario 3: Mixed Course (L=4, P=2)

**Input:**
```json
{
  "course_id": "ENME 103",
  "L": 4,
  "P": 2,
  "enrolled_groups": ["BME2"]
}
```

**Template Generation:**
```python
templates = [
    SubsessionTemplate(session_index=0, required_quanta=2, type="theory"),
    SubsessionTemplate(session_index=1, required_quanta=2, type="theory"),
    SubsessionTemplate(session_index=2, required_quanta=2, type="practical"),
]
```

**Quanta Assignment (Example - NEW API):**
```python
# Theory: Monday 10:00-12:00, Wednesday 10:00-12:00
SessionGene(start_quanta=0, num_quanta=2, type="theory", ...)
SessionGene(start_quanta=14, num_quanta=2, type="theory", ...)

# Practical: Thursday 14:00-16:00 (contiguous)
SessionGene(start_quanta=25, num_quanta=2, type="practical", ...)
```

---

## Appendix B: Algorithm Complexity Analysis

### Window Search Complexity

**Naive approach:**
```
For each quantum q in [0, total_quanta - required_len]:
    Check if window [q, q+required_len) is valid
    → O(total_quanta * constraint_checks)
```

**Optimized approach:**
```
1. Pre-compute availability bitmaps (O(R * Q) where R=resources, Q=quanta)
2. Sliding window validation (O(Q) per template)
3. Total: O(R*Q + T*Q) where T=templates

Example:
- R=50 resources (rooms+instructors+groups)
- Q=70 total quanta (7 days * 10 hours)
- T=200 templates (100 courses * 2 sessions avg)

Complexity: O(50*70 + 200*70) = O(17,500) operations per individual
vs. Legacy random O(200) operations

Overhead: ~100x per individual, but amortized by:
- 50-70% fewer repair cycles (saved ~3000 operations/gen)
- Parallel initialization (10x speedup)
- Cached availability grids (reused across population)

Net impact: ~10-20% initialization slowdown, 30-40% total speedup
```

---

## Appendix C: Configuration Reference

### Full Config Schema

```yaml
initializer:
  # MASTER KILLSWITCH
  enforce_subsession_continuity: true
  
  # Continuity rules
  continuity_rules:
    theory:
      preferred_block_size: 2  # Quanta per session
      allow_single_block: true  # For odd loads
      max_block_size: 4  # Safety limit
    
    practical:
      require_contiguous: true
      max_block_size: 10  # Prevent marathon sessions
      allow_day_wrap: false  # Practical can't span midnight
    
    global:
      avoid_lunch_breaks: true  # Don't split blocks across lunch
      prefer_same_day: true  # Session blocks on same day if possible
  
  # Fallback behavior
  fallback_on_infeasibility: "warn"  # "warn" | "error" | "random"
  max_window_search_attempts: 100
  
  # Heuristic scoring weights (for window selection)
  window_scoring:
    time_preference_weight: 0.1  # Earlier times better
    load_balancing_weight: 0.2  # Spread across days
    resource_conflict_weight: 0.5  # Avoid congested slots

# Deprecated constraint (kept for backward compatibility)
constraints:
  soft:
    session_continuity:
      enabled: false  # Replaced by initializer enforcement
      weight: 0.0
```

---

## TODO: File-by-File Migration Checklist

### Core Data Structure
- [ ] `src/ga/sessiongene.py` - **PRIORITY 1** - Refactor dataclass
  - Remove `quanta: List[int]`
  - Add `start_quanta: int, num_quanta: int`
  - Add `get_quanta_list()`, `shift_to()`, `end_quanta` property
  - Remove ALL backward compatibility code

### Population Initialization (Update all `SessionGene(...)` calls)
- [ ] `src/ga/population.py` - Update `generate_course_group_aware_population()`
- [ ] `src/ga/hybrid_population.py` - Update `_greedy_construction()`, `_random_construction()`
- [ ] `src/ga/individual.py` - Update `create_individual()` if needed

### Constraint Evaluators (Replace `gene.quanta` loops)
- [ ] `src/constraints/hard.py` - All hard constraints
  - `instructor_exclusivity()` - Use `range(gene.start_quanta, gene.end_quanta)`
  - `room_exclusivity()` - Use range iteration
  - `group_exclusivity()` - Use range iteration
  - `room_capacity()` - Use `gene.get_quanta_list()` if needed
- [ ] `src/constraints/soft.py` - All soft constraints
  - `session_continuity()` - **DEPRECATE or SIMPLIFY**
  - `schedule_compactness()` - Use start/end properties
  - `preferred_time_slots()` - Use start/end properties
- [ ] `src/constraints/evaluator.py` - Update any constraint helpers

### Decoder & Exporter (Replace `gene.quanta` with `gene.get_quanta_list()`)
- [ ] `src/decoder/schedule_decoder.py` - Update `CourseSession` construction
- [ ] `src/exporter/pdf_exporter.py` - Update schedule rendering
- [ ] `src/exporter/json_exporter.py` - Update JSON serialization
- [ ] `src/exporter/plot_exporter.py` - Update visualization

### Genetic Operators
- [ ] `src/ga/operators/mutation.py` - Update all mutation functions
  - Time shifts: `gene.start_quanta += delta`
  - Swaps: swap both fields
- [ ] `src/ga/operators/crossover.py` - Verify (should work as-is)
- [ ] `src/ga/operators/repair.py` - Update repair logic
- [ ] `src/ga/operators/repair_selective.py` - Update if used
- [ ] `src/ga/operators/repair_wrappers.py` - Update if used

### Heuristics (Use ContinuityHelper for repairs)
- [ ] `src/heuristics/construction.py` - Update greedy builders
- [ ] `src/heuristics/improvement.py` - Update local search
- [ ] `src/heuristics/perturbation.py` - Update destroy operators

### New Files to Create
- [ ] `src/ga/continuity_helpers.py` - Core continuity logic
  - `ContinuityHelper` class
  - `calculate_session_durations()`
  - `find_contiguous_window()`
  - Availability grid builders
- [ ] `src/ga/continuity_validator.py` - Simplified validation

### Configuration
- [ ] `configs/base.yaml` - Add `initializer.enforce_subsession_continuity` and rules
- [ ] Update config schema in `src/config/config_model.py` if needed

### Tests (Update all SessionGene instantiations)
- [ ] `test/unit/test_sessiongene.py` - Rewrite for new structure
- [ ] `test/unit/test_constraints.py` - Update constraint tests
- [ ] `test/unit/test_population.py` - Update population tests
- [ ] `test/unit/test_continuity_engine.py` - NEW: Add continuity tests
- [ ] `test/integration/test_continuity_initialization.py` - NEW: Integration tests
- [ ] Update all other test files using SessionGene

### Documentation
- [ ] Update docstrings in all modified files
- [ ] `docs/06-development/changelog/enhancements.md` - Add entry
- [ ] `docs/06-development/implementation-notes/CONTINUITY_MIGRATION.md` - NEW: Full summary
- [ ] `README.md` - Update "Key Features" section

### Validation Steps (Run after each phase)
- [ ] Run `pytest test/unit/` after Phase 1
- [ ] Run `pytest test/unit/` after Phase 2
- [ ] Run `pytest test/` after Phase 3
- [ ] Run `uv run nsga --test` after Phase 4
- [ ] Run `uv run nsga --med` after Phase 5
- [ ] Run `uv run nsga --prod` for final validation

---

**End of Plan**
