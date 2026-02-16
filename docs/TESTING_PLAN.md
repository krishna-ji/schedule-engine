# Comprehensive Testing Plan — schedule-engine

> **Scope**: Unit, integration, semantic, intent, and algorithm-correctness testing  
> **Components**: 14 constraints, 12+ repair operators, 26 heuristics, GA pipeline  
> **Goal**: Verify algorithms work correctly, not just syntax — "does this do what it's supposed to?"

---

## Testing Philosophy

| Layer | What it Tests | Example |
|-------|---------------|---------|
| **Unit** | Single function/class in isolation | `StudentGroupExclusivity.evaluate()` returns 0 when no overlap |
| **Semantic** | Does the output *mean* what it should? | A penalty of 3 means exactly 3 time-slot overlaps, not 3 genes |
| **Intent** | Does the algorithm achieve its *purpose*? | After `repair_group_overlaps()`, group conflicts should be 0 |
| **Integration** | Do components work together correctly? | Evaluator + Timetable + Constraint produce correct weighted fitness |
| **Algorithm** | Is the algorithm mathematically correct? | NSGA-II fronts satisfy non-domination; improvement never worsens fitness |

---

## Phase 0: Shared Test Infrastructure

**File**: `tests/conftest.py`

### Fixtures Needed

| Fixture | Purpose |
|---------|---------|
| `make_course()` | Create Course with configurable type/quanta/room features/groups/instructors |
| `make_instructor()` | Create Instructor with full-time/part-time/available_quanta control |
| `make_group()` | Create Group with student count and enrolled courses |
| `make_room()` | Create Room with capacity, features, optional available_quanta |
| `make_gene()` | Create SessionGene with full control over all 7 fields |
| `make_context()` | Build SchedulingContext from lists of the above |
| `make_timetable()` | Build Timetable from genes + context (shorthand) |
| `make_violation_free_tt()` | A timetable with zero hard constraint violations |
| `qts()` | Default QuantumTimeSystem (6 days, 7 quanta/day, 42 total) |

### Helper Functions

```python
def assert_constraint_zero(constraint, timetable):
    """Assert constraint evaluates to exactly 0 penalty."""

def assert_constraint_positive(constraint, timetable, expected=None):
    """Assert constraint evaluates to >0 penalty, optionally exact value."""

def count_hard_violations(timetable, context) -> int:
    """Sum all hard constraint violations for a timetable."""

def genes_differ_only_in(gene1, gene2, fields: set[str]) -> bool:
    """Assert two genes are identical except for specified mutable fields."""
```

### QuantumTimeSystem Defaults (reference)

- 6 operational days: Sunday–Friday
- 7 quanta/day: 10:00–17:00 (each quantum = 60 min)
- 42 total quanta: 0–41
- Day offsets: Sun=0, Mon=7, Tue=14, Wed=21, Thu=28, Fri=35
- Midday break: 12:00–13:00 → within-day quantum **2** (1 quantum per day)
- Break window: 12:00–14:00 → within-day quanta **{2, 3}** (2 quanta per day)

---

## Phase 1: Hard Constraint Unit Tests

**File**: `tests/test_constraints_hard.py`  
**Priority**: HIGHEST — these define schedule validity

### HC1: StudentGroupExclusivity

Delegates to `Timetable.count_group_violations()` which counts slots where a group appears in 2+ genes.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 1.1 | No overlap | G1 at q=0-1, G1 at q=2-3 | penalty = 0 | Unit |
| 1.2 | Exact overlap | G1 at q=0-1, G1 at q=0-1 | penalty = 2 (2 shared quanta) | Semantic |
| 1.3 | Partial overlap | G1 at q=0-2, G1 at q=1-2 | penalty = 2 (quanta 1,2 shared) | Semantic |
| 1.4 | Two groups independent | G1 at q=0, G2 at q=0 | penalty = 0 | Unit |
| 1.5 | Multiple overlaps | 3 genes with G1 at q=0 | penalty = 2 (len(3)-1=2 at q=0) | Semantic |
| 1.6 | Empty timetable | No genes | penalty = 0 | Edge |
| 1.7 | Single gene | 1 gene | penalty = 0 | Edge |
| 1.8 | Multi-group gene | Gene with [G1, G2] at q=0; another gene with G1 at q=0 | penalty = 1 | Semantic |

**Intent test**: "A group physically cannot attend two sessions at the same time. This constraint catches that."

### HC2: InstructorExclusivity

Delegates to `Timetable.count_instructor_violations()`.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 2.1 | No conflict | I1 at q=0-1, I1 at q=2-3 | penalty = 0 | Unit |
| 2.2 | Exact overlap | I1 at q=0-1, I1 at q=0-1 | penalty = 2 | Semantic |
| 2.3 | Partial overlap | I1 at q=0-2, I1 at q=1-2 | penalty = 2 | Semantic |
| 2.4 | Different instructors | I1 at q=0, I2 at q=0 | penalty = 0 | Unit |
| 2.5 | Empty timetable | No genes | penalty = 0 | Edge |

### HC3: RoomExclusivity

Delegates to `Timetable.count_room_violations()`.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 3.1 | No conflict | R1 at q=0-1, R1 at q=2-3 | penalty = 0 | Unit |
| 3.2 | Exact overlap | R1 at q=0-1, R1 at q=0-1 | penalty = 2 | Semantic |
| 3.3 | Different rooms | R1 at q=0, R2 at q=0 | penalty = 0 | Unit |
| 3.4 | 3 genes same room+time | 3 genes R1 at q=0 | penalty = 2 | Semantic |

### HC4: InstructorQualifications

Per-gene check: `gene.instructor_id in course.qualified_instructor_ids`.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 4.1 | Qualified instructor | I1 teaches CS101, CS101.qualified=[I1] | penalty = 0 | Unit |
| 4.2 | Unqualified instructor | I2 teaches CS101, CS101.qualified=[I1] | penalty = 1 | Unit |
| 4.3 | Missing course def | Gene references unknown course key | penalty = 1 | Edge |
| 4.4 | Empty qualification list | CS101.qualified=[] | penalty = 1 | Edge |
| 4.5 | Multiple violations | 3 genes with unqualified instructors | penalty = 3 | Semantic |
| 4.6 | All valid, 5 genes | 5 genes all properly qualified | penalty = 0 | Unit |
| 4.7 | Mixed valid/invalid | 2 qualified + 1 unqualified | penalty = 1 | Semantic |

**Intent test**: "An instructor must be qualified to teach the specific course they're assigned to."

### HC5: RoomSuitability

Uses `is_room_type_compatible(required, room_features)`.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 5.1 | Lecture in lecture room | required="lecture", room="lecture" | penalty = 0 | Unit |
| 5.2 | Lab in lab room | required="practical", room="lab" | penalty = 0 | Unit |
| 5.3 | Lecture in lab room | required="lecture", room="lab" | penalty = 1 | Unit |
| 5.4 | Lab in lecture room | required="practical", room="lecture" | penalty = 1 | Unit |
| 5.5 | Lecture in auditorium | required="lecture", room="auditorium" | penalty = 0 | Unit |
| 5.6 | Lecture in seminar | required="lecture", room="seminar" | penalty = 0 | Unit |
| 5.7 | Practical in computer_lab | required="practical", room="computer_lab" | penalty = 0 | Unit |
| 5.8 | Case insensitive | required="LECTURE", room="Lecture" | penalty = 0 | Edge |
| 5.9 | No room found | Room missing from context | penalty = 0 (skip) | Edge |

**Intent test**: "Theory classes can't be in labs; practicals can't be in lecture halls."

### HC6: InstructorTimeAvailability

Per-quantum check for part-time instructors.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 6.1 | Full-time instructor | is_full_time=True, any quanta | penalty = 0 | Unit |
| 6.2 | Part-time, available | available_quanta={0,1,2,3}, gene at q=0-1 | penalty = 0 | Unit |
| 6.3 | Part-time, unavailable | available_quanta={0,1}, gene at q=2-3 | penalty = 2 (2 quanta) | Semantic |
| 6.4 | Part-time, partial avail | available_quanta={0}, gene at q=0-1 | penalty = 1 (q=1 unavailable) | Semantic |
| 6.5 | Empty timetable | No genes | penalty = 0 | Edge |
| 6.6 | Instructor missing | Gene references unknown instructor | penalty = 0 (skip) | Edge |

**Semantic verification**: Penalty counts the *number of quanta* the instructor is unavailable, not genes.

### HC7: RoomTimeAvailability

Per-quantum check for room availability.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 7.1 | Room always available | available_quanta=empty (no restriction) | penalty depends on impl | Edge |
| 7.2 | Room available for slot | available_quanta={0,1}, gene at q=0-1 | penalty = 0 | Unit |
| 7.3 | Room unavailable | available_quanta={2,3}, gene at q=0-1 | penalty = 2 | Semantic |
| 7.4 | Partial availability | available_quanta={0}, gene at q=0-1 | penalty = 1 | Semantic |
| 7.5 | Room missing in context | Gene references unknown room | penalty = 0 (skip) | Edge |

### HC8: CourseCompleteness

Checks `sum(gene.num_quanta) == course.quanta_per_week` per (course, group).

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 8.1 | Exact match | CS101 needs 4q, scheduled for 4q | penalty = 0 | Unit |
| 8.2 | Under-scheduled | CS101 needs 4q, only 2q scheduled | penalty = 1 | Semantic |
| 8.3 | Over-scheduled | CS101 needs 4q, 6q scheduled | penalty = 1 | Semantic |
| 8.4 | Not scheduled at all | CS101 needs 4q, 0 genes for group | penalty = 1 | Unit |
| 8.5 | Multiple groups | CS101 enrolled by G1,G2; G1 has 4q, G2 has 2q | penalty = 1 (G2 short) | Semantic |
| 8.6 | Split sessions | CS101 needs 4q, two 2q genes | penalty = 0 | Unit |
| 8.7 | Multi-group gene | Gene with [G1,G2], both need 4q, gene=4q | penalty = 0 | Semantic |

**Intent test**: "Every course-group pair must receive exactly the required weekly hours."

---

## Phase 2: Soft Constraint Unit Tests

**File**: `tests/test_constraints_soft.py`  
**Priority**: HIGH

### SC1: StudentScheduleCompactness

Counts gap quanta between first and last session per group per day, excluding midday break.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 1.1 | No gap | G1 has q=0,1,2 on Sunday | penalty = 0 | Unit |
| 1.2 | One gap | G1 has q=0,2 on Sunday (q=1 is gap) | penalty = 1 | Semantic |
| 1.3 | Gap during break excluded | G1 has q=1,3 on Sunday (q=2 is midday break) | penalty = 0 | Semantic |
| 1.4 | Multiple gaps | G1 has q=0,3,6 on Sunday | penalty = 4 (q=1,3,4,5 minus break at q=2) | Semantic |
| 1.5 | Single session/day | G1 has only q=0 on Sunday | penalty = 0 (skipped, <2 quanta) | Edge |
| 1.6 | Multiple days | Gaps on 2 days → sum of both | penalty = sum | Integration |
| 1.7 | Multiple groups | G1 + G2 with different gaps | penalty = sum for all groups | Integration |
| 1.8 | Empty timetable | No genes | penalty = 0 | Edge |

**Semantic verification**: "A gap is an empty quantum between the first and last class on a day, excluding the designated break slot."

### SC2: InstructorScheduleCompactness

Same algorithm as SC1 but for instructors.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 2.1 | No gap | I1 at q=0,1,2 on Sunday | penalty = 0 | Unit |
| 2.2 | One gap | I1 at q=0,2 (q=1 is gap) | penalty = 1 | Semantic |
| 2.3 | Break excluded | I1 at q=1,3 (q=2 is break) | penalty = 0 | Semantic |
| 2.4 | Empty | No genes | penalty = 0 | Edge |

### SC3: StudentLunchBreak

**Known Bug**: `get_midday_break_quanta()` returns 1-quantum set (q=2 per day) but `break_min_quanta=2`. This means every group with any class that day gets penalized.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 3.1 | Full break free | G1 occupies q=0,1,4,5 (break q=2,3 free) → but midday_break={2} | depends on defaults | Semantic |
| 3.2 | Break occupied | G1 occupies q=2 → midday_break={2}, free=0, need 2 → missing=2 | penalty = 2 × 5.0 = 10 | Semantic |
| 3.3 | **BUG TEST**: Default config | midday_break returns {2} (1 quantum), need 2 → always missing 1 | penalty > 0 always | Bug |
| 3.4 | Custom break_min=1 | With break_min_quanta=1, free={2} → sufficient | penalty = 0 | Config |
| 3.5 | No classes that day | G1 has no quanta on Sunday | not checked (no entry in days) | Edge |
| 3.6 | Different penalty rate | penalty_per_missing=10.0, missing 1 | penalty = 10 | Config |

**Intent test**: "Students need a lunch break. If the break window has too few free quanta, penalize."

### SC4: SessionContinuity

Penalizes isolated single-quantum theory blocks (after the first one).

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 4.1 | All contiguous | CS101 theory, q=0,1,2 on same day → 1 block of 3 | penalty = 0 | Unit |
| 4.2 | Single isolated block OK | CS101 theory, q=0 → 1 block of 1 (first one excused) | penalty = 0 | Semantic |
| 4.3 | Two isolated blocks | CS101 theory, q=0,4 → 2 blocks of 1 → 1 excess | penalty = 10 | Semantic |
| 4.4 | Three isolated blocks | CS101 theory, q=0,3,6 → 3 blocks of 1 → 2 excess | penalty = 20 | Semantic |
| 4.5 | Practical skipped | CS101 practical, any fragmentation | penalty = 0 | Semantic |
| 4.6 | Mixed blocks | q=0,1,4 → block(0,1) + block(4) → only 1 isolated, excused | penalty = 0 | Semantic |
| 4.7 | Empty timetable | No genes | penalty = 0 | Edge |
| 4.8 | Custom penalty | isolated_slot_penalty=50 | scales accordingly | Config |

**Intent test**: "Theory sessions shouldn't be scattered into isolated single-hour slots throughout the day."

### SC5: PairedCohortPracticalAlignment

Measures symmetric difference of practical quanta between cohort pairs.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 5.1 | Perfect alignment | G1A and G1B both have CS101-practical at q=0-1 | penalty = 0 | Unit |
| 5.2 | Total misalignment | G1A at q=0-1, G1B at q=2-3 | penalty = 4 (sym diff) | Semantic |
| 5.3 | Partial overlap | G1A at q=0-2, G1B at q=1-3 | penalty = 2 (q=0 in A only, q=3 in B only) | Semantic |
| 5.4 | No cohort pairs | cohort_pairs=[] | penalty = 0 | Edge |
| 5.5 | No shared practicals | Pairs exist but different courses | penalty = 0 | Edge |
| 5.6 | Theory ignored | Pair shares theory course, not practical | penalty = 0 | Semantic |

**Intent test**: "Subgroup practicals should happen at the same time so lab resources are shared."

### SC6: BreakPlacementCompliance

Penalizes groups lacking free quanta in designated break windows.

| # | Test Case | Setup | Expected | Type |
|---|-----------|-------|----------|------|
| 6.1 | Break fully free | G1 has no classes in break window {2,3} | penalty = 0 | Unit |
| 6.2 | Break occupied | G1 has q=2,3 both occupied, free=0, need 2 | penalty = 1 | Semantic |
| 6.3 | Partial break free | G1 has q=2 occupied, q=3 free → free=1, need 2 | penalty = 1 | Semantic |
| 6.4 | enforce_break=False | QTS.enforce_break_placement=False | penalty = 0 | Config |
| 6.5 | No classes that day | G1 has no quanta that day | not checked | Edge |
| 6.6 | Multiple groups multiple days | Accumulates across all | penalty = sum | Integration |

---

## Phase 3: Constraint Infrastructure Tests

**File**: `tests/test_constraint_infra.py`

### build_constraints() Factory

| # | Test Case | Expected | Type |
|---|-----------|----------|------|
| 3.1 | Default call | Returns 14 constraints (8 hard + 6 soft), all weight=1.0 | Unit |
| 3.2 | hard_weight=10 | All 8 hard constraints have weight=10 | Unit |
| 3.3 | soft_weight=0 | All 6 soft constraints have weight=0 | Unit |
| 3.4 | **BUG**: weight=0.0 | `student_group_exclusivity_weight=0.0` → `0.0 or hard_weight` = hard_weight (wrong!) | Bug |
| 3.5 | Individual override | `instructor_exclusivity_weight=5.0` → that constraint=5, others=hard_weight | Unit |
| 3.6 | Custom params | `gap_penalty_per_quantum=2.0` passed to compactness constraints | Config |
| 3.7 | Custom break params | `break_min_quanta=4` → LunchBreak and BreakPlacement both get 4 | Config |

### Registries

| # | Test Case | Expected | Type |
|---|-----------|----------|------|
| 3.8 | HARD_CONSTRAINT_CLASSES length | len = 8 | Unit |
| 3.9 | SOFT_CONSTRAINT_CLASSES length | len = 6 | Unit |
| 3.10 | ALL_CONSTRAINTS | len = 14, = HARD + SOFT | Unit |
| 3.11 | All implement Protocol | Each element `isinstance(c, Constraint)` | Unit |
| 3.12 | All have unique names | No duplicate `.name` values | Semantic |
| 3.13 | Kind correctness | Each hard has `kind="hard"`, each soft has `kind="soft"` | Semantic |

### Evaluator Integration

**File**: `tests/test_evaluator.py`

| # | Test Case | Expected | Type |
|---|-----------|----------|------|
| E.1 | fitness() with zero violations | Returns (0.0, 0.0) or equivalent zero tuple | Integration |
| E.2 | fitness() applies weights | `constraint.weight * constraint.evaluate()` | Semantic |
| E.3 | breakdown() returns raw | Unweighted penalty values per constraint | Semantic |
| E.4 | hard_breakdown() | Only hard constraints in result | Unit |
| E.5 | soft_breakdown() | Only soft constraints in result | Unit |
| E.6 | Total consistency | sum(fitness()) == sum(weight * penalty for each constraint) | Semantic |
| E.7 | Deterministic | Same timetable → same fitness (no randomness) | Unit |

---

## Phase 4: Violation Detector Tests

**File**: `tests/test_violation_detector.py`

### ViolationDetector

| # | Test Case | Expected | Type |
|---|-----------|----------|------|
| V.1 | No violations → empty set | Clean schedule → no violated gene indices | Unit |
| V.2 | Group overlap detected | 2 genes with same group at same time → both returned | Semantic |
| V.3 | Instructor overlap detected | 2 genes with same instructor at same time → both returned | Semantic |
| V.4 | Room overlap detected | 2 genes with same room at same time → both returned | Semantic |
| V.5 | Fast vs Full consistency | Both strategies return same results for same input | Integration |
| V.6 | **BUG**: Self-overlap dead code | `gene.num_quanta != gene.num_quanta` never True | Bug |
| V.7 | **BUG**: No hierarchy in full | BME1A and BME1AB at same time → not detected | Bug |
| V.8 | Hybrid strategy threshold | Below threshold uses fast, above uses full | Unit |

---

## Phase 5: Repair Operator Tests

**File**: `tests/test_repairs.py`  
**Priority**: HIGH — repairs must actually fix what they claim to fix

### Test Pattern for Each Repair

```
1. PRE:  Create schedule WITH specific violation
2. ACT:  Run repair operator
3. POST: Verify violation count decreased or reached 0
4. SIDE: Verify no NEW violations were introduced
5. PRESERVE: Verify structural invariants maintained (course_id, group_ids, num_quanta)
```

### R1: repair_instructor_availability (Priority 1)

Fixes HC6 by shifting gene to time when instructor IS available.

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| R1.1 | Instructor unavailable at scheduled time | After repair: instructor available at new time | Intent |
| R1.2 | No available slot exists | Gene preserved as-is (repair fails gracefully) | Edge |
| R1.3 | Structural preservation | course_id, group_ids, num_quanta unchanged | Semantic |
| R1.4 | Only time changed | room_id, instructor_id preserved | Semantic |
| R1.5 | Already available | No change needed, gene unchanged | Edge |

### R2: repair_instructor_availability_reassign (Priority 1)

Fixes HC6 by swapping to a different available instructor.

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| R2.1 | Swap to available instructor | New instructor available at that time | Intent |
| R2.2 | No alternative instructor | Repair fails gracefully | Edge |
| R2.3 | New instructor must be qualified | instructor_id in course.qualified_instructor_ids | Semantic |
| R2.4 | Only instructor_id changed | time, room, course preserved | Semantic |

### R3: repair_group_overlaps (Priority 2)

Fixes HC1 by shifting gene to conflict-free time.

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| R3.1 | Two genes with same group overlapping | After repair: no overlap | Intent |
| R3.2 | Hierarchy-aware overlap | Parent group BME1A + subgroup BME1AB | Intent |
| R3.3 | New time doesn't create instructor conflict | No new HC2 violations | Side-effect |
| R3.4 | New time doesn't create room conflict | No new HC3 violations | Side-effect |
| R3.5 | No conflict-free slot | Repair fails gracefully or tries room swap | Edge |
| R3.6 | Structural preservation | course_id, group_ids, num_quanta unchanged | Semantic |

### R4: repair_room_overlap_reassign (Priority 3)

Fixes HC3 by swapping to a different available room.

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| R4.1 | Two genes in same room at same time | After repair: different rooms | Intent |
| R4.2 | New room must be compatible | Room type matches course requirement | Semantic |
| R4.3 | No alternative room | Repair fails gracefully | Edge |
| R4.4 | Only room_id changed | time, instructor, course preserved | Semantic |

### R5: repair_room_conflicts (Priority 4)

Fixes HC3 by time shift first, then room swap as fallback.

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| R5.1 | Room conflict resolved by time shift | Gene moved to different time, same room | Intent |
| R5.2 | Time shift fails → room swap | Different room assigned | Intent |
| R5.3 | Both fail | Repair fails gracefully | Edge |
| R5.4 | Fallback hierarchy | Time shift attempted before room swap | Semantic |

### R6: repair_instructor_conflicts (Priority 5)

Fixes HC2 by time shift first, then instructor swap.

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| R6.1 | Instructor conflict resolved by time shift | Gene at new time, same instructor | Intent |
| R6.2 | Time shift fails → instructor swap | Different instructor assigned | Intent |
| R6.3 | New instructor must be qualified | Validates qualification | Semantic |
| R6.4 | Both fail | Graceful failure | Edge |

### R7: repair_instructor_qualifications (Priority 6)

Fixes HC4 by swapping to a qualified instructor.

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| R7.1 | Unqualified instructor replaced | New instructor in course.qualified | Intent |
| R7.2 | No qualified instructor available | Repair fails | Edge |
| R7.3 | Qualified instructor has conflict | Bug: selective version doesn't check | Bug |
| R7.4 | Only instructor_id changed | Preserves everything else | Semantic |

### R8: repair_room_type_mismatches (Priority 7)

Fixes HC5 by swapping to a compatible room.

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| R8.1 | Lab in lecture room → moved to lab | New room compatible | Intent |
| R8.2 | No compatible room | Repair fails | Edge |
| R8.3 | New room doesn't create overlap | No new HC3 | Side-effect |
| R8.4 | Only room_id changed | Preserves everything else | Semantic |

### R9-R12: Soft Constraint Repairs

| # | Repair | Intent Test |
|---|--------|------------|
| R9 | repair_paired_cohort_practicals | After repair, cohort pairs have aligned practical times |
| R10 | repair_student_compactness | After repair, gap count decreased for worst group |
| R11 | repair_instructor_compactness | After repair, gap count decreased for worst instructor |
| R12 | repair_student_lunch_break | After repair, lunch break quanta are free |

### Orchestration: repair_individual_unified

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| O.1 | Full mode runs all repairs | All hard violations addressed | Integration |
| O.2 | Selective mode targets only violated genes | Fewer genes modified than full mode | Semantic |
| O.3 | Priority ordering respected | Higher-priority repairs run first | Semantic |
| O.4 | Idempotent on clean schedule | No changes to violation-free schedule | Intent |
| O.5 | Never increases hard violations | Total hard violations <= before repair | Intent |

---

## Phase 6: Heuristic Tests

**File**: `tests/test_heuristics.py`

### Construction Heuristics (3)

These build a complete schedule from scratch.

| # | Heuristic | Tests | Type |
|---|-----------|-------|------|
| H.C1 | largest_degree_first | Output has gene for every course-group pair; total quanta match | Intent |
| H.C2 | most_constrained_first | Same completeness as H.C1; most constrained courses scheduled first | Intent |
| H.C3 | earliest_deadline_first | Same completeness; produces valid individual | Intent |
| H.C4 | All three, same context | All produce same number of genes (same course-group pairs) | Cross-check |
| H.C5 | Output validity | Every gene has valid instructor_id, room_id, start_quanta | Semantic |
| H.C6 | Gene structure | course_type matches course definition; num_quanta correct | Semantic |

### Perturbation Heuristics (5)

These modify an existing schedule.

| # | Heuristic | Tests | Type |
|---|-----------|-------|------|
| H.P1 | random_swap | Two genes swap mutable attributes; structure preserved | Intent |
| H.P2 | temporal_shift | Gene time changed; course/group/instructor preserved | Intent |
| H.P3 | room_shuffle | Room changed; must remain compatible with course type | Intent |
| H.P4 | instructor_reassign | Instructor changed; must remain qualified | Intent |
| H.P5 | multi_perturbation | Multiple perturbations applied; structure preserved | Intent |
| H.P6 | All: structural invariants | course_id, course_type, group_ids, num_quanta NEVER changed | Semantic |

### Improvement Heuristics (3)

These must monotonically improve (or maintain) fitness.

| # | Heuristic | Tests | Type |
|---|-----------|-------|------|
| H.I1 | kempe_chain | Violations after <= violations before | Algorithm |
| H.I2 | ejection_chain | Violations after <= violations before | Algorithm |
| H.I3 | variable_depth_search | Violations after <= violations before | Algorithm |
| H.I4 | All: no degradation | fitness_after <= fitness_before guaranteed | Intent |
| H.I5 | All: structural preservation | Gene count unchanged, course_ids preserved | Semantic |

### Diversity Heuristics (4)

| # | Heuristic | Tests | Type |
|---|-----------|-------|------|
| H.D1 | distance_preserving_crossover | Output maintains genetic distance from population | Intent |
| H.D2 | crowding_mutation | Mutated individual differs from nearest neighbor | Intent |
| H.D3 | niching_selection | Selected set has better diversity than random selection | Intent |
| H.D4 | adaptive_diversity_maintenance | Population diversity metric >= threshold | Intent |

### Meta-Heuristics (4)

| # | Heuristic | Tests | Type |
|---|-----------|-------|------|
| H.M1 | variable_neighborhood_descent | Fitness improves or stays same | Algorithm |
| H.M2 | iterated_local_search | Better than initial solution or same | Algorithm |
| H.M3 | adaptive_large_neighborhood | Returns valid schedule | Unit |
| H.M4 | guided_local_search | Uses penalty augmentation correctly | Intent |

### Heuristic Registry

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| H.R1 | get_all_heuristics() count | Returns exactly 26 HeuristicInfo entries | Unit |
| H.R2 | All categories present | 6 categories: construction, perturbation, improvement, diversity, meta, repair | Unit |
| H.R3 | All functions callable | Each `.function` is callable | Unit |
| H.R4 | get_enabled_heuristics() respects config | Disabled heuristics excluded | Config |
| H.R5 | No duplicate names | All heuristic names unique | Semantic |

---

## Phase 7: GA Operator Tests

**File**: `tests/test_ga_operators.py`

### Crossover

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| X.1 | Structural preservation | Same (course_id, group_ids) keys before and after | Intent |
| X.2 | Only mutable fields swap | instructor_id, room_id, start_quanta may change | Semantic |
| X.3 | num_quanta never changes | Duration preserved for all genes | Semantic |
| X.4 | start_quanta clipped | After swap, start_quanta in [0, total_quanta - num_quanta] | Semantic |
| X.5 | cx_prob=0 → no changes | Both parents unchanged | Unit |
| X.6 | cx_prob=1 → all swap | All matching genes swap mutable attributes | Unit |
| X.7 | Validation error | Parents with different course structures → ValueError | Edge |
| X.8 | In-place modification | Returns the same list objects (modified) | Unit |

### Mutation

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| M.1 | course_id immutable | Never changes across mutations | Semantic |
| M.2 | course_type immutable | Never changes across mutations | Semantic |
| M.3 | group_ids immutable | Never changes across mutations | Semantic |
| M.4 | num_quanta preserved | Duration never changes | Semantic |
| M.5 | Instructor qualification | New instructor is qualified for course | Semantic |
| M.6 | Room suitability | New room is compatible via find_suitable_rooms | Semantic |
| M.7 | mutate_time_quanta length | Output length == gene.num_quanta always | Algorithm |
| M.8 | mut_prob=0 → no changes | Individual unchanged | Unit |
| M.9 | DEAP tuple format | Returns `(individual,)` tuple | Unit |

### Constraint-Aware Mutation

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| CA.1 | No new group overlaps | Refuses mutation that creates group overlap | Intent |
| CA.2 | Still mutates valid moves | Accepts mutation when no conflict | Unit |
| CA.3 | **BUG**: Room change unconstrained | 30% room change in constraint_guided may assign incompatible room | Bug |

### Selection (NSGA-II)

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| S.1 | Dominates correctness | A dominates B iff A ≤ in all, < in at least one | Algorithm |
| S.2 | Dominates transitivity | If A dom B and B dom C, then A dom C | Algorithm |
| S.3 | Dominates reflexivity | A does NOT dominate A (strict) | Algorithm |
| S.4 | Front 0 is Pareto-optimal | No individual in front 0 is dominated by any | Algorithm |
| S.5 | Front partitioning | Every individual in exactly one front | Algorithm |
| S.6 | Crowding distance boundary | Boundary individuals get infinity | Algorithm |
| S.7 | Crowding distance non-negative | All distances >= 0 | Algorithm |
| S.8 | sel_nsga2_fast returns k | Output length = k (or all if fewer) | Unit |
| S.9 | Front priority | All of front 0 selected before front 1 | Semantic |
| S.10 | Crowding tiebreak | Within same front, higher crowding preferred | Semantic |

---

## Phase 8: System Integration Tests

**File**: `tests/test_system_integration.py`

### End-to-End Pipeline

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| E2E.1 | Construct → Evaluate | Construction heuristic produces evaluable schedule | Integration |
| E2E.2 | Construct → Repair → Evaluate | Repair reduces violations vs pre-repair | Integration |
| E2E.3 | Construct → Mutate → Evaluate | Mutation produces different but valid schedule | Integration |
| E2E.4 | Construct → Crossover → Evaluate | Crossover of two schedules produces valid offspring | Integration |
| E2E.5 | Full GA iteration | Select → Crossover → Mutate → Repair → Evaluate → Select | Integration |
| E2E.6 | 10 generations improvement | Fitness at gen 10 <= fitness at gen 0 (non-worsening) | Algorithm |

### Repair-Constraint Alignment

| # | Test Case | Expected | Type |
|---|-----------|-------|------|
| RCA.1 | repair_group_overlaps  HC1 | HC1.evaluate() should be 0 after repair | Intent |
| RCA.2 | repair_instructor_conflicts  HC2 | HC2.evaluate() should be 0 or decreased | Intent |
| RCA.3 | repair_room_overlap  HC3 | HC3.evaluate() should be 0 after repair | Intent |
| RCA.4 | repair_instructor_qualifications  HC4 | HC4.evaluate() should be 0 after repair | Intent |
| RCA.5 | repair_room_type_mismatches  HC5 | HC5.evaluate() should be 0 after repair | Intent |
| RCA.6 | repair_instructor_availability  HC6 | HC6.evaluate() should be 0 after repair | Intent |
| RCA.7 | repair_room_availability  HC7 | HC7.evaluate() should be 0 after repair | Intent |
| RCA.8 | Full repair pipeline  all HC | All hard constraints should be 0 after full repair | Intent |

---

## Phase 9: Algorithm Correctness Tests

**File**: `tests/test_algorithm_correctness.py`

### NSGA-II Properties

| # | Property | How to Test |
|---|----------|-------------|
| A.1 | Non-domination invariant | After sorting, no individual in front $i$ dominated by front $i$ |
| A.2 | Crowding distance symmetry | Boundary inds always get $\infty$ |
| A.3 | Elitism preservation | Best front preserved across generations |

### Strategy Selectors

| # | Property | How to Test |
|---|----------|-------------|
| A.4 | RoundRobin cycles through all | After N calls (N=num_heuristics), each used exactly once |
| A.5 | Adaptive probabilities sum to 1 | `sum(probabilities) ≈ 1.0` after updates |
| A.6 | Adaptive rewards best | Heuristic with best improvement gets highest probability |
| A.7 | SimpleRL Q-values converge | After many updates, best action Q > others |

### Repair Engine (RL-ready)

| # | Property | How to Test |
|---|----------|-------------|
| A.8 | MoveTimeOperator changes time only | All other fields preserved |
| A.9 | SwapRoomOperator changes room only | All other fields preserved |
| A.10 | ReassignInstructorOperator changes instructor only | All other fields preserved |
| A.11 | EpsilonGreedy exploration | With ε=1.0, uniform random selection |
| A.12 | EpsilonGreedy exploitation | With ε=0.0, always selects best operator |
| A.13 | **BUG**: RepairPipeline.default() | Always fails (imports non-existent classes) |

---

## Phase 10: Known Bug Verification Tests

**File**: `tests/test_known_bugs.py`

| # | Bug | Test | Status |
|---|-----|------|--------|
| B.1 | `build_constraints()` weight=0 | `build_constraints(student_group_exclusivity_weight=0.0)` → weight should be 0, actually gets hard_weight | Document & test |
| B.2 | Midday break  lunch mismatch | `get_midday_break_quanta()` returns 1q, `StudentLunchBreak` needs 2 → always penalizes | Document & test |
| B.3 | ViolationDetector self-overlap | `gene.num_quanta != gene.num_quanta` is always False | Document & test |
| B.4 | ViolationDetector no hierarchy | Doesn't detect parentsubgroup overlaps (BME1ABME1AB) | Document & test |
| B.5 | RepairPipeline.default() imports | Imports TimeSlotOperator etc. which don't exist | Document & test |
| B.6 | Selective repair no-conflict-check | `repair_instructor_qualifications` selective doesn't verify new instructor isn't conflicting | Document & test |
| B.7 | Constraint-guided room mutation | `_mutate_session()` 30% room change has no type/capacity check | Document & test |
| B.8 | SessionContinuity unused params | `preferred_block_min/max` stored but never used in evaluate() | Document & test |
| B.9 | InstructorQualifications singleton state | `_warned_missing`/`_warned_empty` accumulate unboundedly across evaluations | Document & test |

---

## Execution Order & Dependencies

```
Phase 0: conftest.py (shared fixtures)           ← FIRST
   ↓
Phase 1: test_constraints_hard.py (8 HC)          ← No dependencies
Phase 2: test_constraints_soft.py (6 SC)          ← No dependencies
Phase 3: test_constraint_infra.py (factory/reg)   ← No dependencies
   ↓ (parallel with phases 1-3)
Phase 4: test_violation_detector.py               ← Uses Phase 0 fixtures
Phase 5: test_repairs.py                          ← Depends on Phase 1 (constraints to verify repairs)
   ↓
Phase 6: test_heuristics.py                       ← Uses Phase 0 fixtures
Phase 7: test_ga_operators.py                     ← Uses Phase 0 fixtures
   ↓
Phase 8: test_system_integration.py               ← Depends on all above
Phase 9: test_algorithm_correctness.py            ← Depends on Phases 7-8
Phase 10: test_known_bugs.py                      ← Can run anytime
```

---

## Test Count Summary

| Phase | File | Tests |
|-------|------|-------|
| 1 | test_constraints_hard.py | ~42 |
| 2 | test_constraints_soft.py | ~36 |
| 3 | test_constraint_infra.py | ~20 |
| 4 | test_violation_detector.py | ~8 |
| 5 | test_repairs.py | ~50 |
| 6 | test_heuristics.py | ~30 |
| 7 | test_ga_operators.py | ~28 |
| 8 | test_system_integration.py | ~14 |
| 9 | test_algorithm_correctness.py | ~13 |
| 10 | test_known_bugs.py | ~9 |
| **Total** | | **~250 new tests** |

---

## Marker Strategy

```python
# pytest markers for selective execution
@pytest.mark.hard_constraint    # Phase 1
@pytest.mark.soft_constraint    # Phase 2
@pytest.mark.infra             # Phase 3
@pytest.mark.detector          # Phase 4
@pytest.mark.repair            # Phase 5
@pytest.mark.heuristic         # Phase 6
@pytest.mark.ga_operator       # Phase 7
@pytest.mark.integration       # Phase 8
@pytest.mark.algorithm         # Phase 9
@pytest.mark.known_bug         # Phase 10
@pytest.mark.semantic          # Tests verifying output meaning
@pytest.mark.intent            # Tests verifying purpose achieved
```

Usage: `pytest -m hard_constraint` to run just Phase 1.
