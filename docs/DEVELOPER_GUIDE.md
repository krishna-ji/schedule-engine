# Schedule Engine — Complete Developer Guide

> **Audience**: New developers joining this project.  
> **Scope**: Full-depth coverage of the mathematical model, algorithms, constraint system, software architecture, data pipeline, and code organisation.

---

## Table of Contents

1. [What Is This Project?](#1-what-is-this-project)
2. [The Mathematical Problem: University Course Timetabling (UCTP)](#2-the-mathematical-problem)
3. [Time Representation: The Quantum Time System](#3-time-representation-the-quantum-time-system)
4. [Domain Model (Entity Layer)](#4-domain-model-entity-layer)
5. [Chromosome Encoding & Gene Representation](#5-chromosome-encoding--gene-representation)
6. [Constraint System: Hard & Soft](#6-constraint-system-hard--soft)
7. [The Optimisation Algorithm: NSGA-II via pymoo](#7-the-optimisation-algorithm-nsga-ii-via-pymoo)
8. [The Pipeline: From JSON to Optimised Schedule](#8-the-pipeline-from-json-to-optimised-schedule)
9. [Operators: Sampling, Crossover, Mutation, Repair](#9-operators-sampling-crossover-mutation-repair)
10. [Vectorized Evaluation: The Performance Engine](#10-vectorized-evaluation-the-performance-engine)
11. [Experiment System & Modes](#11-experiment-system--modes)
12. [Reinforcement Learning Integration](#12-reinforcement-learning-integration)
13. [Software Engineering Architecture](#13-software-engineering-architecture)
14. [Data Flow Walkthrough: A Full Run](#14-data-flow-walkthrough-a-full-run)
15. [Key Design Decisions & Trade-offs](#15-key-design-decisions--trade-offs)
16. [How to Run & Extend](#16-how-to-run--extend)

---

## 1. What Is This Project?

**Schedule Engine** is a university course timetabling optimiser built for the Institute of Engineering (IOE), Tribhuvan University, Nepal. It takes JSON descriptions of courses, instructors, rooms, and student groups, then uses a **multi-objective evolutionary algorithm (NSGA-II)** to produce weekly schedules that satisfy hard constraints (no double-bookings) and optimise soft constraints (compact schedules, lunch breaks).

### Scale of the real problem

| Dimension | Typical count |
|-----------|---------------|
| Courses (theory + practical) | ~240 course-type pairs |
| Student groups (including subgroups) | ~80 |
| Instructors | ~120 |
| Rooms (lecture + lab) | ~40 |
| Events (genes) to schedule | ~600 |
| Time quanta per week | 42 (6 days × 7 hours) |
| Decision variables per event | 3 (instructor, room, start time) |
| Total decision variables | ~1800 |

This is an NP-hard combinatorial optimisation problem — there is no known polynomial-time algorithm that guarantees the globally optimal solution. Metaheuristics (evolutionary algorithms) are the standard approach.

---

## 2. The Mathematical Problem

### 2.1 Formal Definition

The University Course Timetabling Problem (UCTP) can be stated as:

> **Given** a set of events $E$, resources (instructors $I$, rooms $R$, student groups $G$), and a discrete time grid $T$, **find** an assignment $\sigma: E \to I \times R \times T$ that **minimises** constraint violations.

Each event $e \in E$ is a teaching session characterised by:

- A course $c_e$ (with type: theory or practical)
- A set of student groups $G_e \subseteq G$
- A duration $d_e$ (in time quanta)

The assignment for each event consists of:

- An instructor $i_e \in I_e$ (from the qualified set)
- A room $r_e \in R_e$ (from the suitable set)
- A start time $t_e \in T$ (such that the session $[t_e, t_e + d_e)$ fits within a single day)

### 2.2 Multi-Objective Formulation

This is a **bi-objective minimisation** problem:

$$\min_{\sigma} \quad \mathbf{F}(\sigma) = \begin{pmatrix} f_{\text{hard}}(\sigma) \\ f_{\text{soft}}(\sigma) \end{pmatrix}$$

Where:

$$f_{\text{hard}}(\sigma) = \sum_{c \in \mathcal{H}} w_c \cdot v_c(\sigma)$$

$$f_{\text{soft}}(\sigma) = \sum_{c \in \mathcal{S}} w_c \cdot v_c(\sigma)$$

$v_c(\sigma)$ is the violation count for constraint $c$, and $w_c$ is its weight. The goal is to reach $f_{\text{hard}} = 0$ (feasible schedule) while minimising $f_{\text{soft}}$ (schedule quality).

### 2.3 Why Multi-Objective?

A single-objective formulation ($f = \alpha \cdot f_\text{hard} + \beta \cdot f_\text{soft}$) forces you to choose weights $\alpha, \beta$ a priori. NSGA-II instead returns a **Pareto front** — the set of solutions where you can't improve one objective without worsening the other. The decision-maker (university scheduler) can then choose their preferred trade-off.

### 2.4 Constraint Hierarchy

The system distinguishes two classes:

| Type | Meaning | Goal | Examples |
|------|---------|------|----------|
| **Hard** | Physical impossibilities | Must reach 0 | No double-booking of rooms, instructors, groups |
| **Soft** | Quality preferences | Minimise | Compact schedules, lunch breaks, session continuity |

---

## 3. Time Representation: The Quantum Time System

**File**: `src/io/time_system.py` — class `QuantumTimeSystem`

### 3.1 Concept

Time is discretised into **quanta** — fixed-duration atomic time units (default: 60 minutes). The key insight is that quanta are **continuous and contiguous**: non-operating hours and closed days receive no index.

```
Default config: 6 operating days (Sun–Fri), 10:00–17:00 each → 7 quanta/day

Day       | Quanta indices | Human time
----------|----------------|------------------
Sunday    | 0–6            | 10:00–17:00
Monday    | 7–13           | 10:00–17:00
Tuesday   | 14–20          | 10:00–17:00
Wednesday | 21–27          | 10:00–17:00
Thursday  | 28–34          | 10:00–17:00
Friday    | 35–41          | 10:00–17:00
Saturday  | (closed)       | —
                               Total: 42 quanta
```

### 3.2 Why Quanta?

1. **Compact representation**: A session at "Monday 10:00 for 2 hours" is just `(start_quanta=7, num_quanta=2)` → occupies quanta {7, 8}.
2. **Overlap detection**: Two sessions overlap iff their quanta ranges intersect — a simple integer comparison.
3. **Day-boundary enforcement**: The QTS automatically clips sessions so they don't span across days.
4. **Encoding**: Each event's start time is a single integer ∈ [0, 42), perfect for an evolutionary algorithm's decision variable.

### 3.3 Key Methods

| Method | What it does |
|--------|-------------|
| `time_to_quanta("Monday", "10:00")` → `7` | Convert human time → quantum index |
| `quanta_to_time(7)` → `("Monday", "10:00")` | Reverse conversion |
| `quantum_to_day_and_within_day(7)` → `("Monday", 0)` | Day name + offset within day |
| `get_midday_break_quanta()` → `{day: {2}}` | Which within-day quanta are break time |
| `get_all_operating_quanta()` → `{0,1,...,41}` | All valid quanta |

---

## 4. Domain Model (Entity Layer)

**Directory**: `src/domain/`

This is the core data model — plain Python dataclasses representing the scheduling universe.

### 4.1 Entity Hierarchy

```
SchedulingContext                ← top-level container
 ├── courses:  dict[(code, type), Course]
 ├── groups:   dict[str, Group]
 ├── instructors: dict[str, Instructor]
 ├── rooms:    dict[str, Room]
 ├── available_quanta: list[int]
 ├── cohort_pairs: list[(str, str)]   ← paired groups (e.g. BAE2A, BAE2B)
 └── family_map: dict[str, set[str]]  ← parent↔subgroup relationships
```

### 4.2 Course

```python
@dataclass(slots=True)
class Course:
    course_id: str          # "ENCT 101"
    name: str               # "Computer Programming"
    quanta_per_week: int    # Total quanta needed (L+T for theory, P for practical)
    required_room_features: str  # "lecture" or "practical"
    enrolled_group_ids: list[str]
    qualified_instructor_ids: list[str]
    course_type: str        # "theory" or "practical"
    L: int; T: int; P: int  # Lecture/Tutorial/Practical hours from syllabus
    specific_lab_features: list[str]  # e.g. ["General Programming Lab"]
```

**Critical**: Each syllabus course with both L+T > 0 and P > 0 becomes **two** Course objects:

- `("ENCT 101", "theory")` with `quanta_per_week = L + T = 4`
- `("ENCT 101", "practical")` with `quanta_per_week = P = 3`

This is because theory and practical sessions have fundamentally different constraints (room type, group splitting).

### 4.3 Group (Student Group)

```python
@dataclass(slots=True)
class Group:
    group_id: str         # "BME1AB" (parent) or "BME1A" (subgroup)
    name: str
    student_count: int    # affects room capacity constraint
    enrolled_courses: list[str]  # course codes
```

**Group hierarchy**: Parent groups (e.g. `BME1AB`) have subgroups (e.g. `BME1A`, `BME1B`).

- **Theory** sessions are scheduled for all subgroups simultaneously (e.g. `group_ids=["BME1A", "BME1B"]`)
- **Practical** sessions are scheduled per-subgroup (e.g. `group_ids=["BME1A"]` alone)

This models the real-world pattern where lectures are for the full class but labs split into halves.

### 4.4 Instructor

```python
@dataclass(slots=True)
class Instructor:
    instructor_id: str
    name: str
    qualified_courses: list[tuple[str, str]]  # [(course_code, course_type), ...]
    is_full_time: bool
    available_quanta: set[int]  # only for part-time instructors
```

### 4.5 Room

```python
@dataclass(slots=True)
class Room:
    room_id: str          # "D101"
    name: str             # "Computer Lab #1 [D 101]"
    capacity: int         # 24
    room_features: str    # "practical" or "lecture"
    specific_features: list[str]  # ["Digital Architecture Lab", "General Programming Lab"]
```

Room suitability matching is centralised in `src/utils/room_compatibility.py` — a practical course requiring "General Programming Lab" matches rooms with that specific feature.

### 4.6 SessionGene (The Chromosome Gene)

```python
@dataclass
class SessionGene:
    course_id: str         # "ENCT 101"
    course_type: str       # "theory" or "practical"
    instructor_id: str     # decision variable
    group_ids: list[str]   # structural (never mutated)
    room_id: str           # decision variable
    start_quanta: int      # decision variable
    num_quanta: int        # structural (fixed by course requirements)
```

**Immutable fields** (structural): `course_id`, `course_type`, `group_ids`, `num_quanta`  
**Mutable fields** (decision variables): `instructor_id`, `room_id`, `start_quanta`

The contiguous representation `(start_quanta, num_quanta)` replaced an older `quanta: list[int]` design. This makes **temporal fragmentation structurally impossible** — a session is always a single contiguous block.

### 4.7 Timetable (Pre-Indexed Schedule View)

```python
class Timetable:
    """Wraps list[SessionGene] + SchedulingContext with O(1) lookup indexes."""
```

On construction, `Timetable` builds these indexes **once** (computed during `__init__`, reused by all consumers):

| Index | Type | Purpose |
|-------|------|---------|
| `group_occupancy` | `(group_id, quantum) → [gene_idx]` | Detect group double-booking |
| `instructor_occupancy` | `(instructor_id, quantum) → [gene_idx]` | Detect instructor double-booking |
| `room_occupancy` | `(room_id, quantum) → [gene_idx]` | Detect room double-booking |
| `group_daily` | `group_id → day → {within_day_q}` | Schedule compactness |
| `course_group_quanta` | `(course, type, group) → int` | Course completeness check |

This eliminates redundant recomputation — previously, the same decode-and-map logic was scattered across 14 different callsites.

### 4.8 Supergroup & Cluster (Decomposition)

For CP-SAT repair, the system clusters academic programmes that share many courses/instructors:

```
ARCH     — BAR (independent)
CIVIL    — BCE (independent)
IT       — BCT + BEI (12+ shared courses)
MECH     — BAM + BME + BIE (15+ shared courses)
MASTERS  — MEE + MIISE + MMDM
```

Detection uses **Union-Find**: programmes sharing ≥2 courses are merged into the same cluster. This enables parallel CP-SAT solving per cluster.

---

## 5. Chromosome Encoding & Gene Representation

### 5.1 The SessionGene (OOP) Encoding

In the original GA layer (`src/ga/`), an individual is:

```python
Individual = list[SessionGene]  # ~600 genes for the IOE dataset
```

Each gene has both structural (fixed) and decision (mutable) fields. All individuals have genes in the **same deterministic order** — gene at position $k$ always represents the same (course, group) pair. This is critical for crossover.

### 5.2 The 3×E Interleaved (pymoo) Encoding

For the pymoo pipeline (`src/pipeline/`), genes are flattened into a dense NumPy integer vector:

```
X = [I₀, R₀, T₀,  I₁, R₁, T₁,  ...,  I_{E-1}, R_{E-1}, T_{E-1}]
                                     ↑
                               3×E integers total (~1800)
```

Where for event $e$:

- `X[3e + 0]` = instructor **index** (into a sorted instructor list)
- `X[3e + 1]` = room **index** (into a sorted room list)
- `X[3e + 2]` = start quantum (0–41)

The structural fields (course_id, course_type, group_ids, num_quanta) are stored separately in `events_with_domains.pkl` since they're the same for all individuals.

### 5.3 Domain Constraints per Event

Each event has a **precomputed domain** — the set of valid values for each decision variable:

```python
allowed_instructors[e] = [3, 7, 12]     # indices of qualified instructors
allowed_rooms[e]       = [0, 5, 8, 14]  # indices of suitable rooms
allowed_starts[e]      = [0, 1, ..., 40] # valid start quanta (= max_quanta - duration)
```

These are computed once by `build_events_with_domains()` and stored in the pkl file. Every operator (sampling, mutation, repair) respects these domains.

### 5.4 Subsession Splitting

Theory courses with `quanta_per_week > 2` are split into pedagogically appropriate sub-sessions:

```python
def get_subsession_durations(quanta_per_week, course_type):
    if course_type == "practical":
        return [quanta_per_week]          # Single continuous block
    # Theory: 2-hour blocks + 1-hour remainder
    if quanta_per_week % 2 == 0:
        return [2] * (quanta_per_week // 2)    # e.g. 6→[2,2,2]
    blocks = [2] * (quanta_per_week // 2)
    blocks.append(1)                           # e.g. 5→[2,2,1]
    return blocks
```

This means a 5-hour theory course becomes **3 separate genes** (two 2-hour + one 1-hour), each independently scheduled.

---

## 6. Constraint System: Hard & Soft

**Directory**: `src/constraints/`

### 6.1 Architecture

All constraints implement the `Constraint` protocol:

```python
class Constraint(Protocol):
    name: str
    weight: float
    kind: str   # "hard" or "soft"
    
    def evaluate(self, tt: Timetable) -> float:
        """Return penalty (0 = no violation)."""
```

Each constraint is a self-contained class with its own evaluation logic. The `Evaluator` aggregates them:

```python
class Evaluator:
    def fitness(genes, context, qts) -> (hard_penalty, soft_penalty):
        tt = Timetable(genes, context, qts)
        hard = Σ(c.weight × c.evaluate(tt))  for c ∈ hard_constraints
        soft = Σ(c.weight × c.evaluate(tt))  for c ∈ soft_constraints
```

### 6.2 Hard Constraints (8 total)

These represent physical impossibilities. A feasible schedule has all hard penalties = 0.

| # | Constraint | What it checks | Math |
|---|-----------|----------------|------|
| 1 | **StudentGroupExclusivity** | A student group can't be in two places at once | $\sum_{(g,q)} \max(0, \|occ(g,q)\| - 1)$ |
| 2 | **InstructorExclusivity** | An instructor can't teach two classes simultaneously | $\sum_{(i,q)} \max(0, \|occ(i,q)\| - 1)$ |
| 3 | **RoomExclusivity** | A room can't host two sessions at the same time | $\sum_{(r,q)} \max(0, \|occ(r,q)\| - 1)$ |
| 4 | **InstructorQualifications** | Instructor must be qualified for the course | $\sum_e \mathbb{1}[i_e \notin \text{qualified}(c_e)]$ |
| 5 | **RoomSuitability** | Room type must match course requirements | $\sum_e \mathbb{1}[\text{type}(r_e) \not\sim \text{req}(c_e)]$ |
| 6 | **InstructorTimeAvailability** | Part-time instructors only during their available slots | $\sum_e \sum_{q \in [t_e, t_e+d_e)} \mathbb{1}[q \notin \text{avail}(i_e)]$ |
| 7 | **RoomTimeAvailability** | Rooms only usable during their available hours | $\sum_e \sum_{q \in [t_e, t_e+d_e)} \mathbb{1}[q \notin \text{avail}(r_e)]$ |
| 8 | **CourseCompleteness** | Each (course, group) must have exactly the required quanta | $\sum_{(c,g)} \mathbb{1}[\text{actual}(c,g) \neq \text{required}(c)]$ |

For exclusivity constraints, $occ(x, q)$ is the set of events assigned to resource $x$ at quantum $q$. If $|occ| > 1$, there's a conflict.

### 6.3 Soft Constraints (6 total)

These encode quality preferences. Lower is better, but non-zero is acceptable.

| # | Constraint | What it measures |
|---|-----------|-----------------|
| 1 | **StudentScheduleCompactness** | Penalise idle gaps in student schedules (per group per day) |
| 2 | **InstructorScheduleCompactness** | Penalise idle gaps in instructor schedules |
| 3 | **StudentLunchBreak** | Students should have free quanta during 12:00–14:00 window |
| 4 | **SessionContinuity** | Penalise isolated 1-hour slots (prefer 2+ hour blocks) |
| 5 | **PairedCohortPracticalAlignment** | Paired subgroups (A/B) should have practicals at same time |
| 6 | **BreakPlacementCompliance** | Groups must have ≥1 free quantum during designated break window |

**Compactness** is measured as the number of gap quanta between first and last session per entity per day, excluding designated break quanta:

$$\text{gap}(g, d) = |\{q \in [\min(S_{g,d}), \max(S_{g,d})] : q \notin S_{g,d} \land q \notin \text{break}(d)\}|$$

where $S_{g,d}$ is the set of within-day quanta occupied by group $g$ on day $d$.

### 6.4 Constraint Factory

```python
constraints = build_constraints(
    hard_weight=10.0,           # scale all hard constraints
    soft_weight=1.0,
    isolated_slot_penalty=50.0, # custom parameter for SessionContinuity
    break_min_quanta=2,         # require 2 free quanta for lunch
)
evaluator = Evaluator(constraints)
```

---

## 7. The Optimisation Algorithm: NSGA-II via pymoo

### 7.1 What is NSGA-II?

**Non-dominated Sorting Genetic Algorithm II** (Deb et al., 2002) is the gold standard for multi-objective evolutionary optimisation. It maintains a population of candidate solutions and evolves them over generations using selection, crossover, and mutation.

### 7.2 Key Concepts

**Pareto Dominance**: Solution $A$ dominates $B$ (written $A \prec B$) iff $A$ is at least as good in all objectives and strictly better in at least one:

$$A \prec B \iff \forall i: f_i(A) \leq f_i(B) \land \exists j: f_j(A) < f_j(B)$$

**Non-dominated Front**: The set of solutions not dominated by any other solution in the population. NSGA-II sorts the population into successive fronts $F_1, F_2, \ldots$ where $F_1$ is the Pareto-optimal front.

**Crowding Distance**: Within each front, NSGA-II uses crowding distance to maintain diversity. Solutions in less crowded regions of objective space are preferred.

### 7.3 NSGA-II Pseudocode

```
1. Initialize population P₀ of size N (via ConstructiveSampling)
2. Evaluate F(P₀) = (f_hard, f_soft) for each individual
3. For generation t = 1, 2, ..., ngen:
   a. Select parents via binary tournament (based on rank + crowding)
   b. Apply crossover (EventBlockCrossover): swap event triples between parents
   c. Apply mutation (EventLocalMutation): randomly reassign instructor/room/time
   d. Apply repair (PymooVectorizedRepair): fix domain violations & conflicts
   e. Evaluate offspring
   f. Combine parent + offspring → 2N candidates
   g. Non-dominated sort → fronts F₁, F₂, ...
   h. Fill next generation: take all of F₁, then F₂, ... 
      until last front doesn't fit — use crowding distance to select survivors
4. Return final population (Pareto front = F₁)
```

### 7.4 pymoo Integration

The project uses **pymoo** (Python Multi-objective Optimization), which provides the NSGA-II framework. The project plugs in custom:

| Component | Class | File |
|-----------|-------|------|
| Problem definition | `SchedulingProblem` | `src/pipeline/scheduling_problem.py` |
| Sampling | `ConstructiveSampling` | `src/pipeline/pymoo_operators.py` |
| Crossover | `EventBlockCrossover` | `src/pipeline/pymoo_operators.py` |
| Mutation | `EventLocalMutation` | `src/pipeline/pymoo_operators.py` |
| Repair | `PymooVectorizedRepair` | `src/pipeline/repair_operator_vectorized.py` |

### 7.5 Performance Characteristics

From actual runs on the IOE dataset (~600 events):

| Configuration | Hard penalty | Soft penalty | Time |
|--------------|-------------|-------------|------|
| pop=100, gen=100 | 47 | 633 | ~420s |
| pop=100, gen=200 | 81 | 680 | ~960s |
| pop=50, gen=50 | 49 | 669 | ~130s |
| pop=10, gen=200 | 55 | 630 | ~1740s |

The problem has not yet reached full feasibility (hard=0) which is expected for this scale — the search space is enormous (each of ~600 events has ~10 valid instructors × ~15 valid rooms × ~40 valid start times = ~6000 combinations per event).

---

## 8. The Pipeline: From JSON to Optimised Schedule

### 8.1 Overview

```
JSON Data Files                                                    
    ↓                                                              
DataStore.from_json()          ← Load + link entities              
    ↓                                                              
build_events_with_domains()    ← Generate events + precompute domains
    ↓                                                              
events_with_domains.pkl        ← Cached intermediate representation
    ↓                                                              
SchedulingProblem              ← pymoo Problem (n_vars, objectives) 
    ↓                                                              
NSGA-II (pymoo.minimize)       ← Evolution loop                   
    ↓                                                              
Best chromosome                ← Flat int array                    
    ↓                                                              
chromosome_to_genes()          ← Convert back to SessionGene list  
    ↓                                                              
export_everything()            ← schedule.json, calendar.pdf, plots
```

### 8.2 Data Loading (`DataStore`)

**File**: `src/io/data_store.py`

`DataStore.from_json("data")` does:

1. **Create QTS** with operating hours (default: Sun–Fri, 10:00–17:00)
2. **Load Groups** from `Groups.json` — populate available_quanta, handle subgroups
3. **Load Courses** from `Course.json` — split into theory/practical, compute `quanta_per_week`
4. **Filter Courses** — only keep courses enrolled by at least one loaded group
5. **Load Instructors** from `Instructors.json` — link qualified courses as `(code, type)` tuples
6. **Load Rooms** from `Rooms.json` — encode availability as quantum sets
7. **Link relationships** — `Course.enrolled_group_ids`, `Course.qualified_instructor_ids`
8. **Derive cohort pairs** — automatically detect A/B subgroup pairs (e.g. BME1A↔BME1B)
9. **Feasibility preflight** — check for impossible combinations (instructor with 0 qualifications, etc.)

### 8.3 Event Building (`build_events_with_domains`)

**File**: `src/pipeline/build_events.py`

This is the bridge between the OOP domain model and the dense numeric representation:

1. Generate a **reference individual** (one random population member) to get the canonical event list
2. **Sort events** by a stable key: `(course_id, course_type, sorted(group_ids), num_quanta)` — ensures deterministic ordering
3. Build **mapping tables**: `instructor_id ↔ instructor_idx`, `room_id ↔ room_idx`
4. For each event, compute:
   - `allowed_instructors[e]`: indices of all qualified instructors
   - `allowed_rooms[e]`: indices of all suitable rooms (type + capacity + specific features)
   - `allowed_starts[e]`: valid start quanta (0 to max_quanta - duration)
5. Record instructor/room **availability** as quantum sets
6. Compute a `data_hash` (SHA-256 of all input JSONs) for change detection
7. Pickle everything to `events_with_domains.pkl`

### 8.4 The PKL Schema

```python
{
    "schema_version": 2,
    "data_hash": "abc123...",
    "events": [
        {"idx": 0, "course_id": "ENCT 101", "course_type": "theory",
         "group_ids": ["BME1A", "BME1B"], "num_quanta": 2},
        ...
    ],
    "allowed_instructors": [[3, 7], [1, 4, 9], ...],  # per event
    "allowed_rooms": [[0, 5, 8], [2, 6], ...],         # per event
    "allowed_starts": [[0,1,...,40], [0,1,...,39], ...], # per event
    "instructor_to_idx": {"INS001": 0, "INS002": 1, ...},
    "room_to_idx": {"A997": 0, "B211": 1, ...},
    "idx_to_instructor": {0: "INS001", 1: "INS002", ...},
    "idx_to_room": {0: "A997", 1: "B211", ...},
    "instructor_available_quanta": {0: None, 5: {7,8,9,...}},  # None=full-time
    "room_available_quanta": {0: None, ...},
}
```

---

## 9. Operators: Sampling, Crossover, Mutation, Repair

### 9.1 Sampling: `ConstructiveSampling`

Instead of pure random initialisation, the constructive sampler builds near-feasible individuals:

1. **Priority ordering**: Events with tightest groups (most quanta needed) are placed first
2. For each event (in priority order):
   - Find domain-valid (instructor, room, start) triples
   - Prefer triples that avoid conflicts with already-placed events
   - Use occupancy maps (`room_occ`, `inst_occ`, `group_occ`) for O(duration) conflict checks
3. Result: initial population with far fewer conflicts than pure random

An alternative `RandomDomainSampling` is also available (100× faster, but lower quality).

### 9.2 Crossover: `EventBlockCrossover`

**Type**: Uniform event crossover with probability $p$ (default 0.5)

For each mating:

1. Generate a boolean mask of size $E$ — each event independently has probability $p$ of being swapped
2. **Offspring 1** = parent 1's events where mask=True, parent 2's events where mask=False
3. **Offspring 2** = the complement

```python
# Fully vectorized (no Python loop over events):
mask_events = np.random.random((n_matings, E)) < prob
mask_genes = np.repeat(mask_events, 3, axis=1)  # expand to gene-level
Y[0] = np.where(mask_genes, X[0], X[1])
Y[1] = np.where(mask_genes, X[1], X[0])
```

This swaps the complete (instructor, room, time) triple for each selected event, preserving internal gene consistency.

### 9.3 Mutation: `EventLocalMutation`

**Type**: Per-event mutation with probability $p_m$ (default 0.05)

For each selected event:

1. Randomly choose which gene(s) to mutate (instructor, room, time — at least 1)
2. Sample new value from the event's **allowed domain**
3. This is also fully vectorized using padded domain arrays

### 9.4 Repair: Multi-Stage Constraint Repair

The repair operator is the most sophisticated component. It runs after crossover+mutation to fix constraint violations.

**Stage 1 — Domain Repair**: Fix any out-of-domain values

```python
if inst[e] not in allowed_instructors[e]:
    inst[e] = allowed_instructors[e][0]
```

**Stage 2 — Conflict Repair**: Fix exclusivity violations using incremental occupancy maps

```python
for each conflicting event e (by severity, worst-first):
    remove e from occupancy maps
    try alternative (instructor, room, time) placements
    add best placement back to maps
```

**Stage 3 — Group Conflict Repair**: Handle the family hierarchy (parent↔subgroup) conflicts that aren't captured by simple occupancy

The repair runs for multiple passes (configurable, default 5), alternating between worst-first and best-first ordering to avoid cycles.

### 9.5 Vectorized Repair (`PymooVectorizedRepair`)

The production repair operator processes the **entire population** at once, using NumPy for conflict detection and domain enforcement across all individuals simultaneously.

---

## 10. Vectorized Evaluation: The Performance Engine

### 10.1 The Problem

Evaluating ~600 events across ~100 individuals per generation requires checking ~480,000 event-quantum pairs. Doing this with Python loops and `Timetable` construction would be prohibitively slow.

### 10.2 Hard Constraint Vectorization

**File**: `src/pipeline/fast_evaluator_vectorized.py`

The key insight: **expand events into quanta**, then use NumPy broadcasting over the entire population.

1. **Precompute expansion arrays** (once):

   ```python
   # For E events with total Q quanta:
   exp_event[q_idx] = which event this quantum belongs to
   exp_offset[q_idx] = offset within the event's duration
   # For groups: further expand by (event, group, quantum)
   ```

2. **Evaluate all individuals simultaneously**:

   ```python
   # Extract start times for all (individuals, events):
   start_times = X[:, 2::3]  # shape (N, E)
   
   # Expand to quanta: actual_quanta[n, q] = start_times[n, exp_event[q]] + exp_offset[q]
   actual_quanta = start_times[:, exp_event] + exp_offset  # shape (N, Q)
   
   # Build occupancy tensors using np.add.at / np.bincount
   # Detect where occupancy > 1 → violations
   ```

3. **Result**: `G` matrix of shape `(N, 9)` — 9 hard constraint violation counts per individual.

### 10.3 Soft Constraint Vectorization

**File**: `src/pipeline/soft_evaluator_vectorized.py`

Similar approach for the top 3 soft constraints:

- Build `(individual, group/instructor, day)` occupancy bitmasks
- Compute gap penalties vectorized across all dimensions
- Return `S` vector of shape `(N,)` — total soft penalty per individual

### 10.4 Performance Impact

The vectorized evaluator processes all 100 individuals in a **single NumPy call** (no Python loop over individuals). This is ~50-100× faster than the OOP `Timetable` → `Evaluator` path.

---

## 11. Experiment System & Modes

### 11.1 Architecture

```
BaseExperiment (ABC)            ← Logging, timing, output dirs, JSON export
  └── GAExperiment              ← pymoo pipeline orchestration
       ├── BaselineExperiment   ← Pure NSGA-II + repair
       ├── MemeticExperiment    ← + elite bitset repair
       ├── AggressiveExperiment ← Large pop, high mutation, full repair
       ├── AdaptiveExperiment   ← Stagnation-aware mutation scaling
       └── CPHybridExperiment   ← + periodic CP-SAT deep polish
```

### 11.2 Mode Details

| Mode | Pop | Gen | Mutation | Special |
|------|-----|-----|----------|---------|
| **Baseline** | 100 | 200 | 0.05 | Pure NSGA-II, vectorized repair only |
| **Memetic** | 80 | 150 | 0.08 | Top 5% elite get additional bitset repair |
| **Aggressive** | 200 | 100 | 0.15 | 2× offspring, full-population repair |
| **Adaptive** | 100 | 300 | 0.05→0.20 | Ramps mutation when stagnating for 15 gens |
| **CP Hybrid** | 60 | 100 | 0.05 | Every 10 gens, best individual gets CP-SAT polish (ortools) |

### 11.3 Callback System

Each mode can inject a per-generation callback that:

- Logs convergence metrics (best hard, best soft)
- Computes MOEA quality indicators (hypervolume, spacing, IGD)
- Applies mode-specific operators (elite repair, mutation scaling)
- Records timing data

### 11.4 Output Artefacts

Each run produces (in `output/ga_<mode>/<timestamp>/`):

| File | Content |
|------|---------|
| `results.json` | Full results including convergence history |
| `run.log` | Detailed text log |
| `feasibility_report.txt` | Pre-run data quality check |
| `hard_violations.png` | Hard penalty convergence plot |
| `soft_violations.png` | Soft penalty convergence plot |
| `pareto_front.png` | Final population in objective space |
| `convergence_dashboard.png` | Combined HV, spacing, diversity, feasibility |
| `calendar_schedule.pdf` | Group-wise weekly timetable |
| `instructor_schedules.pdf` | Instructor-wise view |
| `room_schedules.pdf` | Room-wise view |
| `log_violations.log` | Remaining constraint violations |

---

## 12. Reinforcement Learning Integration

**Directory**: `src/rl/`

The project has an RL layer that wraps the GA repair process as a Gymnasium environment:

- **State**: Current constraint violation counts + schedule features
- **Action**: Which repair operator to apply (time shift, room swap, instructor swap)
- **Reward**: Reduction in hard/soft violations

Agents (PPO, DQN from Stable-Baselines3) learn which repair operator is most effective in different situations. This is used in the advanced experiment modes (`rl_01` through `rl_10`).

The RL integration follows a **hybrid** architecture:

- GA handles population-level search (exploration)
- RL guides per-individual repair (exploitation)

---

## 13. Software Engineering Architecture

### 13.1 Layer Diagram

```
┌────────────────────────────────────────────┐
│            runs/ (Entry Points)            │  ga_01_baseline.py, etc.
├────────────────────────────────────────────┤
│         src/experiments/                   │  BaseExperiment → GAExperiment
├────────────────────────────────────────────┤
│         src/pipeline/                      │  pymoo operators, encoding,
│                                            │  vectorized evaluators, repair
├────────────────────────────────────────────┤
│         src/constraints/                   │  Constraint protocol, Evaluator
├────────────────────────────────────────────┤
│         src/ga/                            │  Core population, operators,
│                                            │  heuristics, repair strategies
├────────────────────────────────────────────┤
│         src/domain/                        │  Entity dataclasses, Timetable
├────────────────────────────────────────────┤
│         src/io/                            │  Data loading, time system,
│                                            │  decoding, export (PDF/JSON)
├────────────────────────────────────────────┤
│         src/utils/                         │  Logging, parallelism, profiling
└────────────────────────────────────────────┘
```

### 13.2 Dependency Flow

```
runs/ → experiments/ → pipeline/ → constraints/ → domain/
                    ↘                            ↗
                     → ga/core/ → ga/operators/ ──┘
                                                 
io/ is used across all layers (data loading, time system, export)
```

### 13.3 Key Design Patterns

| Pattern | Where | Why |
|---------|-------|-----|
| **Protocol (structural typing)** | `Constraint` protocol | Constraints don't need a base class; just implement `evaluate()` |
| **Factory** | `PopulationFactory`, `build_constraints()`, `create_algorithm()` | Decouple creation from use |
| **Strategy** | Experiment modes, repair strategies | Swap algorithms without changing the pipeline |
| **Registry** | `ALL_CONSTRAINTS`, `HARD_CONSTRAINT_CLASSES` | Discoverable constraint list |
| **Immutable indexes** | `Timetable._build_indexes()` | Compute once, share across all evaluators |
| **Dependency injection** | `SessionGene.set_time_system()` | Avoid global singleton for time system |
| **Union-Find** | `build_clusters()` | Efficient programme coupling detection |
| **Observer/Callback** | pymoo callbacks for per-gen logging | Extensible without modifying the core loop |

### 13.4 Type Safety

- **Pydantic** for configuration models (runtime validation)
- **Python dataclasses** with `slots=True` for domain entities (memory-efficient)
- **Type hints** throughout with `from __future__ import annotations`
- **mypy** targeted strictness on core modules
- **Runtime protocol checking** for constraints (`@runtime_checkable`)

### 13.5 Testing

Tests in `tests/` cover:

- **Constraint correctness**: Each hard/soft constraint tested in isolation
- **Operator correctness**: Crossover preserves (course, group) structure
- **Equivalence tests**: Vectorized evaluator matches OOP evaluator
- **Algorithm correctness**: Population generation, gene validity
- **Data store**: Loading, serialization roundtrip
- **Integration**: Full pipeline from JSON to evaluated schedule

### 13.6 Parallelism

- **Population generation**: `ProcessPoolExecutor` for parallel individual construction
- **Fitness evaluation**: Fully vectorized with NumPy (no multiprocessing needed — single-threaded NumPy is faster due to BLAS)
- **PDF export**: Optional parallel generation
- **Worker pattern**: `init_worker()` / `get_worker_context()` for passing data to subprocesses without pickling large objects

---

## 14. Data Flow Walkthrough: A Full Run

Let's trace what happens when you run `python runs/ga_01_baseline.py`:

### Step 1: Entry Point

```python
exp = BaselineExperiment(seed=42, pop_size=100, ngen=200, ...)
exp.run()
```

### Step 2: `BaseExperiment.run()`

- Creates timestamped output directory: `output/ga_baseline/20260222_123456/`
- Sets up unified logging (console + file + JSONL)
- Calls `_execute()`

### Step 3: `GAExperiment._execute()`

#### 3a. Ensure PKL exists

```python
pkl_path = self._ensure_pkl()
# → If missing: build_events_with_domains("data")
# → Loads DataStore, generates reference individual, computes domains
# → Saves events_with_domains.pkl
```

#### 3b. Load data

```python
store, ctx, qts = self._load_data()
# → DataStore.from_json("data")
# → Feasibility preflight (warns but doesn't abort)
# → ctx = SchedulingContext with all entities
```

#### 3c. Create problem + algorithm

```python
prob = create_problem(pkl_path, ctx=ctx, qts=qts)
# → SchedulingProblem: n_var=1800, n_obj=2, n_constr=8
# → Precomputes VectorizedEvalData + SoftVectorizedData

algo = create_algorithm(pkl_path, pop_size=100, use_repair=True)
# → NSGA2 with:
#   - ConstructiveSampling (build 100 near-feasible individuals)
#   - EventBlockCrossover (50% per-event swap)
#   - EventLocalMutation (5% per-event)
#   - PymooVectorizedRepair (5-pass conflict resolution)
```

#### 3d. Run evolution

```python
res = minimize(prob, algo, ("n_gen", 200), seed=42, callback=callback)
```

Each generation:

1. **Select** parents (binary tournament on rank + crowding distance)
2. **Crossover**: Parents paired, event triples swapped with p=0.5
3. **Mutation**: Each event in each offspring mutated with p=0.05
4. **Repair**: Fix domain violations, then fix conflicts (5 passes)
5. **Evaluate**: Vectorized hard (8 constraints) + soft (3 constraints)
6. **Select survivors**: Non-dominated sort + crowding distance
7. **Callback**: Log best hard/soft, compute HV/spacing/diversity

#### 3e. Extract results

```python
best_idx = argmin(constraint_violations)
F[best_idx] → (hard=47, soft=633)
```

### Step 4: Output Generation

- Convergence plots (hard, soft, per-constraint)
- Pareto front scatter plot
- MOEA metric trends (hypervolume, spacing, diversity)
- Schedule PDFs (decode best chromosome → SessionGene → CourseSession → PDF)
- Violation report

### Step 5: Save Results

```json
{
    "solver": "pymoo",
    "mode": "baseline",
    "best_hard": 47.0,
    "best_soft": 633.0,
    "n_feasible": 0,
    "elapsed_s": 419.54,
    "convergence_hard": [1200, 980, 750, ..., 47],
    "convergence_soft": [500, 520, 580, ..., 633],
    ...
}
```

---

## 15. Key Design Decisions & Trade-offs

### 15.1 Why NSGA-II over Simulated Annealing / ILP?

| Approach | Pros | Cons |
|----------|------|------|
| **NSGA-II** (chosen) | Multi-objective, good for large instances, anytime (returns best-so-far), parallelizable | Stochastic, no optimality guarantee |
| **ILP/CP-SAT** | Optimal if feasible, provable bounds | Exponential time for large instances, single-objective |
| **Simulated Annealing** | Simple, good for single-objective | Single-objective, poor for multi-objective |

The project uses NSGA-II as primary solver and optionally integrates CP-SAT as a **repair operator** (CPHybridExperiment) to polish specific individuals.

### 15.2 Why Contiguous Quanta?

The `(start_quanta, num_quanta)` design replaced a `quanta: list[int]` field. Benefits:

- **Structural impossibility of fragmentation**: A session can't accidentally have gaps
- **60% memory reduction**: 2 integers vs N-element list
- **Simpler validation**: Just check `start + duration ≤ day_end` vs scanning for gaps
- **Natural crossover**: Swapping `start_quanta` preserves duration automatically

### 15.3 Why Pre-Built Indexes (Timetable)?

Previously, every constraint evaluation rebuilt its own group→quantum map from scratch. The Timetable class builds all indexes once and shares them:

- **14 redundant map-rebuilds eliminated** per evaluation
- **O(1) lookups** for conflict detection
- **Single source of truth** for decoded schedule state

### 15.4 Why the PKL Intermediate?

The `events_with_domains.pkl` serves as a clean boundary between:

- **OOP domain model** (dataclasses, string IDs, Python objects)
- **Dense numeric pipeline** (NumPy arrays, integer indices)

This means the expensive domain computation (room suitability, instructor qualifications) runs **once** and is reused across all pymoo operations.

### 15.5 Dual Encoding System

The project has two parallel encoding systems:

| Aspect | OOP Layer (`src/ga/`) | Numeric Layer (`src/pipeline/`) |
|--------|----------------------|-------------------------------|
| Chromosome | `list[SessionGene]` | `np.ndarray` (3×E ints) |
| Evaluator | `Timetable` → `Evaluator` | Vectorized NumPy kernels |
| Operators | Python crossover/mutation | NumPy vectorized operators |
| Speed | ~10× slower | Production speed |
| Debug | Fully inspectable | Requires chromosome→gene bridge |

The OOP layer exists for debugging, testing, and the RL integration. The numeric layer is the production path.

---

## 16. How to Run & Extend

### 16.1 Running Experiments

```bash
# Install dependencies
uv sync --frozen

# Run baseline (simplest)
python runs/ga_01_baseline.py

# Run with local search
python runs/ga_02_memetic.py
```

Each script has inline configuration — edit `POP_SIZE`, `NGEN`, `MUTATION_PROB` etc. directly.

### 16.2 Adding a New Constraint

1. Create a class in `src/constraints/constraints.py`:

   ```python
   class MyNewConstraint:
       kind: str = "soft"  # or "hard"
       
       def __init__(self, weight: float = 1.0, my_param: float = 5.0):
           self.name = "my_new_constraint"
           self.weight = weight
           self.my_param = my_param
       
       def evaluate(self, tt: Timetable) -> float:
           penalty = 0.0
           for gene in tt.genes:
               # your logic here
               pass
           return penalty
   ```

2. Add to `SOFT_CONSTRAINT_CLASSES` (or `HARD_CONSTRAINT_CLASSES`)
3. Add to `build_constraints()` factory
4. For production speed: add vectorized version in `soft_evaluator_vectorized.py`
5. Add tests in `tests/test_constraints_soft.py`

### 16.3 Adding a New Experiment Mode

1. Create a subclass in `src/experiments/ga_experiment.py`:

   ```python
   class MyExperiment(GAExperiment):
       def __init__(self, **kwargs):
           kwargs.setdefault("mode", "my_mode")
           kwargs.setdefault("pop_size", 150)
           super().__init__(**kwargs)
       
       def _build_callback(self, pkl_path):
           class CB(GACallbackBase):
               def _on_generation(self, algorithm, F, G, cv, best_idx):
                   # your per-generation logic
                   pass
           return CB(self.log_interval)
   ```

2. Create a run script in `runs/`:

   ```python
   from src.experiments import MyExperiment
   exp = MyExperiment(seed=42, ngen=200)
   exp.run()
   ```

### 16.4 Modifying Input Data

Edit the JSON files in `data/`:

- `Course.json`: Add/remove courses, change L/T/P hours
- `Groups.json`: Add student groups, change subgroup structure
- `Instructors.json`: Add instructors, change qualifications
- `Rooms.json`: Add rooms, change features/capacity

After editing, **delete `events_with_domains.pkl`** — it will be rebuilt on next run with the new data hash.

---

## Glossary

| Term | Meaning |
|------|---------|
| **Quantum** | Atomic time unit (default 60 min); index 0–41 for 6-day × 7-hour week |
| **Event** | A single scheduled session (= one gene) — a specific course-type for specific groups |
| **Gene** | `SessionGene` — represents one event's complete assignment |
| **Individual** | `list[SessionGene]` — a complete weekly timetable for all courses |
| **Chromosome** | Flat NumPy encoding of an individual: `[I₀,R₀,T₀, I₁,R₁,T₁, ...]` |
| **Population** | Set of individuals being evolved |
| **Pareto front** | Set of non-dominated solutions (best feasible trade-offs) |
| **Hard constraint** | Physical impossibility (must be 0 for valid schedule) |
| **Soft constraint** | Quality preference (minimise for better schedule) |
| **Domain** | Set of valid values for a decision variable (per event) |
| **Repair** | Post-operator fix-up that resolves constraint violations |
| **QTS** | QuantumTimeSystem — the time→integer mapping module |
| **PKL** | `events_with_domains.pkl` — cached numeric problem representation |
| **CV** | Constraint violation — sum of hard constraint counts |
| **HV** | Hypervolume — MOEA quality indicator (higher = better Pareto front) |
| **IGD** | Inverted Generational Distance — how close front is to a reference |

---

*This document covers the project as of February 2026, branch `feat/pymoo-only`.*
