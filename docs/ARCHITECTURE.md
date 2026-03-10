# Schedule Engine — Architecture

> **Version 1.0.0** · Python 3.12 · MIT License

## Overview

Schedule Engine is a multi-objective university course scheduling system that combines
evolutionary algorithms (NSGA-II via pymoo) with reinforcement learning (PPO/DQN via
Stable-Baselines3) to optimise timetables against 8 hard and 6 soft constraints.

The system solves a real-world NP-hard scheduling problem: assign every
course-session an instructor, a room, and a time-slot so that no resource
conflicts exist (hard constraints) while maximising schedule quality (soft
constraints such as compactness, lunch breaks, and session continuity).

---

## High-Level Data Flow

```
┌──────────────┐     ┌────────────────┐      ┌────────────────────┐
│ data/*.json  │────│  DataStore /    │─────│ SchedulingContext   │
│ (Course,     │     │  QuantumTime   │      │ (courses, groups,   │
│  Groups,     │     │  System        │      │  instructors, rooms,│
│  Instructors,│     └────────────────┘      │  available_quanta)  │
│  Rooms)      │                             └────────┬───────────┘
└──────────────┘                                      │
                                                      ▼
                                        ┌──────────────────────────┐
                                        │   PopulationFactory      │
                                        │ random / greedy / smart  │
                                        └────────────┬─────────────┘
                                                     │
                                     ┌───────────────┼────────────────┐
                                     ▼               ▼                ▼
                            ┌──────────────┐ ┌─────────────┐ ┌──────────────────┐
                            │  GA Pipeline │ │ RL Pipeline  │ │ CP-SAT / LNS     │
                            │  (pymoo      │ │ (gymnasium   │ │ (hybrid repair)  │
                            │   NSGA-II)   │ │  + SB3)      │ │                  │
                            └──────┬───────┘ └──────┬──────┘ └────────┬─────────┘
                                   │                │                  │
                                   ▼                ▼                  ▼
                            ┌────────────────────────────────────────────────┐
                            │           Evaluator (8 hard + 6 soft)         │
                            │   Timetable → constraint violation scores     │
                            └────────────────────────┬──────────────────────┘
                                                     │
                                                     ▼
                            ┌────────────────────────────────────────────────┐
                            │   Export: PDF timetables, plots, CSV, JSON    │
                            └────────────────────────────────────────────────┘
```

---

## Package Structure

```
src/
├── __init__.py                           # Version, author metadata
├── config/                               # Global Config singleton
│   └── __init__.py                       #   Config, init_config(), get_config()
│
├── domain/                               # Core data models
│   ├── course.py                         #   Course dataclass
│   ├── group.py                          #   Group dataclass
│   ├── instructor.py                     #   Instructor dataclass
│   ├── room.py                           #   Room dataclass
│   ├── gene.py                           #   SessionGene — the scheduling atom
│   ├── session.py                        #   CourseSession (decoded view)
│   ├── timetable.py                      #   Timetable (indexed gene collection)
│   └── types.py                          #   Individual, SchedulingContext
│
├── constraints/                          # Constraint evaluation system
│   ├── constraints.py                    #   Constraint protocol + registries
│   ├── evaluator.py                      #   Evaluator (fitness from timetable)
│   ├── hard/                             #   8 hard constraint classes
│   │   ├── cte.py                        #     StudentGroupExclusivity
│   │   ├── fte.py                        #     InstructorExclusivity
│   │   ├── sre.py                        #     RoomExclusivity
│   │   ├── fpc.py                        #     InstructorQualifications
│   │   ├── ffc.py                        #     RoomSuitability
│   │   ├── fca.py                        #     InstructorTimeAvailability
│   │   ├── cqf.py                        #     CourseCompleteness
│   │   └── ictd.py                       #     SiblingSameDay
│   └── soft/                             #   6 soft constraint classes
│       ├── csc.py                        #     StudentScheduleCompactness
│       ├── fsc.py                        #     InstructorScheduleCompactness
│       ├── mip.py                        #     StudentLunchBreak
│       ├── session_continuity.py         #     SessionContinuity
│       ├── sscp.py                       #     PairedCohortPracticalAlignment
│       └── break_placement.py            #     BreakPlacementCompliance
│
├── io/                                   # Data loading and I/O
│   ├── data_store.py                     #   DataStore (JSON → domain objects)
│   ├── loader.py                         #   load_courses/groups/instructors/rooms
│   ├── time_system.py                    #   QuantumTimeSystem (continuous→discrete)
│   ├── validator.py                      #   Input data validation
│   ├── feasibility.py                    #   Pigeonhole feasibility checks
│   ├── decoder.py                        #   Gene → CourseSession decoder
│   └── export/                           #   Output generation
│       ├── pdf.py                        #     PDF timetable export
│       ├── json_export.py               #     JSON result export
│       └── plot_*.py                     #     16 matplotlib plot modules
│
├── ga/                                   # Genetic algorithm components
│   ├── core/                             #   Population, evaluation, indexing
│   │   ├── population_factory.py         #     PopulationFactory
│   │   ├── evaluator.py                  #     evaluate(), evaluate_detailed()
│   │   └── schedule_index.py             #     ScheduleIndex (fast lookup)
│   ├── operators/                        #   Genetic operators
│   │   ├── crossover.py                  #     Course-group-aware crossover
│   │   ├── mutation.py                   #     Gene/individual mutation
│   │   └── local_search.py              #     Hill-climbing local search
│   ├── repair/                           #   15+ repair strategies
│   │   ├── basic.py                      #     7 registered repair operators
│   │   ├── selective.py                  #     Violation-targeted repair (3-4× faster)
│   │   ├── engine.py                     #     RepairEngine (RL-ready, lexicographic)
│   │   ├── pipeline.py                   #     RepairPipeline (orchestrator)
│   │   ├── detector.py                   #     Violation detection
│   │   ├── greedy.py                     #     Greedy repair
│   │   ├── exhaustive.py                 #     Exhaustive repair
│   │   ├── memetic.py                    #     Memetic repair
│   │   ├── igls.py                       #     IGLS repair
│   │   ├── heuristic_repair.py           #     Multi-strategy heuristic repair
│   │   ├── selective_heuristic.py        #     Selective heuristic repair
│   │   ├── parallel.py                   #     Multiprocessing population repair
│   │   ├── break_repair.py              #     Break placement repair
│   │   ├── group_clash_repair.py        #     Group clash resolution
│   │   ├── hierarchy.py                 #     Cascade/family-aware repair
│   │   ├── conflict_detection.py        #     Conflict detection utilities
│   │   ├── wrappers.py                  #     @repair_operator registry
│   │   ├── cp/                           #     CP-SAT constraint programming
│   │   │   ├── pipeline.py              #       CPRepairPipeline
│   │   │   ├── solver.py                #       CPSATSolver (ortools)
│   │   │   ├── partitioner.py           #       Gene partitioning
│   │   │   ├── merger.py                #       Result merging
│   │   │   └── frozen_selector.py       #       Frozen gene selection
│   │   └── lns/                          #     Large Neighbourhood Search
│   │       ├── operator.py              #       LNS-IGLS repair
│   │       ├── repair.py                #       LNS heuristic wrapper
│   │       └── diagnostics.py           #       Conflict graph analysis
│   └── metrics/                          #   Quality metrics
│       ├── hypervolume.py               #     Hypervolume indicator
│       ├── diversity.py                 #     Population diversity
│       ├── convergence.py               #     Convergence rate
│       ├── spacing.py                   #     Spacing metric
│       └── igd.py                       #     Inverted Generational Distance
│
├── pipeline/                             # Pymoo integration layer
│   ├── scheduling_problem.py             #   SchedulingProblem (pymoo Problem)
│   ├── encoding.py                       #   Chromosome  gene encoding
│   ├── pymoo_operators.py               #   Sampling, crossover, mutation
│   ├── repair_operator.py               #   SchedulingRepair (pymoo Repair)
│   ├── repair_operator_bitset.py        #   BitsetSchedulingRepair (Numba JIT)
│   ├── batch_api.py                     #   BatchContext, batch eval/repair
│   ├── fast_evaluator_vectorized.py     #   Vectorized hard eval (numpy)
│   ├── fast_evaluator_batch.py          #   Batch hard eval
│   └── soft_eval_vectorized.py          #   Vectorized soft eval
│
├── rl/                                   # Reinforcement learning
│   ├── gym_env/                          #   Gymnasium environment
│   │   ├── pymoo_env.py                 #     PymooHyperHeuristicEnv
│   │   ├── fast_state_encoder.py        #     VectorizedStateEncoder (39-D obs)
│   │   └── reward_shaper.py             #     PBRS reward shaping
│   ├── actions/                          #   LLH action space
│   │   ├── vectorized_ops.py            #     6 repair action classes
│   │   ├── repairs/                     #     Repair action implementations
│   │   ├── perturbations/               #     Perturbation actions
│   │   ├── optimizations/               #     Optimization actions
│   │   └── utils/                       #     Action utilities
│   ├── agents/                           #   Agent factories
│   │   ├── ppo.py                       #     create_ppo_agent()
│   │   ├── dqn.py                       #     create_dqn_agent()
│   │   └── random_agent.py             #     RandomAgent
│   └── training/                         #   Training infrastructure
│       ├── trainer.py                   #     RLTrainer, create_trainer()
│       ├── callbacks.py                 #     Training callbacks
│       └── thesis_plots.py             #     Publication plot generation
│
├── experiments/                          # Experiment runners (GA modes)
│   ├── base.py                          #   BaseExperiment (ABC)
│   └── ga_experiment.py                 #   GAExperiment + 5 concrete modes
│
└── utils/                                # Cross-cutting utilities
    ├── logging.py                       #   Unified logging (rich console)
    ├── console.py                       #   Rich console singleton
    └── system_info.py                   #   Hardware/software info dump
```

---

## Core Abstractions

### The Scheduling Atom: `SessionGene`

Every scheduled event is a `SessionGene` — the smallest unit of the timetable:

```python
@dataclass
class SessionGene:
    course_id: str        # Which course
    course_type: str      # "theory" or "practical"
    instructor_id: str    # Who teaches
    group_ids: list[str]  # Which student groups attend
    room_id: str          # Where
    start_quanta: int     # When (discrete quantum index)
    num_quanta: int       # Duration (in quanta)
```

An **Individual** (candidate timetable) is `list[SessionGene]`.

### The Quantum Time System

Continuous clock time is discretized into **quanta** (default: 60-minute blocks).
`QuantumTimeSystem` handles:

- Day × time-slot → quantum index mapping
- Operating hours, midday breaks, break windows
- Theory/practical block sizing constraints

### Timetable: The Indexed View

`Timetable` wraps a list of genes with pre-built occupancy indexes
for O(1) conflict detection:

- `group_occupancy[(group_id, quantum)]` → gene indices
- `instructor_occupancy[(instructor_id, quantum)]` → gene indices
- `room_occupancy[(room_id, quantum)]` → gene indices
- Day-level, course-level, and practical-level aggregations

### SchedulingContext: The Problem Instance

`SchedulingContext` bundles all problem data:

```python
@dataclass
class SchedulingContext:
    courses: dict[tuple[str, str], Course]   # (course_id, type) → Course
    groups: dict[str, Group]
    instructors: dict[str, Instructor]
    rooms: dict[str, Room]
    available_quanta: list[int]
    cohort_pairs: list[tuple[str, str]]      # Paired practical groups
    family_map: dict[str, set[str]]          # Course family relationships
```

---

## Constraint System

### Hard Constraints (must be zero for feasibility)

| Code | Constraint | What it checks |
|------|-----------|---------------|
| CTE | StudentGroupExclusivity | No student group in two places at once |
| FTE | InstructorExclusivity | No instructor in two classes at once |
| SRE | RoomExclusivity | No room double-booked |
| FPC | InstructorQualifications | Instructor is qualified for the course |
| FFC | RoomSuitability | Room type matches course type |
| FCA | InstructorTimeAvailability | Part-time instructor available at slot |
| CQF | CourseCompleteness | Correct total hours per course per group |
| ICTD | SiblingSameDay | Sub-sessions of a course not on same day |

### Soft Constraints (minimised for quality)

| Code | Constraint | What it optimises |
|------|-----------|------------------|
| CSC | StudentScheduleCompactness | Minimise gaps in student daily schedules |
| FSC | InstructorScheduleCompactness | Minimise gaps in instructor schedules |
| MIP | StudentLunchBreak | Ensure midday break exists |
| SC | SessionContinuity | Prefer contiguous session blocks |
| SSCP | PairedCohortPracticalAlignment | Synchronise paired practical groups |
| BPC | BreakPlacementCompliance | Breaks in designated windows |

### Evaluator

The `Evaluator` class provides the fitness function:

```python
evaluator = Evaluator()  # uses default ALL_CONSTRAINTS
hard, soft = evaluator.fitness(genes, context, qts)
breakdown = evaluator.breakdown(genes, context, qts)
```

---

## GA Pipeline (pymoo)

The genetic algorithm pipeline uses pymoo's NSGA-II:

1. **Encoding** (`EncodingSpec`): Each gene maps to 3 decision variables
   `[instructor_idx, room_idx, start_quanta]`. A chromosome is a flat numpy array
   of shape `(n_events × 3,)`.

2. **Problem** (`SchedulingProblem`): pymoo `Problem` with 2 objectives (hard, soft)
   and 8 inequality constraints (one per hard constraint).

3. **Operators**:
   - `ConstructiveSampling` / `RandomDomainSampling` — initialisation
   - `EventBlockCrossover` — swap event triples between parents
   - `EventLocalMutation` — per-event (instructor, room, time) mutation

4. **Repair**: Multi-stage pipeline from fast domain fixes to CP-SAT deep polish.

### Repair Hierarchy

```
RepairPipeline (orchestrator)
├── basic.py     — 7 registered operators (priority-ordered)
├── selective.py — violation-targeted variants (3-4× faster)
├── engine.py    — RL-ready engine (epsilon-greedy / round-robin)
├── parallel.py  — multiprocessing across population
├── cp/          — CP-SAT solver (ortools) for deep polish
└── lns/         — Large Neighbourhood Search + IGLS
```

---

## RL Pipeline (Hyper-Heuristic)

The RL system operates as a **hyper-heuristic**: instead of directly manipulating
the timetable, the RL agent selects which low-level heuristic (LLH) to apply
at each generation of the GA.

### Environment: `PymooHyperHeuristicEnv`

- **Observation** (39-D float32): fitness stats, constraint violations, diversity
  metrics, progress indicators, and action history.
- **Actions** (Discrete(6)): six pipeline configuration actions that control
  repair intensity, elite fraction, and strategy.
- **Reward**: phase-transition signal — amplified soft improvement once hard
  constraints converge below threshold, plus one-time feasibility bonus.

### Action Space

| ID | Action | Strategy |
|----|--------|----------|
| 0 | ConservativeRepair | 10% elite, 2 passes |
| 1 | AggressiveRepair | 25% elite, 3 passes |
| 2 | MemeticEliteRepair | 15% elite, 4 passes |
| 3 | SoftFocusRepair | 8% elite, 2 passes + time compaction |
| 4 | DestructiveConstructive | Ruin worst 10%, rebuild 20% elite |
| 5 | IntensifiedRepair | 20% elite, 3 passes |

### Agent Support

- **PPO** (MaskablePPO from sb3-contrib): primary agent with action masking
- **DQN** (Stable-Baselines3): competitor agent
- **RandomAgent**: baseline

### Training Infrastructure

`RLTrainer` manages the full lifecycle:

- Agent creation (PPO/DQN)
- TensorBoard logging
- Checkpointing
- Curriculum learning (3-phase constraint schedule)
- PBRS reward shaping

---

## Experiment System

All experiments inherit from `BaseExperiment`, which provides:

- Timestamped output directories
- Dual logging (file + console via `rich`)
- Timing and JSON result export

`GAExperiment` extends it for pymoo NSGA-II runs with 5 concrete modes:

| Mode | Class | Strategy |
|------|-------|----------|
| 01 | `BaselineExperiment` | Pure NSGA-II (no repair) |
| 02 | `MemeticExperiment` | NSGA-II + bitset elite repair |
| 03 | `AggressiveExperiment` | 2× offspring, high mutation, full repair |
| 04 | `AdaptiveExperiment` | Stagnation-aware mutation escalation |
| 05 | `CPHybridExperiment` | NSGA-II + periodic CP-SAT deep polish |

---

## Design Patterns

### Registry Pattern

Repair operators and constraints use decorator-based registration:

```python
@repair_operator(name="group_overlaps", priority=2, description="Fix HC1")
def repair_group_overlaps(individual, context):
    ...
```

### Protocol-Based Constraints

All constraints implement the `Constraint` protocol — no ABC inheritance required:

```python
class Constraint(Protocol):
    name: str
    weight: float
    kind: str
    def evaluate(self, tt: Timetable) -> float: ...
```

### Config Singleton

A recursive-namespace `Config` object provides dot-access configuration:

```python
config = Config(ga=dict(pop_size=100, ngen=200))
init_config(config)
# later...
cfg = get_config()
print(cfg.ga.pop_size)  # 100
```

### Vectorized Evaluation

Performance-critical evaluation uses:

- **NumPy vectorization** for batch population evaluation
- **Numba JIT** for inner repair loops (bitset repair)
- **Pre-built indexes** in `Timetable` for O(1) lookups

---

## I/O and Data Format

### Input Data (`data/`)

| File | Contents |
|------|----------|
| `Course.json` | Course definitions (id, quanta, type, features) |
| `Groups.json` | Student groups (id, count, enrolled courses) |
| `Instructors.json` | Instructor profiles (id, qualifications, availability) |
| `Rooms.json` | Room inventory (id, capacity, features) |

### Output (`output/`)

Each experiment run creates a timestamped subdirectory containing:

- `result.json` — final objective values and metadata
- Convergence plots (hard/soft penalty vs generation)
- Pareto front visualisation
- Diversity and hypervolume progression
- PDF timetable exports (per group, per instructor)
