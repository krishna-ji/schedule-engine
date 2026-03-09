# Schedule Engine — API Reference

> Auto-generated reference for all public modules, classes, and functions.
> Docstring convention: **Google style** (`Args:`, `Returns:`, `Raises:`).

---

## Table of Contents

- [src.config](#srcconfig)
- [src.domain](#srcdomain)
- [src.constraints](#srcconstraints)
- [src.io](#srcio)
- [src.io.export](#srcioexport)
- [src.ga.core](#srcgacore)
- [src.ga.operators](#srcgaoperators)
- [src.ga.repair](#srcgarepair)
- [src.ga.repair.cp](#srcgarepaircp)
- [src.ga.repair.lns](#srcgarepairlns)
- [src.ga.metrics](#srcgametrics)
- [src.pipeline](#srcpipeline)
- [src.rl.gym\_env](#srcrlgym_env)
- [src.rl.actions](#srcrlactions)
- [src.rl.agents](#srcrlagents)
- [src.rl.training](#srcrltraining)
- [src.experiments](#srcexperiments)
- [src.utils](#srcutils)

---

## `src.config`

**Exports:** `Config`, `init_config`, `get_config`, `get_config_or_default`, `_deep_merge`

### `Config`

Recursive namespace providing dot-access over a nested dict tree.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(**kwargs)` | Nested dicts auto-wrapped as `Config` |
| `get` | `(key: str, default=None) -> Any` | Dict-style get with default |
| `to_dict` | `() -> dict[str, Any]` | Serialize to plain dict |
| `from_dict` | `(d: dict) -> Config` | Classmethod factory |

### Module Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `init_config` | `(config_obj: Config) -> Config` | Set the global config singleton |
| `get_config` | `() -> Config` | Retrieve singleton (raises `RuntimeError` if unset) |
| `get_config_or_default` | `() -> Config` | Retrieve singleton or empty `Config` |
| `_deep_merge` | `(base: dict, overrides: dict) -> dict` | Deep-merge two dicts |

---

## `src.domain`

**Exports:** `Course`, `SessionGene`, `Group`, `Instructor`, `Room`, `CourseSession`,
`ConflictPair`, `Timetable`, `Individual`, `SchedulingContext`

### `Course` — `src.domain.course`

`@dataclass(slots=True)` — a university course definition.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `course_id` | `str` | required | Unique identifier |
| `name` | `str` | required | Human-readable name |
| `quanta_per_week` | `int` | required | Total quanta needed per week |
| `required_room_features` | `str` | required | `"theory"` or `"practical"` |
| `enrolled_group_ids` | `list[str]` | `[]` | Groups enrolled in this course |
| `qualified_instructor_ids` | `list[str]` | `[]` | Qualified instructors |
| `course_type` | `str` | `"theory"` | Session type |
| `L`, `T`, `P` | `int` | `0` | Lecture/tutorial/practical hours |
| `course_code` | `str` | `""` | Short code |
| `department` | `str` | `""` | Department name |
| `semester` | `str` | `""` | Semester identifier |
| `credits` | `int` | `0` | Credit hours |
| `lecture_hours` | `int` | `0` | Weekly lecture hours |
| `practical_hours` | `int` | `0` | Weekly practical hours |
| `specific_lab_features` | `list[str]` | `[]` | Required lab equipment |

| Method | Returns | Description |
|--------|---------|-------------|
| `is_instructor_qualified(instructor_id)` | `bool` | Check qualification |
| `has_group(group_id)` | `bool` | Check enrolment |
| `get_enrolled_groups()` | `set[str]` | Set of enrolled group IDs |

### `Group` — `src.domain.group`

`@dataclass(slots=True)` — a student group/cohort.

| Attribute | Type | Default |
|-----------|------|---------|
| `group_id` | `str` | required |
| `name` | `str` | required |
| `student_count` | `int` | required |
| `enrolled_courses` | `list[str]` | `[]` |
| `available_quanta` | `set[int]` | `set()` |

| Method | Returns | Description |
|--------|---------|-------------|
| `is_enrolled_in_course(course_id)` | `bool` | Check enrolment |
| `get_enrolled_courses_set()` | `set[str]` | Set of course IDs |
| `get_course_count()` | `int` | Number of courses |

### `Instructor` — `src.domain.instructor`

`@dataclass(slots=True)` — an instructor/faculty member.

| Attribute | Type | Default |
|-----------|------|---------|
| `instructor_id` | `str` | required |
| `name` | `str` | required |
| `qualified_courses` | `list[Any]` | required |
| `is_full_time` | `bool` | `True` |
| `available_quanta` | `set[int]` | `set()` |
| `booked_quanta` | `set[int]` | `set()` |
| `max_hours_per_week` | `int` | `40` |

| Method | Returns | Description |
|--------|---------|-------------|
| `is_qualified_for_course(course_id)` | `bool` | Check qualification |
| `is_available_at_quanta(quanta, time_system)` | `bool` | Availability check |
| `get_available_quanta_ranges(time_system)` | `dict[str, list[tuple]]` | Per-day ranges |
| `get_qualification_set()` | `set[str]` | Set of qualified course IDs |

### `Room` — `src.domain.room`

`@dataclass(slots=True)` — a physical room/lab.

| Attribute | Type | Default |
|-----------|------|---------|
| `room_id` | `str` | required |
| `name` | `str` | required |
| `capacity` | `int` | required |
| `room_features` | `str` | required |
| `available_quanta` | `set[int]` | `set()` |
| `specific_features` | `list[str]` | `[]` |

| Method | Returns | Description |
|--------|---------|-------------|
| `can_accommodate_group_size(group_size)` | `bool` | Capacity check |
| `is_suitable_for_course_type(required_features, lab_features)` | `bool` | Feature match |

### `SessionGene` — `src.domain.gene`

`@dataclass` — the scheduling atom: one session assignment.

| Attribute | Type | Description |
|-----------|------|-------------|
| `course_id` | `str` | Course identifier |
| `course_type` | `str` | `"theory"` or `"practical"` |
| `instructor_id` | `str` | Assigned instructor |
| `group_ids` | `list[str]` | Groups attending |
| `room_id` | `str` | Assigned room |
| `start_quanta` | `int` | Starting quantum index |
| `num_quanta` | `int` | Duration in quanta |

| Property/Method | Returns | Description |
|-----------------|---------|-------------|
| `end_quanta` | `int` | `start_quanta + num_quanta` |
| `time_quantum` | `int` | Getter/setter alias for `start_quanta` |
| `duration_quanta` | `int` | Alias for `num_quanta` |
| `get_quanta_list()` | `list[int]` | All occupied quanta |
| `shift_to(new_start)` | `None` | Reassign start time |
| `overlaps_with(other)` | `bool` | Time overlap check |

Module functions: `set_time_system(qts)`, `get_time_system() -> QuantumTimeSystem`

### `CourseSession` — `src.domain.session`

`@dataclass(slots=True)` — decoded session view for constraint checking.

| Attribute | Type |
|-----------|------|
| `course_id` | `str` |
| `instructor_id` | `str` |
| `group_ids` | `list[str]` |
| `room_id` | `str` |
| `session_quanta` | `list[int]` |
| `required_room_features` | `str` |
| `course_type` | `str` |
| `instructor` | `Instructor` |
| `group` | `Group` |
| `room` | `Room` |

### `Timetable` — `src.domain.timetable`

Indexed collection of `SessionGene` objects with pre-built occupancy maps.

**Constructor:** `Timetable(genes, context, qts)`

| Index | Type | Key |
|-------|------|-----|
| `group_occupancy` | `dict` | `(group_id, quantum) → list[gene_idx]` |
| `instructor_occupancy` | `dict` | `(instructor_id, quantum) → list[gene_idx]` |
| `room_occupancy` | `dict` | `(room_id, quantum) → list[gene_idx]` |
| `group_daily` | `dict` | `group_id → {day: set[within_day_q]}` |
| `instructor_daily` | `dict` | `instructor_id → {day: set[within_day_q]}` |
| `course_group_quanta` | `dict` | `(course_code, type, group_id) → int` |
| `course_daily` | `dict` | `(course_id, type) → {day: list[q]}` |
| `practical_quanta` | `dict` | `(course_id, type, group_id) → set[q]` |

| Method | Returns | Description |
|--------|---------|-------------|
| `genes_for_group(group_id)` | `list[int]` | Gene indices for group |
| `genes_for_instructor(instructor_id)` | `list[int]` | Gene indices for instructor |
| `genes_for_room(room_id)` | `list[int]` | Gene indices for room |
| `genes_at_quantum(quantum)` | `list[int]` | Gene indices at time slot |
| `group_conflicts()` | `list[ConflictPair]` | All group double-bookings |
| `instructor_conflicts()` | `list[ConflictPair]` | All instructor clashes |
| `room_conflicts()` | `list[ConflictPair]` | All room clashes |
| `all_conflicts()` | `list[ConflictPair]` | All resource conflicts |
| `count_group_violations()` | `int` | Total group violation count |
| `count_instructor_violations()` | `int` | Total instructor violation count |
| `count_room_violations()` | `int` | Total room violation count |
| `course_for_gene(gene)` | `Course` | Lookup course |
| `instructor_for_gene(gene)` | `Instructor` | Lookup instructor |
| `room_for_gene(gene)` | `Room` | Lookup room |
| `groups_for_gene(gene)` | `list[Group]` | Lookup groups |
| `from_individual(individual, ctx, qts)` | `Timetable` | Classmethod factory |

### `ConflictPair` — `src.domain.timetable`

`@dataclass(frozen=True, slots=True)` — a pair of conflicting genes.

| Attribute | Type |
|-----------|------|
| `gene_a_idx` | `int` |
| `gene_b_idx` | `int` |
| `resource_type` | `str` |
| `resource_id` | `str` |
| `quantum` | `int` |

### `Individual` — `src.domain.types`

Type alias: `Individual = list[SessionGene]`

### `SchedulingContext` — `src.domain.types`

`@dataclass` — complete problem instance.

| Attribute | Type | Default |
|-----------|------|---------|
| `courses` | `dict[tuple[str, str], Course]` | required |
| `groups` | `dict[str, Group]` | required |
| `instructors` | `dict[str, Instructor]` | required |
| `rooms` | `dict[str, Room]` | required |
| `available_quanta` | `list[int]` | required |
| `config` | `Any \| None` | `None` |
| `cohort_pairs` | `list[tuple[str, str]] \| None` | `None` |
| `family_map` | `dict[str, set[str]]` | `{}` |

| Method | Returns | Description |
|--------|---------|-------------|
| `validate()` | `list[str]` | List of validation errors |

---

## `src.constraints`

**Exports:** All 14 constraint classes, `Evaluator`, `ALL_CONSTRAINTS`,
`HARD_CONSTRAINT_CLASSES`, `SOFT_CONSTRAINT_CLASSES`, `build_constraints`

### `Constraint` Protocol — `src.constraints.constraints`

```python
class Constraint(Protocol):
    name: str
    weight: float
    kind: str  # "hard" or "soft"
    def evaluate(self, tt: Timetable) -> float: ...
```

### Hard Constraints — `src.constraints.hard`

| Class | Code | Module | Description |
|-------|------|--------|-------------|
| `StudentGroupExclusivity` | CTE | `cte.py` | No group in two places simultaneously |
| `InstructorExclusivity` | FTE | `fte.py` | No instructor double-booked |
| `RoomExclusivity` | SRE | `sre.py` | No room double-booked |
| `InstructorQualifications` | FPC | `fpc.py` | Instructor qualified for course |
| `RoomSuitability` | FFC | `ffc.py` | Room type matches course type |
| `InstructorTimeAvailability` | FCA | `fca.py` | Part-time instructor availability |
| `CourseCompleteness` | CQF | `cqf.py` | Correct total quanta per course/group |
| `SiblingSameDay` | ICTD | `ictd.py` | Sub-sessions not on same day |

All have signature: `evaluate(self, tt: Timetable) -> float`

### Soft Constraints — `src.constraints.soft`

| Class | Code | Module | Constructor extras |
|-------|------|--------|--------------------|
| `StudentScheduleCompactness` | CSC | `csc.py` | `gap_penalty_per_quantum` |
| `InstructorScheduleCompactness` | FSC | `fsc.py` | `gap_penalty_per_quantum` |
| `StudentLunchBreak` | MIP | `mip.py` | `break_min_quanta`, `penalty_per_missing_quantum` |
| `SessionContinuity` | SC | `session_continuity.py` | `isolated_slot_penalty`, `preferred_block_sizes` |
| `PairedCohortPracticalAlignment` | SSCP | `sscp.py` | — |
| `BreakPlacementCompliance` | BPC | `break_placement.py` | `break_min_quanta` |

### `Evaluator` — `src.constraints.evaluator`

| Method | Signature | Returns |
|--------|-----------|---------|
| `__init__` | `(constraints: list[Constraint] \| None)` | — |
| `fitness` | `(genes, context, qts)` | `tuple[float, float]` (hard, soft) |
| `fitness_from_timetable` | `(tt: Timetable)` | `tuple[float, float]` |
| `breakdown` | `(genes, context, qts)` | `dict[str, float]` |
| `breakdown_from_timetable` | `(tt)` | `dict[str, float]` |
| `hard_breakdown` | `(tt)` | `dict[str, float]` |
| `soft_breakdown` | `(tt)` | `dict[str, float]` |
| `evaluate_all` | `(tt)` | `tuple[float, float, dict, dict]` |

### Factory

`build_constraints(hard_weight, soft_weight, ...) -> list[Constraint]`

---

## `src.io`

**Exports:** `DataStore`, `QuantumTimeSystem`, `check_feasibility`, `decode_individual`,
`derive_cohort_pairs_from_groups`, `encode_availability`, `generate_feasibility_report_file`,
`link_courses_and_groups`, `link_courses_and_instructors`, `load_courses`, `load_groups`,
`load_instructors`, `load_rooms`, `validate_input`

### `DataStore` — `src.io.data_store`

`@dataclass` — loads and holds all input data.

| Attribute | Type |
|-----------|------|
| `courses` | `dict[tuple[str, str], Course]` |
| `groups` | `dict[str, Group]` |
| `instructors` | `dict[str, Instructor]` |
| `rooms` | `dict[str, Room]` |
| `qts` | `QuantumTimeSystem` |
| `cohort_pairs` | `list[tuple[str, str]]` |
| `feasibility_report` | `dict` |

| Method | Signature | Returns |
|--------|-----------|---------|
| `from_json` | `(data_dir, ...) -> DataStore` | Classmethod factory |
| `available_quanta` | property | `list[int]` |
| `to_context` | `() -> SchedulingContext` | Convert to context |
| `summary` | `() -> str` | Human-readable summary |
| `to_dict` / `from_dict` | `()` / `(d)` | Serialization |

### `QuantumTimeSystem` — `src.io.time_system`

`@dataclass` — maps continuous time to discrete quanta (default: 60-min blocks).

| Key Attribute | Type | Description |
|--------------|------|-------------|
| `QUANTUM_MINUTES` | `int` | 60 (class constant) |
| `total_quanta` | `int` | Total quanta in a week |
| `day_quanta_offset` | `dict` | Day → starting quantum |
| `day_quanta_count` | `dict` | Day → quanta count |
| `operating_hours` | `dict` | Day → (start_time, end_time) |

| Method | Signature | Returns |
|--------|-----------|---------|
| `time_to_quanta` | `(day, time_str)` | `int` |
| `quanta_to_time` | `(quantum)` | `tuple[str, str]` |
| `get_all_operating_quanta` | `()` | `list[int]` |
| `is_operational` | `(quantum)` | `bool` |
| `quantum_to_day_and_within_day` | `(quantum)` | `tuple[str, int]` |
| `get_midday_break_quanta` | `(day)` | `set[int]` |

### Loader Functions — `src.io.loader`

| Function | Signature | Returns |
|----------|-----------|---------|
| `load_courses` | `(path)` | `dict[tuple[str,str], Course]` |
| `load_groups` | `(path)` | `dict[str, Group]` |
| `load_instructors` | `(path)` | `dict[str, Instructor]` |
| `load_rooms` | `(path)` | `dict[str, Room]` |
| `link_courses_and_groups` | `(courses, groups)` | `None` (mutates) |
| `link_courses_and_instructors` | `(courses, instructors)` | `None` (mutates) |
| `encode_availability` | `(raw, qts)` | `set[int]` |

### Other I/O Functions

| Function | Module | Description |
|----------|--------|-------------|
| `validate_input` | `validator.py` | Validate input data consistency |
| `check_feasibility` | `feasibility.py` | Pigeonhole feasibility checks |
| `generate_feasibility_report_file` | `feasibility.py` | Write feasibility report |
| `decode_individual` | `decoder.py` | Gene list → CourseSession list |
| `derive_cohort_pairs_from_groups` | `loader.py` | Extract cohort pairs |

---

## `src.io.export`

**Exports:** `export_everything`, `generate_violation_report`, 16 plot functions

| Function | Description |
|----------|-------------|
| `export_everything` | Full export pipeline (PDF + JSON + plots) |
| `generate_violation_report` | Constraint violation breakdown |
| `plot_pareto_front` | Pareto front visualisation |
| `plot_convergence` | Hard/soft convergence curves |
| `plot_diversity` | Population diversity over generations |
| `plot_hard_violations` | Per-constraint hard violation breakdown |
| `plot_soft_violations` | Per-constraint soft violation breakdown |
| `plot_hypervolume` | Hypervolume indicator progression |
| `plot_igd` | IGD progression |
| `plot_spacing` | Spacing metric progression |
| `plot_convergence_comparison` | Multi-run convergence overlay |
| `plot_memetic_analysis` | Memetic repair effectiveness |
| `plot_moea_quality` | Multi-objective quality dashboard |
| `plot_repair_statistics` | Repair operator statistics |
| `plot_stagnation` | Stagnation detection visualisation |
| `plot_constraint_correlation` | Constraint correlation matrix |

---

## `src.ga.core`

**Exports:** `PopulationFactory`, `evaluate`, `evaluate_detailed`, `quanta_list_to_contiguous`

### `PopulationFactory` — `src.ga.core.population_factory`

| Method | Signature | Returns |
|--------|-----------|---------|
| `__init__` | `(context, parallel=True)` | — |
| `random_individual` | `(conflict_aware=True)` | `list[SessionGene]` |
| `greedy_individual` | `()` | `list[SessionGene]` |
| `create_population` | `(n, strategy="smart")` | `list[list[SessionGene]]` |

Population strategies: `"smart"` (mixed), `"hybrid"` (greedy + random), `"random"`

### Evaluation Functions — `src.ga.core.evaluator`

| Function | Signature | Returns |
|----------|-----------|---------|
| `evaluate` | `(individual, context, qts)` | `tuple[float, float]` |
| `evaluate_detailed` | `(individual, context, qts)` | `dict[str, float]` |
| `quanta_list_to_contiguous` | `(quanta_list)` | `list[tuple[int, int]]` |

---

## `src.ga.operators`

**Exports:** `crossover_course_group_aware`, `mutate_gene`, `mutate_individual`

| Function | Signature | Description |
|----------|-----------|-------------|
| `crossover_course_group_aware` | `(parent1, parent2, context)` | Course-group-aware crossover |
| `mutate_gene` | `(gene, context, ...)` | Mutate a single gene |
| `mutate_individual` | `(individual, context, prob)` | Mutate entire individual |

---

## `src.ga.repair`

**Exports:** `RepairEngine`, `RepairPipeline`, `detect_violated_genes`,
`get_all_repair_operators`, `get_enabled_repair_operators`, `repair_individual`,
`repair_individual_selective`, `repair_individual_unified`, `repair_operator`, and more.

### `RepairPipeline` — `src.ga.repair.pipeline`

Unified repair orchestrator with selection policies.

| Method | Description |
|--------|-------------|
| `__init__(context, policy, ...)` | Create with round-robin or epsilon-greedy |
| `repair(individual)` | Run repair pipeline |
| `get_stats()` | Repair statistics |

### `RepairEngine` — `src.ga.repair.engine`

RL-ready repair engine with lexicographic scoring.

| Inner Class | Description |
|-------------|-------------|
| `MoveTimeOperator` | Shift gene to new time |
| `SwapRoomOperator` | Reassign room |
| `ReassignInstructorOperator` | Reassign instructor |

| Data Class | Key Fields |
|------------|------------|
| `RepairCandidate` | `gene_idx`, `new_start`, `new_room_id`, `new_instructor_id` |
| `RepairStepResult` | `applied`, `operator`, `delta_hard`, `delta_soft` |
| `RepairStats` | `steps`, `applied_steps`, `total_delta_hard`, `total_delta_soft` |
| `ViolationState` | `hard`, `soft`, `gene_scores`, occupancy counts |

### Registered Repair Operators — `src.ga.repair.basic`

Decorator-registered, priority-ordered:

| Function | Priority | Constraint |
|----------|----------|-----------|
| `repair_instructor_availability` | 1 | HC5 (FCA) |
| `repair_group_overlaps` | 2 | HC1 (CTE) |
| `repair_room_overlap_reassign` | 3 | HC8 (SRE) |
| `repair_room_conflicts` | 4 | HC8 (SRE) |
| `repair_instructor_conflicts` | 5 | HC2 (FTE) |
| `repair_instructor_qualifications` | 6 | HC3 (FPC) |
| `repair_room_type_mismatches` | 7 | HC4 (FFC) |

Plus soft-constraint repair functions:
`repair_paired_cohort_practicals`, `repair_student_compactness`,
`repair_instructor_compactness`, `repair_student_lunch_break`

### Selective Repair — `src.ga.repair.selective`

Violation-targeted variants (3–4× faster). Only repairs genes with detected violations.

| Function | Description |
|----------|-------------|
| `repair_individual_selective` | Main entry point |
| `repair_instructor_availability_selective` | Selective HC5 |
| `repair_group_overlaps_selective` | Selective HC1 |
| `repair_room_overlap_reassign_selective` | Selective HC8 |
| `repair_room_conflicts_selective` | Selective HC8 |
| `repair_instructor_conflicts_selective` | Selective HC2 |
| `repair_instructor_qualifications_selective` | Selective HC3 |
| `repair_room_type_mismatches_selective` | Selective HC4 |

### Specialised Repair

| Module | Functions |
|--------|-----------|
| `greedy.py` | `greedy_repair(individual, context, ...)` |
| `exhaustive.py` | `exhaustive_repair(individual, context, ...)` |
| `memetic.py` | `memetic_repair(individual, context, ...)` |
| `igls.py` | `igls_repair(individual, context, ...)` |
| `heuristic_repair.py` | `repair_with_heuristic`, `_greedy_assign`, `_local_search_repair` |
| `selective_heuristic.py` | `selective_repair(individual, context, ...)` |
| `break_repair.py` | `repair_break_placement(individual, context, qts)` |
| `group_clash_repair.py` | `repair_group_clashes(individual, context)` |
| `hierarchy.py` | `repair_group_overlaps_hierarchy`, `repair_with_cascade_check` |
| `parallel.py` | `parallel_repair_population(population, context, ...)` |
| `conflict_detection.py` | `find_hard_conflict_sessions`, `select_worst_conflicts` |

### Detection — `src.ga.repair.detector`

| Function | Returns | Description |
|----------|---------|-------------|
| `detect_violated_genes` | `dict[int, list[str]]` | Gene index → violation types |

### Registry — `src.ga.repair.wrappers`

| Symbol | Description |
|--------|-------------|
| `@repair_operator(name, description, priority)` | Registration decorator |
| `get_all_repair_operators()` | `dict` of all registered |
| `get_enabled_repair_operators()` | `dict` of enabled only |
| `get_repair_operator_function(name)` | Get callable by name |
| `get_repair_operator_metadata(name)` | Get metadata by name |
| `RepairOperatorMetadata` | Metadata dataclass |

---

## `src.ga.repair.cp`

**Exports:** `CPRepairPipeline`

| Class/Function | Module | Description |
|---------------|--------|-------------|
| `CPRepairPipeline` | `pipeline.py` | Orchestrates partition → global → cluster → merge |
| `CPRepairStats` | `pipeline.py` | CP repair statistics |
| `CPSATSolver` | `solver.py` | Google OR-Tools CP-SAT constraint solver |
| `CPSolveResult` | `solver.py` | Solver result container |
| `FrozenAssignment` | `solver.py` | Frozen gene assignment |
| `partition_genes` | `partitioner.py` | Graph-based gene partitioning |
| `ClusterPartition` | `partitioner.py` | Partition result |
| `apply_cp_results` | `merger.py` | Merge CP results back to chromosome |
| `audit_hard_violations` | `merger.py` | Post-merge violation audit |
| `select_consistent_frozen_genes` | `frozen_selector.py` | Select genes to freeze |

---

## `src.ga.repair.lns`

**Exports:** `lns_igls_repair`, `lns_repair`

| Function/Class | Module | Description |
|----------------|--------|-------------|
| `lns_igls_repair` | `operator.py` | LNS + IGLS combined repair |
| `LNSRepairStats` | `operator.py` | LNS statistics |
| `apply_lns_to_population` | `operator.py` | Population-level LNS |
| `should_trigger_lns_repair` | `operator.py` | Trigger heuristic |
| `lns_repair` | `repair.py` | Thin heuristic wrapper |
| `build_conflict_graph` | `diagnostics.py` | Gene conflict graph |
| `expand_neighborhood_bfs` | `diagnostics.py` | BFS neighbourhood expansion |

---

## `src.ga.metrics`

**Exports:** `ViolationHeatmap`, `average_pairwise_diversity`, `calculate_convergence_rate`,
`calculate_generational_distance`, `calculate_hypervolume`,
`calculate_inverted_generational_distance`, `calculate_spacing`,
`detect_stagnation`, `individual_distance`, `record_violations_to_heatmap`

| Function | Module | Returns |
|----------|--------|---------|
| `calculate_hypervolume` | `hypervolume.py` | `float` |
| `average_pairwise_diversity` | `diversity.py` | `float` |
| `calculate_convergence_rate` | `convergence.py` | `float` |
| `calculate_spacing` | `spacing.py` | `float` |
| `calculate_inverted_generational_distance` | `igd.py` | `float` |
| `calculate_generational_distance` | `igd.py` | `float` |
| `detect_stagnation` | `convergence.py` | `bool` |
| `individual_distance` | `diversity.py` | `float` |

---

## `src.pipeline`

Pymoo integration layer for vectorized evaluation and repair.

### `SchedulingProblem` — `src.pipeline.scheduling_problem`

`pymoo.core.problem.Problem` — 2 objectives (hard, soft), 8 inequality constraints.

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(pkl_path, ctx, qts)` | Load encoding spec and build lookups |
| `_evaluate` | `(x, out, *args, **kwargs)` | Vectorized batch evaluation |

Factory: `create_problem(pkl_path, ctx, qts, run_preflight) -> SchedulingProblem`

### `EncodingSpec` — `src.pipeline.encoding`

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_events` | `int` | Number of scheduling events |
| `n_vars` | `int` | `n_events × 3` |
| `allowed_instructors` | `list[list[int]]` | Per-event allowed instructor indices |
| `allowed_rooms` | `list[list[int]]` | Per-event allowed room indices |
| `allowed_starts` | `list[list[int]]` | Per-event allowed start quanta |

| Function | Signature | Description |
|----------|-----------|-------------|
| `encode` | `(genes: list[GeneAssignment]) -> ndarray` | Genes → chromosome |
| `decode` | `(x: ndarray) -> list[GeneAssignment]` | Chromosome → genes |
| `chromosome_views` | `(x) -> tuple[inst, room, time]` | Views into interleaved array |

### Repair Operators — `src.pipeline`

| Class | Module | Description |
|-------|--------|-------------|
| `SchedulingRepair` | `repair_operator.py` | Multi-stage: domain fix → conflict → deconfliction |
| `BitsetSchedulingRepair` | `repair_operator_bitset.py` | HPC repair via int16 count tensors + Numba JIT |

### Batch API — `src.pipeline.batch_api`

| Function | Signature | Description |
|----------|-----------|-------------|
| `eval_hard_batch` | `(X, ctx)` | Batch hard constraint evaluation |
| `eval_soft_batch` | `(X, ctx)` | Batch soft constraint evaluation |
| `repair_batch` | `(X, ctx, ...)` | Batch chromosome repair |
| `metrics_batch` | `(X, ctx)` | Batch quality metrics |

### Vectorized Evaluators

| Function | Module | Returns |
|----------|--------|---------|
| `fast_evaluate_hard_vectorized` | `fast_evaluator_vectorized.py` | `ndarray(pop_size, 8)` |
| `fast_evaluate_hard_batch` | `fast_evaluator_batch.py` | `ndarray(pop_size, 8)` |
| `fast_evaluate_hard_single` | `fast_evaluator_batch.py` | `ndarray(8,)` |

---

## `src.rl.gym_env`

**Exports:** `PymooHyperHeuristicEnv`

### `PymooHyperHeuristicEnv` — `src.rl.gym_env.pymoo_env`

`gymnasium.Env` — RL environment wrapping the pymoo GA pipeline.

**Observation space:** `Box(0.0, 1.0, shape=(39,), dtype=float32)`

| Indices | Count | Feature Group |
|---------|-------|---------------|
| 0–4 | 5 | Fitness stats (min, max, mean, std, ptp of hard penalty) |
| 5–7 | 3 | Constraint violation stats (mean, max, frac feasible) |
| 8–12 | 5 | Diversity metrics (pairwise distances) |
| 13–24 | 12 | Constraint breakdown (8 hard + 4 soft means) |
| 25–28 | 4 | Progress (gen ratio, stagnation, convergence, feasibility gain) |
| 29–38 | 10 | Action history (last 10 action IDs) |

**Action space:** `Discrete(6)` — see Architecture doc for action descriptions.

| Method | Signature | Returns |
|--------|-----------|---------|
| `reset` | `(seed, options)` | `(obs, info)` |
| `step` | `(action)` | `(obs, reward, terminated, truncated, info)` |
| `action_masks` | `()` | `ndarray[bool]` |

### `VectorizedStateEncoder` — `src.rl.gym_env.fast_state_encoder`

Extracts 39-D `[0,1]` observation from population state with zero per-individual Python loops.

---

## `src.rl.actions`

**Exports:** `VECTORIZED_ACTION_SPACE`

### Action Classes — `src.rl.actions.vectorized_ops`

| Class | Action ID | Strategy |
|-------|-----------|----------|
| `ConservativeRepair` | 0 | 10% elite, 2 passes |
| `AggressiveRepair` | 1 | 25% elite, 3 passes |
| `MemeticEliteRepair` | 2 | 15% elite, 4 passes |
| `SoftFocusRepair` | 3 | 8% elite, 2 passes + compact |
| `DestructiveConstructive` | 4 | Ruin worst 10%, rebuild 20% |
| `IntensifiedRepair` | 5 | 20% elite, 3 passes |

### `PostGenConfig` — `src.rl.actions.vectorized_ops`

`@dataclass(frozen=True)` — configuration applied after each GA generation.

| Field | Type | Description |
|-------|------|-------------|
| `elite_fraction` | `float` | Fraction of population to repair |
| `passes` | `int` | Repair iteration count |
| `stochastic_alternate` | `bool` | Alternate stochastic/deterministic |
| `ruin_fraction` | `float` | Fraction to ruin (destructive only) |
| `compact_soft` | `bool` | Apply time compaction |

---

## `src.rl.agents`

**Exports:** `RandomAgent`, `create_ppo_agent`, `create_dqn_agent`

| Symbol | Description |
|--------|-------------|
| `create_ppo_agent(env, **kwargs)` | Create MaskablePPO agent (sb3-contrib) |
| `create_dqn_agent(env, **kwargs)` | Create DQN agent (stable-baselines3) |
| `RandomAgent` | Baseline random action selection |

---

## `src.rl.training`

**Exports:** `RLTrainer`, `create_trainer`

### `RLTrainer` — `src.rl.training.trainer`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(env, agent_type, save_dir, ...)` | Create trainer |
| `train` | `(total_timesteps, ...)` | Run training loop |
| `evaluate` | `(n_episodes, ...)` | Deterministic evaluation |
| `save` | `(path)` | Save model |
| `load` | `(path)` | Load model |

Constructor params: `env`, `agent_type="ppo"`, `save_dir`, `tensorboard_log`,
`verbose`, `n_envs`, `use_subproc`, `device`, `debug_logging`, `**agent_kwargs`

### `RolloutProgressCallback` — `src.rl.training.trainer`

`BaseCallback` — logs timestep progress with timing during training.

---

## `src.experiments`

**Exports:** `BaseExperiment`, `GAExperiment`, `BaselineExperiment`,
`MemeticExperiment`, `AggressiveExperiment`, `AdaptiveExperiment`, `CPHybridExperiment`

### `BaseExperiment(ABC)` — `src.experiments.base`

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `(name, tag, seed, data_dir, output_dir, verbose)` | Setup |
| `run` | `()` | Execute with timing + logging |
| `_execute` | `() -> dict` | **Abstract** — implement experiment logic |

### `GAExperiment(BaseExperiment)` — `src.experiments.ga_experiment`

Base for all pymoo NSGA-II experiments.

| Concrete Class | GA Mode | Strategy |
|---------------|---------|----------|
| `BaselineExperiment` | 01 | Pure NSGA-II |
| `MemeticExperiment` | 02 | NSGA-II + bitset elite repair |
| `AggressiveExperiment` | 03 | 2× offspring, high mutation, full repair |
| `AdaptiveExperiment` | 04 | Stagnation-aware mutation escalation |
| `CPHybridExperiment` | 05 | NSGA-II + periodic CP-SAT polish |

---

## `src.utils`

**Exports:** `LogContext`, `LogStats`, `get_console`, `get_log_stats`,
`get_logger`, `log_call`, `log_duration`, `quick_setup`, `setup_unified_logging`

| Symbol | Module | Description |
|--------|--------|-------------|
| `get_logger(name)` | `logging.py` | Get configured logger |
| `setup_unified_logging(level)` | `logging.py` | Configure root logging |
| `quick_setup(level)` | `logging.py` | One-line logging setup |
| `get_console()` | `console.py` | Rich console singleton |
| `log_call` | `logging.py` | Decorator — log function calls |
| `log_duration` | `logging.py` | Context manager — log timing |
| `LogContext` | `logging.py` | Structured log context |
| `LogStats` | `logging.py` | Aggregate statistics tracker |
| `get_log_stats()` | `logging.py` | Global stats singleton |
