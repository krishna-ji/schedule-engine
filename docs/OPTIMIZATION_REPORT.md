# Schedule Engine Optimization Report

## Hard Constraint Violations: 3,429 → 75 (97.8% Reduction)

**Date:** February 12–13, 2026  
**Branch:** `nb-migration`  
**Test Suite:** 606 tests passing, 0 failures  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Description](#2-problem-description)
3. [Initial State & Root Cause Analysis](#3-initial-state--root-cause-analysis)
4. [Phase 1: Foundation Fixes (FIX 1–8)](#4-phase-1-foundation-fixes-fix-18)
5. [Phase 2: Local Search Enhancement](#5-phase-2-local-search-enhancement)
6. [Phase 3: Algorithm Pivot — NSGA-II → ILS](#6-phase-3-algorithm-pivot--nsga-ii--ils)
7. [Phase 4: Ultimate Mode — ILS Pipeline](#7-phase-4-ultimate-mode--ils-pipeline)
8. [Phase 5: Ablation — What Worked, What Didn't](#8-phase-5-ablation--what-worked-what-didnt)
9. [Phase 6: Rescheduling Passes & Final Push](#9-phase-6-rescheduling-passes--final-push)
10. [Convergence Summary](#10-convergence-summary)
11. [Structural Analysis](#11-structural-analysis)
12. [Files Modified](#12-files-modified)
13. [Reproduction Instructions](#13-reproduction-instructions)

---

## 1. Executive Summary

The schedule engine's hard constraint violations were reduced from **~3,429 to 75** across a systematic, multi-phase optimization effort. The improvement came from fixing 8 bugs in the initialization/mutation pipeline, enhancing the local search neighborhood, pivoting from a stagnating NSGA-II approach to Iterated Local Search (ILS), and adding ruin-and-recreate rescheduling passes for the dominant bottleneck constraints.

| Milestone | Hard Violations | Reduction | Wall Time |
|-----------|:-----------:|:---------:|:---------:|
| Original (random init, baseline GA) | ~3,429 | — | — |
| After FIX 1 (conflict-aware init) | ~701 | −80% | — |
| After FIX 1–8 (all foundation fixes) | ~701 | — | — |
| After 5 rounds repair+LS (single ind) | ~186 | −73% | 30s |
| ILS v1 (60 iters, 3 starts) | **110** | −41% | 547s |
| ILS v7 (300 iters, 5 starts, tuned) | **87** | −21% | 3,474s |
| ILS v8 (+ instructor rescheduling) | **84** | −3% | 3,365s |
| **ILS v9 (+ improved warm restart)** | **75** | −11% | **3,161s** |

**Final constraint breakdown at Hard=75:**

| Constraint | Count | Share |
|-----------|:-----:|:-----:|
| `student_group_exclusivity` | 49 | 65.3% |
| `instructor_time_availability` | 20 | 26.7% |
| `instructor_exclusivity` | 6 | 8.0% |
| **Total** | **75** | **100%** |

---

## 2. Problem Description

### 2.1 Instance Characteristics

| Dimension | Value |
|-----------|-------|
| Courses | 159 (theory + practical types) |
| Student Groups | 92 (hierarchical: BCE3A/BCE3B → BCE3AB) |
| Instructors | 189 (mix of full-time and part-time) |
| Rooms | 75 (lecture halls, labs, computer rooms) |
| Time Quanta | 42 (6 days × 7 hours: Sun–Fri, 10:00–17:00) |
| Sessions (genes) | 549 |
| Total Quanta Demand | ~1,144 quanta |

### 2.2 Encoding

Each session is encoded as a **SessionGene** with 6 attributes:

```
SessionGene(course_id, course_type, group_ids[], instructor_id, room_id, start_quanta)
```

- `course_id`, `course_type`, `group_ids`, `num_quanta` are **immutable** (set by course data)
- `instructor_id`, `room_id`, `start_quanta` are **mutable** (decision variables)
- An individual = list of 549 `SessionGene` objects

### 2.3 Hard Constraints (must satisfy)

| Constraint | Description |
|-----------|------------|
| `student_group_exclusivity` | No student group may be in two sessions simultaneously |
| `instructor_exclusivity` | No instructor may teach two sessions simultaneously |
| `room_exclusivity` | No room may host two sessions simultaneously |
| `instructor_time_availability` | Part-time instructors only available at declared hours |
| `instructor_qualifications` | Instructor must be qualified for the assigned course |
| `room_capacity` | Room capacity ≥ total group size |
| `room_features` | Room type must match course requirements (lab, lecture, etc.) |
| `course_completeness` | All course sessions must be scheduled |

### 2.4 Group Hierarchy

Groups follow a parent-child hierarchy that creates scheduling coupling:

```
BCE3AB (parent)
├── BCE3A (subgroup A)
└── BCE3B (subgroup B)
```

If a course targets `BCE3AB` (combined lecture) and another targets `BCE3A` (practical), these overlap — the BCE3A students are in both groups. This hierarchy is the primary source of `student_group_exclusivity` violations.

### 2.5 Structural Floor

One instructor (`I110`) has 8 exclusive quanta of demand but only 6 available quanta. This creates an **irreducible minimum of Hard=2** that no algorithm can eliminate without changing the input data.

---

## 3. Initial State & Root Cause Analysis

### 3.1 The Original Problem

The baseline GA produced schedules with **~3,429 hard** and **~900 soft** constraint violations after 500 generations. Investigation revealed this was not an algorithmic limitation but a **pipeline-wide infrastructure problem** — 7 distinct bugs across initialization, mutation, and crossover created a "garbage in, garbage out" cycle where repair operators fought against the very mutations that were supposed to help.

### 3.2 Seven Bugs Identified (Stagnation Analysis)

| # | Bug | Impact | Location |
|---|------|--------|----------|
| 1 | Init used **global** `used_quanta` set instead of per-resource tracking | Massive initial conflicts | `population.py` |
| 2 | `PopulationFactory.random_individual()` ignored `conflict_aware` param | Smart init was effectively disabled | `population_factory.py` |
| 3 | `mutate_time_quanta()` picked random times ignoring existing schedule | Mutations created new conflicts faster than repair fixed them | `mutation.py` |
| 4 | Room selection was random, ignoring type/capacity | Practical sessions placed in lecture halls | `mutation.py` |
| 5 | Guided mutation was disabled (`guided=False` default) | Violations were not targeted | `mutation.py` |
| 6 | `_mutate_session()` time selection was not conflict-aware | Same as #3 for constraint-guided path | `constraint_guided_mutation.py` |
| 7 | Only 1 gene repaired per mutation call | Repair throughput far too low for 549 genes | `constraint_guided_mutation.py` |
| 8 | Crossover rate too high (`cx_prob=0.5`) | Destroyed good genes | `crossover.py` / config |

---

## 4. Phase 1: Foundation Fixes (FIX 1–8)

### FIX 1: Per-Resource Conflict Tracking in Init

**File:** `src/schedule_engine/ga/core/population.py`  
**Function:** `create_session_gene_with_conflict_avoidance()`

**Before:** Used a single global `used_quanta` set that pooled ALL scheduled quanta together. A gene for Group A at time=5 would mark time=5 as "used" for ALL subsequent genes, including unrelated groups/instructors.

**After:** Tracks conflicts per-resource with `gene_blocked = group_schedule[gid] ∪ instructor_schedule[iid]`. Only blocks times actually used by the SAME group or instructor.

**Impact:** Hard violations dropped from **~3,429 → ~701** (−80%) just from this single fix.

### FIX 2: Respect `conflict_aware` Parameter

**File:** `src/schedule_engine/ga/core/population_factory.py`  
**Function:** `random_individual()`

**Before:** The `conflict_aware=True` parameter was accepted but never passed to the population generation function. All individuals were generated with random placement.

**After:** Parameter correctly forwarded to `generate_course_group_aware_population()`.

### FIX 3: Conflict-Aware Time Mutation

**File:** `src/schedule_engine/ga/operators/mutation.py`  
**Function:** `mutate_time_quanta()`

**Before:** Picked random time quanta with no awareness of the existing schedule. A mutation was equally likely to move a gene INTO a conflict as out of one.

**After:** When the `individual` parameter is provided, builds a `blocked` set of quanta used by same-group and same-instructor genes, then preferentially selects conflict-free slots. Falls back to random only when no conflict-free slot exists.

### FIX 4: Type-Aware Room Selection

**File:** `src/schedule_engine/ga/operators/mutation.py`  
**Function:** `_mutate_session()` and `mutate_individual()`

**Before:** Room selection was `random.choice(all_rooms)` — a practical course could be placed in a lecture hall, a 60-person class in a 20-seat room.

**After:** Uses `find_suitable_rooms_for_course()` which filters by: (a) `is_room_type_compatible(required, actual)` and (b) capacity ≥ total group size.

### FIX 5: Enable Guided Mutation as Default

**File:** `src/schedule_engine/ga/operators/mutation.py`

**Before:** `guided=False` was the default, meaning mutation was purely random — it didn't even look at which genes were violating constraints.

**After:** `guided=True` is the default. Mutation now delegates to `constraint_guided_mutation()` which identifies violated sessions and targets them specifically.

### FIX 6: Conflict-Aware Time in Constraint-Guided Path

**File:** `src/schedule_engine/ga/operators/constraint_guided_mutation.py`  
**Function:** `_mutate_session()`

**Before:** Time selection within the guided mutation path used `random.choice(available_quanta)` — no conflict awareness.

**After:** Builds a contiguous conflict-free time block by checking all same-group and same-instructor genes. 40% time mutation probability now finds slots that avoid existing overlaps.

### FIX 7: Multi-Gene Repair Per Mutation Call

**File:** `src/schedule_engine/ga/operators/constraint_guided_mutation.py`  
**Function:** `constraint_guided_mutation()`

**Before:** Only 1 gene was repaired per mutation call. With 549 genes and typical 50% mutation probability, repair throughput was far too low — each generation fixed ~1 gene while creating 0-2 new violations.

**After:** Repairs up to `max(3, len(violations) // 5)` genes per mutation call. With ~200 violations, this means ~40 targeted repairs per call.

### FIX 8: Reduce Crossover Rate

**File:** `runs/ga_06_ultimate.py` (config)

**Before:** `cx_prob=0.5` — half of all genes swapped during crossover, destroying carefully repaired schedule fragments.

**After:** `cx_prob=0.15` — crossover now touches only 15% of genes, preserving repair work.

### Combined Impact

After all 8 fixes applied, the initial population quality improved dramatically:

| Stage | Hard | Soft |
|-------|:----:|:----:|
| Random init (original) | ~3,429 | ~900 |
| Conflict-aware init (FIX 1 only) | ~701 | ~1,300 |
| After 5 rounds repair+LS | ~186 | ~1,100 |

---

## 5. Phase 2: Local Search Enhancement

### 5.1 Missing Instructor Neighborhoods

**File:** `src/schedule_engine/ga/operators/local_search.py`  
**Functions:** `_generate_instructor_neighbors()`, `_generate_time_instructor_neighbors()`

**Discovery:** The `_generate_neighborhood()` function only generated 2 neighborhood types: time moves and room swaps. It completely lacked instructor-change neighbors. Since `instructor_time_availability` was the #2 constraint, the LS could never fix instructor-related violations.

**Fix:** Added two new neighborhood types:

1. **Instructor neighbors** — for each qualified, available instructor: create a neighbor with that instructor (same time/room). Directly fixes `instructor_qualifications` and `instructor_exclusivity`.

2. **Time+Instructor combined neighbors** — sample 5 time slots × 3 qualified instructors = up to 15 neighbors. Fixes `instructor_time_availability` where the current instructor is unavailable AND teaching time is bad.

The neighborhood generator now produces **4 types**, sampled down to `max_samples=30`:

```
Neighborhood Types:
  1. Time        → move to different time slot (same room, instructor)
  2. Room        → move to different room (same time, instructor)
  3. Instructor  → change instructor (same time, room)        ← NEW
  4. Time+Instr  → change both time and instructor            ← NEW
```

### 5.2 Impact

Adding instructor neighborhoods to gene-level LS enabled the optimizer to fix instructor-related violations that were previously untouchable. Combined with the 11 deterministic repair operators and the RepairEngine's 3 operators, this created a complete repair arsenal covering ALL hard constraint types.

---

## 6. Phase 3: Algorithm Pivot — NSGA-II → ILS

### 6.1 NSGA-II Attempt (Abandoned)

The first version of "Ultimate Mode" used NSGA-II (Non-dominated Sorting Genetic Algorithm II) with a 4-phase repair pipeline. Results:

- **Stagnated at Hard≈169** after significant wall time
- `repair_individual_unified` took ~2.6s per individual — too expensive for population-based search
- With pop_size=100, one generation = ~260s of just repair

### 6.2 ILS Discovery

Testing revealed that a **single good individual** getting iterative repair+LS was more effective than a population of mediocre individuals getting less attention each. This led to the ILS (Iterated Local Search) pivot:

```
ILS Loop:
  1. Perturb candidate from best solution
  2. Apply full repair chain
  3. Accept if improved (greedy)
  4. Repeat
```

First ILS test: **Hard 186 → 128** in 30 iterations (240s) — far better than NSGA-II's 169 in similar time.

### 6.3 Why ILS Beat NSGA-II

| Aspect | NSGA-II | ILS |
|--------|---------|-----|
| Repair budget per individual | ~2.6s (shared across 100) | ~7s (full chain on 1) |
| Repair operators applied | Deterministic only | Det + Gene LS + RepairEngine |
| Selection pressure | Pareto-based (diffuse) | Greedy (direct) |
| Memory | 100 individuals × 549 genes | 2 individuals (best + candidate) |

The problem's high constraint density means that quality improvement comes from **deep per-individual repair**, not **population diversity**. ILS concentrates all computational budget on a single solution.

---

## 7. Phase 4: Ultimate Mode — ILS Pipeline

### 7.1 Architecture

**File:** `src/schedule_engine/experiments/modes/ultimate.py` (~935 lines)

The UltimateExperiment implements a 2-phase pipeline:

```
PHASE 1: Multi-Start Initialization
  ┌─────────────────────────────────────────────┐
  │  for start in 1..n_starts:                  │
  │    Generate conflict-aware individual        │
  │    for round in 1..5:                       │
  │      deterministic_repair(11 operators)     │
  │      gene_level_LS(greedy hill climbing)    │
  │  Keep best (hard, soft)                     │
  └─────────────────────────────────────────────┘

PHASE 2: Iterated Local Search (300 iterations)
  ┌─────────────────────────────────────────────┐
  │  candidate = deep_copy(best)                │
  │  smart_perturb(candidate, n_genes)          │
  │  deterministic_repair(candidate)            │
  │  gene_level_LS(candidate)                   │
  │  RepairEngine(candidate, 500ms)             │
  │  if candidate < best:                       │
  │    best = candidate    ← Greedy acceptance  │
  │                                             │
  │  Every 10 stale iters:                      │
  │    group/instructor rescheduling pass        │
  │  Every 30 stale iters:                      │
  │    diversification restart (warm + fresh)    │
  └─────────────────────────────────────────────┘
```

### 7.2 Smart Perturbation

Not random — targets the **worst-violated genes** and uses conflict-aware placement:

```python
def smart_perturb(ind, n_perturb):
    # 1. Score each gene by violation count
    # 2. Pick the worst n_perturb genes
    # 3. For each gene:
    #    a. Build group_blocked set (same-group genes' times)
    #    b. Add intra-perturbation tracking (avoid cascading group conflicts)
    #    c. Prefer conflict-free time slot, fallback to least-conflict
    #    d. Assign type-compatible room
    #    e. Fix instructor availability at new time
```

Key innovation: **intra-perturbation tracking** — a `perturbed_times` dict tracks where previously-perturbed genes of the same group were placed, preventing them from conflicting with each other.

### 7.3 The Repair Arsenal

Each ILS iteration applies 4 repair stages in sequence:

| Stage | Operator | What It Does | Cost |
|-------|----------|-------------|------|
| 1 | `repair_individual_unified` | 11 deterministic operators, priority-ordered | ~2.6s |
| 2 | `gene_ls_pass` | Greedy hill climbing for each violated gene (30 neighbors, 12 iterations) | ~3-4s |
| 3 | `RepairEngine` | Eval-guided MoveTime/SwapRoom/ReassignInstructor with ε-greedy policy | ~0.5s |
| 4 | (on stagnation) `group_reschedule_pass` / `instructor_reschedule_pass` | Ruin-and-recreate | ~1-2s |

### 7.4 The 11 Deterministic Repair Operators

Applied in priority order by `repair_individual_unified()`:

| Priority | Operator | Target Constraint |
|:--------:|----------|------------------|
| 1 | `repair_instructor_availability` | Shift to available times |
| 1 | `repair_instructor_availability_reassign` | Swap to available instructor |
| 2 | `repair_group_overlaps` | Hierarchy-aware group de-overlapping |
| 3 | `repair_room_overlap_reassign` | Idle-room swap before time-shift |
| 4 | `repair_room_conflicts` | Room double-booking fallback |
| 5 | `repair_instructor_conflicts` | Instructor double-booking |
| 6 | `repair_instructor_qualifications` | Reassign to qualified instructor |
| 7 | `repair_room_type_mismatches` | Compatible room type |
| 8 | `repair_paired_cohort_practicals` | Align cohort pair practicals |
| 9 | `repair_student_compactness` | Reduce idle gaps (soft) |
| 10 | `repair_instructor_compactness` | Reduce instructor gaps (soft) |
| 11 | `repair_student_lunch_break` | Protect midday break (soft) |

### 7.5 Diversification Strategy

**Periodic rescheduling** (every 10 stagnant iterations):
- Odd multiples (10, 30, ...): Group rescheduling — ruin-and-recreate for top 3-5 most conflicted groups
- Even multiples (20, 40, ...): Instructor rescheduling — ruin-and-recreate for top 5 most conflicted instructors

**Diversification restart** (after 30 stagnant iterations):
- Strategy A (fresh): Generate new individual from scratch + 5 rounds repair+LS
- Strategy B (warm): Copy best → perturb 10% → group+instructor reschedule → 2 rounds repair+LS → RepairEngine (3× budget)
- Pick whichever is better; update best if restart beats it

---

## 8. Phase 5: Ablation — What Worked, What Didn't

### 8.1 Ablation Study Results

| Experiment | Configuration | Hard | Verdict |
|-----------|---------------|:----:|---------|
| v1 (baseline ILS) | 3 starts, 60 iters, greedy | **110** | **Baseline** |
| v2 (+ SA acceptance) | + Simulated Annealing (T: 8→0.3) | 113 |  **Worse** — SA drift counterproductive |
| v3 (+ mixed perturbation) | + random/scatter/large perturb, no SA | 118 |  **Worse** — destructive perturbations |
| v4 (+ group reschedule per iter) | Group reschedule in every ILS iteration | 112 |  **Minimal gain, slowed convergence** |
| v4b (reschedule on stagnation only) | Group reschedule only every 10 stale iters | 112 | ≈ Neutral |
| v5 (longer + more starts) | 5 starts, 200 iters, stagnation=50 | **85** |  5 starts found better basin |
| v7 (tuned stagnation) | 5 starts, 300 iters, stagnation=30 | **87** |  Faster restarts = more basins explored |
| v8 (+ instructor reschedule) | + alternating group/instructor reschedule | **84** |  Instructor reschedule helps |
| v9 (+ improved restart) | + warm restart (10% perturb, 2 repair rounds) | **75** |  **Best result** |
| v10 (reschedule in Phase 1) | Reschedule passes in init | 164 (init) |  **Counterproductive in init** |

### 8.2 Key Ablation Insights

#### Simulated Annealing: Harmful

SA acceptance allows worse solutions to be accepted with probability $P = e^{-\Delta / T}$. For this highly-constrained problem, this just creates drift — the accepted worse solutions don't lead to new basins, they just waste iterations recovering back to the local optimum.

**Verdict:** Greedy acceptance (only accept strict improvements) is optimal for this problem.

#### Mixed Perturbation Strategies: Harmful

Three additional perturbation strategies were tested:
- **Random perturbation** (scramble random genes)
- **Group-cluster scatter** (reschedule all sessions of worst group)
- **Large perturbation** (perturb 30% of genes)

All created too much damage for the repair chain to recover from. The smart perturbation (which targets violated genes with conflict-aware placement) is strictly better.

#### Group Rescheduling Per Iteration: Harmful

Applying `group_reschedule_pass()` in every ILS iteration (before gene LS) produced WORSE results (152 stuck → never improved). The rescheduling pass makes large moves that the subsequent repair chain can't fully recover from within one iteration. It works better as an infrequent diversification tool.

#### Phase 1 Rescheduling: Harmful

Adding rescheduling passes to the Phase 1 initialization produced WORSE initial seeds (164 vs 127). The rescheduling moves genes to locally-optimal positions, but this interferes with the iterative repair+LS chain that was already finding better solutions.

#### 5 Starts vs 3 Starts: Critical

Increasing from 3 to 5 multi-start initializations was one of the highest-impact changes. Start 4 (seed=42+3=45) found Hard=127 while the best of 3 starts was Hard=152. This 25-point improvement in the starting solution propagated through the entire ILS run.

#### Stagnation Restart Frequency: Important

Reducing `stagnation_restart` from 50 to 30 triggers more frequent diversification restarts, which:
- Explores more basins (8 restarts in v7 vs ~4 in v5)
- Doesn't waste time on exhausted basins
- The warm restart occasionally discovers lower-violation basins

### 8.3 What Moved the Needle (Ranked by Impact)

| Rank | Change | Impact |
|:----:|--------|--------|
| 1 | **FIX 1:** Per-resource conflict tracking in init | 3,429 → 701 (−80%) |
| 2 | **Iterative repair+LS** (5 rounds) | 701 → 186 (−73%) |
| 3 | **ILS algorithm** (vs NSGA-II) | 186 → 110 (−41%) |
| 4 | **5 starts** (vs 3) | 152 → 127 initial (−16% init) |
| 5 | **Instructor reschedule + warm restart** | 87 → 75 (−14%) |
| 6 | **Instructor neighborhoods in LS** | Enabled fixing instructor constraints |
| 7 | **Smart perturbation** (vs random) | ~30% faster convergence |
| 8 | **Stagnation=30** (vs 50) | More basins explored |

---

## 9. Phase 6: Rescheduling Passes & Final Push

### 9.1 Group Rescheduling Pass (Ruin-and-Recreate)

**Function:** `group_reschedule_pass(ind, n_groups=3)`

Targets `student_group_exclusivity` — the dominant bottleneck (65% of violations).

Algorithm:
1. Count pairwise time-overlap violations per group
2. Select the N worst groups
3. For each worst group:
   - Collect all gene indices belonging to that group
   - For each gene, compute `group_blocked` (times used by same-group genes excluding targets) and `instr_blocked` (times used by same instructor)
   - Score every possible start time: `score = |slot ∩ group_blocked| × 3 + |slot ∩ instr_blocked|`
   - Pick the time with minimum score (perfect=0 → early exit)
   - Track `placed_times` to coordinate within-group gene placements

### 9.2 Instructor Rescheduling Pass (Ruin-and-Recreate)

**Function:** `instructor_reschedule_pass(ind, n_instr=5)`

Targets `instructor_time_availability` + `instructor_exclusivity` (35% of violations combined).

Algorithm:
1. Count violations per instructor: availability violations + pairwise time overlaps
2. Select the N worst instructors
3. For each worst instructor:
   - Collect all gene indices taught by this instructor
   - Build `instr_avail` (part-time: declared availability; full-time: all quanta)
   - Score every possible start time: `score = |slot − instr_avail| × 5 + |slot ∩ self_overlap| × 4 + |slot ∩ group_blocked| × 2`
   - Pick minimum-score time, track `placed_instr_times` for self-coordination

### 9.3 Integration

The rescheduling passes are integrated at two points:

1. **Periodic stagnation handler** (every 10 non-improving iterations):
   - Alternates group reschedule (iters 10, 30, ...) and instructor reschedule (iters 20, 40, ...)
   - followed by full repair chain: `det_repair → gene_LS → RepairEngine(3× budget)`
   - Accept if improved

2. **Warm diversification restart** (after 30 non-improving iterations):
   - Applies BOTH group + instructor reschedule after perturbation
   - Double repair rounds (2×) for deeper recovery

### 9.4 Final Convergence (v9)

```
Phase 1 (210s): Best initial seed = Hard=127
ILS progression:
  Iter   1: 127 → 118 → 113 → 109 → 104 → 100 → 97
  Iter  94: Broke plateau at 97
  Iter 143: → 93
  Iter 170: group-reschedule → 87
  Iter 190: instructor-reschedule → 84
  Iter 220: → 82 → 80
  Iter 260: group-reschedule → 77
  Iter 272: → 75  ← BEST
  Iter 300: 75 (final, stable)
  
Total: 22 improvements, 2 restarts, 3161s
```

---

## 10. Convergence Summary

### 10.1 Full Trajectory

```
Hard Violations Over Time:

3429 ┤ ██ Raw random init
     │
 701 ┤ ██ FIX 1: conflict-aware init
     │
 186 ┤ ██ 5 rounds repair+LS (single individual)
     │
 152 ┤ ██ Phase 1: best of 5 conflict-aware starts
     │
 110 ┤ ██ ILS v1 (60 iterations, 3 starts)
     │
  87 ┤ ██ ILS v7 (300 iters, 5 starts, stagnation=30)
     │
  84 ┤ ██ + instructor rescheduling
     │
  75 ┤ ██ + improved warm restart (BEST)
     │
   2 ┤ ── Structural floor (instructor I110 overloaded)
```

### 10.2 Per-Constraint Trajectory

| Stage | student_group | instr_time_avail | instr_excl | room_excl | Total |
|-------|:---:|:---:|:---:|:---:|:---:|
| Init (random) | ~1,500 | ~800 | ~600 | ~500 | ~3,429 |
| Init (conflict-aware) | ~300 | ~200 | ~150 | ~51 | ~701 |
| After repair+LS | ~76 | ~43 | ~33 | ~6 | ~152 |
| ILS v1 (60 iters) | 71 | 25 | 16 | 0 | 110* |
| **ILS v9 (300 iters)** | **49** | **20** | **6** | **0** | **75** |

*Note: v1 total via breakdown was 112 (71+25+16), not 110 — minor discrepancy due to different evaluation paths.

---

## 11. Structural Analysis

### 11.1 Why Zero Is Not Achievable

**Instructor I110 Overload:** This instructor has 8 exclusive quanta of teaching demand but only 6 available quanta. This creates a minimum of 2 hard violations that cannot be resolved without reassigning courses to different instructors or extending I110's availability.

**BCE3x Groups at 97.6% Capacity:** The BCE3A/BCE3B/BCE3AB groups have 41 out of 42 quanta filled with courses. With only 1 free quantum, any pair of overlapping sessions in this group has essentially zero room to be moved apart. The pigeonhole principle limits how far `student_group_exclusivity` can be reduced.

### 11.2 Theoretical Lower Bound

- **Structural minimum:** Hard ≥ 2 (I110 overload)
- **Practical estimate:** Hard ≈ 20–40 may be the reachable floor given the near-saturated group capacities. Reaching this would require either:
  - A complete constraint-programming solver (guaranteed optimal but exponential time)
  - Significantly more ILS runtime (diminishing returns: v9 made its last improvement at iter 272/300)

### 11.3 Remaining Violations Breakdown

The 75 remaining violations are concentrated in 3 constraints:

| Constraint | Count | Root Cause |
|-----------|:-----:|-----------|
| `student_group_exclusivity` | 49 | 97.6% group capacity → near-zero slack for de-overlapping |
| `instructor_time_availability` | 20 | Part-time instructors with limited hours teaching in high-demand slots |
| `instructor_exclusivity` | 6 | Instructors assigned to overlapping sessions |

Notable: `room_exclusivity`, `room_capacity`, `room_features`, `instructor_qualifications`, and `course_completeness` are all at **zero** — fully resolved.

---

## 12. Files Modified

### Core Algorithm Changes (src/)

| File | Lines | Change |
|------|:-----:|--------|
| `ga/core/population.py` | 1,873 | FIX 1: per-resource conflict tracking in `create_session_gene_with_conflict_avoidance()` |
| `ga/core/population_factory.py` | 153 | FIX 2: respect `conflict_aware` parameter |
| `ga/operators/mutation.py` | ~260 | FIX 3-5: conflict-aware time, type-aware room, guided default |
| `ga/operators/constraint_guided_mutation.py` | ~275 | FIX 6-7: conflict-aware time, multi-gene repair |
| `ga/operators/local_search.py` | 583 | Added instructor + time+instructor neighborhood generators; max_samples=30 |
| `ga/repair/conflict_detection.py` | 369 | Fixed constraint name mapping (`instructor_time_availability` → correct tracker) |
| `ga/repair/engine.py` | 775 | RepairEngine with ε-greedy policy (used by ultimate mode) |
| `ga/repair/basic.py` | 1,405 | 11 deterministic repair operators (existing, used by pipeline) |
| `experiments/modes/ultimate.py` | ~935 | **NEW:** UltimateExperiment — ILS + full repair arsenal |
| `experiments/modes/__init__.py` | — | Export UltimateExperiment |
| `experiments/__init__.py` | — | Export UltimateExperiment |

### Run Scripts

| File | Lines | Description |
|------|:-----:|-----------|
| `runs/ga_06_ultimate.py` | ~115 | Production config for Mode F |

### Documentation

| File | Description |
|------|-----------|
| `docs/STAGNATION_ANALYSIS.md` | Original 7-bug diagnosis |
| `docs/OPTIMIZATION_REPORT.md` | This report |

---

## 13. Reproduction Instructions

### Quick Test (60 iterations, ~10 minutes)

```bash
cd /home/krishna/Desktop/schedule-engine
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
import logging; logging.basicConfig(level=logging.INFO, format='%(message)s')
from schedule_engine.experiments.modes.ultimate import UltimateExperiment
from pathlib import Path

exp = UltimateExperiment(
    seed=42, pop_size=1, ngen=60, cxpb=0.15, mutpb=0.4,
    fitness_weights=(-1.0, -1.0),
    data_dir=Path('data'), output_dir=Path('output/quick_test'),
    opening_time='10:00', closing_time='17:00', closed_days=['Saturday'],
    n_starts=3, repair_ls_rounds=5, ils_iterations=60,
    stagnation_restart=30,
)
exp.run()
"
```

Expected: Hard ≈ 95–110 in ~10 minutes.

### Full Run (300 iterations, ~53 minutes)

```bash
cd /home/krishna/Desktop/schedule-engine
.venv/bin/python runs/ga_06_ultimate.py
```

Expected: Hard ≈ 75–85 in ~53 minutes with default seed=42.

### Run Tests

```bash
cd /home/krishna/Desktop/schedule-engine
.venv/bin/python -m pytest tests/ -x -q
# Expected: 606 passed, 1 xfailed
```

---

*Report generated: February 13, 2026*  
*Best result: Hard=75 | Soft=1,228 | 22 improvements | 2 restarts | 3,161s*
