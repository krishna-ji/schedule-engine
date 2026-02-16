# GA Stagnation Root Cause Analysis

**Date**: 2025 | **Branch**: nb-migration  
**Problem**: GA stagnates at Hard≈880, Soft≈728 across all modes after 1000–2000 generations.  
**Theoretical minimum**: Hard=2, Soft≈0 (only I110's structural infeasibility prevents Hard=0).

---

## Executive Summary

The GA is **440× worse than the theoretical minimum** due to **7 interacting bugs**.  
The single most damaging bug is the "smart" initialization that piles **96% of all 549 genes onto quantum 0**, creating a starting fitness of Hard=3429 — which is **2.3× worse than pure random initialization** (Hard=1462). The remaining 6 bugs prevent the GA from recovering.

---

## Problem Dimensions

| Metric | Value |
|--------|-------|
| Courses | 159 (88 lecture, 71 practical) |
| Groups | 92 |
| Instructors | 189 (96 full-time, 93 part-time) |
| Rooms | 75 (52 lecture, 23 practical) |
| **Genes (chromosome size)** | **549** |
| Available quanta | 42 (6 days × 7 hours, Saturday closed) |
| Total quanta demand | 1144 |
| Average sessions per quantum (ideal) | 27.2 |
| Tightest group (BCE3x) | 41/42 quanta (97.6%) |
| Structurally infeasible instructor | I110: demand=8, available=6 |

---

## Bug Inventory

### BUG 1 (CRITICAL): "Smart" initialization piles 96% of genes at quantum 0

**File**: `src/schedule_engine/ga/core/population.py`, function `create_session_gene_with_conflict_avoidance()` (line 899)

**What happens**: The conflict-avoidance algorithm tracks a `used_quanta` set across all genes. With 549 genes competing for 42 quanta, the set fills up after ~20 genes. Once full:

1. `find_qualified_instructors_with_availability()` returns empty (all quanta "used")
2. Falls back to basic instructor selection — but `assigned_quanta` stays empty `[]`
3. `assign_conflict_free_quanta()` is called — can't find free consecutive blocks
4. Falls through to `_find_consecutive_block(available_quanta, block_size)` which scans sorted quanta [0,1,2,...,41] and returns the **first** consecutive block found — **always starting at quantum 0**

**Evidence**:
```
SMART INIT:  528/549 genes at start_quanta=0 (96.2%)  →  Hard=3429, Soft=2622
PURE RANDOM: 19/549 genes at start_quanta=0 (3.5%)    →  Hard=1462, Soft=1339
```

**Impact**: Smart init is **2.3× worse than pure random**. The entire GA starts handicapped.

---

### BUG 2 (HIGH): PopulationFactory.random_individual() ignores `conflict_aware` parameter

**File**: `src/schedule_engine/ga/core/population_factory.py`, line 62

```python
def random_individual(self, *, conflict_aware: bool = True) -> list[SessionGene]:
    # BUG: Always calls generate_course_group_aware_population regardless of conflict_aware!
    from schedule_engine.ga.core.population import generate_course_group_aware_population
    pop = generate_course_group_aware_population(n=1, context=self._context, parallel=False)
    return pop[0] if pop else []
```

The `conflict_aware` flag is **never used**. Even `strategy="random"` goes through `random_individual(conflict_aware=False)` → still calls the conflict-aware (bug-ridden) function.

---

### BUG 3 (HIGH): Mutation is not conflict-aware

**File**: `src/schedule_engine/ga/operators/mutation.py`, function `mutate_time_quanta()` (line 85)

The mutation picks a new time slot from `context.available_quanta` (a flat list of [0..41]) without any awareness of what's occupied by the gene's group, instructor, or room. With ~27 sessions per quantum on average, a random pick is **almost guaranteed to create a new conflict** while resolving the old one.

Additionally, `mutate_time_quanta()` has a **30% "keep current time" probability**. When the current time is quantum 0 (from bad init), this preserves the clustering.

**In `mutate_gene()`**: 
- 70% probability to keep current instructor (even if causing conflicts)
- 50% probability to keep current room (even if unsuitable)
- Net effect: mutations are too conservative to escape local optima

---

### BUG 4 (MEDIUM): Constraint-guided mutation room selection is blind

**File**: `src/schedule_engine/ga/operators/constraint_guided_mutation.py`, function `_mutate_session()` (line 190)

When mutating room (30% of the time):
```python
gene.room_id = random.choice(list(context.rooms.keys()))
```

This ignores room type (lecture vs practical), capacity requirements, and room availability. It assigns a random room from all 75, so ~30% of the time it picks the wrong type — immediately creating a new `room_suitability` hard violation.

---

### BUG 5 (MEDIUM): Baseline uses non-guided mutation

**File**: `src/schedule_engine/ga/run_helpers.py`, function `smart_mutation()` (line 597)

```python
(mutated,) = mutate_individual(individual, data.context, mut_prob=gene_mut_prob, guided=False)
```

The baseline always uses `guided=False`, meaning each gene has an independent 20% mutation chance. With 549 genes, ~110 are randomly mutated per individual — each creating random new conflicts without targeting existing violations. This is essentially random noise.

---

### BUG 6 (MEDIUM): Crossover cannot fix time clustering

**File**: `src/schedule_engine/ga/operators/crossover.py`, function `crossover_course_group_aware()`

The crossover swaps `instructor_id`, `room_id`, and `start_quanta` between parents. If both parents have most genes clustered at quantum 0 (from bad init), **swapping preserves the clustering** — Parent A's quantum 0 gene swaps with Parent B's quantum 0 gene, resulting in quantum 0 in both offspring.

Crossover only helps if parents have sufficiently **different** time slot assignments. With the init bug, the entire population starts in the same cluster, destroying crossover's effectiveness.

---

### BUG 7 (LOW): Structural infeasibility — instructor I110

**Data issue**: Instructor I110 is the sole qualified instructor for courses requiring 8 quanta, but I110 is part-time with only 6 available quanta. This creates a **minimum of 2 hard violations** (`instructor_time_availability`) that no algorithm can resolve.

---

## Observed Convergence (All Modes)

| Mode | Generations | Best Hard | Best Soft | Hard Breakdown (top 3) |
|------|------------|-----------|-----------|----------------------|
| Baseline | 2000 | 933 | 671 | group_excl=457, instr_avail=193, room_excl=178 |
| Memetic | 1000 | 898 | 797 | group_excl=435, instr_avail=177, room_excl=155 |
| Repair Sequential | 1000 | 881 | 728 | group_excl=441, instr_avail=179, room_excl=154 |
| Repair Bandit | 1000 | 881 | 728 | group_excl=441, instr_avail=179, room_excl=154 |

All modes converge to the same region because they share the same broken initialization and weak mutation operators.

---

## Why The Barrier Exists at ~880

The barrier is a **fitness plateau** caused by the interaction of all bugs:

1. **Init** creates solutions where ~27 sessions compete for each quantum slot (group_excl ≈ 500)
2. **Mutation** moves 1 gene at a time from one crowded slot to another crowded slot (net improvement ≈ 0)
3. **Crossover** swaps between equally-clustered parents (no improvement)
4. **NSGA-II selection** preserves the best-of-bad solutions, but the entire population is trapped in the same basin
5. Even **repair operators** can only make local improvements — they move individual genes to better slots, but can't perform the global redistributions needed to break the plateau

The fundamental issue: **resolving one conflict (e.g., moving gene X from quantum 5) creates new conflicts at the destination quantum** because the entire schedule is dense. The GA needs coordinated multi-gene moves, but only performs single-gene mutations.

---

## Root Cause Cascade

```
BUG 1 (bad init: 96% at q=0)
  └─→ Entire population starts at Hard≈3400
      └─→ BUG 6 (crossover can't fix clustering)
          └─→ Population stays homogeneous
              └─→ BUG 3 (blind mutation) + BUG 5 (non-guided)
                  └─→ Random single-gene changes → net zero improvement
                      └─→ GA converges to local optimum at Hard≈880
                          └─→ BUG 4 (blind room mutation) creates new violations
                              └─→ Oscillation, no further progress
```

---

## Fix Priority (ordered by impact)

### FIX 1: Switch default init_strategy to "random"

**Immediate impact**: Start from Hard≈1462 instead of Hard≈3429 (57% improvement).

In `base.py` line 120, change:
```python
init_strategy: str = "smart",  # → change to "random"
```

Or fix `PopulationFactory.random_individual()` to actually respect `conflict_aware=False`.

### FIX 2: Make mutation conflict-aware

In `mutate_time_quanta()`, instead of picking from the flat 42-quanta pool, check what's actually available for this gene's group/instructor/room:
```python
# Compute conflict-free quanta for this gene
occupied_by_group = {q for g in gene.group_ids for q2, glist in tt.group_occ if ...}
valid_quanta = [q for q in available_quanta if q not in occupied_for_this_gene]
```

### FIX 3: Make room mutation type-aware

In `_mutate_session()`, replace `random.choice(list(context.rooms.keys()))` with `find_suitable_rooms_for_course()`.

### FIX 4: Enable guided mutation in baseline

Change `smart_mutation()` to use `guided=True`:
```python
(mutated,) = mutate_individual(individual, data.context, mut_prob=gene_mut_prob, guided=True)
```

### FIX 5: Add multi-gene coordinated mutation

Introduce a "swap" operator that exchanges time slots between two conflicting genes simultaneously, reducing the net conflict count.

### FIX 6: Increase mutation pressure

With 549 genes and complex interdependencies, 20% per-gene mutation rate is too low for exploration. Consider adaptive mutation rates that increase when stagnation is detected.

### FIX 7: Address I110 data issue

Either add more available quanta for I110, or add a second qualified instructor for I110's courses.
