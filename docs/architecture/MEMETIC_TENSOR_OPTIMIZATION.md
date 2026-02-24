# Memetic Tensor Optimisation for University Timetabling

> *A vectorized NSGA-II repair architecture with population-level bincount
> occupancy detection and post-repair SSCP projection.*

---

## Abstract

This document formalises the algorithmic and systems-level design of a
memetic evolutionary timetabling engine that achieves **zero SSCP
(Symmetric Sub-Cohort Parallelism) violations from generation 1** while
maintaining **1.32 s/generation** throughput on a 120-individual,
790-event, 42-quantum instance.  The key innovations are:

1. A **population-level vectorized repair operator** that replaces
   $O(N \cdot E^2)$ per-individual Python loops with $O(N \cdot Q)$
   NumPy bincount histograms.
2. A **post-repair structural projection** that enforces paired
   practical synchronisation as an algebraic invariant, not as
   optimisation pressure.
3. A **density-aware soft evaluator** that computes CSC, FSC, and MIP
   penalties over 4-D occupancy tensors with zero Python loops.
4. A **pre-feasibility topology analyser** that proves structural
   (in)feasibility before the GA launches.

---

## 1. Constraint Topologies

### 1.1 Hard Constraints (Encoding)

Each individual is a flat chromosome $\mathbf{x} \in \mathbb{Z}^{3E}$
with interleaved triples $(i_e, r_e, t_e)$ for each event
$e \in \{0, \ldots, E-1\}$.

| Symbol | Domain | Semantics |
|--------|--------|-----------|
| $i_e$ | $\mathcal{D}_e^{\text{inst}}$ | Instructor index |
| $r_e$ | $\mathcal{D}_e^{\text{room}}$ | Room index |
| $t_e$ | $\mathcal{D}_e^{\text{time}}$ | Start quantum (0-indexed into $T = 42$) |

Hard constraints are partitioned into three families:

**HC-1: Resource exclusivity.**  No two events may share the same
(resource, quantum) cell:

$$
\forall\, q \in [t_e,\, t_e + d_e)\!:\quad
\text{rc}[r_e, q] \leq 1,\quad
\text{ic}[i_e, q] \leq 1,\quad
\text{gc}[g, q] \leq 1 \;\;\forall g \in G_e
$$

**HC-2: Availability.**  Assignments must respect boolean availability:

$$
\forall\, q \in [t_e,\, t_e + d_e)\!:\quad
\text{ia}[i_e, q] = \top,\quad
\text{ra}[r_e, q] = \top
$$

**HC-3: Domain membership.**  Every gene must lie within its domain:

$$
i_e \in \mathcal{D}_e^{\text{inst}},\quad
r_e \in \mathcal{D}_e^{\text{room}},\quad
t_e \in \mathcal{D}_e^{\text{time}}
$$

### 1.2 Soft Constraints (Objectives)

| Abbr. | Full Name | Formula |
|-------|-----------|---------|
| **CSC** | Cohort Schedule Contiguity | $\sum_{g,d} \rho_g \sum_{q=q_{\min}^{g,d}}^{q_{\max}^{g,d}} \mathbb{1}[\lnot\text{occ} \wedge q \notin \mathcal{B}]$ |
| **FSC** | Faculty Schedule Contiguity | Same as CSC over instructor dimension |
| **MIP** | Meridian Interval Preservation | $\sum_{g,d} \max(\ell_{\min} - \text{free}(\mathcal{W}),\, 0)$ |
| **SSCP** | Symmetric Sub-Cohort Parallelism | $\sum_{(A,B)} \max(\|\mathbf{P}_A \oplus \mathbf{P}_B\|_1 - |L_A - L_B|,\, 0)$ |

Where:
- $\mathcal{B} = \{2, 3, 4\}$ — floating lunch exclusion quanta
- $\mathcal{W} = \{2, 3, 4\}$ — lunch window (same set for this instance)
- $\rho_g = L_g / (D \cdot Q_d)$ — density ratio weighting
- $\mathbf{P}_{n,g} \in \{0,1\}^T$ — practical-event occupancy vector

---

## 2. Algorithmic Breakthroughs

### 2.1 Population-Level Bincount Occupancy

The central HPC innovation is replacing per-individual occupancy
dictionaries with **population-level linearised histogram keys**.

For room double-booking detection, every quantum of every event across
all $N$ individuals is mapped to a flat key:

$$
k_{n,e,\delta} = n \cdot (R \cdot T) + r_e \cdot T + (t_e + \delta)
\quad\text{where } \delta \in [0, d_e)
$$

A single `np.bincount(k)` call produces a histogram of size
$N \cdot R \cdot T$.  The histogram is **gathered** back at the same
keys to yield per-quantum conflict flags:

```python
room_cnt = np.bincount(room_keys, minlength=N * R * T)
room_conflict = (room_cnt[room_keys] > 1).astype(np.float64)
```

These flags are then aggregated back to per-event scores via a second
`np.bincount` on event-linearised keys.  The same pattern is applied
for instructor ($N \cdot I \cdot T$) and group ($N \cdot G \cdot T$)
dimensions.

**Expansion arrays** pre-compute the event→quantum mapping at
construction time:

| Array | Shape | Content |
|-------|-------|---------|
| `exp_event` | $(Q,)$ | Event index owning expanded quantum $q'$ |
| `exp_offset` | $(Q,)$ | Within-event offset $\delta$ |
| `grp_exp_event` | $(GQ,)$ | Event index (group-expanded) |
| `grp_exp_group` | $(GQ,)$ | Group index |

Where $Q = \sum_e d_e$ and $GQ = \sum_e d_e \cdot |G_e|$.

**Complexity**: $O(N \cdot Q)$ for room/instructor scoring,
$O(N \cdot GQ)$ for group scoring.  For the reference instance
($E = 790$, $Q \approx 1{,}550$, $GQ \approx 2{,}200$, $N = 120$),
the total scoring pass takes $\sim 3\text{ms}$.

### 2.2 CSC Density Ratio

Standard gap-based compactness treats all groups equally, but groups
with high load (many scheduled quanta) have structurally more gap
opportunities.  The density ratio:

$$
\rho_g = \frac{L_g}{D \cdot Q_d}
$$

where $L_g = \sum_{e : g \in G_e} d_e$ is the total quanta load of
group $g$, normalises the penalty so that a heavily-loaded group's gap
receives proportionally more weight.

Implementation: `np.bincount(grp_exp_group, minlength=n_groups)` computes
$L_g$ in $O(GQ)$, then the density scale is broadcast as a
$(1, G, 1)$ tensor multiplied element-wise with the $(N, G, D)$ gap
array.

### 2.3 MIP {2, 3, 4} Intersection

The Meridian Interval Preservation constraint protects the floating
lunch window at within-day quanta $\{2, 3, 4\}$ (12:00–15:00).

The MIP kernel exploits the fact that the lunch window and the break
exclusion set are **identical** ($\mathcal{W} = \mathcal{B}$) for this
instance, but keeps them as separate configuration to support
asymmetric break/lunch schemas.

Implementation: a boolean mask `lunch_mask[QPD]` is broadcast against
the 4-D occupancy tensor via NumPy broadcasting rules
$(1, 1, 1, Q_d)$ → $(N, G, D, Q_d)$, avoiding any explicit loop.

### 2.4 SSCP Tensor Projection

**The breakthrough**: SSCP = 0 is achieved not through evolutionary
pressure or even scoring, but through a **post-repair algebraic
projection** that runs on every individual of every generation.

For each pair $(a, b)$ with precomputed common domain
$\mathcal{T}_a \cap \mathcal{T}_b$:

1. **Detect** desynchronisation: `ta = time[:, pair_a]; tb = time[:, pair_b]; desync = (ta != tb)` — shape $(N, P)$.
2. **Project**: for each desynchronised $(n, p)$, sample $t \sim \text{Uniform}(\mathcal{T}_a \cap \mathcal{T}_b)$ and set $t_{n,a} = t_{n,b} = t$.
3. **Separate rooms**: if $r_{n,a} = r_{n,b}$, reassign $r_{n,b}$ to a different room from $\mathcal{D}_b^{\text{room}} \setminus \{r_{n,a}\}$.

Because this projection runs **after** every repair call and **before**
fitness evaluation, the invariant $t_a = t_b \wedge r_a \neq r_b$ is
structurally maintained.  Crossover and mutation may break it, but the
next generation's repair restores it deterministically.

**Convergence proof**: SSCP = 0 was achieved at generation 1 and
maintained through all 50 generations, with no SSCP regression
observed.

---

## 3. Dual Repair Architecture

The engine employs two complementary repair operators:

### 3.1 VectorizedRepair (Bulk, Every Generation)

| Property | Value |
|----------|-------|
| Granularity | Population-level ($N$ individuals simultaneously) |
| Strategy | Stochastic resample with 30% mutation mask |
| Occupancy | `np.bincount` on linearised keys |
| Pairs | Post-repair SSCP projection |
| Complexity | $O(\text{passes} \cdot N \cdot Q)$ |
| Runtime | ~1.3 s/gen for $N = 120$, $E = 790$ |

### 3.2 BitsetSchedulingRepair (Elite, Every 5th Generation)

| Property | Value |
|----------|-------|
| Granularity | Per-individual (top 15% elites) |
| Strategy | Greedy remove/re-place with vectorized cost matrices |
| Occupancy | 2-D `int16` count arrays (R×T, I×T, G×T) |
| Pairs | Simultaneous dual placement (`_find_paired_placement`) |
| Complexity | $O(E \cdot |\mathcal{I}| \cdot |\mathcal{T}| \cdot d_{\max})$ per individual |
| Soft proxy | Lunch, compactness, paired-cohort magnet (sub-integer) |

The dual architecture exploits the **speed/quality tradeoff**: the
vectorized operator provides fast population-wide constraint reduction,
while the bitset operator performs deep local search on elite
individuals to escape local minima.

The memetic loop (in `MemeticExperiment`):

```
for gen in range(NGEN):
    X = crossover_mutation(X)
    X = VectorizedRepair.repair_batch(X, passes=7)  # bulk
    F = evaluate(X)
    if gen % 5 == 0:
        elite_idx = top_15%(F)
        X[elite_idx] = BitsetSchedulingRepair.repair(X[elite_idx])  # deep
```

---

## 4. HPC Performance Profile

### 4.1 Memory Layout

| Tensor | Shape | Dtype | Bytes |
|--------|-------|-------|-------|
| `rc` (bitset) | $(R, T) = (45, 42)$ | int16 | 3,780 |
| `ic` (bitset) | $(I, T) = (70, 42)$ | int16 | 5,880 |
| `gc` (bitset) | $(G, T) = (92, 42)$ | int16 | 7,728 |
| **Total count arrays** | | | **17,388** (~17 KiB) |
| `inst_domains` (vec) | $(E, D_I^{\max})$ | int64 | variable |
| `room_domains` (vec) | $(E, D_R^{\max})$ | int64 | variable |
| `time_domains` (vec) | $(E, D_T^{\max})$ | int64 | variable |
| `exp_event` | $(Q,)$ | int32 | ~6,200 |
| `grp_exp_event` | $(GQ,)$ | int32 | ~8,800 |

The bitset operator's count arrays fit entirely in L1 cache (~32 KiB),
enabling stride-1 sequential access patterns with minimal cache misses.

### 4.2 Vectorization Techniques

| Technique | Where Used | Benefit |
|-----------|------------|---------|
| `np.bincount` on linearised keys | `_score_all_batch` | Eliminates per-individual Python loops |
| Padded domain matrices | `_fix_domains_vec` | Single `rng.random(K) * dom_len` call |
| Fancy indexing gather | `_find_placement` cost matrix | $(|\mathcal{T}|, |\mathcal{R}|)$ in one shot |
| Boolean broadcast masking | `eval_soft_vectorized` | 4-D tensor ops without explicit loops |
| Sentinel-based min/max | `occ_masked_{min,max}` | Axis-3 reduction replaces argwhere loops |

### 4.3 Benchmark Results

| Metric | Value |
|--------|-------|
| Population size | 120 |
| Generations | 50 |
| Events | 790 |
| Paired event tuples | 186 (93 pairs) |
| Total runtime | 65.8 s |
| Per-generation | 1.32 s |
| SSCP at Gen 1 | **0** |
| SSCP at Gen 50 | **0** |
| Best hard constraint | 67 (Gen 22) |
| Hard trajectory | 1466 → 78 → 67 → 70 → 67 → 76 |

Compared to the bitset-only bulk repair (naively wired): **250×
speedup** (1.32 s/gen vs 327 s/gen) with identical SSCP = 0 guarantee.

---

## 5. Pre-Feasibility Proof

The pre-feasibility reporter (`feasibility_reporter.py`) establishes
mathematical bounds on solvability before the GA begins.

### 5.1 Spatial Resource Equilibrium (SRE)

Events are grouped by their allowed-room set (room "feature class"):

$$
\mathcal{F}_k = \{e : \mathcal{R}_e = \text{frozenset}(k)\}
$$

For each class, supply and demand are computed:

$$
\text{supply}_k = \sum_{r \in k} |\text{avail}(r)|, \qquad
\text{demand}_k = \sum_{e \in \mathcal{F}_k} d_e
$$

If $\text{supply}_k < \text{demand}_k$, the deficit is
**mathematically irreducible** — no feasible schedule exists for those
events without relaxing room constraints.

### 5.2 Faculty Capacity Analysis (FCA)

For each instructor $i$, the assigned load across all qualified events
is:

$$
\text{load}_i = \sum_{e : i \in \mathcal{D}_e^{\text{inst}}} d_e
$$

If $\text{load}_i > |\text{avail}(i)|$, instructor $i$ is
**overloaded** — at least one event must be reassigned to a different
instructor.

### 5.3 SSCP Cascade Risk

For each cohort pair $(L, R)$, the net unique load (assuming maximal
practical alignment):

$$
L_{\text{net}} = L_{\text{total}}^L + L_{\text{total}}^R
  - \min(L_{\text{prac}}^L, L_{\text{prac}}^R)
$$

If $L_{\text{net}} > T$, the pair has **HIGH cascade risk** — their
combined scheduling obligations exceed the available time horizon,
and at least one constraint must be violated.

---

## 6. Conclusion

The combination of population-level bincount occupancy, post-repair
SSCP projection, and density-aware soft evaluation yields a
timetabling engine that:

- **Guarantees** SSCP = 0 as a structural invariant (not evolved).
- **Achieves** 1.32 s/gen throughput via zero-Python-loop NumPy kernels.
- **Proves** infeasibility pre-emptively via topological analysis.
- **Separates** bulk repair (vectorized, stochastic) from elite repair
  (greedy, exact) in a principled memetic architecture.
