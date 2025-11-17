# Constraint Checking Performance & Complexity Analysis


---

## 1) Short Prompt (StackOverflow/Quick Ask)

Title: "Algorithmic complexity analysis for constraint checking in DEAP timetabling engine"

Body:
> I maintain a DEAP timetabling engine (Python) where constraint violation checking seems very slow. The checker is centralized and registry-based, but it uses nested loops and is very slow. Typical sizes: population ≈ 50–200, genes ≈ 50–300. What practical algorithm/data structure changes or profiling strategies can I use to reduce time complexity for constraint/violation checks? I'm targeting real-time per-step evaluation for RL (fast per-step). Key constraints: instructor availability, room conflicts, timeslot overlaps, and capacity. Code is in `src/constraints` and uses list-based checks. Any recommendations for incremental evaluation, caching, suitable data structures (maps, bitsets), or existing library patterns would be appreciated.

Tags: python, complexity, optimization, timetabling, genetic-algorithms, profiling

---

## 2) Detailed Prompt ( Full Code Review)

Title:
> Request: Full algorithmic complexity analysis and optimization plan for constraint violation checking (`src/constraints`)

Body:
> **Context**:
> - Project: University timetabling engine (NSGA-II + RL)
> - Problem: Constraint evaluation is expensive; RL training slowed to 1 it/s.
> - Code areas: `src/constraints`, `src/rl/gym_env/schedule_env.py`, `src/core/ga_scheduler.py`.
> - Typical sizes:
>   - Population 50–200
>   - Genes/chromosome: 50–300
>   - Timeslots: 40–80
>
> **Request**: Provide a full algorithmic complexity analysis and actionable optimization plan. Specifically:
> 1. Complexity (Big-O) of primary functions, especially `evaluate`, `evaluate_detailed`, and any constraint functions in `src/constraints`.
> 2. Identify loops/nested loops causing O(n^k) behavior and quantify k (e.g., O(n^3), O(n^4)).
> 3. Suggest exact refactors and data-structures to reduce time complexity. (Delta/incremental evaluation, per-resource maps, bitsets, interval trees, segment trees, conflict maps, etc.)
> 4. Provide a code sketch implementing delta evaluation for one expensive constraint (e.g., instructors or timeslot overlap) and complexity analysis after the fix.
> 5. Suggest microbenchmark and profiling steps to reproduce the performance difference and show before/after metrics.
> 6. Provide a prioritized list of optimizations and an implementation risk/impact table.
>
> **Deliverables**:
> - Big-O per function and per constraint type (worst-case & typical case)
> - Sample Python code sketch of delta checks for a constraint
> - A benchmark script and profiling commands
> - Unit test outline for correctness checks on delta/refactor

Attachments: Link to full repository or minimal reproducer, and the slow function for direct analysis.

---

## 3) LLM / ChatGPT Prompt (Deep Analysis)

Use this to ask an LLM for a complete analysis:

> "Analyze the schedule-engine repo for algorithmic complexity and performance focusing on constraint checking in `src/constraints`, `ScheduleEnv.step()`, and the evaluation pipeline. Provide a Big-O analysis per function, identify all O(n^k) hotspots, propose 3 prioritized refactors (with code sketches), and provide a micro-benchmark harness and profiling commands (cProfile/pyinstrument/py-spy) to validate improvements. Include tests and complexity claims verification with sample data." 

---

## 4) Minimal Reproducible Example Request (Use in issue or ticket)

> "Please create a minimal script that:
> - Generates a synthetic `SchedulingContext` with N courses and M genes per individual.
> - Creates a population of size P.
> - Runs the current `evaluate` / `evaluate_detailed` function and records timings.
> - Then runs a proposed delta-check change or alternate method and records timings.
> - Provide ms per evaluation and growth (P and M scaling) figures and the difference in run-time." 

Provide a small code snippet example (pseudocode included below) and concrete values (e.g., pop_size 50, genes 100, timeslots 40).

Pseudocode Sample:
```python
import time
from src.encoder.quantum_time_system import QuantumTimeSystem
from src.ga.population import generate_course_group_aware_population
from src.ga.evaluator.fitness import evaluate as eval_full

# Setup context & pop
context = load_context(data_dir)
population = generate_course_group_aware_population(n=50, context=context, parallel=False)
for ind in population:
    ind.fitness.values = eval_full(ind, courses=context.courses, instructors=context.instructors, groups=context.groups, rooms=context.rooms)

# bench
start = time.perf_counter()
for ind in population:
    eval_full(ind, courses=context.courses, instructors=context.instructors, groups=context.groups, rooms=context.rooms)
elapsed = time.perf_counter() - start
print(f"Avg ms per evaluation: {elapsed*1000/len(population):.2f}")
```

---

## 5) Profiling & Benchmark Commands

Use these to gather data:

- cProfile
```bash
python -m cProfile -o profile.out scripts/bench_constraint_check.py
python -m pstats profile.out
```

- py-spy (sampling profiler)
```bash
py-spy top -- python scripts/bench_constraint_check.py --pop 100 --genes 200
```

- pyinstrument
```bash
pyinstrument scripts/bench_constraint_check.py
```

- line_profiler (if configured)
```bash
kernprof -l -v scripts/bench_constraint_check.py
```

Ask for:
- `pstats` dump or callgraph
- CPU hotspots & call counts
- Per-invocation time for handlers

---

## 6) Complexity Checklist (Ask the reviewer to answer these)
- [ ] Worst-case complexity of main functions (e.g., `evaluate`, `evaluate_detailed`)
- [ ] Complexity per constraint type (instructor, timeslot overlap, capacity)
- [ ] Complexity of replacing operations (mutation/crossover) - how many constraints affected per op
- [ ] Per-step RL complexity in `ScheduleEnv.step()` (sum of evaluation + other overheads)
- [ ] Proposed final per-step complexity after optimizations
- [ ] Provide before/after microbench metrics (ms/step)

---

## 7) Tags and Post Meta (for issues/StackOverflow)
`python`, `optimization`, `profiling`, `complexity`, `timetabling`, `genetic-algorithms`, `deap`, `stable-baselines3`.

---

## 8) Follow-ups I can do for you (choose any):
1. Add a `scripts/bench_constraint_check.py` microbenchmark script in the repo.
2. Implement a per-resource map delta-check for `instructor_availability` constraint, with unit tests.
3. Add profiling instrumentation and run benchmarks for `pop_size` in {10,50,100,200}.
4. Demonstrate a full refactor for `ScheduleEnv.step()` using incremental updates.

Tell me which of these you'd like me to implement next and I’ll plan the work and make changes accordingly.

---

*End of prompt library.*
