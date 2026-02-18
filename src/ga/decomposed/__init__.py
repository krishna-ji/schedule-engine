"""Decomposed GA + CP-SAT Hybrid Scheduling.

This package implements a **supergroup-decomposed** approach:

1. **Cluster Detection** — Programmes sharing courses/instructors are grouped
   into independent clusters (ARCH, CIVIL, IT, MECH, MASTERS).

2. **Per-Cluster GA** — Each cluster runs its own small GA population.
   Smaller search space → faster convergence.

3. **CP-SAT Polish** — After each GA generation, a CP-SAT solver fixes
   hard constraint violations within each cluster.

4. **Global Coordination** — Bridge genes (cross-cluster shared resources)
   are solved first, then frozen for cluster-level solves.

5. **Merge** — Cluster solutions are combined into a full schedule.

Usage::

    from src.ga.decomposed import DecomposedScheduler

    scheduler = DecomposedScheduler(ctx)
    best_schedule = scheduler.run()
"""

from src.ga.decomposed.coordinator import DecomposedScheduler

__all__ = ["DecomposedScheduler"]
