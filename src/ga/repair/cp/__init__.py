"""CP-SAT Repair package: Constraint Programming based repair for GA individuals.

Architecture:
    1. **Partitioner** — decomposes a chromosome into cluster-scoped subproblems
       plus a set of cross-cluster "bridge" genes.
    2. **Global Phase** — solves the bridge subproblem first (foundation courses,
       cross-cluster instructors) using OR-Tools CP-SAT.
    3. **Cluster Phase** — for each cluster, solves an independent CP-SAT model
       in parallel, with bridge gene assignments frozen as constraints.
    4. **Merger** — reassembles the full chromosome and audits for residual
       cross-cluster conflicts.
    5. **Pipeline** — orchestrates the whole flow: partition → global → cluster → merge.

Usage::

    from src.ga.repair.cp import CPRepairPipeline

    pipeline = CPRepairPipeline()
    repaired = pipeline.repair(individual, context, family_map)
"""

from src.ga.repair.cp.pipeline import CPRepairPipeline

__all__ = ["CPRepairPipeline"]
