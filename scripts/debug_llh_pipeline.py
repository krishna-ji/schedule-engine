#!/usr/bin/env python3
"""Phase 55: Diagnostic script to find why LLH repair produces ~1374 hard.

Runs 4 comparisons:
1. Direct repair_batch on random population (no Pymoo)
2. PymooVectorizedRepair._do (working reference from ga_02_memetic)
3. ConservativeRepair._do (LLH wrapper)
4. Full env step() with ConservativeRepair

If #1 works but #3 doesn't → bug in _AtomicRepairBase._do
If #1 also fails → bug in VectorizedRepair itself
If #1-3 work but #4 doesn't → bug in how env.step() uses repair
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np


def main():
    from src.pipeline.pymoo_operators import RandomDomainSampling
    from src.pipeline.repair_operator_vectorized import VectorizedRepair
    from src.pipeline.scheduling_problem import SchedulingProblem, create_problem

    pkl_path = ".cache/events_with_domains.pkl"

    # ── Create problem & random population ──────────────────────────
    print("=" * 70)
    print("PHASE 55: LLH Pipeline Diagnostic")
    print("=" * 70)

    problem = create_problem(pkl_path)
    spec = problem.spec

    # Generate random population using domain sampling
    sampler = RandomDomainSampling(pkl_path)
    X_random = sampler._do(problem, 20)  # 20 individuals
    print(f"\nRandom population: shape={X_random.shape}, dtype={X_random.dtype}")

    # Evaluate initial hard violations
    F_init = np.zeros((20, 2))
    G_init = np.zeros((20, 8))
    out = {}
    problem._evaluate(X_random, out)
    F_init = out["F"]
    G_init = out["G"]
    print(f"Initial best hard: {F_init[:, 0].min():.0f}")
    print(f"Initial mean hard: {F_init[:, 0].mean():.0f}")

    # ── Test 1: Direct repair_batch ─────────────────────────────────
    print("\n" + "─" * 70)
    print("TEST 1: Direct VectorizedRepair.repair_batch(passes=3)")
    print("─" * 70)

    engine = VectorizedRepair(pkl_path)
    X1 = engine.repair_batch(X_random.copy(), passes=3)
    print(f"  repair changed X: {not np.array_equal(X1, X_random)}")
    print(f"  delta (L1): {np.sum(np.abs(X1.astype(int) - X_random.astype(int)))}")

    out1 = {}
    problem._evaluate(X1, out1)
    F1 = out1["F"]
    print(
        f"  After repair: best_hard={F1[:, 0].min():.0f}, mean_hard={F1[:, 0].mean():.0f}"
    )

    # ── Test 1b: Direct repair_batch passes=7 ──────────────────────
    print("\n" + "─" * 70)
    print("TEST 1b: Direct VectorizedRepair.repair_batch(passes=7)")
    print("─" * 70)

    X1b = engine.repair_batch(X_random.copy(), passes=7)
    out1b = {}
    problem._evaluate(X1b, out1b)
    F1b = out1b["F"]
    print(
        f"  After repair: best_hard={F1b[:, 0].min():.0f}, mean_hard={F1b[:, 0].mean():.0f}"
    )

    # ── Test 2: PymooVectorizedRepair._do ───────────────────────────
    print("\n" + "─" * 70)
    print("TEST 2: PymooVectorizedRepair._do(problem, X) [working reference]")
    print("─" * 70)

    from src.pipeline.repair_operator_vectorized import PymooVectorizedRepair

    pymoo_repair = PymooVectorizedRepair(pkl_path, passes=7)
    X2 = pymoo_repair._do(problem, X_random.copy())
    print(f"  Return type: {type(X2)}, shape: {X2.shape}, dtype: {X2.dtype}")
    print(f"  repair changed X: {not np.array_equal(X2, X_random)}")

    out2 = {}
    problem._evaluate(X2, out2)
    F2 = out2["F"]
    print(
        f"  After repair: best_hard={F2[:, 0].min():.0f}, mean_hard={F2[:, 0].mean():.0f}"
    )

    # ── Test 3: ConservativeRepair._do ──────────────────────────────
    print("\n" + "─" * 70)
    print("TEST 3: ConservativeRepair._do(problem, X) [LLH wrapper]")
    print("─" * 70)

    from src.rl.actions.vectorized_ops import AggressiveRepair, ConservativeRepair

    conservative = ConservativeRepair(pkl_path)
    X3 = conservative._do(problem, X_random.copy())
    print(f"  Return type: {type(X3)}, shape: {X3.shape}, dtype: {X3.dtype}")
    print(f"  repair changed X: {not np.array_equal(X3, X_random)}")

    out3 = {}
    problem._evaluate(X3, out3)
    F3 = out3["F"]
    print(
        f"  After repair: best_hard={F3[:, 0].min():.0f}, mean_hard={F3[:, 0].mean():.0f}"
    )

    # ── Test 3b: AggressiveRepair._do ──────────────────────────────
    print("\n" + "─" * 70)
    print("TEST 3b: AggressiveRepair._do(problem, X) [LLH wrapper]")
    print("─" * 70)

    aggressive = AggressiveRepair(pkl_path)
    X3b = aggressive._do(problem, X_random.copy())

    out3b = {}
    problem._evaluate(X3b, out3b)
    F3b = out3b["F"]
    print(
        f"  After repair: best_hard={F3b[:, 0].min():.0f}, mean_hard={F3b[:, 0].mean():.0f}"
    )

    # ── Test 4: Check what Pymoo's Repair.do() actually does ─────
    print("\n" + "─" * 70)
    print("TEST 4: Pymoo's Repair.do() on Population object")
    print("─" * 70)

    from pymoo.core.population import Population

    # Create a Population object (like pymoo does internally)
    pop = Population.new("X", X_random.copy())

    # Use PymooVectorizedRepair.do() (the public method, not _do)
    repaired_pop = pymoo_repair.do(problem, pop)
    X4a = repaired_pop.get("X")
    print(f"  PymooVectorizedRepair.do(): type={type(X4a)}, shape={X4a.shape}")

    out4a = {}
    problem._evaluate(X4a, out4a)
    F4a = out4a["F"]
    print(
        f"  After repair: best_hard={F4a[:, 0].min():.0f}, mean_hard={F4a[:, 0].mean():.0f}"
    )

    # Now do the same with ConservativeRepair.do()
    pop_b = Population.new("X", X_random.copy())
    repaired_pop_b = conservative.do(problem, pop_b)
    X4b = repaired_pop_b.get("X")
    print(f"  ConservativeRepair.do():   type={type(X4b)}, shape={X4b.shape}")

    out4b = {}
    problem._evaluate(X4b, out4b)
    F4b = out4b["F"]
    print(
        f"  After repair: best_hard={F4b[:, 0].min():.0f}, mean_hard={F4b[:, 0].mean():.0f}"
    )

    # ── Test 5: Full env simulation (3 gens) ─────────────────────
    print("\n" + "─" * 70)
    print("TEST 5: Full PymooHyperHeuristicEnv — 10 gens, action=0 (Conservative)")
    print("─" * 70)

    from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

    env = PymooHyperHeuristicEnv(
        pkl_path=pkl_path,
        max_generations=50,
        pop_size=120,
        algorithm_name="nsga2",
        seed=42,
    )
    obs, info = env.reset()
    print(f"  After reset (gen 1): best_hard={info['best_hard']:.0f}")

    for g in range(10):
        obs, reward, done, trunc, info = env.step(0)  # Conservative
        print(
            f"  Gen {g + 2:2d}: best_hard={info['best_hard']:.0f}  "
            f"mean_hard={info['mean_hard']:.0f}  reward={reward:.4f}"
        )

    # ── Test 6: Full env with PymooVectorizedRepair instead ──────
    print("\n" + "─" * 70)
    print("TEST 6: Env with PymooVectorizedRepair (7 passes) injected manually")
    print("─" * 70)

    env2 = PymooHyperHeuristicEnv(
        pkl_path=pkl_path,
        max_generations=50,
        pop_size=120,
        algorithm_name="nsga2",
        seed=42,
    )
    obs2, info2 = env2.reset()
    print(f"  After reset (gen 1): best_hard={info2['best_hard']:.0f}")

    # Inject the working repair manually
    working_repair = PymooVectorizedRepair(pkl_path, passes=7)
    for g in range(10):
        env2._algorithm.mating.repair = working_repair
        env2._algorithm.next()
        env2._gen += 1
        pop = env2._algorithm.pop
        F, G, X = env2._extract_pop(pop)
        best_hard = float(F[:, 0].min())
        mean_hard = float(F[:, 0].mean())
        print(f"  Gen {g + 2:2d}: best_hard={best_hard:.0f}  mean_hard={mean_hard:.0f}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Initial random:           best_hard = {F_init[:, 0].min():.0f}")
    print(f"  Direct repair_batch(3):   best_hard = {F1[:, 0].min():.0f}")
    print(f"  Direct repair_batch(7):   best_hard = {F1b[:, 0].min():.0f}")
    print(f"  PymooVectorizedRepair._do: best_hard = {F2[:, 0].min():.0f}")
    print(f"  ConservativeRepair._do:    best_hard = {F3[:, 0].min():.0f}")
    print(f"  AggressiveRepair._do:      best_hard = {F3b[:, 0].min():.0f}")
    print(f"  Pymoo Repair.do() (Pymoo): best_hard = {F4a[:, 0].min():.0f}")
    print(f"  Pymoo Repair.do() (LLH):   best_hard = {F4b[:, 0].min():.0f}")
    print(f"  Env 10 gens (LLH):         best_hard = {info['best_hard']:.0f}")
    print(f"  Env 10 gens (PymooRepair):  best_hard = {best_hard:.0f}")


if __name__ == "__main__":
    main()
