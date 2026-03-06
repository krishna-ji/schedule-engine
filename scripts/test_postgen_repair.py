"""Test post-generation BitsetRepair on BEST survivors.

Mirrors the memetic GA callback strategy: after each algorithm.next(),
repair the top K% of the surviving population (lowest hard penalty)
using BitsetSchedulingRepair, then re-evaluate.
"""

import time

import numpy as np
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population

from src.pipeline.repair_operator_bitset import BitsetSchedulingRepair
from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

print("Test: Post-gen BitsetRepair on BEST 15% of SURVIVORS")
print("=" * 60)

env = PymooHyperHeuristicEnv(
    pkl_path=".cache/events_with_domains.pkl",
    max_generations=50,
    pop_size=120,
    algorithm_name="nsga2",
    seed=42,
)
obs, info = env.reset()
bh = info["best_hard"]
mh = info["mean_hard"]
print(f"Gen  1: best_hard={bh:.0f}  mean_hard={mh:.0f}")

repairer = BitsetSchedulingRepair(".cache/events_with_domains.pkl")
ELITE_FRAC = 0.15
REPAIR_PASSES = 4

t0 = time.perf_counter()
for g in range(25):
    # Use action 0 (Conservative) for mating — domain fix + small BitsetRepair
    obs, reward, done, trunc, info = env.step(0)

    # === POST-GENERATION: repair BEST survivors (like memetic GA) ===
    pop = env._algorithm.pop
    F = pop.get("F")
    hard_vals = F[:, 0]

    n_elite = max(1, int(len(pop) * ELITE_FRAC))
    elite_idx = np.argsort(hard_vals)[:n_elite]
    elite_idx = elite_idx[hard_vals[elite_idx] > 0]

    modified = []
    for idx in elite_idx:
        xi = pop[idx].get("X").copy()
        for p in range(REPAIR_PASSES):
            rng = np.random.default_rng() if p % 2 == 0 else None
            xi_new = repairer.repair(xi, rng=rng)
            if np.array_equal(xi_new, xi):
                break
            xi = xi_new
        pop[idx].set("X", xi)
        pop[idx].set("F", None)
        pop[idx].set("G", None)
        pop[idx].set("CV", None)
        for tag in ["F", "G", "CV"]:
            if tag in pop[idx].evaluated:
                pop[idx].evaluated.remove(tag)
        modified.append(pop[idx])

    if modified:
        eval_pop = Population.create(*modified)
        Evaluator().eval(env._problem, eval_pop)

    F2 = pop.get("F")
    bh = float(F2[:, 0].min())
    mh = float(F2[:, 0].mean())
    t = time.perf_counter() - t0
    n_rep = len(modified)
    print(
        f"Gen {g+2:2d}: best_hard={bh:.0f}  mean_hard={mh:.0f}  repaired={n_rep}  elapsed={t:.1f}s"
    )
    if bh < 100:
        print(f"  >>> MILESTONE: hard < 100 at gen {g+2}!")
    if bh == 0:
        print(f"  >>> FEASIBLE at gen {g+2}!")
        break
