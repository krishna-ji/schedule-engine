"""Phase 55b verification: post-gen BitsetRepair on BEST survivors.

Tests LLH actions using the integrated env (post-gen repair is now
built into env.step()).
"""

import time

import numpy as np

from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

ACTION_NAMES = {
    0: "Conservative (10%, 2p)",
    1: "Aggressive (25%, 3p)",
    2: "Memetic (15%, 4p)",
    3: "SoftFocus (8%, 2p)",
    4: "Destructive (20%, 2p)",
    5: "Intensified (20%, 3p)",
}


def test_action(action_id: int, n_gens: int = 15):
    """Run a single LLH action for n_gens and report results."""
    name = ACTION_NAMES[action_id]
    print(f"\n{'='*60}")
    print(f"Action {action_id}: {name}")
    print(f"{'='*60}")

    env = PymooHyperHeuristicEnv(
        pkl_path=".cache/events_with_domains.pkl",
        max_generations=50,
        pop_size=120,
        algorithm_name="nsga2",
        seed=42,
    )
    obs, info = env.reset()
    print(f"  Gen  1: hard={info['best_hard']:.0f}  mean={info['mean_hard']:.0f}")

    t0 = time.perf_counter()
    best_ever = info["best_hard"]
    milestone_gen = None

    for g in range(n_gens):
        obs, reward, done, trunc, info = env.step(action_id)
        bh = info["best_hard"]
        mh = info["mean_hard"]
        n_rep = info.get("n_repaired", 0)
        t = time.perf_counter() - t0
        best_ever = min(best_ever, bh)
        tag = ""
        if bh < 100 and milestone_gen is None:
            milestone_gen = g + 2
            tag = " <<< MILESTONE"
        print(
            f"  Gen {g+2:2d}: hard={bh:.0f}  mean={mh:.0f}  repaired={n_rep}  {t:.1f}s{tag}"
        )

    total_time = time.perf_counter() - t0
    per_gen = total_time / n_gens

    print(
        f"\n  Result: best_ever={best_ever:.0f}, milestone={'N/A' if milestone_gen is None else f'gen {milestone_gen}'}, {per_gen:.1f}s/gen"
    )
    return best_ever, milestone_gen, per_gen


if __name__ == "__main__":
    print("Phase 55b Verification: Post-gen BitsetRepair on BEST survivors")
    print("Testing Conservative (action=0) and Memetic (action=2)")

    results = {}
    for aid in [0, 2]:
        best, milestone, spg = test_action(aid, n_gens=15)
        results[aid] = (best, milestone, spg)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for aid, (best, milestone, spg) in results.items():
        name = ACTION_NAMES[aid]
        status = "PASS" if best < 100 else "FAIL"
        mg = f"gen {milestone}" if milestone else "N/A"
        print(
            f"  [{status}] Action {aid} ({name}): best={best:.0f}, milestone={mg}, {spg:.1f}s/gen"
        )
