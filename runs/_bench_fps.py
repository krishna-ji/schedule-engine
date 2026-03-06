"""Quick FPS benchmark for Elite 8 at pop_size=120."""

import sys
import time

sys.path.insert(0, ".")
import numpy as np

from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

env = PymooHyperHeuristicEnv(
    pkl_path=".cache/events_with_domains.pkl",
    max_generations=10,
    pop_size=120,
    seed=42,
)
obs, info = env.reset()
print(f"Reset OK. Best hard: {info['best_hard']}")

t0 = time.perf_counter()
# Force each of the 8 actions once, then 12 more random
actions_to_test = list(range(8)) + [None] * 12
for i, forced_action in enumerate(actions_to_test):
    masks = env.action_masks()
    valid = np.where(masks)[0]
    if forced_action is not None and masks[forced_action]:
        action = forced_action
    else:
        action = int(np.random.choice(valid))
    t_step = time.perf_counter()
    obs, r, term, trunc, info = env.step(action)
    dt = time.perf_counter() - t_step
    print(
        f"  Step {i+1:2d}: action={action} ({info['action_name']:<35s}) "
        f"dt={dt:.3f}s hard={info['best_hard']:.0f} r={r:.4f}"
    )
    if term or trunc:
        break

elapsed = time.perf_counter() - t0
steps = min(i + 1, 20)
print(f"\nTotal: {elapsed:.2f}s | FPS: {steps/elapsed:.1f}")
print(f"ETA for 150k steps: {150000/max(steps/elapsed, 0.01)/60:.1f} minutes")
env.close()
