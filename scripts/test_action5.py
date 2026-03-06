"""Quick run of Action 5 (Intensified) only — the one that crashed."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rl.gym_env.pymoo_env import PymooHyperHeuristicEnv

POP_SIZE = 120
MAX_GEN = 50
SEED = 42

print("Running Action 5 (Intensified, 20%, 3p alt)...")
env = PymooHyperHeuristicEnv(
    pkl_path=".cache/events_with_domains.pkl",
    max_generations=MAX_GEN,
    pop_size=POP_SIZE,
    algorithm_name="nsga2",
    seed=SEED,
)
obs, info = env.reset()
print(
    f"  Gen  1: hard={info['best_hard']:.0f}  soft={info['best_soft']:.0f}  mean={info['mean_hard']:.0f}"
)

t0 = time.perf_counter()
for g in range(MAX_GEN - 1):
    obs, reward, done, trunc, info = env.step(5)
    t = time.perf_counter() - t0
    gen = info["generation"]
    bh = info["best_hard"]
    bs = info["best_soft"]
    mh = info["mean_hard"]
    nr = info.get("n_repaired", 0)
    print(
        f"  Gen {gen:>2}: hard={bh:.0f}  soft={bs:.0f}  mean={mh:.0f}  repaired={nr}  {t:.1f}s"
    )
    if done or trunc:
        break

total = time.perf_counter() - t0
print(
    f"\nDone. Final: hard={info['best_hard']:.0f}  soft={info['best_soft']:.0f}  total={total:.1f}s  ({total/(MAX_GEN-1):.1f}s/gen)"
)
env.close()
