"""Quick baseline run for verification."""

import math
import time

from src.experiments.ga_experiment import BaselineExperiment

exp = BaselineExperiment(
    pop_size=20,
    ngen=50,
    seed=42,
    export_pdf=False,
    verbose=True,
)
t0 = time.time()
result = exp.run()
total = time.time() - t0

print()
print("=" * 60)
print("BASELINE RESULT SUMMARY")
print("=" * 60)
print(f"  Total wall time:   {total:.1f}s")
print(f"  GA time:           {result['elapsed_s']}s")
print(f"  sec/gen:           {result['sec_per_gen']}")
print(f"  best_hard:         {result['best_hard']}")
print(f"  best_soft:         {result['best_soft']}")
print(f"  best_cv:           {result['best_cv']}")
print(f"  n_feasible:        {result['n_feasible']}")
print(f"  HV points:         {len(result['hypervolumes'])}")
print(f"  Spacing points:    {len(result['spacings'])}")
print(f"  Feasibility pts:   {len(result['feasibility_rates'])}")
print(f"  IGD points:        {len(result['igds'])}")

hv_list = result["hypervolumes"]
hv_real = [v for v in hv_list if not math.isnan(v)]
if hv_real:
    print(f"  HV (last finite):  {hv_real[-1]:.2f}")
else:
    print("  HV: all nan (no feasible solutions)")

feas = result["feasibility_rates"]
if feas:
    print(f"  Feas rate (last):  {feas[-1]:.3f}")

igd_list = result["igds"]
igd_real = [v for v in igd_list if not math.isnan(v)]
if igd_real:
    print(f"  IGD (last finite): {igd_real[-1]:.4f}")
else:
    print("  IGD: all nan (no reference front)")
