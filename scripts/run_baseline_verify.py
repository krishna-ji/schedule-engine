"""Quick baseline run for verification."""

import logging
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_config import quick_setup

logger = quick_setup()

from src.experiments.ga_experiment import BaselineExperiment

exp = BaselineExperiment(
    pop_size=20,
    ngen=50,
    seed=42,
    export_pdf=True,
    verbose=True,
)
t0 = time.time()
result = exp.run()
total = time.time() - t0

logger.info("")
logger.info("=" * 60)
logger.info("BASELINE RESULT SUMMARY")
logger.info("=" * 60)
logger.info("  Total wall time:   %.1fs", total)
logger.info("  GA time:           %ss", result["elapsed_s"])
logger.info("  sec/gen:           %s", result["sec_per_gen"])
logger.info("  best_hard:         %s", result["best_hard"])
logger.info("  best_soft:         %s", result["best_soft"])
logger.info("  best_cv:           %s", result["best_cv"])
logger.info("  n_feasible:        %s", result["n_feasible"])
logger.info("  HV points:         %d", len(result["hypervolumes"]))
logger.info("  Spacing points:    %d", len(result["spacings"]))
logger.info("  Feasibility pts:   %d", len(result["feasibility_rates"]))
logger.info("  IGD points:        %d", len(result["igds"]))

hv_list = result["hypervolumes"]
hv_real = [v for v in hv_list if not math.isnan(v)]
if hv_real:
    logger.info("  HV (last finite):  %.2f", hv_real[-1])
else:
    logger.info("  HV: all nan (no feasible solutions)")

feas = result["feasibility_rates"]
if feas:
    logger.info("  Feas rate (last):  %.3f", feas[-1])

igd_list = result["igds"]
igd_real = [v for v in igd_list if not math.isnan(v)]
if igd_real:
    logger.info("  IGD (last finite): %.4f", igd_real[-1])
else:
    logger.info("  IGD: all nan (no reference front)")
