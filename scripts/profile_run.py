#!/usr/bin/env python3
"""cProfile + pyinstrument on a real GA run (pop=200, gens=50)."""

import cProfile
import io
import json
import logging
import os
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logging_config import quick_setup

logger = quick_setup()

POP = int(sys.argv[1]) if len(sys.argv) > 1 else 200
NGEN = int(sys.argv[2]) if len(sys.argv) > 2 else 50
OUT = Path("results")
OUT.mkdir(exist_ok=True)

pkl_path = "events_with_domains.pkl"

from pymoo.optimize import minimize

from src.pipeline.pymoo_operators import create_algorithm
from src.pipeline.scheduling_problem import create_problem

problem = create_problem(pkl_path)
algo = create_algorithm(pkl_path, pop_size=POP)

logger.info("Profiling: pop=%d, gens=%d", POP, NGEN)

# cProfile
profiler = cProfile.Profile()
profiler.enable()
t0 = time.perf_counter()
res = minimize(problem, algo, ("n_gen", NGEN), seed=42, verbose=False)
wall = time.perf_counter() - t0
profiler.disable()

logger.info("Wall time: %.2fs", wall)
if res.F is not None:
    logger.info("Best F: %s", res.F[:3])

# Save cumulative
s = io.StringIO()
pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats(60)
(OUT / "profile_cprofile.txt").write_text(s.getvalue(), encoding="utf-8")

# Save tottime
s2 = io.StringIO()
pstats.Stats(profiler, stream=s2).sort_stats("tottime").print_stats(60)
(OUT / "profile_cprofile_tottime.txt").write_text(s2.getvalue(), encoding="utf-8")

# Extract top hotspots
stats_list = []
for key, val in profiler.stats.items():
    fn, lineno, func = key
    cc, nc, tt, ct, callers = val
    stats_list.append(
        {
            "file": os.path.basename(fn),
            "lineno": lineno,
            "function": func,
            "ncalls": nc,
            "tottime": round(tt, 4),
            "cumtime": round(ct, 4),
        }
    )
stats_list.sort(key=lambda x: x["tottime"], reverse=True)

logger.info("\n=== TOP 15 BY TOTTIME ===")
for h in stats_list[:15]:
    logger.info(
        "  %-45s %8.3fs tot  %8.3fs cum  (%6d calls)  [%s:%s]",
        h['function'], h['tottime'], h['cumtime'], h['ncalls'], h['file'], h['lineno']
    )

# Save JSON summary
summary = {
    "params": {"pop": POP, "gens": NGEN},
    "wall_time_s": round(wall, 2),
    "top_by_tottime": stats_list[:30],
    "top_by_cumtime": sorted(stats_list, key=lambda x: x["cumtime"], reverse=True)[:30],
}
(OUT / "profile_summary.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

# pyinstrument
logger.info("\n=== pyinstrument run ===")
import pyinstrument

problem2 = create_problem(pkl_path)
algo2 = create_algorithm(pkl_path, pop_size=POP)
profiler2 = pyinstrument.Profiler()
profiler2.start()
res2 = minimize(problem2, algo2, ("n_gen", NGEN), seed=42, verbose=False)
profiler2.stop()

(OUT / "profile_pyinstrument.html").write_text(
    profiler2.output_html(), encoding="utf-8"
)
text = profiler2.output_text(unicode=False)
(OUT / "profile_pyinstrument.txt").write_text(text, encoding="utf-8")
# Show first 40 lines
for line in text.split("\n")[:40]:
    logger.info("%s", line)

logger.info("\nDone. Outputs: results/profile_*.{txt,json,html}")
