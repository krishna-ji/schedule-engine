#!/usr/bin/env python3
"""Profile baseline GA to find bottlenecks."""
from __future__ import annotations

import cProfile
import pstats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run():
    from src.experiments import BaselineExperiment

    exp = BaselineExperiment(
        seed=42,
        pop_size=50,
        ngen=50,
        crossover_prob=0.5,
        mutation_event_prob=0.05,
        data_dir=ROOT / "data",
        log_interval=5,
        verbose=False,
    )
    exp.run()


if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()

    s = pstats.Stats(pr, stream=sys.stdout)
    print("\n\n===== TOP 50 BY CUMULATIVE TIME =====")
    s.sort_stats("cumulative")
    s.print_stats(50)

    print("\n\n===== TOP 40 BY TOTAL (SELF) TIME =====")
    s.sort_stats("tottime")
    s.print_stats(40)
