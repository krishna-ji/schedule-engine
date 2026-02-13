#!/usr/bin/env python3
"""
Compare all 3 initialization strategies: smart, hybrid, random
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from schedule_engine.experiments import BaselineExperiment

# Test config - same for all strategies
SEED = 42
POP_SIZE = 50
NGEN = 50  # Enough to see meaningful differences
CXPB = 0.9
MUTPB = 0.2
FITNESS_WEIGHTS = (-1.0, -1.0)
DATA_DIR = PROJECT_ROOT / "data"
OPENING_TIME = "10:00"
CLOSING_TIME = "17:00"
CLOSED_DAYS = ["Saturday"]
LOG_INTERVAL = 10
VERBOSE = True

strategies = ["smart", "hybrid", "random"]
results = {}

for strategy in strategies:
    print(f"\n{'='*60}")
    print(f"RUNNING: {strategy.upper()} initialization")
    print(f"{'='*60}\n")
    
    exp = BaselineExperiment(
        seed=SEED,
        pop_size=POP_SIZE,
        ngen=NGEN,
        cxpb=CXPB,
        mutpb=MUTPB,
        fitness_weights=FITNESS_WEIGHTS,
        data_dir=DATA_DIR,
        output_dir=None,
        opening_time=OPENING_TIME,
        closing_time=CLOSING_TIME,
        closed_days=CLOSED_DAYS,
        init_strategy=strategy,
        log_interval=LOG_INTERVAL,
        verbose=VERBOSE,
    )
    exp.run()
    
    # Store results from experiment's internal stats
    results[strategy] = {
        "hard": exp._stats.min_hard[-1] if exp._stats.min_hard else None,
        "soft": exp._stats.min_soft[-1] if exp._stats.min_soft else None,
        "time": exp._stats.elapsed_time,
        "output_dir": str(exp.output_dir),
    }

print("\n" + "="*60)
print("INITIALIZATION STRATEGY COMPARISON RESULTS")
print("="*60)
print(f"\nConfig: pop={POP_SIZE}, ngen={NGEN}, seed={SEED}\n")
print(f"{'Strategy':<12} {'Hard':<12} {'Soft':<12} {'Time (s)':<12}")
print("-"*48)
for strategy, data in results.items():
    hard = data['hard']
    soft = data['soft']
    time_s = f"{data['time']:.1f}"
    print(f"{strategy:<12} {hard:<12} {soft:<12} {time_s:<12}")

# Find winner
best_hard = min(results.items(), key=lambda x: x[1]['hard'])
best_soft = min(results.items(), key=lambda x: x[1]['soft'])
print("\n" + "-"*48)
print(f"Best hard violations: {best_hard[0]} ({best_hard[1]['hard']})")
print(f"Best soft violations: {best_soft[0]} ({best_soft[1]['soft']})")
