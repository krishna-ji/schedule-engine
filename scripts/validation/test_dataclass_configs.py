#!/usr/bin/env python3
"""
Test script: Verify config loader creates valid Pydantic models.

This script exercises schedule_engine.config.loader.dict_to_pydantic
with a handful of profiles used in notebooks and scripts.
"""

from schedule_engine.config.loader import dict_to_pydantic

print("CONFIG LOADER - VALIDATION TEST")

print()

profiles = [
    (
        "Baseline (test)",
        {"experiment_name": "baseline-test", "ngen": 30, "pop_size": 10},
    ),
    (
        "Baseline (prod)",
        {"experiment_name": "baseline-prod", "ngen": 200, "pop_size": 50},
    ),
    ("Memetic", {"experiment_name": "memetic", "heuristics_mode": "memetic"}),
    ("Adaptive", {"experiment_name": "adaptive", "heuristics_mode": "adaptive"}),
    ("RL-guided", {"experiment_name": "rl-guided", "rl_enabled": True}),
]

for name, profile in profiles:
    config = dict_to_pydantic(profile)
    print(f"Testing: {name}")
    print(f"  ✓ name={config.name}")
    print(f"  ✓ ngen={config.ga.ngen}, pop={config.ga.pop_size}")
    print(f"  ✓ lns_enabled={config.lns.enabled}, rl_enabled={config.rl.enabled}")
    print()


print("✓ ALL TESTS PASSED")
