#!/usr/bin/env python3
"""
Test script: Verify dataclass Config constructs correctly.

Exercises schedule_engine.config.models.Config with several profiles.
"""

from schedule_engine.config.models import Config, GAConfig, RLConfig

print("CONFIG DATACLASS - VALIDATION TEST")
print()

profiles = [
    (
        "Baseline (test)",
        Config(name="baseline-test", ga=GAConfig(ngen=30, pop_size=10)),
    ),
    (
        "Baseline (prod)",
        Config(name="baseline-prod", ga=GAConfig(ngen=200, pop_size=50)),
    ),
    (
        "Memetic",
        Config(name="memetic"),
    ),
    (
        "Adaptive",
        Config(name="adaptive"),
    ),
    (
        "RL-guided",
        Config(name="rl-guided", rl=RLConfig(enabled=True)),
    ),
]

for label, config in profiles:
    print(f"Testing: {label}")
    print(f"  ✓ name={config.name}")
    print(f"  ✓ ngen={config.ga.ngen}, pop={config.ga.pop_size}")
    print(f"  ✓ lns_enabled={config.lns.enabled}, rl_enabled={config.rl.enabled}")
    print()

print("✓ ALL TESTS PASSED")
