"""
Experiment A: Pure NSGA-II Baseline

Minimal NSGA-II with all enhancements disabled.
Serves as baseline for comparing other experiments.
"""

from src.config.presets.blueprints import PureNsgaBlueprint
from src.config.presets.profiles import Profile

# Instantiate blueprint
experiment_a = PureNsgaBlueprint()

# Experiment metadata
EXPERIMENT_ID = "A"
EXPERIMENT_NAME = "Pure NSGA-II Baseline"
EXPERIMENT_DESCRIPTION = "Minimal NSGA-II (no repairs, no heuristics, no enhancements)"

# Killswitches (explicit documentation)
KILLSWITCHES = {
    "repair.enabled": False,
    "heuristics.adaptive_priority.enabled": False,
    "lns.enabled": False,
    "rl.enabled": False,
    "enhancements.master_enabled": False,
}


# Quick usage
def get_config(profile: Profile = Profile.TEST):
    """Get config for Experiment A."""
    return experiment_a.build(profile)


if __name__ == "__main__":
    # Test the config
    config = get_config(Profile.TEST)
    print(f"✓ {EXPERIMENT_NAME}")
    print(f"  Repair: {config.repair.enabled}")
    print(f"  RL: {config.rl.enabled}")
    print(f"  Enhancements: {config.enhancements.master_enabled}")
