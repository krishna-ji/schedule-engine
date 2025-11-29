"""
Experiment B: Memetic NSGA-II

NSGA-II with memetic local search repairs enabled.
Tests effectiveness of repair system.
"""

from src.config.presets.blueprints import MemeticNsgaBlueprint
from src.config.presets.profiles import Profile

# Instantiate blueprint
experiment_b = MemeticNsgaBlueprint()

# Experiment metadata
EXPERIMENT_ID = "B"
EXPERIMENT_NAME = "Memetic NSGA-II"
EXPERIMENT_DESCRIPTION = "NSGA-II + memetic local search repairs"

# Killswitches (explicit documentation)
KILLSWITCHES = {
    "repair.enabled": True,
    "repair.memetic_mode": True,
    "heuristics.adaptive_priority.enabled": False,
    "lns.enabled": False,
    "rl.enabled": False,
    "enhancements.master_enabled": False,
}


# Quick usage
def get_config(profile: Profile = Profile.TEST):
    """Get config for Experiment B."""
    return experiment_b.build(profile)


if __name__ == "__main__":
    config = get_config(Profile.TEST)
    print(f"✓ {EXPERIMENT_NAME}")
    print(f"  Repair: {config.repair.enabled}")
    print(f"  Memetic mode: {config.repair.memetic_mode}")
    print(f"  RL: {config.rl.enabled}")
