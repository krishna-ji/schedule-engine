"""
Experiment C: Round-Robin Heuristics

NSGA-II with fixed rotation through heuristic operators.
Tests effectiveness of heuristic toolbox.
"""

from src.config.presets.blueprints import RoundRobinHeuristicBlueprint
from src.config.presets.profiles import Profile

# Instantiate blueprint
experiment_c = RoundRobinHeuristicBlueprint()

# Experiment metadata
EXPERIMENT_ID = "C"
EXPERIMENT_NAME = "Round-Robin Heuristics"
EXPERIMENT_DESCRIPTION = "NSGA-II + round-robin heuristic selection"

# Killswitches (explicit documentation)
KILLSWITCHES = {
    "repair.enabled": True,
    "repair.memetic_mode": True,
    "heuristics.adaptive_priority.enabled": False,  # Fixed rotation, not adaptive
    "heuristics.construction.largest_degree_first.enabled": True,
    "heuristics.perturbation.random_swap.enabled": True,
    "lns.enabled": False,
    "rl.enabled": False,
    "enhancements.master_enabled": True,
}


# Quick usage
def get_config(profile: Profile = Profile.TEST):
    """Get config for Experiment C."""
    return experiment_c.build(profile)


if __name__ == "__main__":
    config = get_config(Profile.TEST)
    print(f"✓ {EXPERIMENT_NAME}")
    print(f"  Repair: {config.repair.enabled}")
    print(f"  Adaptive priority: {config.heuristics.adaptive_priority.enabled}")
    print(f"  Enhancements: {config.enhancements.master_enabled}")
