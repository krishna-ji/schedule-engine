"""
Experiment D: Adaptive Heuristic Selection

NSGA-II with performance-based adaptive heuristic priority.
Tests effectiveness of adaptive selection mechanism.
"""

from src.config.presets.blueprints import AdaptiveHeuristicBlueprint
from src.config.presets.profiles import Profile

# Instantiate blueprint
experiment_d = AdaptiveHeuristicBlueprint()

# Experiment metadata
EXPERIMENT_ID = "D"
EXPERIMENT_NAME = "Adaptive Heuristic Selection"
EXPERIMENT_DESCRIPTION = "NSGA-II + adaptive performance-based heuristic selection"

# Killswitches (explicit documentation)
KILLSWITCHES = {
    "repair.enabled": True,
    "repair.memetic_mode": True,
    "ga.use_adaptive_probabilities": True,
    "heuristics.adaptive_priority.enabled": True,  # KEY: Adaptive selection
    "heuristics.adaptive_priority.evaluation_window": 10,
    "heuristics.adaptive_priority.reorder_interval": 10,
    "lns.enabled": False,
    "rl.enabled": False,
    "enhancements.master_enabled": True,
    "enhancements.hypermutation.enabled": True,
}


# Quick usage
def get_config(profile: Profile = Profile.TEST):
    """Get config for Experiment D."""
    return experiment_d.build(profile)


if __name__ == "__main__":
    config = get_config(Profile.TEST)
    print(f"✓ {EXPERIMENT_NAME}")
    print(f"  Adaptive priority: {config.heuristics.adaptive_priority.enabled}")
    print(f"  Adaptive probabilities: {config.ga.use_adaptive_probabilities}")
    print(f"  Hypermutation: {config.enhancements.hypermutation.enabled}")
