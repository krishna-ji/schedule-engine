"""
Experiment E: RL-Guided Hyper-Heuristic

Full NSGA-II stack with RL controlling heuristic selection.
Tests effectiveness of reinforcement learning guidance.
"""

from src.config.presets.blueprints import RlGuidedBlueprint
from src.config.presets.profiles import Profile

# Instantiate blueprint
experiment_e = RlGuidedBlueprint()

# Experiment metadata
EXPERIMENT_ID = "E"
EXPERIMENT_NAME = "RL-Guided Hyper-Heuristic"
EXPERIMENT_DESCRIPTION = "Full NSGA-II + RL-guided heuristic selection"

# Killswitches (explicit documentation)
KILLSWITCHES = {
    "repair.enabled": True,
    "repair.memetic_mode": True,
    "ga.use_adaptive_probabilities": True,
    "heuristics.adaptive_priority.enabled": False,  # RL takes over
    "heuristics.construction.largest_degree_first.enabled": True,
    "heuristics.perturbation.random_swap.enabled": True,
    "heuristics.improvement.kempe_chain.enabled": True,
    "heuristics.meta.variable_neighborhood_descent.enabled": True,
    "lns.enabled": True,
    "rl.enabled": True,  # KEY: RL enabled
    "rl.mode": "rl_primary",
    "rl.hybrid.rl_probability": 0.8,
    "enhancements.master_enabled": True,
    "enhancements.memetic_mode": True,
    "enhancements.hypermutation.enabled": True,
    "enhancements.population_restart.enabled": True,
}


# Quick usage
def get_config(profile: Profile = Profile.TEST):
    """Get config for Experiment E."""
    return experiment_e.build(profile)


if __name__ == "__main__":
    config = get_config(Profile.TEST)
    print(f"✓ {EXPERIMENT_NAME}")
    print(f"  RL enabled: {config.rl.enabled}")
    print(f"  RL mode: {config.rl.mode}")
    print(f"  LNS enabled: {config.lns.enabled}")
    print(f"  Memetic mode: {config.enhancements.memetic_mode}")
