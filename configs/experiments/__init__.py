"""
Experiment configurations using dataclass-based hierarchy.

Architecture:
- BaseConfig: Global defaults (time system, penalties, feature flags)
- TestConfig/ProdConfig: Scaling profiles (ngen, pop_size, metrics)
- ExperimentXBaseConfig: Experiment semantics (killswitches) as mixin
- ExperimentXTestConfig/ProdConfig: Combine semantics + scaling via multiple inheritance

Import experiment configs from this module:
- baseline: Pure NSGA-II (Mode A)
- memetic: NSGA-II + Memetic local search (Mode B)
- roundrobin: NSGA-II + Round-robin heuristics (Mode C)
- adaptive: NSGA-II + Adaptive heuristics (Mode D)
- rl_guided: NSGA-II + RL-guided control (Mode E)
- heuristic_testing: Individual heuristic testing (Mode F)
"""

# Import all experiment modules
from . import adaptive, baseline, heuristic_testing, memetic, rl_guided, roundrobin

# Import all dataclass configs for direct use
from .adaptive import (
    AdaptiveBaseConfig,
    AdaptiveProdConfig,
    AdaptiveTestConfig,
    adaptive_prod,
    adaptive_test,
)
from .baseline import (
    BaselineBaseConfig,
    BaselineProdConfig,
    BaselineTestConfig,
    baseline_prod,
    baseline_test,
)
from .heuristic_testing import (
    HeuristicTestingBaseConfig,
    HeuristicTestingProdConfig,
    HeuristicTestingTestConfig,
    heuristic_testing_prod,
    heuristic_testing_test,
)
from .memetic import (
    MemeticBaseConfig,
    MemeticProdConfig,
    MemeticTestConfig,
    memetic_prod,
    memetic_test,
)
from .rl_guided import (
    RlGuidedBaseConfig,
    RlGuidedProdConfig,
    RlGuidedTestConfig,
    rl_guided_prod,
    rl_guided_test,
)
from .roundrobin import (
    RoundRobinBaseConfig,
    RoundRobinProdConfig,
    RoundRobinTestConfig,
    roundrobin_prod,
    roundrobin_test,
)

# Experiment registry for CLI/launcher integration
EXPERIMENT_REGISTRY = {
    "baseline": {
        "id": "A",
        "name": "Pure NSGA-II Baseline",
        "description": "Minimal NSGA-II (no repairs, no heuristics, no enhancements)",
        "test_config": baseline_test,
        "prod_config": baseline_prod,
        "module": baseline,
    },
    "memetic": {
        "id": "B",
        "name": "Memetic NSGA-II",
        "description": "NSGA-II with memetic local search on elite solutions",
        "test_config": memetic_test,
        "prod_config": memetic_prod,
        "module": memetic,
    },
    "roundrobin": {
        "id": "C",
        "name": "Round-Robin Heuristics",
        "description": "NSGA-II with round-robin heuristic selection",
        "test_config": roundrobin_test,
        "prod_config": roundrobin_prod,
        "module": roundrobin,
    },
    "adaptive": {
        "id": "D",
        "name": "Adaptive Heuristic Selection",
        "description": "NSGA-II with performance-based adaptive heuristic priority",
        "test_config": adaptive_test,
        "prod_config": adaptive_prod,
        "module": adaptive,
    },
    "rl_guided": {
        "id": "E",
        "name": "RL-Guided Hyper-Heuristic",
        "description": "Full NSGA-II with RL-guided heuristic selection",
        "test_config": rl_guided_test,
        "prod_config": rl_guided_prod,
        "module": rl_guided,
    },
    "heuristic_testing": {
        "id": "F",
        "name": "Individual Heuristic Testing",
        "description": "Test individual heuristics in isolation",
        "test_config": heuristic_testing_test,
        "prod_config": heuristic_testing_prod,
        "module": heuristic_testing,
    },
}

__all__ = [
    # Modules
    "baseline",
    "memetic",
    "roundrobin",
    "adaptive",
    "rl_guided",
    "heuristic_testing",
    # Base configs (mixins)
    "BaselineBaseConfig",
    "MemeticBaseConfig",
    "RoundRobinBaseConfig",
    "AdaptiveBaseConfig",
    "RlGuidedBaseConfig",
    "HeuristicTestingBaseConfig",
    # Test configs
    "BaselineTestConfig",
    "MemeticTestConfig",
    "RoundRobinTestConfig",
    "AdaptiveTestConfig",
    "RlGuidedTestConfig",
    "HeuristicTestingTestConfig",
    # Prod configs
    "BaselineProdConfig",
    "MemeticProdConfig",
    "RoundRobinProdConfig",
    "AdaptiveProdConfig",
    "RlGuidedProdConfig",
    "HeuristicTestingProdConfig",
    # Instances
    "baseline_test",
    "baseline_prod",
    "memetic_test",
    "memetic_prod",
    "roundrobin_test",
    "roundrobin_prod",
    "adaptive_test",
    "adaptive_prod",
    "rl_guided_test",
    "rl_guided_prod",
    "heuristic_testing_test",
    "heuristic_testing_prod",
    # Registry
    "EXPERIMENT_REGISTRY",
]
