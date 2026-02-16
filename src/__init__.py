"""
Schedule Engine - University Course Scheduling with NSGA-II

A genetic algorithm-based course scheduling engine that optimizes university
timetables using NSGA-II (Non-dominated Sorting Genetic Algorithm II) with
constraint-based optimization.

Main Features:
    - Multi-objective optimization (hard constraints vs soft preferences)
    - Constraint-guided repair mechanisms
    - Parallel evaluation support
    - Rich terminal UI with progress tracking
    - Comprehensive feasibility analysis
    - PDF calendar export and evolution plots

Package Structure:
    - core/: Domain models, constraints, evaluation
    - ga/: Genetic algorithm implementation
    - rl/: Reinforcement learning agents
    - output/: Unified export and visualization
    - experiments/: Experiment modes and runners

Usage:
    from src.config import Config, init_config

    config = Config(ga=dict(ngen=100, pop_size=50), name="example")
    init_config(config)

Authors:
    Krishna Acharya, Dinanath Padhya, Bipul Dahal

License:
    MIT
"""

__version__ = "1.0.0"
__author__ = "Krishna Acharya, Dinanath Padhya, Bipul Dahal"
__license__ = "MIT"

__all__ = [
    "__author__",
    "__license__",
    "__version__",
]
