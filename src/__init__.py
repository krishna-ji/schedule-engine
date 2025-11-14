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

Usage:
    from src.config import init_config
    from src.workflows import run_standard_workflow

    config = init_config()
    result = run_standard_workflow(
        pop_size=config.ga.pop_size,
        generations=config.ga.ngen,
        config=config
    )

Authors:
    Krishna Acharya, Dinanath Padhya, Bipul Dahal

License:
    MIT
"""

__version__ = "1.0.0"
__author__ = "Krishna Acharya, Dinanath Padhya, Bipul Dahal"
__license__ = "MIT"

__all__ = [
    "__version__",
    "__author__",
    "__license__",
]
