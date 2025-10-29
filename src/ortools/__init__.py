"""
Google OR-Tools CP-SAT Constraint Programming Module

This module provides constraint programming-based scheduling using Google OR-Tools
CP-SAT solver. It generates feasible solutions that satisfy all hard constraints.

Main Components:
    - cp_scheduler: Multi-solution CP-SAT solver orchestration
    - model_builder: CP-SAT model construction
    - variable_factory: Decision variable creation
    - constraint_factory: Hard constraint definitions
    - solution_decoder: CP solution to CourseSession conversion

Usage:
    from src.ortools import CPScheduler

    scheduler = CPScheduler(context, config)
    solutions = scheduler.generate_feasible_solutions(num_solutions=50)
"""

from src.ortools.cp_scheduler import CPScheduler
from src.ortools.solution_decoder import decode_cp_solution

__all__ = ["CPScheduler", "decode_cp_solution"]
