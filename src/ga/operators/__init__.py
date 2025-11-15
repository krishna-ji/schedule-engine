"""
GA Operators Package

Exports genetic operators and repair heuristics.
"""

from src.ga.operators.mutation import mutate_individual, mutate_gene
from src.ga.operators.crossover import crossover_course_group_aware
from src.ga.operators.repair import repair_individual, repair_individual_unified
from src.ga.operators.repair_selective import repair_individual_selective
from src.ga.operators.violation_detector import detect_violated_genes

# Modern decorator-based repair registry
from src.ga.operators.repair_wrappers import (
    repair_operator,
    get_all_repair_operators,
    get_enabled_repair_operators,
    get_repair_operator_metadata,
    get_repair_operator_function,
    get_repair_statistics_template,
)

__all__ = [
    # Mutation operators
    "mutate_individual",
    "mutate_gene",
    # Crossover operators
    "crossover_course_group_aware",
    # Repair operators
    "repair_individual",
    "repair_individual_unified",
    "repair_individual_selective",
    # Violation detection
    "detect_violated_genes",
    # Repair registry (decorator-based)
    "repair_operator",
    "get_all_repair_operators",
    "get_enabled_repair_operators",
    "get_repair_operator_metadata",
    "get_repair_operator_function",
    "get_repair_statistics_template",
]
