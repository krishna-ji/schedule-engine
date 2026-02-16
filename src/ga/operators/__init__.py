"""
GA Operators Package

Exports genetic operators (crossover, mutation, selection, local search).
Repair operators have moved to src.ga.repair.
"""

from src.ga.operators.crossover import crossover_course_group_aware
from src.ga.operators.mutation import mutate_gene, mutate_individual

# Re-export repair symbols for backward compatibility
from src.ga.repair.basic import repair_individual, repair_individual_unified
from src.ga.repair.detector import detect_violated_genes
from src.ga.repair.engine import RepairEngine
from src.ga.repair.selective import repair_individual_selective
from src.ga.repair.wrappers import (
    get_all_repair_operators,
    get_enabled_repair_operators,
    get_repair_operator_function,
    get_repair_operator_metadata,
    get_repair_statistics_template,
    repair_operator,
)

__all__ = [
    "RepairEngine",
    # Crossover operators
    "crossover_course_group_aware",
    # Violation detection
    "detect_violated_genes",
    "get_all_repair_operators",
    "get_enabled_repair_operators",
    "get_repair_operator_function",
    "get_repair_operator_metadata",
    "get_repair_statistics_template",
    "mutate_gene",
    # Mutation operators
    "mutate_individual",
    # Repair operators (re-exported from ga.repair for compat)
    "repair_individual",
    "repair_individual_selective",
    "repair_individual_unified",
    # Repair registry (decorator-based)
    "repair_operator",
]
