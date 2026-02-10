"""
GA Operators Package

Exports genetic operators (crossover, mutation, selection, local search).
Repair operators have moved to schedule_engine.ga.repair.
"""

from schedule_engine.ga.operators.crossover import crossover_course_group_aware
from schedule_engine.ga.operators.mutation import mutate_gene, mutate_individual

# Re-export repair symbols for backward compatibility
from schedule_engine.ga.repair.basic import repair_individual, repair_individual_unified
from schedule_engine.ga.repair.selective import repair_individual_selective
from schedule_engine.ga.repair.engine import RepairEngine
from schedule_engine.ga.repair.wrappers import (
    get_all_repair_operators,
    get_enabled_repair_operators,
    get_repair_operator_function,
    get_repair_operator_metadata,
    get_repair_statistics_template,
    repair_operator,
)
from schedule_engine.ga.repair.detector import detect_violated_genes

__all__ = [
    # Mutation operators
    "mutate_individual",
    "mutate_gene",
    # Crossover operators
    "crossover_course_group_aware",
    # Repair operators (re-exported from ga.repair for compat)
    "repair_individual",
    "repair_individual_unified",
    "repair_individual_selective",
    "RepairEngine",
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
