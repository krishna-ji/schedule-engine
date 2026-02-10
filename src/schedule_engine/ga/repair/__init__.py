"""GA Repair: All repair operators consolidated in one package.

Provides:
    - repair_individual, repair_individual_unified: Core repair functions
    - repair_individual_selective: Targeted repair (violated genes only)
    - RepairEngine: RL-ready repair engine with pluggable policies
    - RepairPipeline: Unified repair orchestration
    - detect_violated_genes: Violation detection for targeted repair
    - repair_operator: Decorator for repair operator registration
    - Specialized repairs: LNS, IGLS, greedy, exhaustive, memetic, etc.
"""

from __future__ import annotations

from schedule_engine.ga.repair.basic import (
    repair_individual,
    repair_individual_unified,
)
from schedule_engine.ga.repair.detector import detect_violated_genes
from schedule_engine.ga.repair.engine import RepairEngine
from schedule_engine.ga.repair.pipeline import RepairPipeline
from schedule_engine.ga.repair.selective import repair_individual_selective
from schedule_engine.ga.repair.wrappers import (
    get_all_repair_operators,
    get_enabled_repair_operators,
    get_repair_operator_function,
    get_repair_operator_metadata,
    get_repair_statistics_template,
    repair_operator,
)

__all__ = [
    # Core repair
    "repair_individual",
    "repair_individual_unified",
    "repair_individual_selective",
    # Engine & pipeline
    "RepairEngine",
    "RepairPipeline",
    # Detection
    "detect_violated_genes",
    # Registry (decorator-based)
    "repair_operator",
    "get_all_repair_operators",
    "get_enabled_repair_operators",
    "get_repair_operator_metadata",
    "get_repair_operator_function",
    "get_repair_statistics_template",
]
