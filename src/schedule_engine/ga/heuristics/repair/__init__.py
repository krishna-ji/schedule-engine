"""
Repair Heuristics Module

Repair operators for constraint violation restoration, now integrated as heuristics.
These operators fix hard constraint violations in GA individuals by projecting
invalid solutions onto the feasible region.

Repair Categories:
1. IGLS Repair - Iterative Greedy Local Search for violated sessions
2. Greedy Repair - First-improving hill climb
3. Selective Repair - Targeted repair of only violated genes
4. LNS Repair - Large Neighborhood Search with subproblem solving
5. Exhaustive Repair - Intensive steepest descent local search
6. Memetic Repair - Elite IGLS for hybrid GA modes
7. Break Placement Repair - Ensures proper break windows for groups

All repair operators are now registered as heuristics in the repair category,
providing unified management and RL integration.
"""

from schedule_engine.ga.heuristics.repair.break_repair import repair_break_placement

# LNS components (migrated into schedule_engine.heuristics.repair)
from schedule_engine.ga.heuristics.repair.conflict_detection import find_hard_conflict_sessions
from schedule_engine.ga.heuristics.repair.exhaustive_repair import exhaustive_repair
from schedule_engine.ga.heuristics.repair.greedy_repair import greedy_repair
from schedule_engine.ga.heuristics.repair.igls_repair import igls_repair
from schedule_engine.ga.heuristics.repair.lns_operator import lns_igls_repair
from schedule_engine.ga.heuristics.repair.lns_repair import lns_repair
from schedule_engine.ga.heuristics.repair.memetic_repair import memetic_repair
from schedule_engine.ga.heuristics.repair.selective_repair import selective_repair

__all__ = [
    "igls_repair",
    "greedy_repair",
    "selective_repair",
    "lns_repair",
    "exhaustive_repair",
    "memetic_repair",
    "repair_break_placement",
    # LNS components
    "find_hard_conflict_sessions",
    "lns_igls_repair",
]
