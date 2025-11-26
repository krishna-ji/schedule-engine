"""
Repair Heuristics Module

Repair operators for constraint violation restoration, now integrated as heuristics.
These operators fix hard constraint violations in GA individuals by projecting
invalid solutions onto the feasible region.

Repair Categories:
1. IGLS Repair - Iterative Greedy Local Search for violated sessions
2. LNS Repair - Large Neighborhood Search with subproblem solving
3. Selective Repair - Targeted repair of only violated genes
4. Exhaustive Search - Intensive steepest descent local search
5. Specialized Repairs - Individual constraint-specific repairs

All repair operators are now registered as heuristics in the repair category,
providing unified management and RL integration.
"""

from src.heuristics.repair.igls_repair import igls_repair
from src.heuristics.repair.lns_repair import lns_repair  
from src.heuristics.repair.selective_repair import selective_repair
from src.heuristics.repair.exhaustive_search import exhaustive_search

__all__ = [
    "igls_repair",
    "lns_repair", 
    "selective_repair",
    "exhaustive_search",
]