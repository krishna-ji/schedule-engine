"""
Hybrid control system combining RL and heuristics.

Provides multiple integration modes:
- RL-Primary: RL selects heuristics, execute directly
- RL-Fallback: Try RL first, fallback to fixed strategy on failure
- RL-Assisted: Heuristics with RL guidance/ranking
"""

from schedule_engine.rl.hybrid.hybrid_controller import HybridController

__all__ = ["HybridController"]
