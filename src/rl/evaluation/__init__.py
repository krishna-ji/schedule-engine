"""
RL agent evaluation and baseline comparison.

Provides comprehensive evaluation suite:
- Baseline strategies (random, round-robin, greedy, expert)
- Metrics collection (fitness, convergence, diversity)
- Statistical analysis (t-tests, effect sizes)
- Performance benchmarking
"""

from src.rl.evaluation.baselines import BaselineStrategies
from src.rl.evaluation.evaluator import RLEvaluator
from src.rl.evaluation.metrics import MetricsCollector

__all__ = ["BaselineStrategies", "RLEvaluator", "MetricsCollector"]
