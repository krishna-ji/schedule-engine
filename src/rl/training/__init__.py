"""
RL training infrastructure.

Handles agent training, hyperparameter tuning, curriculum learning:
- Training loop with checkpointing and logging
- TensorBoard integration for metrics
- Curriculum learning (easy → medium → hard problems)
- Hyperparameter optimization with Optuna
"""

from src.rl.training.trainer import RLTrainer
from src.rl.training.curriculum import CurriculumManager

__all__ = ["RLTrainer", "CurriculumManager"]
