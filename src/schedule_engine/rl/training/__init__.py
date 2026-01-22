"""
RL Training Infrastructure.

Provides training utilities for RL agents:
- RLTrainer: Main training class with TensorBoard logging
- Callbacks: Training callbacks (periodic eval, early stopping, checkpoints)
- CurriculumManager: Multi-stage curriculum learning
- Checkpoints: Checkpoint metadata management
"""

from schedule_engine.rl.training.trainer import RLTrainer, create_trainer

__all__ = ["RLTrainer", "create_trainer"]
