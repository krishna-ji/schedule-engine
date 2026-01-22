"""
Production deployment utilities for trained RL models.

Provides fast model loading and inference for production use:
- Model loading from checkpoints (<100ms)
- Fast inference (<10ms per prediction)
- Version management and model registry
- CPU-only device selection
- Registry-driven promotions with rollback
"""

from schedule_engine.rl.deployment.inference import RLInference
from schedule_engine.rl.deployment.model_loader import ModelLoader
from schedule_engine.rl.deployment.registry import ModelRegistration, ModelRegistry

__all__ = ["ModelLoader", "RLInference", "ModelRegistry", "ModelRegistration"]
