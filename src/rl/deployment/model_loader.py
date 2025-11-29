"""
Model loader for RL deployment.

Provides fast model loading with caching for production inference.

Features:
- Model caching to avoid repeated loads (<100ms target)
- Version management and validation
- Model metadata tracking
- CPU-only device placement for consistent execution
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """
    Fast model loader with caching for production inference.

    Caches loaded models to avoid repeated I/O operations.
    Tracks loading times and validates models before use.
    """

    def __init__(
        self,
        model_dir: str = "models/rl_agents",
        cache_models: bool = True,
        device: str = "cpu",
    ):
        """
        Initialize model loader.

        Args:
            model_dir: Directory containing trained models
            cache_models: Enable model caching
            device: Device for model inference (CPU-only)
        """
        self.model_dir = Path(model_dir)
        self.cache_models = cache_models
        self.device = self._normalize_device(device)

        # Model cache: {model_path: (model, load_time, metadata)}
        self._cache: dict[str, tuple] = {}

        # Loading statistics
        self.load_count = 0
        self.cache_hit_count = 0
        self.total_load_time = 0.0

        logger.info(
            f"Initialized ModelLoader (cache_models={cache_models}, device={self.device})"
        )

    def _normalize_device(self, requested: str) -> str:
        """Force CPU device regardless of requested value."""

        normalized = (requested or "cpu").lower()
        if normalized != "cpu":
            logger.warning(
                "GPU inference is disabled; forcing CPU device (requested '%s').",
                normalized,
            )
        return "cpu"

    def load_model(
        self,
        model_path: str,
        agent_type: str = "ppo",
        force_reload: bool = False,
    ) -> BaseAlgorithm:
        """
        Load trained model from checkpoint.

        Args:
            model_path: Path to model file (.zip)
            agent_type: Agent type (ppo or dqn)
            force_reload: Force reload even if cached

        Returns:
            Loaded model
        """
        # Resolve path
        model_path = self._resolve_path(model_path)

        # Check cache
        if self.cache_models and not force_reload and model_path in self._cache:
            self.cache_hit_count += 1
            model, load_time, _ = self._cache[model_path]

            logger.debug(
                f"Model loaded from cache: {model_path} (original load time: {load_time:.1f}ms)"
            )
            return model

        # Load model
        start_time = time.time()

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        try:
            if agent_type.lower() == "ppo":
                model = PPO.load(model_path, device=self.device)
            elif agent_type.lower() == "dqn":
                model = DQN.load(model_path, device=self.device)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")

            load_time_ms = (time.time() - start_time) * 1000

            # Update statistics
            self.load_count += 1
            self.total_load_time += load_time_ms

            # Cache model
            if self.cache_models:
                metadata = {
                    "load_time_ms": load_time_ms,
                    "load_timestamp": datetime.now().isoformat(),
                    "agent_type": agent_type,
                    "device": str(self.device),
                }
                self._cache[model_path] = (model, load_time_ms, metadata)

            # Log performance
            if load_time_ms > 100:
                logger.warning(
                    f"Model load time {load_time_ms:.1f}ms exceeds target (100ms): {model_path}"
                )
            else:
                logger.info(f"Model loaded in {load_time_ms:.1f}ms: {model_path}")

            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            raise

    def _resolve_path(self, model_path: str) -> str:
        """
        Resolve model path (handle relative paths, add .zip extension).

        Resolves paths relative to project root to avoid working directory issues.
        """
        path = Path(model_path)

        # Add .zip if missing
        if not path.suffix:
            path = path.with_suffix(".zip")

        # Handle relative paths - resolve relative to project root
        if not path.is_absolute():
            # Try model_dir first
            resolved = self.model_dir / path
            if not resolved.exists():
                # Fall back to project root
                project_root = Path(__file__).parent.parent.parent.parent
                resolved = project_root / path

            path = resolved

        return str(path.resolve())

    def preload_models(self, model_paths: list, agent_type: str = "ppo"):
        """
        Preload multiple models into cache.

        Args:
            model_paths: List of model paths to preload
            agent_type: Agent type
        """
        logger.info(f"Preloading {len(model_paths)} models...")

        for model_path in model_paths:
            try:
                self.load_model(model_path, agent_type=agent_type)
            except Exception as e:
                logger.error(f"Failed to preload {model_path}: {e}")

        logger.info(f"Preloaded {len(self._cache)} models")

    def clear_cache(self):
        """Clear model cache."""
        num_cached = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared model cache ({num_cached} models)")

    def get_cached_models(self) -> list:
        """Get list of cached model paths."""
        return list(self._cache.keys())

    def validate_model(
        self,
        model: BaseAlgorithm,
        expected_action_space: Any | None = None,
        expected_observation_space: Any | None = None,
    ) -> bool:
        """
        Validate model has expected action/observation spaces.

        Args:
            model: Model to validate
            expected_action_space: Expected action space (optional)
            expected_observation_space: Expected observation space (optional)

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check action space
            if expected_action_space is not None:
                if model.action_space != expected_action_space:
                    logger.error(
                        f"Action space mismatch: expected {expected_action_space}, "
                        f"got {model.action_space}"
                    )
                    return False

            # Check observation space
            if expected_observation_space is not None:
                if model.observation_space != expected_observation_space:
                    logger.error(
                        f"Observation space mismatch: expected {expected_observation_space}, "
                        f"got {model.observation_space}"
                    )
                    return False

            logger.debug("Model validation passed")
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """Get loader statistics."""
        avg_load_time = 0.0
        if self.load_count > 0:
            avg_load_time = self.total_load_time / self.load_count

        cache_hit_rate = 0.0
        total_requests = self.load_count + self.cache_hit_count
        if total_requests > 0:
            cache_hit_rate = self.cache_hit_count / total_requests

        return {
            "load_count": self.load_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_hit_rate": cache_hit_rate,
            "avg_load_time_ms": avg_load_time,
            "cached_models": len(self._cache),
            "total_requests": total_requests,
        }

    def benchmark_load_time(
        self, model_path: str, agent_type: str = "ppo", runs: int = 10
    ) -> dict[str, float]:
        """
        Benchmark model loading time.

        Args:
            model_path: Model to benchmark
            agent_type: Agent type
            runs: Number of benchmark runs

        Returns:
            Benchmark results (mean, min, max, std)
        """
        import numpy as np

        logger.info(f"Benchmarking model load time ({runs} runs)...")

        times = []

        for i in range(runs):
            # Clear cache to force reload
            self.clear_cache()

            start = time.time()
            model = self.load_model(
                model_path, agent_type=agent_type, force_reload=True
            )
            elapsed_ms = (time.time() - start) * 1000

            times.append(elapsed_ms)
            logger.debug(f"Run {i+1}/{runs}: {elapsed_ms:.2f}ms")

        results = {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "median_ms": float(np.median(times)),
        }

        logger.info(
            f"Benchmark results: {results['mean_ms']:.2f} ± {results['std_ms']:.2f}ms"
        )

        if results["mean_ms"] > 100:
            logger.warning(
                f"[WARNING] Mean load time {results['mean_ms']:.2f}ms exceeds target (100ms)"
            )
        else:
            logger.info(
                f"[OK] Mean load time {results['mean_ms']:.2f}ms within target (<100ms)"
            )

        return results


def create_model_loader(**kwargs) -> ModelLoader:
    """
    Convenience function to create model loader.

    Args:
        **kwargs: ModelLoader arguments

    Returns:
        Configured ModelLoader
    """
    return ModelLoader(**kwargs)
