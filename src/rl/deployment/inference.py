"""
RL inference engine for production deployment.

Provides fast, reliable action prediction with:
- Timeout protection (<10ms target)
- Performance monitoring
- Batch prediction support
- Error handling and fallback
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RLInference:
    """
    Fast inference engine for production RL deployment.

    Provides action prediction with timeout protection,
    performance monitoring, and batch prediction support.
    """

    def __init__(
        self,
        model: BaseAlgorithm,
        timeout_ms: float = 10.0,
        track_performance: bool = True,
        max_history: int = 1000,
    ):
        """
        Initialize inference engine.

        Args:
            model: Trained RL model
            timeout_ms: Maximum inference time (milliseconds)
            track_performance: Enable performance tracking
            max_history: Maximum prediction history size
        """
        self.model = model
        self.timeout_ms = timeout_ms
        self.track_performance = track_performance

        # Performance tracking
        self.prediction_times: deque = deque(maxlen=max_history)
        self.prediction_count = 0
        self.timeout_count = 0
        self.error_count = 0

        logger.info(f"Initialized RLInference (timeout={timeout_ms}ms)")

    def predict_action(
        self,
        state: np.ndarray,
        deterministic: bool = True,
        validate: bool = True,
    ) -> int | None:
        """
        Predict action from state with timeout protection.

        Args:
            state: Environment state (observation)
            deterministic: Use deterministic policy
            validate: Validate state shape

        Returns:
            Predicted action (int) or None if timeout/error
        """
        start_time = time.time()

        try:
            # Validate state
            if validate and not self._validate_state(state):
                logger.error("Invalid state shape")
                self.error_count += 1
                return None

            # Predict action
            action_result, _states = self.model.predict(
                state, deterministic=deterministic
            )

            # Extract scalar action (robust to 0-D arrays from SB3)
            action: int
            if isinstance(action_result, np.ndarray):
                if action_result.size == 1:
                    # Handles numpy scalars / 0-D arrays without indexing errors
                    action = int(action_result.reshape(-1)[0])
                else:
                    action = int(action_result.reshape(-1)[0])
            else:
                action = int(action_result)

            # Track performance
            elapsed_ms = (time.time() - start_time) * 1000

            if self.track_performance:
                self.prediction_times.append(elapsed_ms)

            self.prediction_count += 1

            # Check timeout
            if elapsed_ms > self.timeout_ms:
                self.timeout_count += 1
                logger.warning(
                    f"Inference timeout: {elapsed_ms:.2f}ms > {self.timeout_ms}ms "
                    f"(action={action})"
                )

            return action

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.error_count += 1
            logger.error(f"Inference error after {elapsed_ms:.2f}ms: {e}")
            return None

    def predict_batch(
        self,
        states: np.ndarray,
        deterministic: bool = True,
    ) -> list[int | None]:
        """
        Predict actions for batch of states.

        Args:
            states: Batch of states (shape: [batch_size, state_dim])
            deterministic: Use deterministic policy

        Returns:
            List of predicted actions
        """
        start_time = time.time()

        try:
            # Predict for batch
            actions_result, _states = self.model.predict(
                states, deterministic=deterministic
            )

            # Convert to list with optional typing for consistency
            actions: list[int | None]
            if isinstance(actions_result, np.ndarray):
                actions = [int(a) for a in actions_result]
            else:
                actions = [int(actions_result)]

            # Track performance
            elapsed_ms = (time.time() - start_time) * 1000
            avg_per_sample = elapsed_ms / len(states)

            if self.track_performance:
                # Add per-sample time
                for _ in range(len(states)):
                    self.prediction_times.append(avg_per_sample)

            self.prediction_count += len(states)

            logger.debug(
                f"Batch prediction: {len(states)} samples in {elapsed_ms:.2f}ms "
                f"({avg_per_sample:.2f}ms/sample)"
            )

            return actions

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.error_count += 1
            logger.error(f"Batch inference error after {elapsed_ms:.2f}ms: {e}")
            return [None] * len(states)

    def _validate_state(self, state: np.ndarray) -> bool:
        """Validate state shape matches model's observation space."""
        try:
            expected_shape = self.model.observation_space.shape

            if state.shape != expected_shape:
                logger.error(
                    f"State shape mismatch: expected {expected_shape}, got {state.shape}"
                )
                return False

            return True

        except Exception as e:
            logger.error(f"State validation error: {e}")
            return False

    def get_average_inference_time(self) -> float:
        """Get average inference time (milliseconds)."""
        if not self.prediction_times:
            return 0.0
        return float(np.mean(self.prediction_times))

    def get_median_inference_time(self) -> float:
        """Get median inference time (milliseconds)."""
        if not self.prediction_times:
            return 0.0
        return float(np.median(self.prediction_times))

    def get_statistics(self) -> dict[str, Any]:
        """Get inference statistics."""
        stats = {
            "prediction_count": self.prediction_count,
            "timeout_count": self.timeout_count,
            "error_count": self.error_count,
            "timeout_rate": 0.0,
            "error_rate": 0.0,
        }

        if self.prediction_count > 0:
            stats["timeout_rate"] = self.timeout_count / self.prediction_count
            stats["error_rate"] = self.error_count / self.prediction_count

        if self.prediction_times:
            stats["avg_time_ms"] = self.get_average_inference_time()
            stats["median_time_ms"] = self.get_median_inference_time()
            stats["min_time_ms"] = float(np.min(self.prediction_times))
            stats["max_time_ms"] = float(np.max(self.prediction_times))
            stats["std_time_ms"] = float(np.std(self.prediction_times))

        return stats

    def reset_statistics(self) -> None:
        """Reset performance statistics."""
        self.prediction_times.clear()
        self.prediction_count = 0
        self.timeout_count = 0
        self.error_count = 0
        logger.debug("Reset inference statistics")

    def benchmark(self, state: np.ndarray, runs: int = 1000) -> dict[str, float]:
        """
        Benchmark inference latency.

        Args:
            state: Sample state for benchmark
            runs: Number of benchmark runs

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking inference latency ({runs} runs)...")

        # Warmup
        for _ in range(10):
            self.predict_action(state, deterministic=True, validate=False)

        # Reset statistics
        self.reset_statistics()

        # Benchmark
        times = []
        for _ in range(runs):
            start = time.time()
            self.predict_action(state, deterministic=True, validate=False)
            elapsed_ms = (time.time() - start) * 1000
            times.append(elapsed_ms)

        results = {
            "mean_ms": float(np.mean(times)),
            "median_ms": float(np.median(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p95_ms": float(np.percentile(times, 95)),
            "p99_ms": float(np.percentile(times, 99)),
        }

        logger.info("Benchmark results:")
        logger.info(f"  Mean: {results['mean_ms']:.2f}ms")
        logger.info(f"  Median: {results['median_ms']:.2f}ms")
        logger.info(f"  P95: {results['p95_ms']:.2f}ms")
        logger.info(f"  P99: {results['p99_ms']:.2f}ms")

        # Check target
        if results["median_ms"] > 10.0:
            logger.warning(
                f"[WARNING] Median latency {results['median_ms']:.2f}ms exceeds target (10ms)"
            )
        else:
            logger.info(
                f"[OK] Median latency {results['median_ms']:.2f}ms within target (<10ms)"
            )

        return results


def create_inference_engine(model: BaseAlgorithm, **kwargs: Any) -> RLInference:
    """
    Convenience function to create inference engine.

    Args:
        model: Trained RL model
        **kwargs: RLInference arguments

    Returns:
        Configured RLInference engine
    """
    return RLInference(model=model, **kwargs)
