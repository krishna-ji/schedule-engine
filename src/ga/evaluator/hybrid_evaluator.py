"""Hybrid CPU/GPU evaluator that automatically selects best strategy.

Intelligently chooses between GPU (for large batches) and CPU multiprocessing
(for small batches) to maximize performance across all population sizes.
"""

from typing import List, Tuple, Callable
from src.ga.sessiongene import SessionGene
from src.ga.evaluator.gpu_batch_evaluator import get_gpu_evaluator
import logging

logger = logging.getLogger(__name__)


class HybridEvaluator:
    """Automatically choose CPU or GPU based on batch size and availability.

    This evaluator provides the best of both worlds:
    - GPU acceleration for large populations (10-50x speedup)
    - CPU multiprocessing for small populations (avoids GPU overhead)
    - Graceful fallback if GPU unavailable
    """

    def __init__(
        self,
        gpu_threshold: int = 100,
        gpu_device: str = "auto",
        fallback_to_cpu: bool = True,
    ):
        """Initialize hybrid evaluator.

        Args:
            gpu_threshold: Minimum population size to use GPU (default: 100)
            gpu_device: GPU device ('auto', 'cuda', or 'cpu')
            fallback_to_cpu: Fall back to CPU if GPU fails
        """
        self.gpu_threshold = gpu_threshold
        self.fallback_to_cpu = fallback_to_cpu

        # Try to initialize GPU evaluator
        try:
            self.gpu_evaluator = get_gpu_evaluator(device=gpu_device)
            self.gpu_available = self.gpu_evaluator.is_available()
        except Exception as e:
            logger.warning(f"Failed to initialize GPU evaluator: {e}")
            self.gpu_evaluator = None
            self.gpu_available = False

        if self.gpu_available:
            logger.info(
                f"✓ Hybrid Evaluator: GPU available (threshold={gpu_threshold})"
            )
        else:
            logger.info("Hybrid Evaluator: CPU-only mode")

    def evaluate_population(
        self,
        population: List,
        cpu_evaluate_func: Callable,
        batch_size: int = None,
    ) -> List[Tuple[int, int]]:
        """Evaluate population using optimal strategy.

        Args:
            population: List of individuals to evaluate
            cpu_evaluate_func: CPU evaluation function (fallback)
            batch_size: GPU batch size (None = auto)

        Returns:
            List of (hard_violations, soft_violations) tuples
        """
        pop_size = len(population)

        # Decision logic: GPU vs CPU
        use_gpu = (
            self.gpu_available
            and pop_size >= self.gpu_threshold
            and self.gpu_evaluator is not None
        )

        if use_gpu:
            try:
                logger.debug(f"Using GPU for {pop_size} individuals")
                return self.gpu_evaluator.batch_evaluate_conflicts(
                    population, batch_size=batch_size
                )
            except Exception as e:
                logger.error(f"GPU evaluation failed: {e}")
                if not self.fallback_to_cpu:
                    raise

                logger.info("Falling back to CPU evaluation")
                # Fall through to CPU evaluation

        # CPU evaluation (multiprocessing handled by caller)
        logger.debug(f"Using CPU for {pop_size} individuals")
        return [cpu_evaluate_func(ind) for ind in population]

    def is_gpu_available(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_available

    def get_recommended_strategy(self, population_size: int) -> str:
        """Get recommended evaluation strategy for given population size.

        Args:
            population_size: Size of population

        Returns:
            'gpu' or 'cpu'
        """
        if not self.gpu_available:
            return "cpu"

        return "gpu" if population_size >= self.gpu_threshold else "cpu"


# Singleton instance
_hybrid_evaluator = None


def get_hybrid_evaluator(
    gpu_threshold: int = 100, gpu_device: str = "auto", fallback_to_cpu: bool = True
) -> HybridEvaluator:
    """Get or create hybrid evaluator singleton.

    Args:
        gpu_threshold: Minimum population size to use GPU
        gpu_device: GPU device selection
        fallback_to_cpu: Fall back to CPU if GPU fails

    Returns:
        HybridEvaluator instance
    """
    global _hybrid_evaluator
    if _hybrid_evaluator is None:
        _hybrid_evaluator = HybridEvaluator(
            gpu_threshold=gpu_threshold,
            gpu_device=gpu_device,
            fallback_to_cpu=fallback_to_cpu,
        )
    return _hybrid_evaluator
