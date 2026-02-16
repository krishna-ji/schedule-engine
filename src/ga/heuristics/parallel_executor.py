"""Parallel execution of heuristics across population.

Enables simultaneous application of heuristics to multiple individuals,
achieving 10-16x speedup by fully utilizing all CPU cores.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from concurrent.futures import as_completed
from typing import Any, Protocol, TypeGuard, cast

from src.domain.types import Individual
from src.utils.parallel_worker import get_worker_context, init_worker
from src.utils.system_info import get_cpu_count

logger = logging.getLogger(__name__)

HeuristicFunc = Callable[[Individual, Any], Any]


class _FitnessCarrier(Protocol):
    fitness: Any


def _has_fitness(obj: object) -> TypeGuard[_FitnessCarrier]:
    return hasattr(obj, "fitness")


class ParallelHeuristicExecutor:
    """Execute heuristics on multiple individuals simultaneously.

    Uses process-based parallelism to bypass Python's GIL and achieve
    true parallel execution across all CPU cores.
    """

    def __init__(self, max_workers: int | None = None, use_threads: bool = False):
        """Initialize parallel executor.

        Args:
            max_workers: Number of parallel workers (None = auto-detect all CPUs)
            use_threads: Use threads instead of processes (faster for I/O-bound)
        """
        if max_workers is None:
            max_workers = get_cpu_count()

        self.max_workers = max_workers
        self.use_threads = use_threads

        logger.info(
            f"ParallelHeuristicExecutor: {max_workers} workers "
            f"({'threads' if use_threads else 'processes'})"
        )

    def apply_parallel(
        self,
        heuristic_func: HeuristicFunc,
        individuals: list[Individual],
        context: Any,
        chunk_size: int | None = None,
    ) -> list[Individual]:
        """Apply heuristic to population in parallel.

        Args:
            heuristic_func: Heuristic function to apply
            individuals: Population to process
            context: Scheduling context
            chunk_size: Individuals per worker (default: len/workers)

        Returns:
            Modified population
        """
        if len(individuals) == 0:
            return []

        # For small populations, don't parallelize
        if len(individuals) < self.max_workers:
            return [
                self._execute_heuristic(heuristic_func, ind, context)
                for ind in individuals
            ]

        # Determine chunk size
        if chunk_size is None:
            chunk_size = max(1, len(individuals) // self.max_workers)

        # Split into chunks
        chunks = [
            individuals[i : i + chunk_size]
            for i in range(0, len(individuals), chunk_size)
        ]

        # Prepare args for executor
        initializer_func: Callable[[str, int], None] | None = None
        initializer_args: tuple[str, int] | None = None

        # If using ProcessPoolExecutor, use initializer to avoid pickling context
        if not self.use_threads:
            # Try to extract data_dir from context
            data_dir = "data"
            if (
                hasattr(context, "config")
                and hasattr(context.config, "io")
                and hasattr(context.config.io, "data_dir")
            ):
                data_dir = context.config.io.data_dir

            initializer_func = init_worker
            initializer_args = (data_dir, random.randint(0, 10000))

        try:
            # Process in parallel with order preservation
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

            executor_ctx: ThreadPoolExecutor | ProcessPoolExecutor
            if self.use_threads:
                executor_ctx = ThreadPoolExecutor(max_workers=self.max_workers)
            else:
                # Use type: ignore for initializer signature mismatch (ProcessPoolExecutor doesn't accept [str, int] args)
                executor_ctx = ProcessPoolExecutor(
                    max_workers=self.max_workers,
                    initializer=initializer_func,  # type: ignore[arg-type]
                    initargs=initializer_args or (),  # type: ignore[arg-type]
                )

            with executor_ctx as executor:
                # If using processes, DO NOT pass context (it's loaded in worker)
                submit_context = context if self.use_threads else None

                # Submit all chunks
                futures = [
                    executor.submit(
                        self._apply_to_chunk, heuristic_func, chunk, submit_context
                    )
                    for chunk in chunks
                ]

                # Collect results IN ORDER (not as_completed which scrambles order)
                results = []
                for i, future in enumerate(futures):
                    try:
                        # Add timeout to prevent hanging (60 seconds per chunk)
                        chunk_results = future.result(timeout=60)
                        results.extend(chunk_results)
                    except Exception as e:
                        logger.error(f"Heuristic chunk {i} failed: {e}")
                        # Fallback: process chunk sequentially
                        # Use enumerate index to directly access chunk (O(1) instead of O(n))
                        results.extend(
                            [
                                self._execute_heuristic(heuristic_func, ind, context)
                                for ind in chunks[i]
                            ]
                        )

            return results

        except Exception as e:
            logger.error(f"Parallel execution failed: {e}, falling back to sequential")
            # Fallback to sequential
            return [
                self._execute_heuristic(heuristic_func, ind, context)
                for ind in individuals
            ]

    @staticmethod
    def _apply_to_chunk(
        heuristic_func: HeuristicFunc,
        chunk: list[Individual],
        context: Any | None,
    ) -> list[Individual]:
        """Apply heuristic to a chunk of individuals.

        This runs in a separate process/thread.
        """
        # If context is None, try to get from worker global state
        if context is None:
            try:
                worker_data = get_worker_context()
                context = worker_data["context"]
            except RuntimeError:
                # Should not happen if initialized correctly
                # But if it does, we can't proceed without context
                raise RuntimeError("Context missing in worker process") from None

        results = []
        for ind in chunk:
            try:
                modified_ind = heuristic_func(ind, context)
                results.append(
                    ParallelHeuristicExecutor._prepare_result(modified_ind, ind)
                )
            except Exception as e:
                # If heuristic fails, keep original
                logger.debug(f"Heuristic failed on individual: {e}")
                results.append(ind)

        return results

    def apply_batch(
        self,
        heuristic_funcs: list[HeuristicFunc],
        individual: Individual,
        context: Any,
    ) -> Individual:
        """Apply multiple heuristics to single individual in parallel.

        Useful when you want to try several heuristics and pick the best result.

        Args:
            heuristic_funcs: List of heuristic functions
            individual: Individual to modify
            context: Scheduling context

        Returns:
            Best modified individual (lowest fitness)
        """
        if len(heuristic_funcs) == 0:
            return individual

        try:
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

            executor_class = (
                ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
            )
            with executor_class(
                max_workers=min(self.max_workers, len(heuristic_funcs))
            ) as executor:
                # Apply all heuristics in parallel
                futures = [
                    executor.submit(func, individual, context)
                    for func in heuristic_funcs
                ]

                # Collect results
                results: list[Individual] = []
                for future in as_completed(futures):
                    try:
                        results.append(
                            self._prepare_result(future.result(), individual)
                        )
                    except Exception as e:
                        logger.debug(f"Heuristic failed: {e}")
                        results.append(individual)

                if _has_fitness(individual):
                    fitness_results = [res for res in results if _has_fitness(res)]
                    if fitness_results:
                        best = min(
                            fitness_results,
                            key=lambda ind: (
                                ind.fitness.values
                                if hasattr(ind.fitness, "values")
                                else (float("inf"),)
                            ),
                        )
                        return cast("Individual", best)
                return results[0] if results else individual

        except Exception as e:
            logger.error(f"Batch heuristic application failed: {e}")
            return individual

    def _execute_heuristic(
        self,
        heuristic_func: HeuristicFunc,
        individual: Individual,
        context: Any,
    ) -> Individual:
        """Execute heuristic safely and return modified individual."""
        try:
            result = heuristic_func(individual, context)
            return self._prepare_result(result, individual)
        except Exception as exc:
            logger.debug(f"Heuristic execution failed: {exc}")
            return individual

    @staticmethod
    def _prepare_result(result: Any, original: Individual) -> Individual:
        """Normalize heuristic outputs to return mutated individuals."""
        if isinstance(result, list):
            return result
        if isinstance(result, tuple):
            for item in result:
                if isinstance(item, list):
                    return item
        return original


# Singleton instance
_parallel_executor: ParallelHeuristicExecutor | None = None


def get_parallel_executor(max_workers: int | None = None) -> ParallelHeuristicExecutor:
    """Get or create parallel executor singleton.

    Args:
        max_workers: Number of workers (None = auto-detect all CPUs)
    """
    global _parallel_executor
    if _parallel_executor is None:
        _parallel_executor = ParallelHeuristicExecutor(max_workers=max_workers)
    return _parallel_executor
