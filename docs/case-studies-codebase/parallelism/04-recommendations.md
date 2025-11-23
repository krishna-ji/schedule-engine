# Recommendations for Parallelism Optimization

## Executive Summary

Your parallelism strategy is **excellent** with room for **targeted improvements**. This document provides actionable recommendations ranked by impact.

---

## High-Impact Recommendations 

### 1. Enable GPU for Production Runs

**Impact:** 5-10x speedup (6-10 hours → 1-2 hours for 2000 generations)

**Effort:** Medium (2-4 hours implementation)

**Implementation:**
```bash
# Step 1: Verify GPU
uv run diagnose-gpu

# Step 2: Create GPU config
# Create configs/gpu-prod.yaml inheriting from prod.yaml
```

```yaml
# configs/gpu-prod.yaml
# Inherits from: prod.yaml
_inherit: prod.yaml

gpu:
  enabled: true
  device: auto  # cuda if available, else cpu
  batch_size: 128
  min_population_for_gpu: 100

parallel:
  use_multiprocessing: false  # GPU replaces multiprocessing
```

**Code changes:**
- Modify `GAScheduler.__init__()` to accept GPU config
- Add GPU evaluation branch in evolution loop
- Add hybrid CPU/GPU selector based on batch size

**Testing:**
```bash
# Benchmark GPU vs CPU
uv run benchmark-gpu --generations 50

# Full production run with GPU
python main.py --config configs/gpu-prod.yaml
```

**Expected results:**
- Small populations (< 100): CPU still faster
- Large populations (200+): GPU 10-20x faster
- **Total time: 34 hours → 2-3 hours** (with GPU)

---

### 2. Fix Double Executor in ParallelExecutor

**Impact:** Eliminate wasted resources (2x executor creation per call)

**Effort:** Low (15 minutes)

**Current code:**
```python
# src/heuristics/parallel_executor.py:95-160
# Creates executor TWICE - first with as_completed, then in order
```

**Fixed code:**
```python
def apply_parallel(
    self,
    heuristic_func: Callable,
    individuals: List,
    context: Any,
    chunk_size: int = None,
) -> List:
    """Apply heuristic to population in parallel (ORDER-PRESERVING)."""
    if len(individuals) == 0:
        return []
    
    # For small populations, don't parallelize
    if len(individuals) < self.max_workers:
        return [heuristic_func(ind, context) for ind in individuals]
    
    # Determine chunk size
    if chunk_size is None:
        chunk_size = max(1, len(individuals) // self.max_workers)
    
    # Split into chunks
    chunks = [
        individuals[i : i + chunk_size]
        for i in range(0, len(individuals), chunk_size)
    ]
    
    # Select executor type
    ExecutorClass = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
    
    # Prepare executor kwargs
    executor_kwargs = {"max_workers": self.max_workers}
    
    if not self.use_threads:
        data_dir = getattr(context, 'data_dir', 'data')
        executor_kwargs["initializer"] = init_worker
        executor_kwargs["initargs"] = (data_dir, random.randint(0, 10000))
    
    try:
        with ExecutorClass(**executor_kwargs) as executor:
            submit_context = context if self.use_threads else None
            
            # Submit all chunks
            futures = [
                executor.submit(
                    self._apply_to_chunk, heuristic_func, chunk, submit_context
                )
                for chunk in chunks
            ]
            
            # Collect results IN ORDER (use enumerate, not futures.index)
            results = []
            for i, future in enumerate(futures):
                try:
                    results.extend(future.result(timeout=60))  # Add timeout
                except concurrent.futures.TimeoutError:
                    logger.error(f"Chunk {i} timed out, falling back to sequential")
                    results.extend([heuristic_func(ind, context) for ind in chunks[i]])
                except Exception as e:
                    logger.error(f"Chunk {i} failed: {e}, falling back to sequential")
                    results.extend([heuristic_func(ind, context) for ind in chunks[i]])
            
            return results
    
    except Exception as e:
        logger.error(f"Parallel execution failed: {e}, falling back to sequential")
        return [heuristic_func(ind, context) for ind in individuals]
```

**Testing:**
```python
# test/unit/test_parallel_executor.py
def test_order_preservation():
    """Verify parallel executor preserves order."""
    executor = get_parallel_executor()
    individuals = [MockIndividual(i) for i in range(100)]
    
    results = executor.apply_parallel(
        heuristic_func=lambda ind, ctx: ind,
        individuals=individuals,
        context=test_context
    )
    
    # Verify order
    assert [r.id for r in results] == list(range(100))
```

---

### 3. Add GPU Batch Size Auto-Tuning

**Impact:** Prevent OOM errors, optimize memory usage

**Effort:** Medium (1 hour)

**Implementation:**
```python
# src/ga/evaluator/gpu_batch_evaluator.py

class GPUConstraintEvaluator:
    def __init__(self, device="cuda"):
        # ... existing code ...
        
        # Auto-tune batch size based on GPU memory
        if self.enabled:
            self.optimal_batch_size = self._auto_tune_batch_size()
        else:
            self.optimal_batch_size = 128
    
    def _auto_tune_batch_size(self) -> int:
        """Automatically determine optimal batch size."""
        if not torch.cuda.is_available():
            return 128
        
        # Get GPU memory
        gpu_props = torch.cuda.get_device_properties(0)
        total_mem = gpu_props.total_memory / (1024**3)  # Convert to GB
        
        # Estimate memory per individual
        # 100 genes * 5 features * 4 bytes = 2KB per individual
        mem_per_individual = 0.002  # MB
        
        # Use 80% of GPU memory for safety
        usable_mem = (total_mem * 1024 * 0.8)  # MB
        
        # Calculate batch size
        batch_size = int(usable_mem / mem_per_individual)
        
        # Clamp to reasonable range [32, 512]
        batch_size = max(32, min(512, batch_size))
        
        logger.info(
            f"Auto-tuned GPU batch size: {batch_size} "
            f"(GPU: {total_mem:.1f}GB)"
        )
        
        return batch_size
    
    def batch_evaluate_conflicts(
        self, population: List[List[SessionGene]], batch_size: int = None
    ) -> List[Tuple[int, int]]:
        """Evaluate constraints for entire population on GPU."""
        if not self.enabled:
            return [(0, 0) for _ in population]
        
        # Use auto-tuned batch size if not specified
        if batch_size is None:
            batch_size = self.optimal_batch_size
        
        # ... rest of method
```

---

## Medium-Impact Recommendations 

### 4. Implement Hybrid CPU/GPU Evaluator

**Impact:** Best of both worlds (GPU for large batches, CPU for small)

**Effort:** Medium (2 hours)

**Implementation:**
```python
# src/ga/evaluator/hybrid_evaluator.py

class HybridEvaluator:
    """Automatically choose CPU or GPU based on batch size."""
    
    def __init__(self, cpu_threshold=100):
        """
        Args:
            cpu_threshold: Use CPU for batches smaller than this
        """
        self.cpu_threshold = cpu_threshold
        
        # Try to initialize GPU
        try:
            self.gpu_evaluator = get_gpu_evaluator()
            self.gpu_available = self.gpu_evaluator.is_available()
        except Exception as e:
            logger.warning(f"GPU initialization failed: {e}")
            self.gpu_available = False
        
        logger.info(
            f"HybridEvaluator initialized (GPU: {self.gpu_available}, "
            f"threshold: {cpu_threshold})"
        )
    
    def evaluate_population(
        self, population: List, evaluate_func: Callable
    ) -> List[Tuple[float, float]]:
        """Evaluate population using optimal backend."""
        batch_size = len(population)
        
        # Decision logic
        if batch_size >= self.cpu_threshold and self.gpu_available:
            # Large batch: use GPU
            logger.debug(f"Using GPU for batch of {batch_size}")
            return self.gpu_evaluator.batch_evaluate_conflicts(population)
        else:
            # Small batch: use CPU multiprocessing
            logger.debug(f"Using CPU for batch of {batch_size}")
            # Delegate to existing CPU evaluation
            return [evaluate_func(ind) for ind in population]

# Usage in GAScheduler
self.hybrid_evaluator = HybridEvaluator(cpu_threshold=100)
fitness_values = self.hybrid_evaluator.evaluate_population(invalid, self.toolbox.evaluate)
```

**Benefits:**
- No manual configuration needed
- Optimal performance for all batch sizes
- Graceful fallback if GPU unavailable

---

### 5. Add Parallel Operation Timeouts

**Impact:** Prevent hanging on worker deadlocks

**Effort:** Low (30 minutes)

**Implementation:**
```python
# src/heuristics/parallel_executor.py

import concurrent.futures

OPERATION_TIMEOUT = 60  # seconds per chunk

for i, future in enumerate(futures):
    try:
        result = future.result(timeout=OPERATION_TIMEOUT)
        results.extend(result)
    except concurrent.futures.TimeoutError:
        logger.error(
            f"Chunk {i} timed out after {OPERATION_TIMEOUT}s, "
            f"falling back to sequential"
        )
        # Process chunk sequentially as fallback
        results.extend([heuristic_func(ind, context) for ind in chunks[i]])
```

---

### 6. Standardize CPU Count Detection

**Impact:** Code consistency, better error handling

**Effort:** Low (30 minutes)

**Current inconsistency:**
```python
# Multiple methods used across codebase
from multiprocessing import cpu_count  # Method 1
import os; os.cpu_count()              # Method 2
import multiprocessing as mp; mp.cpu_count()  # Method 3
```

**Standardized approach:**
```python
# src/utils/system_info.py (new file)

import os
import logging

DEFAULT_CPU_COUNT = 8  # Fallback if detection fails

logger = logging.getLogger(__name__)

def get_cpu_count() -> int:
    """
    Get number of available CPU cores with fallback.
    
    Returns:
        Number of logical CPU cores (includes hyperthreading)
    """
    try:
        count = os.cpu_count()
        if count is None or count < 1:
            logger.warning(
                f"CPU count detection returned {count}, "
                f"using default {DEFAULT_CPU_COUNT}"
            )
            return DEFAULT_CPU_COUNT
        return count
    except Exception as e:
        logger.error(f"Failed to detect CPU count: {e}, using default")
        return DEFAULT_CPU_COUNT

# Usage everywhere:
from src.utils.system_info import get_cpu_count
num_workers = get_cpu_count()
```

**Files to update:**
- `src/ga/population.py`
- `src/ga/hybrid_population.py`
- `src/heuristics/parallel_executor.py`
- `src/workflows/standard_run.py`
- `src/validation/feasibility_checker.py`

---

## Low-Impact Recommendations 

### 7. Optimize GPU Tensor Creation

**Impact:** Minor GPU performance improvement (~5-10%)

**Effort:** Low (15 minutes)

**Current (inefficient):**
```python
# Creates tensor on GPU immediately
tensor = torch.zeros((batch, genes, 5), device=self.device, dtype=torch.long)

# Populate on GPU (many small transfers)
for i, individual in enumerate(batch):
    for j, gene in enumerate(individual):
        tensor[i, j, 0] = gene.quanta[0]
        # ... more assignments
```

**Optimized (batch transfer):**
```python
# Create on CPU
tensor = torch.zeros((batch, genes, 5), dtype=torch.long)

# Populate on CPU (fast)
for i, individual in enumerate(batch):
    for j, gene in enumerate(individual):
        tensor[i, j, 0] = gene.quanta[0]
        # ... more assignments

# Single batch transfer to GPU
tensor = tensor.to(self.device, non_blocking=True)
```

---

### 8. Add Performance Profiling

**Impact:** Identify bottlenecks for future optimization

**Effort:** Low (30 minutes)

**Implementation:**
```python
# src/utils/profiler.py (new file)

import time
import logging
from contextlib import contextmanager
from typing import Dict

logger = logging.getLogger(__name__)

_TIMINGS: Dict[str, list] = {}

@contextmanager
def profile_section(name: str):
    """Context manager for profiling code sections."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if name not in _TIMINGS:
            _TIMINGS[name] = []
        _TIMINGS[name].append(elapsed)

def print_profile_summary():
    """Print summary of profiled sections."""
    if not _TIMINGS:
        return
    
    logger.info("=== Performance Profile ===")
    for name, times in sorted(_TIMINGS.items()):
        total = sum(times)
        avg = total / len(times)
        logger.info(
            f"{name:30s}: {total:8.2f}s total, "
            f"{avg:8.4f}s avg ({len(times)} calls)"
        )

# Usage in GAScheduler
from src.utils.profiler import profile_section, print_profile_summary

def _evolve_generation(self, gen: int):
    with profile_section("selection"):
        offspring = self.toolbox.select(self.population, len(self.population))
    
    with profile_section("crossover"):
        offspring = _parallel_crossover(offspring, cxpb, self.toolbox)
    
    with profile_section("mutation"):
        offspring = _parallel_mutation(offspring, mutpb, self.toolbox)
    
    with profile_section("evaluation"):
        fitness_values = list(self.toolbox.map(self.toolbox.evaluate, invalid))

# At end of run
print_profile_summary()
```

**Output example:**
```
=== Performance Profile ===
selection                     :     2.34s total,   0.0012s avg (2000 calls)
crossover                     :    12.56s total,   0.0063s avg (2000 calls)
mutation                      :     8.91s total,   0.0045s avg (2000 calls)
evaluation                    :   450.23s total,   0.2251s avg (2000 calls)  <-- BOTTLENECK
repair                        :    67.89s total,   0.0339s avg (2000 calls)
```

---

### 9. Add Unit Tests for Parallel Operations

**Impact:** Catch race conditions, ensure correctness

**Effort:** Medium (2 hours)

**Test cases:**
```python
# test/unit/test_parallel_operations.py

import pytest
from src.heuristics.parallel_executor import get_parallel_executor

def test_order_preservation():
    """Verify parallel executor preserves individual order."""
    executor = get_parallel_executor()
    individuals = [MockIndividual(i) for i in range(100)]
    
    def identity_heuristic(ind, ctx):
        return ind
    
    results = executor.apply_parallel(
        heuristic_func=identity_heuristic,
        individuals=individuals,
        context=test_context
    )
    
    assert [r.id for r in results] == list(range(100))

def test_error_handling():
    """Verify parallel executor handles worker failures."""
    executor = get_parallel_executor()
    
    def failing_heuristic(ind, ctx):
        if ind.id == 50:
            raise ValueError("Intentional failure")
        return ind
    
    individuals = [MockIndividual(i) for i in range(100)]
    
    # Should not crash, should fallback to sequential for failed chunk
    results = executor.apply_parallel(
        heuristic_func=failing_heuristic,
        individuals=individuals,
        context=test_context
    )
    
    assert len(results) == 100

def test_timeout_handling():
    """Verify parallel executor handles timeouts."""
    executor = get_parallel_executor()
    
    def slow_heuristic(ind, ctx):
        import time
        if ind.id == 50:
            time.sleep(100)  # Intentional timeout
        return ind
    
    individuals = [MockIndividual(i) for i in range(100)]
    
    # Should timeout and fallback, not hang forever
    results = executor.apply_parallel(
        heuristic_func=slow_heuristic,
        individuals=individuals,
        context=test_context
    )
    
    assert len(results) == 100
```

---

## Summary Roadmap

### Phase 1: Quick Wins (1 day)
- [ ] Fix double executor context 
- [ ] Add operation timeouts 
- [ ] Standardize CPU count detection 
- [ ] Optimize GPU tensor creation 

### Phase 2: GPU Integration (2-3 days)
- [ ] Add GPU batch size auto-tuning 
- [ ] Implement hybrid CPU/GPU evaluator 
- [ ] Test GPU on small dataset 
- [ ] Benchmark GPU vs CPU 

### Phase 3: Production Deployment (1 week)
- [ ] Enable GPU for production runs 
- [ ] Run full benchmarks (2000 gens) 
- [ ] Document GPU setup guide 
- [ ] Add performance profiling 

### Phase 4: Quality Assurance (ongoing)
- [ ] Add unit tests for parallel operations 
- [ ] Add integration tests 
- [ ] Performance regression testing 
- [ ] Monitor GPU memory usage 

---

## Expected Performance Improvements

| Change | Time Saved | Effort | Priority |
|--------|------------|--------|----------|
| Enable GPU (prod) | 5-9 hours/run | Medium | **HIGH** |
| Fix double executor | 1-2 sec/run | Low | **HIGH** |
| GPU batch auto-tune | Prevent crashes | Medium | **HIGH** |
| Hybrid evaluator | Best of both | Medium | Medium |
| Add timeouts | Prevent hangs | Low | Medium |
| Standardize CPU detect | Code quality | Low | Low |
| Optimize GPU tensors | ~5-10% GPU | Low | Low |

**Total potential speedup: 5-10x for production runs** (34 hours → 3-6 hours)

---

## Conclusion

Your parallelism implementation is **production-ready** with excellent fundamentals. The recommended improvements focus on:

1. **GPU enablement** (biggest impact)
2. **Code cleanup** (fix double executor)
3. **Robustness** (timeouts, auto-tuning)
4. **Quality** (tests, profiling)

**Next steps:**
1. Implement Phase 1 fixes (1 day)
2. Test GPU integration (2-3 days)
3. Run full production benchmark
4. Document results
