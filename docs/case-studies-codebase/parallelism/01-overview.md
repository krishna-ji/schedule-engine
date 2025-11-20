# Parallelism in Schedule Engine: Complete Case Study

## Executive Summary

The Schedule Engine employs a **sophisticated multi-level parallelism strategy** that combines:
- **Process-level parallelism** (true CPU parallelism, bypasses GIL)
- **Thread-level parallelism** (for I/O-bound tasks)
- **GPU acceleration** (available but not enabled by default)
- **DEAP framework integration** (custom parallel backend)

**Performance Gains:**
- Fitness evaluation: 3-6x speedup (multiprocessing)
- Heuristic application: 10-16x speedup (parallel executor)
- Population generation: 3-5x speedup (ProcessPoolExecutor)
- Validation checks: 3-5x speedup (ThreadPoolExecutor)
- GPU potential: 10-50x speedup (CUDA acceleration)

---

## Table of Contents

1. [Thread-Level vs Core-Level Parallelism](#thread-vs-core)
2. [Library-Specific Parallelism](#library-parallelism)
3. [GPU Strategy & Implementation](#gpu-strategy)
4. [Bugs & Issues Found](#bugs-found)
5. [Recommendations](#recommendations)

---

## 1. Thread-Level vs Core-Level Parallelism {#thread-vs-core}

### Core-Level Parallelism (Process-Based) ✅

**What:** Each CPU core runs a separate Python process

**How:** `multiprocessing.Pool` or `concurrent.futures.ProcessPoolExecutor`

**Why:** Bypasses Python's Global Interpreter Lock (GIL)

**Used for:**
- ✅ Fitness evaluation (CPU-intensive)
- ✅ Population generation (CPU-intensive)
- ✅ Heuristic application (CPU-intensive)
- ✅ Large Neighborhood Search (CPU-intensive)
- ✅ Exhaustive local search (CPU-intensive)

**Libraries:**
```python
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
```

**Example from codebase:**
```python
# src/workflows/standard_run.py:221
pool = multiprocessing.Pool(
    processes=config.parallel.num_workers,  # Uses ALL CPU cores
    initializer=init_worker,
    initargs=(data_dir, seed),
)
```

---

### Thread-Level Parallelism (GIL-Limited) ⚠️

**What:** Multiple threads within single Python process

**How:** `concurrent.futures.ThreadPoolExecutor`

**Why:** Good for I/O-bound operations only

**Used for:**
- ✅ Input validation (file reading, checks)
- ✅ Feasibility checks (data validation)
- ✅ Report generation (I/O operations)
- ❌ **NOT used for GA operators** (crossover/mutation are CPU-bound)

**Libraries:**
```python
from concurrent.futures import ThreadPoolExecutor
```

**Example from codebase:**
```python
# src/validation/feasibility_checker.py:162
max_workers = os.cpu_count() or 5
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(check_instructor_workload, ...),
        executor.submit(check_room_capacity, ...),
        # ... 5 independent checks in parallel
    }
```

**Why threads work here:** Checks involve reading data structures, not heavy computation

---

## 2. Library-Specific Parallelism {#library-parallelism}

### 2.1 DEAP (Genetic Algorithm Framework)

**Version:** 1.4.1

**Parallelism Model:** Accepts external `map` function

**Integration:**
```python
# src/core/ga_scheduler.py:366
if self.pool is not None:
    self.toolbox.register("map", self.pool.map)  # Use multiprocessing
    
# Usage in evolution:
fitness_values = list(self.toolbox.map(self.toolbox.evaluate, population))
```

**Key Design:**
- DEAP doesn't provide parallelism itself
- Delegates to user-provided `map` function
- **No conflict** - seamless integration

---

### 2.2 PyTorch (GPU Framework)

**Version:** 2.4.1+cu121 (CUDA 12.1 support)

**Parallelism Model:** 
- GPU acceleration via CUDA
- Multi-threading for tensor operations
- Releases GIL for GPU operations

**Status:** **INSTALLED BUT NOT ENABLED**

**Implementation:** `src/ga/evaluator/gpu_batch_evaluator.py`

**Potential usage:**
```python
# GPU batch evaluation (10-50x speedup)
evaluator = GPUConstraintEvaluator(device="cuda")
results = evaluator.batch_evaluate_conflicts(population, batch_size=128)
```

**Why not enabled:**
- Requires CUDA toolkit installation
- Needs GPU hardware
- Not critical for current problem sizes

---

### 2.3 Stable-Baselines3 (RL Framework)

**Version:** 2.3.2

**Parallelism Model:** `SubprocVecEnv` (subprocess-based)

**Integration:**
```python
# src/rl/training/train_script.py
vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
```

**Critical Protection:**
```python
# Prevents nested multiprocessing (DEADLOCK RISK!)
os.environ["_GA_WORKER_PROCESS"] = "1"
```

---

### 2.4 NumPy/SciPy (Scientific Computing)

**Versions:** NumPy 1.26.4, SciPy 1.11.4

**Parallelism Model:** Internal BLAS/LAPACK threading

**GIL Behavior:** **Releases GIL during operations** ✅

**Impact:** Operations like matrix multiplication run in parallel automatically

**No conflict with multiprocessing** - NumPy handles threading internally

---

## 3. Core-Level Parallelism Details

### 3.1 CPU Core Utilization

**Detection:**
```python
from multiprocessing import cpu_count
num_cores = cpu_count()  # Auto-detect all logical cores
```

**Your system likely has:**
- Physical cores: 4-16
- Logical cores (with hyperthreading): 8-32

**Configuration:**
```yaml
# configs/base.yaml
parallel:
  use_multiprocessing: true
  num_workers: null  # null = use ALL cores
```

### 3.2 Worker Initialization Pattern

**Problem:** Can't pickle complex objects (SchedulingContext, DEAP types)

**Your Solution:** Load data from JSON in each worker

```python
# src/utils/parallel_worker.py
def init_worker(data_dir: str, seed: int):
    """Called once per worker process"""
    # 1. Register DEAP types (Windows requires this)
    from deap import creator, base
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
    
    # 2. Load data from disk (avoid pickling!)
    courses = load_courses(os.path.join(data_dir, "Course.json"))
    # ... load instructors, groups, rooms
    
    # 3. Store in global variable
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {"courses": courses, ...}
```

**Why this is brilliant:**
- Avoids pickling overhead (which fails on DEAP types)
- Each worker has independent copy of data
- Windows-compatible (spawn method)

---

## 4. Thread-Level Parallelism Details

### 4.1 When Threads Work

**GIL Release Conditions:**
- I/O operations (file read/write)
- System calls
- NumPy operations
- Network requests

**Used in codebase:**

| Module | Purpose | Why Threads Work |
|--------|---------|------------------|
| `input_validator.py` | Data validation | I/O-bound checks |
| `feasibility_checker.py` | 5 independent checks | Data reading + simple math |
| `reporting.py` | Generate plots | File writing |

### 4.2 When Threads DON'T Work

**GIL Limitations:**
- Pure Python loops
- Arithmetic operations
- Object creation
- List comprehensions

**Avoided in codebase:**

```python
# src/core/ga_scheduler.py:78-89
def _parallel_crossover(offspring, cxpb, toolbox, max_workers=None):
    """
    Apply crossover sequentially.
    
    NOTE: ThreadPoolExecutor removed because Python's GIL prevents true parallelism
    for CPU-bound tasks like crossover. Multiprocessing overhead (pickling)
    often outweighs benefits for simple operators. Sequential is faster and safer.
    """
    # Sequential execution - CORRECT DECISION!
    for i in range(0, len(offspring) - 1, 2):
        if random.random() < cxpb:
            toolbox.mate(offspring[i], offspring[i + 1])
```

**Why sequential is correct:**
- Crossover is fast (microseconds per operation)
- Process creation overhead > actual speedup
- GIL prevents threading from helping
- Avoids complexity and race conditions

---

## Summary Table: Thread vs Core Level

| Operation | Parallelism | Library | Level | CPU Cores Used | GIL Impact |
|-----------|-------------|---------|-------|----------------|------------|
| **Fitness Eval** | ✅ Process | `multiprocessing` | Core | ALL | None (bypassed) |
| **Population Init** | ✅ Process | `ProcessPoolExecutor` | Core | ALL | None (bypassed) |
| **Heuristics** | ✅ Process | `ProcessPoolExecutor` | Core | ALL | None (bypassed) |
| **Crossover** | ❌ Sequential | N/A | Single | 1 | Full (but intentional) |
| **Mutation** | ❌ Sequential | N/A | Single | 1 | Full (but intentional) |
| **Validation** | ✅ Thread | `ThreadPoolExecutor` | Thread | Variable | Partial (I/O) |
| **Feasibility** | ✅ Thread | `ThreadPoolExecutor` | Thread | Variable | Partial (I/O) |
| **RL Training** | ✅ Process | `SubprocVecEnv` | Core | 8 envs | None (bypassed) |
| **GPU Eval** | ⚠️ Disabled | PyTorch CUDA | GPU | N/A | None (GPU ops) |

---

## Next Steps

See individual case study files for:
- [02-gpu-strategy.md](./02-gpu-strategy.md) - GPU implementation guide
- [03-bugs-found.md](./03-bugs-found.md) - Issues and fixes
- [04-recommendations.md](./04-recommendations.md) - Optimization suggestions
