# GPU Acceleration Strategy

## Current Status

**GPU Support:** ✅ Implemented, ❌ Not Enabled

**Implementation:** `src/ga/evaluator/gpu_batch_evaluator.py`

**PyTorch:** 2.4.1+cu121 (CUDA 12.1 support installed)

**Device Config:** CPU-only by default

---

## Where to Use GPU?

### 1. Fitness Evaluation (HIGHEST PRIORITY) 🎯

**File:** `src/ga/evaluator/gpu_batch_evaluator.py`

**Current Implementation:**
```python
class GPUConstraintEvaluator:
    """Evaluate constraints on GPU for 10-50x speedup."""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.enabled = self.device.type == "cuda"
    
    def batch_evaluate_conflicts(
        self, population: List[List[SessionGene]], batch_size: int = 128
    ) -> List[Tuple[int, int]]:
        """Vectorized constraint checking on GPU."""
```

**How it works:**
1. Convert population to GPU tensors [batch_size, genes, features]
2. Vectorized overlap detection using PyTorch operations
3. Results transferred back to CPU

**Expected speedup:** 10-50x for large populations (500+ individuals)

**Best for:**
- Large population sizes (200-1000)
- Long generation runs (2000+ generations)
- Complex constraint checking

---

### 2. RL Training (SECONDARY) 🤖

**File:** `src/rl/training/train_script.py`

**Current Implementation:**
```python
# Device setting: always CPU-only execution
device = "cpu"  # Hardcoded!
```

**Why disabled:**
```python
# CRITICAL: Inside SubprocVecEnv worker processes, 
# we CANNOT use nested multiprocessing
```

**Potential usage:**
```python
# For single-environment training
model = PPO(
    policy="MlpPolicy",
    env=env,
    device="cuda",  # GPU for neural network
    # ... other params
)
```

**Expected speedup:** 2-5x for neural network forward/backward passes

**Best for:**
- Long training runs (100k+ timesteps)
- Large neural networks
- Single environment (no SubprocVecEnv)

---

### 3. Batch Heuristic Evaluation (FUTURE)

**Not yet implemented**

**Potential:**
```python
class GPUHeuristicEvaluator:
    """Evaluate multiple heuristic applications on GPU."""
    
    def batch_apply_heuristics(
        self,
        population: List,
        heuristics: List[Callable],
        batch_size: int = 64
    ) -> List:
        """Try multiple heuristics in parallel on GPU."""
```

**Use case:**
- RL action selection (try all 19 heuristics on GPU)
- Local search (evaluate all neighbors simultaneously)

---

## How to Enable GPU

### Step 1: Check GPU Availability

```bash
# Use built-in diagnostic
uv run diagnose-gpu
```

**Expected output:**
```
CUDA Available: True
GPU Count: 1
GPU 0: NVIDIA GeForce RTX 3060 (12GB)
PyTorch Version: 2.4.1+cu121
```

### Step 2: Update Configuration

**Option A: Enable GPU for fitness evaluation**

Create `configs/gpu.yaml`:
```yaml
# GPU-accelerated fitness evaluation
gpu:
  enabled: true
  device: cuda  # or "cuda:0" for specific GPU
  batch_size: 128  # Tune based on GPU memory
  fallback_to_cpu: true  # Fallback if GPU unavailable

parallel:
  use_multiprocessing: false  # Disable CPU multiprocessing (GPU replaces it)
  num_workers: 1
```

**Option B: Enable GPU for RL training**

Modify `configs/base.yaml`:
```yaml
rl:
  agent:
    device: cuda  # Change from "cpu"
  
  training:
    # Reduce number of parallel environments for GPU mode
    n_envs: 1  # Single environment to avoid multiprocessing conflict
```

### Step 3: Modify GA Scheduler

**File:** `src/core/ga_scheduler.py`

**Current (CPU multiprocessing):**
```python
# Parallel mode: use worker evaluation
if self.pool is not None:
    self.toolbox.register("map", self.pool.map)
    self.toolbox.register("evaluate", _worker_evaluate)
```

**Proposed (GPU batch evaluation):**
```python
# GPU mode: use batch evaluator
if config.gpu.enabled and torch.cuda.is_available():
    from src.ga.evaluator.gpu_batch_evaluator import get_gpu_evaluator
    
    self.gpu_evaluator = get_gpu_evaluator(device=config.gpu.device)
    self.toolbox.register("evaluate_batch", self._evaluate_batch_gpu)
    logger.info(f"GPU evaluation enabled: {self.gpu_evaluator.device}")
else:
    # Fallback to CPU multiprocessing
    if self.pool is not None:
        self.toolbox.register("map", self.pool.map)
```

**Add GPU evaluation method:**
```python
def _evaluate_batch_gpu(self, population):
    """Evaluate entire population on GPU."""
    return self.gpu_evaluator.batch_evaluate_conflicts(
        population, 
        batch_size=self.config.gpu.batch_size
    )
```

### Step 4: Update Evolution Loop

**File:** `src/core/ga_scheduler.py` (line ~1100)

**Current:**
```python
# Evaluate invalid individuals
invalid = [ind for ind in offspring if not ind.fitness.valid]
if invalid:
    fitness_values = list(self.toolbox.map(self.toolbox.evaluate, invalid))
    for ind, fit in zip(invalid, fitness_values):
        ind.fitness.values = fit
```

**Proposed (GPU-aware):**
```python
# Evaluate invalid individuals
invalid = [ind for ind in offspring if not ind.fitness.valid]
if invalid:
    # Use GPU batch evaluator if available
    if hasattr(self, 'gpu_evaluator') and self.gpu_evaluator.is_available():
        fitness_values = self._evaluate_batch_gpu(invalid)
    else:
        # Fallback to CPU multiprocessing
        fitness_values = list(self.toolbox.map(self.toolbox.evaluate, invalid))
    
    for ind, fit in zip(invalid, fitness_values):
        ind.fitness.values = fit
```

---

## Best Strategy: Hybrid CPU + GPU

### Optimal Configuration

**For large populations (200+):**
```yaml
gpu:
  enabled: true
  device: cuda
  batch_size: 128
  min_population_for_gpu: 100  # Use GPU only if pop >= 100

parallel:
  use_multiprocessing: false  # GPU replaces multiprocessing
  num_workers: 1
```

**For small populations (<100):**
```yaml
gpu:
  enabled: false  # GPU overhead > speedup for small batches

parallel:
  use_multiprocessing: true
  num_workers: null  # Use all CPU cores
```

### Hybrid Evaluator

**Proposed implementation:**
```python
class HybridEvaluator:
    """Automatically choose CPU or GPU based on batch size."""
    
    def __init__(self, gpu_threshold=100):
        self.gpu_evaluator = get_gpu_evaluator()
        self.gpu_threshold = gpu_threshold
    
    def evaluate_population(self, population):
        if len(population) >= self.gpu_threshold and self.gpu_evaluator.is_available():
            # Large batch: use GPU
            return self.gpu_evaluator.batch_evaluate_conflicts(population)
        else:
            # Small batch: use CPU multiprocessing (faster)
            return [self._evaluate_cpu(ind) for ind in population]
```

---

## GPU Memory Considerations

### Memory Usage Estimates

**Per individual:**
- Genes: ~50-100 genes/individual
- Features: 5 floats/gene = 20 bytes/gene
- Total: ~1-2 KB/individual

**Population of 200:**
- Total: ~200-400 KB
- GPU batch (128): ~128-256 KB
- **Conclusion:** Memory is NOT a bottleneck (GPUs have 4-12GB)

### Optimal Batch Sizes

| GPU Memory | Batch Size | Population Size |
|------------|------------|-----------------|
| 4GB | 64-128 | 100-500 |
| 8GB | 128-256 | 500-1000 |
| 12GB+ | 256-512 | 1000+ |

---

## Performance Benchmarks (Expected)

### Fitness Evaluation Times

**CPU (multiprocessing, 8 cores):**
- 200 individuals: ~2-5 seconds/generation
- 500 individuals: ~5-12 seconds/generation
- 1000 individuals: ~10-25 seconds/generation

**GPU (RTX 3060, batch=128):**
- 200 individuals: ~0.2-0.5 seconds/generation
- 500 individuals: ~0.5-1 seconds/generation
- 1000 individuals: ~1-2 seconds/generation

**Speedup:** 10-20x for large populations

### Total Run Time Impact

**Test environment (30 generations):**
- CPU: ~5 minutes
- GPU: ~1-2 minutes
- **Savings:** 3-4 minutes

**Production (2000 generations):**
- CPU: ~6-10 hours
- GPU: ~0.5-1 hour
- **Savings:** 5-9 hours (5-10x faster!)

---

## Implementation Checklist

- [ ] Install CUDA Toolkit 12.1
- [ ] Verify GPU with `uv run diagnose-gpu`
- [ ] Create `configs/gpu.yaml`
- [ ] Add GPU config to Pydantic models (`src/config/models.py`)
- [ ] Modify `GAScheduler.__init__()` to accept GPU config
- [ ] Add `_evaluate_batch_gpu()` method
- [ ] Update evolution loop to use GPU evaluator
- [ ] Add hybrid CPU/GPU selection logic
- [ ] Test with small population (benchmark)
- [ ] Test with large population (validate speedup)
- [ ] Update documentation

---

## Troubleshooting

### "CUDA out of memory"
**Solution:** Reduce `batch_size` in config
```yaml
gpu:
  batch_size: 64  # Reduce from 128
```

### "GPU slower than CPU"
**Cause:** Small populations (GPU overhead > speedup)
**Solution:** Use hybrid evaluator with threshold

### "RuntimeError: CUDA error"
**Cause:** GPU in use by another process
**Solution:** Check with `nvidia-smi`, kill conflicting processes

### "No GPU available"
**Cause:** CUDA not installed or wrong PyTorch version
**Solution:** 
```bash
# Verify PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
uv add torch==2.4.1+cu121 --index https://download.pytorch.org/whl/cu121
```

---

## Summary

**Recommendation:** 
1. **Enable GPU for production runs** (2000 gens) - saves 5-9 hours
2. **Keep CPU multiprocessing for test runs** (30 gens) - simpler, good enough
3. **Use hybrid evaluator** - best of both worlds

**Priority:**
1. ✅ GPU fitness evaluation (HIGHEST IMPACT)
2. ⏳ GPU RL training (MEDIUM IMPACT)
3. ⏳ GPU heuristic batch eval (FUTURE)
