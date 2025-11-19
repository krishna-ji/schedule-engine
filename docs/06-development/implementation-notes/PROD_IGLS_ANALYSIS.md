# Production Configuration Analysis for IGLS

## Current Production Settings

### GA Parameters (prod.yaml)
- **ngen**: 500 generations
- **pop_size**: 100 individuals
- **Multiprocessing**: Enabled (auto-detect cores)

### IGLS Settings (from common.yaml)

#### Tier 1: Exhaustive Search
```yaml
enabled: true
generations: [3, 25]
population_coverage: 0.3  # Top 30 individuals
max_neighborhood_size: 80
timeout_seconds: 120
```

#### Tier 2: Stagnation Repair
```yaml
enabled: true
patience: 5
min_generation: 8
population_coverage: 0.5  # Top 50 individuals
max_iterations: 10
timeout_seconds: 60
cooldown: 3
```

#### Tier 3: Selective Probabilistic
```yaml
enabled: true
apply_probability: 0.3
apply_after_mutation: true
apply_after_crossover: false
```

## ⚠️ Issues with Current Production Settings

### Issue 1: Exhaustive Search Only at Gen 3 & 25
**Problem**: With 500 generations, exhaustive search only fires twice (gen 3, 25), leaving gens 26-500 without intensive repair.

**Impact**: 
- 95% of evolution (475 gens) has no exhaustive optimization
- Likely to stagnate after gen 100-200
- Wasting computational resources on unoptimized late generations

**Recommendation**: Add more exhaustive triggers for long runs
```yaml
generations: [3, 25, 100, 200, 350, 475]  # Strategic throughout evolution
```

### Issue 2: Timeout Too Short for Large Population
**Problem**: 
- Exhaustive timeout: 120s for top 30% of 100 = 30 individuals
- Test run: 10 pop × 30% = 3 individuals took 72s
- Projected: 30 individuals could take 720s (12 minutes!)

**Impact**: High probability of timeout, incomplete optimization

**Recommendation**: Increase timeout or reduce coverage
```yaml
# Option A: Increase timeout
timeout_seconds: 600  # 10 minutes

# Option B: Reduce coverage
population_coverage: 0.2  # Top 20 individuals = 20 × 4min ≈ 8 min
```

### Issue 3: VM Resource Constraints
**Concern**: Multiprocessing with 100 pop × 527 genes × constraint checks

**Questions to Answer**:
1. How many CPU cores does your VM have?
2. How much RAM?
3. What's acceptable runtime? (hours? days?)

## Recommended Production Configurations

### Option A: Conservative (Safe for VM)
```yaml
# prod.yaml
ga:
  ngen: 300  # Reduced from 500
  pop_size: 50  # Reduced from 100

# common.yaml adjustments:
repair:
  exhaustive_search:
    generations: [3, 25, 100, 200, 275]
    population_coverage: 0.2  # Top 10 individuals
    timeout_seconds: 300  # 5 minutes
  
  stagnation_repair:
    patience: 10  # Less aggressive for long runs
    population_coverage: 0.3
    timeout_seconds: 120
```

**Estimated Runtime**: 8-12 hours
**Peak Memory**: ~4-6 GB
**CPU Cores**: 4-8 recommended

### Option B: Aggressive (Requires Good Hardware)
```yaml
# prod.yaml
ga:
  ngen: 500
  pop_size: 100

# common.yaml adjustments:
repair:
  exhaustive_search:
    generations: [3, 25, 100, 200, 350, 475]
    population_coverage: 0.15  # Top 15 individuals
    timeout_seconds: 600  # 10 minutes
  
  stagnation_repair:
    patience: 15  # Less aggressive
    population_coverage: 0.2
    timeout_seconds: 180
```

**Estimated Runtime**: 24-36 hours
**Peak Memory**: ~8-12 GB
**CPU Cores**: 8-16 recommended

### Option C: Quick Production (Fast Iteration)
```yaml
# prod.yaml
ga:
  ngen: 200
  pop_size: 50

# common.yaml adjustments:
repair:
  exhaustive_search:
    generations: [3, 25, 100, 175]
    population_coverage: 0.3  # Top 15 individuals
    timeout_seconds: 240
  
  stagnation_repair:
    patience: 8
    population_coverage: 0.4
    timeout_seconds: 90
```

**Estimated Runtime**: 4-6 hours
**Peak Memory**: ~3-4 GB
**CPU Cores**: 4-8 recommended

## Critical Recommendations Before Running Production

### 1. Test Resource Usage First
```bash
# Run with current settings but limit generations
python main.py --env prod  # Monitor first 30 gens

# Check:
# - CPU usage (should be 80-100% if multiprocessing working)
# - Memory usage (watch for OOM)
# - Time per generation
```

### 2. Create Custom Production Config
Don't modify common.yaml directly. Instead, override IGLS settings in prod.yaml:

```yaml
# prod.yaml
name: "Production Configuration - IGLS System"
environment: prod

ga:
  ngen: 300
  pop_size: 50

parallel:
  use_multiprocessing: true
  num_workers: 6  # Explicit core count

# Override IGLS settings for production
repair:
  exhaustive_search:
    generations: [3, 25, 100, 200, 275]
    population_coverage: 0.2
    timeout_seconds: 300
  
  stagnation_repair:
    patience: 10
    timeout_seconds: 120
```

### 3. Monitor IGLS Performance
Watch the logs for:
```
 Gen X: EXHAUSTIVE SEARCH triggered
    Exhaustive search complete: N genes improved, total reduction: X, time: Ys
```

If timeout occurs frequently, reduce coverage or increase timeout.

### 4. Adjust Based on First Run
After first production run:
- If completes quickly (< 4 hours): Increase ngen or pop_size
- If times out often: Reduce coverage or increase timeout
- If stagnates early (< gen 100): Add more exhaustive triggers
- If memory issues: Reduce pop_size, disable multiprocessing

## VM Compatibility Check

### Minimum Requirements
- **CPU**: 4 cores (8+ recommended)
- **RAM**: 8 GB (16+ recommended)
- **Storage**: 10 GB free (for outputs)
- **OS**: Windows/Linux with Python 3.8+

### Expected Performance by VM Size
| VM Size | Cores | RAM | Recommended Config | Est. Runtime |
|---------|-------|-----|-------------------|--------------|
| Small   | 2-4   | 4-8 GB | ngen=100, pop=20 | 2-3 hours |
| Medium  | 4-8   | 8-16 GB | ngen=200, pop=50 | 6-8 hours |
| Large   | 8-16  | 16-32 GB | ngen=500, pop=100 | 24+ hours |

## Immediate Action Items

1. **Tell me your VM specs** (cores, RAM)
2. **Choose a config option** (A, B, or C above)
3. **Test with limited generations first**:
   ```bash
   # Modify prod.yaml temporarily:
   ga:
     ngen: 30  # Test IGLS triggers
   ```
4. **Then scale up** after confirming no issues

## Summary: Can You Run Current Prod Settings?

**Short Answer**: ⚠️ **Not recommended as-is**

**Issues**:
-  Exhaustive only at gen 3 & 25 (insufficient for 500 gens)
-  Timeout likely insufficient for 100 pop
-  No VM-specific tuning

**Next Steps**:
1. Tell me your VM specs
2. I'll create optimized prod config for your hardware
3. Test with 30 gens first
4. Scale up after validation

**Safe Starting Point** (works on most VMs):
```yaml
ga:
  ngen: 200
  pop_size: 40
repair:
  exhaustive_search:
    generations: [3, 25, 100, 175]
    population_coverage: 0.25
    timeout_seconds: 240
```

This should complete in 4-6 hours on a modest VM (4 cores, 8GB RAM).
