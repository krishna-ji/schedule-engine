# RL Training Performance Fix (Nov 21, 2025)

## Problem
RL test profile was **frozen for 5+ minutes** during initialization:
- 4 parallel environments × 16 population size = **64 GA initializations**
- Each environment needs to initialize a full GA population
- With 4 envs in parallel, this takes 5-10 minutes just to start!

## Root Cause
```yaml
# OLD configs/training/test.yaml
parallel:
  n_envs: 4              # Too many for test!
  use_subproc: true      # Overhead for small test
population_size: 16      # Too large for quick test
timesteps: 10000         # Way too many for smoke test
```

**Math**: 4 envs × 16 pop × 40 gens = 2,560 fitness evaluations just for first rollout!

## Solution

### Updated Test Profile
```yaml
# NEW configs/training/test.yaml
parallel:
  n_envs: 1              # Single env - no parallelism overhead
  use_subproc: false     # No need for subproc with 1 env
population_size: 10      # Smaller population
timesteps: 500           # Just enough to verify RL works
max_generations: 30      # Reduced from 40
```

**New Math**: 1 env × 10 pop × 30 gens = 300 fitness evaluations (8x faster!)

## Performance Impact

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Initialization** | 5-10 min | ~30 sec | **10-20x faster** |
| **Total Time** | N/A (frozen) | ~2-3 min | **Actually works!** |
| **Timesteps** | 10,000 | 500 | More realistic for test |
| **Parallel Envs** | 4 | 1 | No overhead |
| **Pop Size** | 16 | 10 | Lighter load |

## Why This Works

1. **Single Environment**: No multiprocessing overhead, direct execution
2. **Smaller Population**: 10 instead of 16 = 37% less work
3. **Fewer Timesteps**: 500 is enough to verify RL pipeline works
4. **No Subproc**: DummyVecEnv instead of SubprocVecEnv = instant startup

## Usage

```bash
# Quick smoke test (now actually quick!)
uv run train-rl --test     # ~2-3 min

# Medium training (still uses 4 envs)
uv run train-rl --med      # ~30-45 min

# Production (uses 8 envs)
uv run train-rl --prod     # ~1-2 hours
```

## Files Changed

1. **`configs/training/test.yaml`**:
   - `n_envs: 4 → 1`
   - `use_subproc: true → false`
   - `population_size: 16 → 10`
   - `timesteps: 10000 → 500`
   - `max_generations: 40 → 30`

2. **`scripts/launcher.py`**:
   - Updated test timesteps: `10000 → 500`
   - Updated description: "~5-10 min" → "~2-3 min"

3. **`CLI_REFERENCE.md`**:
   - Updated all test profile documentation
   - Added env counts to table
   - Clarified actual timings

## Lesson Learned

**For test/smoke profiles:**
- Use single environment (no parallelism)
- Small population (10-20)
- Minimal timesteps (500-1000)
- Goal: Verify pipeline works, not train quality model

**For production:**
- Use multiple envs (4-8) for speed
- Larger populations (50-100)
- Many timesteps (50K-300K)
- Goal: Train actual useful model

## Summary

Test profile was trying to do production-scale work in "test" mode. Now it actually tests the pipeline quickly instead of freezing forever.

**Before**: 10,000 timesteps × 4 envs = waiting forever  
**After**: 500 timesteps × 1 env = done in 2-3 minutes ✅
