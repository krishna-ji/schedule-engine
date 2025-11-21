# Training Profiles

Three preconfigured profiles for different use cases and compute budgets.

## Profile Comparison

| Aspect | **Test** | **Med** | **Prod** |
|--------|----------|---------|----------|
| **Purpose** | Smoke test | Experiments | **Thesis** |
| **Timesteps** | 500 | 100,000 | **300,000** |
| **Time** | ~2-3 min | ~30-45 min | **~1-2 hours** |
| **Envs** | 1 | 4 | **8** |
| **Pop Size** | 10 | 50 | **80** |
| **Generations** | 30 | 120 | **200** |
| **Max Steps** | 20 | 60 | **80** |
| **Eval Episodes** | 1 | 5 | **10** |
| **Use Case** | Verify works | Tuning | **Deployment** |

## Test Profile

**Config**: `configs/training/test.yaml`

```yaml
profile: test
timesteps: 500
max_generations: 30
max_steps: 20
population_size: 10
eval_episodes: 1
save_prefix: rl_agent_test

parallel:
  n_envs: 1           # Single env (no overhead)
  use_subproc: false  # DummyVecEnv

debug_logging: true
debug_log_interval: 10
```

**When to use:**
- ✅ Quick sanity check (CI/CD)
- ✅ Verify code changes work
- ✅ Local development
- ✅ Debug training loop
- ❌ NOT for actual model training

**Command:**
```bash
uv run train-rl --test
```

**Output:**
- Model: `models/rl_agents/rl_agent_test.zip` (~5 MB)
- Time: 2-3 minutes
- Quality: **Throwaway** (not useful for deployment)

---

## Medium Profile

**Config**: `configs/training/med.yaml`

```yaml
profile: med
timesteps: 100000
max_generations: 120
max_steps: 60
population_size: 50
eval_episodes: 5
save_prefix: rl_agent_med

# Inherits parallel config from base.yaml
# (4 envs, subproc=true by default)
```

**When to use:**
- ✅ Hyperparameter tuning
- ✅ Ablation studies
- ✅ Iterative experiments
- ✅ Compare agent configurations
- ⚠️ OK for preliminary results
- ❌ NOT for final thesis/paper

**Command:**
```bash
uv run train-rl --med
```

**Output:**
- Model: `models/rl_agents/rl_agent_med.zip` (~10 MB)
- Time: 30-45 minutes
- Quality: **Usable** for testing, not production

---

## Production Profile

**Config**: `configs/training/prod.yaml`

```yaml
profile: prod
timesteps: 300000
max_generations: 200
max_steps: 80
population_size: 80
eval_episodes: 10
save_prefix: rl_agent_prod

parallel:
  n_envs: 8           # Maximum stable parallelism
  use_subproc: true   # True parallelism (bypass GIL)

debug_logging: true
debug_log_interval: 25
```

**When to use:**
- ✅ **Final thesis/paper experiments**
- ✅ Production deployment
- ✅ Benchmark comparisons
- ✅ Published results
- ✅ Real-world usage

**Command:**
```bash
uv run train-rl --prod
```

**Output:**
- Model: `models/rl_agents/rl_agent_prod.zip` (~15 MB)
- Time: 1-2 hours (300K timesteps)
- Quality: **Production-ready** 🎯

---

## Customizing Profiles

### Override via CLI

```bash
# Use prod profile but fewer timesteps
uv run train-rl --prod --timesteps 50000

# Use test profile but with more gens
uv run train-rl --test --max-generations 50

# Custom name
uv run train-rl --med --name "ablation-study-1"
```

### Create Custom Profile

```yaml
# configs/training/custom.yaml
profile: custom
timesteps: 150000
max_generations: 150
max_steps: 70
population_size: 60
eval_episodes: 8
save_prefix: rl_agent_custom

parallel:
  n_envs: 6
  use_subproc: true

device: cpu
debug_logging: true
```

```bash
# Use custom config
python src/rl/training/train_script.py --config configs/training/custom.yaml
```

### Environment-Specific Overrides

Profiles automatically inherit from `configs/base.yaml` and can be overridden:

```
base.yaml (shared defaults)
  ↓
test.yaml (test-specific overrides)
  ↓
Runtime flags (--timesteps, --agent, etc.)
```

---

## Profile Selection Guide

### Choose Test If:
- Running in CI/CD pipeline
- Quick code verification needed
- < 5 minutes available
- Don't care about model quality

### Choose Med If:
- Experimenting with hyperparameters
- Comparing agent configurations (PPO vs DQN)
- Running ablation studies
- Have 30-60 minutes
- Need "good enough" model

### Choose Prod If:
- **Writing thesis/paper**
- Benchmarking for publication
- Deploying to production
- Have 1-3 hours
- Need best possible model

---

## Computational Requirements

### Test Profile
- **RAM**: ~2 GB
- **CPU**: 1 core utilized
- **GPU**: Not used
- **Disk**: ~100 MB logs + 5 MB model
- **Network**: None

### Med Profile
- **RAM**: ~4-8 GB (4 parallel envs)
- **CPU**: 4 cores utilized (4 envs)
- **GPU**: Not used (CPU-only RL)
- **Disk**: ~500 MB logs + 10 MB model
- **Network**: None

### Prod Profile
- **RAM**: ~8-16 GB (8 parallel envs)
- **CPU**: 8 cores utilized (8 envs)
- **GPU**: Not used (CPU-only RL)
- **Disk**: ~2 GB logs + 15 MB model
- **Network**: None (TensorBoard optional)

---

## Training Time Breakdown

### Test (2-3 min)
```
Environment setup:  ~30s  (1 env × 10 pop)
Training (500 steps): ~90s
Evaluation (1 ep):    ~10s
Saving model:         ~5s
Total:               ~135s
```

### Med (30-45 min)
```
Environment setup:   ~2 min  (4 envs × 50 pop)
Training (100K steps): ~25 min
Evaluation (5 eps):    ~3 min
Saving/checkpoints:    ~2 min
Total:                ~32 min
```

### Prod (1-2 hours)
```
Environment setup:    ~5 min  (8 envs × 80 pop)
Training (300K steps):  ~90 min
Evaluation (10 eps):    ~10 min
Saving/checkpoints:     ~5 min
Total:                 ~110 min
```

---

## Cost-Benefit Analysis

| Profile | Time | CPU Hours | Model Quality | Best Use |
|---------|------|-----------|---------------|----------|
| Test | 3 min | 0.05h | ⭐ | Debug |
| Med | 45 min | 3h | ⭐⭐⭐ | Experiments |
| Prod | 2 hrs | 16h | ⭐⭐⭐⭐⭐ | **Thesis** |

**Recommendation**: 
- Use **Test** for development (100+ runs OK)
- Use **Med** for experiments (10-20 runs)
- Use **Prod** for final results (1-3 runs max)

---

## Next Steps

- **[02-quickstart.md](02-quickstart.md)** - Run first training
- **[08-trainer.md](08-trainer.md)** - Training system details
- **[15-configuration.md](15-configuration.md)** - All config options
- **[18-tensorboard.md](18-tensorboard.md)** - Monitor training
