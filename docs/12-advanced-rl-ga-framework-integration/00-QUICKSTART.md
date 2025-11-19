# Phase 1 & 2 Integration - Quick Start Guide

**Date**: November 17, 2025  
**Status**: Phase 2 Execution Pending

---

## What's Been Done

###  Phase 1: Heuristic Toolbox (COMPLETE)
- **19 heuristic operators** across 5 categories (construction, perturbation, improvement, diversity, meta)
- Registry-based architecture with decorators
- Config-driven killswitches
- Full test coverage
- **Location**: `src/heuristics/`, `test/test_heuristics.py`

###  Phase 2: RL Integration (CODE COMPLETE)
- **Phase 2.1**: Gymnasium environment (21-dimensional state space, 20 actions)
- **Phase 2.2**: Training infrastructure (trainer, curriculum, callbacks, checkpoints)
- **Phase 2.3**: Deployment (model loader, inference engine, hybrid controller)
- **Phase 2.4**: GA integration hooks
- **Location**: `src/rl/`, `scripts/`, `config-train/`

---

## What Needs to Be Done (Execution Tasks)

### Step 1: Generate Validation Sets (30 minutes)
```bash
cd /path/to/schedule-engine

# Generate validation problems for each curriculum stage
python scripts/generate_validation_set.py --stage easy --num-problems 30
python scripts/generate_validation_set.py --stage medium --num-problems 30
python scripts/generate_validation_set.py --stage hard --num-problems 30
```

**Output**: `data/validation/{easy,medium,hard}/` with 90 problems total

---

### Step 2: Run Curriculum Training (24-48 hours)

#### Option A: Full Curriculum (Recommended for production)
```bash
# 300K timesteps, ~24-48 hours on GPU
python src/rl/training/train_script.py \
    --timesteps 300000 \
    --agent ppo \
    --device cuda \
    --config config-train/prod.yaml \
    --experiment-name phase2_curriculum_full
```

#### Option B: Medium Run (Balanced)
```bash
# 100K timesteps, ~8-16 hours on GPU
python src/rl/training/train_script.py \
    --timesteps 100000 \
    --agent ppo \
    --device cuda \
    --config config-train/med.yaml \
    --experiment-name phase2_curriculum_med
```

#### Option C: Smoke Test (Quick validation)
```bash
# 10K timesteps, ~1-2 hours on GPU
python src/rl/training/train_script.py \
    --timesteps 10000 \
    --agent ppo \
    --device cuda \
    --config config-train/test.yaml \
    --experiment-name phase2_smoke_test
```

**Monitor Training**:
```bash
tensorboard --logdir logs/tensorboard --port 6006
# Open browser: http://localhost:6006
```

**Expected Output**:
- Checkpoints in `models/rl_agents/checkpoints/`
- TensorBoard logs in `logs/tensorboard/`
- Manifest entries in `models/rl_agents/manifest.json`

---

### Step 3: Select Best Checkpoint (5 minutes)
```bash
# Evaluate all checkpoints on validation set
python scripts/select_best_checkpoint.py \
    --validation-dir data/validation \
    --metric median_reward \
    --promote
```

**Output**: Selected checkpoint ID and validation metrics

---

### Step 4: Promote to Production (2 minutes)
```bash
# Promote selected checkpoint to production
python scripts/promote_model_to_prod.py \
    --checkpoint-id <checkpoint_id_from_step3> \
    --update-config configs/prod.yaml
```

**Changes**:
- Copies model to `models/rl_agents/best_model.zip`
- Updates `configs/prod.yaml` (sets `rl.enabled: true`)
- Updates manifest with deployment metadata

---

### Step 5: Run Baseline Comparisons (4-8 hours)

#### A. Baseline (GA without RL)
```bash
# Ensure RL is disabled
python main.py --env prod
```

#### B. Enhanced (GA with RL)
```bash
# Ensure configs/prod.yaml has rl.enabled: true (done by promotion script)
python main.py --env prod
```

**Compare**:
- Final fitness (hard violations, soft penalty)
- Convergence speed (generations to feasibility)
- Time to completion
- Pareto front quality

---

### Step 6: Update Documentation (1 hour)
```bash
# Update Phase 2 completion doc with results
vim docs/06-development/implementation-notes/PHASE_2_RL_COMPLETE.md

# Add sections:
# - Empirical Results
# - Training convergence plots
# - Baseline comparison
# - Inference latency benchmarks
```

---

## Expected Timeline

- **Step 1** (Validation sets): 30 minutes
- **Step 2** (Training): 1-48 hours (depends on option chosen)
- **Step 3** (Checkpoint selection): 5 minutes
- **Step 4** (Promotion): 2 minutes
- **Step 5** (Baseline comparison): 4-8 hours
- **Step 6** (Documentation): 1 hour

**Total**: ~6-58 hours (mostly wall time for training)

---

## Troubleshooting

### Training Not Converging
**Symptom**: Reward not improving, flat learning curves

**Solutions**:
1. Check TensorBoard for entropy decay (should decrease gradually)
2. Adjust hyperparameters in `config-train/base.yaml`:
   ```yaml
   rl:
     agent:
       ppo:
         learning_rate: 0.0001  # Try lower if unstable
         ent_coef: 0.02         # Try higher for more exploration
   ```
3. Verify heuristics are enabled in `configs/base.yaml`
4. Check for errors in logs: `tail -f logs/training.log`

### Checkpoint Selection Fails
**Symptom**: No checkpoints meet advancement threshold

**Solutions**:
1. Lower advancement_patience in config
2. Lower threshold values in curriculum config
3. Check validation set difficulty (may be too hard)

### Inference Too Slow
**Symptom**: RL inference exceeds 10ms timeout

**Solutions**:
1. Ensure model loaded to GPU: check `device: cuda` in config
2. Enable model caching (should be default)
3. Reduce max_steps_per_episode if needed
4. Use fallback_strategy if RL consistently times out

---

## Success Indicators

### Training Success
-  Reward curve shows improvement over curriculum stages
-  Agent advances through all 3 stages (easy→medium→hard)
-  Final checkpoints show positive median_reward
-  No crashes or errors during training

### Deployment Success
-  Model loads in <100ms
-  Inference completes in <10ms (median)
-  No errors in production run with RL enabled
-  RL actions successfully applied (check logs)

### Performance Success (Ideal but not required initially)
-  RL-enabled GA matches baseline performance (no regression)
-  RL-enabled GA exceeds baseline (improvement)
-  Learned policy shows interpretable patterns

---

## Phase 3 Readiness

Once Phase 2 execution is complete:

1. **Review Phase 3 roadmap**: `docs/12-advanced-rl-ga-framework-integration/31-phase3-roadmap.md`
2. **Check dependency analysis**: `docs/12-advanced-rl-ga-framework-integration/30-dependency-analysis.md`
3. **Decide on Tier 1 implementation**: Start with Enhancement #2 (constraint-specific state)

---

## Documentation Locations

- **Integration Status**: `docs/12-advanced-rl-ga-framework-integration/03-integration-status.md`
- **Dependency Analysis**: `docs/12-advanced-rl-ga-framework-integration/30-dependency-analysis.md`
- **Phase 3 Roadmap**: `docs/12-advanced-rl-ga-framework-integration/31-phase3-roadmap.md`
- **Updated Todo**: `Todo.md` (Phase 3 tasks appended)
- **Phase 2 Details**: `docs/06-development/implementation-notes/PHASE_2_RL_COMPLETE.md`

---

## Quick Commands Reference

```bash
# Generate validation sets
for stage in easy medium hard; do
    python scripts/generate_validation_set.py --stage $stage --num-problems 30
done

# Run training (medium)
python src/rl/training/train_script.py --timesteps 100000 --agent ppo --device cuda --config config-train/med.yaml

# Monitor training
tensorboard --logdir logs/tensorboard --port 6006

# Select & promote
python scripts/select_best_checkpoint.py --validation-dir data/validation --metric median_reward --promote
python scripts/promote_model_to_prod.py --checkpoint-id <id> --update-config configs/prod.yaml

# Run with RL
python main.py --env prod  # rl.enabled must be true in configs/prod.yaml
```

---

**Next Action**: Start with Step 1 (generate validation sets)

**Questions?** See detailed documentation in `docs/12-advanced-rl-ga-framework-integration/`
