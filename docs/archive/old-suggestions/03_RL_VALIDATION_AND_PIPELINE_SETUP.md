# RL Model Validation & Pipeline Setup Guide

**Date**: November 17, 2025  
**Version**: 1.0  
**Status**: Production Guide

---

## Executive Summary

This document provides comprehensive guidance on:
1. **Training Pipeline**: How to train RL agents from scratch
2. **Validation Strategy**: How to validate RL model performance
3. **Model Selection**: How to choose the best checkpoint
4. **Production Deployment**: How to load and use trained models in GA runtime
5. **Inference Integration**: The complex part - how GA loads and uses RL models

---

## Table of Contents

1. [Training Pipeline Setup](#training-pipeline-setup)
2. [RL Model Validation](#rl-model-validation)
3. [Checkpoint Selection](#checkpoint-selection)
4. [Production Deployment](#production-deployment)
5. [GA Runtime Integration](#ga-runtime-integration)
6. [Troubleshooting](#troubleshooting)

---

## Training Pipeline Setup

### Overview

The training pipeline consists of:
1. Data preparation (validation sets)
2. Training configuration (profiles)
3. Training execution (curriculum learning)
4. Checkpoint management
5. TensorBoard monitoring

### Step 1: Generate Validation Sets

Validation sets are used to evaluate checkpoints during and after training.

```bash
# Generate validation sets for all curriculum stages
python scripts/generate_validation_set.py --stage all --num-problems 20

# Or generate per-stage
python scripts/generate_validation_set.py --stage easy --num-problems 30
python scripts/generate_validation_set.py --stage medium --num-problems 25
python scripts/generate_validation_set.py --stage hard --num-problems 20
```

**Output Structure**:
```
data/validation/
├── easy/
│   ├── problem_001.json
│   ├── problem_002.json
│   └── ... (20-30 problems)
├── medium/
│   └── ... (20-25 problems)
└── hard/
    └── ... (15-20 problems)
```

**Important**: Use the same seed for consistency:
```python
# In scripts/generate_validation_set.py
def generate_validation_set(stage, num_problems, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # ... generation logic ...
```

### Step 2: Configure Training Profile

Training profiles are in `config-train/`:

```yaml
# config-train/prod.yaml
training:
  total_timesteps: 300000  # Total training steps
  agent_type: "ppo"        # Agent type (ppo or dqn)
  seed: 42                 # Random seed
  
  curriculum:
    - name: "easy"
      num_episodes: 200
      max_generations: 100
      sample_config:
        num_courses: 10
      threshold: -5.0         # Advancement threshold
      advancement_patience: 5  # Episodes before advancing
    
    - name: "medium"
      num_episodes: 300
      max_generations: 200
      sample_config:
        num_courses: 20
      threshold: -3.0
      advancement_patience: 5
    
    - name: "hard"
      num_episodes: 500
      max_generations: 400
      sample_config:
        num_courses: 40
      threshold: -2.0
      advancement_patience: 5
  
  ppo:
    learning_rate: 0.0003
    n_steps: 2048
    batch_size: 64
    n_epochs: 10
    gamma: 0.99
    gae_lambda: 0.95
    clip_range: 0.2
    ent_coef: 0.01
  
  checkpoint_settings:
    save_freq: 50           # Save every 50 episodes
    validation_freq: 25     # Validate every 25 episodes
    keep_best_n: 5          # Keep top 5 checkpoints
    manifest_path: "models/rl_agents/manifest.json"
    validation_set_dir: "data/validation"
  
  logging:
    tensorboard_dir: "logs/tensorboard"
    log_dir: "logs/training"
    log_level: "INFO"
```

**Quick Test Profile** (`config-train/test.yaml`):
```yaml
training:
  total_timesteps: 10000
  curriculum:
    - name: "easy"
      num_episodes: 50
      max_generations: 50
      threshold: -10.0  # Easier threshold
```

### Step 3: Start Training

```bash
# Quick smoke test (5-10 minutes)
uv run train --profile test

# Medium training (30-60 minutes)
uv run train --profile med

# Full production training (60-120 minutes)
uv run train --profile prod

# With custom overrides
uv run train --profile prod --timesteps 500000 --seed 123
```

**Training Output**:
```
[cyan]Starting RL Training...[/cyan]
[cyan]Profile: prod[/cyan]
[cyan]Agent: PPO[/cyan]
[cyan]Total Timesteps: 300000[/cyan]

[bold]Curriculum Stage 1: easy[/bold]
  Episodes: 200
  Max Generations: 100
  Threshold: -5.0

Episode 1/200: reward=-15.2, length=45
Episode 2/200: reward=-12.8, length=42
...
Episode 50/200: reward=-6.1, length=38 [cyan]✓ Checkpoint saved[/cyan]
...
Episode 200/200: reward=-4.8, length=35 [green]✓ Stage complete[/green]

[bold]Curriculum Stage 2: medium[/bold]
...
```

### Step 4: Monitor with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir logs/tensorboard --port 6006

# Open browser to http://localhost:6006
```

**Key Metrics to Monitor**:

1. **Training Metrics**:
   - `rollout/ep_rew_mean`: Average episode reward (should increase)
   - `rollout/ep_len_mean`: Average episode length
   - `train/policy_loss`: Policy gradient loss
   - `train/value_loss`: Value function loss
   - `train/entropy_loss`: Exploration entropy

2. **Validation Metrics**:
   - `eval/mean_reward`: Validation set performance
   - `eval/success_rate`: % episodes reaching goal
   - `eval/mean_violations`: Average constraint violations

3. **Curriculum Progress**:
   - `curriculum/stage`: Current stage (0=easy, 1=medium, 2=hard)
   - `curriculum/stage_episodes`: Episodes in current stage
   - `curriculum/advancement_progress`: Progress toward threshold

---

## RL Model Validation

### Validation Strategy

RL model validation is **different from supervised learning** because:
- No ground truth labels
- Success measured by constraint satisfaction
- Need to test on unseen problems
- Must compare against baseline strategies

### Three-Level Validation

#### Level 1: Training Validation (Online)

Happens **during training** via callbacks:

```python
# src/rl/training/callbacks.py
class PeriodicEvaluationCallback(BaseCallback):
    def __init__(self, eval_freq=25, validation_set_dir="data/validation"):
        self.eval_freq = eval_freq
        self.validation_problems = load_validation_set(validation_set_dir)
    
    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate on validation set
            rewards = []
            for problem in self.validation_problems:
                env = create_env(problem)
                episode_reward = evaluate_episode(env, self.model)
                rewards.append(episode_reward)
            
            mean_reward = np.mean(rewards)
            self.logger.record("eval/mean_reward", mean_reward)
            
            # Save checkpoint if best so far
            if mean_reward > self.best_reward:
                self.best_reward = mean_reward
                self.model.save(f"{self.checkpoint_dir}/best_checkpoint.zip")
```

**Metrics Tracked**:
- Mean episode reward
- Success rate (% episodes with reward > threshold)
- Mean hard violations
- Mean soft violations
- Episode length distribution

#### Level 2: Checkpoint Validation (Offline)

Happens **after training** to select best checkpoint:

```bash
# Evaluate all checkpoints on full validation set
python scripts/select_best_checkpoint.py \
    --metric mean_reward \
    --validation-set data/validation/hard \
    --num-episodes 50
```

**Script Logic**:
```python
# scripts/select_best_checkpoint.py
def evaluate_checkpoint(checkpoint_path, validation_set, num_episodes):
    """Evaluate single checkpoint on validation set."""
    model = PPO.load(checkpoint_path)
    
    results = []
    for problem in validation_set:
        for episode in range(num_episodes):
            env = create_env(problem)
            obs = env.reset(seed=episode)  # Different seeds per episode
            
            total_reward = 0
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                total_reward += reward
            
            results.append({
                'problem': problem.name,
                'episode': episode,
                'reward': total_reward,
                'hard_violations': info['hard_violations'],
                'soft_violations': info['soft_violations'],
                'steps': info['steps']
            })
    
    return aggregate_results(results)

def select_best_checkpoint(checkpoints, validation_set, metric='mean_reward'):
    """Select best checkpoint by validation metric."""
    checkpoint_scores = []
    
    for checkpoint in checkpoints:
        results = evaluate_checkpoint(checkpoint, validation_set, num_episodes=50)
        score = results[metric]
        checkpoint_scores.append((checkpoint, score, results))
    
    # Sort by metric (higher is better)
    checkpoint_scores.sort(key=lambda x: x[1], reverse=True)
    
    best_checkpoint, best_score, best_results = checkpoint_scores[0]
    return best_checkpoint, best_results
```

**Validation Metrics**:

| Metric | Description | Target |
|--------|-------------|--------|
| `mean_reward` | Average episode reward | > -2.0 (hard stage) |
| `median_reward` | Median reward (robust to outliers) | > -2.5 |
| `success_rate` | % episodes with reward > -1.0 | > 80% |
| `mean_hard_violations` | Average hard constraint violations | < 1.0 |
| `mean_soft_violations` | Average soft constraint penalty | < 10.0 |
| `convergence_rate` | Average steps to convergence | < 50 |
| `robustness_score` | Std dev of rewards (lower = better) | < 2.0 |

#### Level 3: Baseline Comparison (Production)

Happens **in production** to verify RL improves over baseline:

```bash
# Run baseline (pure GA, no RL)
uv run prod  # with rl.enabled: false

# Run RL-enhanced GA
uv run prod  # with rl.enabled: true

# Compare results
python scripts/compare_runs.py \
    output/baseline_run_20251117/ \
    output/rl_run_20251117/
```

**Comparison Script**:
```python
# scripts/compare_runs.py
def compare_runs(baseline_dir, rl_dir):
    """Compare baseline vs RL runs."""
    baseline_metrics = load_metrics(f"{baseline_dir}/metrics.json")
    rl_metrics = load_metrics(f"{rl_dir}/metrics.json")
    
    comparison = {
        'hard_violations': {
            'baseline': baseline_metrics['final_hard_violations'],
            'rl': rl_metrics['final_hard_violations'],
            'improvement': improvement_percentage(...)
        },
        'soft_violations': {
            'baseline': baseline_metrics['final_soft_violations'],
            'rl': rl_metrics['final_soft_violations'],
            'improvement': improvement_percentage(...)
        },
        'convergence_speed': {
            'baseline': baseline_metrics['generations_to_feasible'],
            'rl': rl_metrics['generations_to_feasible'],
            'speedup': speedup_factor(...)
        },
        'runtime': {
            'baseline': baseline_metrics['total_time_seconds'],
            'rl': rl_metrics['total_time_seconds'],
            'overhead': overhead_percentage(...)
        }
    }
    
    # Statistical significance test
    comparison['statistical_test'] = mann_whitney_u_test(
        baseline_metrics['per_generation_fitness'],
        rl_metrics['per_generation_fitness']
    )
    
    return comparison
```

### Validation Best Practices

1. **Use Held-Out Problems**: Never validate on training data
2. **Multiple Seeds**: Run each problem with different seeds (5-10)
3. **Deterministic Policy**: Use `deterministic=True` for validation
4. **Statistical Testing**: Use Mann-Whitney U or Wilcoxon signed-rank test
5. **Document Results**: Save all validation metrics for comparison

### Validation Checklist

- [ ] Validation set has at least 20-30 problems per stage
- [ ] Validation problems are unseen during training
- [ ] Each checkpoint evaluated with 50+ episodes
- [ ] Results compared against random and greedy baselines
- [ ] Statistical significance tested (p < 0.05)
- [ ] Validation metrics tracked over training
- [ ] Best checkpoint identified by robust metric (median reward)
- [ ] Production comparison shows improvement

---

## Checkpoint Selection

### Checkpoint Management

Checkpoints are automatically saved during training:

```
models/rl_agents/
├── checkpoints/
│   ├── ppo_stage1_easy_ep025.zip
│   ├── ppo_stage1_easy_ep050.zip
│   ├── ppo_stage1_easy_ep075.zip
│   ├── ppo_stage2_medium_ep025.zip
│   ├── ppo_stage3_hard_ep050.zip    # Best checkpoint
│   └── ...
├── manifest.json                     # Checkpoint metadata
└── best_model.zip → checkpoints/...  # Symlink
```

### Manifest Structure

```json
{
  "checkpoints": [
    {
      "checkpoint_id": "ppo_stage3_hard_ep050",
      "model_path": "models/rl_agents/checkpoints/ppo_stage3_hard_ep050.zip",
      "created_at": "2025-11-17T10:30:45",
      "stage": "hard",
      "episode": 50,
      "timesteps": 250000,
      "training_metrics": {
        "mean_train_reward": -2.1,
        "episode_length": 35
      },
      "validation_metrics": {
        "mean_reward": -1.8,
        "median_reward": -1.6,
        "success_rate": 0.85,
        "mean_hard_violations": 0.3,
        "mean_soft_violations": 8.2
      },
      "metadata": {
        "agent_type": "ppo",
        "observation_space": "Box(21,)",
        "action_space": "Discrete(20)"
      }
    },
    // ... more checkpoints ...
  ],
  "best_checkpoint": "ppo_stage3_hard_ep050",
  "last_updated": "2025-11-17T12:00:00"
}
```

### Selection Strategies

#### Strategy 1: Best Mean Reward

```bash
python scripts/select_best_checkpoint.py --metric mean_reward
```

**Pros**: Simple, intuitive
**Cons**: Sensitive to outliers

#### Strategy 2: Best Median Reward (Recommended)

```bash
python scripts/select_best_checkpoint.py --metric median_reward
```

**Pros**: Robust to outliers, consistent performance
**Cons**: May miss high peaks

#### Strategy 3: Best Success Rate

```bash
python scripts/select_best_checkpoint.py --metric success_rate --threshold -1.0
```

**Pros**: Focuses on feasibility
**Cons**: Ignores soft constraint quality

#### Strategy 4: Multi-Objective

```bash
python scripts/select_best_checkpoint.py \
    --metrics mean_reward,success_rate,robustness_score \
    --weights 0.5,0.3,0.2
```

**Pros**: Balanced selection
**Cons**: Requires weight tuning

### Promotion Workflow

Once best checkpoint selected:

```bash
# Promote to production
python scripts/promote_model_to_prod.py \
    --checkpoint-id ppo_stage3_hard_ep050 \
    --validate

# This will:
# 1. Validate checkpoint integrity
# 2. Update configs/prod.yaml
# 3. Create registry entry
# 4. Backup previous config
```

---

## Production Deployment

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  PRODUCTION DEPLOYMENT                   │
└─────────────────────────────────────────────────────────┘

1. MODEL STORAGE
   models/rl_agents/
   ├── checkpoints/           # All training checkpoints
   ├── production/            # Current production model
   │   └── current.zip → ../checkpoints/ppo_stage3_hard_ep050.zip
   └── registry.json          # Deployment history

2. CONFIGURATION
   configs/prod.yaml
   ├── rl.enabled: true
   ├── rl.agent.model_path: "models/rl_agents/production/current.zip"
   └── rl.agent.type: "ppo"

3. RUNTIME LOADING
   src/core/ga_scheduler.py
   ├── _init_rl()            # Load model on GA start
   ├── _apply_rl_operators() # Use model each generation
   └── _cleanup_rl()         # Release resources on end
```

### Deployment Steps

#### Step 1: Validate Checkpoint

```python
# scripts/promote_model_to_prod.py
def validate_checkpoint(checkpoint_path):
    """Validate checkpoint before promotion."""
    checks = []
    
    # 1. File exists
    if not Path(checkpoint_path).exists():
        checks.append(("File exists", False, "Not found"))
        return checks, False
    checks.append(("File exists", True, "OK"))
    
    # 2. Can load model
    try:
        model = PPO.load(checkpoint_path)
        checks.append(("Model loads", True, "OK"))
    except Exception as e:
        checks.append(("Model loads", False, str(e)))
        return checks, False
    
    # 3. Action/observation space matches
    expected_obs_shape = (21,)
    expected_action_space = 20
    
    if model.observation_space.shape != expected_obs_shape:
        checks.append(("Observation space", False, f"Expected {expected_obs_shape}"))
        return checks, False
    checks.append(("Observation space", True, "OK"))
    
    if model.action_space.n != expected_action_space:
        checks.append(("Action space", False, f"Expected {expected_action_space}"))
        return checks, False
    checks.append(("Action space", True, "OK"))
    
    # 4. Can predict
    try:
        obs = model.observation_space.sample()
        action, _ = model.predict(obs)
        checks.append(("Prediction works", True, "OK"))
    except Exception as e:
        checks.append(("Prediction works", False, str(e)))
        return checks, False
    
    return checks, True
```

#### Step 2: Update Configuration

```python
def update_production_config(checkpoint_path, agent_type):
    """Update prod.yaml with new model."""
    config_path = "configs/prod.yaml"
    
    # Backup current config
    backup_path = f"{config_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(config_path, backup_path)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Update RL section
    config['rl']['enabled'] = True
    config['rl']['mode'] = 'inference'
    config['rl']['agent']['type'] = agent_type
    config['rl']['agent']['model_path'] = checkpoint_path
    
    # Write atomically (temp file + rename)
    temp_path = f"{config_path}.tmp"
    with open(temp_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    os.replace(temp_path, config_path)
    
    print(f"[green]✓ Updated {config_path}[/green]")
    print(f"[dim]  Backup: {backup_path}[/dim]")
```

#### Step 3: Register Deployment

```python
def register_deployment(checkpoint_id, model_path, validation_metrics):
    """Record deployment in registry."""
    registry_path = "models/rl_agents/registry.json"
    
    # Load existing registry
    if Path(registry_path).exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {"deployments": []}
    
    # Add new deployment
    deployment = {
        "deployment_id": f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "checkpoint_id": checkpoint_id,
        "model_path": model_path,
        "validation_metrics": validation_metrics,
        "deployed_at": datetime.now().isoformat(),
        "deployed_by": os.getenv("USER", "unknown"),
        "status": "active"
    }
    
    # Mark previous deployment as inactive
    for d in registry["deployments"]:
        d["status"] = "inactive"
    
    registry["deployments"].insert(0, deployment)
    registry["current_deployment"] = deployment["deployment_id"]
    
    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
```

### Rollback Procedure

```bash
# View deployment history
python scripts/promote_model_to_prod.py --list

# Rollback to previous deployment
python scripts/promote_model_to_prod.py --rollback

# Rollback to specific deployment
python scripts/promote_model_to_prod.py --rollback --deployment-id deploy_20251115_093000
```

---

## GA Runtime Integration

This is the **complex part**: How does the GA load and use the trained RL model at runtime?

### Integration Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                GA RUNTIME RL INTEGRATION                      │
└──────────────────────────────────────────────────────────────┘

PHASE 1: INITIALIZATION (Once, at GA start)
   ├─ Load Config (configs/prod.yaml)
   ├─ Check: rl.enabled == true?
   ├─ Check: rl.mode == "inference" or "hybrid"?
   ├─ Initialize Components:
   │  ├─ StateEncoder (population → 21-dim vector)
   │  ├─ ActionMapper (action ID → heuristic function)
   │  ├─ ModelLoader (load .zip file)
   │  ├─ RLInference (prediction engine)
   │  └─ HybridController (RL + fallback)
   └─ Set flag: self.rl_enabled = True

PHASE 2: EVOLUTION LOOP (Each Generation)
   ├─ Standard GA Operators (crossover, mutation)
   ├─ Fitness Evaluation
   ├─ **RL OPERATOR APPLICATION**:
   │  ├─ StateEncoder.encode(population) → state vector
   │  ├─ HybridController.select_action(state) → action ID
   │  ├─ ActionMapper.apply_action(action, individual) → modified ind
   │  ├─ Fitness evaluation of modified individual
   │  └─ Population update
   └─ Metrics tracking

PHASE 3: CLEANUP (At GA end)
   ├─ Save results
   ├─ Release RL resources
   └─ Log statistics
```

### Detailed Integration Flow

#### Initialization Phase (`_init_rl`)

**Location**: `src/core/ga_scheduler.py::_init_rl()`

**Step-by-Step**:

1. **Check Configuration**:
```python
def _init_rl(self) -> bool:
    rl_config = get_config().rl
    
    # Check if enabled
    if not rl_config.enabled:
        return False
    
    # Check mode
    if rl_config.mode not in ["inference", "hybrid"]:
        console.print(f"[yellow]RL mode '{rl_config.mode}' not compatible[/yellow]")
        return False
```

2. **Initialize State Encoder**:
```python
from src.rl.gym_env.state_encoder import StateEncoder

self.rl_state_encoder = StateEncoder(
    max_generations=self.config.generations,
    history_size=rl_config.environment.observation_history_size,
    normalize=True
)
console.print("   [green]✓ StateEncoder initialized[/green]")
```

3. **Initialize Action Mapper**:
```python
from src.rl.gym_env.action_mapper import ActionMapper

self.rl_action_mapper = ActionMapper(context=self.context)
console.print(f"   [green]✓ ActionMapper initialized ({len(self.rl_action_mapper.valid_actions)} actions)[/green]")
```

4. **Load Trained Model**:
```python
from src.rl.deployment.model_loader import ModelLoader

model_path = rl_config.agent.model_path

# Try to auto-detect best checkpoint
if not model_path or model_path == "models/rl_agents/best_model.zip":
    from src.rl.training.checkpoints import CheckpointManager
    
    manager = CheckpointManager(rl_config.training.checkpoint_settings.manifest_path)
    best_checkpoint = manager.get_best_checkpoint(metric="mean_reward")
    if best_checkpoint:
        model_path = best_checkpoint.model_path

# Load model
loader = ModelLoader(cache_models=True)
model, metadata = loader.load_model(model_path, agent_type=rl_config.agent.type)
console.print(f"   [green]✓ Model loaded: {rl_config.agent.type.upper()}[/green]")
```

5. **Create Inference Engine**:
```python
from src.rl.deployment.inference import RLInference

inference_engine = RLInference(
    model=model,
    timeout_ms=rl_config.inference.timeout_ms
)
```

6. **Initialize Hybrid Controller**:
```python
from src.rl.hybrid.hybrid_controller import HybridController

self.rl_controller = HybridController(
    inference_engine=inference_engine,
    action_mapper=self.rl_action_mapper,
    mode=rl_config.hybrid.mode,
    fallback_strategy=rl_config.hybrid.fallback_strategy,
    rl_probability=rl_config.hybrid.rl_probability
)
console.print(f"   [green]✓ HybridController initialized (mode: {rl_config.hybrid.mode})[/green]")
```

7. **Set Flag**:
```python
self.rl_enabled = True
console.print("[green]RL Integration: ENABLED[/green]")
return True
```

#### Evolution Phase (`_apply_rl_operators`)

**Location**: `src/core/ga_scheduler.py::_apply_rl_operators(gen)`

**Called**: Once per generation, after selection

**Step-by-Step**:

1. **Check RL Enabled**:
```python
def _apply_rl_operators(self, gen: int) -> None:
    if not self.rl_enabled:
        return
```

2. **Encode Population State**:
```python
state = self.rl_state_encoder.encode(
    population=self.population,
    current_generation=gen,
    generations_without_improvement=self.stagnation_counter,
    max_generations=self.config.generations
)
# state: numpy array shape (21,), values in [0, 1]
```

3. **Select Action (via RL + Hybrid Controller)**:
```python
action_id = self.rl_controller.select_action(
    state=state,
    valid_actions=self.rl_action_mapper.get_valid_actions(),
    deterministic=True  # Use deterministic policy in production
)
# action_id: integer in [0, 19]
```

4. **Apply Selected Heuristic**:
```python
# Get best individual to modify
best_ind = tools.selBest(self.population, 1)[0]

# Apply heuristic
modified_individuals, success = self.rl_action_mapper.apply_action(
    action_id=action_id,
    individual=best_ind,
    context=self.context,
    population=self.population,
    generation=gen
)

if not success:
    logger.warning(f"Heuristic {action_id} failed to apply")
    return
```

5. **Evaluate Modified Individuals**:
```python
if modified_individuals:
    # Compute fitness for modified individuals
    fitness_values = list(self.toolbox.map(
        self.toolbox.evaluate,
        modified_individuals
    ))
    
    # Assign fitness
    for ind, fit in zip(modified_individuals, fitness_values):
        ind.fitness.values = fit
```

6. **Update Population (Optional)**:
```python
# Replace worst individuals with improved ones
if modified_individuals:
    worst_indices = self._get_worst_indices(len(modified_individuals))
    for idx, new_ind in zip(worst_indices, modified_individuals):
        self.population[idx] = new_ind
```

7. **Record Heuristic Usage**:
```python
self.rl_state_encoder.record_heuristic_application(action_id)
```

### Critical Implementation Details

#### Detail 1: Model Loading Performance

**Problem**: Model loading can take 50-100ms, blocking GA start.

**Solution**: Cache model in ModelLoader:
```python
class ModelLoader:
    def __init__(self, cache_models=True):
        self.cache_models = cache_models
        self._cache = {}  # {model_path: (model, load_time)}
    
    def load_model(self, model_path, agent_type):
        # Check cache
        if self.cache_models and model_path in self._cache:
            return self._cache[model_path]
        
        # Load from disk
        start_time = time.time()
        if agent_type == "ppo":
            model = PPO.load(model_path)
        elif agent_type == "dqn":
            model = DQN.load(model_path)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        load_time = time.time() - start_time
        
        # Cache for future use
        if self.cache_models:
            self._cache[model_path] = (model, load_time)
        
        return model, {"load_time": load_time}
```

#### Detail 2: Inference Timeout Protection

**Problem**: RL prediction may hang or take too long.

**Solution**: Wrap prediction with timeout:
```python
class RLInference:
    def __init__(self, model, timeout_ms=10.0):
        self.model = model
        self.timeout_ms = timeout_ms
    
    def predict(self, state, deterministic=True):
        """Predict action with timeout protection."""
        start_time = time.time()
        
        try:
            # Predict action
            action, _ = self.model.predict(state, deterministic=deterministic)
            
            # Check timeout
            elapsed_ms = (time.time() - start_time) * 1000
            if elapsed_ms > self.timeout_ms:
                logger.warning(f"Inference timeout: {elapsed_ms:.1f}ms > {self.timeout_ms}ms")
            
            return action
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
```

#### Detail 3: Heuristic Execution Safety

**Problem**: Heuristics may fail or return invalid individuals.

**Solution**: Wrap execution in ActionMapper:
```python
def apply_action(self, action, individual, context, **kwargs):
    action_info = self.actions[action]
    
    if action_info.function is None:
        return [individual], True  # No-op
    
    try:
        # Clone individual to avoid mutation
        individual_copy = copy.deepcopy(individual)
        
        # Apply heuristic
        result = action_info.function(individual_copy, context, **kwargs)
        
        # Validate result
        if result is None or not isinstance(result, list):
            logger.warning(f"Heuristic {action_info.name} returned invalid result")
            return [individual], False
        
        # Check genes are valid
        if any(not hasattr(gene, 'course_id') for gene in result):
            logger.warning(f"Heuristic {action_info.name} returned invalid genes")
            return [individual], False
        
        return [result], True
    
    except Exception as e:
        logger.error(f"Heuristic {action_info.name} failed: {e}")
        return [individual], False
```

### Configuration Example

**Production Configuration** (`configs/prod.yaml`):
```yaml
rl:
  enabled: true
  mode: inference  # Options: disabled, training, inference, hybrid
  
  agent:
    type: ppo
    model_path: models/rl_agents/production/current.zip
  
  inference:
    timeout_ms: 10.0
    deterministic: true
    track_performance: true
  
  hybrid:
    mode: rl_primary  # Trust RL, fallback only on error
    fallback_strategy: random
    rl_probability: 1.0
  
  environment:
    max_steps_per_episode: 100
    observation_history_size: 10
  
  logging:
    log_heuristic_usage: true
    log_level: INFO
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Model Not Found

```
FileNotFoundError: Model not found at models/rl_agents/best_model.zip
```

**Solution**:
```bash
# Check model path
ls -la models/rl_agents/

# Train a model if missing
uv run train --profile test

# Or promote existing checkpoint
python scripts/promote_model_to_prod.py --checkpoint-id <CHECKPOINT_ID>
```

#### Issue 2: Inference Timeout

```
Warning: Inference timeout: 15.2ms > 10.0ms
```

**Solution**: Increase timeout in config:
```yaml
rl:
  inference:
    timeout_ms: 20.0
```

#### Issue 3: Action Space Mismatch

```
AssertionError: Expected action space Discrete(20), got Discrete(15)
```

**Solution**: Retrain model with current heuristic configuration.

#### Issue 4: RL Not Improving Results

**Debugging Steps**:
1. Check RL is actually being used: `grep "RL action" output/*/logger.txt`
2. Check action distribution: Are all actions being selected?
3. Check fallback usage: Is RL timing out frequently?
4. Validate checkpoint: Did we select the right checkpoint?
5. Compare validation vs production: Different data characteristics?

**Diagnostic Script**:
```python
# scripts/diagnose_rl_usage.py
def diagnose_rl_usage(log_file):
    """Diagnose RL usage from GA log file."""
    rl_actions = []
    fallback_actions = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "RL action:" in line:
                action_id = extract_action_id(line)
                rl_actions.append(action_id)
            elif "Fallback action:" in line:
                action_id = extract_action_id(line)
                fallback_actions.append(action_id)
    
    print(f"Total RL actions: {len(rl_actions)}")
    print(f"Total fallback actions: {len(fallback_actions)}")
    print(f"RL usage rate: {len(rl_actions)/(len(rl_actions)+len(fallback_actions))*100:.1f}%")
    
    # Action distribution
    action_counts = Counter(rl_actions)
    print("\nMost used actions:")
    for action_id, count in action_counts.most_common(5):
        action_name = get_action_name(action_id)
        print(f"  {action_name}: {count} times")
```

---

## Summary

### Complete Pipeline Checklist

**Training**:
- [ ] Generate validation sets (20-30 problems per stage)
- [ ] Configure training profile (test/med/prod)
- [ ] Start training with monitoring
- [ ] Monitor TensorBoard metrics
- [ ] Wait for completion (10min-2hours depending on profile)

**Validation**:
- [ ] Evaluate all checkpoints on validation set
- [ ] Select best checkpoint by robust metric (median_reward)
- [ ] Compare against baselines (random, greedy)
- [ ] Run statistical significance tests
- [ ] Document validation results

**Deployment**:
- [ ] Validate checkpoint integrity
- [ ] Update configs/prod.yaml
- [ ] Register deployment in registry.json
- [ ] Test inference latency (<10ms)
- [ ] Backup previous configuration

**Production Testing**:
- [ ] Run baseline (RL disabled)
- [ ] Run RL-enhanced GA
- [ ] Compare results (violations, convergence, runtime)
- [ ] Monitor RL usage (action distribution, fallback rate)
- [ ] Verify improvement over baseline

---

**Document Status**: ✅ Complete - Ready for production use
