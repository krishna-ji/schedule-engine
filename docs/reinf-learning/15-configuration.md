# RL Configuration Reference

Complete reference for all RL-related configuration options.

## Configuration Files

```
configs/
├── base.yaml           # Shared defaults
├── test.yaml           # Test overrides
├── med.yaml            # Medium overrides
├── prod.yaml           # Production overrides
└── training/           # RL-specific configs
    ├── test.yaml       # RL test profile
    ├── med.yaml        # RL medium profile
    └── prod.yaml       # RL production profile
```

**Inheritance**:
```
base.yaml (all common settings)
  ↓
training/test.yaml (RL test overrides)
  ↓
Runtime flags (--timesteps, --agent, etc.)
```

---

## Base Configuration

**File**: `configs/base.yaml`

```yaml
# ============================================================
# Reinforcement Learning Configuration
# ============================================================
rl:
  enabled: true                    # Master killswitch
  
  # Training settings
  timesteps: 100000                # Total training timesteps
  max_steps: 100                   # Max steps per episode
  eval_episodes: 5                 # Episodes for evaluation
  save_prefix: "rl_agent"          # Model save prefix
  
  # Agent configuration
  agent:
    algorithm: "PPO"               # or "DQN"
    policy: "MlpPolicy"            # or "MultiInputPolicy"
    learning_rate: 0.0003
    gamma: 0.99                    # Discount factor
    n_steps: 2048                  # Steps before update (PPO)
    batch_size: 64
    n_epochs: 10                   # Update epochs (PPO)
    ent_coef: 0.01                 # Entropy coefficient
    clip_range: 0.2                # PPO clip range
    target_update_interval: 1000   # DQN only
    buffer_size: 100000            # DQN only
    exploration_fraction: 0.1      # DQN only
    exploration_final_eps: 0.05    # DQN only
  
  # Parallel environments
  parallel:
    n_envs: 4                      # Number of parallel envs
    use_subproc: true              # True parallelism
  
  # Reward function
  reward:
    hard_weight: 10.0              # α: Hard constraint weight
    soft_weight: 1.0               # β: Soft constraint weight
    diversity_bonus: 0.5           # γ: Diversity bonus
    feasibility_bonus: 50.0        # One-time bonus
    time_penalty: -0.01            # Per-step penalty
  
  # State normalization
  normalization:
    enabled: true
    method: "running_mean"         # or "min_max"
    clip_obs: 10.0                 # Clip to ±10
  
  # Checkpointing
  checkpoint:
    enabled: true
    frequency: 10000               # Save every 10K steps
    save_replay_buffer: true       # DQN only
  
  # Logging
  tensorboard:
    enabled: true
    log_dir: "logs/tensorboard/"
    log_interval: 100              # Log every 100 steps
  
  # Curriculum learning
  curriculum:
    enabled: false                 # Not yet implemented
    stages:
      - name: "easy"
        max_generations: 50
        population_size: 20
      - name: "medium"
        max_generations: 100
        population_size: 50
      - name: "hard"
        max_generations: 200
        population_size: 100
```

---

## Test Profile

**File**: `configs/training/test.yaml`

```yaml
profile: test

# Minimal timesteps for smoke test
timesteps: 500

# GA configuration (small for speed)
max_generations: 30
max_steps: 20
population_size: 10
eval_episodes: 1

# Single environment (no parallelism overhead)
parallel:
  n_envs: 1
  use_subproc: false  # DummyVecEnv

# Fast checkpointing
checkpoint:
  frequency: 100

# Verbose logging
debug_logging: true
debug_log_interval: 10

# Model save prefix
save_prefix: "rl_agent_test"
```

**Purpose**: Quick validation (~2-3 min)

**Command**:
```bash
uv run train-rl --test
```

---

## Medium Profile

**File**: `configs/training/med.yaml`

```yaml
profile: med

# Moderate timesteps for experiments
timesteps: 100000

# GA configuration (balanced)
max_generations: 120
max_steps: 60
population_size: 50
eval_episodes: 5

# Parallel environments (inherit from base: 4 envs)
# parallel:
#   n_envs: 4
#   use_subproc: true

# Standard checkpointing
checkpoint:
  frequency: 10000

# Model save prefix
save_prefix: "rl_agent_med"
```

**Purpose**: Hyperparameter tuning (~30-45 min)

**Command**:
```bash
uv run train-rl --med
```

---

## Production Profile

**File**: `configs/training/prod.yaml`

```yaml
profile: prod

# Maximum timesteps for best quality
timesteps: 300000

# GA configuration (full scale)
max_generations: 200
max_steps: 80
population_size: 80
eval_episodes: 10

# Maximum stable parallelism
parallel:
  n_envs: 8
  use_subproc: true

# Less frequent checkpointing (save disk space)
checkpoint:
  frequency: 25000

# Reduce log verbosity
debug_logging: true
debug_log_interval: 25

# Model save prefix
save_prefix: "rl_agent_prod"
```

**Purpose**: Final thesis results (~1-2 hours)

**Command**:
```bash
uv run train-rl --prod
```

---

## PPO Configuration

**Algorithm**: Proximal Policy Optimization

```yaml
rl:
  agent:
    algorithm: "PPO"
    policy: "MlpPolicy"
    
    # Learning
    learning_rate: 0.0003          # Adam learning rate
    gamma: 0.99                    # Discount factor
    gae_lambda: 0.95               # GAE parameter
    
    # Rollout
    n_steps: 2048                  # Steps before update
    batch_size: 64                 # Minibatch size
    n_epochs: 10                   # Gradient updates per rollout
    
    # PPO-specific
    clip_range: 0.2                # Clipping parameter
    clip_range_vf: null            # Value function clipping (optional)
    ent_coef: 0.01                 # Entropy coefficient
    vf_coef: 0.5                   # Value function coefficient
    max_grad_norm: 0.5             # Gradient clipping
    
    # Network architecture
    net_arch:
      pi: [64, 64]                 # Policy network
      vf: [64, 64]                 # Value network
    activation_fn: "tanh"          # or "relu"
```

**When to use**:
- ✅ Default choice (stable, sample-efficient)
- ✅ Continuous/discrete action spaces
- ✅ On-policy learning

---

## DQN Configuration

**Algorithm**: Deep Q-Network

```yaml
rl:
  agent:
    algorithm: "DQN"
    policy: "MlpPolicy"
    
    # Learning
    learning_rate: 0.0001          # Adam learning rate
    gamma: 0.99                    # Discount factor
    
    # Replay buffer
    buffer_size: 100000            # Experience replay size
    learning_starts: 10000         # Start learning after N steps
    batch_size: 32                 # Minibatch size
    tau: 0.005                     # Soft update coefficient
    
    # Target network
    target_update_interval: 1000   # Update target every N steps
    
    # Exploration
    exploration_fraction: 0.1      # Fraction for exploration decay
    exploration_initial_eps: 1.0   # Initial epsilon
    exploration_final_eps: 0.05    # Final epsilon
    
    # DQN variants
    double_q: true                 # Double DQN
    prioritized_replay: false      # Prioritized experience replay
    prioritized_replay_alpha: 0.6  # Priority exponent
    prioritized_replay_beta0: 0.4  # Importance sampling
    
    # Network architecture
    net_arch: [128, 128]           # Q-network layers
    activation_fn: "relu"
```

**When to use**:
- ✅ Discrete action spaces only
- ✅ Off-policy learning (sample-efficient)
- ✅ Stable exploration strategy

---

## Reward Shaping

### Default Weights

```yaml
rl:
  reward:
    hard_weight: 10.0              # Most important
    soft_weight: 1.0               # Baseline
    diversity_bonus: 0.5           # Minor bonus
    feasibility_bonus: 50.0        # One-time bonus
    time_penalty: -0.01            # Encourage speed
```

### Aggressive Hard Constraint Focus

```yaml
rl:
  reward:
    hard_weight: 100.0             # 100x more important
    soft_weight: 1.0
    diversity_bonus: 0.0           # Ignore diversity
    feasibility_bonus: 200.0       # Big bonus
    time_penalty: -0.05            # Strong time pressure
```

### Balanced Multi-Objective

```yaml
rl:
  reward:
    hard_weight: 5.0               # Less aggressive
    soft_weight: 2.0               # More soft focus
    diversity_bonus: 1.0           # Encourage exploration
    feasibility_bonus: 25.0        # Moderate bonus
    time_penalty: 0.0              # No time pressure
```

---

## State Normalization

### Running Mean (Recommended)

```yaml
rl:
  normalization:
    enabled: true
    method: "running_mean"         # Online normalization
    clip_obs: 10.0                 # Clip to ±10 std
    epsilon: 1e-8                  # Numerical stability
```

**Pros**: Adapts to changing state distribution.

### Min-Max Scaling

```yaml
rl:
  normalization:
    enabled: true
    method: "min_max"
    min_vals: [0, 0, 0, -1, 0]     # Per-feature mins
    max_vals: [100, 500, 1, 1, 1]  # Per-feature maxs
```

**Pros**: Fixed bounds, deterministic.

### No Normalization

```yaml
rl:
  normalization:
    enabled: false
```

**Use case**: When state already normalized (e.g., custom env).

---

## Checkpointing

### Frequent Checkpoints

```yaml
rl:
  checkpoint:
    enabled: true
    frequency: 5000                # Every 5K steps
    save_replay_buffer: true       # DQN only
    keep_n_latest: 5               # Keep last 5 checkpoints
```

**Use case**: Long training runs, experimentation.

### Minimal Checkpoints

```yaml
rl:
  checkpoint:
    enabled: true
    frequency: 50000               # Every 50K steps
    save_replay_buffer: false      # Save disk space
    keep_n_latest: 2               # Keep last 2 only
```

**Use case**: Production training, limited disk.

---

## TensorBoard Logging

### Verbose Logging

```yaml
rl:
  tensorboard:
    enabled: true
    log_dir: "logs/tensorboard/"
    log_interval: 50               # Every 50 steps
    log_histograms: true           # Weight histograms
    log_gradients: true            # Gradient norms
```

**Use case**: Debugging, hyperparameter tuning.

### Minimal Logging

```yaml
rl:
  tensorboard:
    enabled: true
    log_dir: "logs/tensorboard/"
    log_interval: 1000             # Every 1K steps
    log_histograms: false
    log_gradients: false
```

**Use case**: Production runs, reduce overhead.

---

## Curriculum Learning

**Status**: ⚠️ Not yet implemented (placeholder config)

```yaml
rl:
  curriculum:
    enabled: true
    
    stages:
      # Stage 1: Easy problems
      - name: "easy"
        max_generations: 50        # Short episodes
        population_size: 20        # Small pop
        constraints:
          hard_only: true          # Ignore soft initially
        
      # Stage 2: Medium problems
      - name: "medium"
        max_generations: 100
        population_size: 50
        constraints:
          hard_only: false         # Include soft
        
      # Stage 3: Hard problems
      - name: "hard"
        max_generations: 200
        population_size: 100
        constraints:
          hard_only: false
          adaptive: true           # Adaptive difficulty
    
    # Transition criteria
    transition:
      method: "performance"        # or "timesteps"
      threshold: 0.8               # 80% success rate
      min_episodes: 100            # Min episodes per stage
```

---

## Override via CLI

### Common Overrides

```bash
# Change timesteps
uv run train-rl --test --timesteps 1000

# Change algorithm
uv run train-rl --med --agent DQN

# Change learning rate
uv run train-rl --prod --learning-rate 0.001

# Custom config file
python src/rl/training/train_script.py --config path/to/custom.yaml
```

### Multiple Overrides

```bash
# Combine flags
uv run train-rl --med \
  --timesteps 50000 \
  --agent PPO \
  --learning-rate 0.0005 \
  --n-envs 8 \
  --name "experiment-1"
```

---

## Environment Variables

```bash
# Override device
export RL_DEVICE=cuda

# Override log directory
export RL_LOG_DIR=logs/my_experiment/

# Override model directory
export RL_MODEL_DIR=models/custom/
```

**Access in code**:
```python
import os

device = os.getenv("RL_DEVICE", "cpu")
log_dir = os.getenv("RL_LOG_DIR", "logs/tensorboard/")
```

---

## Validation

### Check Config Syntax

```bash
# Verify YAML syntax
uv run verify-config
```

### Validate RL Config

```python
from src.config import get_config

config = get_config()

# Check required fields
assert config.rl.enabled, "RL not enabled"
assert config.rl.timesteps > 0, "Invalid timesteps"
assert config.rl.agent.algorithm in ["PPO", "DQN"], "Invalid algorithm"
```

---

## Common Patterns

### 1. Quick Test Run

```yaml
# configs/training/quick.yaml
profile: quick
timesteps: 100
max_generations: 10
max_steps: 10
population_size: 5
eval_episodes: 1
parallel:
  n_envs: 1
  use_subproc: false
```

```bash
python src/rl/training/train_script.py --config configs/training/quick.yaml
```

### 2. Ablation Study

```yaml
# configs/training/ablation_no_diversity.yaml
profile: ablation
timesteps: 50000
reward:
  hard_weight: 10.0
  soft_weight: 1.0
  diversity_bonus: 0.0    # Disable diversity
  feasibility_bonus: 50.0
  time_penalty: -0.01
```

### 3. Hyperparameter Grid Search

```python
# scripts/hyperparameter_search.py
import yaml

learning_rates = [0.0001, 0.0003, 0.001]
ent_coefs = [0.0, 0.01, 0.05]

for lr in learning_rates:
    for ent in ent_coefs:
        config = {
            "profile": "grid_search",
            "timesteps": 50000,
            "agent": {
                "learning_rate": lr,
                "ent_coef": ent
            }
        }
        
        with open(f"configs/training/grid_lr{lr}_ent{ent}.yaml", "w") as f:
            yaml.dump(config, f)
```

---

## Next Steps

- **[02-quickstart.md](02-quickstart.md)** - Run first training
- **[03-profiles.md](03-profiles.md)** - Profile comparison
- **[08-trainer.md](08-trainer.md)** - Training system
- **[09-agents.md](09-agents.md)** - PPO/DQN details
- **[16-checkpoints.md](16-checkpoints.md)** - Model saving
- **[18-tensorboard.md](18-tensorboard.md)** - Monitor training
