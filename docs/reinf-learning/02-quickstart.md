# Quickstart Guide

Get your first RL agent trained in under 5 minutes!

## Prerequisites

```bash
# Ensure you have UV installed and dependencies synced
uv sync

# Verify GPU/system status (optional)
uv run diagnose
```

## Option 1: Unified CLI (Recommended)

### Smoke Test (~2-3 minutes)
```bash
# Train RL agent with test profile
uv run train-rl --test
```

**What happens:**
1. Loads scheduling data (courses, rooms, instructors, groups)
2. Creates 1 parallel environment (no overhead)
3. Initializes PPO agent
4. Trains for 500 timesteps (~2-3 min)
5. Evaluates on 1 test episode
6. Saves model to `models/rl_agents/rl_agent_test.zip`

**Output:**
```
INFO  RL AGENT TRAINING
INFO  Profile: test (500 timesteps)
INFO  Agent: PPO
...
INFO  Env created (1 worker, 10 pop, 30 gens)
INFO  Training...
  20% ━━━━━━━━━━━━━━━━━━━━ 100/500 [00:45<02:30, 2.5 it/s]
INFO  Training complete!
INFO  Mean reward: +12.5 ± 3.2
```

### Medium Training (~30-45 minutes)
```bash
# More thorough training
uv run train-rl --med
```

**What's different:**
- 100,000 timesteps (vs 500)
- 4 parallel envs (4x faster)
- 50 population size
- 120 generations per episode
- Better model quality

### Production Training (~1-2 hours)
```bash
# Full production-quality training
uv run train-rl --prod
```

**What's different:**
- 300,000 timesteps
- 8 parallel envs (8x faster)
- 80 population size
- 200 generations per episode
- **Thesis-quality model**

## Option 2: Interactive Launcher

```bash
# Start interactive menu
uv run launcher
```

**Select from menu:**
```
RL Training
  4. uv run train-rl --test     Smoke test (500 steps, ~2-3 min)
  5. uv run train-rl --med      Medium run (50K steps, ~30-45 min)
  6. uv run train-rl --prod     Production (100K steps, ~1-2 hrs)
  c. uv run train-rl --test --curriculum  Curriculum learning
```

Press `4` and hit Enter!

## Option 3: Python Direct

```python
# scripts/train_rl_example.py
from src.rl.training.train_script import main

# Run with test profile
import sys
sys.argv = ['train_script.py', '--profile', 'test', '--agent', 'ppo']
main()
```

```bash
python scripts/train_rl_example.py
```

## Monitor Training Progress

### TensorBoard (Recommended)

Training automatically starts TensorBoard at `http://localhost:6006`

```bash
# Or manually start
uv run tensorboard

# Then open: http://localhost:6006
```

**What you'll see:**
- Episode rewards over time
- Mean/std reward curves
- Action selection frequency heatmap
- Policy loss, value loss
- Entropy (exploration measure)

### Console Logs

Training logs to `logs/training/train_<timestamp>.log`:

```bash
# Watch logs in real-time
tail -f logs/training/train_<timestamp>.log
```

### Rich Progress Bar

In terminal, you'll see:

```
Training Progress:
  45% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 22,500/50,000 [15:30<18:45, 24.5 it/s]
```

## Check Results

### Model Saved
```bash
# Model location
ls -lh models/rl_agents/rl_agent_test.zip

# ~5-10 MB file
```

### Evaluation Metrics

Check console output:
```
Evaluation Results:
  Mean Reward: 15.3 ± 4.2
  Min/Max Reward: 8.1 / 23.7
  Episodes: 1
  Success Rate: 100%
```

### TensorBoard Graphs

Open `http://localhost:6006` and check:
- `episode_reward_mean` - Should trend upward
- `action_distribution` - Which operators agent prefers
- `policy_loss` - Should decrease over time

## Use Trained Model

### Load and Test

```python
from stable_baselines3 import PPO
from src.rl.environment import ScheduleEnv
from src.workflows.standard_run import load_input_data

# Load data
courses, rooms, instructors, groups = load_input_data("data/")

# Create env
env = ScheduleEnv(
    courses=courses,
    rooms=rooms,
    instructors=instructors,
    groups=groups,
    max_generations=100,
    population_size=50
)

# Load trained model
agent = PPO.load("models/rl_agents/rl_agent_test.zip")

# Run episode
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action, _states = agent.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    
print(f"Total reward: {total_reward:.2f}")
print(f"Best fitness: {env.get_best_fitness()}")
```

### Compare vs Baseline

Run baseline NSGA-II for comparison:
```bash
# Baseline (no RL)
uv run nsga --test

# RL-guided
uv run train-rl --test

# Then load both models and compare:
# - Best fitness achieved
# - Convergence speed
# - Hard constraint satisfaction
```

## Troubleshooting

### Training Freezes

**Problem**: No progress for 5+ minutes

**Solution**: Reduce parallel envs in config
```yaml
# configs/training/test.yaml
parallel:
  n_envs: 1  # Reduce from 4
```

### Out of Memory

**Problem**: Process killed, OOM error

**Solution**: Reduce population size
```yaml
# configs/training/test.yaml
population_size: 10  # Reduce from 16
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Run from project root
```bash
cd ~/schedule-engine
uv run train-rl --test
```

### Slow Training

**Problem**: Very slow (>10 min for test)

**Solution**: Check CPU usage, reduce n_envs to 1

## Next Steps

✅ **You've trained your first RL agent!**

**Recommended path:**

1. **Understand the system**: Read [01-overview.md](01-overview.md)

2. **Explore components**: 
   - [04-environment.md](04-environment.md) - How env works
   - [07-reward-function.md](07-reward-function.md) - How agent learns

3. **Customize training**:
   - [03-profiles.md](03-profiles.md) - Training profiles
   - [15-configuration.md](15-configuration.md) - Config parameters

4. **Analyze results**:
   - [18-tensorboard.md](18-tensorboard.md) - TensorBoard guide
   - [19-visualization.md](19-visualization.md) - Result plots

5. **Deploy to production**:
   - [17-deployment.md](17-deployment.md) - Production guide

## Quick Reference

```bash
# Smoke test
uv run train-rl --test          # 2-3 min

# Medium
uv run train-rl --med           # 30-45 min

# Production
uv run train-rl --prod          # 1-2 hours

# With curriculum
uv run train-rl --prod --curriculum

# Monitor
uv run tensorboard              # http://localhost:6006

# Interactive
uv run launcher                 # Select 4-6

# Help
uv run train-rl --help
```

Happy training! 🚀
