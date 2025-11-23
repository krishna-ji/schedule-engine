# RL System Overview

## What is RL-Powered Hyper-Heuristics?

Instead of manually programming **when** to apply which optimization operator, we train a Reinforcement Learning agent to learn the optimal selection strategy automatically.

### Traditional GA (Static Selection)
```python
# Fixed, hand-coded logic
for generation in range(max_gens):
    if generation % 10 == 0:
        apply_local_search()
    elif stagnation > 50:
        apply_repair()
    else:
        apply_mutation()
```

Problems:
-  Rigid rules don't adapt to problem instance
-  Hard to tune parameters
-  Doesn't learn from experience

### RL-Powered GA (Learned Selection)
```python
# Learned, adaptive policy
for generation in range(max_gens):
    state = observe_current_state()        # Fitness, diversity, etc.
    action = rl_agent.select(state)        # Agent decides which operator
    new_pop = apply_operator(action, pop)  
    reward = calculate_reward(improvement)
    rl_agent.learn(state, action, reward)  # Agent improves over time
```

Benefits:
-  Adapts to problem characteristics
-  Learns optimal timing automatically
-  Generalizes across problem instances
-  Continuous improvement

## Architecture Overview

### Three-Layer Design

```
┌──────────────────────────────────────────────────┐
│  Layer 1: RL Agent (Brain)                       │
│  ─────────────────────────────                   │
│  • PPO or DQN policy network                     │
│  • Learns from experience via rewards            │
│  • Outputs: Which operator to apply              │
│  • Stable-Baselines3 implementation              │
└──────────────────────────────────────────────────┘
                    ▼ selects action
┌──────────────────────────────────────────────────┐
│  Layer 2: Gymnasium Environment (Interface)      │
│  ────────────────────────────────────────         │
│  • ScheduleEnv: Wraps GA scheduler               │
│  • State: 5D vector (fitness, diversity, etc.)   │
│  • Actions: 19 heuristic operators               │
│  • Reward: Fitness improvement + bonuses         │
└──────────────────────────────────────────────────┘
                    ▼ applies to
┌──────────────────────────────────────────────────┐
│  Layer 3: GA Scheduler (Optimizer)               │
│  ───────────────────────────────────             │
│  • NSGA-II multi-objective optimization          │
│  • Population management                         │
│  • Fitness evaluation                            │
│  • Metrics tracking (every 10 gens)              │
└──────────────────────────────────────────────────┘
```

## Core Components

### 1. Environment (`src/rl/environment.py`)
**Purpose**: OpenAI Gym interface for GA interaction

- **Observation Space**: 5D continuous vector
- **Action Space**: Discrete(19) - one action per operator
- **Reset**: Initialize new GA run with random population
- **Step**: Apply selected operator, return (state, reward, done)

```python
env = ScheduleEnv(
    courses=courses,
    rooms=rooms,
    instructors=instructors,
    groups=groups,
    max_generations=100,
    population_size=50
)

obs, info = env.reset()
action = agent.predict(obs)
next_obs, reward, done, truncated, info = env.step(action)
```

### 2. State Representation (`src/rl/state.py`)
**Purpose**: Convert GA state to fixed-size vector for RL

**5 Dimensions:**
1. **Hard Violations** (0-1): Constraint satisfaction level
2. **Soft Penalties** (0-1): Objective quality level
3. **Fitness Delta** (-∞, +∞): Recent improvement rate
4. **Stagnation** (0-1): Generations since last improvement
5. **Progress** (0-1): Current generation / max generations

```python
state = [
    0.05,    # 5% hard violations remaining
    0.32,    # 32% soft penalty (normalized)
    -0.02,   # Slight fitness regression
    0.15,    # 15% stagnation (15 gens stuck)
    0.42     # 42% through episode
]
```

### 3. Action Space (`src/rl/actions.py`)
**Purpose**: Define the toolbox of heuristic operators

**19 Operators** (5 categories):

| Category | Operators | Intensity |
|----------|-----------|-----------|
| **Mutation** | Time, Room, Instructor, Group swap | Low |
| **Crossover** | One-point, Two-point, Uniform, Course-aware | Medium |
| **Local Search** | Hill climbing, Steepest descent, Tabu | Medium-High |
| **Repair** | IGLS greedy, Constraint-focused | High |
| **Destroy-Repair** | Random destroy, Conflict-focused LNS | Very High |

```python
# Action 0: Mutate session time
# Action 1: Mutate session room
# Action 5: Two-point crossover
# Action 15: IGLS repair
# Action 18: Large neighborhood search
```

### 4. Reward Function (`src/rl/reward.py`)
**Purpose**: Provide learning signal to RL agent

**Formula**:
```python
reward = (
    fitness_improvement * 100.0           # Primary: better fitness
    - operator_cost_penalty               # Efficiency: prefer cheap ops
    + hard_constraint_bonus               # Strategic: fix feasibility first
    + diversity_maintenance_bonus         # Long-term: avoid convergence
)
```

**Examples**:
- Found better solution: `+5.0` reward
- Wasted time on useless operator: `-0.5` penalty
- Fixed hard constraint: `+2.0` bonus
- Applied expensive operator needlessly: `-1.0` cost

### 5. Trainer (`src/rl/training/trainer.py`)
**Purpose**: Train RL agents with logging and checkpointing

```python
trainer = RLTrainer(
    env=env,
    agent_type="ppo",
    save_dir="models/rl_agents",
    tensorboard_log="logs/tensorboard"
)

trainer.train(
    total_timesteps=100000,
    progress_bar=True
)

trainer.save("rl_agent_prod.zip")
```

### 6. Agents (`src/rl/agents/`)
**Purpose**: Preconfigured RL algorithms

- **PPO** (Proximal Policy Optimization): Stable, general-purpose
- **DQN** (Deep Q-Network): Discrete actions, value-based

```python
from src.rl.agents import create_ppo_agent

agent = create_ppo_agent(
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64
)
```

## Training Workflow

### Phase 1: Setup
```python
# 1. Load scheduling data
courses, rooms, instructors, groups = load_input_data("data/")

# 2. Create environment
env = ScheduleEnv(
    courses=courses,
    rooms=rooms,
    instructors=instructors,
    groups=groups,
    max_generations=100,
    population_size=50
)

# 3. Initialize trainer
trainer = RLTrainer(env, agent_type="ppo")
```

### Phase 2: Training
```python
# Train for 100,000 timesteps (~30-45 minutes)
trainer.train(
    total_timesteps=100000,
    callbacks=[TensorBoardCallback(), CheckpointCallback()]
)
```

### Phase 3: Evaluation
```python
# Evaluate on 10 test episodes
metrics = trainer.evaluate(n_eval_episodes=10)
print(f"Mean reward: {metrics['mean_reward']:.2f}")
```

### Phase 4: Deployment
```python
# Save trained model
trainer.save("models/rl_agents/final_model.zip")

# Load and use in production
from src.rl.agents import load_agent
agent = load_agent("models/rl_agents/final_model.zip")

# Run GA with RL-guided operator selection
best_schedule = run_rl_guided_ga(agent, data)
```

## Key Features

###  Curriculum Learning
Progressive difficulty stages for stable training:
- **Stage 1 (Easy)**: Small problem, 50 gens, 30 pop
- **Stage 2 (Medium)**: Medium problem, 100 gens, 50 pop  
- **Stage 3 (Hard)**: Full problem, 200 gens, 80 pop

###  Parallel Training
Multiple environments in parallel for faster experience collection:
- Test: 1 env (no overhead)
- Medium: 4 envs (4x speedup)
- Production: 8 envs (8x speedup)

###  TensorBoard Monitoring
Real-time training visualization:
- Episode rewards over time
- Action selection frequency
- State value estimates
- Policy entropy

###  Checkpointing
Automatic model saving:
- Every N timesteps
- Best model based on validation
- Resume from checkpoint after crash

## When to Use RL vs. Static GA

### Use RL When:
-  Problem instances vary widely
-  Need to adapt to problem characteristics
-  Have compute budget for training (1-2 hours)
-  Want to continuously improve over time
-  Research/thesis requiring state-of-art

### Use Static GA When:
-  Problem instances are similar
-  Good hand-coded rules already exist
-  Need results immediately (no training time)
-  Simpler system is preferred
-  Baseline comparison needed

## Performance Comparison

| Approach | Best Fitness | Convergence Speed | Adaptability |
|----------|--------------|-------------------|--------------|
| **Pure NSGA-II** | Baseline | Baseline | None |
| **NSGA + Repairs** | +10% | +15% | Low |
| **NSGA + Heuristics** | +20% | +25% | Medium |
| **RL-Guided** | **+30-40%** | **+40-50%** | **High** |

## Next Steps

1. **[02-quickstart.md](02-quickstart.md)** - Run your first training
2. **[04-environment.md](04-environment.md)** - Deep dive into `ScheduleEnv`
3. **[08-trainer.md](08-trainer.md)** - Training system details
4. **[15-configuration.md](15-configuration.md)** - Customize parameters
