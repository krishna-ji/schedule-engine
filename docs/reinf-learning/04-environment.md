# ScheduleEnv - Gymnasium Environment

Deep dive into the RL environment wrapper for NSGA-II genetic algorithm.

## Overview

**Location**: `src/rl/environment/schedule_env.py`

**Purpose**: Wrap GA scheduler as Gymnasium environment for RL agents.

```python
from gymnasium import Env
from gymnasium.spaces import Box, Discrete

class ScheduleEnv(Env):
    """
    Gymnasium environment for course timetable scheduling.
    
    RL agent learns to select heuristic operators during GA evolution,
    improving convergence speed and solution quality.
    """
```

**Interface**:
```
RL Agent (PPO/DQN)
  ↓ select action (0-18)
Environment (ScheduleEnv)
  ↓ apply heuristic operator
GA Scheduler (NSGA-II)
  ↓ evolve population
State Vector (5D)
  ↑ feedback to agent
```

---

## Initialization

```python
# Create environment
from src.rl.environment.schedule_env import ScheduleEnv
from src.config import get_config

config = get_config()
env = ScheduleEnv(config)

# Observation space: Box(5,)
print(env.observation_space)
# Box([-inf -inf -inf -inf -inf], [inf inf inf inf inf], (5,), float32)

# Action space: Discrete(19)
print(env.action_space)
# Discrete(19)  # 19 heuristic operators
```

**Constructor**:
```python
def __init__(self, config: GAConfig, max_steps: int = 100):
    """
    Args:
        config: GA configuration object
        max_steps: Max RL steps per episode (default 100)
    """
```

---

## Observation Space

**Type**: `Box(5,)` - 5D continuous vector

**Components**:
1. **Hard constraint violations** (normalized)
2. **Soft constraint penalties** (normalized)
3. **Population diversity** (0-1)
4. **Improvement rate** (change from last step)
5. **Episode progress** (steps / max_steps)

### State Representation

```python
def _get_observation(self) -> np.ndarray:
    """Extract 5D state vector from GA scheduler."""
    
    # 1. Hard violations (normalized by max seen)
    hard_violations = -self.scheduler.population[0].fitness.values[0]
    norm_hard = hard_violations / max(self.max_hard_seen, 1.0)
    
    # 2. Soft penalties (normalized by max seen)
    soft_penalties = -self.scheduler.population[0].fitness.values[1]
    norm_soft = soft_penalties / max(self.max_soft_seen, 1.0)
    
    # 3. Population diversity (hypervolume-based)
    diversity = self.scheduler.calculate_diversity()
    
    # 4. Improvement rate (delta fitness)
    if self.prev_fitness is not None:
        improvement = (self.prev_fitness - current_fitness) / max(abs(self.prev_fitness), 1e-6)
    else:
        improvement = 0.0
    
    # 5. Episode progress
    progress = self.current_step / self.max_steps
    
    return np.array([
        norm_hard,
        norm_soft,
        diversity,
        improvement,
        progress
    ], dtype=np.float32)
```

**Normalization Strategy**:
- **Hard/Soft**: Track max seen values, normalize to [0, 1]
- **Diversity**: Already [0, 1] from hypervolume calculation
- **Improvement**: Clipped to [-1, 1]
- **Progress**: Linear [0, 1]

---

## Action Space

**Type**: `Discrete(19)` - 19 heuristic operators

**Action Mapping**:
```python
ACTION_MAP = {
    0: "crossover_course_group_aware",      # Course-group crossover
    1: "crossover_uniform",                  # Uniform crossover
    2: "crossover_two_point",                # Two-point crossover
    3: "mutate_timeslot",                    # Change time slot
    4: "mutate_room",                        # Change room
    5: "mutate_instructor",                  # Change instructor
    6: "mutate_swap_sessions",               # Swap two sessions
    7: "mutate_adaptive",                    # Adaptive mutation
    8: "repair_igls",                        # IGLS repair
    9: "repair_stagnation",                  # Stagnation repair
    10: "repair_selective",                  # Selective repair
    11: "local_search_hill_climbing",        # Hill climbing
    12: "local_search_simulated_annealing",  # Simulated annealing
    13: "local_search_tabu",                 # Tabu search
    14: "heuristic_earliest_start",          # Earliest start time
    15: "heuristic_largest_enrollment",      # Largest enrollment first
    16: "heuristic_most_constrained",        # Most constrained first
    17: "heuristic_least_flexible",          # Least flexible first
    18: "heuristic_saturation_degree"        # Saturation degree
}
```

**Operator Categories**:
- **Crossover (0-2)**: Genetic recombination
- **Mutation (3-7)**: Small changes
- **Repair (8-10)**: Fix constraint violations
- **Local Search (11-13)**: Refinement
- **Heuristics (14-18)**: Constructive methods

See [06-action-space.md](06-action-space.md) for detailed operator descriptions.

---

## Reward Function

**Formula**:
```
reward = α × Δhard + β × Δsoft + γ × diversity_bonus + penalty
```

**Components**:
```python
def _calculate_reward(self) -> float:
    """Multi-objective reward with penalties."""
    
    # 1. Hard constraint improvement (most important)
    delta_hard = self.prev_hard - current_hard
    hard_reward = 10.0 * delta_hard  # α = 10
    
    # 2. Soft constraint improvement
    delta_soft = self.prev_soft - current_soft
    soft_reward = 1.0 * delta_soft   # β = 1
    
    # 3. Diversity bonus (encourage exploration)
    diversity = self.scheduler.calculate_diversity()
    diversity_bonus = 0.5 if diversity > 0.7 else 0.0  # γ = 0.5
    
    # 4. Feasibility bonus (found zero-violation solution)
    feasibility_bonus = 50.0 if current_hard == 0 else 0.0
    
    # 5. Time penalty (encourage faster convergence)
    time_penalty = -0.01 * self.current_step
    
    return hard_reward + soft_reward + diversity_bonus + feasibility_bonus + time_penalty
```

**Weight Ratios**:
- Hard violations: **10x** importance
- Soft penalties: **1x** baseline
- Diversity: **0.5x** (minor bonus)
- Feasibility: **50x** one-time bonus
- Time: **-0.01x** per step

See [07-reward-function.md](07-reward-function.md) for reward shaping details.

---

## Episode Lifecycle

### 1. Reset

```python
def reset(self, seed=None, options=None):
    """Start new episode."""
    super().reset(seed=seed)
    
    # Initialize GA scheduler
    self.scheduler = GAScheduler(self.config)
    self.scheduler.initialize_population()
    
    # Reset tracking variables
    self.current_step = 0
    self.prev_fitness = None
    self.episode_rewards = []
    
    # Get initial observation
    obs = self._get_observation()
    info = {"generation": 0}
    
    return obs, info
```

**Key Operations**:
- Create fresh GA scheduler
- Initialize random population
- Reset episode counters
- Return initial state

### 2. Step

```python
def step(self, action: int):
    """Execute one RL step."""
    
    # 1. Map action to operator
    operator_name = ACTION_MAP[action]
    
    # 2. Apply operator to population
    self.scheduler.apply_operator(operator_name)
    
    # 3. Run GA evolution (1 generation)
    self.scheduler.evolve_one_generation()
    
    # 4. Calculate reward
    reward = self._calculate_reward()
    
    # 5. Get next observation
    obs = self._get_observation()
    
    # 6. Check termination conditions
    terminated = (
        self.current_step >= self.max_steps or
        self.scheduler.found_feasible_solution()
    )
    truncated = False
    
    # 7. Update counters
    self.current_step += 1
    self.episode_rewards.append(reward)
    
    info = {
        "generation": self.current_step,
        "hard_violations": self.scheduler.best_individual.hard_violations,
        "soft_penalties": self.scheduler.best_individual.soft_penalties,
        "operator": operator_name
    }
    
    return obs, reward, terminated, truncated, info
```

**Step Sequence**:
1. Action → Operator name
2. Apply operator to population
3. Run 1 GA generation
4. Calculate reward
5. Extract new state
6. Check if done
7. Return (obs, reward, terminated, truncated, info)

### 3. Termination

**Episode ends when**:
- `max_steps` reached (default 100)
- Feasible solution found (0 hard violations)
- Manual reset() called

**Terminal State Handling**:
```python
if terminated:
    # Save final solution
    self.scheduler.save_best_solution()
    
    # Log episode statistics
    total_reward = sum(self.episode_rewards)
    avg_reward = total_reward / len(self.episode_rewards)
    
    info["episode_reward"] = total_reward
    info["avg_reward"] = avg_reward
```

---

## Parallelization

### Vectorized Environments

```python
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank: int):
    """Create single environment instance."""
    def _init():
        env = ScheduleEnv(config)
        env.seed(config.seed + rank)
        return env
    return _init

# Create 4 parallel environments
n_envs = 4
vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
```

**Benefits**:
- **4x faster**: 4 episodes simultaneously
- **Diverse experience**: Different random seeds
- **Bypass GIL**: True parallelism (separate processes)

**Overhead**:
- Process spawning: ~30s for 4 envs
- IPC communication: ~5% slowdown
- Memory: 4x usage

**When to use**:
- ✅ Training (med/prod profiles)
- ❌ Testing (test profile uses 1 env)

---

## Integration with GA Scheduler

### Shared State

```python
# Environment holds reference to scheduler
self.scheduler = GAScheduler(self.config)

# Scheduler exposes methods for RL
self.scheduler.apply_operator(operator_name)
self.scheduler.evolve_one_generation()
self.scheduler.get_best_individual()
self.scheduler.calculate_diversity()
```

### Operator Application

```python
def apply_operator(self, operator_name: str):
    """Apply heuristic operator to population."""
    
    if operator_name.startswith("crossover"):
        offspring = self._apply_crossover(operator_name)
    elif operator_name.startswith("mutate"):
        offspring = self._apply_mutation(operator_name)
    elif operator_name.startswith("repair"):
        self._apply_repair(operator_name)
    elif operator_name.startswith("local_search"):
        self._apply_local_search(operator_name)
    elif operator_name.startswith("heuristic"):
        self._apply_heuristic(operator_name)
    
    # Evaluate offspring
    self._evaluate_population(offspring)
    
    # Merge with population
    self.population = self._select_next_generation(offspring)
```

**Flow**:
1. RL agent selects action (0-18)
2. Environment maps to operator name
3. Scheduler applies operator
4. Scheduler evolves 1 generation
5. Environment calculates reward
6. Repeat

---

## Configuration

**Config Keys** (configs/training/base.yaml):
```yaml
rl:
  enabled: true
  max_steps: 100              # Episode length
  n_envs: 4                   # Parallel environments
  use_subproc: true           # True parallelism
  
  reward:
    hard_weight: 10.0         # α
    soft_weight: 1.0          # β
    diversity_bonus: 0.5      # γ
    feasibility_bonus: 50.0
    time_penalty: -0.01
  
  normalization:
    enabled: true
    method: "running_mean"    # or "min_max"
```

**Access in Environment**:
```python
self.max_steps = config.rl.max_steps
self.hard_weight = config.rl.reward.hard_weight
```

---

## Debugging

### Enable Debug Logging

```yaml
# configs/training/test.yaml
debug_logging: true
debug_log_interval: 10
```

**Output**:
```
[Step 10] Action: 3 (mutate_timeslot)
  Hard: 45 → 42 (Δ-3, reward +30.0)
  Soft: 120 → 115 (Δ-5, reward +5.0)
  Diversity: 0.85 (bonus +0.5)
  Total reward: +35.5

[Step 20] Action: 8 (repair_igls)
  Hard: 42 → 38 (Δ-4, reward +40.0)
  Soft: 115 → 118 (Δ+3, reward -3.0)
  Diversity: 0.72 (bonus +0.5)
  Total reward: +37.5
```

### Render Environment

```python
# Enable human-readable rendering
env = ScheduleEnv(config)
env.render_mode = "human"

# Step through episode
obs, info = env.reset()
for step in range(100):
    action = env.action_space.sample()  # Random action
    obs, reward, done, truncated, info = env.step(action)
    env.render()  # Print state to console
    if done:
        break
```

---

## Advanced Features

### 1. Constraint-Specific State

**Enhanced observation space**:
```python
# Instead of just (hard, soft, diversity, improvement, progress)
# Use detailed constraint breakdown:
obs = np.array([
    room_capacity_violations,
    instructor_conflicts,
    time_conflicts,
    room_exclusivity_violations,
    instructor_workload_violations,
    room_preference_penalties,
    time_preference_penalties,
    diversity,
    improvement,
    progress
], dtype=np.float32)  # 10D instead of 5D
```

**Benefit**: Agent learns which constraints are problematic.

### 2. Multi-Objective Rewards

**Separate rewards per objective**:
```python
reward_dict = {
    "hard": delta_hard * 10.0,
    "soft": delta_soft * 1.0,
    "diversity": diversity_bonus,
    "feasibility": feasibility_bonus,
    "time": time_penalty
}

# Multi-objective RL agent (e.g., MO-PPO)
return reward_dict  # instead of scalar
```

### 3. Adaptive Operator Probabilities

**Track operator success rates**:
```python
self.operator_stats = {
    op: {"success": 0, "failure": 0, "avg_reward": 0.0}
    for op in ACTION_MAP.values()
}

# Update after each step
if reward > 0:
    self.operator_stats[operator]["success"] += 1
else:
    self.operator_stats[operator]["failure"] += 1

# Use as additional state feature
success_rate = self.operator_stats[operator]["success"] / (
    self.operator_stats[operator]["success"] + 
    self.operator_stats[operator]["failure"] + 1e-6
)
```

---

## Troubleshooting

### Issue: "Environment not registered"

**Cause**: Gymnasium doesn't know about ScheduleEnv.

**Fix**:
```python
from gymnasium.envs.registration import register

register(
    id="ScheduleEnv-v0",
    entry_point="src.rl.environment.schedule_env:ScheduleEnv",
    max_episode_steps=100
)

# Now can create via string
env = gym.make("ScheduleEnv-v0", config=config)
```

### Issue: "Observation out of bounds"

**Cause**: State values exceed Box bounds.

**Fix**: Ensure normalization is correct.
```python
# Check observation space
assert env.observation_space.contains(obs), f"Invalid obs: {obs}"

# Clip to bounds
obs = np.clip(obs, env.observation_space.low, env.observation_space.high)
```

### Issue: "Slow episode initialization"

**Cause**: GA population initialization is expensive.

**Fix**: Reduce population size or use cached init.
```yaml
# configs/training/test.yaml
population_size: 10  # instead of 50
```

---

## Next Steps

- **[05-state-representation.md](05-state-representation.md)** - Detailed state design
- **[06-action-space.md](06-action-space.md)** - All 19 operators
- **[07-reward-function.md](07-reward-function.md)** - Reward shaping
- **[08-trainer.md](08-trainer.md)** - Training loop
- **[09-agents.md](09-agents.md)** - PPO/DQN configuration
