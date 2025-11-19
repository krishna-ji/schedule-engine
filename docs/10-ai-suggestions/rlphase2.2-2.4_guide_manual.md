# Phase 2.2-2.4 Implementation Guide: From Basics to Production

**Your Current Status**:  Phase 2.1 Complete - You have a working Gym environment  
**What's Next**: Train RL agents, deploy them, and evaluate performance  
**Target**: Production-ready RL system integrated with your GA scheduler

---

## Table of Contents

1. [RL Basics: What You Need to Know](#rl-basics)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Phase 2.2: Training Infrastructure](#phase-22-training)
4. [Phase 2.3: Deployment & Integration](#phase-23-deployment)
5. [Phase 2.4: Evaluation & Comparison](#phase-24-evaluation)
6. [Phase 3: Advanced RL Features](#phase-3-advanced-features)
7. [Testing & Validation](#testing-validation)
8. [Step-by-Step Commands](#step-by-step-commands)
9. [Troubleshooting Guide](#troubleshooting)

---

## RL Basics: What You Need to Know

### What is Reinforcement Learning?

**Reinforcement Learning** is a machine learning paradigm where an agent learns to make sequential decisions by interacting with an environment to maximize cumulative reward.

Think of RL like training a dog:
- **State** $s_t$: What the dog sees (your GA's current population quality)
- **Action** $a_t$: What the dog does (which heuristic to apply)
- **Reward** $r_t$: Treat or scolding (fitness improvement or not)
- **Learning**: Dog learns which actions get treats in which situations

### The RL Framework

In formal terms, we model the problem as a **Markov Decision Process (MDP)**:

$$\text{MDP} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

Where:
- $\mathcal{S}$: State space (all possible GA population states)
- $\mathcal{A}$: Action space (20 heuristics: 0=no-op, 1-19=heuristics)
- $\mathcal{P}$: Transition probability $P(s_{t+1} | s_t, a_t)$
- $\mathcal{R}$: Reward function $R(s_t, a_t, s_{t+1})$
- $\gamma \in [0,1]$: Discount factor (default: 0.99)

**Goal**: Learn policy $\pi(a|s)$ that maximizes expected cumulative reward:

$$J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

### Your RL Setup

```
┌─────────────────────────────────────────────────────────┐
│  YOUR GA SCHEDULER (existing code)                      │
│  - Has population of schedules                          │
│  - Has 19 heuristics to improve schedules               │
│  - Currently picks heuristics randomly or fixed order   │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  RL AGENT (what you're building)                        │
│  - Observes: population quality (25 numbers)            │
│  - Decides: which heuristic to use next (0-19)          │
│  - Gets reward: +1 if improved, -0.5 if worse           │
│  - Learns: "use heuristic X when in situation Y"        │
└─────────────────────────────────────────────────────────┘
```

### Why Use RL?

**Without RL** (current):
```python
# Random selection - stupid
heuristic = random.choice([0, 1, 2, ..., 19])

# Fixed order - inflexible
for heuristic in [0, 1, 2, ..., 19]:
    apply_heuristic(heuristic)
```

**With RL** (smart):
```python
# RL agent learns when to use which heuristic
state = observe_population()  # [fitness=0.8, diversity=0.5, ...]
action = rl_agent.predict(state)  # "Use heuristic 5 now!"
# Agent learned: "When diversity is low, use perturbation heuristic"
```

---

## Mathematical Foundations

### 1. State Encoding

Your **state vector** $s_t \in \mathbb{R}^{25}$ encodes GA population quality:

$$s_t = \begin{bmatrix}
f_{\text{best}} & f_{\text{worst}} & f_{\text{mean}} & f_{\text{median}} & f_{\text{std}} \\
v_{\text{hard}} & v_{\text{soft}} & v_{\text{avg}} & v_{\text{ratio}} & v_{\text{std}} \\
d_{\text{unique}} & d_{\text{avg}} & d_{\text{hamming}} & d_{\text{entropy}} & d_{\text{cluster}} \\
p_{\text{gen}} & p_{\text{stag}} & p_{\text{conv}} & i_{\text{rate}} & i_{\text{trend}} \\
h_{\text{avg\_reward}} & h_{\text{success\_rate}} & h_{\text{last\_action}} & m_{\text{time}} & m_{\text{eval}}
\end{bmatrix}^T$$

All features normalized to $[0, 1]$ using min-max scaling:

$$s_{\text{norm}} = \frac{s - s_{\min}}{s_{\max} - s_{\min}}$$

### 2. Reward Function

Multi-component reward balances fitness improvement, diversity, and efficiency:

$$R(s_t, a_t, s_{t+1}) = w_f \cdot R_f + w_d \cdot R_d - w_t \cdot R_t$$

Where:
- **Fitness reward** (primary objective):
  $$R_f = \frac{f_{\text{best}}^{(t)} - f_{\text{best}}^{(t+1)}}{f_{\text{best}}^{(t)}} \in [-1, 1]$$

- **Diversity bonus** (maintain exploration):
  $$R_d = \frac{d_{\text{unique}}^{(t+1)} - d_{\text{unique}}^{(t)}}{|P|} \in [-1, 1]$$

- **Time penalty** (encourage efficiency):
  $$R_t = \frac{\Delta t}{\Delta t_{\max}} \in [0, 1]$$

Default weights: $w_f = 1.0$, $w_d = 0.1$, $w_t = 0.01$

### 3. Policy Gradient (PPO Algorithm)

**Proximal Policy Optimization** updates policy to maximize:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is probability ratio
- $\hat{A}_t$ is advantage estimate (how good action $a_t$ was)
- $\epsilon = 0.2$ is clipping parameter (prevents large updates)

**Advantage estimation** using Generalized Advantage Estimation (GAE):

$$\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ and $\lambda = 0.95$

### 4. Value-Based Learning (DQN Algorithm)

**Deep Q-Network** learns action-value function:

$$Q(s, a) = \mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_t \mid s_0=s, a_0=a\right]$$

**Loss function** (TD error minimization):

$$L(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\left[\left(r + \gamma\max_{a'}Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right]$$

Where:
- $\mathcal{D}$ is replay buffer (experience memory)
- $\theta^-$ is target network (updated slowly for stability)
- $\epsilon$-greedy exploration: $\epsilon_t = \epsilon_{\max} \cdot \text{decay}^t$

### 5. Curriculum Learning

Progressive difficulty training with adaptive stage transitions:

$$\text{Stage}_i = \{D_i, T_i, \tau_i\}$$

Where:
- $D_i$: Problem difficulty (number of courses: 10 → 20 → 40)
- $T_i$: Training timesteps for stage $i$
- $\tau_i$: Performance threshold for advancement

**Transition criterion**:

$$\text{Advance if: } \bar{R}_{\text{recent}} \geq \tau_i \text{ and } \text{var}(R_{\text{recent}}) < \sigma_{\max}$$

---

## Phase 2.2: Training Infrastructure

### What is Training?

**Training** = Running the GA scheduler many times while the RL agent learns from experience.

Like practicing chess:
- Play 1000 games (training episodes)
- Learn from mistakes (bad heuristic choices)
- Get better over time (higher rewards)

### 2.2.1: Implement Basic Trainer

**What**: A script that trains your RL agent  
**Why**: You need to actually teach the agent  
**Where**: `src/rl/training/trainer.py`

#### What This File Does

```python
# Pseudocode of what trainer.py will do:
def train_rl_agent():
    # 1. Create environment
    env = ScheduleEnv(population, context)
    
    # 2. Create agent
    agent = PPO("MlpPolicy", env)
    
    # 3. Train for many episodes
    for episode in range(1000):
        state = env.reset()  # Start new GA run
        
        for step in range(100):
            action = agent.predict(state)  # Pick heuristic
            state, reward, done = env.step(action)  # Apply it
            agent.learn()  # Update agent's brain
        
        if episode % 100 == 0:
            agent.save(f"checkpoint_{episode}.zip")
    
    # 4. Save final model
    agent.save("trained_model.zip")
```

#### Implementation Steps

**Step 1**: Create the trainer class structure

```python
# src/rl/training/trainer.py

from stable_baselines3 import PPO
from src.rl.gym_env import ScheduleEnv
from src.rl.agents import create_ppo_agent
from pathlib import Path
import time

class RLTrainer:
    """Trains RL agent to select heuristics."""
    
    def __init__(self, env: ScheduleEnv, agent_type: str = "ppo"):
        self.env = env
        self.agent_type = agent_type
        self.agent = None
        
    def train(self, total_timesteps: int = 100000):
        """Main training loop."""
        print(f"Starting training for {total_timesteps} timesteps...")
        
        # Create agent
        self.agent = create_ppo_agent(self.env, verbose=1)
        
        # Train
        start_time = time.time()
        self.agent.learn(
            total_timesteps=total_timesteps,
            progress_bar=True,
        )
        elapsed = time.time() - start_time
        
        print(f"Training completed in {elapsed:.1f}s")
        
        return self.agent
    
    def save_model(self, path: str):
        """Save trained model."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(path)
        print(f"Model saved to {path}")
```

**Step 2**: Create a training script you can run

```python
# src/rl/training/train_script.py

from src.encoder import load_scheduling_data
from src.rl.gym_env import create_schedule_env
from src.rl.training.trainer import RLTrainer

def main():
    # Load your data
    print("Loading scheduling data...")
    context = load_scheduling_data("data/")
    
    # Create initial population (use your existing GA code)
    from src.core.ga_scheduler import GAScheduler
    scheduler = GAScheduler(context)
    initial_pop = scheduler.create_initial_population(pop_size=50)
    
    # Create environment
    print("Creating RL environment...")
    env = create_schedule_env(
        initial_population=initial_pop,
        context=context,
        max_generations=100,  # Short episodes for training
        max_steps_per_episode=50,
    )
    
    # Create trainer
    trainer = RLTrainer(env, agent_type="ppo")
    
    # Train agent
    print("Training agent...")
    trainer.train(total_timesteps=50000)  # Start small
    
    # Save model
    trainer.save_model("models/rl_agents/first_ppo_model.zip")
    
    print("Done! Model saved.")

if __name__ == "__main__":
    main()
```

**Step 3**: Run your first training

```bash
# In terminal
python src/rl/training/train_script.py
```

**What you'll see**:
```
Loading scheduling data...
Creating RL environment...
Training agent...
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 42.5     |
|    ep_rew_mean     | 0.35     |
| time/              |          |
|    fps             | 324      |
|    total_timesteps | 2048     |
---------------------------------
Training completed in 154.2s
Model saved to models/rl_agents/first_ppo_model.zip
Done!
```

### 2.2.2: Add TensorBoard Logging

**What**: Visualize training progress in your browser  
**Why**: See if agent is learning or stuck

#### Implementation

**Update trainer.py**:
```python
def train(self, total_timesteps: int = 100000):
    """Train with TensorBoard logging."""
    
    # Create agent with TensorBoard
    self.agent = create_ppo_agent(
        self.env, 
        verbose=1,
        tensorboard_log="logs/tensorboard/",  # Important!
    )
    
    # Train
    self.agent.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        tb_log_name="ppo_schedule_training",  # Run name
    )
    
    return self.agent
```

**View in browser**:
```bash
# In another terminal
tensorboard --logdir logs/tensorboard/

# Open browser to: http://localhost:6006
```

**What you'll see**:
- Reward over time (should increase!)
- Episode length (how long episodes last)
- Policy loss (should decrease)
- Value loss (should decrease)

### 2.2.3: Implement Curriculum Learning

**What**: Train on easy problems first, then harder  
**Why**: Like learning math - start with addition before calculus

#### Concept

```
Easy Problems (10 courses)     → Agent learns basics
    ↓
Medium Problems (20 courses)   → Agent gets better
    ↓
Hard Problems (40 courses)     → Agent masters it
```

#### Implementation

```python
# src/rl/training/curriculum.py

class CurriculumManager:
    """Manages progressive difficulty in training."""
    
    def __init__(self):
        self.stages = [
            {"name": "easy", "num_courses": 10, "timesteps": 20000},
            {"name": "medium", "num_courses": 20, "timesteps": 30000},
            {"name": "hard", "num_courses": 40, "timesteps": 50000},
        ]
        self.current_stage = 0
    
    def get_current_stage(self):
        """Get current difficulty stage."""
        if self.current_stage >= len(self.stages):
            return None
        return self.stages[self.current_stage]
    
    def advance_stage(self):
        """Move to next difficulty."""
        self.current_stage += 1
    
    def create_env_for_stage(self, stage):
        """Create environment with stage difficulty."""
        # Filter courses to match difficulty
        num_courses = stage["num_courses"]
        filtered_context = self.filter_courses(num_courses)
        
        # Create environment
        return create_schedule_env(
            initial_population=self.create_pop(filtered_context),
            context=filtered_context,
        )
```

**Use in training**:
```python
def train_with_curriculum():
    curriculum = CurriculumManager()
    agent = None
    
    for stage_info in curriculum.stages:
        print(f"Training on {stage_info['name']} problems...")
        
        # Create environment for this difficulty
        env = curriculum.create_env_for_stage(stage_info)
        
        # Train (continue from previous stage if exists)
        if agent is None:
            agent = create_ppo_agent(env)
        else:
            agent.set_env(env)  # Reuse agent, new environment
        
        agent.learn(total_timesteps=stage_info["timesteps"])
        
        # Save checkpoint
        agent.save(f"models/stage_{stage_info['name']}.zip")

#### Example Configuration (configs/base.yaml)

```yaml
rl:
    training:
        curriculum:
            - name: "easy"
                enabled: true
                num_episodes: 200
                max_generations: 100
                checkpoint_every: 25
                validation_episodes: 5
                sample_config:
                    num_courses: 10
            - name: "medium"
                enabled: true
                num_episodes: 300
                max_generations: 200
                checkpoint_every: 25
                validation_episodes: 5
                sample_config:
                    num_courses: 20
            - name: "hard"
                enabled: true
                num_episodes: 500
                max_generations: 400
                checkpoint_every: 50
                validation_episodes: 10
                sample_config:
                    num_courses: 40
        checkpoint_settings:
            manifest_path: "models/rl_agents/manifest.json"
            checkpoint_dir: "models/rl_agents/checkpoints/"
            checkpoint_keep_last: 5
            validation_threshold: 0.05  # relative improvement for stage advancement
            adaptive_advance_patience: 3  # number of validated checkpoints above threshold for advancement
```


    ### Why Many Episodes per Stage (Justification)

    One episode is a single, complete GA run (initialize population -> run GA for configured generations -> collect episode reward). Multiple episodes per stage are essential for the following reasons:

    - Statistical Robustness: A neural policy needs many examples to learn reliable action-to-outcome relationships. One episode is an anecdote; hundreds show an actual pattern.
    - Diversity of Problems: Each episode uses a different seed/initial population and problem instance (especially with curriculum sampling), exposing the agent to varied constraints and local optima.
    - Stable Gradient Updates: Batch gradient updates from multiple collected episodes yield smoother and more stable learning than one-shot changes.
    - Avoid Overfitting: Repeating episodes with different instances prevents memorization of a single seed or instance.

    Recommended episode counts (tunable based on compute and dataset):
    - Stage 1 (easy problems): 200 episodes (10 courses)
    - Stage 2 (medium problems): 300 episodes (20 courses)
    - Stage 3 (hard/production): 500 episodes (40+ courses)

    Checkpoint & Validation Process:
    - Save checkpoints every 25–50 episodes depending on speed and memory (`checkpoint_every` configurable).
    - Maintain a validation set (hold-out seed/instances) for each stage; run 5-10 validation episodes per checkpoint and compute aggregated validation metrics.
    - Use a composite validation metric (weighted fitness improvement + hypervolume + diversity) to select the best checkpoint per stage.
    - Promote the best validated model to the next stage or to production if all criteria are met.

    Adaptive Stage Advancement (optional):
    - Skip to the next stage early if validation metric reaches the stage threshold (`9tau_i`) consistently for `k` checkpoints.
    - If validation degrades, either 1) revert to a previous checkpoint or 2) increase training episodes at the current stage.

    ### NSGA-II Diversity Metrics in the State Vector

    For multi-objective optimization using NSGA-II, include explicit diversity metrics in the `StateEncoder` so the agent can make decisions that balance convergence and diversity.

    Suggested metrics (normalized to [0,1]):
    - Pareto front size (|Pareto| / pop_size)
    - Hypervolume or Pareto spread (normalized by known bounds)
    - Mean & std of crowding distances for rank-1 solutions
    - Genotype diversity (average normalised Hamming distance in population)
    - Phenotype diversity (avg pairwise Euclidean distance in the objective space)
    - Unique fitness ratio (#unique fitness vectors / population size)

    How to use them:
    - Low genotype diversity → trigger stronger perturbation (higher mutation) or more exploratory crossovers
    - Low phenotype diversity / shrinking Pareto spread → apply niching crossover or diversity-promoting operators
    - High crowding distance variance → rebalance selection pressure (e.g., lower selection pressure or add tournament diversity)

    Implementation notes:
    - Use subsampling for pairwise diversity computations if population size is large (O(sample_limit) instead of O(N^2)).
    - Maintain running statistics (moving averages) for improvement rates to detect stagnation quickly.
    - Normalization of features is critical for stable learning (min-max or z-score with clamping).

    #### Example diversity helper (pseudocode)

    ```python
    def genotype_diversity(population_genotypes, sample_limit=200):
        # population_genotypes: list of lists/tuples of genes
        # return avg normalized Hamming distance
        from random import randrange
        import numpy as np
        n = len(population_genotypes)
        L = len(population_genotypes[0])
        pairs = []
        if n * (n - 1) // 2 <= sample_limit:
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append((population_genotypes[i], population_genotypes[j]))
        else:
            for _ in range(sample_limit):
                i = randrange(n)
                j = randrange(n)
                if i == j: j = (i+1) % n
                pairs.append((population_genotypes[i], population_genotypes[j]))
        distances = [sum(1 for a,b in zip(x,y) if a != b) / float(L) for x,y in pairs]
        return float(np.mean(distances))

    def phenotype_diversity(objectives, sample_limit=200):
        # objectives: list of tuples like (hard, soft)
        import numpy as np
        n = len(objectives)
        if n < 2:
            return 0.0
        arr = np.array(objectives, dtype=float)
        col_std = np.std(arr, axis=0, ddof=1)
        col_std[col_std == 0] = 1.0
        arr = arr / col_std
        pairs = []
        if n * (n - 1) // 2 <= sample_limit:
            for i in range(n):
                for j in range(i+1, n):
                    pairs.append((arr[i], arr[j]))
        else:
            for _ in range(sample_limit):
                i = randrange(n)
                j = randrange(n)
                if i == j: j = (i+1) % n
                pairs.append((arr[i], arr[j]))
        distances = [np.linalg.norm(a - b) for a,b in pairs]
        max_possible = np.sqrt(arr.shape[1])
        return float(np.mean(distances) / max_possible)
    ```


```

### 2.2.4: Hyperparameter Tuning

**What**: Find best learning settings  
**Why**: Default settings might not be optimal for your problem

#### Key Hyperparameters

**PPO Hyperparameters**:
- $\alpha$ (`learning_rate`): Step size for gradient descent, typically $10^{-4}$ to $10^{-3}$
- $N$ (`n_steps`): Rollout length before update, typically 1024-4096
- $B$ (`batch_size`): Mini-batch size for SGD, typically 32-128
- $\gamma$ (`gamma`): Discount factor for future rewards, typically 0.95-0.99
- $\epsilon$ (`clip_range`): PPO clipping parameter, typically 0.1-0.3
- $\lambda$ (`gae_lambda`): GAE parameter, typically 0.9-0.99
- $K$ (`n_epochs`): Gradient updates per rollout, typically 3-10

**DQN Hyperparameters**:
- $\alpha$ (`learning_rate`): Learning rate, typically $10^{-4}$
- $|\mathcal{D}|$ (`buffer_size`): Replay buffer capacity, typically 10K-1M
- $B$ (`batch_size`): Sample size from buffer, typically 32-128
- $\epsilon_{\text{start}}$ (`exploration_initial_eps`): Initial exploration, typically 1.0
- $\epsilon_{\text{end}}$ (`exploration_final_eps`): Final exploration, typically 0.01
- $f_{\text{target}}$ (`target_update_interval`): Target network update frequency

#### Algorithm: Bayesian Optimization with Optuna

```python
# src/rl/training/tune_hyperparameters.py

import optuna
from optuna.pruners import MedianPruner
from src.rl.training.trainer import RLTrainer

def objective(trial):
    """Objective function for hyperparameter optimization."""
    
    # Sample hyperparameters
    config = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'n_steps': trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'gamma': trial.suggest_uniform('gamma', 0.95, 0.999),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.3),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.9, 0.99),
        'n_epochs': trial.suggest_int('n_epochs', 3, 10),
    }
    
    # Create environment
    env = create_schedule_env(...)
    
    # Train with these hyperparameters
    agent = create_ppo_agent(env, **config)
    
    # Use callback to report intermediate values
    eval_callback = TrialEvalCallback(trial, eval_env, n_eval_episodes=5)
    
    agent.learn(
        total_timesteps=50000,
        callback=eval_callback,
    )
    
    # Return final performance (negative because we minimize)
    return -eval_callback.best_mean_reward

# Run optimization
study = optuna.create_study(
    direction='maximize',
    pruner=MedianPruner(),
    study_name='ppo_schedule_optimization',
)

study.optimize(objective, n_trials=50, timeout=3600)  # 1 hour budget

# Best hyperparameters
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_trial.value}")
print(f"Best params: {study.best_trial.params}")
```

#### Quick Tuning Strategy (Grid Search)

```python
# src/rl/training/grid_search.py

import itertools
import numpy as np

def grid_search_hyperparameters():
    """Simple grid search for quick tuning."""
    
    param_grid = {
        'learning_rate': [0.0001, 0.0003, 0.001],
        'n_steps': [1024, 2048, 4096],
        'batch_size': [32, 64, 128],
    }
    
    results = []
    
    # Generate all combinations
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    
    print(f"Testing {len(combinations)} configurations...")
    
    for combo in combinations:
        config = dict(zip(keys, combo))
        
        print(f"\nTesting: {config}")
        
        # Train
        env = create_schedule_env(...)
        agent = create_ppo_agent(env, **config)
        agent.learn(total_timesteps=50000)
        
        # Evaluate
        mean_reward = evaluate_agent(agent, env, n_episodes=10)
        
        results.append({
            'config': config,
            'mean_reward': mean_reward,
        })
        
        print(f"Mean reward: {mean_reward:.2f}")
    
    # Find best configuration
    best = max(results, key=lambda x: x['mean_reward'])
    print(f"\nBest configuration: {best['config']}")
    print(f"Best mean reward: {best['mean_reward']:.2f}")
    
    return best['config']
```

### 2.2.5: Training Evaluation Callbacks

**What**: Monitor training quality and save best models  
**Why**: Prevent overfitting and track progress

#### Algorithm: Custom Callback System

```python
# src/rl/training/callbacks.py

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import numpy as np

class PeriodicEvaluationCallback(BaseCallback):
    """Evaluates agent periodically during training."""
    
    def __init__(
        self,
        eval_env,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        log_path: str = "logs/eval/",
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.eval_history = []
    
    def _on_step(self) -> bool:
        """Called at each training step."""
        
        if self.n_calls % self.eval_freq == 0:
            # Run evaluation
            episode_rewards = []
            
            for _ in range(self.n_eval_episodes):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _ = self.eval_env.step(action)
                    episode_reward += reward
                
                episode_rewards.append(episode_reward)
            
            # Log results
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/std_reward', std_reward)
            
            self.eval_history.append({
                'timestep': self.n_calls,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
            })
            
            print(f"Eval @ {self.n_calls}: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return True

class EarlyStoppingCallback(BaseCallback):
    """Stops training if no improvement for N evaluations."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.01,
    ):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.wait = 0
    
    def _on_step(self) -> bool:
        """Check if training should stop."""
        
        # Get current performance from logger
        if 'eval/mean_reward' in self.logger.name_to_value:
            current_reward = self.logger.name_to_value['eval/mean_reward']
            
            # Check for improvement
            if current_reward > self.best_mean_reward + self.min_delta:
                self.best_mean_reward = current_reward
                self.wait = 0
            else:
                self.wait += 1
                
                if self.wait >= self.patience:
                    print(f"Early stopping: no improvement for {self.patience} evaluations")
                    return False  # Stop training
        
        return True  # Continue training

class CheckpointCallback(BaseCallback):
    """Saves best model during training."""
    
    def __init__(
        self,
        save_freq: int = 10000,
        save_path: str = "models/checkpoints/",
        name_prefix: str = "rl_model",
    ):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.name_prefix = name_prefix
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        """Save checkpoint if best model."""
        
        if self.n_calls % self.save_freq == 0:
            # Save periodic checkpoint
            checkpoint_path = self.save_path / f"{self.name_prefix}_step_{self.n_calls}"
            self.model.save(checkpoint_path)
            
            # Check if best model
            if 'eval/mean_reward' in self.logger.name_to_value:
                current_reward = self.logger.name_to_value['eval/mean_reward']
                
                if current_reward > self.best_mean_reward:
                    self.best_mean_reward = current_reward
                    best_path = self.save_path / f"{self.name_prefix}_best"
                    self.model.save(best_path)
                    print(f"New best model saved: {current_reward:.2f}")
        
        return True
```

**Usage in training**:

```python
# Combine multiple callbacks
from stable_baselines3.common.callbacks import CallbackList

eval_callback = PeriodicEvaluationCallback(eval_env, eval_freq=5000)
early_stop_callback = EarlyStoppingCallback(patience=5)
checkpoint_callback = CheckpointCallback(save_freq=10000)

callback_list = CallbackList([
    eval_callback,
    early_stop_callback,
    checkpoint_callback,
])

# Train with callbacks
agent.learn(
    total_timesteps=100000,
    callback=callback_list,
)
```

---

## Phase 2.3: Deployment & Integration

### What is Deployment?

**Deployment** = Using your trained RL agent in the actual GA scheduler (production).

### 2.3.1: Implement Model Loader

**What**: Load trained model quickly  
**Why**: Production needs fast startup (<100ms)

```python
# src/rl/deployment/model_loader.py

from stable_baselines3 import PPO, DQN
from pathlib import Path
import time

class ModelLoader:
    """Loads trained RL models for production use."""
    
    def __init__(self, model_dir: str = "models/rl_agents"):
        self.model_dir = Path(model_dir)
        self.cache = {}  # Cache loaded models
    
    def load_model(self, model_name: str, agent_type: str = "ppo"):
        """Load model from disk."""
        
        # Check cache first
        if model_name in self.cache:
            return self.cache[model_name]
        
        # Load from disk
        model_path = self.model_dir / f"{model_name}.zip"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        start = time.time()
        
        if agent_type == "ppo":
            model = PPO.load(model_path)
        elif agent_type == "dqn":
            model = DQN.load(model_path)
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        elapsed = (time.time() - start) * 1000
        print(f"Model loaded in {elapsed:.1f}ms")
        
        # Cache it
        self.cache[model_name] = model
        
        return model
    
    def list_available_models(self):
        """List all trained models."""
        return [p.stem for p in self.model_dir.glob("*.zip")]

### 2.3.4: Model Promotion & Registry

**What**: Promote validated models to production and maintain a small model registry with metadata.
**Why**: Production should use only validated models; registry ensures reproducibility and safe promotion.

Manifest format (`models/rl_agents/manifest.json`):
```
[
    {
        "name": "ppo_scheduler_final",
        "path": "models/rl_agents/ppo_scheduler_final.zip",
        "created_at": "2025-11-10T12:00:00Z",
        "seed": 12345,
        "config_path": "configs/rl_training/hyperparams_ppo.yaml",
        "validation_score": -2.3,
        "hypervolume": 0.67,
        "status": "validated"  # validated|staging|prod
    }
]
```

Promotion script responsibilities:
- Validate the checkpoint selected by the validation stage, copy to `models/rl_agents/` as `*_final.zip`, and add a manifest entry.
- Update `configs/prod.yaml` with the chosen `rl.agent.model_path` and `rl.agent.version` atomically. Prefer writing to a tmp file then renaming.
- Add a rollback mechanism to revert `configs/prod.yaml` to a previous version.

Registry helpers (`src/rl/deployment/registry.py`):
- `list_models()`
- `get_model_metadata(name)`
- `promote_model(name)`
- `demote_model(name)`

Add automated tests for registry functionality and promotion script.

```

### 2.3.2: Implement Inference Engine

**What**: Fast prediction (<10ms per action)  
**Why**: GA runs in real-time, can't wait

```python
# src/rl/deployment/inference.py

import numpy as np
import time

class RLInference:
    """Fast inference engine for production."""
    
    def __init__(self, model, timeout_ms: float = 10.0):
        self.model = model
        self.timeout_ms = timeout_ms
        self.prediction_times = []
    
    def predict_action(self, state: np.ndarray, deterministic: bool = True):
        """Predict action with timeout protection."""
        
        start = time.time()
        
        try:
            # Predict action
            action, _states = self.model.predict(state, deterministic=deterministic)
            
            # Check timing
            elapsed_ms = (time.time() - start) * 1000
            self.prediction_times.append(elapsed_ms)
            
            if elapsed_ms > self.timeout_ms:
                print(f"Warning: Prediction took {elapsed_ms:.2f}ms (> {self.timeout_ms}ms)")
            
            return int(action)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None
    
    def get_average_inference_time(self):
        """Get average prediction time."""
        if not self.prediction_times:
            return 0.0
        return np.mean(self.prediction_times)
```

### 2.3.3: Implement Hybrid Controller

**What**: Combines RL with fallback strategies  
**Why**: RL might fail, need backup plan

#### Three Modes

**Mode 1: RL-Primary** (Trust RL completely)
```python
action = rl_agent.predict(state)
if action is valid:
    apply_heuristic(action)
else:
    apply_heuristic(fallback_random())
```

**Mode 2: RL-Fallback** (Try RL, fallback if fails)
```python
try:
    action = rl_agent.predict(state, timeout=10ms)
    apply_heuristic(action)
except TimeoutError:
    action = fallback_greedy()
    apply_heuristic(action)
```

**Mode 3: RL-Assisted** (Mix RL with other strategies)
```python
if random() < 0.8:  # 80% RL
    action = rl_agent.predict(state)
else:  # 20% exploration
    action = random_heuristic()
apply_heuristic(action)
```

#### Implementation

```python
# src/rl/hybrid/hybrid_controller.py

from enum import Enum
import random
from typing import Optional
import numpy as np

class HybridMode(Enum):
    RL_PRIMARY = "rl_primary"
    RL_FALLBACK = "rl_fallback"
    RL_ASSISTED = "rl_assisted"

class HybridController:
    """Manages hybrid RL + heuristic selection."""
    
    def __init__(
        self,
        rl_inference,
        mode: HybridMode = HybridMode.RL_PRIMARY,
        fallback_strategy: str = "random",
        rl_probability: float = 0.8,
    ):
        self.rl_inference = rl_inference
        self.mode = mode
        self.fallback_strategy = fallback_strategy
        self.rl_probability = rl_probability
        
        # Statistics
        self.rl_calls = 0
        self.fallback_calls = 0
    
    def select_action(self, state: np.ndarray, valid_actions: list) -> int:
        """Select action using hybrid strategy."""
        
        if self.mode == HybridMode.RL_PRIMARY:
            return self._rl_primary(state, valid_actions)
        elif self.mode == HybridMode.RL_FALLBACK:
            return self._rl_fallback(state, valid_actions)
        else:  # RL_ASSISTED
            return self._rl_assisted(state, valid_actions)
    
    def _rl_primary(self, state, valid_actions):
        """Always use RL, fallback only on failure."""
        action = self.rl_inference.predict_action(state)
        
        if action is not None and action in valid_actions:
            self.rl_calls += 1
            return action
        else:
            self.fallback_calls += 1
            return self._fallback_action(valid_actions)
    
    def _rl_fallback(self, state, valid_actions):
        """Try RL with timeout, fallback on timeout/error."""
        try:
            action = self.rl_inference.predict_action(state)
            if action in valid_actions:
                self.rl_calls += 1
                return action
        except Exception:
            pass
        
        self.fallback_calls += 1
        return self._fallback_action(valid_actions)
    
    def _rl_assisted(self, state, valid_actions):
        """Mix RL with exploration."""
        if random.random() < self.rl_probability:
            action = self.rl_inference.predict_action(state)
            if action in valid_actions:
                self.rl_calls += 1
                return action
        
        self.fallback_calls += 1
        return self._fallback_action(valid_actions)
    
    def _fallback_action(self, valid_actions):
        """Fallback strategy when RL unavailable."""
        if self.fallback_strategy == "random":
            return random.choice(valid_actions)
        elif self.fallback_strategy == "greedy":
            return valid_actions[0]  # Assume sorted by priority
        else:
            return random.choice(valid_actions)
    
    def get_statistics(self):
        """Get usage statistics."""
        total = self.rl_calls + self.fallback_calls
        rl_pct = 100 * self.rl_calls / total if total > 0 else 0
        return {
            "rl_calls": self.rl_calls,
            "fallback_calls": self.fallback_calls,
            "rl_percentage": rl_pct,
        }
```

### 2.3.4: Integrate with GA Scheduler

**What**: Connect RL agent to your existing GA  
**Where**: Modify `src/core/ga_scheduler.py`

#### Current GA Code (simplified)

```python
# Your existing code probably looks like:
class GAScheduler:
    def run(self):
        population = self.create_initial_population()
        
        for generation in range(self.ngen):
            # Evaluate fitness
            for ind in population:
                ind.fitness.values = self.evaluate(ind)
            
            # Apply operators
            for ind in population:
                if random() < mutpb:
                    self.mutate(ind)  # Picks random heuristic
```

#### Add RL Integration

```python
# Modified with RL:
class GAScheduler:
    def __init__(self, context, use_rl: bool = False):
        self.context = context
        self.use_rl = use_rl
        
        # Initialize RL if enabled
        if self.use_rl:
            self._init_rl()
    
    def _init_rl(self):
        """Initialize RL components."""
        from src.rl.deployment import ModelLoader, RLInference
        from src.rl.hybrid import HybridController
        from src.config import get_config
        
        config = get_config()
        
        # Load trained model
        loader = ModelLoader()
        model = loader.load_model(
            model_name=config.rl.agent.model_path.replace(".zip", ""),
            agent_type=config.rl.agent.type,
        )
        
        # Create inference engine
        inference = RLInference(
            model=model,
            timeout_ms=config.rl.inference.timeout_ms,
        )
        
        # Create hybrid controller
        self.rl_controller = HybridController(
            rl_inference=inference,
            mode=config.rl.hybrid.mode,
            fallback_strategy=config.rl.hybrid.fallback_strategy,
        )
        
        # Create state encoder
        from src.rl.gym_env import StateEncoder
        self.state_encoder = StateEncoder()
    
    def run(self):
        """Run GA with optional RL integration."""
        population = self.create_initial_population()
        
        for generation in range(self.ngen):
            # Evaluate fitness
            for ind in population:
                ind.fitness.values = self.evaluate(ind)
            
            # Apply operators
            if self.use_rl:
                self._apply_rl_operators(population, generation)
            else:
                self._apply_standard_operators(population)
    
    def _apply_rl_operators(self, population, generation):
        """Apply operators selected by RL agent."""
        
        # Encode current state
        state = self.state_encoder.encode(
            population=population,
            current_generation=generation,
            generations_without_improvement=self.stagnation_counter,
        )
        
        # Get action from RL agent
        from src.rl.gym_env import ActionMapper
        action_mapper = ActionMapper()
        valid_actions = action_mapper.enabled_actions
        
        action = self.rl_controller.select_action(state, valid_actions)
        
        # Apply selected heuristic
        for ind in population:
            modified_ind, success = action_mapper.apply_action(
                action, ind, self.context
            )
            if success:
                # Replace individual with modified version
                ind[:] = modified_ind
    
    def _apply_standard_operators(self, population):
        """Standard GA operators (existing code)."""
        for ind in population:
            if random() < self.mutpb:
                self.mutate(ind)
```

#### Enable RL in Config

```yaml
# configs/prod.yaml (add this)
rl:
  enabled: true
  mode: inference  # Use trained model
  agent:
    type: ppo
    model_path: models/rl_agents/best_ppo_model.zip
  hybrid:
    mode: rl_fallback
    fallback_strategy: random
```

#### Run with RL

```bash
# Train first (Phase 2.2)
python src/rl/training/train_script.py

# Then run GA with RL enabled
uv run prod  # Will use RL if enabled in config
```

---

## Phase 2.4: Evaluation & Comparison

### What is Evaluation?

**Evaluation** = Prove that RL is better than alternatives.

### 2.4.1: Implement Baseline Strategies

**What**: Other strategies to compare against  
**Why**: Need to show RL > Random, RL > Fixed Order, etc.

```python
# src/rl/evaluation/baselines.py

class BaselineStrategies:
    """Collection of baseline heuristic selection strategies."""
    
    @staticmethod
    def random_selection(valid_actions):
        """Uniformly random selection."""
        return random.choice(valid_actions)
    
    @staticmethod
    def round_robin(valid_actions, step_counter):
        """Cycle through actions in order."""
        return valid_actions[step_counter % len(valid_actions)]
    
    @staticmethod
    def fixed_priority(valid_actions):
        """Always pick highest priority."""
        # Assume valid_actions sorted by priority
        return valid_actions[0]
    
    @staticmethod
    def greedy(valid_actions, recent_rewards):
        """Pick action with best recent performance."""
        if not recent_rewards:
            return random.choice(valid_actions)
        
        # Pick action with highest average reward
        best_action = max(recent_rewards, key=lambda a: np.mean(recent_rewards[a]))
        return best_action if best_action in valid_actions else valid_actions[0]
```

### 2.4.2: Implement Evaluator

**What**: Run multiple strategies and collect results  
**Why**: Compare performance scientifically

```python
# src/rl/evaluation/evaluator.py

import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class EvaluationResult:
    strategy_name: str
    final_fitness: float
    convergence_generation: int
    total_time: float
    heuristic_usage: Dict[int, int]

class RLEvaluator:
    """Evaluates and compares RL agent with baselines."""
    
    def __init__(self, context, num_runs: int = 10):
        self.context = context
        self.num_runs = num_runs
        self.results = {}
    
    def evaluate_strategy(self, strategy_name: str, strategy_fn):
        """Evaluate a single strategy over multiple runs."""
        print(f"Evaluating {strategy_name}...")
        
        results = []
        
        for run in range(self.num_runs):
            print(f"  Run {run+1}/{self.num_runs}")
            
            # Create fresh GA scheduler
            scheduler = GAScheduler(self.context)
            
            # Run with this strategy
            start = time.time()
            best_solution = scheduler.run_with_strategy(strategy_fn)
            elapsed = time.time() - start
            
            # Collect results
            result = EvaluationResult(
                strategy_name=strategy_name,
                final_fitness=best_solution.fitness.values[0],
                convergence_generation=scheduler.convergence_gen,
                total_time=elapsed,
                heuristic_usage=scheduler.heuristic_counts,
            )
            results.append(result)
        
        self.results[strategy_name] = results
        return results
    
    def compare_strategies(self):
        """Compare all evaluated strategies."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        for strategy_name, results in self.results.items():
            fitnesses = [r.final_fitness for r in results]
            times = [r.total_time for r in results]
            
            print(f"\n{strategy_name}:")
            print(f"  Avg Fitness: {np.mean(fitnesses):.2f} ± {np.std(fitnesses):.2f}")
            print(f"  Best Fitness: {np.min(fitnesses):.2f}")
            print(f"  Worst Fitness: {np.max(fitnesses):.2f}")
            print(f"  Avg Time: {np.mean(times):.1f}s ± {np.std(times):.1f}s")
    
    def run_full_evaluation(self):
        """Run comprehensive evaluation."""
        
        # 1. RL Agent (trained)
        self.evaluate_strategy("RL-PPO", self._rl_strategy)
        
        # 2. Random baseline
        self.evaluate_strategy("Random", BaselineStrategies.random_selection)
        
        # 3. Round-robin
        self.evaluate_strategy("Round-Robin", BaselineStrategies.round_robin)
        
        # 4. Fixed priority
        self.evaluate_strategy("Fixed-Priority", BaselineStrategies.fixed_priority)
        
        # 5. Greedy
        self.evaluate_strategy("Greedy", BaselineStrategies.greedy)
        
        # Compare
        self.compare_strategies()
        
        # Statistical tests
        self.statistical_analysis()
    
    def _rl_strategy(self, state, valid_actions):
        """RL agent strategy."""
        if not hasattr(self, 'rl_controller'):
            # Initialize RL once
            from src.rl.hybrid import HybridController
            # ... setup code ...
            pass
        
        return self.rl_controller.select_action(state, valid_actions)
    
    def statistical_analysis(self):
        """Perform statistical significance tests."""
        from scipy import stats
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        
        rl_results = self.results.get("RL-PPO", [])
        random_results = self.results.get("Random", [])
        
        if rl_results and random_results:
            rl_fitness = [r.final_fitness for r in rl_results]
            random_fitness = [r.final_fitness for r in random_results]
            
            # T-test
            t_stat, p_value = stats.ttest_ind(rl_fitness, random_fitness)
            
            print(f"\nRL vs Random:")
            print(f"  T-statistic: {t_stat:.3f}")
            print(f"  P-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  ✓ Statistically significant (p < 0.05)")
                
                improvement = (np.mean(random_fitness) - np.mean(rl_fitness)) / np.mean(random_fitness) * 100
                print(f"  ✓ RL improves by {improvement:.1f}%")
            else:
                print(f"  ✗ Not statistically significant")
```

### 2.4.3: Metrics Collection

**What**: Track comprehensive performance metrics  
**Why**: Understand agent behavior beyond final fitness

```python
# src/rl/evaluation/metrics.py

from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

@dataclass
class PerformanceMetrics:
    """Comprehensive metrics for strategy evaluation."""
    
    # Fitness metrics
    final_fitness: float
    best_fitness: float
    fitness_improvement: float
    fitness_trajectory: List[float] = field(default_factory=list)
    
    # Convergence metrics
    convergence_generation: int
    convergence_speed: float  # Generations per improvement
    stagnation_episodes: int
    
    # Efficiency metrics
    total_time: float
    time_per_generation: float
    evaluations_per_second: float
    
    # Diversity metrics
    final_diversity: float
    diversity_trajectory: List[float] = field(default_factory=list)
    diversity_maintenance: float  # % of initial diversity retained
    
    # Heuristic usage
    heuristic_usage: Dict[int, int] = field(default_factory=dict)
    heuristic_success_rate: Dict[int, float] = field(default_factory=dict)
    heuristic_avg_reward: Dict[int, float] = field(default_factory=dict)
    
    # Robustness metrics
    fitness_variance: float = 0.0
    time_variance: float = 0.0
    reliability: float = 1.0  # % runs without failures

class MetricsCollector:
    """Collects and aggregates metrics across multiple runs."""
    
    def __init__(self):
        self.runs = []
    
    def collect_run(self, metrics: PerformanceMetrics):
        """Add metrics from one run."""
        self.runs.append(metrics)
    
    def compute_statistics(self):
        """Compute aggregate statistics."""
        
        if not self.runs:
            return {}
        
        stats = {}
        
        # Fitness statistics
        fitness_values = [r.final_fitness for r in self.runs]
        stats['fitness'] = {
            'mean': np.mean(fitness_values),
            'std': np.std(fitness_values),
            'min': np.min(fitness_values),
            'max': np.max(fitness_values),
            'median': np.median(fitness_values),
            'iqr': np.percentile(fitness_values, 75) - np.percentile(fitness_values, 25),
        }
        
        # Convergence statistics
        conv_gens = [r.convergence_generation for r in self.runs]
        stats['convergence'] = {
            'mean_generation': np.mean(conv_gens),
            'std_generation': np.std(conv_gens),
            'min_generation': np.min(conv_gens),
        }
        
        # Time statistics
        times = [r.total_time for r in self.runs]
        stats['time'] = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
        }
        
        # Diversity statistics
        diversity = [r.final_diversity for r in self.runs]
        stats['diversity'] = {
            'mean': np.mean(diversity),
            'std': np.std(diversity),
        }
        
        # Heuristic usage aggregation
        all_heuristics = set()
        for run in self.runs:
            all_heuristics.update(run.heuristic_usage.keys())
        
        stats['heuristics'] = {}
        for h_id in all_heuristics:
            usage_counts = [r.heuristic_usage.get(h_id, 0) for r in self.runs]
            stats['heuristics'][h_id] = {
                'total_usage': sum(usage_counts),
                'avg_usage': np.mean(usage_counts),
                'usage_frequency': sum(1 for c in usage_counts if c > 0) / len(self.runs),
            }
        
        return stats
    
    def export_to_json(self, filepath: str):
        """Export metrics to JSON for analysis."""
        import json
        
        data = {
            'runs': [asdict(run) for run in self.runs],
            'statistics': self.compute_statistics(),
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
```

### 2.4.4: Statistical Analysis

**What**: Rigorous statistical testing of results  
**Why**: Prove RL superiority with confidence

#### Statistical Tests

```python
# src/rl/evaluation/statistical_tests.py

from scipy import stats
import numpy as np
from typing import List, Tuple

class StatisticalAnalyzer:
    """Performs statistical significance testing."""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha  # Significance level
    
    def independent_t_test(
        self,
        sample1: List[float],
        sample2: List[float],
    ) -> Tuple[float, float, bool]:
        """
        Two-sample t-test for comparing means.
        
        H0: μ1 = μ2 (no difference)
        H1: μ1 ≠ μ2 (significant difference)
        """
        
        t_stat, p_value = stats.ttest_ind(sample1, sample2)
        significant = p_value < self.alpha
        
        return t_stat, p_value, significant
    
    def paired_t_test(
        self,
        sample1: List[float],
        sample2: List[float],
    ) -> Tuple[float, float, bool]:
        """
        Paired t-test for matched samples.
        
        Use when same problems tested with different strategies.
        """
        
        t_stat, p_value = stats.ttest_rel(sample1, sample2)
        significant = p_value < self.alpha
        
        return t_stat, p_value, significant
    
    def mann_whitney_u_test(
        self,
        sample1: List[float],
        sample2: List[float],
    ) -> Tuple[float, float, bool]:
        """
        Non-parametric test (doesn't assume normal distribution).
        
        More robust when data is not normally distributed.
        """
        
        u_stat, p_value = stats.mannwhitneyu(
            sample1, sample2, alternative='two-sided'
        )
        significant = p_value < self.alpha
        
        return u_stat, p_value, significant
    
    def effect_size_cohens_d(
        self,
        sample1: List[float],
        sample2: List[float],
    ) -> float:
        """
        Cohen's d effect size.
        
        Interpretation:
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large
        """
        
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
        n1, n2 = len(sample1), len(sample2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        
        d = (mean1 - mean2) / pooled_std
        
        return d
    
    def confidence_interval(
        self,
        sample: List[float],
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for mean.
        
        Returns: (lower_bound, upper_bound)
        """
        
        mean = np.mean(sample)
        sem = stats.sem(sample)  # Standard error of mean
        ci = stats.t.interval(
            confidence,
            len(sample)-1,
            loc=mean,
            scale=sem,
        )
        
        return ci
    
    def anova_test(
        self,
        *samples: List[float],
    ) -> Tuple[float, float, bool]:
        """
        One-way ANOVA for comparing multiple strategies.
        
        H0: All strategies have same mean
        H1: At least one strategy differs
        """
        
        f_stat, p_value = stats.f_oneway(*samples)
        significant = p_value < self.alpha
        
        return f_stat, p_value, significant
    
    def posthoc_tukey(
        self,
        groups: Dict[str, List[float]],
    ):
        """
        Tukey's HSD post-hoc test for pairwise comparisons.
        
        Use after ANOVA to find which pairs differ.
        """
        
        from statsmodels.stats.multicomp import pairwise_tukeyhsd
        
        # Prepare data
        all_data = []
        all_labels = []
        for label, values in groups.items():
            all_data.extend(values)
            all_labels.extend([label] * len(values))
        
        # Run Tukey HSD
        tukey = pairwise_tukeyhsd(all_data, all_labels, alpha=self.alpha)
        
        return tukey

def generate_comparison_report(
    rl_results: List[float],
    baseline_results: Dict[str, List[float]],
    output_path: str = "rl_comparison_report.txt",
):
    """Generate comprehensive comparison report."""
    
    analyzer = StatisticalAnalyzer()
    
    with open(output_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("RL AGENT STATISTICAL COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        
        # RL statistics
        f.write("RL AGENT PERFORMANCE\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean: {np.mean(rl_results):.4f}\n")
        f.write(f"Std:  {np.std(rl_results):.4f}\n")
        f.write(f"Min:  {np.min(rl_results):.4f}\n")
        f.write(f"Max:  {np.max(rl_results):.4f}\n")
        ci = analyzer.confidence_interval(rl_results)
        f.write(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]\n\n")
        
        # Compare with each baseline
        for baseline_name, baseline_data in baseline_results.items():
            f.write(f"\nRL vs {baseline_name}\n")
            f.write("-"*60 + "\n")
            
            # Baseline statistics
            f.write(f"{baseline_name} Mean: {np.mean(baseline_data):.4f}\n")
            f.write(f"{baseline_name} Std:  {np.std(baseline_data):.4f}\n\n")
            
            # T-test
            t_stat, p_value, sig = analyzer.independent_t_test(
                rl_results, baseline_data
            )
            f.write(f"Independent t-test:\n")
            f.write(f"  t-statistic: {t_stat:.4f}\n")
            f.write(f"  p-value:     {p_value:.6f}\n")
            f.write(f"  Significant: {'YES' if sig else 'NO'} (α=0.05)\n\n")
            
            # Mann-Whitney U test
            u_stat, p_value_u, sig_u = analyzer.mann_whitney_u_test(
                rl_results, baseline_data
            )
            f.write(f"Mann-Whitney U test:\n")
            f.write(f"  U-statistic: {u_stat:.4f}\n")
            f.write(f"  p-value:     {p_value_u:.6f}\n")
            f.write(f"  Significant: {'YES' if sig_u else 'NO'}\n\n")
            
            # Effect size
            d = analyzer.effect_size_cohens_d(rl_results, baseline_data)
            f.write(f"Effect size (Cohen's d): {d:.4f}\n")
            if abs(d) >= 0.8:
                f.write(f"  Interpretation: LARGE effect\n")
            elif abs(d) >= 0.5:
                f.write(f"  Interpretation: MEDIUM effect\n")
            elif abs(d) >= 0.2:
                f.write(f"  Interpretation: SMALL effect\n")
            else:
                f.write(f"  Interpretation: NEGLIGIBLE effect\n")
            
            # Performance improvement
            improvement = (
                (np.mean(baseline_data) - np.mean(rl_results))
                / np.mean(baseline_data) * 100
            )
            f.write(f"\nPerformance improvement: {improvement:+.2f}%\n")
        
        # ANOVA across all strategies
        f.write("\n" + "="*60 + "\n")
        f.write("ANOVA: COMPARING ALL STRATEGIES\n")
        f.write("="*60 + "\n")
        
        all_samples = [rl_results] + list(baseline_results.values())
        f_stat, p_value_anova, sig_anova = analyzer.anova_test(*all_samples)
        
        f.write(f"F-statistic: {f_stat:.4f}\n")
        f.write(f"p-value:     {p_value_anova:.6f}\n")
        f.write(f"Significant: {'YES' if sig_anova else 'NO'}\n\n")
        
        if sig_anova:
            f.write("At least one strategy significantly differs from others.\n")
    
    print(f"Comparison report saved to {output_path}")
```

### 2.4.5: Visualization

**What**: Create plots to visualize results  
**Why**: Pictures are worth 1000 numbers

```python
# src/rl/visualization/performance_plots.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class PerformanceVisualizer:
    """Create performance comparison plots."""
    
    def __init__(self, results_dict):
        self.results = results_dict
    
    def plot_fitness_comparison(self, save_path: str = "fitness_comparison.png"):
        """Box plot comparing final fitness across strategies."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        strategies = []
        fitness_data = []
        
        for strategy_name, results in self.results.items():
            strategies.append(strategy_name)
            fitness_data.append([r.final_fitness for r in results])
        
        # Create box plot
        ax.boxplot(fitness_data, labels=strategies)
        ax.set_ylabel("Final Fitness (lower is better)")
        ax.set_title("Strategy Comparison: Final Fitness")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    
    def plot_convergence_speed(self, save_path: str = "convergence_speed.png"):
        """Bar chart comparing convergence generations."""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        strategies = []
        avg_gens = []
        std_gens = []
        
        for strategy_name, results in self.results.items():
            strategies.append(strategy_name)
            gens = [r.convergence_generation for r in results]
            avg_gens.append(np.mean(gens))
            std_gens.append(np.std(gens))
        
        x = np.arange(len(strategies))
        ax.bar(x, avg_gens, yerr=std_gens, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45)
        ax.set_ylabel("Convergence Generation")
        ax.set_title("Strategy Comparison: Convergence Speed")
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    
    def plot_heuristic_usage(self, strategy_name: str, save_path: str = "heuristic_usage.png"):
        """Histogram of heuristic usage for a strategy."""
        
        results = self.results.get(strategy_name, [])
        if not results:
            return
        
        # Aggregate heuristic usage
        total_usage = {}
        for result in results:
            for heuristic_id, count in result.heuristic_usage.items():
                total_usage[heuristic_id] = total_usage.get(heuristic_id, 0) + count
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        heuristics = sorted(total_usage.keys())
        counts = [total_usage[h] for h in heuristics]
        
        ax.bar(heuristics, counts)
        ax.set_xlabel("Heuristic ID")
        ax.set_ylabel("Usage Count")
        ax.set_title(f"Heuristic Usage - {strategy_name}")
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
```

---

## Phase 3: Advanced RL Features

### 3.1: Multi-Agent RL

**What**: Multiple specialized agents working together  
**Why**: Different heuristic categories need different strategies

#### Architecture: Hierarchical Multi-Agent System

```
┌─────────────────────────────────────────────────┐
│           Meta-Controller Agent                 │
│    (Decides which specialist to use)            │
└─────────────────────────────────────────────────┘
         │           │           │
         ▼           ▼           ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│Construction │ │Perturbation │ │Improvement  │
│   Agent     │ │   Agent     │ │   Agent     │
│             │ │             │ │             │
│ Heuristics: │ │ Heuristics: │ │ Heuristics: │
│  0-6        │ │  7-12       │ │  13-19      │
└─────────────┘ └─────────────┘ └─────────────┘
```

#### Mathematical Formulation

**Hierarchical MDP**:
- High-level state: $s_t^{\text{high}} = [\text{progress}, \text{phase}, \text{specialist\_success}]$
- Low-level state: $s_t^{\text{low}} = [25\text{-dim state vector}]$
- High-level action: $a_t^{\text{high}} \in \{\text{construction}, \text{perturbation}, \text{improvement}\}$
- Low-level action: $a_t^{\text{low}} \in \{\text{heuristics within category}\}$

**Option Framework**:

$$\text{Option } \omega = \langle I_\omega, \pi_\omega, \beta_\omega \rangle$$

Where:
- $I_\omega \subseteq \mathcal{S}$: Initiation set (when option can start)
- $\pi_\omega: \mathcal{S} \times \mathcal{A} \rightarrow [0,1]$: Policy over low-level actions
- $\beta_\omega: \mathcal{S}^+ \rightarrow [0,1]$: Termination condition

#### Implementation

```python
# src/rl/multi_agent/specialist_agents.py

from enum import Enum
from typing import List, Dict
import numpy as np

class HeuristicCategory(Enum):
    CONSTRUCTION = "construction"  # Build solutions from scratch
    PERTURBATION = "perturbation"  # Shake up solutions
    IMPROVEMENT = "improvement"    # Local optimization

class SpecialistAgent:
    """Specialist agent for one heuristic category."""
    
    def __init__(
        self,
        category: HeuristicCategory,
        heuristic_ids: List[int],
        model_path: str,
    ):
        self.category = category
        self.heuristic_ids = heuristic_ids
        self.model = self._load_model(model_path)
        self.stats = {
            'calls': 0,
            'successes': 0,
            'total_reward': 0.0,
        }
    
    def predict(self, state: np.ndarray) -> int:
        """Predict heuristic within category."""
        # Get low-level action (relative to category)
        local_action, _ = self.model.predict(state)
        
        # Map to global heuristic ID
        global_action = self.heuristic_ids[local_action]
        
        self.stats['calls'] += 1
        
        return global_action
    
    def update_stats(self, success: bool, reward: float):
        """Update performance statistics."""
        if success:
            self.stats['successes'] += 1
        self.stats['total_reward'] += reward
    
    def get_success_rate(self) -> float:
        """Get success rate of this specialist."""
        if self.stats['calls'] == 0:
            return 0.0
        return self.stats['successes'] / self.stats['calls']

class MetaController:
    """High-level controller selecting specialists."""
    
    def __init__(self):
        # Initialize specialists
        self.specialists = {
            HeuristicCategory.CONSTRUCTION: SpecialistAgent(
                category=HeuristicCategory.CONSTRUCTION,
                heuristic_ids=[0, 1, 2, 3, 4, 5, 6],
                model_path="models/specialist_construction.zip",
            ),
            HeuristicCategory.PERTURBATION: SpecialistAgent(
                category=HeuristicCategory.PERTURBATION,
                heuristic_ids=[7, 8, 9, 10, 11, 12],
                model_path="models/specialist_perturbation.zip",
            ),
            HeuristicCategory.IMPROVEMENT: SpecialistAgent(
                category=HeuristicCategory.IMPROVEMENT,
                heuristic_ids=[13, 14, 15, 16, 17, 18, 19],
                model_path="models/specialist_improvement.zip",
            ),
        }
        
        # Load meta-policy
        self.meta_policy = self._load_meta_policy()
    
    def select_action(self, state: np.ndarray, phase: str) -> int:
        """Two-level decision making."""
        
        # High-level: Select specialist
        meta_state = self._encode_meta_state(state, phase)
        specialist_idx, _ = self.meta_policy.predict(meta_state)
        
        category = list(HeuristicCategory)[specialist_idx]
        specialist = self.specialists[category]
        
        # Low-level: Specialist selects heuristic
        action = specialist.predict(state)
        
        return action
    
    def _encode_meta_state(self, state: np.ndarray, phase: str) -> np.ndarray:
        """Encode high-level state for meta-controller."""
        
        # Extract key features
        progress = state[15]  # Generation progress
        stagnation = state[16]  # Stagnation counter
        
        # Specialist performance
        specialist_success_rates = [
            s.get_success_rate() for s in self.specialists.values()
        ]
        
        # Phase encoding (one-hot)
        phase_encoding = {
            'exploration': [1, 0, 0],
            'exploitation': [0, 1, 0],
            'intensification': [0, 0, 1],
        }.get(phase, [0, 0, 0])
        
        meta_state = np.array([
            progress,
            stagnation,
            *specialist_success_rates,
            *phase_encoding,
        ])
        
        return meta_state

# Training specialist agents separately
def train_specialist_agents():
    """Train each specialist on filtered heuristics."""
    
    categories = {
        'construction': [0, 1, 2, 3, 4, 5, 6],
        'perturbation': [7, 8, 9, 10, 11, 12],
        'improvement': [13, 14, 15, 16, 17, 18, 19],
    }
    
    for category_name, heuristic_ids in categories.items():
        print(f"Training {category_name} specialist...")
        
        # Create environment with filtered actions
        env = ScheduleEnv(
            ...,
            allowed_actions=heuristic_ids,
        )
        
        # Train specialist
        agent = PPO("MlpPolicy", env)
        agent.learn(total_timesteps=100000)
        
        # Save
        agent.save(f"models/specialist_{category_name}.zip")
        
        print(f"{category_name} specialist trained!")
```

### 3.2: Transfer Learning

**What**: Pre-train on synthetic problems, fine-tune on real problems  
**Why**: Faster convergence and better generalization

#### Transfer Learning Pipeline

```
Stage 1: Pre-training          Stage 2: Fine-tuning
─────────────────────          ────────────────────
Synthetic problems (1000s)  →  Real problems (100s)
Simple structure            →  Complex constraints
Fast generation             →  Real-world data
```

#### Algorithm: Progressive Neural Network

```python
# src/rl/transfer/transfer_learning.py

from stable_baselines3 import PPO
from pathlib import Path

class TransferLearningPipeline:
    """Implements transfer learning for scheduling."""
    
    def __init__(self):
        self.pretrained_model = None
        self.finetuned_model = None
    
    def stage1_pretrain(
        self,
        synthetic_env,
        timesteps: int = 500000,
        save_path: str = "models/pretrained_base.zip",
    ):
        """Pre-train on synthetic problems."""
        
        print("Stage 1: Pre-training on synthetic problems...")
        
        # Create agent
        self.pretrained_model = PPO(
            "MlpPolicy",
            synthetic_env,
            learning_rate=0.0003,
            n_steps=2048,
            verbose=1,
        )
        
        # Train
        self.pretrained_model.learn(
            total_timesteps=timesteps,
            progress_bar=True,
        )
        
        # Save
        self.pretrained_model.save(save_path)
        
        print(f"Pre-training complete! Model saved to {save_path}")
    
    def stage2_finetune(
        self,
        real_env,
        pretrained_path: str,
        timesteps: int = 100000,
        save_path: str = "models/finetuned_model.zip",
        freeze_layers: bool = False,
    ):
        """Fine-tune on real problems."""
        
        print("Stage 2: Fine-tuning on real problems...")
        
        # Load pretrained model
        self.finetuned_model = PPO.load(pretrained_path, env=real_env)
        
        # Optionally freeze early layers
        if freeze_layers:
            self._freeze_early_layers()
        
        # Reduce learning rate for fine-tuning
        self.finetuned_model.learning_rate = 0.0001
        
        # Fine-tune
        self.finetuned_model.learn(
            total_timesteps=timesteps,
            progress_bar=True,
            reset_num_timesteps=False,  # Continue from pretrained
        )
        
        # Save
        self.finetuned_model.save(save_path)
        
        print(f"Fine-tuning complete! Model saved to {save_path}")
    
    def _freeze_early_layers(self):
        """Freeze early layers of neural network."""
        # Freeze first half of policy network
        policy = self.finetuned_model.policy
        
        for i, (name, param) in enumerate(policy.named_parameters()):
            if i < len(list(policy.parameters())) // 2:
                param.requires_grad = False
                print(f"Froze layer: {name}")

def generate_synthetic_problems(n_problems: int = 1000):
    """Generate synthetic scheduling problems."""
    
    import random
    
    problems = []
    
    for i in range(n_problems):
        # Gradually increase difficulty
        difficulty = min(i / n_problems, 1.0)
        
        n_courses = int(10 + difficulty * 30)  # 10-40 courses
        n_rooms = int(5 + difficulty * 15)     # 5-20 rooms
        n_instructors = int(n_courses * 0.8)
        
        problem = {
            'courses': generate_random_courses(n_courses),
            'rooms': generate_random_rooms(n_rooms),
            'instructors': generate_random_instructors(n_instructors),
        }
        
        problems.append(problem)
    
    return problems
```

### 3.3: Online Learning

**What**: Agent continues learning from production runs  
**Why**: Adapt to new problem characteristics over time

#### Algorithm: Experience Replay from Production

```python
# src/rl/online/online_learning.py

from collections import deque
import pickle
import time

class OnlineLearningSystem:
    """Enables continuous learning from production."""
    
    def __init__(
        self,
        model_path: str,
        buffer_size: int = 10000,
        update_frequency: int = 100,  # Update every N episodes
    ):
        self.model = PPO.load(model_path)
        self.experience_buffer = deque(maxlen=buffer_size)
        self.update_frequency = update_frequency
        self.episodes_since_update = 0
    
    def collect_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store experience from production run."""
        
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'timestamp': time.time(),
        }
        
        self.experience_buffer.append(experience)
    
    def should_update(self) -> bool:
        """Check if model should be updated."""
        
        self.episodes_since_update += 1
        
        if self.episodes_since_update >= self.update_frequency:
            self.episodes_since_update = 0
            return True
        
        return False
    
    def update_model(self):
        """Update model with collected experiences."""
        
        if len(self.experience_buffer) < 1000:
            print("Not enough experiences for update")
            return
        
        print(f"Updating model with {len(self.experience_buffer)} experiences...")
        
        # Create temporary environment
        temp_env = self._create_replay_env()
        
        # Train on replay buffer
        self.model.set_env(temp_env)
        self.model.learn(
            total_timesteps=10000,
            reset_num_timesteps=False,
        )
        
        # Save updated model
        timestamp = int(time.time())
        self.model.save(f"models/online_updated_{timestamp}.zip")
        
        print("Model updated and saved!")
    
    def save_buffer(self, filepath: str):
        """Save experience buffer for offline analysis."""
        
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.experience_buffer), f)
```

### 3.4: Meta-RL (Learning to Learn)

**What**: Agent learns how to quickly adapt to new problems  
**Why**: Few-shot adaptation for unseen problem instances

#### Algorithm: MAML (Model-Agnostic Meta-Learning)

```python
# src/rl/meta/meta_rl.py

import torch
import torch.nn as nn
from copy import deepcopy

class MAML_Scheduler:
    """Meta-RL for fast adaptation to new scheduling problems."""
    
    def __init__(
        self,
        base_model: PPO,
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
    ):
        self.base_model = base_model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
    
    def meta_train(
        self,
        task_distribution: List,
        n_iterations: int = 1000,
    ):
        """Meta-training loop."""
        
        print("Starting meta-training...")
        
        for iteration in range(n_iterations):
            # Sample batch of tasks
            task_batch = random.sample(task_distribution, 10)
            
            # Compute meta-gradient
            meta_loss = 0
            
            for task in task_batch:
                # Clone model for task-specific adaptation
                adapted_model = deepcopy(self.base_model)
                
                # Inner loop: Fast adaptation to task
                for _ in range(self.inner_steps):
                    # Collect experience on task
                    experiences = self._collect_task_experiences(
                        adapted_model, task
                    )
                    
                    # Gradient step with inner learning rate
                    loss = self._compute_loss(adapted_model, experiences)
                    self._gradient_step(adapted_model, loss, self.inner_lr)
                
                # Outer loop: Evaluate adapted model
                eval_experiences = self._collect_task_experiences(
                    adapted_model, task
                )
                task_loss = self._compute_loss(adapted_model, eval_experiences)
                
                meta_loss += task_loss
            
            # Meta-update: Update base model
            meta_loss /= len(task_batch)
            self._gradient_step(self.base_model, meta_loss, self.meta_lr)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Meta-loss = {meta_loss:.4f}")
        
        print("Meta-training complete!")
    
    def fast_adapt(
        self,
        new_task,
        n_adaptation_steps: int = 5,
    ):
        """Quickly adapt to new task using few examples."""
        
        # Clone meta-learned model
        adapted_model = deepcopy(self.base_model)
        
        # Few-shot adaptation
        for step in range(n_adaptation_steps):
            experiences = self._collect_task_experiences(adapted_model, new_task)
            loss = self._compute_loss(adapted_model, experiences)
            self._gradient_step(adapted_model, loss, self.inner_lr)
        
        return adapted_model
```

---

## Testing & Validation

### Unit Tests

```python
# test/rl/test_gym_env.py

import pytest
import numpy as np
from src.rl.gym_env import (
    StateEncoder,
    ActionMapper,
    RewardCalculator,
    ScheduleEnv,
)

class TestStateEncoder:
    """Test state encoding functionality."""
    
    def test_state_dimensions(self):
        """Test that state has correct dimensions."""
        encoder = StateEncoder()
        population = create_test_population()
        
        state = encoder.encode(population, 0, 0)
        
        assert state.shape == (25,)
        assert np.all((state >= 0) & (state <= 1))
    
    def test_state_normalization(self):
        """Test that all features are normalized."""
        encoder = StateEncoder()
        population = create_test_population()
        
        state = encoder.encode(population, 0, 0)
        
        assert np.all(state >= 0)
        assert np.all(state <= 1)
    
    def test_state_consistency(self):
        """Test that same population produces same state."""
        encoder = StateEncoder()
        population = create_test_population()
        
        state1 = encoder.encode(population, 0, 0)
        state2 = encoder.encode(population, 0, 0)
        
        np.testing.assert_array_almost_equal(state1, state2)

class TestActionMapper:
    """Test action mapping functionality."""
    
    def test_action_space_size(self):
        """Test that action space has 20 actions."""
        mapper = ActionMapper()
        
        assert len(mapper.enabled_actions) == 20
        assert mapper.action_space == 20
    
    def test_action_validity(self):
        """Test that all actions are valid."""
        mapper = ActionMapper()
        
        for action in range(20):
            assert action in mapper.enabled_actions
    
    def test_heuristic_application(self):
        """Test that heuristics are applied correctly."""
        mapper = ActionMapper()
        individual = create_test_individual()
        context = create_test_context()
        
        for action in mapper.enabled_actions:
            if action > 0:  # Skip no-op
                result, success = mapper.apply_action(action, individual, context)
                assert result is not None

class TestRewardCalculator:
    """Test reward calculation."""
    
    def test_reward_range(self):
        """Test that rewards are in valid range."""
        calculator = RewardCalculator()
        
        for _ in range(100):
            population = create_random_population()
            reward = calculator.calculate_reward(population, population, 0.1)
            
            assert -1 <= reward <= 1
    
    def test_improvement_reward(self):
        """Test that improvement gives positive reward."""
        calculator = RewardCalculator()
        
        pop_before = create_population_with_fitness(100)
        pop_after = create_population_with_fitness(80)  # Better
        
        reward = calculator.calculate_reward(pop_before, pop_after, 0.1)
        
        assert reward > 0

class TestScheduleEnv:
    """Test Gym environment."""
    
    def test_reset(self):
        """Test environment reset."""
        env = create_test_env()
        
        obs = env.reset()
        
        assert obs.shape == (25,)
        assert env.current_generation == 0
    
    def test_step(self):
        """Test environment step."""
        env = create_test_env()
        
        obs = env.reset()
        obs, reward, done, info = env.step(1)  # Apply heuristic 1
        
        assert obs.shape == (25,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_episode_termination(self):
        """Test that episodes terminate correctly."""
        env = create_test_env(max_steps=10)
        
        obs = env.reset()
        
        for i in range(15):
            obs, reward, done, info = env.step(0)  # No-op
            
            if i < 10:
                assert not done
            else:
                assert done
                break
```

### Integration Tests

```python
# test/rl/test_integration.py

def test_full_training_pipeline():
    """Test complete training pipeline."""
    
    # Create environment
    env = create_test_env()
    
    # Create agent
    agent = PPO("MlpPolicy", env)
    
    # Train
    agent.learn(total_timesteps=1000)
    
    # Evaluate
    mean_reward = evaluate_agent(agent, env, n_episodes=5)
    
    assert mean_reward is not None
    
    # Save
    agent.save("test_model.zip")
    
    # Load
    loaded_agent = PPO.load("test_model.zip")
    
    # Predict
    obs = env.reset()
    action, _ = loaded_agent.predict(obs)
    
    assert 0 <= action < 20

def test_ga_integration():
    """Test RL integration with GA scheduler."""
    
    # Create scheduler with RL enabled
    scheduler = GAScheduler(context, use_rl=True)
    
    # Run
    best_solution = scheduler.run()
    
    # Check that RL was used
    assert scheduler.rl_controller.rl_calls > 0
    
    # Check solution quality
    assert best_solution.fitness.values[0] < float('inf')
```

---

## Step-by-Step Commands

### Complete Workflow

```bash
# ========================================
# PHASE 2.2: TRAINING
# ========================================

# Step 1: Train your first RL agent (small scale)
python src/rl/training/train_script.py

# Step 2: View training progress
tensorboard --logdir logs/tensorboard/
# Open browser to http://localhost:6006

# Step 3: Train with curriculum (better results)
python src/rl/training/train_with_curriculum.py

# Step 4: Hyperparameter tuning (optional)
python src/rl/training/tune_hyperparameters.py

# ========================================
# PHASE 2.3: DEPLOYMENT
# ========================================

# Step 5: Test model loading speed
python -c "
from src.rl.deployment import ModelLoader
loader = ModelLoader()
model = loader.load_model('first_ppo_model', 'ppo')
print('Model loaded successfully!')
"

# Step 6: Test inference speed
python src/rl/deployment/test_inference.py

# Step 7: Run GA with RL enabled
# First, update configs/prod.yaml:
#   rl:
#     enabled: true
#     mode: inference
#     agent:
#       model_path: models/rl_agents/first_ppo_model.zip

uv run prod

# ========================================
# PHASE 2.4: EVALUATION
# ========================================

# Step 8: Run full evaluation
python src/rl/evaluation/run_evaluation.py

# Step 9: Generate plots
python src/rl/visualization/create_plots.py

# Step 10: View results
ls output/rl_metrics/
# Should see: fitness_comparison.png, convergence_speed.png, etc.
```

---

## Troubleshooting Guide

### Problem: Training is very slow

**Symptoms**: Takes hours to train  
**Solutions**:
1. Reduce `total_timesteps` (start with 10,000)
2. Use smaller population size (50 instead of 200)
3. Reduce `max_steps_per_episode` (50 instead of 100)
4. Check if GPU is being used: `torch.cuda.is_available()`

### Problem: Reward stays at 0 or negative

**Symptoms**: Agent not learning, reward flat  
**Solutions**:
1. Check reward calculation - is it actually changing?
2. Increase reward weights for fitness
3. Try different learning rate (0.001 instead of 0.0003)
4. Make sure environment is returning correct rewards

### Problem: Model loading fails

**Symptoms**: "Model not found" error  
**Solutions**:
1. Check file path: `models/rl_agents/model_name.zip`
2. Make sure you saved model after training
3. Use absolute paths if relative paths fail

### Problem: Inference too slow (>10ms)

**Symptoms**: GA runs very slowly with RL  
**Solutions**:
1. Use `deterministic=True` in predict (faster)
2. Batch predictions if possible
3. Use CPU-optimized model (avoid GPU transfer overhead)
4. Enable action caching for repeated states

### Problem: RL worse than random

**Symptoms**: Evaluation shows RL underperforms  
**Solutions**:
1. Train longer (maybe not learned yet)
2. Check if reward function makes sense
3. Try curriculum learning
4. Verify environment is correct (state/action/reward)
5. Use hyperparameter tuning

---

## Complete Implementation Roadmap

### Phase 2.2: Training Infrastructure (Week 1-2)

**Task 1: RLTrainer class** (2 days)
- [ ] Create `src/rl/training/trainer.py`
- [ ] Implement basic training loop with progress tracking
- [ ] Add model saving/loading functionality
- [ ] Add TensorBoard integration
- [ ] Test with 10K timesteps smoke test

**Task 2: Training script** (1 day)
- [ ] Create `src/rl/training/train_script.py`
- [ ] Load scheduling data and create initial population
- [ ] Initialize environment and trainer
- [ ] Add command-line arguments for configuration
- [ ] Test end-to-end training

**Task 3: CurriculumManager** (2 days)
- [ ] Create `src/rl/training/curriculum.py`
- [ ] Implement 3-stage curriculum (easy→medium→hard)
- [ ] Add stage transition logic with performance thresholds
- [ ] Implement problem filtering by difficulty
- [ ] Test curriculum progression

**Task 4: Hyperparameter tuning** (2 days)
- [ ] Create `src/rl/training/tune_hyperparameters.py`
- [ ] Implement Optuna integration for Bayesian optimization
- [ ] Add grid search alternative for quick tuning
- [ ] Test 10-20 configurations
- [ ] Save best configuration to config file

**Task 5: Training callbacks** (1 day)
- [ ] Create `src/rl/training/callbacks.py`
- [ ] Implement PeriodicEvaluationCallback
- [ ] Implement EarlyStoppingCallback
- [ ] Implement CheckpointCallback
- [ ] Test callback integration

**Deliverables**: Trained PPO agent (100K timesteps), training logs, best hyperparameters

---

### Phase 2.3: Deployment & Integration (Week 3-4)

**Task 6: ModelLoader** (1 day)
- [ ] Create `src/rl/deployment/model_loader.py`
- [ ] Implement model caching for fast reloading
- [ ] Add model version management
- [ ] Benchmark loading time (<100ms target)
- [ ] Test with multiple models

**Task 7: RLInference engine** (1 day)
- [ ] Create `src/rl/deployment/inference.py`
- [ ] Implement timeout protection (10ms target)
- [ ] Add performance monitoring
- [ ] Add batch prediction support
- [ ] Benchmark inference latency

**Task 8: HybridController** (2 days)
- [ ] Create `src/rl/hybrid/hybrid_controller.py`
- [ ] Implement 3 modes (RL-primary, RL-fallback, RL-assisted)
- [ ] Add fallback strategies (random, greedy, round-robin)
- [ ] Implement usage statistics tracking
- [ ] Test mode switching

**Task 9: GA integration** (2 days)
- [ ] Modify `src/core/ga_scheduler.py` to support RL mode
- [ ] Add `_init_rl()` method for component initialization
- [ ] Implement `_apply_rl_operators()` method
- [ ] Add RL enable/disable configuration
- [ ] Test full GA run with RL enabled

**Task 10: Deployment tests** (1 day)
- [ ] Create `test/rl/test_deployment.py`
- [ ] Test model loading speed
- [ ] Test inference latency
- [ ] Test hybrid controller switching
- [ ] Test full GA integration

**Deliverables**: Production-ready RL system integrated with GA

---

### Phase 2.4: Evaluation & Comparison (Week 5-6)

**Task 11: BaselineStrategies** (1 day)
- [ ] Create `src/rl/evaluation/baselines.py`
- [ ] Implement 5 baseline strategies
- [ ] Test each baseline independently
- [ ] Document baseline characteristics

**Task 12: RLEvaluator** (2 days)
- [ ] Create `src/rl/evaluation/evaluator.py`
- [ ] Implement multi-strategy evaluation (10 runs each)
- [ ] Add results aggregation and comparison
- [ ] Add progress tracking and logging
- [ ] Test with all strategies

**Task 13: MetricsCollector** (1 day)
- [ ] Create `src/rl/evaluation/metrics.py`
- [ ] Implement comprehensive metrics tracking
- [ ] Add JSON export functionality
- [ ] Test metrics collection across runs

**Task 14: Statistical analysis** (2 days)
- [ ] Create `src/rl/evaluation/statistical_tests.py`
- [ ] Implement t-tests and Mann-Whitney U tests
- [ ] Add effect size calculations (Cohen's d)
- [ ] Implement ANOVA and post-hoc tests
- [ ] Generate comparison report

**Task 15: TrainingVisualizer** (1 day)
- [ ] Create `src/rl/visualization/training_plots.py`
- [ ] Implement reward curves over episodes
- [ ] Add loss plots (policy, value)
- [ ] Add entropy plots
- [ ] Test plot generation

**Task 16: PerformanceVisualizer** (1 day)
- [ ] Create `src/rl/visualization/performance_plots.py`
- [ ] Implement box plots for fitness comparison
- [ ] Add bar charts for convergence speed
- [ ] Add scatter plots for time vs quality
- [ ] Generate comparison figures

**Task 17: HeuristicAnalyzer** (1 day)
- [ ] Create `src/rl/visualization/heuristic_plots.py`
- [ ] Implement usage frequency histograms
- [ ] Add effectiveness heatmaps
- [ ] Add state-action correlation plots
- [ ] Test visualizations

**Task 18: Evaluation runner** (1 day)
- [ ] Create `src/rl/evaluation/run_evaluation.py`
- [ ] Orchestrate full evaluation pipeline
- [ ] Generate all reports and plots
- [ ] Add command-line interface
- [ ] Test end-to-end evaluation

**Deliverables**: Comprehensive evaluation report with statistical significance

---

### Testing Suite (Week 7)

**Task 19: Unit tests for gym environment** (1 day)
- [ ] Create `test/rl/test_gym_env.py`
- [ ] Test StateEncoder (dimensions, normalization, consistency)
- [ ] Test ActionMapper (space size, validity, application)
- [ ] Test RewardCalculator (range, improvement detection)
- [ ] Test ScheduleEnv (reset, step, termination)

**Task 20: Integration tests for training** (1 day)
- [ ] Create `test/rl/test_training.py`
- [ ] Test trainer initialization
- [ ] Test training loop execution
- [ ] Test checkpointing
- [ ] Test model saving/loading

**Task 21: Integration tests for deployment** (1 day)
- [ ] Create `test/rl/test_deployment.py`
- [ ] Test model loader
- [ ] Test inference engine
- [ ] Test hybrid controller
- [ ] Test GA integration

**Task 22: Performance benchmarks** (1 day)
- [ ] Create `test/rl/test_performance.py`
- [ ] Benchmark inference latency (<10ms)
- [ ] Benchmark model loading (<100ms)
- [ ] Benchmark memory usage
- [ ] Generate performance report

**Deliverables**: Complete test suite with >80% coverage

---

### Documentation (Week 8)

**Task 23: RL Architecture Guide** (1 day)
- [ ] Create `docs/RL_ARCHITECTURE.md`
- [ ] Document system overview
- [ ] Explain component interactions
- [ ] Add data flow diagrams
- [ ] Document design decisions

**Task 24: RL Training Guide** (1 day)
- [ ] Create `docs/RL_TRAINING_GUIDE.md`
- [ ] Setup instructions
- [ ] Training procedures
- [ ] Hyperparameter tuning guide
- [ ] Curriculum learning guide

**Task 25: RL Integration Guide** (1 day)
- [ ] Create `docs/RL_INTEGRATION_GUIDE.md`
- [ ] How to use RL in GA
- [ ] Configuration options
- [ ] Hybrid modes explanation
- [ ] Troubleshooting guide

**Task 26: RL Quick Start** (Half day)
- [ ] Create `docs/RL_QUICKSTART.md`
- [ ] 5-minute getting started
- [ ] Simple examples
- [ ] Common use cases
- [ ] FAQ section

**Deliverables**: Complete documentation for thesis and future maintenance

---

### Phase 3: Advanced Features (Week 9-12, Optional)

**Task 27: Multi-agent RL** (1 week)
- [ ] Research multi-agent architectures
- [ ] Train specialist agents (construction, perturbation, improvement)
- [ ] Implement meta-controller
- [ ] Evaluate vs single-agent baseline
- [ ] Document findings

**Task 28: Transfer learning** (1 week)
- [ ] Generate synthetic problems (1000s)
- [ ] Pre-train on synthetic data
- [ ] Fine-tune on real problems
- [ ] Measure transfer effectiveness
- [ ] Compare with training from scratch

**Task 29: Online learning** (1 week)
- [ ] Implement experience replay system
- [ ] Add online update mechanism
- [ ] Deploy in production
- [ ] Monitor adaptation over time
- [ ] Evaluate long-term performance

**Task 30: Meta-RL** (1 week)
- [ ] Research MAML and related algorithms
- [ ] Implement meta-training loop
- [ ] Test few-shot adaptation
- [ ] Compare with standard RL
- [ ] Document meta-learning benefits

**Deliverables**: Advanced RL features for thesis contributions

---

## Quick Start Checklist (Minimum Viable Product - 3 Weeks)

### Week 1: Training (Phase 2.2)
- [ ] Day 1-2: Implement `trainer.py` and `train_script.py`
- [ ] Day 3: Train first model (50K timesteps) with TensorBoard
- [ ] Day 4: Implement curriculum learning in `curriculum.py`
- [ ] Day 5: Train with curriculum (100K timesteps)
- [ ] Weekend: Quick hyperparameter tuning (grid search 3x3)

### Week 2: Deployment (Phase 2.3)
- [ ] Day 1: Implement `model_loader.py` and `inference.py`
- [ ] Day 2: Implement `hybrid_controller.py`
- [ ] Day 3: Integrate with `ga_scheduler.py`
- [ ] Day 4: Test full GA run with RL enabled
- [ ] Day 5: Debug and optimize performance
- [ ] Weekend: Write deployment tests

### Week 3: Evaluation (Phase 2.4)
- [ ] Day 1: Implement baseline strategies in `baselines.py`
- [ ] Day 2: Implement `evaluator.py` and `metrics.py`
- [ ] Day 3: Run full evaluation (RL vs 4 baselines, 10 runs each)
- [ ] Day 4: Statistical analysis with `statistical_tests.py`
- [ ] Day 5: Create all visualizations
- [ ] Weekend: Write evaluation report for thesis

**Success Criteria**:
- ✓ RL agent beats random baseline by >20%
- ✓ Statistical significance (p < 0.05)
- ✓ Model loads <100ms, inference <10ms
- ✓ Complete evaluation report with plots

---

## Success Criteria

You'll know you're done when:

### Phase 2.2 Complete ✓
- [ ] Can train RL agent for 100K timesteps
- [ ] Training reward increases over time (visible in TensorBoard)
- [ ] Model saves successfully
- [ ] Curriculum training works

### Phase 2.3 Complete ✓
- [ ] Model loads in <100ms
- [ ] Inference runs in <10ms per prediction
- [ ] GA scheduler runs with RL enabled
- [ ] Hybrid controller switches between RL and fallback correctly

### Phase 2.4 Complete ✓
- [ ] Evaluated 5 strategies (RL + 4 baselines) with 10 runs each
- [ ] RL beats random baseline by >20%
- [ ] Statistical significance confirmed (p < 0.05)
- [ ] Plots generated and look good

---

## What You'll Have at the End

1. **Trained RL Agent**: A `.zip` file with learned policy
2. **Deployment System**: Fast inference engine (<10ms)
3. **Integrated GA**: Your GA scheduler now uses RL to pick heuristics
4. **Evaluation Results**: Proof that RL works better than alternatives
5. **Visualizations**: Plots for your thesis showing RL performance
6. **Production Ready**: Can run `uv run prod` with RL enabled

---

## Next Steps After Phase 2

Once Phase 2 is complete, you can:

1. **Thesis Writing**: Document RL integration in methodology chapter
2. **Further Optimization**: Multi-agent RL, transfer learning
3. **Production Deployment**: Use RL in real scheduling scenarios
4. **Research Publication**: Write paper on RL-based hyper-heuristics

---

## Summary of Mathematical Concepts

### Key Equations Reference

**MDP Framework**:
$$\text{MDP} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$$

**Objective Function**:
$$J(\pi) = \mathbb{E}_{\pi}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

**Reward Function**:
$$R(s_t, a_t, s_{t+1}) = w_f \cdot R_f + w_d \cdot R_d - w_t \cdot R_t$$

**PPO Loss**:
$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

**DQN Loss**:
$$L(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\left[\left(r + \gamma\max_{a'}Q_{\theta^-}(s',a') - Q_\theta(s,a)\right)^2\right]$$

**GAE Advantage**:
$$\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**State Normalization**:
$$s_{\text{norm}} = \frac{s - s_{\min}}{s_{\max} - s_{\min}}$$

**Effect Size (Cohen's d)**:
$$d = \frac{\mu_1 - \mu_2}{\text{pooled\_std}}$$

---

## Algorithm Pseudocode Summary

### Algorithm 1: PPO Training

```
Initialize policy π_θ and value function V_ϕ
for iteration = 1 to N do:
    Collect trajectories {τ_i} using π_θ
    Compute advantages Â_t using GAE
    for epoch = 1 to K do:
        for mini-batch in trajectories do:
            Compute ratio r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            Compute L^CLIP(θ) with clipping
            Update θ using gradient of L^CLIP
            Update ϕ using value loss
    θ_old ← θ
```

### Algorithm 2: Curriculum Learning

```
stages = [(easy, T₁), (medium, T₂), (hard, T₃)]
agent = initialize_agent()
for (difficulty, timesteps) in stages do:
    env = create_env(difficulty)
    agent.set_env(env)
    agent.learn(timesteps)
    if performance_threshold_met():
        continue
    else:
        repeat_stage()
```

### Algorithm 3: Multi-Agent Hierarchical RL

```
Initialize specialists: {Construction, Perturbation, Improvement}
Initialize meta-controller
while not done do:
    # High-level decision
    meta_state = encode_progress_and_phase()
    specialist = meta_controller.select(meta_state)
    
    # Low-level decision
    state = encode_population()
    action = specialist.predict(state)
    
    # Execute and learn
    execute_heuristic(action)
    update_statistics()
```

### Algorithm 4: Transfer Learning Pipeline

```
# Stage 1: Pre-training
synthetic_problems = generate_synthetic(n=1000)
agent = PPO(env=synthetic_problems)
agent.learn(timesteps=500000)
agent.save("pretrained.zip")

# Stage 2: Fine-tuning
agent = PPO.load("pretrained.zip", env=real_problems)
agent.learning_rate *= 0.1  # Reduce LR
freeze_early_layers(agent)
agent.learn(timesteps=100000)
agent.save("finetuned.zip")
```

---

## File Structure Summary

```
src/rl/
├── gym_env/
│   ├── state_encoder.py       (StateEncoder: 25-dim observations)
│   ├── action_space.py        (ActionMapper: 20 discrete actions)
│   ├── reward_calculator.py   (RewardCalculator: multi-component)
│   └── schedule_env.py        (ScheduleEnv: Gym interface)
│
├── agents/
│   ├── ppo_agent.py           (PPO wrapper)
│   ├── dqn_agent.py           (DQN wrapper)
│   └── random_agent.py        (Random baseline)
│
├── training/
│   ├── trainer.py             (RLTrainer: main training loop)
│   ├── train_script.py        (Executable training script)
│   ├── curriculum.py          (CurriculumManager: progressive difficulty)
│   ├── tune_hyperparameters.py (Optuna/grid search)
│   └── callbacks.py           (Evaluation, early stopping, checkpoints)
│
├── deployment/
│   ├── model_loader.py        (ModelLoader: fast loading <100ms)
│   └── inference.py           (RLInference: fast prediction <10ms)
│
├── hybrid/
│   └── hybrid_controller.py   (HybridController: 3 modes + fallbacks)
│
├── evaluation/
│   ├── baselines.py           (5 baseline strategies)
│   ├── evaluator.py           (RLEvaluator: multi-strategy comparison)
│   ├── metrics.py             (MetricsCollector: comprehensive metrics)
│   ├── statistical_tests.py   (t-tests, effect sizes, ANOVA)
│   └── run_evaluation.py      (Evaluation orchestration)
│
├── visualization/
│   ├── training_plots.py      (TrainingVisualizer: reward/loss curves)
│   ├── performance_plots.py   (PerformanceVisualizer: box plots, bars)
│   └── heuristic_plots.py     (HeuristicAnalyzer: usage/effectiveness)
│
├── multi_agent/              (Phase 3: Advanced)
│   └── specialist_agents.py   (Multi-agent hierarchical RL)
│
├── transfer/                  (Phase 3: Advanced)
│   └── transfer_learning.py   (Pre-training + fine-tuning)
│
├── online/                    (Phase 3: Advanced)
│   └── online_learning.py     (Experience replay, continuous adaptation)
│
└── meta/                      (Phase 3: Advanced)
    └── meta_rl.py             (MAML: learning to learn)

test/rl/
├── test_gym_env.py           (Unit tests: StateEncoder, ActionMapper, etc.)
├── test_training.py          (Integration: training pipeline)
├── test_deployment.py        (Integration: model loading, inference)
└── test_performance.py       (Benchmarks: latency, memory)

docs/
├── RL_ARCHITECTURE.md        (System overview, design decisions)
├── RL_TRAINING_GUIDE.md      (Training procedures, hyperparameters)
├── RL_INTEGRATION_GUIDE.md   (GA integration, configuration)
└── RL_QUICKSTART.md          (5-minute getting started)

models/rl_agents/
├── first_ppo_model.zip       (Initial trained model)
├── best_ppo_model.zip        (Best performing model)
├── stage_easy.zip            (Curriculum checkpoints)
├── stage_medium.zip
├── stage_hard.zip
└── checkpoints/              (Training checkpoints)

logs/
├── tensorboard/              (TensorBoard training logs)
└── eval/                     (Evaluation logs)

output/rl_metrics/
├── fitness_comparison.png    (Box plots)
├── convergence_speed.png     (Bar charts)
├── heuristic_usage.png       (Histograms)
├── comparison_report.txt     (Statistical analysis)
└── evaluation_results.json   (Raw metrics)
```

---

## Key Performance Targets

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Model Loading Time** | <100ms | Production startup requirement |
| **Inference Latency** | <10ms | Real-time GA integration |
| **Memory Usage** | <500MB | Efficient resource utilization |
| **Training Time (100K)** | <2 hours | Practical experimentation |
| **RL vs Random Improvement** | >20% | Meaningful performance gain |
| **Statistical Significance** | p<0.05 | Scientific rigor |
| **Test Coverage** | >80% | Code quality assurance |
| **Success Rate (Production)** | >95% | Reliability requirement |

---

## Common Pitfalls & Solutions

### Pitfall 1: Reward Signal Too Sparse
**Problem**: Agent receives 0 reward most of the time  
**Solution**: Add intermediate rewards (diversity bonus, partial improvement)

### Pitfall 2: State Space Too Large
**Problem**: 25 dimensions hard to learn  
**Solution**: Use feature selection, PCA, or curriculum learning

### Pitfall 3: Overfitting to Training Problems
**Problem**: Agent performs well in training, poor on new problems  
**Solution**: Diverse training set, regularization, early stopping

### Pitfall 4: Training Instability
**Problem**: Reward oscillates wildly  
**Solution**: Reduce learning rate, increase batch size, use PPO instead of A2C

### Pitfall 5: Slow Inference
**Problem**: Prediction takes >100ms  
**Solution**: Use deterministic=True, batch predictions, CPU-optimized model

---

## Research Contributions

Your RL integration provides several **novel contributions** for your thesis:

1. **Hyper-Heuristic Selection via Deep RL**: First application of PPO/DQN to university course scheduling heuristic selection

2. **Multi-Component Reward Design**: Novel reward function balancing fitness, diversity, and efficiency

3. **Hierarchical Multi-Agent System**: Specialist agents for different heuristic categories with meta-controller

4. **Transfer Learning for Scheduling**: Pre-training on synthetic problems accelerates adaptation to real instances

5. **Curriculum Learning for Combinatorial Optimization**: Progressive difficulty training improves generalization

6. **Hybrid RL-Heuristic System**: Production-ready fallback mechanisms ensure reliability

7. **Comprehensive Empirical Evaluation**: Rigorous statistical comparison with multiple baselines

**Potential Publications**:
- Conference paper: "Deep Reinforcement Learning for Hyper-Heuristic Selection in University Course Scheduling"
- Journal paper: "A Hierarchical Multi-Agent Reinforcement Learning Approach to Constraint-Based Scheduling"

---

## Next Steps After Implementation

### Short-term (1-2 months)
1. Write methodology chapter in thesis
2. Create presentation slides with visualizations
3. Prepare demo for thesis defense
4. Document lessons learned

### Medium-term (3-6 months)
1. Submit conference paper
2. Extend to other scheduling domains (job shop, nurse rostering)
3. Explore explainable RL (attention mechanisms, SHAP values)
4. Deploy in production for real university

### Long-term (6-12 months)
1. Journal paper with extended results
2. Open-source release with documentation
3. Collaborate with other universities
4. Explore multi-objective RL

---

**Good luck! You've got this!** 

*Remember: Start small, test often, and celebrate progress.*

*This guide contains everything you need from mathematical foundations to production deployment. Follow the roadmap, implement incrementally, and document your journey. Your RL-enhanced scheduling engine will be a significant contribution to the field!*

**Questions or stuck?** Review the troubleshooting section, check the algorithms, and test incrementally. The math is there to understand, the code is there to implement, and the roadmap is there to guide you. You can do this! 
