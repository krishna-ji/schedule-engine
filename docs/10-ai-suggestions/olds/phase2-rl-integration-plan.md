# Phase 2: RL Integration - Complete Implementation Plan

**Status**:  Planning (Phase 1.5 Heuristic Toolbox  Complete)  
**Target**: Full RL-guided heuristic selection system  
**Estimated Effort**: 4-6 weeks

---

## Phase 1.5 Completion Status 

**What's Ready:**
-  19 heuristic operators across 5 categories
-  Decorator-based registry system
-  Config-driven killswitches
-  Metadata system for introspection
-  Statistics tracking template
-  Action space defined (19 operators)

**Architecture Foundation:**
- Registry provides function access for RL action space
- Metadata supports reward shaping
- Statistics template ready for RL metrics
- Heuristics can serve as expert demonstrations

---

## Phase 2 Overview

### Goal
Integrate Reinforcement Learning agent to **intelligently select which heuristic to apply** during GA evolution, replacing random/fixed heuristic strategies with learned policies.

### Core Concept
```
RL Agent observes GA state → selects heuristic → applies heuristic → receives reward
```

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     RL-Enhanced GA System                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐      ┌──────────────┐     ┌────────────┐ │
│  │  GA Engine   │ ───► │  RL Agent    │ ──► │ Heuristic  │ │
│  │  (NSGA-II)   │      │  (PPO/DQN)   │     │  Toolbox   │ │
│  └──────────────┘      └──────────────┘     └────────────┘ │
│         │                      │                    │        │
│         │                      ▼                    │        │
│         │              ┌──────────────┐             │        │
│         └──────────────│ Gym Env      │◄────────────┘        │
│                        │ (Wrapper)    │                      │
│                        └──────────────┘                      │
│                               │                              │
│                               ▼                              │
│                        ┌──────────────┐                      │
│                        │ Reward Fn    │                      │
│                        └──────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 2.1: Gym Environment (Week 1-2)

### Goal
Create OpenAI Gym environment wrapping GA + Heuristic Toolbox

### Tasks

#### 2.1.1: Environment Skeleton
**Files to create:**
- `src/rl/gym_env/__init__.py`
- `src/rl/gym_env/schedule_env.py` - Main Gym environment
- `src/rl/gym_env/state_encoder.py` - Convert GA state to observation
- `src/rl/gym_env/reward_calculator.py` - Compute rewards

**State Space Design:**
```python
observation = {
    # Population metrics
    "best_fitness": float,           # Best fitness (hard violations)
    "avg_fitness": float,            # Average fitness
    "worst_fitness": float,          # Worst fitness
    "fitness_std": float,            # Fitness standard deviation
    
    # Diversity metrics
    "population_diversity": float,   # Pairwise distance metric
    "unique_solutions": int,         # Number of unique individuals
    
    # Progress metrics
    "generation": int,               # Current generation
    "generations_without_improvement": int,
    "convergence_rate": float,       # Rate of fitness improvement
    
    # Constraint violations
    "total_hard_violations": int,
    "total_soft_violations": int,
    "avg_violations_per_individual": float,
    
    # Heuristic history (last N applications)
    "recent_heuristic_ids": [int] * 10,  # Last 10 heuristics applied
    "recent_improvements": [float] * 10,  # Their improvement scores
    
    # Context
    "elapsed_time": float,           # Seconds since start
    "population_size": int,
    "total_genes": int,
}
```

**Action Space:**
```python
# Discrete action space: 19 heuristics + "no-op"
action_space = Discrete(20)

action_mapping = {
    0: "no_op",  # Do nothing
    1: "largest_degree_first",
    2: "most_constrained_first",
    # ... 3-19: all other heuristics
}
```

**Environment Interface:**
```python
class ScheduleEnv(gym.Env):
    def __init__(self, config, context):
        self.action_space = Discrete(20)
        self.observation_space = Dict({...})
        
    def reset(self):
        """Initialize new GA run"""
        return observation
    
    def step(self, action):
        """Apply heuristic, advance GA, return reward"""
        return observation, reward, done, info
    
    def _apply_heuristic(self, action_id):
        """Call selected heuristic from toolbox"""
        
    def _calculate_reward(self):
        """Compute reward based on fitness improvement"""
```

**Deliverables:**
-  Gym environment with 20 discrete actions
-  State encoder converting GA metrics to observation
-  Reward calculator with multiple strategies
-  Unit tests for environment logic

---

#### 2.1.2: Reward Function Design
**File:** `src/rl/gym_env/reward_calculator.py`

**Reward Strategies:**

1. **Fitness Improvement (Primary)**
   ```python
   reward = (old_best_fitness - new_best_fitness) * 100
   # Positive reward for improvement, negative for degradation
   ```

2. **Population-Level Improvement**
   ```python
   reward = (old_avg_fitness - new_avg_fitness) * 50
   # Encourage overall population quality
   ```

3. **Diversity Bonus**
   ```python
   diversity_bonus = population_diversity * 10
   # Reward maintaining diversity
   ```

4. **Time Penalty**
   ```python
   time_penalty = -execution_time_seconds * 0.1
   # Penalize slow heuristics
   ```

5. **Combined Reward**
   ```python
   total_reward = (
       fitness_improvement * 1.0 +
       diversity_bonus * 0.2 +
       time_penalty * 0.5
   )
   ```

**Reward Shaping Options:**
- Dense rewards (every step)
- Sparse rewards (only on improvement)
- Curriculum learning (easier → harder problems)

**Deliverables:**
-  Multiple reward calculation strategies
-  Configurable reward weights
-  Reward normalization/scaling
-  Logging for reward analysis

---

#### 2.1.3: Integration Hooks
**File:** `src/core/ga_scheduler.py` (modify)

**Hook Points:**
```python
class GAScheduler:
    def __init__(self, context, config, rl_env=None):
        self.rl_env = rl_env  # Optional RL environment
        
    def evolve(self, population):
        for gen in range(self.ngen):
            # Standard GA operations
            offspring = self.select_and_vary(population)
            
            # RL Hook: Query agent for heuristic
            if self.rl_env:
                state = self.rl_env.get_state(population, gen)
                action = self.rl_env.agent.predict(state)
                self.rl_env.apply_action(action, population)
            
            # Continue GA
            population = self.update_population(offspring)
```

**Deliverables:**
-  GA scheduler accepts optional RL environment
-  Hook points for state observation
-  Hook points for action application
-  Backward compatible (works without RL)

---

## Phase 2.2: RL Agent Training (Week 3-4)

### Goal
Train RL agents to select heuristics intelligently

### Tasks

#### 2.2.1: Training Infrastructure
**Files to create:**
- `src/rl/agents/__init__.py`
- `src/rl/agents/ppo_agent.py` - PPO implementation
- `src/rl/agents/dqn_agent.py` - DQN implementation
- `src/rl/agents/random_agent.py` - Random baseline
- `src/rl/training/trainer.py` - Training loop
- `src/rl/training/evaluator.py` - Agent evaluation
- `src/rl/training/logger.py` - Metrics logging

**Training Loop:**
```python
def train_rl_agent(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state)
            agent.learn()
            
            state = next_state
            episode_reward += reward
        
        log_metrics(episode, episode_reward, info)
```

**Deliverables:**
-  PPO agent (Stable-Baselines3)
-  DQN agent (Stable-Baselines3)
-  Random baseline for comparison
-  Training loop with checkpointing
-  TensorBoard logging

---

#### 2.2.2: Hyperparameter Tuning
**File:** `configs/rl_config.yaml`

```yaml
rl:
  agent_type: ppo  # Options: ppo, dqn, random
  
  training:
    num_episodes: 1000
    max_steps_per_episode: 100
    learning_rate: 0.0003
    batch_size: 64
    gamma: 0.99  # Discount factor
    
  ppo:
    n_steps: 2048
    n_epochs: 10
    clip_range: 0.2
    ent_coef: 0.01  # Entropy coefficient
    
  dqn:
    buffer_size: 100000
    learning_starts: 1000
    target_update_interval: 1000
    epsilon_start: 1.0
    epsilon_end: 0.05
    epsilon_decay: 0.995
    
  reward:
    fitness_weight: 1.0
    diversity_weight: 0.2
    time_penalty_weight: 0.5
    
  curriculum:
    enabled: true
    stages:
      - name: easy
        generations: 50
        population_size: 20
      - name: medium
        generations: 100
        population_size: 50
      - name: hard
        generations: 200
        population_size: 100
```

**Deliverables:**
-  RL config model in Pydantic
-  Hyperparameter sweep scripts
-  Grid search / Bayesian optimization
-  Best hyperparameters documented

---

#### 2.2.3: Curriculum Learning
**File:** `src/rl/training/curriculum.py`

**Stages:**
1. **Stage 1 (Easy)**: Small problem, 50 generations
   - Goal: Learn basic heuristic effects
   - Success: Agent beats random baseline

2. **Stage 2 (Medium)**: Medium problem, 100 generations
   - Goal: Learn when to apply which heuristic
   - Success: Agent beats fixed strategy

3. **Stage 3 (Hard)**: Full problem, 200 generations
   - Goal: Handle complex scheduling
   - Success: Agent matches/beats human expert

**Progressive Difficulty:**
```python
curriculum_stages = [
    {"num_courses": 10, "num_rooms": 5, "generations": 50},
    {"num_courses": 20, "num_rooms": 10, "generations": 100},
    {"num_courses": 40, "num_rooms": 20, "generations": 200},
]
```

**Deliverables:**
-  Curriculum stage definitions
-  Automatic progression logic
-  Stage-specific success criteria
-  Gradual difficulty scaling

---

## Phase 2.3: Agent Evaluation (Week 5)

### Goal
Compare RL agents against baselines and analyze performance

### Tasks

#### 2.3.1: Baseline Strategies
**File:** `src/rl/baselines/strategies.py`

**Strategies to Compare:**
1. **Random**: Random heuristic selection
2. **Round Robin**: Cycle through heuristics
3. **Fixed Priority**: Always apply heuristics in priority order
4. **Greedy**: Apply heuristic that improved most recently
5. **Expert Rules**: Hand-crafted if-then rules
6. **RL Agent**: Trained agent

**Evaluation Metrics:**
```python
metrics = {
    # Performance
    "final_best_fitness": float,
    "convergence_speed": float,  # Generations to convergence
    "solution_quality": float,   # Final violation count
    
    # Efficiency
    "total_time_seconds": float,
    "heuristic_applications": int,
    "avg_improvement_per_application": float,
    
    # Diversity
    "final_population_diversity": float,
    "diversity_over_time": List[float],
    
    # Heuristic usage
    "heuristic_usage_counts": Dict[str, int],
    "most_effective_heuristics": List[str],
}
```

**Deliverables:**
-  6 baseline strategies implemented
-  Evaluation harness
-  Statistical significance tests
-  Performance comparison tables

---

#### 2.3.2: Visualization & Analysis
**Files to create:**
- `src/rl/visualization/training_plots.py`
- `src/rl/visualization/heuristic_heatmap.py`
- `src/rl/visualization/action_distribution.py`

**Plots to Generate:**
1. **Training Curves**
   - Reward over episodes
   - Loss over episodes
   - Action entropy over time

2. **Performance Comparison**
   - Final fitness: RL vs baselines (box plot)
   - Convergence speed (bar chart)
   - Time efficiency (scatter plot)

3. **Heuristic Analysis**
   - Heuristic usage frequency (histogram)
   - Heuristic effectiveness (heatmap)
   - State-action correlation (confusion matrix)

4. **Policy Visualization**
   - Action probability distribution by state
   - Q-value heatmaps (for DQN)
   - Value function landscape

**Deliverables:**
-  Comprehensive visualization suite
-  Automated report generation
-  Interactive plots (Plotly)
-  Export to PDF/HTML

---

## Phase 2.4: Production Integration (Week 6)

### Goal
Integrate trained RL agent into production GA scheduler

### Tasks

#### 2.4.1: Model Deployment
**Files to create:**
- `src/rl/deployment/model_loader.py`
- `src/rl/deployment/inference.py`
- `models/rl_agents/` - Trained model checkpoints

**Model Loading:**
```python
class RLModelLoader:
    @staticmethod
    def load_agent(model_path: str, device: str = "cpu"):
        """Load trained RL agent from checkpoint"""
        agent = PPO.load(model_path, device=device)
        return agent
    
    @staticmethod
    def load_best_agent(config):
        """Load best performing agent from experiments"""
        best_model_path = config.rl.best_model_path
        return RLModelLoader.load_agent(best_model_path)
```

**Inference Mode:**
```python
class RLInference:
    def __init__(self, agent, deterministic=True):
        self.agent = agent
        self.deterministic = deterministic
    
    def select_heuristic(self, state):
        """Select heuristic using trained policy"""
        action, _ = self.agent.predict(state, deterministic=self.deterministic)
        return action
```

**Deliverables:**
-  Model serialization/deserialization
-  Fast inference (<1ms per prediction)
-  Fallback to heuristics if model unavailable
-  Model versioning system

---

#### 2.4.2: Configuration Integration
**File:** `configs/base.yaml` (extend)

```yaml
rl:
  enabled: false  # Master killswitch
  
  mode: inference  # Options: training, inference, disabled
  
  agent:
    type: ppo  # Options: ppo, dqn
    model_path: models/rl_agents/ppo_best.zip
    deterministic: true  # Use deterministic policy
    
  fallback:
    enabled: true  # Fallback to heuristics if RL fails
    strategy: fixed_priority  # Fallback strategy
    
  monitoring:
    log_actions: true
    log_states: false
    log_rewards: true
    
  performance:
    max_inference_time_ms: 10  # Timeout for prediction
    batch_prediction: false  # Batch multiple predictions
```

**Deliverables:**
-  RL config section in YAML
-  Pydantic model for RLConfig
-  Config validation
-  Environment-specific overrides (test/prod)

---

#### 2.4.3: Hybrid Mode
**File:** `src/rl/hybrid/hybrid_controller.py`

**Hybrid Strategies:**
1. **RL-Primary**: RL selects, heuristics execute
2. **RL-Assisted**: Heuristics select, RL adjusts parameters
3. **RL-Fallback**: Try RL first, fallback to heuristics on failure
4. **RL-Ensemble**: Combine RL predictions with heuristic scores

**Implementation:**
```python
class HybridController:
    def select_heuristic(self, state, population):
        if self.config.rl.enabled:
            try:
                # Try RL agent
                action = self.rl_agent.predict(state)
                
                # Validate action
                if self._is_valid_action(action, state):
                    return action
            except Exception as e:
                logger.warning(f"RL prediction failed: {e}, using fallback")
        
        # Fallback to heuristics
        return self._fallback_strategy(state, population)
```

**Deliverables:**
-  Hybrid controller implementation
-  Multiple hybrid strategies
-  Seamless switching between modes
-  Performance monitoring

---

## Phase 2.5: Advanced Features (Optional)

### 2.5.1: Multi-Agent RL
- Multiple RL agents for different heuristic categories
- Cooperative agent coordination
- Hierarchical RL (high-level: category, low-level: specific heuristic)

### 2.5.2: Transfer Learning
- Pre-train on small problems
- Fine-tune on production problems
- Domain adaptation techniques

### 2.5.3: Online Learning
- Continue learning during production runs
- Adaptive policy updates
- Experience replay from production data

### 2.5.4: Meta-RL
- Learn to learn across problem instances
- Few-shot adaptation to new scheduling scenarios
- Universal heuristic selection policy

---

## Dependencies & Tools

### Required Libraries
```toml
[tool.poetry.dependencies]
# RL frameworks
gymnasium = "^0.29.0"  # OpenAI Gym fork
stable-baselines3 = "^2.0.0"  # RL algorithms
tensorboard = "^2.14.0"  # Logging

# Optional: Advanced RL
ray = "^2.7.0"  # Distributed training
optuna = "^3.3.0"  # Hyperparameter optimization

# Visualization
plotly = "^5.17.0"
matplotlib = "^3.8.0"
seaborn = "^0.13.0"
```

### Hardware Requirements
- **Training**: GPU recommended (CUDA 11.8+)
- **Inference**: CPU sufficient
- **Memory**: 8GB+ RAM for training

---

## Testing Strategy

### Unit Tests
- `test/rl/test_gym_env.py` - Environment logic
- `test/rl/test_state_encoder.py` - State encoding
- `test/rl/test_reward_calculator.py` - Reward computation
- `test/rl/test_agents.py` - Agent behavior

### Integration Tests
- `test/rl/test_ga_rl_integration.py` - GA + RL integration
- `test/rl/test_training_pipeline.py` - Full training pipeline
- `test/rl/test_inference.py` - Model loading & inference

### Performance Tests
- Inference latency (<10ms per prediction)
- Memory usage (<500MB additional)
- Training time (<24 hours for basic model)

---

## Success Metrics

### Training Success
-  Agent learns non-random policy (entropy decreases)
-  Training reward increases over episodes
-  Agent converges to stable policy

### Performance Success
-  RL agent beats random baseline by 20%+
-  RL agent matches or beats fixed strategy
-  Solution quality improved by 10%+
-  Convergence speed improved by 15%+

### Production Success
-  Inference latency <10ms per prediction
-  Model size <100MB
-  Zero crashes in 100 production runs
-  Graceful fallback works

---

## Risk Mitigation

### Technical Risks
1. **RL doesn't converge**
   - Mitigation: Start with simple reward, use curriculum learning
   
2. **Training takes too long**
   - Mitigation: Use smaller problems, distributed training (Ray)
   
3. **Agent overfits to training problems**
   - Mitigation: Diverse training set, regularization, validation set
   
4. **Inference too slow for production**
   - Mitigation: Model pruning, quantization, fallback to heuristics

### Implementation Risks
1. **Complex integration with GA**
   - Mitigation: Modular design, backward compatibility
   
2. **Config complexity increases**
   - Mitigation: Sensible defaults, clear documentation
   
3. **Debugging RL is hard**
   - Mitigation: Extensive logging, visualization tools

---

## Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | 2.1.1 | Gym environment skeleton |
| 1-2 | 2.1.2 | Reward function design |
| 2 | 2.1.3 | GA integration hooks |
| 3 | 2.2.1 | Training infrastructure |
| 3-4 | 2.2.2 | Hyperparameter tuning |
| 4 | 2.2.3 | Curriculum learning |
| 5 | 2.3.1 | Baseline comparison |
| 5 | 2.3.2 | Visualization & analysis |
| 6 | 2.4.1 | Model deployment |
| 6 | 2.4.2 | Config integration |
| 6 | 2.4.3 | Hybrid mode |

**Total**: 6 weeks (assumes 20-30 hours/week effort)

---

## Documentation Plan

### Developer Docs
- `docs/RL_ARCHITECTURE.md` - System architecture
- `docs/RL_TRAINING_GUIDE.md` - How to train agents
- `docs/RL_INTEGRATION_GUIDE.md` - How to use in production
- `docs/RL_TROUBLESHOOTING.md` - Common issues & solutions

### User Docs
- `docs/RL_QUICKSTART.md` - Get started in 5 minutes
- `docs/RL_CONFIG_GUIDE.md` - Configuration options
- `docs/RL_FAQ.md` - Frequently asked questions

### Research Docs
- `docs/for_report/RL_METHODOLOGY.md` - Thesis chapter
- `docs/for_report/RL_RESULTS.md` - Experimental results
- `docs/for_report/RL_ANALYSIS.md` - Performance analysis

---

## Next Immediate Steps

### Week 1 Action Items

1. **Day 1-2: Environment Setup**
   - [ ] Create `src/rl/` directory structure
   - [ ] Install gymnasium and stable-baselines3
   - [ ] Create basic Gym environment shell
   - [ ] Test import and basic reset()

2. **Day 3-4: State Space**
   - [ ] Implement state encoder
   - [ ] Test state observation from GA
   - [ ] Validate state space dimensions
   - [ ] Add state normalization

3. **Day 5-6: Action Space**
   - [ ] Map actions to heuristics
   - [ ] Implement action application
   - [ ] Test heuristic execution from action
   - [ ] Add action masking (optional)

4. **Day 7: Reward Function**
   - [ ] Implement basic fitness reward
   - [ ] Test reward calculation
   - [ ] Add reward logging
   - [ ] Validate reward signals

**First Milestone**: Working Gym environment with random agent

---

## References

### RL for Combinatorial Optimization
- "Learning to Solve NP-Complete Problems" - Oriol Vinyals et al.
- "Attention, Learn to Solve Routing Problems!" - Wouter Kool et al.
- "Learning Heuristics for the TSP by Policy Gradient" - Michel Deudon et al.

### RL in Scheduling
- "Job Shop Scheduling with Deep RL" - various papers
- "University Timetabling with RL" - literature review needed

### Stable-Baselines3
- Official docs: https://stable-baselines3.readthedocs.io/
- PPO paper: "Proximal Policy Optimization Algorithms"
- DQN paper: "Deep Q-Network"

---

## Questions to Resolve

1. **State Space**: What's the optimal state representation?
2. **Action Space**: Discrete vs continuous? Hierarchical?
3. **Reward Shaping**: Dense vs sparse rewards?
4. **Training Data**: How many problem instances needed?
5. **Generalization**: Will agent work on unseen problems?
6. **Real-time Learning**: Online updates during production?

---

## Conclusion

Phase 2 builds on the solid foundation of Phase 1.5 Heuristic Toolbox to create an **intelligent, adaptive scheduling system**. The RL agent will learn when and which heuristics to apply, potentially discovering novel strategies that outperform hand-crafted approaches.

**Key Innovation**: Instead of replacing domain knowledge (heuristics) with RL, we **augment** it—RL learns to orchestrate existing heuristics intelligently.

This hybrid approach:
-  Leverages decades of scheduling research (heuristics)
-  Adds adaptive intelligence (RL)
-  Maintains robustness (fallback to heuristics)
-  Enables continuous improvement (online learning)

**Let's build the future of intelligent scheduling! **
