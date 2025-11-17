# Specialist Agents: Task-Specific RL Policies

**Enhancement**: #2 - Separate Agents for Different Solution Regions  
**Difficulty**: Medium  
**Impact**: Medium  
**Priority**: 4

---

## Problem Statement

### Current: Single Monolithic Agent

```python
# One agent handles all scenarios
agent = PPO(policy, env, ...)

# Same agent must learn to:
# 1. Repair infeasible solutions (hard > 0)
# 2. Optimize feasible solutions (hard = 0, minimize soft)
# 3. Diversify stagnated populations
# 4. Intensify promising solutions
```

**Problem**: These tasks require **conflicting strategies**:
- **Infeasible solutions**: Aggressive repair (prioritize feasibility)
- **Feasible solutions**: Careful optimization (preserve feasibility)
- **Stagnated population**: Exploration (increase diversity)
- **Converged population**: Exploitation (refine best solutions)

**Result**: Agent learns compromise policy, suboptimal for all scenarios.

---

## Solution: Specialist Agents

Train multiple agents, each specialized for specific solution characteristics.

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              Agent Coordinator                      │
│  "Which specialist should handle this situation?"   │
└────────────┬────────────────────────────────────────┘
             │
    ┌────────┴────────┬──────────┬──────────┐
    ▼                 ▼          ▼          ▼
┌────────┐      ┌──────────┐  ┌──────┐  ┌──────────┐
│Repair  │      │Optimizer │  │Explor│  │Intensify │
│Agent   │      │Agent     │  │-er   │  │Agent     │
│        │      │          │  │Agent │  │          │
│hard>0  │      │hard=0    │  │div↓  │  │exploit   │
└────────┘      └──────────┘  └──────┘  └──────────┘
```

---

## Specialist Agent Types

### 1. Repair Agent (Infeasible Solutions)

**Specialization**: Move infeasible solutions toward feasibility

**State Features** (prioritized):
```python
state = [
    hard_violations,           # Primary concern
    constraint_breakdown,      # Which constraints to fix
    repair_attempts,           # How many repairs tried
    distance_to_feasibility,   # Estimated effort needed
    # Less important:
    soft_penalty,              # Secondary
    diversity_metrics          # Ignored
]
```

**Action Space**: Focus on repair heuristics
```python
actions = [
    "kempe_chain",           # Conflict resolution
    "ejection_chain",        # Cascading repair
    "random_swap",           # Escape local optima
    "temporal_shift",        # Time adjustment
    "instructor_reassign"    # Availability matching
]
# Excludes: diversity operators, meta-heuristics
```

**Reward Function**: Prioritize hard constraint reduction
```python
def repair_reward(old_fitness, new_fitness):
    hard_delta = old_fitness[0] - new_fitness[0]
    
    # Heavily reward feasibility improvement
    if hard_delta > 0:
        return hard_delta * 10.0
    
    # Small penalty for no improvement (encourage trying different repairs)
    return -0.1
```

**Training Data**: Episodes starting from infeasible solutions
```python
def generate_repair_episode(context):
    # Start with intentionally infeasible solution
    individual = generate_random_individual()  # High violations
    
    # Episode: Try to make it feasible
    for step in range(max_steps):
        if individual.fitness.values[0] == 0:
            break  # Success!
        
        action = repair_agent.select_action(state)
        individual = apply_heuristic(individual, action)
```

---

### 2. Optimizer Agent (Feasible Solutions)

**Specialization**: Optimize soft constraints while preserving feasibility

**State Features** (prioritized):
```python
state = [
    soft_penalty,              # Primary concern
    constraint_breakdown,      # Which soft constraints to improve
    feasibility_margin,        # How safe is current feasibility
    recent_improvements,       # Track optimization progress
    # Less important:
    hard_violations,           # Should be 0
    diversity_metrics          # Not critical
]
```

**Action Space**: Focus on improvement heuristics
```python
actions = [
    "kempe_chain",           # Local improvement
    "ejection_chain",        # Quality refinement
    "room_shuffle",          # Preference optimization
    "temporal_shift",        # Gap reduction
    # Careful, cautious operators that preserve feasibility
]
# Excludes: aggressive perturbations, risky moves
```

**Reward Function**: Soft constraint improvement + feasibility preservation
```python
def optimizer_reward(old_fitness, new_fitness):
    old_hard, old_soft = old_fitness
    new_hard, new_soft = new_fitness
    
    # Severe penalty for losing feasibility
    if new_hard > 0 and old_hard == 0:
        return -100.0
    
    # Reward soft improvement
    soft_delta = old_soft - new_soft
    return soft_delta * 1.0
```

**Training Data**: Episodes starting from feasible solutions
```python
def generate_optimizer_episode(context):
    # Start with feasible solution (hard = 0)
    individual = generate_feasible_individual()
    
    # Episode: Optimize soft constraints
    for step in range(max_steps):
        action = optimizer_agent.select_action(state)
        individual = apply_heuristic(individual, action)
        
        # Early termination if feasibility lost
        if individual.fitness.values[0] > 0:
            break  # Failure
```

---

### 3. Explorer Agent (Stagnated Populations)

**Specialization**: Increase population diversity when stagnated

**State Features** (prioritized):
```python
state = [
    diversity_metrics,         # Primary concern
    unique_fitness_ratio,      # Convergence indicator
    stagnation_counter,        # How long stagnated
    pareto_front_size,         # Quality-diversity balance
    # Less important:
    fitness_values             # Not the focus
]
```

**Action Space**: Focus on diversity operators
```python
actions = [
    "random_restart",          # Fresh solution injection
    "multi_perturbation",      # Aggressive shaking
    "random_swap",             # Escaping local optima
    "adaptive_restart",        # Partial reconstruction
    "archive_injection"        # Import from archive
]
# Includes: risky, exploratory moves
```

**Reward Function**: Diversity increase
```python
def explorer_reward(old_state, new_state):
    diversity_delta = (
        new_state.phenotype_diversity - old_state.phenotype_diversity
    )
    
    # Bonus for breaking convergence
    if new_state.unique_fitness_ratio > old_state.unique_fitness_ratio:
        diversity_delta += 0.5
    
    return diversity_delta * 10.0
```

---

### 4. Intensifier Agent (Promising Solutions)

**Specialization**: Exploit promising regions, refine elite solutions

**State Features** (prioritized):
```python
state = [
    best_fitness,              # How good is the best
    improvement_rate,          # Recent progress
    local_search_budget,       # Computational resources left
    solution_quality_rank,     # Pareto rank
]
```

**Action Space**: Focus on local search and refinement
```python
actions = [
    "variable_depth_search",   # Deep local search
    "kempe_chain",             # Iterative improvement
    "ejection_chain",          # Cascading refinement
    "run_igls"                 # Intensive local search
]
# Computationally expensive, high-quality operators
```

**Reward Function**: Elite solution improvement
```python
def intensifier_reward(old_best, new_best):
    # Only care about improving the best solution
    if dominates(new_best, old_best):
        improvement = fitness_distance(old_best, new_best)
        return improvement * 5.0
    return -0.1
```

---

## Agent Coordination

### Strategy 1: State-Based Routing

```python
class AgentCoordinator:
    def __init__(self):
        self.repair_agent = PPO(...)
        self.optimizer_agent = PPO(...)
        self.explorer_agent = PPO(...)
        self.intensifier_agent = PPO(...)
    
    def select_agent(self, population_state):
        """
        Route to appropriate specialist based on state.
        """
        best_hard = min(ind.fitness.values[0] for ind in population_state.population)
        diversity = population_state.phenotype_diversity
        stagnation = population_state.stagnation_counter
        
        # Decision tree routing
        if best_hard > 0:
            # Population contains infeasible solutions
            return self.repair_agent, "repair"
        
        elif stagnation > 50 or diversity < 0.1:
            # Population is stagnated
            return self.explorer_agent, "explore"
        
        elif diversity < 0.3:
            # Population is converging, but still diverse enough
            return self.intensifier_agent, "intensify"
        
        else:
            # Population is feasible and diverse
            return self.optimizer_agent, "optimize"
    
    def select_action(self, state):
        agent, mode = self.select_agent(state)
        action = agent.predict(state)
        return action, mode
```

### Strategy 2: Bandit-Based Selection

```python
class BanditCoordinator:
    def __init__(self, num_agents=4):
        self.agents = [repair_agent, optimizer_agent, explorer_agent, intensifier_agent]
        self.ucb_scores = np.zeros(num_agents)
        self.counts = np.zeros(num_agents)
        self.total_rewards = np.zeros(num_agents)
    
    def select_agent(self, state, generation):
        """
        Use Upper Confidence Bound (UCB) for agent selection.
        """
        # UCB formula: reward_mean + sqrt(2 * log(t) / n_i)
        exploration_bonus = np.sqrt(2 * np.log(generation + 1) / (self.counts + 1))
        
        mean_rewards = self.total_rewards / (self.counts + 1)
        ucb_scores = mean_rewards + exploration_bonus
        
        agent_id = np.argmax(ucb_scores)
        return self.agents[agent_id], agent_id
    
    def update(self, agent_id, reward):
        """Update statistics after agent application."""
        self.counts[agent_id] += 1
        self.total_rewards[agent_id] += reward
```

### Strategy 3: Meta-Agent

```python
class MetaAgent:
    """
    An RL agent that learns WHICH specialist to use.
    """
    def __init__(self):
        self.specialists = [repair_agent, optimizer_agent, explorer_agent, intensifier_agent]
        
        # Meta-policy: state → specialist_id
        self.meta_policy = PPO(
            policy="MlpPolicy",
            env=MetaEnv(self.specialists),
            ...
        )
    
    def select_action(self, state):
        # Meta-agent selects which specialist to use
        specialist_id = self.meta_policy.predict(state)
        
        # Selected specialist chooses actual heuristic
        specialist = self.specialists[specialist_id]
        action = specialist.predict(state)
        
        return action, specialist_id
```

---

## Training Strategy

### Phase 1: Independent Training

Train each specialist independently on curated datasets:

```python
# 1. Repair Agent
repair_episodes = generate_infeasible_scenarios(1000)
train_agent(repair_agent, repair_episodes, reward_fn=repair_reward)

# 2. Optimizer Agent
optimizer_episodes = generate_feasible_scenarios(1000)
train_agent(optimizer_agent, optimizer_episodes, reward_fn=optimizer_reward)

# 3. Explorer Agent
explorer_episodes = generate_converged_scenarios(1000)
train_agent(explorer_agent, explorer_episodes, reward_fn=explorer_reward)

# 4. Intensifier Agent
intensifier_episodes = generate_promising_scenarios(1000)
train_agent(intensifier_agent, intensifier_episodes, reward_fn=intensifier_reward)
```

### Phase 2: Coordinator Training

Train coordinator to select specialists:

```python
# Train meta-agent or bandit selector
for episode in range(num_episodes):
    state = env.reset()
    
    for step in range(max_steps):
        # Coordinator selects specialist
        specialist, specialist_id = coordinator.select_agent(state)
        
        # Specialist selects action
        action = specialist.predict(state)
        
        # Apply action
        next_state, reward, done, _ = env.step(action)
        
        # Update coordinator
        coordinator.update(specialist_id, reward)
        
        state = next_state
```

---

## Implementation Roadmap

### Week 1-2: Infrastructure
- [ ] Create `src/rl/agents/specialist_agent.py` base class
- [ ] Define specialist-specific state encoders
- [ ] Implement reward functions for each specialist
- [ ] Create data generators for specialist-specific scenarios

### Week 3-4: Train Specialists
- [ ] Train repair agent on infeasible scenarios
- [ ] Train optimizer agent on feasible scenarios
- [ ] Train explorer agent on converged scenarios
- [ ] Train intensifier agent on promising scenarios

### Week 5-6: Coordination
- [ ] Implement state-based routing coordinator
- [ ] Implement UCB-based coordinator
- [ ] Compare coordination strategies
- [ ] Select best coordinator for production

### Week 7-8: Integration & Evaluation
- [ ] Integrate into GA scheduler
- [ ] Benchmark against single-agent baseline
- [ ] Measure per-specialist performance
- [ ] Document findings

---

## Expected Benefits

### 1. Better Task-Specific Performance
- **Repair agent**: 30% faster infeasible → feasible transition
- **Optimizer agent**: 20% better soft constraint optimization
- **Explorer agent**: 50% faster escape from stagnation
- **Intensifier agent**: 15% better elite solution quality

### 2. Reduced Negative Transfer
- **Current**: Agent learns compromise (bad at all tasks)
- **Expected**: Each specialist is expert in its domain

### 3. Interpretability
- **Current**: Black-box policy (unclear why action chosen)
- **Expected**: "Repair agent activated because hard > 0" (transparent reasoning)

---

## Evaluation Metrics

### 1. Per-Specialist Effectiveness
```python
def evaluate_specialist(agent, test_scenarios):
    successes = 0
    for scenario in test_scenarios:
        result = run_episode(agent, scenario)
        if result.goal_achieved:
            successes += 1
    return successes / len(test_scenarios)
```

**Targets**:
- Repair agent: 80% success on infeasible → feasible
- Optimizer agent: 70% success on soft improvement
- Explorer agent: 60% success on diversity increase
- Intensifier agent: 50% success on elite improvement

### 2. Coordination Quality
```python
def evaluate_coordination(coordinator, test_runs):
    correct_selections = 0
    for run in test_runs:
        selected_agent = coordinator.select_agent(run.state)
        oracle_agent = oracle_selector(run.state)  # Ground truth
        if selected_agent == oracle_agent:
            correct_selections += 1
    return correct_selections / len(test_runs)
```

**Target**: 75% match with oracle selector

---

## Risks and Mitigations

### Risk 1: Overhead of Multiple Agents
**Issue**: 4× memory and training time

**Mitigation**:
- Share feature extractors (only policy heads differ)
- Train in parallel on multi-GPU system
- Start with 2 agents (repair + optimizer) before adding more

### Risk 2: Coordinator Failure
**Issue**: Wrong specialist selected → poor performance

**Mitigation**:
- Fallback to round-robin if coordinator uncertain
- Log specialist selections for debugging
- Use simple rule-based coordinator initially

### Risk 3: Specialist Overfitting
**Issue**: Agent only good on training scenarios

**Mitigation**:
- Train on diverse problem instances
- Use regularization (dropout, entropy bonus)
- Test on held-out validation set

---

## Configuration

```yaml
# configs/base.yaml
rl:
  mode: "specialist_agents"  # vs "single_agent"
  
  specialists:
    enabled: true
    
    repair_agent:
      model_path: "models/rl_agents/repair_specialist.zip"
      reward_weight_hard: 10.0
      
    optimizer_agent:
      model_path: "models/rl_agents/optimizer_specialist.zip"
      reward_weight_soft: 1.0
      
    explorer_agent:
      model_path: "models/rl_agents/explorer_specialist.zip"
      reward_weight_diversity: 10.0
      
    intensifier_agent:
      model_path: "models/rl_agents/intensifier_specialist.zip"
      reward_weight_elite: 5.0
  
  coordination:
    strategy: "state_based"  # Options: "state_based", "ucb", "meta_agent"
    
    state_based:
      hard_threshold: 0
      stagnation_threshold: 50
      diversity_threshold_low: 0.1
      diversity_threshold_high: 0.3
```

---

## Related Work

### Papers
1. **"Specialist-Generalist Networks"** (Rosenbaum et al., 2017)
   - Task-specific specialist modules
   - Gating network for module selection

2. **"PathNet"** (Fernando et al., 2017)
   - Evolutionary selection of neural network paths
   - Different paths for different tasks

3. **"Mixture of Experts"** (Jacobs et al., 1991)
   - Classical approach to combining specialized models
   - Gating network learns to route inputs

---

## Summary

**Problem**: Single agent learns suboptimal compromise policy

**Solution**: Train specialist agents for repair, optimization, exploration, intensification

**Coordination**: State-based routing, UCB bandit, or meta-agent selection

**Expected Impact**: 20-30% improvement in task-specific metrics

**Recommended Start**: Repair + Optimizer specialists with state-based routing

**Next Steps**: Phase 1 infrastructure → Phase 2 specialist training → Phase 3 coordination
