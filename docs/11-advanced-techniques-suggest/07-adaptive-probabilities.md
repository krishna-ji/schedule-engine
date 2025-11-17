# Adaptive Operator Probabilities

**Enhancement**: #7 - RL Controls Crossover/Mutation Rates  
**Difficulty**: Medium  
**Impact**: Medium  
**Priority**: 3

---

## Problem Statement

### Current: Fixed Probabilities

```python
# src/core/ga_scheduler.py (current)
crossover_prob = 0.7  # Hardcoded
mutation_prob = 0.2   # Hardcoded

# Applied uniformly throughout evolution
offspring = toolbox.mate(parent1, parent2) if random.random() < crossover_prob
offspring = toolbox.mutate(offspring) if random.random() < mutation_prob
```

**Problem**: Optimal exploration/exploitation balance changes during search:
- **Early generations**: Need more exploration (higher mutation)
- **Late generations**: Need more exploitation (higher crossover, lower mutation)
- **Stagnation**: Need dramatic shaking (very high mutation)
- **Converging**: Need careful refinement (lower mutation)

---

## Solution: RL-Adaptive Probabilities

Let RL agent learn to tune operator probabilities dynamically.

### State Features for Probability Control

```python
class ProbabilityControlState:
    """State for deciding operator probabilities."""
    
    def __init__(self, population, generation):
        # Convergence indicators
        self.diversity = compute_diversity(population)
        self.unique_fitness_ratio = len(set(fitness)) / len(population)
        self.stagnation_counter = population.stagnation_counter
        
        # Search progress
        self.generation_progress = generation / max_generations
        self.improvement_rate = population.recent_improvement_rate
        
        # Population quality
        self.best_fitness = population.best_fitness
        self.avg_fitness = population.avg_fitness
        self.pareto_front_size = len(population.pareto_front)
        
        # Operator effectiveness history
        self.crossover_success_rate = population.crossover_success_rate
        self.mutation_success_rate = population.mutation_success_rate
```

### Action Space: Probability Levels

```python
class ProbabilityActions:
    """
    Discrete probability levels for operators.
    """
    # Crossover probability
    CROSSOVER_LOW = 0.3
    CROSSOVER_MEDIUM = 0.7
    CROSSOVER_HIGH = 0.9
    
    # Mutation probability
    MUTATION_LOW = 0.05
    MUTATION_MEDIUM = 0.2
    MUTATION_HIGH = 0.5
    
    # Combined actions (3 × 3 = 9 actions)
    ACTIONS = [
        (CROSSOVER_LOW, MUTATION_LOW),
        (CROSSOVER_LOW, MUTATION_MEDIUM),
        (CROSSOVER_LOW, MUTATION_HIGH),
        (CROSSOVER_MEDIUM, MUTATION_LOW),
        (CROSSOVER_MEDIUM, MUTATION_MEDIUM),  # Default
        (CROSSOVER_MEDIUM, MUTATION_HIGH),
        (CROSSOVER_HIGH, MUTATION_LOW),
        (CROSSOVER_HIGH, MUTATION_MEDIUM),
        (CROSSOVER_HIGH, MUTATION_HIGH),
    ]
```

### RL Policy

```python
class ProbabilityPolicy(nn.Module):
    """
    Policy network that outputs operator probabilities.
    """
    def __init__(self, state_dim=15, num_actions=9):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions)
        )
    
    def forward(self, state):
        action_logits = self.network(state)
        return action_logits
    
    def select_probabilities(self, state):
        """
        Select crossover and mutation probabilities.
        """
        logits = self.forward(state)
        action_id = torch.argmax(logits).item()
        
        crossover_prob, mutation_prob = ProbabilityActions.ACTIONS[action_id]
        return crossover_prob, mutation_prob
```

### Reward Function

```python
def compute_probability_reward(old_population, new_population, probabilities_used):
    """
    Reward based on population improvement.
    """
    # Fitness improvement
    old_best = min(ind.fitness.values[0] for ind in old_population)
    new_best = min(ind.fitness.values[0] for ind in new_population)
    fitness_improvement = old_best - new_best
    
    # Diversity maintenance
    old_div = compute_diversity(old_population)
    new_div = compute_diversity(new_population)
    diversity_delta = new_div - old_div
    
    # Combined reward
    reward = fitness_improvement * 1.0 + diversity_delta * 0.5
    
    return reward
```

---

## Mathematical Foundation

### Exploration-Exploitation Trade-off

**Exploration** (high mutation): $$P(\text{explore}) = \mu \cdot (1 - p)$$
**Exploitation** (high crossover): $$P(\text{exploit}) = \chi \cdot p$$

Where:
- $\mu$ = mutation probability
- $\chi$ = crossover probability (chi)
- $p$ = progress ratio $= t / T$ (current gen / total gens)

**Optimal balance** (theoretical):
$$\mu^*(t) = \mu_0 \cdot e^{-\lambda t}$$

RL learns this decay schedule automatically.

---

## Approach 1: Direct Parameter Control

RL directly outputs continuous probability values.

```python
class ContinuousProbabilityPolicy(nn.Module):
    """
    Output continuous probabilities in [0, 1].
    """
    def __init__(self, state_dim=15):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.crossover_head = nn.Linear(32, 1)
        self.mutation_head = nn.Linear(32, 1)
    
    def forward(self, state):
        features = self.feature_extractor(state)
        
        crossover_logit = self.crossover_head(features)
        mutation_logit = self.mutation_head(features)
        
        # Sigmoid to [0, 1]
        crossover_prob = torch.sigmoid(crossover_logit)
        mutation_prob = torch.sigmoid(mutation_logit)
        
        return crossover_prob, mutation_prob
```

---

## Approach 2: Schedule Learning

Learn parameters of a schedule function.

```python
class SchedulePolicy(nn.Module):
    """
    Learn parameters of exponential/linear schedule.
    """
    def __init__(self):
        super().__init__()
        
        # Learnable schedule parameters
        self.crossover_init = nn.Parameter(torch.tensor(0.7))
        self.crossover_decay = nn.Parameter(torch.tensor(0.001))
        
        self.mutation_init = nn.Parameter(torch.tensor(0.3))
        self.mutation_decay = nn.Parameter(torch.tensor(0.002))
    
    def get_probabilities(self, generation):
        """
        Compute probabilities based on learned schedule.
        """
        t = generation
        
        crossover_prob = self.crossover_init * torch.exp(-self.crossover_decay * t)
        mutation_prob = self.mutation_init * torch.exp(-self.mutation_decay * t)
        
        # Clamp to reasonable range
        crossover_prob = torch.clamp(crossover_prob, 0.1, 0.95)
        mutation_prob = torch.clamp(mutation_prob, 0.01, 0.5)
        
        return crossover_prob.item(), mutation_prob.item()
```

---

## Approach 3: Credit Assignment

Track which probabilities led to good offspring.

```python
class CreditAssignment:
    """
    Assign credit to operator probability settings.
    """
    def __init__(self):
        self.history = []  # (probabilities, offspring_quality)
    
    def record(self, crossover_prob, mutation_prob, offspring):
        """
        Record probability setting and resulting offspring quality.
        """
        quality = -offspring.fitness.values[0]  # Negative for minimization
        self.history.append((crossover_prob, mutation_prob, quality))
    
    def learn_best_probabilities(self):
        """
        Find probabilities that produced best offspring (on average).
        """
        if not self.history:
            return 0.7, 0.2  # Defaults
        
        # Group by probability settings
        from collections import defaultdict
        prob_to_qualities = defaultdict(list)
        
        for cx_prob, mut_prob, quality in self.history:
            # Discretize probabilities
            cx_discrete = round(cx_prob, 1)
            mut_discrete = round(mut_prob, 1)
            
            prob_to_qualities[(cx_discrete, mut_discrete)].append(quality)
        
        # Find best average quality
        best_probs = max(prob_to_qualities.items(), 
                        key=lambda x: np.mean(x[1]))
        
        return best_probs[0]
```

---

## Integration with GA

```python
class GASchedulerWithAdaptiveProbs:
    def __init__(self, rl_policy):
        self.rl_policy = rl_policy
        self.crossover_prob = 0.7  # Initial
        self.mutation_prob = 0.2   # Initial
    
    def evolve_one_generation(self, population, generation):
        """
        Evolution with RL-adaptive probabilities.
        """
        # 1. Get current probabilities from RL policy
        state = self.encode_state(population, generation)
        self.crossover_prob, self.mutation_prob = self.rl_policy.select_probabilities(state)
        
        # 2. Apply operators with adaptive probabilities
        offspring = []
        for _ in range(len(population)):
            # Selection
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            # Crossover with adaptive probability
            if random.random() < self.crossover_prob:
                child = self.toolbox.mate(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation with adaptive probability
            if random.random() < self.mutation_prob:
                child = self.toolbox.mutate(child)
            
            offspring.append(child)
        
        # 3. Evaluate offspring
        for ind in offspring:
            ind.fitness.values = self.evaluate(ind)
        
        # 4. Compute reward for RL
        reward = compute_probability_reward(population, offspring, 
                                            (self.crossover_prob, self.mutation_prob))
        
        # 5. Update RL policy
        self.rl_policy.update(state, (self.crossover_prob, self.mutation_prob), reward)
        
        # 6. Selection for next generation
        next_gen = self.nsga2_select(population + offspring, len(population))
        
        return next_gen
```

---

## Expected Benefits

### 1. Better Exploration-Exploitation Balance
- **Current**: Fixed 70/20 may be suboptimal at different stages
- **Expected**: Adaptive balance based on search state
- **Impact**: 10-15% faster convergence

### 2. Automatic Stagnation Recovery
- **Current**: Manual detection and intervention
- **Expected**: RL automatically increases mutation when stagnated
- **Impact**: 20% fewer stagnation episodes

### 3. Problem-Specific Adaptation
- **Current**: Same probabilities for all problem instances
- **Expected**: Learn different strategies for different problem types
- **Impact**: 15% better generalization

---

## Evaluation Metrics

### 1. Probability Trajectory
```python
def plot_probability_trajectory(run_history):
    """
    Visualize how probabilities change during evolution.
    """
    generations = [h.generation for h in run_history]
    crossover_probs = [h.crossover_prob for h in run_history]
    mutation_probs = [h.mutation_prob for h in run_history]
    
    plt.plot(generations, crossover_probs, label="Crossover")
    plt.plot(generations, mutation_probs, label="Mutation")
    plt.legend()
    plt.xlabel("Generation")
    plt.ylabel("Probability")
```

### 2. Correlation with Performance
```python
def analyze_prob_performance_correlation(run_history):
    """
    Measure: Do high mutation probabilities correlate with diversity increase?
    """
    mutation_probs = [h.mutation_prob for h in run_history]
    diversity_deltas = [h.diversity_delta for h in run_history]
    
    correlation = np.corrcoef(mutation_probs, diversity_deltas)[0, 1]
    return correlation
```

---

## Configuration

```yaml
# configs/base.yaml
ga:
  adaptive_probabilities:
    enabled: true
    method: "rl_direct"  # Options: "rl_direct", "rl_schedule", "credit_assignment"
    
    rl_direct:
      state_dim: 15
      action_space: "discrete"  # 9 combinations
      update_frequency: 1  # Every generation
      
    rl_schedule:
      initial_crossover: 0.7
      initial_mutation: 0.3
      learnable_decay: true
      
    credit_assignment:
      window_size: 50  # Recent history
      discretization: 0.1
```

---

## Related Work

### Papers
1. **"Adaptive Parameter Control"** (Eiben et al., 1999)
   - Survey of parameter adaptation methods
   - Self-adaptive GAs

2. **"Meta-EA"** (Grefenstette, 1986)
   - Evolving EA parameters
   - Meta-level evolution

---

## Summary

**Problem**: Fixed probabilities suboptimal across evolution stages

**Solution**: RL learns to adapt probabilities based on search state

**Expected Impact**: 10-15% faster convergence, 20% better stagnation handling

**Next Steps**: Implement discrete policy → Train on curriculum → Benchmark
