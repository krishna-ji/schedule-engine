# Memetic RL: RL-Guided Local Search

**Enhancement**: #6 - Adaptive Local Search Intensity  
**Difficulty**: High  
**Impact**: High  
**Priority**: 5

---

## Problem Statement

### Current Local Search: Fixed Budget

```python
# src/lns/lns_operator.py (current)
def apply_igls_repair(individual, context, config):
    """IGLS with fixed computational budget."""
    max_iterations = 100  # Hardcoded
    
    for iteration in range(max_iterations):
        # Try repair moves
        improved = try_repair_move(individual)
        if not improved:
            break
    
    return individual
```

**Problems**:
1. **Wasted computation**: Near-feasible solutions don't need 100 iterations
2. **Insufficient search**: Far-from-feasible solutions need more than 100
3. **No adaptation**: Same budget regardless of population state, generation, or individual quality

---

## Solution: RL Controls Local Search

Let RL agent decide:
1. **When** to apply local search
2. **Which** solutions to search
3. **How much** computational budget to allocate

---

## Architecture Overview

```
┌──────────────────────────────────────────────────┐
│          RL Local Search Controller              │
│  "Which solution needs how much local search?"   │
└────────────────┬─────────────────────────────────┘
                 │
        ┌────────┴────────┬────────────┬────────────┐
        ▼                 ▼            ▼            ▼
   ┌─────────┐      ┌──────────┐  ┌───────┐  ┌───────────┐
   │ Best    │      │ Promising│  │ Near  │  │ Infeasible│
   │ (100it) │      │ (50 it)  │  │(20it) │  │  (200 it) │
   └─────────┘      └──────────┘  └───────┘  └───────────┘
```

---

## Component 1: Local Search Budget Allocation

### State Representation

```python
class LocalSearchState:
    """
    State features for deciding local search budget.
    """
    def __init__(self, individual, population, generation):
        # Individual quality
        self.hard_violations = individual.fitness.values[0]
        self.soft_penalty = individual.fitness.values[1]
        self.pareto_rank = individual.pareto_rank
        
        # Individual characteristics
        self.constraint_breakdown = compute_constraint_breakdown(individual)
        self.estimated_distance_to_feasibility = estimate_repair_difficulty(individual)
        self.previous_repair_attempts = individual.repair_count
        
        # Population context
        self.population_diversity = compute_diversity(population)
        self.generation_progress = generation / max_generations
        self.stagnation_counter = population.stagnation_counter
        
        # Resource constraints
        self.remaining_budget = population.remaining_computational_budget
        self.time_spent_so_far = population.total_time_spent
```

### Action Space: Budget Levels

```python
class LocalSearchAction:
    """
    Discrete action: How many iterations to allocate.
    """
    SKIP = 0          # No local search
    LIGHT = 10        # Quick refinement
    MEDIUM = 50       # Standard search
    HEAVY = 100       # Intensive search
    EXHAUSTIVE = 200  # Maximum effort
```

### RL Policy

```python
class LocalSearchPolicy(nn.Module):
    """
    Neural network that outputs budget allocation.
    """
    def __init__(self, state_dim=30, num_actions=5):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.policy_head = nn.Linear(64, num_actions)  # Budget levels
        self.value_head = nn.Linear(64, 1)             # Expected improvement
    
    def forward(self, state):
        features = self.feature_extractor(state)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        return action_logits, value
```

### Reward Function

```python
def compute_local_search_reward(old_fitness, new_fitness, budget_used):
    """
    Reward = improvement / computational cost
    """
    hard_improvement = old_fitness[0] - new_fitness[0]
    soft_improvement = old_fitness[1] - new_fitness[1]
    
    # Weighted fitness improvement
    fitness_delta = hard_improvement * 10.0 + soft_improvement * 0.1
    
    # Efficiency: improvement per iteration
    efficiency = fitness_delta / (budget_used + 1)
    
    # Bonus for reaching feasibility
    feasibility_bonus = 10.0 if new_fitness[0] == 0 and old_fitness[0] > 0 else 0.0
    
    return efficiency + feasibility_bonus
```

---

## Component 2: Solution Selection for Local Search

### Multi-Armed Bandit Approach

```python
class SolutionSelector:
    """
    Select which solutions to apply local search to.
    """
    def __init__(self, population_size):
        self.ucb_scores = np.zeros(population_size)
        self.improvement_history = [[] for _ in range(population_size)]
    
    def select_solutions(self, population, budget):
        """
        Select top-k solutions based on UCB scores.
        """
        # Compute UCB scores
        for i, ind in enumerate(population):
            avg_improvement = np.mean(self.improvement_history[i]) if self.improvement_history[i] else 0
            exploration_bonus = np.sqrt(2 * np.log(len(population)) / (len(self.improvement_history[i]) + 1))
            
            self.ucb_scores[i] = avg_improvement + exploration_bonus
        
        # Select top solutions
        num_selections = budget // 50  # Assume avg 50 iterations per solution
        selected_indices = np.argsort(self.ucb_scores)[-num_selections:]
        
        return [population[i] for i in selected_indices]
    
    def update(self, solution_index, improvement):
        """
        Update improvement history.
        """
        self.improvement_history[solution_index].append(improvement)
```

---

## Component 3: Meta-Learning Local Search Operators

### Operator Portfolio

```python
class LocalSearchOperators:
    """
    Portfolio of local search operators with learned selection.
    """
    def __init__(self):
        self.operators = {
            "kempe_chain": kempe_chain_search,
            "ejection_chain": ejection_chain_search,
            "variable_depth": variable_depth_search,
            "simulated_annealing": simulated_annealing,
            "tabu_search": tabu_search,
            "large_neighborhood": lns_search
        }
        
        # Track operator effectiveness
        self.operator_rewards = {op: [] for op in self.operators}
    
    def select_operator(self, state):
        """
        Select operator based on Thompson sampling.
        """
        samples = {}
        for op in self.operators:
            if self.operator_rewards[op]:
                # Sample from posterior
                mean = np.mean(self.operator_rewards[op])
                std = np.std(self.operator_rewards[op]) + 1e-6
                samples[op] = np.random.normal(mean, std)
            else:
                # High uncertainty → high sample
                samples[op] = np.random.normal(10, 10)
        
        return max(samples.items(), key=lambda x: x[1])[0]
    
    def apply_operator(self, individual, operator_name, budget):
        """
        Apply selected operator with allocated budget.
        """
        operator_fn = self.operators[operator_name]
        improved = operator_fn(individual, max_iterations=budget)
        return improved
    
    def update_rewards(self, operator_name, reward):
        """
        Update operator effectiveness.
        """
        self.operator_rewards[operator_name].append(reward)
```

---

## Component 4: Adaptive Search Depth

### Dynamic Termination

Instead of fixed iterations, terminate when:
1. No improvement for N consecutive steps
2. Improvement rate drops below threshold
3. Budget exhausted

```python
class AdaptiveLocalSearch:
    def __init__(self, max_budget=200):
        self.max_budget = max_budget
        self.patience = 10  # Stop after 10 non-improving steps
        self.improvement_threshold = 0.01
    
    def search(self, individual, operator):
        """
        Adaptive local search with dynamic termination.
        """
        best_fitness = individual.fitness.values
        no_improvement_count = 0
        budget_used = 0
        
        while budget_used < self.max_budget:
            # Try improvement move
            neighbor = operator.generate_neighbor(individual)
            neighbor_fitness = evaluate_fitness(neighbor)
            
            # Check improvement
            if dominates(neighbor_fitness, best_fitness):
                improvement = fitness_distance(best_fitness, neighbor_fitness)
                
                if improvement > self.improvement_threshold:
                    individual = neighbor
                    best_fitness = neighbor_fitness
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            budget_used += 1
            
            # Early termination
            if no_improvement_count >= self.patience:
                break
        
        return individual, budget_used
```

---

## Integration with GA

### Modified Evolution Loop

```python
class GASchedulerWithRLLocalSearch:
    def evolve_one_generation(self, population):
        """
        NSGA-II evolution with RL-controlled local search.
        """
        # 1. Selection + variation (standard NSGA-II)
        offspring = self.nsga2_variation(population)
        
        # 2. RL decides: Apply local search or not?
        population_state = self.encode_population_state(population + offspring)
        apply_local_search = self.rl_controller.should_apply_local_search(population_state)
        
        if apply_local_search:
            # 3. RL allocates budget to selected solutions
            selected_for_ls = self.rl_controller.select_solutions(
                population + offspring,
                remaining_budget=self.computational_budget
            )
            
            for individual in selected_for_ls:
                # 4. RL determines budget for this individual
                state = self.encode_individual_state(individual, population)
                budget = self.rl_controller.allocate_budget(state)
                
                # 5. Apply local search
                old_fitness = individual.fitness.values
                individual = self.local_search.search(individual, budget)
                new_fitness = individual.fitness.values
                
                # 6. Compute reward and update RL
                reward = compute_local_search_reward(old_fitness, new_fitness, budget)
                self.rl_controller.update(state, budget, reward)
        
        # 7. NSGA-II selection for next generation
        next_gen = self.nsga2_select(population + offspring, self.pop_size)
        
        return next_gen
```

---

## Training Strategy

### Phase 1: Imitation Learning (Warm Start)

```python
def generate_imitation_data():
    """
    Generate expert demonstrations of good budget allocation.
    """
    demonstrations = []
    
    for problem in training_problems:
        population = initialize_population(problem)
        
        for generation in range(max_generations):
            for individual in population:
                state = encode_state(individual, population)
                
                # Expert policy: Allocate based on heuristics
                if individual.fitness.values[0] == 0:
                    budget = 50  # Feasible: moderate search
                elif individual.fitness.values[0] < 5:
                    budget = 100  # Near-feasible: intensive search
                else:
                    budget = 200  # Infeasible: exhaustive search
                
                demonstrations.append((state, budget))
    
    return demonstrations

# Train policy with supervised learning
policy = LocalSearchPolicy()
optimizer = torch.optim.Adam(policy.parameters())

for state, target_budget in demonstrations:
    action_logits, _ = policy(state)
    loss = nn.CrossEntropyLoss()(action_logits, target_budget)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Phase 2: Reinforcement Learning (Fine-Tuning)

```python
# Continue training with PPO
ppo_agent = PPO(
    policy=policy,
    env=LocalSearchEnv(),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64
)

ppo_agent.learn(total_timesteps=100000)
```

---

## Expected Benefits

### 1. Computational Efficiency
- **Current**: Wastes 80% of local search budget on easy cases
- **Expected**: Allocates budget where it's most effective
- **Impact**: 50% reduction in total computational time

### 2. Solution Quality
- **Current**: Under-searches hard problems, over-searches easy ones
- **Expected**: Adaptive effort based on problem difficulty
- **Impact**: 15% better final fitness

### 3. Scalability
- **Current**: Fixed budget doesn't scale with problem size
- **Expected**: Learns appropriate budget for different problem scales
- **Impact**: Handles 2× larger problems with same time budget

---

## Evaluation Metrics

### 1. Budget Efficiency
```python
def evaluate_budget_efficiency(runs):
    """
    Measure: improvement per iteration spent.
    """
    total_improvement = 0
    total_iterations = 0
    
    for run in runs:
        for ls_application in run.local_search_applications:
            improvement = ls_application.fitness_improvement
            iterations = ls_application.budget_used
            
            total_improvement += improvement
            total_iterations += iterations
    
    return total_improvement / total_iterations
```

### 2. Allocation Quality
```python
def evaluate_allocation_quality(runs):
    """
    Correlation: budget allocated vs improvement achieved.
    """
    budgets = [app.budget_allocated for run in runs 
              for app in run.local_search_applications]
    improvements = [app.fitness_improvement for run in runs 
                   for app in run.local_search_applications]
    
    correlation = np.corrcoef(budgets, improvements)[0, 1]
    return correlation
```

---

## Configuration

```yaml
# configs/base.yaml
rl:
  local_search_control:
    enabled: true
    
    budget_allocation:
      action_space: [0, 10, 50, 100, 200]
      max_total_budget: 10000  # Per generation
      
    solution_selection:
      method: "ucb"  # Options: "ucb", "rl_policy", "top_k"
      num_solutions: 5
      
    operator_selection:
      method: "thompson_sampling"
      operators: ["kempe_chain", "ejection_chain", "variable_depth"]
      
    adaptive_termination:
      enabled: true
      patience: 10
      improvement_threshold: 0.01
```

---

## Related Work

### Papers
1. **"Adaptive Operator Selection"** (Fialho et al., 2010)
   - Credit assignment for operators
   - MAB-based selection

2. **"Learning to Optimize"** (Chen et al., 2017)
   - Meta-learning for optimization algorithms
   - LSTM-based controllers

3. **"AlphaGo Zero"** (Silver et al., 2017)
   - MCTS with RL policy
   - Computational budget allocation

---

## Summary

**Problem**: Fixed local search budget is inefficient

**Solution**: RL learns to allocate budget adaptively

**Key Decisions**: When, which solutions, how much budget, which operator

**Expected Impact**: 50% faster, 15% better quality

**Difficulty**: High (requires sophisticated RL training)

**Next Steps**: Imitation learning → RL fine-tuning → Benchmark
