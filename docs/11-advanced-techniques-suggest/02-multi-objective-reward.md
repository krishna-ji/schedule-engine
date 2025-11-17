# Multi-Objective Reward Functions for RL

**Enhancement**: #1 - Pareto-Aware RL Rewards  
**Difficulty**: Medium  
**Impact**: High  
**Priority**: 1

---

## Problem Statement

### Current Approach: Scalar Reward
```python
# src/rl/reward.py (current)
def calculate_reward(old_fitness, new_fitness, diversity_delta, time_spent):
    fitness_improvement = (old_fitness[0] - new_fitness[0]) + 
                         (old_fitness[1] - new_fitness[1]) * 0.01
    diversity_bonus = diversity_delta * 0.1
    time_penalty = time_spent * 0.001
    
    return fitness_improvement + diversity_bonus - time_penalty
```

**Problem**: This collapses a multi-objective optimization problem into a single scalar. Information about Pareto trade-offs is lost.

### Why This Fails

#### Example Scenario
```
Old solution: (hard_violations=5, soft_penalty=100)
New solution A: (hard_violations=3, soft_penalty=150)  # Better hard, worse soft
New solution B: (hard_violations=5, soft_penalty=50)   # Same hard, better soft

Scalar rewards:
A: (5-3) + (100-150)*0.01 = 2 - 0.5 = 1.5
B: (5-5) + (100-50)*0.01 = 0 + 0.5 = 0.5

Agent learns: A is better (reward 1.5 > 0.5)
Reality: Both are Pareto improvements, neither dominates
```

**Issue**: The agent learns to prioritize hard constraints (due to higher weight), missing opportunities to optimize soft constraints when hard constraints are already satisfied.

---

## Solution: Pareto-Aware Rewards

### Approach 1: Hypervolume Indicator

The **hypervolume** measures the volume of objective space dominated by a Pareto front, relative to a reference point.

#### Mathematical Definition

Given:
- Pareto front $P = \{s_1, s_2, \ldots, s_n\}$
- Reference point $r = (r_1, r_2)$ (worst acceptable values)
- Objective space $\mathbb{R}^2$ (hard violations, soft penalty)

Hypervolume:
$$HV(P, r) = \text{volume}\left(\bigcup_{s \in P} [s_1, r_1] \times [s_2, r_2]\right)$$

#### Why Hypervolume?
1. **Pareto-compliant**: Monotonic with Pareto dominance
2. **Diversity-aware**: Rewards spread along Pareto front
3. **Single metric**: Despite being multi-objective aware

#### Implementation Strategy

```python
# src/rl/reward.py (enhanced)
import numpy as np
from pymoo.indicators.hv import HV

class HypervolumeReward:
    def __init__(self, ref_point=(100, 10000)):
        """
        Args:
            ref_point: (max_hard, max_soft) - worst acceptable solution
        """
        self.ref_point = np.array(ref_point)
        self.hv_calculator = HV(ref_point=self.ref_point)
    
    def calculate(self, old_population, new_population):
        """
        Calculate reward as change in hypervolume.
        
        Args:
            old_population: List of (hard, soft) tuples before action
            new_population: List of (hard, soft) tuples after action
        
        Returns:
            float: HV increase (positive = improvement)
        """
        old_front = self._extract_pareto_front(old_population)
        new_front = self._extract_pareto_front(new_population)
        
        old_hv = self.hv_calculator(old_front) if len(old_front) > 0 else 0
        new_hv = self.hv_calculator(new_front) if len(new_front) > 0 else 0
        
        return new_hv - old_hv
    
    def _extract_pareto_front(self, population):
        """Extract non-dominated solutions."""
        fronts = []
        for ind in population:
            fitness = np.array([ind.fitness.values[0], ind.fitness.values[1]])
            fronts.append(fitness)
        
        fronts = np.array(fronts)
        is_pareto = self._is_pareto_efficient(fronts)
        return fronts[is_pareto]
    
    def _is_pareto_efficient(self, costs):
        """
        Find Pareto-efficient points (minimization).
        """
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                # Remove dominated points
                is_efficient[is_efficient] = np.any(
                    costs[is_efficient] < c, axis=1
                ) | np.all(costs[is_efficient] == c, axis=1)
        return is_efficient
```

#### Reward Formula with Hypervolume

```python
def calculate_reward(self, state_before, state_after, action, time_spent):
    """
    Multi-component reward with hypervolume as primary signal.
    """
    # 1. Hypervolume contribution
    hv_delta = self.hypervolume_reward.calculate(
        state_before.population,
        state_after.population
    )
    
    # 2. Diversity bonus (secondary)
    diversity_delta = (
        state_after.diversity_metrics['phenotype_diversity'] -
        state_before.diversity_metrics['phenotype_diversity']
    )
    
    # 3. Efficiency penalty (computational cost)
    time_penalty = time_spent * 0.001
    
    # Weighted combination
    reward = (
        hv_delta * 1.0 +              # Primary: Pareto improvement
        diversity_delta * 0.1 +        # Secondary: Diversity
        -time_penalty                  # Cost: Computation time
    )
    
    return reward
```

---

### Approach 2: Decomposition Methods (MOEA/D-Style)

Instead of one agent, train multiple agents with different preference vectors.

#### Mathematical Foundation

**Weight vector**: $\mathbf{w} = (w_1, w_2)$ where $w_1 + w_2 = 1$

**Scalarization function** (Tchebycheff):
$$g^{\text{te}}(\mathbf{x} | \mathbf{w}, \mathbf{z}^*) = \max_{i=1,2} \left\{ w_i |f_i(\mathbf{x}) - z_i^*| \right\}$$

Where:
- $f_1(\mathbf{x})$ = hard violations
- $f_2(\mathbf{x})$ = soft penalty
- $\mathbf{z}^* = (z_1^*, z_2^*)$ = ideal point (best values seen)

#### Implementation

```python
class DecomposedReward:
    def __init__(self, weight_vectors):
        """
        Args:
            weight_vectors: List of (w1, w2) tuples
                e.g., [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]
        """
        self.weight_vectors = weight_vectors
        self.ideal_point = np.array([0, 0])  # Updated dynamically
    
    def calculate(self, fitness, weight_vector):
        """
        Calculate Tchebycheff scalar for given fitness and weight.
        """
        hard, soft = fitness
        w1, w2 = weight_vector
        
        # Update ideal point
        self.ideal_point[0] = min(self.ideal_point[0], hard)
        self.ideal_point[1] = min(self.ideal_point[1], soft)
        
        # Tchebycheff aggregation
        term1 = w1 * abs(hard - self.ideal_point[0])
        term2 = w2 * abs(soft - self.ideal_point[1])
        
        return -max(term1, term2)  # Negative because we want to minimize
    
    def get_reward(self, old_fitness, new_fitness, agent_id):
        """
        Reward for specific agent (with specific weight vector).
        """
        weight = self.weight_vectors[agent_id]
        
        old_scalar = self.calculate(old_fitness, weight)
        new_scalar = self.calculate(new_fitness, weight)
        
        return new_scalar - old_scalar
```

#### Multi-Agent Architecture

```python
# src/rl/agents/ensemble_agent.py
class MOEADEnsemble:
    def __init__(self, num_agents=5):
        # Create weight vectors uniformly distributed
        self.weight_vectors = self._uniform_weights(num_agents)
        
        # Create one RL agent per weight vector
        self.agents = [
            PPO(policy, env, ...) 
            for _ in range(num_agents)
        ]
        
        self.reward_calculator = DecomposedReward(self.weight_vectors)
    
    def _uniform_weights(self, n):
        """Generate n uniformly distributed weight vectors."""
        return [(i/(n-1), 1 - i/(n-1)) for i in range(n)]
    
    def select_action(self, state, agent_id):
        """Agent-specific action selection."""
        return self.agents[agent_id].predict(state)
    
    def update(self, transitions):
        """Update all agents with their respective rewards."""
        for agent_id, agent in enumerate(self.agents):
            # Filter transitions for this agent
            agent_transitions = [
                t for t in transitions if t['agent_id'] == agent_id
            ]
            
            # Calculate agent-specific rewards
            for t in agent_transitions:
                t['reward'] = self.reward_calculator.get_reward(
                    t['old_fitness'],
                    t['new_fitness'],
                    agent_id
                )
            
            # Update agent
            agent.learn(agent_transitions)
```

---

### Approach 3: Preference Articulation

Let the user specify preferences, and learn a policy that respects them.

#### Mathematical Foundation

**User preference**: "Hard constraints are 100x more important than soft"

**Reward function**:
$$r(\mathbf{s}, \mathbf{s}') = -\alpha \cdot \Delta f_1 - \beta \cdot \Delta f_2$$

Where:
- $\alpha, \beta$ = preference weights (e.g., $\alpha=100, \beta=1$)
- $\Delta f_i = f_i(\mathbf{s}') - f_i(\mathbf{s})$ = change in objective $i$

#### Dynamic Preference Learning

```python
class AdaptivePreferenceReward:
    def __init__(self, initial_alpha=1.0, initial_beta=0.01):
        self.alpha = initial_alpha  # Hard constraint weight
        self.beta = initial_beta    # Soft constraint weight
    
    def calculate(self, old_fitness, new_fitness):
        hard_delta = old_fitness[0] - new_fitness[0]
        soft_delta = old_fitness[1] - new_fitness[1]
        
        return self.alpha * hard_delta + self.beta * soft_delta
    
    def update_preferences(self, current_state):
        """
        Adapt preferences based on solution quality.
        
        Strategy:
        - If infeasible (hard > 0): Focus entirely on hard constraints
        - If feasible (hard = 0): Balance hard and soft
        - If highly feasible: Focus more on soft constraints
        """
        best_hard = current_state.best_fitness[0]
        
        if best_hard > 10:
            # Infeasible: prioritize hard constraints
            self.alpha = 100.0
            self.beta = 0.01
        elif best_hard > 0:
            # Near-feasible: balanced approach
            self.alpha = 10.0
            self.beta = 0.1
        else:
            # Feasible: optimize soft constraints
            self.alpha = 1.0
            self.beta = 1.0
```

---

## Implementation Roadmap

### Phase 1: Hypervolume Reward (2 weeks)
1. **Week 1**:
   - Add `pymoo` dependency for hypervolume calculation
   - Implement `HypervolumeReward` class
   - Add Pareto front extraction utilities
   - Unit tests for HV calculation

2. **Week 2**:
   - Integrate into `src/rl/gym_env/schedule_env.py`
   - Update reward calculation in step() method
   - Add configuration flag: `rl.reward.type: hypervolume`
   - Benchmark: Compare HV vs scalar rewards on validation set

### Phase 2: Decomposition (4 weeks)
1. **Week 3-4**:
   - Implement `DecomposedReward` class
   - Create `MOEADEnsemble` agent wrapper
   - Add agent selection strategy (round-robin or performance-based)

2. **Week 5-6**:
   - Train ensemble of 5 agents with different weights
   - Implement agent coordination (which agent to use when)
   - Evaluate: Does ensemble cover Pareto front better?

### Phase 3: Adaptive Preferences (2 weeks)
1. **Week 7**:
   - Implement `AdaptivePreferenceReward`
   - Add dynamic weight adjustment logic
   - Test: Does adaptation help infeasible → feasible transition?

2. **Week 8**:
   - Compare all three approaches
   - Select best performer for production
   - Document findings

---

## Evaluation Metrics

### 1. Hypervolume Quality
```python
def evaluate_hypervolume(final_population, ref_point):
    """Measure quality of final Pareto front."""
    front = extract_pareto_front(final_population)
    hv = HV(ref_point=ref_point)
    return hv(front)
```

**Target**: 20% improvement over scalar reward baseline

### 2. Pareto Front Coverage
```python
def evaluate_coverage(front, num_divisions=10):
    """Measure how well solutions spread across front."""
    # Divide objective space into grid
    # Count occupied cells
    return occupied_cells / total_cells
```

**Target**: 80% of Pareto front covered

### 3. Convergence Speed
```python
def evaluate_convergence(hv_history):
    """Measure how quickly HV increases."""
    # Find generation where HV reaches 90% of final value
    final_hv = hv_history[-1]
    threshold = 0.9 * final_hv
    
    for gen, hv in enumerate(hv_history):
        if hv >= threshold:
            return gen
    return len(hv_history)
```

**Target**: 30% faster convergence than scalar reward

---

## Expected Benefits

### 1. Better Pareto Approximation
- Current: RL agent biased toward hard constraints (due to weighting)
- Expected: Balanced exploration of hard/soft trade-offs
- Impact: **More diverse solution set**, better for decision makers

### 2. Improved Sample Efficiency
- Current: Agent must learn hard/soft balance from scratch
- Expected: Reward directly encodes Pareto dominance
- Impact: **Faster training**, fewer episodes needed

### 3. Principled Multi-Objective Handling
- Current: Ad-hoc weight selection (trial and error)
- Expected: Theoretically grounded (MOEA/D, HV proven in literature)
- Impact: **Reproducible results**, easier to explain to stakeholders

---

## Risks and Mitigations

### Risk 1: Computational Overhead
**Issue**: Hypervolume calculation is O(n log n) for 2D, can be expensive

**Mitigation**:
- Approximate HV using sampling for large fronts
- Cache Pareto front between generations
- Only recalculate when population changes significantly

### Risk 2: Sparse Reward Signal
**Issue**: HV might not change every action, leading to sparse rewards

**Mitigation**:
- Add shaped reward components (diversity bonus, constraint reduction)
- Use reward shaping: $r_{\text{shaped}} = r_{\text{HV}} + \phi(s') - \phi(s)$
- Potential-based shaping with distance to Pareto front

### Risk 3: Multiple Agents Overhead
**Issue**: Ensemble of 5 agents requires 5× training time

**Mitigation**:
- Share feature extractors across agents (only policy heads differ)
- Use parallel training (train agents simultaneously on separate GPUs)
- Start with 3 agents (hard-focused, balanced, soft-focused)

---

## Configuration Changes

```yaml
# configs/base.yaml (additions)
rl:
  reward:
    type: "hypervolume"  # Options: "scalar", "hypervolume", "decomposed", "adaptive"
    
    hypervolume:
      reference_point: [100, 10000]  # (max_hard, max_soft)
      normalize: true
      
    decomposed:
      num_agents: 5
      weight_vectors: "uniform"  # or "manual" with custom weights
      
    adaptive:
      initial_alpha: 1.0
      initial_beta: 0.01
      adaptation_frequency: 50  # Update preferences every N generations
```

---

## Related Work

### Papers
1. **"Hypervolume-based Multi-objective RL"** (Van Moffaert & Nowé, 2014)
   - First application of HV to RL rewards
   - Proves convergence to Pareto optimal policies

2. **"MOEA/D"** (Zhang & Li, 2007)
   - Decomposition approach for multi-objective optimization
   - Foundation for decomposed reward approach

3. **"Pareto Policy Pool for Constrained RL"** (Alegre et al., 2022)
   - Ensemble of agents with different constraint tolerances
   - Relevant for hard/soft constraint balance

### Open-Source Libraries
- **pymoo**: Multi-objective optimization with HV indicators
- **mo-gymnasium**: Multi-objective RL environments
- **morl-baselines**: Baseline algorithms for MORL

---

## Summary

**Current Problem**: Scalar rewards lose multi-objective information

**Solution**: Use Pareto-aware rewards (hypervolume, decomposition, or adaptive)

**Recommended Start**: Hypervolume indicator (proven, implementable in 2 weeks)

**Expected Impact**: 20-30% improvement in Pareto front quality, better solution diversity

**Next Steps**: Implement Phase 1, benchmark against current scalar approach
