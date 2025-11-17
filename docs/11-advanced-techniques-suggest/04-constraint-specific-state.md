# Constraint-Specific State Representation

**Enhancement**: #3 - Per-Constraint Breakdown in RL State  
**Difficulty**: Low  
**Impact**: High  
**Priority**: 2

---

## Problem Statement

### Current State Representation

```python
# src/rl/state.py (current - aggregated)
state = [
    total_hard_violations,      # Single number
    total_soft_penalty,         # Single number
    avg_fitness,
    diversity_metrics,
    stagnation_counter,
    ...
]
```

**Problem**: RL agent sees **aggregated** constraint violations but doesn't know:
- **Which** constraints are violated
- **How many** violations per constraint
- **Where** violations are concentrated

**Example**:
```
Solution A: {group_overlap: 5, instructor_overlap: 0, ...}
Solution B: {group_overlap: 0, instructor_overlap: 5, ...}

Current state: Both show "total_hard = 5"
Agent can't distinguish them!
```

**Impact**: Agent learns generic repair strategy instead of constraint-targeted repairs.

---

## Solution: Constraint-Specific State Encoding

### Approach 1: Per-Constraint Violation Counts

Expand state to include individual constraint breakdowns.

#### State Vector Design

```python
# src/rl/state.py (enhanced)
class EnhancedStateEncoder:
    def __init__(self, context):
        self.context = context
        
        # Hard constraints (from config)
        self.hard_constraints = [
            "no_group_overlap",
            "no_instructor_overlap",
            "no_room_overlap",
            "instructor_availability",
            "room_availability",
            "group_availability",
            "instructor_qualified",
            "room_type_match",
            "room_capacity"
        ]
        
        # Soft constraints (from config)
        self.soft_constraints = [
            "minimize_gaps",
            "instructor_preference",
            "room_preference",
            "block_clustering",
            "midday_break",
            "consecutive_limit"
        ]
    
    def encode(self, population_state):
        """
        Create 45-dimensional state vector:
        - 9 hard constraint violations (per constraint)
        - 6 soft constraint penalties (per constraint)
        - 10 population-level features
        - 20 action history features
        """
        state = []
        
        # 1. Hard constraint breakdown (9 features)
        hard_breakdown = self._compute_hard_breakdown(population_state.best_individual)
        state.extend([hard_breakdown[c] for c in self.hard_constraints])
        
        # 2. Soft constraint breakdown (6 features)
        soft_breakdown = self._compute_soft_breakdown(population_state.best_individual)
        state.extend([soft_breakdown[c] for c in self.soft_constraints])
        
        # 3. Population-level features (10 features)
        state.extend([
            population_state.avg_hard,
            population_state.avg_soft,
            population_state.best_hard,
            population_state.best_soft,
            population_state.diversity,
            population_state.unique_fitness_ratio,
            population_state.stagnation_counter,
            population_state.generation,
            population_state.improvement_rate,
            population_state.pareto_front_size
        ])
        
        # 4. Action history (20 features)
        state.extend(self._encode_action_history())
        
        return np.array(state, dtype=np.float32)
    
    def _compute_hard_breakdown(self, individual):
        """
        Evaluate each hard constraint separately.
        
        Returns:
            dict: {constraint_name: violation_count}
        """
        decoded = decode_individual(individual, self.context)
        breakdown = {}
        
        for constraint_name in self.hard_constraints:
            constraint_fn = get_hard_constraint(constraint_name)
            violations = constraint_fn(decoded, self.context)
            breakdown[constraint_name] = violations
        
        return breakdown
    
    def _compute_soft_breakdown(self, individual):
        """
        Evaluate each soft constraint separately.
        
        Returns:
            dict: {constraint_name: penalty_value}
        """
        decoded = decode_individual(individual, self.context)
        breakdown = {}
        
        for constraint_name in self.soft_constraints:
            constraint_fn = get_soft_constraint(constraint_name)
            penalty = constraint_fn(decoded, self.context)
            breakdown[constraint_name] = penalty
        
        return breakdown
```

#### Normalization

```python
def normalize_state(self, state):
    """
    Normalize each feature to [0, 1] or [-1, 1] range.
    """
    # Hard constraints: normalize by max observed
    hard_max = np.array([50, 30, 20, 40, 30, 30, 10, 5, 15])  # Empirical maxima
    state[0:9] = np.clip(state[0:9] / hard_max, 0, 1)
    
    # Soft constraints: normalize by typical ranges
    soft_max = np.array([1000, 500, 300, 800, 200, 400])
    state[9:15] = np.clip(state[9:15] / soft_max, 0, 1)
    
    # Population features: already in reasonable ranges
    # (Most are ratios or counters with known bounds)
    
    return state
```

---

### Approach 2: Attention Mechanism

Let the RL agent **learn** which constraints to focus on.

#### Architecture

```python
import torch
import torch.nn as nn

class ConstraintAttentionPolicy(nn.Module):
    def __init__(self, num_hard_constraints=9, num_soft_constraints=6):
        super().__init__()
        
        self.num_hard = num_hard_constraints
        self.num_soft = num_soft_constraints
        
        # Constraint embedding
        self.constraint_embedding = nn.Linear(1, 64)
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            batch_first=True
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(64 + 10, 128),  # 64 from attention + 10 population features
            nn.ReLU(),
            nn.Linear(128, 20)  # 20 actions
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(64 + 10, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, state):
        """
        Args:
            state: [batch, 45] tensor
                - [0:9]: hard constraint violations
                - [9:15]: soft constraint penalties
                - [15:25]: population features
                - [25:45]: action history
        
        Returns:
            action_logits: [batch, 20]
            value: [batch, 1]
        """
        batch_size = state.shape[0]
        
        # Extract constraint violations
        hard_violations = state[:, 0:9].unsqueeze(-1)  # [batch, 9, 1]
        soft_penalties = state[:, 9:15].unsqueeze(-1)  # [batch, 6, 1]
        
        # Embed constraints
        hard_embeds = self.constraint_embedding(hard_violations)  # [batch, 9, 64]
        soft_embeds = self.constraint_embedding(soft_penalties)   # [batch, 6, 64]
        
        # Concatenate all constraints
        constraint_embeds = torch.cat([hard_embeds, soft_embeds], dim=1)  # [batch, 15, 64]
        
        # Self-attention over constraints
        attended, attention_weights = self.attention(
            constraint_embeds,
            constraint_embeds,
            constraint_embeds
        )  # [batch, 15, 64]
        
        # Aggregate attended constraints
        attended_pooled = attended.mean(dim=1)  # [batch, 64]
        
        # Concatenate with population features
        population_features = state[:, 15:25]  # [batch, 10]
        combined = torch.cat([attended_pooled, population_features], dim=1)  # [batch, 74]
        
        # Predict action and value
        action_logits = self.policy_head(combined)
        value = self.value_head(combined)
        
        return action_logits, value, attention_weights
```

**Benefit**: Agent learns to **automatically** focus on most problematic constraints.

---

### Approach 3: Hierarchical Encoding

Group constraints by category (overlap, availability, preference).

```python
class HierarchicalStateEncoder:
    def __init__(self):
        self.constraint_categories = {
            "overlap": ["no_group_overlap", "no_instructor_overlap", "no_room_overlap"],
            "availability": ["instructor_availability", "room_availability", "group_availability"],
            "matching": ["instructor_qualified", "room_type_match", "room_capacity"],
            "quality": ["minimize_gaps", "instructor_preference", "room_preference"],
            "structure": ["block_clustering", "midday_break", "consecutive_limit"]
        }
    
    def encode(self, population_state):
        """
        Create hierarchical state:
        - Level 1: Category-level aggregates (5 features)
        - Level 2: Individual constraint counts (15 features)
        - Level 3: Population-level features (10 features)
        """
        state = []
        
        # Level 1: Category aggregates
        hard_breakdown = self._compute_hard_breakdown(population_state.best_individual)
        soft_breakdown = self._compute_soft_breakdown(population_state.best_individual)
        
        overlap_total = sum(hard_breakdown[c] for c in self.constraint_categories["overlap"])
        availability_total = sum(hard_breakdown[c] for c in self.constraint_categories["availability"])
        matching_total = sum(hard_breakdown[c] for c in self.constraint_categories["matching"])
        quality_total = sum(soft_breakdown[c] for c in self.constraint_categories["quality"])
        structure_total = sum(soft_breakdown[c] for c in self.constraint_categories["structure"])
        
        state.extend([overlap_total, availability_total, matching_total, quality_total, structure_total])
        
        # Level 2: Individual constraints (already computed)
        state.extend([hard_breakdown[c] for category in ["overlap", "availability", "matching"] 
                     for c in self.constraint_categories[category]])
        state.extend([soft_breakdown[c] for category in ["quality", "structure"] 
                     for c in self.constraint_categories[category]])
        
        # Level 3: Population features (same as before)
        state.extend([...])
        
        return np.array(state, dtype=np.float32)
```

---

## Guided Repair Strategy

With constraint-specific state, implement **targeted repair**:

```python
class ConstraintGuidedRepair:
    def __init__(self, repair_heuristics):
        # Map constraints to effective repair heuristics
        self.constraint_to_heuristics = {
            "no_group_overlap": ["kempe_chain", "temporal_shift", "ejection_chain"],
            "no_instructor_overlap": ["instructor_reassign", "temporal_shift"],
            "no_room_overlap": ["room_shuffle", "temporal_shift"],
            "instructor_availability": ["instructor_reassign", "temporal_shift"],
            "room_availability": ["room_shuffle"],
            "minimize_gaps": ["temporal_shift", "block_clustering"],
            # ... etc
        }
    
    def select_repair_action(self, state, hard_breakdown):
        """
        Select repair heuristic based on most violated constraint.
        """
        # Find most violated hard constraint
        most_violated = max(hard_breakdown.items(), key=lambda x: x[1])
        constraint_name, violation_count = most_violated
        
        if violation_count == 0:
            # No violations, focus on soft constraints
            return self._select_optimization_action(state)
        
        # Select heuristic effective for this constraint
        candidate_heuristics = self.constraint_to_heuristics[constraint_name]
        
        # Use RL agent to choose among candidates
        action_probs = self.rl_agent.predict_proba(state)
        
        # Filter to only candidate heuristics
        filtered_probs = {h: action_probs[h] for h in candidate_heuristics}
        
        # Sample action
        selected = max(filtered_probs.items(), key=lambda x: x[1])
        return selected[0]
```

---

## Implementation Roadmap

### Week 1: State Encoder
- [ ] Modify `src/rl/state.py` to include per-constraint breakdown
- [ ] Add `_compute_hard_breakdown()` and `_compute_soft_breakdown()`
- [ ] Implement normalization for new features
- [ ] Update state dimension: 19 → 45

### Week 2: Integration
- [ ] Update `src/rl/gym_env/schedule_env.py` to use new encoder
- [ ] Modify observation space in Gym environment
- [ ] Test: Can we still train PPO/DQN with larger state?
- [ ] Benchmark: Does larger state improve sample efficiency?

### Week 3: Attention Mechanism (Optional)
- [ ] Implement `ConstraintAttentionPolicy` in PyTorch
- [ ] Integrate with Stable-Baselines3 (custom policy)
- [ ] Train and compare: Attention vs MLP policy
- [ ] Visualize attention weights (which constraints agent focuses on)

### Week 4: Guided Repair
- [ ] Implement `ConstraintGuidedRepair` class
- [ ] Define constraint-to-heuristic mappings
- [ ] Test: Does targeted repair converge faster?
- [ ] Measure: Reduction in specific constraint types

---

## Expected Benefits

### 1. Faster Targeted Repair
- **Current**: Agent tries random heuristics until one works
- **Expected**: Agent targets specific constraint types
- **Impact**: 30-40% faster infeasible → feasible transition

### 2. Better Sample Efficiency
- **Current**: Agent must learn "which heuristic helps which constraint" from scratch
- **Expected**: State encoding provides direct feedback
- **Impact**: 20% fewer training episodes needed

### 3. Interpretability
- **Current**: "Agent chose operator X" (why?)
- **Expected**: "Agent chose operator X because constraint Y is most violated"
- **Impact**: Easier debugging and trust

### 4. Attention Visualization
```python
# After training with attention mechanism
attention_weights = model.get_attention_weights(state)

print("Agent focused on:")
for constraint, weight in zip(constraint_names, attention_weights):
    if weight > 0.1:
        print(f"  {constraint}: {weight:.2f}")

# Output:
# Agent focused on:
#   no_group_overlap: 0.45
#   instructor_availability: 0.32
#   minimize_gaps: 0.23
```

---

## Evaluation Metrics

### 1. Constraint-Specific Convergence
```python
def evaluate_per_constraint_convergence(run_history):
    """
    Measure how quickly each constraint type is satisfied.
    """
    for constraint in hard_constraints:
        generations_to_zero = find_first_zero(run_history, constraint)
        print(f"{constraint}: {generations_to_zero} generations")
```

**Target**: 20% faster convergence on hardest constraint types

### 2. Action-Constraint Correlation
```python
def measure_action_constraint_correlation(actions_taken, constraint_reductions):
    """
    Measure if agent uses correct heuristics for each constraint.
    """
    # Expected: kempe_chain → reduces overlaps
    # Expected: instructor_reassign → reduces availability violations
    correlation_matrix = compute_correlation(actions_taken, constraint_reductions)
    return correlation_matrix
```

**Target**: 0.7+ correlation between effective actions and constraint reductions

---

## Configuration

```yaml
# configs/base.yaml
rl:
  state_encoder:
    type: "constraint_specific"  # vs "aggregated"
    
    constraint_specific:
      include_hard_breakdown: true
      include_soft_breakdown: true
      include_action_history: true
      normalize: true
      
    attention:
      enabled: false  # Experimental
      num_heads: 4
      embed_dim: 64
```

---

## Related Work

### Papers
1. **"Attention-Based RL"** (Mnih et al., 2014)
   - Visual attention for RL agents
   - Learns what to focus on in complex states

2. **"Graph Neural Networks for Combinatorial Optimization"** (Khalil et al., 2017)
   - Per-node feature representations
   - Attention over problem structure

3. **"Constraint-Aware RL"** (Desai et al., 2020)
   - State encoding for constrained MDPs
   - Separate constraint violation signals

---

## Summary

**Problem**: Aggregated state loses constraint-specific information

**Solution**: Per-constraint violation counts in state vector (45 dimensions)

**Optional**: Attention mechanism to learn constraint importance

**Expected Impact**: 30-40% faster targeted repair, 20% better sample efficiency

**Difficulty**: Low (mostly engineering, minimal algorithmic changes)

**Recommended Start**: Implement per-constraint breakdown first, add attention later

**Next Steps**: Week 1 state encoder → Week 2 integration → Week 3 evaluation
