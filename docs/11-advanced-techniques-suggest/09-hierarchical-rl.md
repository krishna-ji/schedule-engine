# Hierarchical RL: Two-Level Selection

**Enhancement**: #9 - Category → Heuristic Selection  
**Difficulty**: High  
**Impact**: High  
**Priority**: 7

---

## Problem Statement

**Current**: RL directly selects from 19 heuristics (flat action space).

**Problem**: Large action space is hard to learn. Many heuristics are similar within categories.

---

## Solution: Hierarchical Decision Making

### Level 1: Category Selection (High-Level Policy)
```
State: Population-level features
Action: Select category {Construction, Perturbation, Improvement, Diversity, Meta}
```

### Level 2: Heuristic Selection (Low-Level Policy)
```
State: Category + individual-level features
Action: Select specific heuristic within category
```

### Architecture

```python
class HierarchicalPolicy:
    def __init__(self):
        # High-level: Choose category
        self.high_level_policy = PPO(
            action_space=5,  # 5 categories
            ...
        )
        
        # Low-level: Choose heuristic within category
        self.low_level_policies = {
            "construction": PPO(action_space=3, ...),
            "perturbation": PPO(action_space=5, ...),
            "improvement": PPO(action_space=3, ...),
            "diversity": PPO(action_space=4, ...),
            "meta": PPO(action_space=4, ...)
        }
    
    def select_action(self, state):
        # Step 1: High-level selects category
        category = self.high_level_policy.predict(state)
        
        # Step 2: Low-level selects heuristic
        category_state = encode_category_state(state, category)
        heuristic = self.low_level_policies[category].predict(category_state)
        
        return category, heuristic
```

### Reward Shaping

```python
def hierarchical_reward(old_fitness, new_fitness, category, heuristic):
    """
    Reward both levels of the hierarchy.
    """
    improvement = old_fitness - new_fitness
    
    # High-level reward: Did category choice help?
    high_level_reward = improvement
    
    # Low-level reward: Did specific heuristic help? (intrinsic motivation)
    expected_improvement = estimate_category_potential(category, old_fitness)
    low_level_reward = improvement - expected_improvement
    
    return high_level_reward, low_level_reward
```

---

## Mathematical Foundation

**Options Framework** (Sutton et al., 1999):
- **Options** = temporally extended actions (categories in our case)
- **Policy over options**: $\\pi_{high}(o | s)$
- **Intra-option policies**: $\\pi_{low}^o(a | s)$

**Hierarchical Value Function**:
$$Q(s, o) = \\mathbb{E}\\left[r + \\gamma V(s') | s, o\\right]$$

where $o$ is option (category), and value depends on low-level execution.

---

## Expected Benefits

### 1. Faster Learning
- **Flat**: Learn 19-dimensional action space
- **Hierarchical**: Learn 5-dim + max(5-dim) = 10-dim effective space
- **Impact**: 30-40% faster convergence

### 2. Better Generalization
- **Shared knowledge** within categories
- **Transfer** between similar heuristics
- **Impact**: 20% better on unseen problems

### 3. Interpretability
- "Agent chose Improvement → Kempe Chain" (clear reasoning)
- Easier to debug policy decisions

---

## Implementation Roadmap

1. **Week 1-2**: Implement high-level category policy
2. **Week 3-4**: Implement low-level heuristic policies (5 separate)
3. **Week 5-6**: Joint training with hierarchical rewards
4. **Week 7-8**: Benchmark against flat policy

**Difficulty**: High | **Priority**: Medium (after basic enhancements)
