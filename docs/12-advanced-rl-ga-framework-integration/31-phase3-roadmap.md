# Phase 3 Roadmap: Advanced RL-GA Enhancements

**Date**: November 17, 2025  
**Status**: Planning Phase  
**Purpose**: Detailed implementation roadmap for Phase 3+ enhancements

---

## Overview

Phase 3 builds upon completed Phase 1 (Heuristic Toolbox) and Phase 2 (RL Integration) to implement advanced optimization techniques. This roadmap provides actionable tasks, timelines, and success criteria.

---

## Prerequisites

### Must Be Complete Before Starting Phase 3

✅ **Phase 1: Heuristic Toolbox**
- 19 operators across 5 categories
- Registry-based architecture
- Full test coverage

🔄 **Phase 2: RL Integration**
- Gymnasium environment
- Training infrastructure
- Deployment system
- **Pending**: Curriculum training, checkpoint selection, production deployment

**Action**: Complete Phase 2 execution tasks before starting Phase 3

---

## Phase 3 Structure

### Three Tiers of Implementation

**Tier 1: Quick Wins** (Months 1-2, ~30% improvement)
- Low to medium difficulty
- High ROI
- Proven techniques
- Immediate production value

**Tier 2: Advanced** (Months 3-7, ~40% cumulative improvement)
- Medium to high difficulty
- Moderate ROI
- More complex integration
- Requires Tier 1 foundation

**Tier 3: Research** (Months 8-12+, research contributions)
- Very high difficulty
- Research-level uncertainty
- Cutting-edge techniques
- Optional (research goals only)

---

## Tier 1: Quick Wins (Priority)

### Enhancement #2: Constraint-Specific State Encoding

**Priority**: HIGH (Implement First)  
**Difficulty**: Low  
**Timeline**: 1 week  
**Expected Impact**: 30-40% faster targeted repair

#### Week 1: Implementation

**Day 1-2: State Encoder Enhancement**
```python
# File: src/rl/gym_env/state_encoder.py

def _encode_constraint_breakdown(self, hard_penalties, soft_penalties):
    """
    Add per-constraint violation counts to state.
    
    Current state: 21 dimensions
    Enhanced state: 21 + 18 = 39 dimensions
    
    New features:
    - 14 hard constraint violations (one per constraint)
    - 4 soft constraint penalties (one per constraint)
    """
    constraint_features = []
    
    # Hard constraints (14 total)
    for constraint_name in self.hard_constraint_names:
        violation_count = hard_penalties.get(constraint_name, 0)
        constraint_features.append(violation_count)
    
    # Soft constraints (4 total)
    for constraint_name in self.soft_constraint_names:
        penalty = soft_penalties.get(constraint_name, 0)
        constraint_features.append(penalty)
    
    return np.array(constraint_features, dtype=np.float32)
```

**Day 3: Environment Integration**
```python
# File: src/rl/gym_env/schedule_env.py

# Update observation_space
self.observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(39,),  # 21 base + 18 constraint breakdown
    dtype=np.float32
)
```

**Day 4: Testing**
```python
# File: test/rl/test_constraint_state.py

def test_constraint_breakdown_dimensions():
    """Verify state encoder returns 39 dimensions."""
    encoder = StateEncoder(context)
    state = encoder.encode(population, constraints)
    assert state.shape == (39,), f"Expected 39, got {state.shape}"

def test_constraint_isolation():
    """Verify each constraint violation tracked separately."""
    # Create population with known violations
    # Verify state reflects correct constraint breakdown
```

**Day 5: Retraining**
```bash
# Retrain RL agent with enhanced state
python src/rl/training/train_script.py \
    --timesteps 50000 \
    --agent ppo \
    --device cuda \
    --config config-train/med.yaml \
    --experiment-name constraint_specific_state
```

#### Validation
- [ ] State dimensions correct (39)
- [ ] Constraint breakdown accurate
- [ ] Training converges
- [ ] Performance improves over baseline

#### Expected Outcome
- Agent learns to target specific constraints
- Faster convergence on hard constraint violations
- More interpretable learned policy

---

### Enhancement #1: Multi-Objective Reward Shaping

**Priority**: HIGH (Implement Second)  
**Difficulty**: Medium  
**Timeline**: 2 weeks  
**Expected Impact**: 20-30% better solution diversity

#### Week 1: Hypervolume Implementation

**Day 1-3: Hypervolume Calculator**
```python
# File: src/rl/gym_env/hypervolume.py

import numpy as np
from pygmo import hypervolume  # Fast C++ implementation

class HypervolumeCalculator:
    """
    Compute hypervolume indicator for Pareto front quality.
    
    Hypervolume = volume of space dominated by Pareto front.
    Higher hypervolume = better Pareto front quality.
    """
    
    def __init__(self, reference_point: np.ndarray):
        """
        Args:
            reference_point: Worst acceptable point (e.g., [1000, 10000])
                            for (hard_violations, soft_penalty)
        """
        self.reference_point = reference_point
    
    def compute(self, pareto_front: np.ndarray) -> float:
        """
        Compute hypervolume for given Pareto front.
        
        Args:
            pareto_front: (N, 2) array of (hard, soft) fitness values
        
        Returns:
            Hypervolume indicator value
        """
        if len(pareto_front) == 0:
            return 0.0
        
        # pygmo expects minimization objectives
        hv = hypervolume(pareto_front)
        return hv.compute(self.reference_point)
```

**Day 4-5: Reward Integration**
```python
# File: src/rl/gym_env/reward_calculator.py

def compute_hypervolume_reward(
    self,
    old_front: np.ndarray,
    new_front: np.ndarray
) -> float:
    """
    Reward = improvement in hypervolume indicator.
    
    Positive reward: Pareto front improved
    Negative reward: Pareto front degraded
    """
    old_hv = self.hv_calculator.compute(old_front)
    new_hv = self.hv_calculator.compute(new_front)
    
    # Normalize to [-1, 1]
    delta_hv = new_hv - old_hv
    normalized_reward = np.tanh(delta_hv / self.hv_scale)
    
    return normalized_reward
```

#### Week 2: Configuration & Retraining

**Day 1: Config Updates**
```yaml
# File: configs/base.yaml

rl:
  reward:
    type: hypervolume  # Options: scalar, hypervolume, decomposition
    
    hypervolume:
      reference_point: [1000, 10000]  # (hard, soft)
      scale: 100.0  # Normalization scale
      
    # Fallback to scalar if hypervolume computation fails
    fallback_to_scalar: true
```

**Day 2-5: Retraining & Validation**
```bash
# Train with hypervolume reward
python src/rl/training/train_script.py \
    --timesteps 100000 \
    --agent ppo \
    --device cuda \
    --config config-train/prod.yaml \
    --experiment-name hypervolume_reward
```

**Validation**:
- [ ] Hypervolume computation correct
- [ ] Reward correlates with Pareto front quality
- [ ] Agent learns to improve diversity
- [ ] Final Pareto front better than baseline

#### Expected Outcome
- Better spread of solutions along Pareto front
- Improved coverage of objective space
- Higher hypervolume indicator values

---

### Enhancement #6: Archive-Based Diversity

**Priority**: MEDIUM (Implement Third)  
**Difficulty**: Medium  
**Timeline**: 3 weeks  
**Expected Impact**: 30% more diverse solution portfolios

#### Week 1: Archive Implementation

**Behavioral Characterization**:
```python
# File: src/diversity/behavioral_features.py

def extract_behavioral_features(individual, context) -> np.ndarray:
    """
    Extract phenotypic features that characterize solution behavior.
    
    Features:
    1. Time distribution (sessions per day)
    2. Room utilization (sessions per room)
    3. Instructor workload (sessions per instructor)
    4. Course clustering (sessions per course)
    5. Constraint profile (violation pattern)
    
    Returns:
        Feature vector (dimension ~20-30)
    """
    features = []
    
    # Time distribution (7 features: one per day)
    time_dist = compute_sessions_per_day(individual, context)
    features.extend(time_dist)
    
    # Room utilization (normalized)
    room_util = compute_room_utilization(individual, context)
    features.append(room_util.mean())
    features.append(room_util.std())
    
    # ... more features
    
    return np.array(features)
```

**Archive Data Structure**:
```python
# File: src/diversity/archive.py

class BehavioralArchive:
    """
    Maintain archive of elite solutions for novelty search.
    
    Archive stores solutions with diverse behavioral characteristics.
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.solutions = []  # List of (individual, features, fitness)
    
    def add(self, individual, features, fitness):
        """Add solution if sufficiently novel or high quality."""
        if len(self.solutions) < self.max_size:
            self.solutions.append((individual, features, fitness))
        else:
            # Replace least novel or worst fitness solution
            self._replace_solution(individual, features, fitness)
    
    def compute_novelty(self, features) -> float:
        """
        Compute novelty as average distance to k-nearest neighbors.
        
        Higher novelty = more different from archive solutions
        """
        if len(self.solutions) == 0:
            return float('inf')
        
        distances = [
            np.linalg.norm(features - archived_features)
            for _, archived_features, _ in self.solutions
        ]
        
        k = min(15, len(distances))
        k_nearest = sorted(distances)[:k]
        return np.mean(k_nearest)
```

#### Week 2: GA Integration

**Novelty-Fitness Trade-off**:
```python
# File: src/core/ga_scheduler.py

def evaluate_with_archive(self, population):
    """
    Evaluate fitness with novelty bonus from archive.
    
    Final fitness = (1 - α) * quality + α * novelty
    where α controls exploration/exploitation trade-off
    """
    for individual in population:
        # Standard fitness
        quality = self._evaluate_fitness(individual)
        
        # Behavioral features
        features = extract_behavioral_features(individual, self.context)
        
        # Novelty from archive
        novelty = self.archive.compute_novelty(features)
        
        # Combined fitness
        alpha = self.config.archive.novelty_weight  # 0.2 typical
        combined_fitness = (1 - alpha) * quality + alpha * novelty
        
        individual.fitness.values = (combined_fitness,)
        
        # Update archive
        if quality < self.archive.quality_threshold:
            self.archive.add(individual, features, quality)
```

#### Week 3: Testing & Validation

**Metrics**:
- Behavioral diversity (average pairwise distance in feature space)
- Archive coverage (spread of solutions)
- Quality-diversity trade-off

**Validation**:
- [ ] Archive maintains diverse solutions
- [ ] Novelty metric correctly identifies unique solutions
- [ ] Final population more diverse than baseline
- [ ] Quality not significantly degraded

---

## Tier 2: Advanced Techniques (Months 3-7)

### Enhancement #4: Specialist RL Agents

**Timeline**: 4 weeks  
**Dependencies**: #2 (constraint-specific state)

#### Implementation Plan

**Week 1-2: Train Repair Agent**
- Focus on infeasible solutions (hard_violations > 0)
- Reward: hard constraint reduction
- Training data: filtered to infeasible problems only

**Week 2-3: Train Optimizer Agent**
- Focus on feasible solutions (hard_violations == 0)
- Reward: soft constraint reduction
- Training data: filtered to feasible problems only

**Week 4: Coordinator Implementation**
- Switching logic: Use repair agent if hard_violations > 0, else optimizer
- Smooth transitions
- Performance monitoring

---

### Enhancement #3: Adaptive Probabilities

**Timeline**: 2 weeks  
**Dependencies**: #1, #2

#### Implementation Plan

**Week 1: Action Space Extension**
- Add actions: increase_cxpb, decrease_cxpb, increase_mutpb, decrease_mutpb
- Update ActionMapper
- Define probability change delta (e.g., ±0.05)

**Week 2: Retraining & Validation**
- Train agent with probability control actions
- Monitor probability adaptation patterns
- Validate convergence improvement

---

### Enhancement #5: Memetic RL

**Timeline**: 4 weeks  
**Dependencies**: #2

#### Implementation Plan

**Week 1-2: Local Search Budget Control**
- Add actions: increase_igls_budget, decrease_igls_budget
- Track computational cost vs improvement
- Implement adaptive budget allocation

**Week 3: Integration with IGLS**
- Modify IGLS to accept dynamic iteration budget
- Update repair system to use RL-controlled budget

**Week 4: Efficiency Analysis**
- Measure computational savings
- Validate quality maintained or improved

---

### Enhancement #7: Hierarchical RL

**Timeline**: 4 weeks  
**Dependencies**: #1, #2

#### Implementation Plan

**Week 1: High-Level Policy**
- Train policy to select heuristic category (5 actions)
- State: aggregated constraint profile
- Reward: improvement after category application

**Week 2-3: Low-Level Policies**
- Train 5 separate policies (one per category)
- Each selects specific operator within category
- State: detailed constraint breakdown
- Reward: immediate improvement

**Week 4: Hierarchical Coordination**
- Implement options framework
- Test temporal abstraction
- Validate hierarchical decision making

---

## Tier 3: Research Extensions (Optional)

### Enhancement #8: Multi-Agent RL

**Timeline**: 6 weeks  
**Research Level**: High

#### Challenges
- Non-stationarity in multi-agent learning
- Credit assignment across agents
- Computational overhead (K separate models)

#### Success Criteria
- Demonstrate coordination benefits
- Show improvement on complex problems
- Publish research findings

---

### Enhancement #9: Transfer Learning

**Timeline**: 8 weeks  
**Research Level**: High

#### Key Steps
1. Generate 1000+ synthetic problems
2. Pre-train agent (500K steps)
3. Fine-tune on real problems (100K steps)
4. Measure transfer effectiveness

#### Success Criteria
- 50% reduction in fine-tuning time
- Comparable or better final performance
- Demonstrate generalization

---

### Enhancement #10: Online Learning

**Timeline**: 6 weeks  
**Research Level**: High

#### Key Steps
1. Implement experience collection from production
2. Add safe policy update mechanism
3. Deploy online adaptation pipeline
4. Monitor long-term performance

#### Success Criteria
- Adapt to distribution shifts
- Maintain policy stability
- No catastrophic forgetting

---

## Implementation Schedule

### Immediate (November-December 2025)
**Goal**: Complete Phase 2 execution
- [ ] Run curriculum training (24-48 hours)
- [ ] Select and promote best checkpoint
- [ ] Run baseline comparisons
- [ ] Document results

### Q1 2026: Tier 1 Implementation
**Goal**: 30% improvement through quick wins
- Month 1: #2 (constraint-specific state) + #1 (multi-objective reward)
- Month 2: #6 (archive-based diversity) + validation

**Go/No-Go Decision**: End of Q1 2026
- If improvement ≥20%: Proceed to Tier 2
- If improvement <10%: Reconsider strategy

### Q2-Q3 2026: Tier 2 Implementation
**Goal**: 40% cumulative improvement
- Month 3: #4 (specialist agents)
- Month 4: #3 (adaptive probabilities)
- Month 5-6: #5 (memetic RL)
- Month 7: #7 (hierarchical RL)

**Go/No-Go Decision**: End of Q3 2026
- If cumulative ≥40%: Consider Tier 3
- If practical goals met: Enter maintenance mode

### Q4 2026+: Tier 3 (Optional)
**Goal**: Research contributions
- Only if dedicated research team available
- Sufficient compute resources
- Strong research motivation

---

## Success Metrics

### Tier 1 Success
- ✅ 20%+ improvement in key metrics
- ✅ No performance regression
- ✅ Computational overhead <10%
- ✅ Production stability maintained

### Tier 2 Success
- ✅ 40%+ cumulative improvement
- ✅ Positive ROI on development time
- ✅ System remains maintainable
- ✅ Documentation complete

### Tier 3 Success
- ✅ Publishable research results
- ✅ Novel contributions demonstrated
- ✅ Generalization proven

---

## Risk Mitigation

### Risk 1: Tier 1 Doesn't Improve Performance
**Mitigation**: Each enhancement has independent value; can keep subset
**Contingency**: Revert to Phase 2 baseline if all fail

### Risk 2: Training Instability
**Mitigation**: Start with small timesteps; gradually increase
**Contingency**: Tune hyperparameters; use more stable algorithms (PPO > DQN)

### Risk 3: Computational Budget Exceeded
**Mitigation**: Use GPU acceleration; parallelize where possible
**Contingency**: Reduce problem sizes; use shorter training runs

### Risk 4: Research Techniques Fail (Tier 3)
**Mitigation**: Treat as research exploration, not production requirement
**Contingency**: Focus on proven Tier 1-2 techniques

---

## Resource Requirements

### Tier 1 (2 months)
- **Developer**: 1 FTE
- **GPU**: 150 hours
- **Storage**: 20 GB
- **Budget**: ~$500 (GPU cloud costs)

### Tier 2 (5 months)
- **Developer**: 1-2 FTE
- **GPU**: 500 hours
- **Storage**: 50 GB
- **Budget**: ~$1500

### Tier 3 (4+ months)
- **Developer**: 2-3 FTE
- **GPU**: 1000+ hours
- **Storage**: 100+ GB
- **Budget**: ~$3000+

---

## Next Immediate Actions

1. **Complete Phase 2 execution tasks** (See 03-integration-status.md)

2. **Review and approve Phase 3 plan** with stakeholders

3. **Set up Tier 1 development environment**:
   ```bash
   # Create experiment tracking
   mkdir -p experiments/tier1
   
   # Setup TensorBoard
   mkdir -p logs/tensorboard/tier1
   
   # Create config variants
   cp configs/base.yaml configs/tier1_base.yaml
   ```

4. **Begin Tier 1 Enhancement #2** (constraint-specific state)

---

**Last Updated**: November 17, 2025  
**Status**: Ready for Phase 3 implementation after Phase 2 execution completes
