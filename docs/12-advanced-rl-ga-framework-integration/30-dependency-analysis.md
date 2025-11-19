# Dependency Analysis: Enhancement Prerequisites

**Date**: November 17, 2025  
**Purpose**: Analyze dependencies between enhancements from `docs/11-advanced-techniques-suggest/`

---

## Overview

This document analyzes the 10 advanced techniques proposed in `docs/11-advanced-techniques-suggest/` and identifies their dependencies, prerequisites, and implementation order.

---

## Dependency Graph

```
Phase 1 & 2 (Foundation) 
├── Phase 1: Heuristic Toolbox (19 operators)
└── Phase 2: RL Integration (Gym env, training, deployment)
                    │
                    ├─────────────────────────────────────┐
                    │                                     │
                    ▼                                     ▼
        Tier 1 (Quick Wins)              Tier 1 (Strategic)
        ├── #2: Constraint-specific      ├── #1: Multi-objective reward
        │   state (1 week)               │   (2 weeks)
        │   └─ Dependencies: None        │   └─ Dependencies: None
        │                                │
        ▼                                ▼
        #4: Specialist agents            #6: Archive-based diversity
        (4 weeks)                        (3 weeks)
        └─ Dependencies: #2              └─ Dependencies: #1
                    │                                     │
                    └──────────────┬──────────────────────┘
                                   │
                                   ▼
                        Tier 2 (Advanced)
                        ├── #3: Adaptive probabilities (2 weeks)
                        │   └─ Dependencies: #1, #2
                        │
                        ├── #5: Memetic RL (4 weeks)
                        │   └─ Dependencies: #2
                        │
                        └── #7: Hierarchical RL (4 weeks)
                            └─ Dependencies: #1, #2
                                   │
                                   ▼
                        Tier 3 (Research)
                        ├── #8: Multi-agent RL (6 weeks)
                        │   └─ Dependencies: All previous
                        │
                        ├── #9: Transfer learning (8 weeks)
                        │   └─ Dependencies: All previous
                        │
                        └── #10: Online learning (6 weeks)
                            └─ Dependencies: All previous
```

---

## Enhancement Details

### Phase 1 & 2: Foundation 

**Status**: Phase 1 complete, Phase 2 code complete (execution pending)

**Prerequisites for All Enhancements**:
-  Phase 1: 19 heuristic operators in registry
-  Phase 2: RL Gym environment with 21-dimensional state space
-  Phase 2: Training infrastructure (curriculum, callbacks, checkpoints)
-  Phase 2: Deployment infrastructure (loader, inference, hybrid controller)
-  Phase 2: Trained model from curriculum learning

**Files**:
```
src/heuristics/           # Phase 1
src/rl/gym_env/          # Phase 2.1
src/rl/training/         # Phase 2.2
src/rl/deployment/       # Phase 2.3
src/core/ga_scheduler.py # Phase 2.4 (RL integration)
```

---

### Tier 1: Quick Wins (Months 1-2)

#### #2: Constraint-Specific State Encoding

**Enhancement**: Add per-constraint breakdown to state space  
**Priority**: 2 (High)  
**Difficulty**: Low  
**Timeline**: 1 week  
**Dependencies**: None (only Phase 1&2 foundation)

**What Changes**:
- Expand `StateEncoder` from 21 to 35+ dimensions
- Add per-constraint violation counts (14 hard + 4 soft = 18 features)
- Update observation space in `ScheduleEnv`
- Retrain RL agent with enhanced state

**Expected Impact**: 30-40% faster targeted repair

**Files to Modify**:
```
src/rl/gym_env/state_encoder.py  # Add constraint breakdown
src/rl/gym_env/schedule_env.py   # Update observation_space
test/rl/test_diversity_metrics.py # Update tests for new dimensions
```

**Validation**:
- Unit tests for state encoder dimensions
- Training convergence with enhanced state
- Baseline comparison (old state vs new state)

---

#### #1: Multi-Objective Reward

**Enhancement**: Replace scalar reward with hypervolume indicator  
**Priority**: 1 (High)  
**Difficulty**: Medium  
**Timeline**: 2 weeks  
**Dependencies**: None (only Phase 1&2 foundation)

**What Changes**:
- Implement hypervolume calculator in `RewardCalculator`
- Add reference point configuration
- Support Pareto-aware rewards
- Retrain RL agent with MO rewards

**Expected Impact**: 20-30% better solution diversity

**Files to Modify**:
```
src/rl/gym_env/reward_calculator.py  # Add hypervolume calculation
configs/base.yaml                     # Add reference_point config
```

**New Dependencies**:
```python
# May need to add to pyproject.toml
pygmo  # For efficient hypervolume calculation
```

**Validation**:
- Hypervolume monotonicity tests
- Pareto front quality comparison
- Diversity metrics (spacing, spread)

---

### Tier 1: Strategic Enhancements

#### #4: Specialist Agents

**Enhancement**: Train separate RL agents for repair vs optimization  
**Priority**: 4  
**Difficulty**: Medium  
**Timeline**: 4 weeks  
**Dependencies**: #2 (constraint-specific state helps agent specialization)

**What Changes**:
- Train 2 specialist agents:
  - Repair agent (optimized for hard constraint reduction)
  - Optimizer agent (optimized for soft constraint reduction)
- Implement coordinator to switch between agents
- Add agent selection logic based on problem state

**Expected Impact**: 20-30% improvement in task-specific metrics

**Files to Create**:
```
src/rl/agents/specialist_agents.py  # Repair & optimizer agents
src/rl/agents/coordinator.py        # Agent selection logic
scripts/train_specialists.py         # Separate training scripts
```

**Training Strategy**:
- Train repair agent on infeasible solutions only
- Train optimizer agent on feasible solutions only
- Coordinator uses hard_violations > 0 as switching criterion

**Validation**:
- Repair agent performance on infeasible problems
- Optimizer agent performance on feasible problems
- Coordinator switching accuracy

---

#### #6: Archive-Based Diversity

**Enhancement**: Maintain archive of elite solutions for novelty search  
**Priority**: 6  
**Difficulty**: Medium  
**Timeline**: 3 weeks  
**Dependencies**: #1 (multi-objective understanding helps archive selection)

**What Changes**:
- Implement behavioral characterization (phenotypic features)
- Add archive data structure with bounded size
- Integrate novelty metric into fitness evaluation
- Update NSGA-II to consider archive diversity

**Expected Impact**: 30% more diverse solution portfolios

**Files to Create**:
```
src/diversity/archive.py             # Archive management
src/diversity/behavioral_features.py # Phenotypic characterization
src/diversity/novelty_metric.py      # Novelty calculation
```

**Files to Modify**:
```
src/core/ga_scheduler.py  # Integrate archive into selection
```

**Validation**:
- Archive coverage tests
- Behavioral diversity metrics
- Solution quality vs diversity trade-off

---

### Tier 2: Advanced Techniques (Months 3-7)

#### #3: Adaptive Probabilities

**Enhancement**: RL controls crossover/mutation probabilities dynamically  
**Priority**: 3  
**Difficulty**: Medium  
**Timeline**: 2 weeks  
**Dependencies**: #1 (MO rewards), #2 (constraint-specific state)

**Why Dependencies**:
- #1: Need MO rewards to balance exploration/exploitation
- #2: Need constraint state to guide probability adaptation

**What Changes**:
- Expand action space from 20 to 22 (add increase/decrease cxpb/mutpb)
- Add probability control policy
- Track probability effectiveness
- Update GA to use dynamic probabilities

**Expected Impact**: 10-15% faster convergence

**Files to Modify**:
```
src/rl/gym_env/action_mapper.py  # Add probability control actions
src/core/ga_scheduler.py         # Use dynamic probabilities
```

**Validation**:
- Probability adaptation patterns
- Convergence speed comparison
- Sensitivity analysis

---

#### #5: Memetic RL (Adaptive Local Search)

**Enhancement**: RL controls local search budget (IGLS iterations)  
**Priority**: 5  
**Difficulty**: High  
**Timeline**: 4 weeks  
**Dependencies**: #2 (constraint-specific state guides local search decisions)

**What Changes**:
- Expand action space to include IGLS intensity control
- Add local search budget control policy
- Track computational cost vs improvement
- Implement adaptive budget allocation

**Expected Impact**: 50% computational savings + 15% quality improvement

**Files to Create**:
```
src/rl/local_search_controller.py  # LS budget control
```

**Files to Modify**:
```
src/ga/operators/repair_igls.py  # Accept dynamic iteration budget
src/rl/gym_env/action_mapper.py  # Add LS control actions
```

**Validation**:
- Computational efficiency metrics
- Quality vs time trade-off
- Budget allocation patterns

---

#### #7: Hierarchical RL

**Enhancement**: Two-level RL (high-level: category, low-level: operator)  
**Priority**: 7  
**Difficulty**: High  
**Timeline**: 4 weeks  
**Dependencies**: #1 (MO rewards), #2 (constraint-specific state)

**Why Dependencies**:
- #1: High-level policy needs MO understanding
- #2: Low-level policies need constraint-specific guidance

**What Changes**:
- Implement high-level policy (5 actions: select category)
- Implement 5 low-level policies (3-5 actions each: select operator)
- Add hierarchical action space
- Train using options framework

**Expected Impact**: 15-25% better exploration

**Files to Create**:
```
src/rl/hierarchical/high_level_policy.py  # Category selection
src/rl/hierarchical/low_level_policies.py # Operator selection
src/rl/hierarchical/options_framework.py  # Temporal abstraction
```

**Training Complexity**: 2x training time (need to train 6 separate policies)

**Validation**:
- High-level policy interpretability
- Low-level policy effectiveness
- Hierarchical coordination quality

---

### Tier 3: Research Extensions (Months 8-12+)

#### #8: Multi-Agent RL

**Enhancement**: Rank-specific specialist agents with coordination  
**Priority**: 8  
**Difficulty**: Very High  
**Timeline**: 6 weeks  
**Dependencies**: All previous (requires mature understanding of state/action/reward)

**What Changes**:
- Train K agents (one per Pareto rank)
- Implement meta-controller for agent routing
- Add inter-agent communication (optional)
- Coordinate agents across population

**Expected Impact**: 10-20% on hard problems

**Research Challenges**:
- Credit assignment across agents
- Non-stationarity in multi-agent learning
- Computational overhead (K models)

**Files to Create**:
```
src/rl/multi_agent/rank_specialists.py  # K specialist agents
src/rl/multi_agent/meta_controller.py   # Agent routing
src/rl/multi_agent/communication.py     # Inter-agent messaging (optional)
```

**Validation**:
- Individual agent performance
- Meta-controller accuracy
- Coordination quality

---

#### #9: Transfer Learning

**Enhancement**: Pre-train on synthetic problems, fine-tune on real  
**Priority**: 9  
**Difficulty**: Very High  
**Timeline**: 8 weeks  
**Dependencies**: All previous (need robust base system)

**What Changes**:
- Generate 1000+ synthetic scheduling problems
- Pre-train agent on synthetic dataset (500K steps)
- Fine-tune on real problems (100K steps)
- Compare against training from scratch

**Expected Impact**: 50% reduction in fine-tuning time

**Research Challenges**:
- Synthetic problem realism
- Transfer gap between synthetic and real
- Catastrophic forgetting during fine-tuning

**Files to Create**:
```
scripts/generate_synthetic_problems.py  # Problem generation
src/rl/transfer/domain_randomization.py # Synthetic problem sampling
src/rl/transfer/fine_tuner.py           # Fine-tuning logic
```

**Validation**:
- Pre-training effectiveness
- Transfer learning speedup
- Final performance vs from-scratch

---

#### #10: Online Learning

**Enhancement**: Continual learning from production runs  
**Priority**: 10  
**Difficulty**: High  
**Timeline**: 6 weeks  
**Dependencies**: All previous (requires production deployment experience)

**What Changes**:
- Implement experience replay from production logs
- Add safe policy update mechanism
- Deploy online adaptation pipeline
- Monitor long-term performance

**Expected Impact**: Adapt to problem distribution shifts

**Research Challenges**:
- Catastrophic forgetting
- Policy stability
- Safety guarantees

**Files to Create**:
```
src/rl/online/experience_collector.py  # Production logging
src/rl/online/safe_updater.py          # Conservative updates
src/rl/online/drift_detector.py        # Performance monitoring
```

**Validation**:
- Adaptation speed
- Long-term stability
- Rollback mechanism correctness

---

## Implementation Strategy

### Sequential Approach (Recommended)

**Month 1-2**: Tier 1 Quick Wins
1. Week 1: Implement #2 (constraint-specific state)
2. Week 2: Retrain and validate #2
3. Week 3-4: Implement #1 (multi-objective reward)
4. Week 5-6: Retrain and validate #1
5. Week 7-8: Implement #6 (archive diversity)

**Go/No-Go Decision Point**: After Tier 1
- If improvements ≥20%: Continue to Tier 2
- If improvements <10%: Consolidate and reconsider

**Month 3-4**: Tier 1 Strategic
1. Week 9-12: Implement #4 (specialist agents)
2. Week 13-14: Validate specialist agents

**Month 5-7**: Tier 2 Advanced
1. Week 15-16: Implement #3 (adaptive probabilities)
2. Week 17-20: Implement #5 (memetic RL)
3. Week 21-24: Implement #7 (hierarchical RL)

**Go/No-Go Decision Point**: After Tier 2
- If cumulative improvement ≥40%: Consider Tier 3
- If practical goals met: Enter maintenance mode

**Month 8-12+**: Tier 3 Research (Optional)
- Only if research goals, dedicated team, compute resources

---

## Dependency Matrix

| Enhancement | Depends On | Blocks | Can Be Parallel With |
|-------------|------------|--------|---------------------|
| #1: MO Reward | Phase 1&2 | #6, #3, #7 | #2 |
| #2: Constraint State | Phase 1&2 | #4, #3, #5, #7 | #1 |
| #3: Adaptive Probs | #1, #2 | - | #4, #5, #6 |
| #4: Specialists | #2 | - | #1, #3, #5, #6 |
| #5: Memetic RL | #2 | - | #1, #3, #4, #6 |
| #6: Archive Div | #1 | - | #2, #3, #4, #5 |
| #7: Hierarchical | #1, #2 | - | #3, #4, #5, #6 |
| #8: Multi-Agent | All Tier 1&2 | - | #9, #10 |
| #9: Transfer | All Tier 1&2 | - | #8, #10 |
| #10: Online | All Tier 1&2 | - | #8, #9 |

**Key Insight**: #1 and #2 are foundational for most enhancements, so prioritize them first.

---

## Resource Requirements

### Tier 1 (Months 1-2)
- **Developer Time**: 1 FTE
- **GPU Hours**: 100-150
- **Storage**: 20 GB
- **Risk**: Low

### Tier 2 (Months 3-7)
- **Developer Time**: 1-2 FTE
- **GPU Hours**: 300-500
- **Storage**: 50 GB
- **Risk**: Medium

### Tier 3 (Months 8-12+)
- **Developer Time**: 2-3 FTE
- **GPU Hours**: 1000+
- **Storage**: 100+ GB
- **Risk**: High

---

## Testing Dependencies

### Unit Tests
- Each enhancement requires its own unit tests
- Tests must not break existing Phase 1&2 tests

### Integration Tests
- Test enhancement with GA system
- Verify no performance regression on baseline

### System Tests
- Full training pipeline with enhancement
- Baseline comparison on standard test suite

---

## Rollback Strategy

### If Enhancement Fails
1. Disable enhancement via config killswitch
2. Revert to previous model checkpoint
3. Document failure reasons
4. Analyze root cause
5. Decide: fix, defer, or abandon

### Config Killswitches
```yaml
enhancements:
  constraint_specific_state: false  # #2
  multi_objective_reward: false     # #1
  specialist_agents: false          # #4
  archive_diversity: false          # #6
  adaptive_probabilities: false     # #3
  memetic_rl: false                 # #5
  hierarchical_rl: false            # #7
  multi_agent_rl: false             # #8
  transfer_learning: false          # #9
  online_learning: false            # #10
```

---

## Success Metrics

### Tier 1 Success Criteria
- 20%+ improvement in key metric (fitness, convergence, diversity)
- No regression on baseline benchmarks
- Computational overhead <10%

### Tier 2 Success Criteria
- 40%+ cumulative improvement
- Production stability maintained
- Positive ROI on development time

### Tier 3 Success Criteria
- Novel research contributions
- Publishable results
- Generalization demonstrated

---

## Conclusion

**Recommended Path**:
1. **Complete Phase 2 execution** (validation, training, promotion)
2. **Start with Tier 1** (#2 → #1 → #6) for quick wins
3. **Evaluate after Tier 1** (go/no-go decision)
4. **Proceed to Tier 2** (#4 → #3 → #5 → #7) if justified
5. **Consider Tier 3** (#8, #9, #10) only for research goals

**Key Dependencies**:
- #2 (constraint-specific state) is foundational for #3, #4, #5, #7
- #1 (multi-objective reward) is foundational for #3, #6, #7
- Tier 3 requires all Tier 1&2 complete

**Expected Timeline**:
- Tier 1: 2 months
- Tier 2: 5 months (cumulative: 7 months)
- Tier 3: 4+ months (cumulative: 11+ months)

---

**Last Updated**: November 17, 2025
