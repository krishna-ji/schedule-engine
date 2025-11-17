# Phase 2: RL Integration - TODO Checklist

**Status**: 🟡 Ready to Start (Phase 1.5 ✅ Complete)

---

## Phase 2.1: Gym Environment (Week 1-2)

### Environment Setup
- [ ] Create `src/rl/gym_env/` directory structure
- [ ] Install dependencies (gymnasium, stable-baselines3)
- [ ] Create `__init__.py` and module exports

### State Space (Priority: HIGH)
- [ ] Create `state_encoder.py`
- [ ] Implement population metrics encoding
  - [ ] Best/avg/worst fitness
  - [ ] Fitness standard deviation
  - [ ] Population diversity
- [ ] Implement progress metrics encoding
  - [ ] Current generation
  - [ ] Generations without improvement
  - [ ] Convergence rate
- [ ] Implement constraint violation metrics
  - [ ] Total hard violations
  - [ ] Total soft violations
  - [ ] Average violations per individual
- [ ] Add heuristic history (last N applications)
- [ ] Add state normalization
- [ ] Write unit tests for state encoder

### Action Space (Priority: HIGH)
- [ ] Create action mapping (20 actions: 19 heuristics + no-op)
- [ ] Implement action-to-heuristic dispatcher
- [ ] Add action validation
- [ ] Add action masking logic (optional)
- [ ] Write unit tests for action space

### Reward Function (Priority: CRITICAL)
- [ ] Create `reward_calculator.py`
- [ ] Implement fitness improvement reward
- [ ] Implement diversity bonus
- [ ] Implement time penalty
- [ ] Add configurable reward weights
- [ ] Add reward normalization/clipping
- [ ] Test reward signals on sample runs
- [ ] Write unit tests for reward calculator

### Gym Environment (Priority: CRITICAL)
- [ ] Create `schedule_env.py` implementing `gym.Env`
- [ ] Implement `__init__()` with config
- [ ] Implement `reset()` - initialize new GA run
- [ ] Implement `step()` - apply action, return obs/reward/done
- [ ] Implement `render()` for visualization (optional)
- [ ] Add episode termination logic
- [ ] Add info dict with detailed metrics
- [ ] Register environment with Gym
- [ ] Write integration tests

### GA Integration Hooks (Priority: HIGH)
- [ ] Modify `src/core/ga_scheduler.py`
- [ ] Add optional `rl_env` parameter
- [ ] Add state observation hook
- [ ] Add action application hook
- [ ] Ensure backward compatibility (works without RL)
- [ ] Add RL mode flag in config
- [ ] Test GA runs with/without RL

**Milestone 1**: ✅ Working Gym environment with random agent

---

## Phase 2.2: RL Agent Training (Week 3-4)

### Training Infrastructure (Priority: HIGH)
- [ ] Create `src/rl/agents/` directory
- [ ] Create `src/rl/training/` directory
- [ ] Install Stable-Baselines3
- [ ] Setup TensorBoard logging

### Agent Implementation (Priority: HIGH)
- [ ] Create `ppo_agent.py` (Proximal Policy Optimization)
  - [ ] Wrap with SB3 PPO
  - [ ] Configure hyperparameters
  - [ ] Add model saving/loading
- [ ] Create `dqn_agent.py` (Deep Q-Network)
  - [ ] Wrap with SB3 DQN
  - [ ] Configure hyperparameters
  - [ ] Add replay buffer
- [ ] Create `random_agent.py` (baseline)
  - [ ] Random action selection
  - [ ] Used for comparison

### Training Loop (Priority: CRITICAL)
- [ ] Create `trainer.py`
- [ ] Implement training loop
  - [ ] Episode iteration
  - [ ] State-action-reward collection
  - [ ] Agent learning updates
  - [ ] Periodic evaluation
- [ ] Add checkpointing (save best models)
- [ ] Add early stopping
- [ ] Add progress bar/logging
- [ ] Test on toy problem

### Configuration (Priority: HIGH)
- [ ] Create `configs/rl_config.yaml`
- [ ] Add RL config section to Pydantic models
- [ ] Define training hyperparameters
  - [ ] Learning rate
  - [ ] Batch size
  - [ ] Discount factor (gamma)
  - [ ] Number of episodes
- [ ] Define agent-specific hyperparameters
  - [ ] PPO: n_steps, clip_range, entropy coef
  - [ ] DQN: buffer size, epsilon decay
- [ ] Add reward weights configuration

### Hyperparameter Tuning (Priority: MEDIUM)
- [ ] Setup Optuna for hyperparameter optimization
- [ ] Define search space
- [ ] Run hyperparameter sweeps
  - [ ] Learning rate sweep
  - [ ] Batch size sweep
  - [ ] Reward weight sweep
- [ ] Document best hyperparameters
- [ ] Update config with best values

### Curriculum Learning (Priority: MEDIUM)
- [ ] Create `curriculum.py`
- [ ] Define curriculum stages
  - [ ] Easy: 10 courses, 50 generations
  - [ ] Medium: 20 courses, 100 generations
  - [ ] Hard: 40 courses, 200 generations
- [ ] Implement stage progression logic
- [ ] Add success criteria per stage
- [ ] Test curriculum training

**Milestone 2**: ✅ Trained PPO agent beating random baseline

---

## Phase 2.3: Agent Evaluation (Week 5)

### Baseline Strategies (Priority: HIGH)
- [ ] Create `src/rl/baselines/strategies.py`
- [ ] Implement baseline strategies:
  - [ ] Random selection
  - [ ] Round-robin
  - [ ] Fixed priority
  - [ ] Greedy (best recent)
  - [ ] Expert rules (hand-crafted)
- [ ] Create evaluation harness
- [ ] Run all strategies on test set

### Metrics Collection (Priority: HIGH)
- [ ] Create `evaluator.py`
- [ ] Collect performance metrics:
  - [ ] Final best fitness
  - [ ] Convergence speed (generations)
  - [ ] Solution quality (violations)
  - [ ] Total time
  - [ ] Heuristic applications
- [ ] Collect diversity metrics
- [ ] Collect heuristic usage statistics
- [ ] Export metrics to CSV/JSON

### Statistical Analysis (Priority: MEDIUM)
- [ ] Perform t-tests (RL vs baselines)
- [ ] Calculate effect sizes
- [ ] Create comparison tables
- [ ] Generate statistical significance reports

### Visualization (Priority: MEDIUM)
- [ ] Create `src/rl/visualization/` directory
- [ ] Create training curve plots
  - [ ] Reward over episodes
  - [ ] Loss over episodes
  - [ ] Entropy over episodes
- [ ] Create performance comparison plots
  - [ ] Box plots (fitness by strategy)
  - [ ] Bar charts (convergence speed)
  - [ ] Scatter plots (time vs quality)
- [ ] Create heuristic analysis plots
  - [ ] Usage frequency histogram
  - [ ] Effectiveness heatmap
  - [ ] State-action correlation
- [ ] Create policy visualization
  - [ ] Action probability distributions
  - [ ] Q-value heatmaps (DQN)
- [ ] Generate automated HTML reports

**Milestone 3**: ✅ Comprehensive evaluation showing RL > baselines

---

## Phase 2.4: Production Integration (Week 6)

### Model Deployment (Priority: HIGH)
- [ ] Create `src/rl/deployment/` directory
- [ ] Create `model_loader.py`
  - [ ] Load agent from checkpoint
  - [ ] Version management
  - [ ] Device selection (CPU/GPU)
- [ ] Create `inference.py`
  - [ ] Fast prediction interface
  - [ ] Batch prediction support
  - [ ] Timeout handling
- [ ] Create `models/rl_agents/` for checkpoints
- [ ] Add model serialization
- [ ] Test model loading speed (<100ms)

### Configuration Integration (Priority: HIGH)
- [ ] Add RL section to `configs/base.yaml`
  - [ ] Master killswitch (enabled: true/false)
  - [ ] Mode selection (training/inference/disabled)
  - [ ] Model path
  - [ ] Fallback strategy
- [ ] Create RLConfig Pydantic model
- [ ] Add config validation
- [ ] Add environment-specific overrides (test.yaml, prod.yaml)

### Hybrid Controller (Priority: HIGH)
- [ ] Create `src/rl/hybrid/hybrid_controller.py`
- [ ] Implement hybrid strategies:
  - [ ] RL-Primary (RL selects, heuristics execute)
  - [ ] RL-Fallback (try RL, fallback on failure)
  - [ ] RL-Assisted (heuristics + RL guidance)
- [ ] Add action validation
- [ ] Add error handling and fallback
- [ ] Add performance monitoring
- [ ] Test hybrid mode switching

### Integration Testing (Priority: CRITICAL)
- [ ] Test RL mode in GA scheduler
- [ ] Test hybrid mode in GA scheduler
- [ ] Test fallback mechanism
- [ ] Test with production data
- [ ] Performance testing (latency, memory)
- [ ] Stress testing (100+ runs)

### Monitoring & Logging (Priority: MEDIUM)
- [ ] Add RL-specific logging
  - [ ] Action selections
  - [ ] State observations
  - [ ] Rewards received
  - [ ] Fallback triggers
- [ ] Add performance metrics
  - [ ] Inference latency per prediction
  - [ ] Memory usage
  - [ ] Heuristic effectiveness
- [ ] Create RL dashboard (optional)

**Milestone 4**: ✅ RL agent running in production GA scheduler

---

## Phase 2.5: Advanced Features (Optional)

### Multi-Agent RL (Priority: LOW)
- [ ] Research multi-agent approaches
- [ ] Implement category-specific agents
- [ ] Implement agent coordination
- [ ] Test cooperative strategies

### Transfer Learning (Priority: LOW)
- [ ] Pre-train on synthetic problems
- [ ] Fine-tune on real problems
- [ ] Test transfer effectiveness

### Online Learning (Priority: LOW)
- [ ] Implement experience replay from production
- [ ] Add online policy updates
- [ ] Test adaptation to new problems

### Meta-RL (Priority: LOW)
- [ ] Research meta-learning approaches
- [ ] Implement MAML or similar
- [ ] Test few-shot adaptation

---

## Documentation Tasks

### Developer Documentation (Priority: HIGH)
- [ ] Write `docs/RL_ARCHITECTURE.md`
  - [ ] System overview
  - [ ] Component interactions
  - [ ] Data flow diagrams
- [ ] Write `docs/RL_TRAINING_GUIDE.md`
  - [ ] Setup instructions
  - [ ] Training procedure
  - [ ] Hyperparameter tuning
- [ ] Write `docs/RL_INTEGRATION_GUIDE.md`
  - [ ] How to use RL in GA
  - [ ] Configuration options
  - [ ] Troubleshooting

### User Documentation (Priority: MEDIUM)
- [ ] Write `docs/RL_QUICKSTART.md`
  - [ ] 5-minute getting started
  - [ ] Simple examples
- [ ] Write `docs/RL_CONFIG_GUIDE.md`
  - [ ] All config options explained
  - [ ] Example configs
- [ ] Write `docs/RL_FAQ.md`
  - [ ] Common questions
  - [ ] Known issues

### Research Documentation (Priority: MEDIUM)
- [ ] Write `docs/for_report/RL_METHODOLOGY.md`
  - [ ] Thesis chapter content
  - [ ] Experimental design
- [ ] Write `docs/for_report/RL_RESULTS.md`
  - [ ] Performance tables
  - [ ] Graphs and charts
- [ ] Write `docs/for_report/RL_ANALYSIS.md`
  - [ ] Discussion
  - [ ] Insights and findings

---

## Testing Checklist

### Unit Tests (Priority: HIGH)
- [ ] `test/rl/test_gym_env.py` - Environment logic
- [ ] `test/rl/test_state_encoder.py` - State encoding
- [ ] `test/rl/test_reward_calculator.py` - Reward computation
- [ ] `test/rl/test_action_space.py` - Action mapping
- [ ] `test/rl/test_agents.py` - Agent behavior

### Integration Tests (Priority: HIGH)
- [ ] `test/rl/test_ga_rl_integration.py` - GA + RL
- [ ] `test/rl/test_training_pipeline.py` - Full training
- [ ] `test/rl/test_inference.py` - Model loading
- [ ] `test/rl/test_hybrid_mode.py` - Hybrid controller

### Performance Tests (Priority: MEDIUM)
- [ ] Inference latency benchmark (<10ms)
- [ ] Memory usage benchmark (<500MB)
- [ ] Training time benchmark (<24h)
- [ ] Scaling tests (different problem sizes)

---

## Dependencies to Install

```bash
# Core RL
pip install gymnasium stable-baselines3[extra] tensorboard

# Optional: Advanced features
pip install ray[rllib] optuna

# Visualization
pip install plotly seaborn

# Update pyproject.toml
poetry add gymnasium stable-baselines3 tensorboard
poetry add --group dev optuna plotly
```

---

## Success Criteria

### Phase 2.1 (Gym Environment)
- ✅ Environment runs without errors
- ✅ State space correctly encodes GA metrics
- ✅ Actions correctly trigger heuristics
- ✅ Rewards calculated as expected
- ✅ Random agent can interact with environment

### Phase 2.2 (Training)
- ✅ PPO agent trains without errors
- ✅ Training reward increases over episodes
- ✅ Agent learns non-random policy (entropy decreases)
- ✅ Model checkpoints save/load correctly

### Phase 2.3 (Evaluation)
- ✅ RL agent beats random baseline by 20%+
- ✅ RL agent matches/beats fixed strategies
- ✅ Statistical significance confirmed (p < 0.05)
- ✅ Comprehensive plots generated

### Phase 2.4 (Production)
- ✅ Model loads in <100ms
- ✅ Inference latency <10ms per prediction
- ✅ Fallback mechanism works
- ✅ Zero crashes in 100 production runs
- ✅ Solution quality improved by 10%+

---

## Timeline Summary

| Week | Phase | Key Deliverables |
|------|-------|------------------|
| 1 | Environment Setup | Working Gym environment |
| 2 | Reward & Integration | Reward function, GA hooks |
| 3 | Agent Training | Trained PPO/DQN agents |
| 4 | Hyperparameter Tuning | Optimized models |
| 5 | Evaluation | Performance analysis, plots |
| 6 | Production Integration | Deployed in GA scheduler |

**Total Effort**: 6 weeks (assumes 20-30 hours/week)

---

## Priority Legend
- **CRITICAL**: Must be done, blocks other work
- **HIGH**: Important, should be done soon
- **MEDIUM**: Nice to have, can be deferred
- **LOW**: Optional, future work

---

## Next Actions (This Week)

### Monday
- [ ] Create `src/rl/gym_env/` directory structure
- [ ] Install gymnasium and stable-baselines3
- [ ] Create basic environment skeleton

### Tuesday-Wednesday
- [ ] Implement state encoder
- [ ] Test state observation from GA

### Thursday-Friday
- [ ] Implement action space mapping
- [ ] Test heuristic execution from actions

### Weekend
- [ ] Implement reward calculator
- [ ] Test full environment loop

**First Week Goal**: Working Gym environment with random agent ✅

---

## Notes & Questions

### Open Questions
- [ ] Q: Should we use discrete or continuous action space?
  - A: Start with discrete (19 heuristics), can add continuous parameters later

- [ ] Q: Dense or sparse rewards?
  - A: Start dense (every step), test sparse later

- [ ] Q: How many training episodes needed?
  - A: Start with 1000, increase if needed

- [ ] Q: GPU required for training?
  - A: Recommended but not required. CPU training will be slower.

### Design Decisions to Make
- [ ] State normalization strategy
- [ ] Action masking (prevent invalid actions)
- [ ] Episode termination criteria
- [ ] Curriculum learning stages
- [ ] Model architecture (MLP size, layers)

### Risks to Monitor
- [ ] RL convergence issues
- [ ] Overfitting to training problems
- [ ] Inference latency in production
- [ ] Integration complexity with GA

---

## Resources

### Learning Materials
- Stable-Baselines3 docs: https://stable-baselines3.readthedocs.io/
- Gymnasium docs: https://gymnasium.farama.org/
- RL course: Spinning Up in Deep RL (OpenAI)

### Papers to Read
- PPO: "Proximal Policy Optimization Algorithms" (Schulman et al.)
- DQN: "Playing Atari with Deep RL" (Mnih et al.)
- RL for combinatorial optimization papers

### Code Examples
- SB3 custom environments: https://github.com/DLR-RM/stable-baselines3/tree/master/docs/guide/examples
- Gym environment examples: https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs

---

**Status**: Ready to begin Phase 2.1! 🚀

**Last Updated**: November 15, 2025
