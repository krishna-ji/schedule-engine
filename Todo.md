# RL-Based Hyper-Heuristic Implementation TODO

**Project**: Evolving Schedule Engine from Pure GA to RL-Powered Hyper-Heuristic
**Date Created**: 2025-11-14
**Status**: Planning Phase

---

## PHASE 1: LNS-IGLS Hybrid Foundation (4-6 weeks)

**Goal**: Prove that integrating IGLS as a targeted repair tool improves baseline GA

**Status**:  **IMPLEMENTATION COMPLETE** (2025-11-14)

### 1.1 Environment Setup
- [x]  Install `pytorch` library: `uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- [x]  Verify installation with test imports
- [x]  Create new module structure: `src/lns/` directory

### 1.2 Conflict Detection System
- [x]  Create `src/lns/__init__.py`
- [x]  Create `src/lns/conflict_detection.py`
- [x]  Implement `find_hard_conflict_sessions()` function
  - Extract hard constraint violations from evaluation
  - Map violations to session IDs
  - Return list of conflicted `SessionGene` objects
- [x]  Write unit tests for conflict detection (test with known-bad individuals)
- [x]  Test with current dataset to verify correct identification

### 1.3 IGLS Subproblem Solver
- [x]  Create `src/lns/igls_repair.py`
- [x]  Implement IGLS for subproblems:
  - [x]  Adapt IGLS to take a partial schedule as a fixed background.
  - [x]  The search should only modify the "destroyed" sessions.
  - [x]  Evaluation function must check against the fixed background.
- [x]  Add soft constraint optimization to objective.
- [x]  Add time limit or iteration limit to the subproblem solver.
- [x]  Test on small subproblems (5 sessions).

### 1.4 LNS Framework
- [x]  Create `src/lns/lns_operator.py`
- [x]  Implement `LNS_IGLS_Repair()` function (Algorithm 2 from suggestion.md):
  - [x]  Destroy phase: Extract conflicted sessions
  - [x]  Repair phase: Call the IGLS subproblem solver
  - [x]  Reintegration phase: Merge the repaired solution
  - [x]  Return original schedule if IGLS fails to improve
- [x]  Add logging for LNS operations (track success/failure rates)
- [x]  Test end-to-end on full individuals with known conflicts

### 1.5 GA Integration
- [x]  Modify `src/core/ga_scheduler.py`:
  - [x]  Add LNS-CP repair trigger logic (every 100 generations OR stagnation)
  - [x]  Track stagnation counter
  - [x]  Apply LNS-CP to best individual when triggered
  - [x]  Log repair attempts and outcomes
- [x]  Add configuration parameters to `configs/base.yaml`:
  - [x]  `lns.enabled: true/false`
  - [x]  `lns.trigger_interval: 100` (tuned to 100 for production)
  - [x]  `lns.max_subproblem_size: 20`
  - [x]  `lns.cp_time_limit: 10.0`
  - [x]  `lns.stagnation_trigger: 50`
  - [x]  `lns.max_repairs_per_run: 10`

### 1.6 Benchmarking & Evaluation
- [x]  Create benchmarking script: `scripts/benchmark_lns_igls.py`
- [ ]  Run baseline GA (without LNS-IGLS) on test dataset
  - [ ] Record: final fitness, hard violations, soft violations, runtime
  - [ ] Save results to `output/benchmark_baseline/`
- [ ]  Run GA+LNS/IGLS hybrid on same dataset (PRODUCTION RUN IN PROGRESS)
  - [ ] Record: same metrics + LNS success rate
  - [ ] Save results to `output/benchmark_lns_igls/`
- [ ]  Generate comparison report:
  - [ ] Time to first feasible solution
  - [ ] Final solution quality (hard/soft violations)
  - [ ] Convergence plots
  - [ ] Statistical significance tests
- [x]  Document findings in `docs/for_report/PHASE1_LNS_IGLS_IMPLEMENTATION.md` (initial documentation)
- [ ]  Complete results documentation in `docs/for_report/PHASE1_LNS_IGLS_RESULTS.md` (after production run)

### Phase 1 Summary
**Completed**:
-  All core LNS-IGLS modules implemented and tested
-  Conflict detection system with comprehensive unit tests
-  IGLS repair solver adapted for subproblems
-  LNS operator with destroy-repair-reintegrate cycle
-  Full GA integration with interval and stagnation triggers
-  Configuration system for LNS-IGLS parameters
-  Unit tests (all passing in `test/test_lns.py`)
-  Benchmarking script ready for comparative analysis
-  Production configuration enabled and production run started

**In Progress**:
-  Production run with LNS-IGLS enabled (24-48 hours estimated)

**Remaining**:
-  Baseline comparison runs (without LNS-IGLS)
-  Statistical analysis and comparison report
-  Final results documentation with convergence analysis

**Note:** CP-SAT approach has been abandoned due to intractability. Phase 1 now uses LNS-IGLS (pure heuristic repair). See `docs/CP_REMOVAL_SUMMARY.md` for details.

---

## PHASE 2: RL Integration (8-10 weeks)

**Goal**: Implement production-ready RL system for hyper-heuristic selection

**Status**:  **Phase 2.1 Complete** - Gymnasium environment implemented |  **Phase 2.2-2.4 Ready**

**Guide**: Comprehensive implementation guide available in `suggest/rlphase2.2-2.4_guide_manual.md`
- Mathematical foundations (MDP, PPO, DQN, reward functions)
- Detailed algorithms with pseudocode
- Complete code examples for all components
- Performance benchmarks and success criteria

---

### Phase 2.1: Gymnasium Environment  **COMPLETE**

**Status**:  Implemented (2025-11-15) - 21 files, ~1500 lines production code

- [x]  Installed dependencies (gymnasium, stable-baselines3, torch, tensorboard)
- [x]  Created directory structure (`src/rl/` with 7 subdirectories)
- [x]  Implemented StateEncoder (25-dimensional observations)
  - [ ] Add genotype & phenotype diversity features (genotype_diversity, phenotype_diversity, unique_fitness_ratio) to the state vector
- [x]  Implemented ActionMapper (20 discrete actions with registry integration)
- [x]  Implemented RewardCalculator (multi-component: fitness + diversity - time)
- [x]  Implemented ScheduleEnv (full Gym interface: reset, step, render)
- [x]  Created agent wrappers (PPO, DQN, Random)
- [x]  Added RL configuration to `configs/base.yaml` (99 lines)
- [x]  Created Pydantic models for RL config (11 models, 140 lines)
- [x]  Git commit 80b0b10 (Phase 2.1 complete)
- [x]  Created comprehensive implementation guide (`suggest/rlphase2.2-2.4_guide_manual.md`)

---

### Phase 2.2: Training Infrastructure (Weeks 1-2) 

**Goal**: Train RL agents to learn optimal heuristic selection strategies

**Training Budget Example (guideline)**:
- Stage 1 (easy, 10 courses): 200 episodes (short: 100 generations each) ≈ 30s – 2 min overall depending on environment
- Stage 2 (medium, 20 courses): 300 episodes (200 generations each) ≈ 1–5 min total
- Stage 3 (hard, 40+ courses): 500 episodes (400 generations each) ≈ 5–15 min total
- Use shorter episode lengths during early stages and expand during later stages. These timings are approximate and highly dataset dependent.

#### Task 1: RLTrainer Class (Priority: HIGH, Est: 2 days)
- [ ] Create `src/rl/training/trainer.py` with RLTrainer class
- [ ] Implement basic training loop with progress tracking
- [ ] Add model creation, saving, and loading functionality
- [ ] Integrate TensorBoard logging for real-time monitoring
- [ ] Test with 10K timesteps smoke test to verify pipeline
- **Math**: Implements PPO loss $L^{\text{CLIP}}(\theta)$ via SB3
- **Expected output**: Trained model saved to `models/rl_agents/`
- **Files**: `src/rl/training/trainer.py` (~150 lines)
 - [ ] Accept curriculum schedule from config and support stage-specific episode counts (Stage1:200, Stage2:300, Stage3:500 recommended)
 - [ ] Integrate checkpoint callback and manifest logging (save metadata with seed, config)
 - [ ] Provide evaluation hooks for validation set per stage (`--validation-set`) and periodic evaluation

#### Task 2: Training Script (Priority: HIGH, Est: 1 day)
- [ ] Create `src/rl/training/train_script.py` as executable entry point
- [ ] Load scheduling data and create initial population
- [ ] Initialize environment and trainer with configuration
- [ ] Add command-line arguments (--timesteps, --agent-type, --save-path)
- [ ] Test end-to-end training pipeline (50K timesteps)
- **Usage**: `python src/rl/training/train_script.py --timesteps 50000 --agent ppo`
- **Expected runtime**: ~30 minutes for 50K timesteps
- **Files**: `src/rl/training/train_script.py` (~120 lines)

#### Task 3: CurriculumManager (Priority: MEDIUM, Est: 2 days)
- [ ] Create `src/rl/training/curriculum.py` with CurriculumManager class
- [ ] Implement 3-stage curriculum (easy: 10 courses → medium: 20 → hard: 40)
- [ ] Add stage transition logic with performance thresholds ($\bar{R} \geq \tau_i$)
- [ ] Implement problem filtering by difficulty (course count)
- [ ] Test curriculum progression (20K + 30K + 50K timesteps per stage)
- **Math**: Advance if mean reward $\geq$ threshold and variance < limit
- **Expected improvement**: 15-25% better generalization than flat training
- **Files**: `src/rl/training/curriculum.py` (~200 lines)
 - [ ] Implement `checkpoint_every` and `checkpoint_freq` settings per stage (configurable via YAML)
 - [ ] Support adaptive advancement: advance when validation score >= threshold for `k` consecutive checkpoints
 - [ ] Implement `curriculum_validate(stage)` to evaluate current stage against a validation set and return advancement boolean
 - [ ] Provide `restart_from_checkpoint()` to automatically revert to last validated checkpoint if validation degrades

#### Task 4: Hyperparameter Tuning (Priority: MEDIUM, Est: 2 days)
- [ ] Create `src/rl/training/tune_hyperparameters.py`
- [ ] Implement Optuna integration for Bayesian optimization
- [ ] Add grid search alternative for quick tuning (3x3x3 = 27 configs)
- [ ] Test configurations: learning_rate, n_steps, batch_size, gamma
- [ ] Save best configuration to `configs/rl_best_hyperparams.yaml`
- **Search space**: lr ∈ [1e-5, 1e-2], n_steps ∈ {1024, 2048, 4096}, batch ∈ {32, 64, 128}
- **Budget**: 50 Optuna trials or 1 hour wall time
- **Files**: `src/rl/training/tune_hyperparameters.py` (~250 lines)

#### Task 5: Training Callbacks (Priority: LOW, Est: 1 day)
- [ ] Create `src/rl/training/callbacks.py` with custom SB3 callbacks
- [ ] Implement PeriodicEvaluationCallback (evaluate every 5K timesteps)
- [ ] Implement EarlyStoppingCallback (patience=5, min_delta=0.01)
- [ ] Implement CheckpointCallback (save best model + periodic checkpoints)
- [ ] Test callback integration with trainer
- **Usage**: Automatically logs to TensorBoard, saves best model
- **Files**: `src/rl/training/callbacks.py` (~180 lines)

#### Task 5.1: Checkpoints & Validation (Priority: MEDIUM, Est: 2 days)
- [ ] Create `src/rl/training/checkpoints.py` to record checkpoint metadata (seed, config, val metrics, time)
- [ ] Create `scripts/generate_validation_set.py` to produce stage-specific validation datasets
- [ ] Create `scripts/select_best_checkpoint.py` to pick the best model by validation metrics
- [ ] Implement `models/rl_agents/manifest.json` tracker for produced checkpoints and final artifacts
- [ ] Integrate callbacks to automatically write entries to manifest on each checkpoint
- [ ] Add tests for checkpoint manifest correctness and selection logic (`test/rl/test_checkpoints.py`)

#### Task 5.2: Curriculum Metrics (Priority: MEDIUM, Est: 1 day)
- [ ] Implement `src/rl/training/curriculum_metrics.py` that provides genotype/phenotype diversity and population-level metrics
- [ ] Hook `curriculum_metrics` into `StateEncoder` and training callbacks to log diversity metrics
- [ ] Add unit tests for diversity computation to `test/rl/test_diversity_metrics.py`

**Phase 2.2 Deliverables**:
-  Trained PPO agent (100K timesteps, ~2 hours training)
-  TensorBoard logs showing learning progress
-  Best hyperparameters documented in config
-  Curriculum-trained model with better generalization
-  Checkpoint manifest & stage validation artifacts

---

### Phase 2.3: Deployment & Integration (Weeks 3-4) 

**Goal**: Deploy trained RL agent in production GA scheduler

#### Task 6: ModelLoader (Priority: HIGH, Est: 1 day)
- [ ] Create `src/rl/deployment/model_loader.py` with caching
- [ ] Implement fast model loading (<100ms target)
- [ ] Add model version management and validation
- [ ] Implement model caching for repeated loads
- [ ] Benchmark loading time across different models
- **Target**: <100ms model load time (measured via timeit)
- **Files**: `src/rl/deployment/model_loader.py` (~120 lines)

#### Task 7: RLInference Engine (Priority: HIGH, Est: 1 day)
- [ ] Create `src/rl/deployment/inference.py` for fast predictions
- [ ] Implement timeout protection (10ms per prediction)
- [ ] Add performance monitoring (track inference times)
- [ ] Add batch prediction support for efficiency
- [ ] Benchmark inference latency (<10ms deterministic prediction)
- **Target**: <10ms per action prediction (median over 1000 predictions)
- **Files**: `src/rl/deployment/inference.py` (~150 lines)

#### Task 8: HybridController (Priority: HIGH, Est: 2 days)
- [ ] Create `src/rl/hybrid/hybrid_controller.py`
- [ ] Implement 3 modes: RL-primary, RL-fallback, RL-assisted
- [ ] Add fallback strategies: random, greedy, round-robin
- [ ] Implement usage statistics tracking (RL % vs fallback %)
- [ ] Test mode switching and failure recovery
- **Modes**: Primary (trust RL), Fallback (timeout protection), Assisted (80% RL, 20% explore)
- **Files**: `src/rl/hybrid/hybrid_controller.py` (~220 lines)

#### Task 9: GA Integration (Priority: HIGH, Est: 2 days)
- [ ] Modify `src/core/ga_scheduler.py` to support RL mode
- [ ] Add `_init_rl()` method for component initialization
- [ ] Implement `_apply_rl_operators()` method for heuristic selection
- [ ] Add RL enable/disable configuration in YAML
- [ ] Test full GA run with RL enabled (compare vs baseline)
- **Integration points**: After fitness evaluation, before selection
- **Config**: `rl.enabled: true` in `configs/prod.yaml`
- **Files**: Modify `src/core/ga_scheduler.py` (+150 lines)

#### Task 10: Deployment Tests (Priority: MEDIUM, Est: 1 day)
- [ ] Create `test/rl/test_deployment.py`
- [ ] Test model loading speed (<100ms assertion)
- [ ] Test inference latency (<10ms assertion)
- [ ] Test hybrid controller mode switching
- [ ] Test full GA integration (RL vs non-RL comparison)
- **Coverage target**: >80% for deployment modules
- **Files**: `test/rl/test_deployment.py` (~200 lines)

#### Task 10.1: Model Promotion & Registry (Priority: MEDIUM, Est: 1 day)
- [ ] Create `scripts/promote_model_to_prod.py` to promote validated model to production (update `configs/prod.yaml`) and store model metadata in `models/rl_agents/manifest.json`
- [ ] Implement `src/rl/deployment/registry.py` to manage model versions and guarantee atomic updates to `configs/prod.yaml`
- [ ] Add tests to validate promote script and registry (`test/rl/test_registry.py`)

**Phase 2.3 Deliverables**:
-  Production-ready RL inference system
-  GA scheduler with RL integration
-  Hybrid controller with 3 modes
-  Performance benchmarks (<100ms load, <10ms inference)

---

### Phase 2.4: Evaluation & Comparison (Weeks 5-6) 

**Goal**: Rigorous evaluation proving RL superiority over baselines

#### Task 11: BaselineStrategies (Priority: HIGH, Est: 1 day)
- [ ] Create `src/rl/evaluation/baselines.py`
- [ ] Implement 5 baseline strategies: Random, Round-robin, Fixed-priority, Greedy, Expert-rules
- [ ] Test each baseline independently with GA scheduler
- [ ] Document baseline characteristics and expected performance
- **Baselines**: Random (naive), Round-robin (cyclic), Greedy (recent rewards), Expert (hand-crafted rules)
- **Files**: `src/rl/evaluation/baselines.py` (~150 lines)

#### Task 12: RLEvaluator (Priority: HIGH, Est: 2 days)
- [ ] Create `src/rl/evaluation/evaluator.py` with RLEvaluator class
- [ ] Implement multi-strategy evaluation (10 runs per strategy)
- [ ] Add results aggregation and comparison (mean, std, min, max)
- [ ] Add progress tracking and comprehensive logging
- [ ] Test with all strategies (RL + 4 baselines)
- **Output**: EvaluationResult dataclass with fitness, convergence, time, heuristic usage
- **Expected runtime**: ~5-10 hours for full evaluation (50 runs total)
- **Files**: `src/rl/evaluation/evaluator.py` (~280 lines)

#### Task 13: MetricsCollector (Priority: MEDIUM, Est: 1 day)
- [ ] Create `src/rl/evaluation/metrics.py`
- [ ] Implement comprehensive metrics tracking (fitness, convergence, diversity, heuristics)
- [ ] Add JSON export functionality for post-analysis
- [ ] Create metrics aggregation and summary statistics
- [ ] Test metrics collection across multiple runs
- **Metrics**: 15+ metrics including fitness trajectories, convergence speed, diversity maintenance
- **Files**: `src/rl/evaluation/metrics.py` (~200 lines)

#### Task 14: Statistical Analysis (Priority: HIGH, Est: 2 days)
- [ ] Create `src/rl/evaluation/statistical_tests.py`
- [ ] Implement t-tests and Mann-Whitney U tests for pairwise comparison
- [ ] Add Cohen's d effect size calculations
- [ ] Implement ANOVA and Tukey HSD post-hoc tests
- [ ] Generate comparison report with significance tests
- **Tests**: t-test, Mann-Whitney U, ANOVA, effect sizes, confidence intervals
- **Significance level**: α = 0.05, power = 0.80
- **Files**: `src/rl/evaluation/statistical_tests.py` (~300 lines)

#### Task 15: TrainingVisualizer (Priority: MEDIUM, Est: 1 day)
- [ ] Create `src/rl/visualization/training_plots.py`
- [ ] Implement reward curves over training episodes
- [ ] Add policy/value loss plots from TensorBoard logs
- [ ] Add entropy plots for exploration tracking
- [ ] Test plot generation with training logs
- **Plots**: Reward vs timesteps, loss curves, entropy decay, action distribution
- **Format**: High-quality PNG/PDF for thesis
- **Files**: `src/rl/visualization/training_plots.py` (~180 lines)

#### Task 16: PerformanceVisualizer (Priority: HIGH, Est: 1 day)
- [ ] Create `src/rl/visualization/performance_plots.py`
- [ ] Implement box plots for fitness comparison across strategies
- [ ] Add bar charts for convergence speed comparison
- [ ] Add scatter plots for time vs quality trade-off
- [ ] Generate publication-quality comparison figures
- **Plots**: Box plots, bar charts, scatter plots, convergence curves
- **Style**: Consistent colors, legends, axis labels for thesis
- **Files**: `src/rl/visualization/performance_plots.py` (~250 lines)

#### Task 17: HeuristicAnalyzer (Priority: MEDIUM, Est: 1 day)
- [ ] Create `src/rl/visualization/heuristic_plots.py`
- [ ] Implement usage frequency histograms per strategy
- [ ] Add effectiveness heatmaps (heuristic x state features)
- [ ] Add state-action correlation plots
- [ ] Test visualization with evaluation results
- **Plots**: Histograms, heatmaps, correlation matrices
- **Insight**: Which heuristics RL prefers vs baselines
- **Files**: `src/rl/visualization/heuristic_plots.py` (~200 lines)

#### Task 18: Evaluation Runner (Priority: HIGH, Est: 1 day)
- [ ] Create `src/rl/evaluation/run_evaluation.py` orchestration script
- [ ] Integrate all evaluation components into single pipeline
- [ ] Generate all reports and plots automatically
- [ ] Add command-line interface with configuration options
- [ ] Test end-to-end evaluation (full pipeline)
- **Usage**: `python src/rl/evaluation/run_evaluation.py --strategies all --runs 10`
- **Output**: Reports in `output/rl_metrics/`, plots in `output/rl_plots/`
- **Files**: `src/rl/evaluation/run_evaluation.py` (~180 lines)

**Phase 2.4 Deliverables**:
-  Comprehensive evaluation report with 5 strategies
-  Statistical significance confirmation (p < 0.05)
-  12+ publication-quality plots for thesis
-  RL outperforms baselines by >20%

---

### Testing Suite (Week 7) 

**Goal**: Comprehensive testing for code quality and reliability

#### Task 19: Unit Tests for Gym Environment (Priority: HIGH, Est: 1 day)
- [ ] Create `test/rl/test_gym_env.py`
- [ ] Test StateEncoder (dimensions, normalization, consistency)
- [ ] Test ActionMapper (space size, validity, heuristic application)
- [ ] Test RewardCalculator (range checking, improvement detection)
- [ ] Test ScheduleEnv (reset, step, termination conditions)
- **Coverage target**: >85% for gym_env modules
- **Files**: `test/rl/test_gym_env.py` (~250 lines)

#### Task 20: Integration Tests for Training (Priority: MEDIUM, Est: 1 day)
- [ ] Create `test/rl/test_training.py`
- [ ] Test trainer initialization and configuration
- [ ] Test training loop execution (1K timesteps smoke test)
- [ ] Test model checkpointing and loading
- [ ] Test callback integration
- **Tests**: End-to-end training pipeline validation
- **Files**: `test/rl/test_training.py` (~180 lines)

#### Task 20.1: Curriculum & Checkpoint Tests (Priority: HIGH, Est: 1 day)
- [ ] Create `test/rl/test_curriculum.py` to test `CurriculumManager` stage transitions, adaptive advancement, and environment generation
- [ ] Create `test/rl/test_checkpoints.py` to test manifest generation and best-checkpoint selection logic
- [ ] Create `test/rl/test_diversity_metrics.py` to validate genotype/phenotype computations and sub-sampling logic


#### Task 21: Integration Tests for Deployment (Priority: HIGH, Est: 1 day)
- [ ] Create `test/rl/test_deployment.py`
- [ ] Test model loader speed and caching
- [ ] Test inference engine latency
- [ ] Test hybrid controller switching logic
- [ ] Test GA integration end-to-end
- **Assertions**: <100ms load, <10ms inference, correct mode switching
- **Files**: `test/rl/test_deployment.py` (~200 lines)

#### Task 22: Performance Benchmarks (Priority: MEDIUM, Est: 1 day)
- [ ] Create `test/rl/test_performance.py`
- [ ] Benchmark inference latency (target: <10ms median)
- [ ] Benchmark model loading (target: <100ms)
- [ ] Benchmark memory usage (target: <500MB)
- [ ] Generate performance report with timing statistics
- **Tools**: timeit, memory_profiler, line_profiler
- **Files**: `test/rl/test_performance.py` (~150 lines)

**Testing Deliverables**:
-  >80% code coverage for RL modules
-  All tests passing in CI/CD
-  Performance benchmarks documented
-  Regression test suite for future changes

---

### Documentation (Week 8) 

**Goal**: Complete documentation for thesis and maintenance

#### Task 23: RL Architecture Guide (Priority: HIGH, Est: 1 day)
- [ ] Create `docs/RL_ARCHITECTURE.md`
- [ ] Document system overview with component diagram
- [ ] Explain component interactions and data flow
- [ ] Add state machine diagrams for environment
- [ ] Document design decisions and trade-offs
- **Sections**: Overview, Components, Data Flow, Design Rationale, Extension Points
- **Files**: `docs/RL_ARCHITECTURE.md` (~800 lines)

#### Task 24: RL Training Guide (Priority: HIGH, Est: 1 day)
- [ ] Create `docs/RL_TRAINING_GUIDE.md`
- [ ] Write setup instructions (dependencies, environment)
- [ ] Document training procedures (standard, curriculum, tuning)
- [ ] Add hyperparameter tuning guide with recommendations
- [ ] Include troubleshooting section
- **Audience**: Future developers and thesis readers
- **Files**: `docs/RL_TRAINING_GUIDE.md` (~600 lines)

#### Task 25: RL Integration Guide (Priority: HIGH, Est: 1 day)
- [ ] Create `docs/RL_INTEGRATION_GUIDE.md`
- [ ] Explain how to use RL in GA scheduler
- [ ] Document all configuration options
- [ ] Explain 3 hybrid modes with use cases
- [ ] Add troubleshooting and FAQ section
- **Examples**: Code snippets for common use cases
- **Files**: `docs/RL_INTEGRATION_GUIDE.md` (~500 lines)

#### Task 26: RL Quick Start (Priority: MEDIUM, Est: 0.5 days)
- [ ] Create `docs/RL_QUICKSTART.md`
- [ ] Write 5-minute getting started guide
- [ ] Add 3 simple usage examples
- [ ] Document common use cases
- [ ] Include FAQ for quick reference
- **Goal**: Get new users running RL in <10 minutes
- **Files**: `docs/RL_QUICKSTART.md` (~300 lines)

**Documentation Deliverables**:
-  4 comprehensive guides totaling ~2200 lines
-  Architecture documentation for thesis methodology
-  User guides for reproducibility
-  Quick start for demos and presentations

---

### Phase 3: Advanced RL Features (Weeks 9-12, Optional) 

**Goal**: Cutting-edge RL techniques for thesis contributions

#### Task 27: Multi-Agent RL (Priority: LOW, Est: 1 week)
- [ ] Research hierarchical multi-agent architectures
- [ ] Train 3 specialist agents (construction, perturbation, improvement)
- [ ] Implement meta-controller for agent selection
- [ ] Evaluate multi-agent vs single-agent baseline
- [ ] Document findings in thesis appendix
- **Architecture**: Meta-controller + 3 specialists with separate policies
- **Expected gain**: 10-15% improvement over single-agent
- **Files**: `src/rl/multi_agent/` (4 files, ~600 lines)

#### Task 28: Transfer Learning (Priority: LOW, Est: 1 week)
- [ ] Generate 1000 synthetic scheduling problems
- [ ] Pre-train agent on synthetic data (500K timesteps)
- [ ] Fine-tune on real problems (100K timesteps)
- [ ] Measure transfer effectiveness vs training from scratch
- [ ] Compare convergence speed and final quality
- **Hypothesis**: Pre-training reduces fine-tuning time by 50%
- **Files**: `src/rl/transfer/` (3 files, ~400 lines)

#### Task 29: Online Learning (Priority: LOW, Est: 1 week)
- [ ] Implement experience replay from production runs
- [ ] Add online policy update mechanism (safe updates)
- [ ] Deploy in simulated production environment
- [ ] Monitor adaptation over time (drift detection)
- [ ] Evaluate long-term performance stability
- **Safety**: Conservative updates, performance monitoring, rollback mechanism
- **Files**: `src/rl/online/` (3 files, ~350 lines)

#### Task 30: Meta-RL (Priority: LOW, Est: 1 week)
- [ ] Research MAML (Model-Agnostic Meta-Learning) for scheduling
- [ ] Implement meta-training loop with task distribution
- [ ] Test few-shot adaptation (5 steps to new problem)
- [ ] Compare with standard RL on unseen problems
- [ ] Document meta-learning benefits and limitations
- **Goal**: Learn initial policy that adapts quickly to new instances
- **Files**: `src/rl/meta/` (2 files, ~450 lines)

**Phase 3 Deliverables** (Optional):
-  4 advanced RL techniques implemented
-  Novel research contributions for thesis
-  Potential for follow-up journal publication
-  Demonstrates mastery of advanced RL concepts

---

## Summary: Phase 2 Implementation Plan

**Total Tasks**: 30 tasks across 4 sub-phases
**Estimated Time**: 8-12 weeks (excluding Phase 3 advanced features)
**Minimum Viable**: Phase 2.1-2.4 (Tasks 1-18) = 6 weeks
**With Testing & Docs**: +2 weeks = 8 weeks total
**With Advanced Features**: +4 weeks = 12 weeks total

**Priority Levels**:
- **HIGH**: Critical path tasks (Tasks 1-2, 6-9, 11-12, 14, 16, 18, 19, 21, 23-25)
- **MEDIUM**: Important but not blocking (Tasks 3-5, 13, 15, 17, 20, 22, 26)
- **LOW**: Optional advanced features (Tasks 27-30)

**Success Metrics**:
1.  Model loads <100ms, inference <10ms (production-ready)
2.  RL beats random baseline by >20% (statistical significance p<0.05)
3.  >80% test coverage for RL modules
4.  Complete documentation for thesis and reproducibility
5.  Publication-quality evaluation with 10+ plots

**Next Immediate Action**: Start Task 1 (RLTrainer class) using guide in `suggest/rlphase2.2-2.4_guide_manual.md`

---

---

## PHASE 3: DQN Agent Implementation (4-6 weeks)

**Goal**: Train a Deep Reinforcement Learning agent to learn optimal heuristic selection

### 3.1 Neural Network Architecture
- [ ] Create `src/rl/agents/dqn_agent.py`
- [ ] Implement `DQNNetwork` class (PyTorch):
  - [ ] Input layer: 5 neurons (state size)
  - [ ] Hidden layer 1: 128 neurons + ReLU
  - [ ] Hidden layer 2: 128 neurons + ReLU
  - [ ] Hidden layer 3: 64 neurons + ReLU
  - [ ] Output layer: 6 neurons (Q-values for each action)
- [ ] Test network forward pass with dummy data

### 3.2 DQN Agent Core
- [ ] Implement `DQNAgent` class:
  - [ ] Initialize main network and target network
  - [ ] Set up Adam optimizer
  - [ ] Initialize replay buffer (deque, max 10,000 experiences)
  - [ ] Initialize epsilon for exploration
- [ ] Implement `choose_action(state)`:
  - [ ] Epsilon-greedy selection
  - [ ] Random action with probability epsilon
  - [ ] Best Q-value action otherwise
- [ ] Implement `store_experience(s, a, r, s_next)`:
  - [ ] Add to replay buffer
- [ ] Implement epsilon decay logic

### 3.3 Training Logic
- [ ] Implement `train_on_batch(batch_size=32)`:
  - [ ] Sample random batch from replay buffer
  - [ ] Compute Q-values from main network
  - [ ] Compute target Q-values from target network
  - [ ] Calculate Huber loss
  - [ ] Backpropagate and update weights
  - [ ] Return loss for logging
- [ ] Implement `update_target_network()`:
  - [ ] Copy weights from main to target network
- [ ] Add gradient clipping for stability
- [ ] Add learning rate scheduling (optional)

### 3.4 RL Loop Integration
- [ ] Update `src/rl/hyper_heuristic_loop.py`:
  - [ ] Replace random agent with DQN agent
  - [ ] Add agent.learn() call after each step
  - [ ] Add target network update (every 100 steps)
  - [ ] Add experience replay trigger (only train if buffer > 1000)
  - [ ] Add epsilon decay after each iteration
- [ ] Add training metrics logging:
  - [ ] Loss per batch
  - [ ] Epsilon value over time
  - [ ] Q-value statistics
  - [ ] Action distribution over time

### 3.5 Training Execution
- [ ] Create training script: `scripts/train_dqn_agent.py`
- [ ] Set up training run with `test` config (fast iterations):
  - [ ] Max iterations: 500-1000
  - [ ] Small dataset for speed
- [ ] Implement training monitoring:
  - [ ] Real-time loss plotting (optional: use tensorboard)
  - [ ] Action selection heatmap
  - [ ] Reward curve
  - [ ] Fitness improvement curve
- [ ] Save checkpoints every 100 iterations
- [ ] Implement early stopping (if performance plateaus)

### 3.6 Agent Evaluation
- [ ] Create evaluation script: `scripts/evaluate_dqn_agent.py`
- [ ] Load trained agent weights
- [ ] Run agent in greedy mode (epsilon=0) on test problems
- [ ] Compare against baselines:
  - [ ] Random agent
  - [ ] Pure GA baseline
  - [ ] GA+LNS-CP hybrid
- [ ] Generate evaluation metrics:
  - [ ] Final fitness achieved
  - [ ] Time to convergence
  - [ ] Hard constraint violations
  - [ ] Action selection frequency
- [ ] Statistical significance testing (t-tests)

### 3.7 Policy Analysis
- [ ] Create analysis script: `scripts/analyze_learned_policy.py`
- [ ] Extract learned policy patterns:
  - [ ] Which actions are preferred in which states?
  - [ ] Does agent learn to use LNS-CP sparingly (due to cost)?
  - [ ] Does agent prioritize hard constraint fixing?
- [ ] Visualize state-action mappings:
  - [ ] Heatmaps of action selection vs. state features
  - [ ] Decision boundary plots (if possible in 2D projections)
- [ ] Generate interpretable policy summary
- [ ] Document findings in `docs/for_report/PHASE3_DQN_POLICY_ANALYSIS.md`

### 3.8 Hyperparameter Tuning
- [ ] Experiment with different network architectures:
  - [ ] Deeper networks (4-5 layers)
  - [ ] Wider networks (256 neurons per layer)
- [ ] Tune learning rate (try 1e-3, 1e-4, 1e-5)
- [ ] Tune epsilon decay rate (faster vs. slower exploration)
- [ ] Tune batch size (16, 32, 64)
- [ ] Tune replay buffer size (5k, 10k, 20k)
- [ ] Document best configuration in `configs/rl_best.yaml`

---

## PHASE 4: Advanced RL Techniques (Optional, 3-4 weeks)

**Goal**: Enhance the RL agent with state-of-the-art techniques

### 4.1 Double DQN
- [ ] Implement Double DQN variant:
  - [ ] Use main network to select action
  - [ ] Use target network to evaluate action
  - [ ] Reduces Q-value overestimation
- [ ] Compare against vanilla DQN
- [ ] Document improvement (if any)

### 4.2 Prioritized Experience Replay
- [ ] Replace uniform sampling with prioritized sampling:
  - [ ] Prioritize experiences with high TD-error
  - [ ] Use sum-tree for efficient sampling
- [ ] Implement importance sampling weights
- [ ] Benchmark against standard replay

### 4.3 Dueling DQN
- [ ] Modify network architecture:
  - [ ] Split into value stream and advantage stream
  - [ ] Combine to produce Q-values
- [ ] Test if this improves learning speed
- [ ] Compare performance

### 4.4 Multi-Objective Reward Shaping
- [ ] Experiment with different reward formulations:
  - [ ] Weighted sum of hard/soft violations
  - [ ] Lexicographic reward (hard first, then soft)
  - [ ] Sparse rewards (only when solution improves)
- [ ] A/B test reward functions
- [ ] Document optimal reward design

---

## PHASE 5: Additional Heuristics (Optional, 2-3 weeks)

**Goal**: Expand the action space with more diverse operators

### 5.1 Simulated Annealing Action
- [ ] Create `src/rl/actions/simulated_annealing.py`
- [ ] Implement basic SA with temperature schedule
- [ ] Add as Action 6 to toolbox
- [ ] Let RL agent learn when to use SA

### 5.2 Tabu Search Action
- [ ] Create `src/rl/actions/tabu_search.py`
- [ ] Implement tabu list for move prevention
- [ ] Add as Action 7 to toolbox
- [ ] Benchmark contribution

### 5.3 Greedy Construction Action
- [ ] Implement greedy session placement heuristic
- [ ] Use as "emergency reset" action
- [ ] Add to toolbox as Action 8

### 5.4 Re-train Agent with Expanded Toolbox
- [ ] Update action_size in config (now 8-9 actions)
- [ ] Re-train DQN agent
- [ ] Analyze if agent learns meaningful use of new actions

---

## PHASE 6: Comprehensive Evaluation (6-8 weeks)

**Goal**: Rigorous experimental validation and thesis-ready results

### 6.1 Dataset Preparation
- [ ] Create diverse test instances:
  - [ ] Small (50 courses)
  - [ ] Medium (150 courses) - current dataset
  - [ ] Large (300+ courses) - if feasible
  - [ ] Vary constraint density (tight vs. loose)
- [ ] Label each instance with difficulty metrics
- [ ] Create train/test split for RL agent

### 6.2 Baseline Implementations
- [ ] Ensure all baselines are production-ready:
  - [ ] Pure GA (NSGA-II)
  - [ ] GA + IGLS (current system)
  - [ ] GA + LNS-CP hybrid
  - [ ] Random hyper-heuristic
- [ ] Standardize all parameters for fair comparison
- [ ] Use same random seeds across runs

### 6.3 Large-Scale Experiments
- [ ] Run each algorithm on each test instance:
  - [ ] 30 independent runs per instance (statistical validity)
  - [ ] Record: best fitness, avg fitness, std dev, runtime
  - [ ] Save full convergence logs
- [ ] Use `prod` config for final runs (2000 generations)
- [ ] Parallelize experiments across machines if needed
- [ ] Estimated total runtime: ~1-2 weeks of computation

### 6.4 Statistical Analysis
- [ ] Perform hypothesis testing:
  - [ ] Friedman test (non-parametric ANOVA)
  - [ ] Post-hoc pairwise comparisons (Nemenyi test)
  - [ ] Effect size calculations (Cohen's d)
- [ ] Generate statistical significance tables
- [ ] Create ranking tables (mean rank per algorithm)
- [ ] Document methodology in `docs/for_report/STATISTICAL_METHODOLOGY.md`

### 6.5 Visualization & Reporting
- [ ] Generate convergence plots:
  - [ ] All algorithms on same chart per instance
  - [ ] Median + IQR bands
- [ ] Create box plots for final fitness distributions
- [ ] Generate action selection heatmaps (for DQN)
- [ ] Create radar charts for multi-criteria comparison
- [ ] Design infographic summary of results
- [ ] Save all figures in publication-quality format (PDF/SVG)

### 6.6 Ablation Studies
- [ ] Test contribution of each component:
  - [ ] RL agent without LNS-CP action → measure impact
  - [ ] RL agent without cost penalties → measure impact
  - [ ] RL agent with different reward functions → measure impact
- [ ] Create ablation study table
- [ ] Document findings in `docs/for_report/ABLATION_STUDIES.md`

### 6.7 Scalability Analysis
- [ ] Test agent on problems of increasing size
- [ ] Measure runtime scaling (linear? polynomial?)
- [ ] Identify performance bottlenecks
- [ ] Propose optimizations for large-scale problems

### 6.8 Transfer Learning Experiment
- [ ] Train DQN agent on small instances (50 courses)
- [ ] Test agent on larger instances (150 courses) WITHOUT retraining
- [ ] Measure performance degradation
- [ ] If successful, demonstrates generalization ability
- [ ] Document in `docs/for_report/TRANSFER_LEARNING_RESULTS.md`

---

## PHASE 7: Thesis Writing & Documentation (6-8 weeks)

**Goal**: Complete thesis manuscript and prepare for defense

### 7.1 Thesis Structure Setup
- [ ] Create thesis document structure (LaTeX or Word):
  - [ ] Abstract
  - [ ] Chapter 1: Introduction
  - [ ] Chapter 2: Literature Review
  - [ ] Chapter 3: Problem Formulation
  - [ ] Chapter 4: Methodology
  - [ ] Chapter 5: Implementation
  - [ ] Chapter 6: Experimental Results
  - [ ] Chapter 7: Discussion
  - [ ] Chapter 8: Conclusion & Future Work
  - [ ] References
  - [ ] Appendices
- [ ] Set up bibliography management (BibTeX/Zotero)

### 7.2 Literature Review
- [ ] Survey existing work:
  - [ ] University timetabling (UCTP) overview
  - [ ] Genetic algorithms for UCTP
  - [ ] Constraint programming for UCTP
  - [ ] Hybrid approaches (GA+CP, GA+LNS)
  - [ ] Hyper-heuristics in optimization
  - [ ] Reinforcement learning in combinatorial optimization
- [ ] Identify key papers (aim for 50-100 references)
- [ ] Write literature review chapter
- [ ] Create comparison table of related work

### 7.3 Methodology Chapter
- [ ] Document problem formulation:
  - [ ] Mathematical notation for entities (courses, rooms, etc.)
  - [ ] Hard constraints (formal definitions)
  - [ ] Soft constraints (formal definitions)
  - [ ] Objective function (multi-objective formulation)
- [ ] Document GA implementation:
  - [ ] Chromosome encoding
  - [ ] Genetic operators
  - [ ] NSGA-II algorithm
- [ ] Document LNS-IGLS integration:
  - [ ] Algorithm 2 from suggestion.md (with pseudocode)
  - [ ] IGLS subproblem formulation
- [ ] Document RL hyper-heuristic:
  - [ ] State representation
  - [ ] Action space
  - [ ] Reward function
  - [ ] DQN architecture
  - [ ] Algorithm 1 from suggestion.md (with pseudocode)

### 7.4 Implementation Chapter
- [ ] Document system architecture:
  - [ ] Component diagram
  - [ ] Data flow diagram
  - [ ] Class hierarchy
- [ ] Document key implementation details:
  - [ ] Python libraries used
  - [ ] Configuration system
  - [ ] Parallelization approach
- [ ] Include code snippets for critical functions
- [ ] Discuss implementation challenges and solutions

### 7.5 Results Chapter
- [ ] Present experimental setup:
  - [ ] Test instances description
  - [ ] Hardware/software environment
  - [ ] Parameter settings for all algorithms
- [ ] Present main results:
  - [ ] Comparison tables (all algorithms)
  - [ ] Convergence plots
  - [ ] Statistical test results
  - [ ] Box plots
- [ ] Present ablation study results
- [ ] Present transfer learning results (if applicable)
- [ ] Present learned policy analysis

### 7.6 Discussion Chapter
- [ ] Interpret results:
  - [ ] Why does RL hyper-heuristic outperform baselines?
  - [ ] What did the agent learn?
  - [ ] When does LNS-CP help most?
- [ ] Compare against literature:
  - [ ] How do results compare to state-of-the-art?
  - [ ] What is the novel contribution?
- [ ] Discuss limitations:
  - [ ] Computational cost of training
  - [ ] Scalability concerns
  - [ ] Transferability to other timetabling problems
- [ ] Discuss practical implications

### 7.7 Conclusion & Future Work
- [ ] Summarize contributions:
  - [ ] Novel RL-based hyper-heuristic framework
  - [ ] Successful integration of IGLS in LNS context
  - [ ] Demonstrated adaptive optimization
- [ ] Restate key findings
- [ ] Propose future research directions:
  - [ ] Multi-agent RL (multiple cooperating agents)
  - [ ] Meta-learning (learning to learn across problem instances)
  - [ ] Real-time timetabling (dynamic updates)
  - [ ] Application to other scheduling domains

### 7.8 Thesis Finalization
- [ ] Complete all chapters (first draft)
- [ ] Create all figures and tables
- [ ] Proofread entire document
- [ ] Get feedback from advisor
- [ ] Revise based on feedback (2-3 rounds)
- [ ] Format according to university guidelines
- [ ] Generate table of contents, list of figures, list of tables
- [ ] Final proofread
- [ ] Submit thesis

---

## PHASE 8: Publication Preparation (4-6 weeks, parallel with Phase 7)

**Goal**: Prepare conference/journal paper for publication

### 8.1 Target Venue Selection
- [ ] Research suitable venues:
  - [ ] **Conferences**: PATAT (Practice and Theory of Automated Timetabling), GECCO, CEC, EvoCOP
  - [ ] **Journals**: Journal of Scheduling, Computers & Operations Research, European Journal of Operational Research
- [ ] Check deadlines and submission guidelines
- [ ] Select primary target venue

### 8.2 Paper Outline
- [ ] Create conference paper structure (8-12 pages):
  - [ ] Abstract (200 words)
  - [ ] Introduction (1.5 pages)
  - [ ] Related Work (1 page)
  - [ ] Problem Formulation (1 page)
  - [ ] Proposed Method (2 pages)
  - [ ] Experimental Results (2 pages)
  - [ ] Conclusion (0.5 pages)
  - [ ] References
- [ ] Focus on novel contribution: RL hyper-heuristic

### 8.3 Paper Writing
- [ ] Write abstract (compelling summary of contribution)
- [ ] Write introduction (motivate the problem, state contribution)
- [ ] Write related work (concise survey, identify gap)
- [ ] Write method section (focus on RL agent + LNS-CP)
- [ ] Write results section (best results only, key plots)
- [ ] Write conclusion (restate contribution, future work)
- [ ] Create publication-quality figures (consistent style)
- [ ] Format references according to venue style

### 8.4 Paper Review & Submission
- [ ] Internal review (advisor, colleagues)
- [ ] Revise based on feedback
- [ ] Check for plagiarism (self-plagiarism from thesis)
- [ ] Proofread for grammar and clarity
- [ ] Format according to venue template
- [ ] Submit to target venue
- [ ] Prepare for potential revisions

---

## PHASE 9: Presentation & Defense Preparation (3-4 weeks)

**Goal**: Create compelling thesis defense presentation

### 9.1 Presentation Structure
- [ ] Create slide deck (30-40 slides for 45-60 min talk):
  - [ ] Title slide
  - [ ] Motivation (why timetabling is hard)
  - [ ] Problem definition (visual examples)
  - [ ] Literature review (brief, key gaps)
  - [ ] Proposed approach (system architecture diagram)
  - [ ] LNS-CP integration (algorithm visualization)
  - [ ] RL hyper-heuristic (3-layer diagram)
  - [ ] Experimental setup
  - [ ] Main results (comparison tables + plots)
  - [ ] Learned policy analysis
  - [ ] Ablation studies
  - [ ] Conclusion & contributions
  - [ ] Future work
  - [ ] Thank you / Q&A slide
- [ ] Design with clear visuals (avoid text-heavy slides)

### 9.2 Visual Aids
- [ ] Create animations:
  - [ ] GA evolution process (population improving over time)
  - [ ] LNS destroy-repair cycle
  - [ ] RL agent decision-making process
- [ ] Create demo video (optional):
  - [ ] System running in real-time
  - [ ] Visualization of schedule improving
- [ ] Prepare backup slides for potential questions

### 9.3 Defense Practice
- [ ] Practice full presentation (time it, aim for 40-45 min)
- [ ] Practice with advisor (get feedback)
- [ ] Practice with colleagues (mock defense)
- [ ] Prepare answers to anticipated questions:
  - [ ] Why RL instead of other metaheuristics?
  - [ ] How does LNS-CP avoid intractability?
  - [ ] What if RL agent fails to learn?
  - [ ] Can this generalize to other problems?
- [ ] Practice Q&A session

### 9.4 Final Defense
- [ ] Review all thesis chapters night before
- [ ] Review slides one last time
- [ ] Get good sleep before defense
- [ ] Deliver presentation with confidence
- [ ] Answer questions thoughtfully
- [ ] Celebrate successful defense! 

---

## Ongoing Tasks (Throughout All Phases)

### Code Quality & Maintenance
- [ ] Write docstrings for all new functions/classes
- [ ] Add type hints throughout codebase
- [ ] Follow PEP 8 style guidelines
- [ ] Run linter (flake8/ruff) regularly
- [ ] Write unit tests for critical components (aim for >70% coverage)
- [ ] Update `.github/instructions/` for new modules
- [ ] Keep `README.md` updated with new features

### Version Control
- [ ] Commit frequently with descriptive messages
- [ ] Use feature branches for major changes
- [ ] Merge to `dev-krishna` after testing
- [ ] Tag major milestones (e.g., `v2.0-phase1-complete`)
- [ ] Backup code to external drive/cloud regularly

### Documentation
- [ ] Update `docs/code/ENHANCE.md` for each enhancement
- [ ] Create `docs/for_report/` entries for major changes
- [ ] Keep `docs/QUICKREF.md` current
- [ ] Document configuration changes in config files

### Progress Tracking
- [ ] Update this TODO.md weekly (mark completed items)
- [ ] Maintain research journal (log insights, challenges)
- [ ] Schedule regular advisor meetings (bi-weekly)
- [ ] Present progress updates to advisor
- [ ] Adjust timeline based on actual progress

### Learning & Research
- [ ] Read key papers on hyper-heuristics
- [ ] Read key papers on RL for optimization
- [ ] Study PyTorch tutorials (if new to deep learning)
- [ ] Study IGLS implementation details
- [ ] Participate in relevant online communities (Stack Overflow, Reddit r/MachineLearning)

---

## Risk Mitigation

### High-Risk Items (Monitor Closely)
1. **RL Training Instability**: DQN may not converge
   - **Mitigation**: Start with simple problems, tune hyperparameters carefully, use baseline agents as fallback

2. **IGLS May Not Converge on Complex Subproblems**: Even small subproblems may be difficult
   - **Mitigation**: Reduce subproblem size further (5 sessions max), add strict time limits, have greedy fallback

3. **Computational Budget**: Training RL agents is expensive
   - **Mitigation**: Use cloud computing (AWS/Google Cloud), optimize code, parallelize training

4. **Time Constraints**: Thesis deadline approaching
   - **Mitigation**: Prioritize core phases (1-3), treat phases 4-5 as optional enhancements

### Contingency Plans
- **If RL completely fails**: Fall back to GA+LNS-CP hybrid as main contribution (still novel)
- **If LNS-IGLS integration fails**: Focus on pure RL hyper-heuristic with GA operators only
- **If thesis deadline is tight**: Cut optional phases, focus on strong evaluation of core system

---

## Success Metrics

### Minimum Viable Thesis (Pass Threshold)
-  Working GA+LNS-CP hybrid outperforms baseline
-  Functional RL environment and basic agent
-  Comparative evaluation showing some improvement
-  Complete thesis document with all chapters

### Strong Thesis (High Grade)
-  DQN agent learns meaningful policy
-  Demonstrated adaptive behavior (agent changes strategy based on state)
-  Comprehensive benchmarking with statistical validation
-  Published/submitted conference paper

### Outstanding Thesis (Top Grade + Publication)
-  DQN agent significantly outperforms all baselines
-  Transfer learning success (generalization across instances)
-  Rigorous ablation studies
-  Accepted publication in top-tier venue
-  Potential for journal extension

---

## Timeline Summary

| Phase | Duration | Key Deliverable | Status |
|-------|----------|----------------|--------|
| Phase 1: LNS-CP | 4-6 weeks | GA+LNS-CP hybrid system |  **COMPLETE** (Evaluation Pending) |
| Phase 2: RL Env | 3-4 weeks | Working RL environment + random agent |  Not Started |
| Phase 3: DQN | 4-6 weeks | Trained DQN agent |  Not Started |
| Phase 4: Advanced RL | 3-4 weeks | Enhanced RL techniques (optional) |  Not Started |
| Phase 5: More Heuristics | 2-3 weeks | Expanded action space (optional) |  Not Started |
| Phase 6: Evaluation | 6-8 weeks | Comprehensive experimental results |  Not Started |
| Phase 7: Thesis | 6-8 weeks | Complete thesis manuscript |  Not Started |
| Phase 8: Publication | 4-6 weeks | Submitted paper (parallel with Phase 7) |  Not Started |
| Phase 9: Defense | 3-4 weeks | Defense presentation + successful defense |  Not Started |

**Total Estimated Time**: 6-9 months (depending on optional phases)

---

## Notes

- **Priority Order**: Phases 1 → 2 → 3 → 6 → 7 → 9 (skip 4-5 if time-constrained)
- **Current Status**: Planning phase complete, ready to start Phase 1
- **Next Immediate Action**: Implement `find_hard_conflict_sessions()` and adapt IGLS for subproblems
- **Weekly Goal**: Complete at least 3-5 tasks per week
- **Advisor Check-ins**: Every 2 weeks, present progress and get feedback

---

## Recent Updates

### 2025-11-14: Phase 1 Implementation Complete 
- **Milestone**: LNS-CP hybrid system fully implemented and integrated
- **Modules Created**:
  - `src/lns/conflict_detection.py` - Tracks hard constraint violations
  - `src/lns/cp_repair.py` - CP-SAT subproblem solver
  - `src/lns/lns_operator.py` - LNS operator with destroy-repair-reintegrate
- **Integration**: GA scheduler updated with LNS-CP triggers
- **Configuration**: Added LNS parameters to base.yaml and prod.yaml
- **Testing**: All unit tests passing (`test/test_lns.py`)
- **Documentation**: Phase 1 implementation documented in `docs/for_report/`
- **Status**: Production run with LNS-CP enabled (in progress)
- **Next**: Complete evaluation phase (baseline comparison, statistical analysis)

---

## PHASE 2.5: RL-GA INTEGRATION ANALYSIS & ENHANCEMENT (CURRENT)

**Date**: 2025-11-17
**Status**:  **ANALYSIS COMPLETE**

### 2.5.1 Comprehensive Codebase Analysis
- [x]  Complete codebase review of RL-GA integration
- [x]  Identify bugs and issues (28 issues cataloged)
- [x]  Document all heuristic toolbox functions (19 operators)
- [x]  Create RL validation and pipeline setup guide
- [x]  Draft next-level enhancement roadmap
- [x]  Create refactoring plan

### 2.5.2 Documentation Deliverables
**Location**: `docs/_ai__suggestions_11_17/`

Created comprehensive documentation suite:
- [x]  `00_INDEX_AND_SUMMARY.md` - Complete analysis index (11,000 words)
- [x]  `01_CODEBASE_ISSUES_AND_BUGS.md` - 28 issues with mitigations (21,000 words)
- [x]  `02_HEURISTIC_TOOLBOX_DOCUMENTATION.md` - Complete function reference (33,000 words)
- [x]  `03_RL_VALIDATION_AND_PIPELINE_SETUP.md` - Training & deployment guide (32,000 words)
- [x]  `04_NEXT_LEVEL_ENHANCEMENTS.md` - 3-tier enhancement roadmap (27,000 words)
- [x]  `05_REFACTORING_PLAN.md` - 15 refactoring tasks prioritized (18,000 words)

**Total Documentation**: ~150 pages equivalent, ~140,000 words

### 2.5.3 Key Findings Summary

**Critical Issues (P0/P1) - 11 identified:**
1. No integration tests for RL-GA pipeline
2. Missing error handling in ActionMapper
3. State encoder normalization issues
4. RL component cleanup missing
5. Unsafe YAML loading
6. Inconsistent heuristic return types
7. Model path resolution fragile
8. Insufficient validation set size
9. No production RL monitoring
10. Missing model version tracking
11. Curriculum advancement too aggressive

**Heuristic Analysis:**
- 19 operators across 5 categories documented
- Performance benchmarks: 5ms (fastest) to 8000ms (slowest)
- 14/19 enabled by default, 12/19 production-ready
- Comprehensive usage patterns and trade-offs documented

**Enhancement Roadmap:**
- **Tier 1** (Foundational): Multi-agent specialists, adaptive state, enhanced curriculum, reward shaping
- **Tier 2** (Advanced): MAML, HER, PER, multi-task learning
- **Tier 3** (Research): NAS, GNN, Transformers, Offline RL
- Expected impact: +40-80% improvement potential
- Timeline: 6-18 months full implementation

**Refactoring Plan:**
- 15 tasks across 5 categories
- Phase 1 (2 weeks): Critical fixes (27 hours)
- Phase 2 (2 weeks): Quality improvements (21 hours)
- Phase 3 (2 weeks): Enhancements (76 hours)
- Total effort: 124 hours over 6 weeks

### 2.5.4 Immediate Action Items

**Week 1-2 (Critical Fixes):**
- [ ] Fix unsafe YAML loading (1 hour)
- [ ] Standardize heuristic return types (4 hours)
- [ ] Add error handling to ActionMapper (6 hours)
- [ ] Create integration test suite (16 hours)

**Week 3-4 (Quality Improvements):**
- [ ] Refactor state encoder (6 hours)
- [ ] Extract config validation (5 hours)
- [ ] Implement caching (3 hours)
- [ ] Add lazy model loading (4 hours)
- [ ] Add RL cleanup (3 hours)

**Week 5-6 (Foundation):**
- [ ] Unified logging (8 hours)
- [ ] Increase test coverage (30 hours)
- [ ] Add docstrings (15 hours)
- [ ] Add type hints (20 hours)

### 2.5.5 RL Training & Validation

**Training Pipeline Setup:**
- [ ] Generate validation sets (20-30 problems per stage)
- [ ] Configure training profiles (test/med/prod)
- [ ] Execute curriculum training (60-120 minutes)
- [ ] Monitor with TensorBoard
- [ ] Validate checkpoints

**Validation Strategy (3-level):**
1. **Online**: During training via callbacks
2. **Offline**: Post-training checkpoint evaluation
3. **Production**: Baseline comparison

**Deployment Checklist:**
- [ ] Select best checkpoint (median_reward metric)
- [ ] Validate checkpoint integrity
- [ ] Update configs/prod.yaml
- [ ] Register deployment
- [ ] Test inference latency (<10ms)
- [ ] Run baseline comparison
- [ ] Monitor RL usage in production

### Phase 2.5 Summary

**Analysis Scope:**
- 50+ files reviewed
- 5,200+ lines of RL code analyzed
- 19 heuristic operators documented
- 28 bugs/issues identified
- 14 enhancement proposals drafted
- 15 refactoring tasks planned

**Key Deliverables:**
-  Complete issue catalog with mitigations
-  Comprehensive heuristic documentation
-  End-to-end validation & pipeline guide
-  Multi-tier enhancement roadmap
-  Prioritized refactoring plan
-  Updated TODO with action items

**Next Steps:**
1. Start Phase 1 critical fixes (Week 1-2)
2. Begin RL training with validation
3. Execute refactoring plan phases
4. Implement Tier 1 enhancements (Month 2-4)

**Status**: Analysis complete, ready for implementation phase

---

## PHASE 3: ADVANCED RL-GA ENHANCEMENTS (Q1-Q3 2026)

**Date**: 2025-11-17  
**Status**:  **PLANNED**  
**Goal**: Implement advanced optimization techniques for 30-40% improvement  
**Documentation**: `docs/12-advanced-rl-ga-framework-integration/`

### Prerequisites (Must Complete First)

- [ ] ⚠️ **PHASE 2 EXECUTION**: Complete remaining Phase 2.2-2.4 execution tasks
  - [ ] Generate validation sets (30 problems per stage: easy, medium, hard)
  - [ ] Run curriculum training (100K-300K steps with GPU)
  - [ ] Select best checkpoint using validation metrics
  - [ ] Promote checkpoint to production (`scripts/promote_model_to_prod.py`)
  - [ ] Enable RL in `configs/prod.yaml` (set `rl.enabled: true`)
  - [ ] Run baseline comparisons (RL-enabled vs RL-disabled GA)
  - [ ] Update `docs/06-development/implementation-notes/PHASE_2_RL_COMPLETE.md` with empirical results

### 3.1 Tier 1: Quick Wins (Months 1-2, Expected: +30% improvement)

**Goal**: Achieve measurable improvements with proven techniques  
**Resources**: 1 FTE, 150 GPU hours, 2 months  
**Risk**: Low

#### Enhancement #2: Constraint-Specific State Encoding (Week 1-2)
- [ ] Expand `StateEncoder` from 21 to 39 dimensions
- [ ] Add per-constraint violation breakdown (14 hard + 4 soft = 18 features)
- [ ] Update `observation_space` in `ScheduleEnv`
- [ ] Create unit tests for enhanced state encoder
- [ ] Retrain RL agent with constraint-specific state (50K steps)
- [ ] Validate targeted repair improvement (expect 30-40% faster)
- [ ] Document implementation in `docs/12-advanced-rl-ga-framework-integration/`
- **Files**: `src/rl/gym_env/state_encoder.py`, `test/rl/test_constraint_state.py`
- **Dependencies**: None (Phase 1&2 complete)

#### Enhancement #1: Multi-Objective Reward Shaping (Week 3-4)
- [ ] Install `pygmo` for hypervolume calculation
- [ ] Implement `HypervolumeCalculator` class
- [ ] Update `RewardCalculator` with hypervolume-based rewards
- [ ] Add reference point configuration to `configs/base.yaml`
- [ ] Retrain RL agent with MO rewards (100K steps)
- [ ] Validate Pareto front quality improvement (expect 20-30% better diversity)
- [ ] Generate hypervolume convergence plots
- [ ] Document in `docs/12-advanced-rl-ga-framework-integration/`
- **Files**: `src/rl/gym_env/hypervolume.py`, `src/rl/gym_env/reward_calculator.py`
- **Dependencies**: None (Phase 1&2 complete)

#### Enhancement #6: Archive-Based Diversity (Week 5-8)
- [ ] Implement `BehavioralArchive` class with novelty metric
- [ ] Create `behavioral_features.py` for phenotypic characterization
- [ ] Integrate archive into NSGA-II selection process
- [ ] Add novelty-fitness trade-off parameter (α = 0.2 typical)
- [ ] Test archive coverage and diversity metrics
- [ ] Retrain and validate improved solution diversity (expect 30% more diverse)
- [ ] Document in `docs/12-advanced-rl-ga-framework-integration/`
- **Files**: `src/diversity/archive.py`, `src/diversity/behavioral_features.py`, `src/core/ga_scheduler.py`
- **Dependencies**: #1 (multi-objective understanding helps archive selection)

#### Tier 1 Go/No-Go Decision (End of Month 2)
- [ ] Measure cumulative improvement on key metrics
- [ ] Assess: If ≥20% improvement → Proceed to Tier 2
- [ ] Assess: If <10% improvement → Reconsider strategy, consolidate gains
- [ ] Document Tier 1 results and decision in `docs/12-advanced-rl-ga-framework-integration/`

---

### 3.2 Tier 2: Advanced Techniques (Months 3-7, Expected: +40% cumulative)

**Goal**: Implement complex enhancements requiring Tier 1 foundation  
**Resources**: 1-2 FTE, 500 GPU hours, 5 months  
**Risk**: Medium

#### Enhancement #4: Specialist RL Agents (Week 9-12)
- [ ] Design training protocol for repair agent (infeasible solutions only)
- [ ] Design training protocol for optimizer agent (feasible solutions only)
- [ ] Train repair agent (focus: hard constraint reduction, 50K steps)
- [ ] Train optimizer agent (focus: soft constraint reduction, 50K steps)
- [ ] Implement coordinator logic (switch based on hard_violations > 0)
- [ ] Test agent specialization effectiveness
- [ ] Validate 20-30% improvement in task-specific metrics
- [ ] Document in `docs/12-advanced-rl-ga-framework-integration/`
- **Files**: `src/rl/agents/specialist_agents.py`, `src/rl/agents/coordinator.py`
- **Dependencies**: #2 (constraint-specific state helps specialization)

#### Enhancement #3: Adaptive Probabilities (Week 13-14)
- [ ] Expand action space: add increase/decrease cxpb/mutpb actions (4 new actions)
- [ ] Implement probability control policy in `ActionMapper`
- [ ] Update GA to use dynamic crossover/mutation probabilities
- [ ] Train agent with probability control (50K steps)
- [ ] Monitor probability adaptation patterns
- [ ] Validate 10-15% faster convergence
- [ ] Document in `docs/12-advanced-rl-ga-framework-integration/`
- **Files**: `src/rl/gym_env/action_mapper.py`, `src/core/ga_scheduler.py`
- **Dependencies**: #1 (MO rewards), #2 (constraint state)

#### Enhancement #5: Memetic RL (Adaptive Local Search) (Week 15-18)
- [ ] Design RL-controlled IGLS budget allocation
- [ ] Expand action space: add increase/decrease IGLS budget actions
- [ ] Implement computational cost tracking
- [ ] Update `repair_igls.py` to accept dynamic iteration budget
- [ ] Train agent with local search control (100K steps)
- [ ] Measure computational efficiency (expect 50% savings)
- [ ] Validate quality maintained or improved (+15%)
- [ ] Document in `docs/12-advanced-rl-ga-framework-integration/`
- **Files**: `src/rl/local_search_controller.py`, `src/ga/operators/repair_igls.py`
- **Dependencies**: #2 (constraint state guides LS decisions)

#### Enhancement #7: Hierarchical RL (Week 19-22)
- [ ] Design hierarchical action space (high-level: 5 categories, low-level: 3-5 operators each)
- [ ] Train high-level policy (category selection, 50K steps)
- [ ] Train 5 low-level policies (operator selection within category, 30K steps each)
- [ ] Implement options framework for temporal abstraction
- [ ] Test hierarchical coordination
- [ ] Validate 15-25% better exploration
- [ ] Document in `docs/12-advanced-rl-ga-framework-integration/`
- **Files**: `src/rl/hierarchical/high_level_policy.py`, `src/rl/hierarchical/low_level_policies.py`
- **Dependencies**: #1 (MO rewards), #2 (constraint state)

#### Tier 2 Go/No-Go Decision (End of Month 7)
- [ ] Measure cumulative improvement (expect ≥40%)
- [ ] Assess production stability
- [ ] Calculate ROI on development time
- [ ] Decide: Proceed to Tier 3 (research goals) or enter maintenance mode
- [ ] Document Tier 2 results and decision

---

### 3.3 Tier 3: Research Extensions (Months 8-12+, OPTIONAL)

**Goal**: Explore cutting-edge techniques for research contributions  
**Resources**: 2-3 FTE, 1000+ GPU hours, 4+ months  
**Risk**: High (research-level uncertainty)  
**Note**: Only pursue if dedicated research team, compute resources, and strong research motivation

#### Enhancement #8: Multi-Agent RL (Week 23-28, OPTIONAL)
- [ ] Research hierarchical multi-agent architectures for scheduling
- [ ] Train K rank-specific specialist agents (K = 3-5)
- [ ] Implement meta-controller for agent routing
- [ ] Add inter-agent communication (optional)
- [ ] Evaluate multi-agent vs single-agent baseline
- [ ] Target: 10-20% improvement on hard problems
- [ ] Document research findings for publication
- **Files**: `src/rl/multi_agent/rank_specialists.py`, `src/rl/multi_agent/meta_controller.py`
- **Dependencies**: All Tier 1&2 enhancements

#### Enhancement #9: Transfer Learning (Week 29-36, OPTIONAL)
- [ ] Generate 1000+ synthetic scheduling problems
- [ ] Implement domain randomization for synthetic problems
- [ ] Pre-train agent on synthetic data (500K steps)
- [ ] Fine-tune on real problems (100K steps)
- [ ] Measure transfer effectiveness vs training from scratch
- [ ] Target: 50% reduction in fine-tuning time
- [ ] Document research findings for publication
- **Files**: `scripts/generate_synthetic_problems.py`, `src/rl/transfer/domain_randomization.py`
- **Dependencies**: All Tier 1&2 enhancements

#### Enhancement #10: Online Learning (Week 37-42, OPTIONAL)
- [ ] Implement experience replay from production runs
- [ ] Design safe policy update mechanism (conservative updates)
- [ ] Add drift detection and performance monitoring
- [ ] Deploy in simulated production environment
- [ ] Monitor long-term adaptation and stability
- [ ] Target: Adapt to problem distribution shifts
- [ ] Document research findings for publication
- **Files**: `src/rl/online/experience_collector.py`, `src/rl/online/safe_updater.py`
- **Dependencies**: All Tier 1&2 enhancements + production deployment experience

---

## PHASE 4: CONFIGURATION REORGANIZATION (OPTIONAL)

**Date**: 2025-11-17  
**Status**:  **PLANNED** (Optional Enhancement)  
**Goal**: Centralize configs into subdirectories for better organization

### 4.1 Configuration Restructuring (Week 1-2, OPTIONAL)

- [ ] Create subdirectory structure: `configs/ga/`, `configs/rl/`
- [ ] Split `configs/base.yaml` into:
  - `configs/ga/base.yaml` (GA-specific settings)
  - `configs/rl/base.yaml` (RL-specific settings)
  - `configs/common/base.yaml` (shared settings)
- [ ] Create environment variants in each subdirectory:
  - `configs/ga/{base,prod,test,med}.yaml`
  - `configs/rl/{base,prod,test,med}.yaml`
- [ ] Update `src/config/loader.py` to support hierarchical loading
- [ ] Update all documentation references to new config paths
- [ ] Test backward compatibility with existing configs
- [ ] Update `.github/instructions/config.instructions.md`

**Note**: This is a major refactoring. Only pursue if configuration management becomes unwieldy. Current flat structure with base.yaml + env overrides works well for now.

---

## DOCUMENTATION & MAINTENANCE (Ongoing)

### Documentation Updates (Throughout Phase 3)

- [ ] Update `docs/INDEX.md` with Phase 3 documentation links
- [ ] Create mathematical foundations documentation (`docs/12-advanced-rl-ga-framework-integration/10-mathematical-foundations.md`)
- [ ] Create algorithm documentation (`docs/12-advanced-rl-ga-framework-integration/20-nsga-ii-algorithm.md`)
- [ ] Create GPU acceleration guide (`docs/12-advanced-rl-ga-framework-integration/32-gpu-acceleration.md`)
- [ ] Update `README.md` with Phase 3 status and achievements
- [ ] Create Phase 3 implementation notes in `docs/06-development/implementation-notes/`
- [ ] Update thesis report sections in `docs/07-thesis-report/`

### Testing & Quality (Throughout Phase 3)

- [ ] Maintain >80% test coverage for new modules
- [ ] Add integration tests for each enhancement
- [ ] Create benchmarking suite for enhancement comparison
- [ ] Document testing protocols in `test/README.md`
- [ ] Set up continuous benchmarking (track performance over time)

### Code Quality (Throughout Phase 3)

- [ ] Add comprehensive docstrings to all new modules
- [ ] Add type hints throughout new code
- [ ] Run linters (flake8/ruff) regularly
- [ ] Conduct code reviews for major changes
- [ ] Maintain `.github/instructions/` for new modules

---

## FUTURE SUGGESTIONS & IDEAS (Phase 3+)

**Location**: `docs/11-v2-suggestion-updated/` (if additional ideas emerge)

### Potential Future Enhancements (Not Yet Planned)

- [ ] Create `docs/11-v2-suggestion-updated/` directory if new techniques identified
- [ ] Document any novel ideas emerging during Phase 3 implementation
- [ ] Propose research directions for Phase 4 and beyond
- [ ] Explore alternative RL algorithms (SAC, TD3, A3C)
- [ ] Investigate neural architecture search for state encoders
- [ ] Consider graph neural networks for schedule representation
- [ ] Explore offline RL techniques for production deployment
- [ ] Research meta-learning (MAML) for fast adaptation

---

## TIMELINE SUMMARY

### Completed (2025-11)
-  Phase 1: Heuristic Toolbox (19 operators)
-  Phase 2.1-2.4: RL Integration (code complete)
-  Phase 2.5: Comprehensive analysis and planning
-  Documentation structure created (`docs/12-advanced-rl-ga-framework-integration/`)
-  **Critical Bugfixes (2025-11-21)**:
   -  Fixed missing `numpy` imports (`src/constraints/soft.py`, `src/core/ga_scheduler.py`)
   -  Fixed `UnboundLocalError` for `hv`, `spacing`, `spread` variables in metrics tracking
   -  Optimized diversity calculation with sampling (100x speedup: 50s → 0.5s/gen)
   -  Made GPU evaluator robust (skip invalid genes instead of crashing)
   -  **Impact**: Reduced generation time from ~60s to ~5-10s (10x faster!)
-  **Unified CLI Launcher (2025-11-21)**:
   -  Created `scripts/launcher.py` with profile-based routing
   -  Clean command convention: Main (0-9), Helpers (a-z)
   -  Profile hierarchy: test < med < prod (DRY principle)
   -  User documentation: `CLI_REFERENCE.md`
   -  Developer instructions: `.github/instructions/cli.instructions.md`
   -  Backward compatibility maintained for legacy commands

### Immediate (2025-11 to 2025-12)
-  **Pure NSGA-II Baseline (prod-nsga)**: Production run in progress for thesis experiments
-  Complete Phase 2 execution (training, validation, deployment) - **AFTER baseline completes**
-  Baseline comparisons (RL vs non-RL) - **Prioritize pure NSGA-II baseline first**
-  Document Phase 2 empirical results

### Q1 2026 (Jan-Mar)
-  Phase 3 Tier 1: Quick wins (#2, #1, #6)
-  Target: 30% improvement
-  Go/No-Go decision at end of Q1

### Q2-Q3 2026 (Apr-Sep)
-  Phase 3 Tier 2: Advanced techniques (#4, #3, #5, #7)
-  Target: 40% cumulative improvement
-  Go/No-Go decision at end of Q3

### Q4 2026+ (Oct onwards)
-  Phase 3 Tier 3: Research extensions (#8, #9, #10) - OPTIONAL
-  Focus: Research contributions and publications
-  Depends on: Research goals, team availability, compute resources

---

## SUCCESS CRITERIA

### Phase 2 Execution Success (Immediate)
-  Curriculum training completes without errors
-  Model converges on all curriculum stages
-  Best checkpoint selected and promoted
-  RL-enabled GA runs in production
-  Baseline comparison shows promise (may not exceed baseline yet)

### Phase 3 Tier 1 Success (Q1 2026)
-  20%+ improvement in key metrics (fitness, convergence, diversity)
-  No regression on baseline benchmarks
-  Computational overhead <10%
-  All enhancements tested and validated

### Phase 3 Tier 2 Success (Q3 2026)
-  40%+ cumulative improvement
-  Production stability maintained
-  Positive ROI on development time
-  System remains maintainable and documented

### Phase 3 Tier 3 Success (Optional)
-  Publishable research results
-  Novel contributions to field
-  Generalization to other problem types demonstrated

---

**Last Updated**: 2025-11-17  
**Document Owner**: Krishna  
**Project Status**:  Phase 1 Complete |  Phase 2 Code Complete (Execution Pending) |  Phase 3 Planned
