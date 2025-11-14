# RL-Based Hyper-Heuristic Implementation TODO

**Project**: Evolving Schedule Engine from Pure GA to RL-Powered Hyper-Heuristic
**Date Created**: 2025-11-14
**Status**: Planning Phase

---

## PHASE 1: LNS-CP Hybrid Foundation (4-6 weeks)

**Goal**: Prove that integrating CP-SAT as a targeted repair tool improves baseline GA

### 1.1 Environment Setup
- [ ] Install `ortools` library: `uv add ortools`
- [ ] Install `pytorch` library: `uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
- [ ] Verify installations with test imports
- [ ] Create new module structure: `src/lns/` directory

### 1.2 Conflict Detection System
- [ ] Create `src/lns/__init__.py`
- [ ] Create `src/lns/conflict_detection.py`
- [ ] Implement `find_hard_conflict_sessions()` function
  - Extract hard constraint violations from evaluation
  - Map violations to session IDs
  - Return list of conflicted `SessionGene` objects
- [ ] Write unit tests for conflict detection (test with known-bad individuals)
- [ ] Test with current dataset to verify correct identification

### 1.3 CP-SAT Subproblem Solver
- [ ] Create `src/lns/cp_repair.py`
- [ ] Implement basic CP model creation:
  - [ ] Create variables for session start times
  - [ ] Create variables for room assignments
  - [ ] Define domains from available resources
- [ ] Implement internal constraints (conflicts among repaired sessions):
  - [ ] Instructor temporal conflicts
  - [ ] Room temporal conflicts
  - [ ] Student group conflicts
  - [ ] Use `NoOverlap` global constraints
- [ ] Implement constraints against fixed schedule:
  - [ ] No conflicts with partial schedule sessions
  - [ ] Respect already-assigned resources
- [ ] Add soft constraint optimization to objective
- [ ] Add 10-second time limit to solver
- [ ] Test on small subproblems (5 sessions)

### 1.4 LNS Framework
- [ ] Create `src/lns/lns_operator.py`
- [ ] Implement `LNS_CP_Repair()` function (Algorithm 2 from suggestion.md):
  - [ ] Destroy phase: Extract conflicted sessions
  - [ ] Handle case where conflicts > 20 (select worst 20)
  - [ ] Create partial schedule (remove conflicted sessions)
  - [ ] Call CP-SAT repair on subproblem
  - [ ] Reintegrate solution if feasible
  - [ ] Return original schedule if CP-SAT fails
- [ ] Add logging for LNS operations (track success/failure rates)
- [ ] Test end-to-end on full individuals with known conflicts

### 1.5 GA Integration
- [ ] Modify `src/core/ga_scheduler.py`:
  - [ ] Add LNS-CP repair trigger logic (every 50 generations OR stagnation)
  - [ ] Track stagnation counter
  - [ ] Apply LNS-CP to best individual when triggered
  - [ ] Log repair attempts and outcomes
- [ ] Add configuration parameters to `configs/base.yaml`:
  - [ ] `lns.enabled: true/false`
  - [ ] `lns.trigger_interval: 50`
  - [ ] `lns.max_subproblem_size: 20`
  - [ ] `lns.cp_time_limit: 10.0`

### 1.6 Benchmarking & Evaluation
- [ ] Create benchmarking script: `scripts/benchmark_lns_cp.py`
- [ ] Run baseline GA (without LNS-CP) on test dataset
  - [ ] Record: final fitness, hard violations, soft violations, runtime
  - [ ] Save results to `output/benchmark_baseline/`
- [ ] Run GA+LNS-CP hybrid on same dataset
  - [ ] Record: same metrics + LNS success rate
  - [ ] Save results to `output/benchmark_lns_cp/`
- [ ] Generate comparison report:
  - [ ] Time to first feasible solution
  - [ ] Final solution quality
  - [ ] Convergence curves
  - [ ] Statistical significance tests
- [ ] Document findings in `docs/for_report/PHASE1_LNS_CP_RESULTS.md`

---

## PHASE 2: RL Environment Foundation (3-4 weeks)

**Goal**: Build the hyper-heuristic framework and infrastructure for learning agents

### 2.1 Environment Architecture
- [ ] Create `src/rl/` directory structure:
  - [ ] `src/rl/__init__.py`
  - [ ] `src/rl/environment.py`
  - [ ] `src/rl/state.py`
  - [ ] `src/rl/reward.py`
  - [ ] `src/rl/actions.py`

### 2.2 State Representation
- [ ] Create `src/rl/state.py`
- [ ] Implement `StateCalculator` class:
  - [ ] `norm_hard_violations`: Calculate and normalize
  - [ ] `norm_soft_violations`: Calculate and normalize
  - [ ] `fitness_delta`: Track improvement from previous iteration
  - [ ] `norm_stagnation`: Iterations since improvement / 100
  - [ ] `progress`: Current iteration / max iterations
- [ ] Implement `get_state_vector()` → returns 5D numpy array
- [ ] Add normalization utilities for robust scaling
- [ ] Write tests for state calculation edge cases

### 2.3 Heuristic Toolbox
- [ ] Create `src/rl/actions.py`
- [ ] Define `Heuristic` base class with interface:
  - [ ] `apply(individual, **kwargs) → Individual`
  - [ ] `cost: float` (computational cost estimate)
  - [ ] `needs_partner: bool` (for crossover)
- [ ] Implement 6 core heuristics:
  - [ ] **Action 0**: `MutateSessionTime` (low intensity, ~1ms)
  - [ ] **Action 1**: `MutateSessionRoom` (low intensity, ~5ms)
  - [ ] **Action 2**: `CrossoverOnePoint` (medium, ~5ms)
  - [ ] **Action 3**: `LNS_DestroyRandom10Pct` (medium, ~50ms)
  - [ ] **Action 4**: `LNS_DestroyConflicted` (high, ~100ms)
  - [ ] **Action 5**: `LNS_CP_Repair` (very high, ~500ms-10s)
- [ ] Create `HEURISTIC_TOOLBOX` registry (list of heuristic instances)
- [ ] Add cost tracking for each action

### 2.4 Reward Function
- [ ] Create `src/rl/reward.py`
- [ ] Implement `calculate_reward()` function:
  - [ ] Fitness improvement component (primary)
  - [ ] Action cost penalty (efficiency incentive)
  - [ ] Hard constraint bonus (strategic priority)
- [ ] Add reward logging for analysis
- [ ] Create reward visualization utilities
- [ ] Write tests for reward edge cases (no change, degradation, etc.)

### 2.5 Timetabling Environment Class
- [ ] Create `src/rl/environment.py`
- [ ] Implement `TimetablingEnvironment` class:
  - [ ] `__init__(initial_schedule)`: Initialize with starting individual
  - [ ] `get_state_vector(iteration, max_iter)`: Return current state
  - [ ] `apply_heuristic(action_id, **kwargs)`: Execute action, update state
  - [ ] `get_fitness()`: Return current fitness tuple
  - [ ] `evaluate()`: Run full constraint evaluation
  - [ ] `reset()`: Reset to initial state
- [ ] Add archive management for crossover partners
- [ ] Implement observation history tracking
- [ ] Add logging for environment transitions

### 2.6 Random Baseline Agent
- [ ] Create `src/rl/agents/random_agent.py`
- [ ] Implement `RandomAgent` class:
  - [ ] `choose_action(state)`: Uniform random selection
  - [ ] Track action distribution for analysis
- [ ] Integrate with environment for testing
- [ ] Run random agent baseline on test dataset
- [ ] Document random baseline performance

### 2.7 RL Loop Infrastructure
- [ ] Create `src/rl/hyper_heuristic_loop.py`
- [ ] Implement main loop skeleton (Algorithm 1):
  - [ ] Initialize environment and agent
  - [ ] Initialize solution archive
  - [ ] Main iteration loop structure
  - [ ] State observation
  - [ ] Action selection
  - [ ] Heuristic application
  - [ ] Reward calculation
  - [ ] Best solution tracking
- [ ] Add comprehensive logging (state, action, reward per iteration)
- [ ] Add checkpoint saving (save best solutions periodically)
- [ ] Test loop with random agent

### 2.8 Configuration Integration
- [ ] Add RL parameters to `configs/base.yaml`:
  - [ ] `rl.enabled: false` (default off)
  - [ ] `rl.state_size: 5`
  - [ ] `rl.action_size: 6`
  - [ ] `rl.archive_size: 10`
  - [ ] `rl.learning_rate: 0.0001`
  - [ ] `rl.epsilon_start: 1.0`
  - [ ] `rl.epsilon_decay: 0.995`
  - [ ] `rl.epsilon_min: 0.05`
- [ ] Update config models in `src/config/`

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
- [ ] Document LNS-CP integration:
  - [ ] Algorithm 2 from suggestion.md (with pseudocode)
  - [ ] CP-SAT subproblem formulation
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
  - [ ] Successful integration of CP-SAT in LNS context
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
- [ ] Celebrate successful defense! 🎓

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
- [ ] Study CP-SAT documentation (OR-Tools)
- [ ] Participate in relevant online communities (Stack Overflow, Reddit r/MachineLearning)

---

## Risk Mitigation

### High-Risk Items (Monitor Closely)
1. **RL Training Instability**: DQN may not converge
   - **Mitigation**: Start with simple problems, tune hyperparameters carefully, use baseline agents as fallback

2. **CP-SAT Still Intractable on Subproblems**: Even small subproblems may fail
   - **Mitigation**: Reduce subproblem size further (5 sessions max), add strict time limits, have greedy fallback

3. **Computational Budget**: Training RL agents is expensive
   - **Mitigation**: Use cloud computing (AWS/Google Cloud), optimize code, parallelize training

4. **Time Constraints**: Thesis deadline approaching
   - **Mitigation**: Prioritize core phases (1-3), treat phases 4-5 as optional enhancements

### Contingency Plans
- **If RL completely fails**: Fall back to GA+LNS-CP hybrid as main contribution (still novel)
- **If CP-SAT integration fails**: Focus on pure RL hyper-heuristic with GA operators only
- **If thesis deadline is tight**: Cut optional phases, focus on strong evaluation of core system

---

## Success Metrics

### Minimum Viable Thesis (Pass Threshold)
- ✅ Working GA+LNS-CP hybrid outperforms baseline
- ✅ Functional RL environment and basic agent
- ✅ Comparative evaluation showing some improvement
- ✅ Complete thesis document with all chapters

### Strong Thesis (High Grade)
- ✅ DQN agent learns meaningful policy
- ✅ Demonstrated adaptive behavior (agent changes strategy based on state)
- ✅ Comprehensive benchmarking with statistical validation
- ✅ Published/submitted conference paper

### Outstanding Thesis (Top Grade + Publication)
- ✅ DQN agent significantly outperforms all baselines
- ✅ Transfer learning success (generalization across instances)
- ✅ Rigorous ablation studies
- ✅ Accepted publication in top-tier venue
- ✅ Potential for journal extension

---

## Timeline Summary

| Phase | Duration | Key Deliverable | Status |
|-------|----------|----------------|--------|
| Phase 1: LNS-CP | 4-6 weeks | GA+LNS-CP hybrid system | ⬜ Not Started |
| Phase 2: RL Env | 3-4 weeks | Working RL environment + random agent | ⬜ Not Started |
| Phase 3: DQN | 4-6 weeks | Trained DQN agent | ⬜ Not Started |
| Phase 4: Advanced RL | 3-4 weeks | Enhanced RL techniques (optional) | ⬜ Not Started |
| Phase 5: More Heuristics | 2-3 weeks | Expanded action space (optional) | ⬜ Not Started |
| Phase 6: Evaluation | 6-8 weeks | Comprehensive experimental results | ⬜ Not Started |
| Phase 7: Thesis | 6-8 weeks | Complete thesis manuscript | ⬜ Not Started |
| Phase 8: Publication | 4-6 weeks | Submitted paper (parallel with Phase 7) | ⬜ Not Started |
| Phase 9: Defense | 3-4 weeks | Defense presentation + successful defense | ⬜ Not Started |

**Total Estimated Time**: 6-9 months (depending on optional phases)

---

## Notes

- **Priority Order**: Phases 1 → 2 → 3 → 6 → 7 → 9 (skip 4-5 if time-constrained)
- **Current Status**: Planning phase complete, ready to start Phase 1
- **Next Immediate Action**: Install `ortools` and implement `find_hard_conflict_sessions()`
- **Weekly Goal**: Complete at least 3-5 tasks per week
- **Advisor Check-ins**: Every 2 weeks, present progress and get feedback

---

**Last Updated**: 2025-11-14
**Document Owner**: Krishna
**Project Status**: 🟡 Planning Complete, Implementation Starting
