# Research Papers Index

This document lists the key academic papers that form the theoretical foundation of the Schedule Engine project.

---

## Core Algorithms

### 1. NSGA-II (Multi-Objective Genetic Algorithm)

**Paper:** Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II.* IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

**DOI:** [10.1109/4235.996017](https://doi.org/10.1109/4235.996017)

**Key Contributions:**
- Fast non-dominated sorting (O(MN²))
- Crowding distance for diversity preservation
- Elitist selection strategy
- Binary tournament selection

**Relevance to Schedule Engine:**
- Core optimization algorithm
- Two-objective minimization (hard violations, soft penalty)
- Pareto front approximation
- Elite preservation (top 10%)

**Implementation:**
- `src/core/ga_scheduler.py` - Main evolution loop
- Uses DEAP library's `tools.selNSGA2()`
- Custom course-group-aware crossover

---

### 2. Curriculum Timetabling (Survey)

**Paper:** Pillay, N. (2016). *A survey of school timetabling research.* Annals of Operations Research, 218(1), 261-293.

**DOI:** [10.1007/s10479-013-1321-8](https://doi.org/10.1007/s10479-013-1321-8)

**Key Contributions:**
- Comprehensive survey of timetabling techniques
- Classification of constraints (hard vs soft)
- Benchmark datasets (ITC 2007, 2011)
- Performance metrics

**Relevance to Schedule Engine:**
- Problem formulation (14 hard + 4 soft constraints)
- Constraint classification system
- Validation approach

**Implementation:**
- `src/constraints/` - All constraint functions
- `src/validation/feasibility_checker.py` - Feasibility analysis

---

### 3. Hyper-Heuristics

**Paper:** Burke, E. K., Hyde, M., Kendall, G., Ochoa, G., Özcan, E., & Woodward, J. R. (2010). *A classification of hyper-heuristic approaches.* In Handbook of Metaheuristics (pp. 449-468). Springer.

**DOI:** [10.1007/978-1-4419-1665-5_15](https://doi.org/10.1007/978-1-4419-1665-5_15)

**Key Contributions:**
- Classification: Selection vs generation hyper-heuristics
- Low-level heuristic framework
- Online vs offline learning

**Relevance to Schedule Engine:**
- Heuristic toolbox design (19 operators)
- Selection hyper-heuristic (RL-guided)
- Operator categories (construction, perturbation, repair, optimization, diversity)

**Implementation:**
- `src/heuristics/registry.py` - Heuristic registry
- `src/rl/gym_env/action_mapper.py` - Heuristic selection

---

### 4. Deep Reinforcement Learning for Combinatorial Optimization

**Paper:** Bengio, Y., Lodi, A., & Prouvost, A. (2021). *Machine learning for combinatorial optimization: a methodological tour d'horizon.* European Journal of Operational Research, 290(2), 405-421.

**DOI:** [10.1016/j.ejor.2020.07.063](https://doi.org/10.1016/j.ejor.2020.07.063)

**Key Contributions:**
- ML paradigms for CO problems
- End-to-end learning vs policy-based learning
- Construction vs improvement approaches

**Relevance to Schedule Engine:**
- RL-guided heuristic selection
- Policy-based learning (PPO agent)
- Improvement approach (RL selects operators for existing solutions)

**Implementation:**
- `src/rl/gym_env/schedule_env.py` - RL environment
- `src/rl/agents/ppo_agent.py` - PPO agent
- `src/rl/training/` - Training infrastructure

---

### 5. PPO (Proximal Policy Optimization)

**Paper:** Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal policy optimization algorithms.* arXiv preprint arXiv:1707.06347.

**arXiv:** [1707.06347](https://arxiv.org/abs/1707.06347)

**Key Contributions:**
- Clipped surrogate objective
- Stable policy updates
- Sample efficiency
- Easy to implement

**Relevance to Schedule Engine:**
- Primary RL agent (PPO)
- Stable training for heuristic selection
- Multi-component reward function

**Implementation:**
- Uses Stable-Baselines3 PPO implementation
- `src/rl/agents/ppo_agent.py` - Wrapper class
- `src/rl/training/train_script.py` - Training script

---

## Related Work

### 6. Large Neighborhood Search

**Paper:** Pisinger, D., & Ropke, S. (2019). *Large neighborhood search.* In Handbook of Metaheuristics (pp. 99-127). Springer.

**DOI:** [10.1007/978-3-319-91086-4_4](https://doi.org/10.1007/978-3-319-91086-4_4)

**Key Contributions:**
- Destroy-repair framework
- Adaptive large neighborhood search (ALNS)
- Operator selection strategies

**Relevance to Schedule Engine:**
- IGLS repair system inspiration
- Destroy-repair cycle for violated sessions
- Adaptive operator selection (RL-guided)

**Implementation:**
- `src/ga/operators/repair_igls.py` - IGLS repair
- `src/lns/` directory (if LNS-CP was integrated)

---

### 7. Curriculum Learning for RL

**Paper:** Narvekar, S., Peng, B., Leonetti, M., Sinapov, J., Taylor, M. E., & Stone, P. (2020). *Curriculum learning for reinforcement learning domains: A framework and survey.* Journal of Machine Learning Research, 21(181), 1-50.

**arXiv:** [2003.04960](https://arxiv.org/abs/2003.04960)

**Key Contributions:**
- Task progression (easy → hard)
- Transfer learning across tasks
- Performance evaluation

**Relevance to Schedule Engine:**
- Curriculum training (3 stages: 10 → 20 → 40 courses)
- Improved RL convergence
- Better generalization

**Implementation:**
- `src/rl/training/curriculum.py` - CurriculumManager
- Stages: easy (10 courses), medium (20), hard (40+)

---

### 8. Multi-Objective Optimization

**Paper:** Zitzler, E., Laumanns, M., & Thiele, L. (2001). *SPEA2: Improving the strength Pareto evolutionary algorithm.* TIK-Report, 103.

**DOI:** [10.3929/ethz-a-004284029](https://doi.org/10.3929/ethz-a-004284029)

**Key Contributions:**
- Pareto dominance
- Archive-based selection
- Fitness assignment
- Diversity preservation

**Relevance to Schedule Engine:**
- Two-objective optimization (hard, soft)
- Pareto front approximation
- Behavioral archive (Phase 3 enhancement)

**Implementation:**
- Uses NSGA-II (similar to SPEA2)
- `src/diversity/archive.py` - Behavioral archive (Phase 3)

---

### 9. GPU-Accelerated Genetic Algorithms

**Paper:** Pospichal, P., Jaros, J., & Schwarz, J. (2010). *Parallel genetic algorithm on the CUDA architecture.* In European Conference on the Applications of Evolutionary Computation (pp. 442-451). Springer.

**DOI:** [10.1007/978-3-642-12239-2_46](https://doi.org/10.1007/978-3-642-12239-2_46)

**Key Contributions:**
- GPU-parallel fitness evaluation
- Batch processing on CUDA
- Memory management strategies

**Relevance to Schedule Engine:**
- GPU batch constraint evaluation
- 10-50x speedup
- PyTorch CUDA implementation

**Implementation:**
- `src/ga/evaluator/gpu_batch_evaluator.py` - GPU evaluator
- Batch size: 100 (configurable)
- Automatic CPU fallback

---

### 10. Constraint Satisfaction for Timetabling

**Paper:** Schaerf, A. (1999). *A survey of automated timetabling.* Artificial Intelligence Review, 13(2), 87-127.

**DOI:** [10.1023/A:1006576209967](https://doi.org/10.1023/A:1006576209967)

**Key Contributions:**
- Constraint classification
- Solution quality metrics
- Heuristic approaches

**Relevance to Schedule Engine:**
- Constraint design (14 hard, 4 soft)
- Feasibility checking
- Quality metrics (hard violations, soft penalty)

**Implementation:**
- `src/constraints/` - All constraints
- `src/validation/feasibility_checker.py` - Pre-flight checks

---

## Methodology Mapping

### Problem Formulation
**Based on:** Pillay (2016), Schaerf (1999)
- **Constraint classification:** Hard (must-satisfy), Soft (preferences)
- **Entities:** Course, Group, Instructor, Room, Time
- **Objective:** Minimize (hard_violations, soft_penalty)

### Optimization Algorithm
**Based on:** Deb et al. (2002) - NSGA-II
- **Selection:** Non-dominated sorting + crowding distance
- **Crossover:** Course-group-aware (custom)
- **Mutation:** Constraint-guided (custom)
- **Elitism:** Top 10% preserved

### Hyper-Heuristic
**Based on:** Burke et al. (2010)
- **Type:** Selection hyper-heuristic
- **Low-level heuristics:** 19 operators (5 categories)
- **Selection strategy:** RL-guided (PPO agent)

### Reinforcement Learning
**Based on:** Bengio et al. (2021), Schulman et al. (2017)
- **Agent:** PPO (Stable-Baselines3)
- **State:** 25D population metrics + constraint breakdown
- **Action:** 20 discrete (19 heuristics + no-op)
- **Reward:** Multi-component (fitness improvement + diversity - time)

### Local Search Repair
**Based on:** Pisinger & Ropke (2019)
- **Framework:** Destroy-repair (LNS-inspired)
- **Implementation:** IGLS (Iterative Greedy Local Search)
- **Trigger:** Stagnation detection + periodic

### GPU Acceleration
**Based on:** Pospichal et al. (2010)
- **Parallelization:** Batch constraint evaluation
- **Framework:** PyTorch CUDA
- **Speedup:** 10-50x vs CPU

### Curriculum Learning
**Based on:** Narvekar et al. (2020)
- **Stages:** 3 stages (easy → medium → hard)
- **Progression:** 10 → 20 → 40 courses
- **Benefit:** Better convergence + generalization

---

## Citing Schedule Engine

If you use this software in academic work, please cite:

```bibtex
@software{schedule_engine_2025,
  author = {Acharya, Krishna and Padhya, Dinanath and Dahal, Bipul},
  title = {Schedule Engine: RL-Enhanced Genetic Algorithm for University Timetabling},
  year = {2025},
  url = {https://github.com/krishna-ji/schedule-engine},
  version = {2.0.0}
}
```

---

## Additional Resources

### Relevant Conferences
- **PATAT** - Practice and Theory of Automated Timetabling
- **GECCO** - Genetic and Evolutionary Computation Conference
- **EvoCOP** - European Conference on Evolutionary Computation in Combinatorial Optimization
- **ICAPS** - International Conference on Automated Planning and Scheduling

### Relevant Journals
- *IEEE Transactions on Evolutionary Computation*
- *European Journal of Operational Research*
- *Journal of Scheduling*
- *Computers & Operations Research*
- *Artificial Intelligence Review*

### Benchmark Datasets
- **ITC 2007** - International Timetabling Competition 2007
- **ITC 2011** - International Timetabling Competition 2011
- **Carter's Datasets** - Classic timetabling benchmarks

### Online Resources
- [PATAT Conference Series](http://www.patatconference.org/)
- [ITC Website](http://www.cs.qub.ac.uk/itc2007/)
- [Hyper-Heuristics.org](http://www.asap.cs.nott.ac.uk/external/hyper-heuristics/)

---

## See Also

- [NSGA-II Algorithm Reference](01-nsga-ii-algorithm.md) - Detailed NSGA-II documentation
- [Architecture](../architecture/01-high-level-architecture.md) - System design based on these papers
- [References](../references/) - Algorithm and library documentation

---

**Last Updated:** November 20, 2025  
**Status:** Complete (Core papers documented)
