# Minimized Experiment Plan

## **Core Issue Resolution**

You're absolutely correct about the conceptual problems:

1. **IGLS/LNS ARE heuristics** (repair category)
2. **All repair operators ARE heuristics** (construction/perturbation/improvement)  
3. **Mode 4 vs 5 redundancy** - both involve heuristic application
4. **Scattered organization** - repairs should be unified in heuristic toolbox

## **Proposed Clean Architecture**

### **Unified Heuristic Taxonomy** 
```
src/heuristics/
├── construction/     # 3 operators (greedy builders)
├── perturbation/     # 5 operators (destroy/disturb)  
├── improvement/      # 3 operators (local search)
├── diversity/        # 4 operators (exploration)
├── meta/            # 4 operators (high-level strategies)
└── repair/          # IGLS, LNS, selective repair (move here!)
```

### **Cleaner Runtime Modes (6 Total)**

#### **Mode 1: Pure NSGA-II**
- **Components**: Selection + Crossover + Mutation only
- **Heuristics**: NONE
- **Purpose**: Clean baseline for comparison

#### **Mode 2: NSGA-II + Construction Heuristics**  
- **Components**: Mode 1 + Construction heuristics (3 operators)
- **Heuristics**: Largest-degree-first, most-constrained-first, earliest-deadline-first
- **Purpose**: Test greedy initialization impact

#### **Mode 3: NSGA-II + Basic Heuristics**
- **Components**: Mode 2 + Perturbation + Improvement heuristics (8 total)
- **Heuristics**: Construction (3) + Perturbation (5) + Improvement (3)  
- **Purpose**: Traditional heuristic enhancement

#### **Mode 4: NSGA-II + Full Heuristics**
- **Components**: Mode 3 + Diversity + Meta + Repair heuristics (19 total)
- **Heuristics**: All categories including IGLS/LNS repair
- **Purpose**: Complete heuristic toolbox

#### **Mode 5: Hyper-Heuristic (Round-Robin)**
- **Components**: Mode 4 + Fixed heuristic rotation
- **Selection**: Deterministic round-robin through all 19 heuristics
- **Purpose**: Systematic heuristic application baseline

#### **Mode 6: Hyper-Heuristic (RL-Guided)**
- **Components**: Mode 4 + RL agent for heuristic selection  
- **Selection**: PPO/DQN learns optimal heuristic timing
- **Purpose**: Intelligent adaptive heuristic selection

## **Minimized Experiment Framework**

### **Experiment Group A: Progressive Enhancement (6 Experiments)**

#### **A1: Pure NSGA-II Baseline**
```yaml
# configs/experiments/A1-pure-nsga.yaml
ga:
  ngen: 1000
  pop_size: 200
  cxpb: 0.75
  mutpb: 0.25

heuristics:
  enabled: false  # NO heuristics

repair:  
  enabled: false  # NO repair

rl:
  enabled: false  # NO RL
```

#### **A2: Construction Heuristics**
```yaml
# configs/experiments/A2-construction.yaml
extends: A1-pure-nsga.yaml

heuristics:
  enabled: true
  construction:
    largest_degree_first: {enabled: true}
    most_constrained_first: {enabled: true} 
    earliest_deadline_first: {enabled: true}
  # All other categories: disabled
```

#### **A3: Basic Heuristics (Construction + Perturbation + Improvement)**
```yaml
# configs/experiments/A3-basic-heuristics.yaml  
extends: A2-construction.yaml

heuristics:
  perturbation:
    random_swap: {enabled: true}
    temporal_shift: {enabled: true}
    room_shuffle: {enabled: true}
    instructor_reassign: {enabled: true}
    multi_perturbation: {enabled: true}
  improvement:
    kempe_chain: {enabled: true}
    ejection_chain: {enabled: true}
    variable_depth_search: {enabled: true}
```

#### **A4: Full Heuristics (All 19 Operators)**
```yaml
# configs/experiments/A4-full-heuristics.yaml
extends: A3-basic-heuristics.yaml

heuristics:
  diversity:
    distance_preserving_crossover: {enabled: true}
    crowding_mutation: {enabled: true}
    niching_selection: {enabled: true}
    adaptive_diversity_maintenance: {enabled: true}
  meta:
    variable_neighborhood_descent: {enabled: true}
    iterated_local_search: {enabled: true}
    adaptive_large_neighborhood: {enabled: true}
    guided_local_search: {enabled: true}
  repair:  # MOVED from repair: to heuristics:
    igls_repair: {enabled: true}
    lns_repair: {enabled: true}
    selective_repair: {enabled: true}
```

#### **A5: Round-Robin Hyper-Heuristic**
```yaml
# configs/experiments/A5-roundrobin.yaml
extends: A4-full-heuristics.yaml

heuristics:
  selection_strategy: "round_robin"  # Fixed rotation
  rotation_interval: 10  # Switch every 10 generations
```

#### **A6: RL-Guided Hyper-Heuristic**  
```yaml
# configs/experiments/A6-rl-guided.yaml
extends: A4-full-heuristics.yaml

heuristics:
  selection_strategy: "rl_guided"  # RL agent decides

rl:
  enabled: true
  agent_type: "ppo"
  model_path: "output/models/rl_agents/heuristic_selector.zip"
```

### **Experiment Group B: Algorithm Component Analysis (5 Experiments)**

#### **B1: Population Initialization Impact**
- **B1.1**: Random initialization only
- **B1.2**: Greedy initialization only  
- **B1.3**: Smart initialization only
- **B1.4**: Hybrid initialization (current default)

#### **B2: Crossover Operator Comparison**
- **B2.1**: Standard uniform crossover
- **B2.2**: Course-group-aware crossover (current)
- **B2.3**: Distance-preserving crossover

#### **B3: Mutation Strategy Analysis**  
- **B3.1**: Random mutation only
- **B3.2**: Constraint-guided mutation (current)
- **B3.3**: Crowding mutation (diversity-focused)

#### **B4: Selection Mechanism Comparison**
- **B4.1**: Tournament selection  
- **B4.2**: NSGA-II selection (current)
- **B4.3**: Fast NSGA-II selection

#### **B5: Multi-Objective Weight Sensitivity**
- **B5.1**: Weights (-1.0, -0.001) - prioritize hard constraints
- **B5.2**: Weights (-1.0, -0.01) - current default
- **B5.3**: Weights (-1.0, -0.1) - balance objectives  
- **B5.4**: Weights (-1.0, -1.0) - equal objectives

### **Experiment Group C: Scalability & Performance (3 Experiments)**

#### **C1: Problem Size Scaling**
- **C1.1**: Small (50 courses, 10 rooms, 20 instructors)
- **C1.2**: Medium (100 courses, 20 rooms, 40 instructors)  
- **C1.3**: Large (200 courses, 40 rooms, 80 instructors)

#### **C2: Population & Generation Scaling**  
- **C2.1**: Quick test (pop=50, gen=200)
- **C2.2**: Standard (pop=200, gen=1000) 
- **C2.3**: Production (pop=500, gen=2000)

#### **C3: Computational Mode Comparison**
- **C3.1**: CPU sequential evaluation
- **C3.2**: CPU parallel evaluation (32 cores)
- **C3.3**: GPU batch evaluation (if available)

## **Execution Plan**

### **Phase 1: Core Algorithm Progression (A1-A6)**
*Duration: 1 week*
- **Purpose**: Establish the value of each algorithmic enhancement
- **Runs**: 5 runs per experiment (30 total runs)
- **Expected outcome**: Clear performance progression A1 < A2 < A3 < A4 < A5 ≤ A6

### **Phase 2: Component Analysis (B1-B5)**  
*Duration: 1 week*
- **Purpose**: Optimize individual algorithm components
- **Runs**: 3 runs per sub-experiment (45 total runs)
- **Expected outcome**: Optimal configuration for each component

### **Phase 3: Scalability Assessment (C1-C3)**
*Duration: 3 days*  
- **Purpose**: Validate performance across scales and modes
- **Runs**: 2 runs per sub-experiment (18 total runs)
- **Expected outcome**: Scalability characteristics and computational bottlenecks

## **Success Metrics**

### **Primary Metrics** (All Experiments)
1. **Hard Constraint Violations**: Must reach 0 for feasible solution
2. **Soft Constraint Penalty**: Lower is better (quality metric)
3. **Convergence Speed**: Generations to reach best solution
4. **Runtime**: Wall-clock time to completion

### **Secondary Metrics** (Groups A & B)
5. **Solution Diversity**: Population diversity over time
6. **Hypervolume**: Multi-objective solution quality (NSGA-II metric)
7. **Success Rate**: Percentage of runs finding feasible solutions

## **Expected Research Outcomes**

### **Key Research Questions Answered**
1. **What is the incremental value of each algorithmic enhancement?** (Group A)
2. **Which algorithm components are most critical for performance?** (Group B)  
3. **How does the system scale with problem size and computational resources?** (Group C)

### **Practical Recommendations**
1. **Optimal production configuration** for real-world deployment
2. **Algorithm selection guidelines** based on problem characteristics  
3. **Computational resource recommendations** (CPU vs GPU, population size)
4. **Heuristic integration best practices** for timetabling problems

## **Implementation Recommendations**

### **Immediate Actions**
1. **Reorganize heuristic system**: Move IGLS/LNS/repairs into `src/heuristics/repair/`
2. **Unify heuristic registry**: Single registry managing all 19 operators across 6 categories
3. **Simplify runtime modes**: Reduce from 10 to 6 cleaner modes  
4. **Create experiment configs**: 14 YAML files for systematic testing

### **Architecture Cleanup**  
```python
# New unified heuristic structure
src/heuristics/
├── __init__.py           # Unified registry
├── construction/         # 3 operators
├── perturbation/         # 5 operators  
├── improvement/          # 3 operators
├── diversity/           # 4 operators
├── meta/               # 4 operators
└── repair/             # IGLS, LNS, selective (moved from src/ga/operators/)
```

This minimized plan gives you **14 focused experiments** that systematically evaluate your algorithm progression while resolving the conceptual inconsistencies in the current runtime mode organization.