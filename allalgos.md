


Searched codebase for "NSGA-II selection elitism crowding distance tournament selection", 18 results

Searched codebase for "constraint evaluation hard constraints soft constraints validation feasibility", 19 results

Based on my analysis of your codebase, here's a comprehensive categorization of all implemented algorithms, organized by increasing complexity:

## **Algorithm Inventory: Schedule-Engine**

### **1. Basic Genetic Algorithms (Complexity: Low)**

#### **Population Initialization**
- **Random Initialization**: Pure random chromosome generation
- **Greedy Initialization**: Largest-degree-first construction heuristic
- **Constraint-Guided Smart Initialization**: Biased toward feasible regions
- **Hybrid Initialization**: 25% greedy, 50% smart, 25% random

#### **Selection Operators**
- **Tournament Selection**: Binary tournament (DEAP default)
- **NSGA-II Selection**: Non-dominated sorting with crowding distance
- **Fast NSGA-II Selection**: Optimized O(N log^(M-1) N) implementation
- **Elitist Selection**: Top 10% preservation

### **2. Genetic Operators (Complexity: Low-Medium)**

#### **Crossover**
- **Course-Group-Aware Crossover**: Position-independent crossover preserving enrollment structure
- **Distance-Preserving Crossover**: Maintains phenotypic distance between parents

#### **Mutation**
- **Random Mutation**: Traditional uniform mutation
- **Constraint-Guided Mutation**: Targets violation-prone genes
- **Crowding Mutation**: Favors less-explored search space regions
- **Gene-Level Mutation**: Time/room/instructor-specific mutations

### **3. Multi-Objective Optimization (Complexity: Medium)**

#### **NSGA-II Implementation**
- **Fast Non-Dominated Sorting**: O(MN²) complexity
- **Crowding Distance Assignment**: Diversity preservation
- **Pareto Front Approximation**: Two-objective minimization (-hard_violations, -soft_penalty)
- **Lexicographic Multi-Objective**: Weights (-1.0, -0.01)

### **4. Repair & Local Search (Complexity: Medium-High)**

#### **IGLS (Iterative Greedy Local Search)**
- **Stagnation-Triggered Repair**: Patience-based activation
- **Violation Detection**: Identifies constraint-violating sessions
- **Destroy-Repair Cycle**: Removes conflicts, rebuilds with heuristics
- **Timeout Guards**: Prevents repair from stalling evolution

#### **LNS (Large Neighborhood Search)**
- **Conflict Graph Construction**: BFS neighborhood expansion
- **Subproblem Extraction**: Isolates violating session clusters
- **IGLS Integration**: Uses IGLS as subproblem solver
- **Reintegration**: Merges repaired solutions back to population

#### **Selective Repair System**
- **Fast Violation Detection**: Hybrid strategy (fast/full/hybrid)
- **Targeted Repair**: Applies repair only to violating individuals
- **Post-Operator Repair**: Optional repair after crossover/mutation

### **5. Heuristic Operators (19 Total) (Complexity: Medium)**

#### **Construction Heuristics (3)**
1. **Largest Degree First**: Schedule most-constrained courses first
2. **Most Constrained First**: Prioritize by constraint density
3. **Earliest Deadline First**: Time-critical course prioritization

#### **Perturbation Heuristics (5)**
4. **Random Swap**: Random time/room/instructor exchanges  
5. **Temporal Shift**: Move sessions in time dimension
6. **Room Shuffle**: Reassign rooms while preserving time
7. **Instructor Reassign**: Qualified instructor swapping
8. **Multi-Perturbation**: Combined perturbation strategies

#### **Improvement Heuristics (3)**
9. **Kempe Chain**: Graph-based conflict resolution (max 5 iterations)
10. **Ejection Chain**: Sequential improvement moves (max chain length 5)
11. **Variable Depth Search**: Multi-level neighborhood exploration (max depth 3)

#### **Diversity Heuristics (4)**
12. **Distance Preserving Crossover**: Maintains genetic diversity
13. **Crowding Mutation**: Escapes crowded search regions
14. **Niching Selection**: Fitness sharing for diversity (niche radius 0.3)
15. **Adaptive Diversity Maintenance**: Dynamic diversity adjustment

#### **Meta-Heuristics (4)**
16. **Variable Neighborhood Descent**: Systematic neighborhood exploration
17. **Iterated Local Search**: Perturbation + local search cycles
18. **Adaptive Large Neighborhood**: Dynamic neighborhood sizing
19. **Guided Local Search**: Penalty-guided feature avoidance

### **6. Reinforcement Learning (Complexity: High)**

#### **Environment (Gymnasium Interface)**
- **ScheduleEnv**: GA scheduler wrapped as RL environment
- **State Encoder**: 25-dimensional observation vector (fitness, diversity, convergence metrics)
- **Action Mapper**: 20 discrete actions (19 heuristics + no-op)
- **Reward Calculator**: Multi-component rewards (fitness improvement + diversity bonus - time penalty)

#### **RL Agents**
- **PPO (Proximal Policy Optimization)**: 
  - Learning rate: 0.0003, N-steps: 2048, Batch size: 64
  - GAE lambda: 0.95, Clip range: 0.2, Entropy coef: 0.01
- **DQN (Deep Q-Network)**:
  - Buffer size: 100K, Tau: 0.005, Exploration: ε-greedy (1.0→0.05)
- **Random Agent**: Baseline for comparison

#### **Training Infrastructure**
- **Curriculum Learning**: Progressive difficulty scaling
- **Checkpoint Management**: Automated model saving/loading
- **TensorBoard Integration**: Training metrics visualization
- **Hyperparameter Optimization**: Grid search capabilities

### **7. Constraint Evaluation (Complexity: Medium)**

#### **Hard Constraints (8 Total)**
1. **Student Group Exclusivity**: No double-booking of student groups
2. **Instructor Exclusivity**: No instructor conflicts  
3. **Instructor Qualifications**: Qualified instructor assignment
4. **Room Suitability**: Appropriate room types
5. **Instructor Time Availability**: Respect instructor schedules
6. **Room Time Availability**: Respect room availability
7. **Course Completeness**: All required sessions scheduled
8. **Room Exclusivity**: No room double-booking

#### **Soft Constraints (4 Total)**
1. **Student Schedule Compactness**: Minimize gaps in student schedules
2. **Instructor Schedule Compactness**: Minimize gaps in instructor schedules  
3. **Student Lunch Break**: Ensure midday break periods
4. **Session Continuity**: Group consecutive course sessions

#### **Evaluation Modes**
- **CPU Sequential**: Standard constraint evaluation
- **CPU Parallel**: Multiprocessing (32 cores, 3-5x speedup)
- **GPU Batch**: CUDA acceleration (10-50x speedup for large batches)

### **8. Advanced Integration (Complexity: Very High)**

#### **Hybrid Control System**
- **RL-Guided Heuristic Selection**: RL agent chooses operators dynamically
- **Round-Robin Fallback**: Fixed rotation when RL unavailable
- **Adaptive Probabilities**: Dynamic crossover/mutation rates
- **Multi-Agent Coordination**: Specialist agents for different objectives

#### **Runtime Modes (10 Progressive Levels)**
1. **Pure NSGA-II**: Baseline genetic algorithm only
2. **NSGA-II + Repairs**: Add IGLS repair system
3. **NSGA-II + Heuristics**: Add 19 heuristic operators
4. **NSGA-II + Full**: Repairs + heuristics + local search
5. **RL-Guided**: Add reinforcement learning control
6. **Round-Robin**: Fixed heuristic rotation
7. **RL Specialists**: Multiple specialized agents
8. **Archive Diversity**: Behavioral archive integration
9. **Hierarchical RL**: Two-level RL hierarchy
10. **Multi-Agent RL**: Rank-based agent coordination

---

## **Complexity Classification Summary**

- **Simplest (1-2)**: Basic GA operators, random methods
- **Low-Medium (3-4)**: NSGA-II, constraint evaluation  
- **Medium (5-6)**: Repair systems, heuristic operators
- **High (7-8)**: Reinforcement learning, hybrid systems
- **Highest (9-10)**: Multi-agent RL, hierarchical control

Your system implements **60+ distinct algorithms** ranging from basic genetic operators to cutting-edge RL-guided hyper-heuristics, making it one of the most comprehensive metaheuristic frameworks I've analyzed.