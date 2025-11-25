# Schedule-Engine Experiment Architecture Plan

## Overview

This document outlines a comprehensive, incremental experimental framework for evaluating all 60+ algorithms implemented in the Schedule-Engine. Experiments are organized into logical groups that build complexity progressively, allowing for systematic performance analysis and ablation studies.

## Experiment Design Principles

1. **Incremental Complexity**: Each group builds upon the previous group's findings
2. **Controlled Variables**: Single algorithm component changes between experiments  
3. **Statistical Rigor**: Multiple runs with different seeds for statistical significance
4. **Comparative Analysis**: Direct comparisons within and across groups
5. **Scalability Testing**: Multiple problem sizes (small, medium, large datasets)

## Experiment Groups Architecture

### **Group A: Foundation Algorithms (Baseline Establishment)**
*Purpose: Establish performance baselines for core GA components*

#### **A1: Population Initialization Strategies**
- **A1.1** - Pure Random Initialization
- **A1.2** - Greedy Initialization (Largest-Degree-First)  
- **A1.3** - Constraint-Guided Smart Initialization
- **A1.4** - Hybrid Initialization (25% greedy, 50% smart, 25% random)

#### **A2: Selection Operator Comparison**  
- **A2.1** - Tournament Selection (Binary)
- **A2.2** - NSGA-II Selection (Standard)
- **A2.3** - Fast NSGA-II Selection (Optimized)
- **A2.4** - Elitist Selection (Top 10%)

#### **A3: Basic Genetic Operators**
- **A3.1** - Random Crossover + Random Mutation
- **A3.2** - Course-Group-Aware Crossover + Random Mutation
- **A3.3** - Course-Group-Aware Crossover + Constraint-Guided Mutation
- **A3.4** - Distance-Preserving Crossover + Crowding Mutation

### **Group B: Multi-Objective Optimization (NSGA-II Variants)**
*Purpose: Evaluate multi-objective optimization effectiveness*

#### **B1: NSGA-II Core Components**
- **B1.1** - Standard NSGA-II (Baseline)
- **B1.2** - NSGA-II + Fast Non-Dominated Sorting
- **B1.3** - NSGA-II + Enhanced Crowding Distance
- **B1.4** - NSGA-II + Lexicographic Weights Tuning

#### **B2: Fitness Landscape Analysis**
- **B2.1** - Single Objective (Hard Constraints Only)
- **B2.2** - Single Objective (Soft Constraints Only)  
- **B2.3** - Dual Objective (Standard Weights: -1.0, -0.01)
- **B2.4** - Dual Objective (Balanced Weights: -1.0, -1.0)

#### **B3: Population Size Impact Study**
- **B3.1** - Small Population (50 individuals)
- **B3.2** - Medium Population (200 individuals) 
- **B3.3** - Large Population (500 individuals)
- **B3.4** - Very Large Population (1000 individuals)

### **Group C: Repair & Local Search Systems**
*Purpose: Evaluate constraint repair effectiveness*

#### **C1: IGLS Repair System**
- **C1.1** - No Repair (Pure GA Baseline)
- **C1.2** - Stagnation-Triggered IGLS Repair
- **C1.3** - Periodic IGLS Repair (Fixed Intervals)
- **C1.4** - Adaptive IGLS Repair (Dynamic Triggering)

#### **C2: LNS Integration** 
- **C2.1** - IGLS Only (No LNS)
- **C2.2** - LNS + IGLS (Small Neighborhoods: 5-10 sessions)
- **C2.3** - LNS + IGLS (Medium Neighborhoods: 15-25 sessions)
- **C2.4** - LNS + IGLS (Large Neighborhoods: 30-50 sessions)

#### **C3: Selective Repair Strategies**
- **C3.1** - No Selective Repair
- **C3.2** - Fast Detection + Targeted Repair
- **C3.3** - Full Detection + Population-Wide Repair
- **C3.4** - Hybrid Detection + Smart Repair

### **Group D: Heuristic Operator Evaluation**
*Purpose: Systematic evaluation of 19 heuristic operators*

#### **D1: Construction Heuristics (3 Operators)**
- **D1.1** - Largest Degree First
- **D1.2** - Most Constrained First  
- **D1.3** - Earliest Deadline First
- **D1.4** - Combined Construction Heuristics

#### **D2: Perturbation Heuristics (5 Operators)**
- **D2.1** - Random Swap
- **D2.2** - Temporal Shift
- **D2.3** - Room Shuffle  
- **D2.4** - Instructor Reassign
- **D2.5** - Multi-Perturbation
- **D2.6** - All Perturbation Heuristics Combined

#### **D3: Improvement Heuristics (3 Operators)**
- **D3.1** - Kempe Chain (Graph-Based Conflict Resolution)
- **D3.2** - Ejection Chain (Sequential Improvement)
- **D3.3** - Variable Depth Search (Multi-Level Exploration)
- **D3.4** - All Improvement Heuristics Combined

#### **D4: Diversity Heuristics (4 Operators)**
- **D4.1** - Distance Preserving Crossover
- **D4.2** - Crowding Mutation
- **D4.3** - Niching Selection  
- **D4.4** - Adaptive Diversity Maintenance
- **D4.5** - All Diversity Heuristics Combined

#### **D5: Meta-Heuristics (4 Operators)**
- **D5.1** - Variable Neighborhood Descent
- **D5.2** - Iterated Local Search
- **D5.3** - Adaptive Large Neighborhood Search
- **D5.4** - Guided Local Search
- **D5.5** - All Meta-Heuristics Combined

### **Group E: Constraint System Analysis**
*Purpose: Evaluate constraint evaluation modes and impact*

#### **E1: Hard Constraint Analysis (8 Constraints)**
- **E1.1** - Individual Hard Constraint Impact Study
- **E1.2** - Hard Constraint Weighting Sensitivity
- **E1.3** - Hard Constraint Ordering Effects
- **E1.4** - Hard Constraint Interaction Analysis

#### **E2: Soft Constraint Analysis (4 Constraints)**  
- **E2.1** - Individual Soft Constraint Impact Study
- **E2.2** - Soft Constraint Weighting Sensitivity
- **E2.3** - Soft Weight Factor Optimization (0.001 to 0.1)
- **E2.4** - Soft Constraint Priority Ranking

#### **E3: Evaluation Mode Performance**
- **E3.1** - CPU Sequential Evaluation (Baseline)
- **E3.2** - CPU Parallel Evaluation (32 cores)
- **E3.3** - GPU Batch Evaluation (CUDA)
- **E3.4** - Hybrid CPU-GPU Evaluation

### **Group F: Reinforcement Learning Integration**
*Purpose: Evaluate RL-guided heuristic selection*

#### **F1: RL Environment Setup**
- **F1.1** - State Encoder Validation (25D observation vector)
- **F1.2** - Action Mapper Testing (20 discrete actions)
- **F1.3** - Reward Function Tuning
- **F1.4** - Episode Length Optimization

#### **F2: RL Agent Comparison**
- **F2.1** - Random Agent (Baseline)
- **F2.2** - PPO Agent (Proximal Policy Optimization)
- **F2.3** - DQN Agent (Deep Q-Network)  
- **F2.4** - Agent Performance Comparison

#### **F3: RL Training Strategies**
- **F3.1** - Standard Training (Fixed Environment)
- **F3.2** - Curriculum Learning (Progressive Difficulty)
- **F3.3** - Transfer Learning (Pre-trained Models)
- **F3.4** - Multi-Environment Training

#### **F4: RL Hyperparameter Optimization**
- **F4.1** - Learning Rate Sensitivity (PPO: 0.0001 to 0.001)
- **F4.2** - Network Architecture Comparison
- **F4.3** - Exploration Strategy Tuning (DQN ε-greedy)
- **F4.4** - Batch Size Impact Study

### **Group G: Advanced Integration Systems**
*Purpose: Evaluate complex hybrid approaches*

#### **G1: Hybrid Control Systems**
- **G1.1** - Pure GA (No RL)
- **G1.2** - RL-Guided Heuristic Selection
- **G1.3** - RL with Round-Robin Fallback
- **G1.4** - Adaptive Probability Control

#### **G2: Multi-Agent Coordination**
- **G2.1** - Single Agent System
- **G2.2** - Specialist Agents (Repair vs Optimization)
- **G2.3** - Hierarchical RL (Two-Level Control)
- **G2.4** - Rank-Based Multi-Agent Coordination

#### **G3: Archive-Based Systems**
- **G3.1** - No Archive (Standard Population)
- **G3.2** - Behavioral Archive Integration  
- **G3.3** - Elite Archive Maintenance
- **G3.4** - Hybrid Archive Strategies

### **Group H: Runtime Mode Validation**
*Purpose: Validate all 10 progressive runtime modes*

#### **H1: Basic Runtime Modes (1-4)**
- **H1.1** - Mode 1: Pure NSGA-II
- **H1.2** - Mode 2: NSGA-II + Repairs  
- **H1.3** - Mode 3: NSGA-II + Heuristics
- **H1.4** - Mode 4: NSGA-II + Full (Best Non-RL)

#### **H2: Advanced Runtime Modes (5-7)**
- **H2.1** - Mode 5: RL-Guided Selection
- **H2.2** - Mode 6: Round-Robin Heuristic Rotation
- **H2.3** - Mode 7: RL Specialist Agents

#### **H3: Cutting-Edge Runtime Modes (8-10)**
- **H3.1** - Mode 8: Archive Diversity Integration
- **H3.2** - Mode 9: Hierarchical RL Control  
- **H3.3** - Mode 10: Multi-Agent RL Coordination

### **Group I: Scalability & Performance Analysis**
*Purpose: Evaluate system performance across scales*

#### **I1: Problem Size Scalability**
- **I1.1** - Small Dataset (50 courses, 10 rooms, 20 instructors)
- **I1.2** - Medium Dataset (100 courses, 20 rooms, 40 instructors)
- **I1.3** - Large Dataset (200 courses, 40 rooms, 80 instructors)
- **I1.4** - Very Large Dataset (500+ courses, 100+ rooms, 200+ instructors)

#### **I2: Computational Performance**
- **I2.1** - Runtime Complexity Analysis
- **I2.2** - Memory Usage Profiling  
- **I2.3** - GPU Acceleration Effectiveness
- **I2.4** - Parallel Processing Efficiency

#### **I3: Convergence Analysis**
- **I3.1** - Convergence Rate Comparison
- **I3.2** - Solution Quality vs Time Trade-offs
- **I3.3** - Stagnation Detection Effectiveness
- **I3.4** - Early Stopping Optimization

### **Group J: Ablation Studies**
*Purpose: Understand individual component contributions*

#### **J1: Algorithm Component Ablation**
- **J1.1** - Remove Repair Systems
- **J1.2** - Remove Heuristic Operators
- **J1.3** - Remove RL Integration  
- **J1.4** - Remove Multi-Objective Optimization

#### **J2: Parameter Sensitivity Analysis**
- **J2.1** - Population Size Sensitivity
- **J2.2** - Generation Count Optimization
- **J2.3** - Crossover/Mutation Probability Tuning
- **J2.4** - Constraint Weight Sensitivity

#### **J3: Feature Importance Analysis**
- **J3.1** - Most Critical Hard Constraints
- **J3.2** - Most Impactful Soft Constraints
- **J3.3** - Most Effective Heuristics
- **J3.4** - Optimal Algorithm Combinations

## Execution Framework

### **Phase 1: Foundation (Groups A-C)**
*Duration: 2-3 weeks*
- Establish baselines and core algorithm performance
- Focus on statistical significance (10+ runs per experiment)
- Generate performance matrices for all basic components

### **Phase 2: Enhancement (Groups D-E)**  
*Duration: 3-4 weeks*
- Systematic heuristic evaluation
- Constraint system optimization
- Identify best-performing operator combinations

### **Phase 3: Intelligence (Groups F-G)**
*Duration: 4-5 weeks*  
- RL integration and training
- Advanced hybrid system evaluation
- Multi-agent coordination testing

### **Phase 4: Validation (Groups H-I)**
*Duration: 2-3 weeks*
- Runtime mode validation
- Scalability assessment  
- Performance benchmarking

### **Phase 5: Analysis (Group J)**
*Duration: 1-2 weeks*
- Ablation studies
- Component contribution analysis
- Final recommendations

## Success Metrics

### **Primary Metrics**
- **Solution Quality**: Hard constraint violations (must reach 0)
- **Optimization Effectiveness**: Soft constraint penalty minimization
- **Convergence Speed**: Generations to reach acceptable solution
- **Computational Efficiency**: Runtime and memory usage

### **Secondary Metrics**  
- **Diversity Maintenance**: Population diversity over generations
- **Scalability**: Performance across different problem sizes
- **Robustness**: Consistency across multiple runs
- **Hyperparameter Sensitivity**: Stability to parameter changes

### **Advanced Metrics**
- **Pareto Front Quality**: Hypervolume, IGD, GD indicators
- **Search Space Exploration**: Coverage and exploitation balance
- **Learning Effectiveness**: RL agent improvement over time
- **System Integration**: Component synergy effects

## Expected Outcomes

### **Research Contributions**
1. **Comprehensive Algorithm Comparison**: First systematic evaluation of 60+ metaheuristic components
2. **Optimal Configuration Discovery**: Best algorithm combinations for timetabling
3. **Scalability Insights**: Performance characteristics across problem sizes  
4. **RL Integration Guidelines**: Best practices for RL in optimization
5. **Hybrid System Design**: Principles for multi-algorithm coordination

### **Practical Applications**
1. **Production Configuration**: Optimal settings for real-world deployment
2. **Problem-Specific Recommendations**: Algorithm selection based on problem characteristics
3. **Performance Prediction Models**: Estimate runtime and quality for new problems
4. **Automated Configuration**: Self-tuning system based on problem analysis

## Implementation Notes

### **Experiment Management**
- Use existing `ExperimentManager` for run tracking
- Store all results in `output/experiments/` with structured naming
- Generate comparison reports automatically
- Maintain experiment manifest with metadata

### **Statistical Requirements**
- Minimum 10 runs per experiment for statistical significance
- Use different random seeds for each run  
- Apply appropriate statistical tests (t-test, Mann-Whitney U)
- Report confidence intervals and effect sizes

### **Resource Planning**
- Estimate total runtime: 200+ hours of computation
- GPU requirements for Groups E3, F, and I
- Parallel execution where possible
- Cloud resources for large-scale experiments

This experimental framework provides a systematic approach to evaluating your comprehensive algorithm suite while maintaining scientific rigor and practical relevance.