# Thesis Experimental Plan: Course Scheduling with Multi-Objective Evolutionary Algorithms

## 📋 **Overview**

This document outlines a comprehensive experimental framework for comparing different evolutionary algorithm approaches to the university course scheduling problem. The experiments progress from simple baselines to advanced RL-enhanced metaheuristics.

---

## 🎯 **Research Questions**

1. **RQ1**: How effective is pure NSGA-II for heavily constrained timetabling problems?
2. **RQ2**: What is the contribution of repair mechanisms vs. pure evolutionary search?
3. **RQ3**: How do domain-specific heuristics improve solution quality?
4. **RQ4**: Can RL-guided heuristic selection outperform fixed strategies?
5. **RQ5**: Which approach provides the best trade-off between solution quality and computational cost?

---

## 🧪 **Experimental Groups**

### **Group A: Baseline Methods (Pure Evolutionary)**
**Purpose**: Establish baseline performance without domain knowledge

| ID | Method | Config | Description | Expected Runtime |
|----|--------|--------|-------------|------------------|
| **A1** | Pure NSGA-II | `configs/baseline/pure-nsga-prod.yaml` | Minimal NSGA-II, no repairs, no heuristics | 3-4 hours |
| **A2** | NSGA-II + Repairs | `configs/baseline/nsga-with-repairs-prod.yaml` | NSGA-II + selective repair after every generation | 4-5 hours |

### **Group B: GA Enhancement Methods**
**Purpose**: Measure incremental improvements from domain knowledge

| ID | Method | Config | Description | Expected Runtime |
|----|--------|--------|-------------|------------------|
| **B1** | NSGA-II + IGLS | `configs/nsga/2-nsga-repairs.yaml` | NSGA-II + strategic repair (stagnation-triggered) | 5-6 hours |
| **B2** | NSGA-II + Heuristics | `configs/nsga/3-nsga-heuristics.yaml` | B1 + 19 domain heuristics | 6-7 hours |
| **B3** | Full GA | `configs/nsga/4-nsga-full.yaml` | B2 + LNS-IGLS local search | 8-10 hours |

### **Group C: Hyper-Heuristic Methods**
**Purpose**: Compare different heuristic selection strategies

| ID | Method | Config | Description | Expected Runtime |
|----|--------|--------|-------------|------------------|
| **C1** | Round-Robin | `configs/hybrid/6-roundrobin.yaml` | Fixed round-robin heuristic selection | 7-8 hours |
| **C2** | RL-Guided | `configs/rl/5-rl-guided.yaml` | PPO agent selects heuristics dynamically | 9-12 hours* |

### **Group D: Advanced RL Methods** *(Future Work)*
**Purpose**: Test state-of-the-art RL enhancements

| ID | Method | Config | Description | Expected Runtime |
|----|--------|--------|-------------|------------------|
| **D1** | RL Specialists | `configs/rl/7-rl-specialists.yaml` | 4 specialist agents for different contexts | 10-15 hours* |
| **D2** | Hierarchical RL | `configs/rl/9-rl-hierarchical.yaml` | Two-level RL policy (category → heuristic) | 12-18 hours* |

*\*Requires trained RL models*

---

## ⚙️ **Configuration Management**

### **Base Configuration Inheritance**
All experiments inherit from `configs/base.yaml` and override specific settings:

```yaml
# Base settings (shared)
configs/base.yaml          # Common GA parameters, constraints, time system
  ↓
configs/prod.yaml          # Production scale (2000 gens, 1000 pop)
  ↓
configs/{category}/{mode}.yaml  # Mode-specific overrides
```

### **Key Configuration Differences**

| Parameter | Pure NSGA-II | With Repairs | Full GA | RL-Guided |
|-----------|--------------|--------------|---------|-----------|
| `repair.enabled` | `false` | `true` | `true` | `true` |
| `repair.selective_repair.apply_after_mutation` | `false` | `true` | `false` | `false` |
| `repair.stagnation_repair.enabled` | `false` | `false` | `true` | `true` |
| `lns.enabled` | `false` | `false` | `true` | `true` |
| `rl.enabled` | `false` | `false` | `false` | `true` |
| `heuristics.*.enabled` | `false` | `false` | `true` | `true` |

### **Ensuring Fair Comparison**
- **Same problem instance**: `data/Course.json`, `data/Groups.json`, etc.
- **Same population size**: 1000 individuals
- **Same generations**: 2000 generations
- **Same random seed**: 69 (for reproducibility)
- **Same hardware**: CPU multiprocessing only (no GPU bias)

---

## 🚀 **Execution Commands**

### **Quick Test (Smoke Tests)**
```bash
# Test all methods with smaller scale (30 gens, 100 pop, ~2-5 min each)
uv run nsga --test                                    # A1: Pure NSGA-II
uv run nsga --test --repair-after-every-generation   # A2: With repairs
uv run baseline --test                                # A1: Alternative command
uv run repairs --test                                 # B1: NSGA-II + IGLS
uv run heuristics --test                              # B2: + Heuristics  
uv run full --test                                    # B3: Full GA
uv run roundrobin --test                              # C1: Round-robin
uv run rl --test                                      # C2: RL-guided*
```

### **Production Runs (Full Scale)**
```bash
# Baseline Group A (6-9 hours total)
uv run nsga --prod --name "A1-pure-nsga"                     # A1: 3-4h
uv run nsga --prod --repair-after-every-generation --name "A2-nsga-repairs"  # A2: 4-5h

# GA Enhancement Group B (19-23 hours total)
uv run repairs --prod --name "B1-nsga-igls"          # B1: 5-6h
uv run heuristics --prod --name "B2-nsga-heuristics" # B2: 6-7h  
uv run full --prod --name "B3-full-ga"               # B3: 8-10h

# Hyper-Heuristic Group C (16-20 hours total)
uv run roundrobin --prod --name "C1-roundrobin"      # C1: 7-8h
uv run rl --prod --name "C2-rl-guided"               # C2: 9-12h*
```

### **Batch Execution Script**
Create `scripts/run_thesis_experiments.ps1`:
```powershell
# Thesis Experiments - Full Run
Write-Host "Starting Thesis Experiments..." -ForegroundColor Green

# Group A: Baselines
uv run nsga --prod --name "A1-pure-nsga-baseline"
uv run nsga --prod --repair-after-every-generation --name "A2-nsga-with-repairs"

# Group B: GA Enhancements  
uv run repairs --prod --name "B1-nsga-igls"
uv run heuristics --prod --name "B2-nsga-heuristics"
uv run full --prod --name "B3-full-ga"

# Group C: Hyper-Heuristics
uv run roundrobin --prod --name "C1-roundrobin"
# uv run rl --prod --name "C2-rl-guided"  # Requires trained model

Write-Host "All experiments completed!" -ForegroundColor Green
```

---

## 📊 **Metrics Collection**

### **Primary Metrics**
- **Hard Constraint Violations** (minimize): Number of constraint breaches
- **Soft Constraint Penalty** (minimize): Weighted preference violations  
- **Runtime** (efficiency): Total execution time per experiment
- **Convergence Speed**: Generations to reach best solution

### **Multi-Objective Metrics**
- **Hypervolume** (maximize): Volume dominated by Pareto front
- **IGD - Inverted Generational Distance** (minimize): Convergence + coverage
- **Spread** (maximize): Diversity of Pareto front solutions
- **Feasibility Rate** (%): Percentage of feasible solutions generated

### **Algorithm-Specific Metrics**
- **Repair Statistics**: Number of fixes per constraint type
- **Heuristic Usage** (RL methods): Which heuristics are selected when
- **Population Diversity**: Average pairwise solution distance

### **Output Locations**
```
output/
├── A1-pure-nsga-baseline_20241123_140000/
│   ├── best_schedule.json                    # Final solution
│   ├── evolution_metrics.json               # Generation-by-generation data
│   ├── pareto_front_plot.png               # Multi-objective visualization
│   ├── constraint_trends_plot.png          # Constraint violation trends
│   └── experiment_report.pdf               # Comprehensive report
├── A2-nsga-with-repairs_20241123_180000/
├── B1-nsga-igls_20241123_220000/
└── ...
```

---

## 📈 **Analysis Framework**

### **Statistical Comparison**
1. **Multiple Independent Runs**: 10 runs per method (different seeds)
2. **Statistical Tests**: Wilcoxon signed-rank test (non-parametric)
3. **Effect Size**: Cohen's d for practical significance
4. **Confidence Intervals**: 95% CI for all metrics

### **Performance Comparison Tables**
Generate comparison tables showing:
```
| Method | Hard Violations | Soft Penalty | Runtime | Hypervolume | IGD |
|--------|----------------|--------------|---------|-------------|-----|
| A1     | 245.2 ± 12.1   | 156.7 ± 8.9  | 3.2h    | 0.67 ± 0.03 | 12.4|
| A2     | 198.6 ± 15.2   | 142.1 ± 7.2  | 4.1h    | 0.72 ± 0.04 | 9.8 |
| B1     | 156.3 ± 9.8    | 128.4 ± 6.1  | 5.4h    | 0.78 ± 0.02 | 7.2 |
| ...    | ...            | ...          | ...     | ...         | ... |
```

### **Visualization Plots**
1. **Convergence Plots**: Fitness vs. generation for all methods
2. **Pareto Front Comparison**: Multi-objective trade-offs
3. **Runtime vs. Quality**: Efficiency analysis
4. **Constraint Heat Maps**: Which constraints are hardest to satisfy

---

## 🛠️ **Implementation Workflow**

### **Phase 1: Setup & Validation (Week 1)**
```bash
# 1. Validate all configurations
uv run nsga --test
uv run repairs --test  
uv run heuristics --test
uv run full --test

# 2. Quick smoke test (~30 min total)
scripts/run_smoke_tests.ps1

# 3. Check output structure
ls output/ | head -5
```

### **Phase 2: Baseline Experiments (Week 2)**
```bash
# Run Group A + B1 (lighter experiments first)
uv run nsga --prod --name "A1-pure-nsga"           # 4h
uv run nsga --prod --repair-after-every-generation --name "A2-repairs"  # 5h  
uv run repairs --prod --name "B1-igls"             # 6h
```

### **Phase 3: Advanced Experiments (Week 3-4)**
```bash
# Run remaining GA methods
uv run heuristics --prod --name "B2-heuristics"    # 7h
uv run full --prod --name "B3-full"                # 10h
uv run roundrobin --prod --name "C1-roundrobin"    # 8h
```

### **Phase 4: RL Training & Evaluation** *(Optional)*
```bash
# Train RL models (if not already done)
uv run train-rl --prod --curriculum                # 2-4h  

# Run RL experiments
uv run rl --prod --name "C2-rl-guided"             # 12h
```

### **Phase 5: Analysis & Reporting (Week 5)**
```bash
# Generate comparison plots and tables
python scripts/analysis/compare_experiments.py
python scripts/analysis/generate_thesis_plots.py
python scripts/analysis/statistical_analysis.py
```

---

## 🎯 **Expected Outcomes & Hypotheses**

### **Hypothesis 1**: Pure NSGA-II Performance
- **H1a**: Pure NSGA-II will struggle with heavily constrained problems (>1000 violations)
- **H1b**: Adding repairs will provide significant improvement (30-50% violation reduction)

### **Hypothesis 2**: Incremental GA Improvements  
- **H2a**: IGLS repairs will reduce violations by 20-40% over pure NSGA-II
- **H2b**: Domain heuristics will provide additional 15-25% improvement
- **H2c**: LNS-IGLS will fine-tune solutions for 5-15% final improvement

### **Hypothesis 3**: RL vs. Fixed Strategies
- **H3a**: RL-guided selection will outperform round-robin by 10-20%
- **H3b**: RL will show faster convergence (fewer generations to optimal)
- **H3c**: RL will have higher computational overhead (20-30% longer runtime)

### **Hypothesis 4**: Runtime vs. Quality Trade-offs
- **H4a**: There will be diminishing returns after B2 (heuristics)
- **H4b**: RL methods will have best quality/runtime ratio for long runs
- **H4c**: Simple repairs (A2) will have best quality/runtime ratio for short runs

---

## 🚨 **Risk Management**

### **Technical Risks**
1. **Long Runtimes**: Start with smaller populations for initial validation
2. **Memory Issues**: Monitor RAM usage, restart between experiments
3. **Config Bugs**: Validate each config with smoke tests first
4. **RL Model Issues**: Have fallback non-RL experiments ready

### **Timeline Risks**
1. **Hardware Availability**: Run experiments in parallel on multiple machines if available
2. **Unexpected Results**: Budget extra time for investigating anomalies
3. **Analysis Complexity**: Prepare analysis scripts in advance

### **Mitigation Strategies**
- **Checkpointing**: Save intermediate results every 100 generations
- **Resume Capability**: Use experiment manager to resume interrupted runs
- **Backup Plans**: Have simplified experimental design ready if needed

---

## 📝 **Documentation & Reproducibility**

### **Configuration Versioning**
All configs stored in git with clear naming:
```
configs/
├── baseline/
│   ├── pure-nsga-prod.yaml          # A1: Pure NSGA-II
│   └── nsga-with-repairs-prod.yaml  # A2: With repairs
├── nsga/
│   ├── 2-nsga-repairs.yaml          # B1: IGLS repairs
│   ├── 3-nsga-heuristics.yaml       # B2: + Heuristics
│   └── 4-nsga-full.yaml             # B3: Full GA
└── rl/
    └── 5-rl-guided.yaml             # C2: RL-guided
```

### **Experiment Logs**
Each experiment automatically generates:
- **Config snapshot**: Exact parameters used  
- **System info**: Hardware, OS, package versions
- **Random seeds**: For reproducibility
- **Git commit hash**: Code version used
- **Runtime logs**: Detailed execution trace

### **Result Archive Structure**
```
results/
├── thesis_experiments_20241123/
│   ├── experiment_manifest.json     # All runs metadata
│   ├── statistical_analysis.json   # Comparison results
│   ├── plots/                       # All visualizations
│   └── raw_data/                    # Individual experiment outputs
├── quick_tests_20241120/            # Smoke test results  
└── pilot_study_20241115/            # Initial validation
```

---

## 🎓 **Thesis Integration**

### **Chapter Structure Alignment**
- **Chapter 3 (Methodology)**: Reference this experimental plan
- **Chapter 4 (Implementation)**: Describe configuration system
- **Chapter 5 (Results)**: Present comparative analysis
- **Chapter 6 (Discussion)**: Interpret findings vs. hypotheses

### **Key Contributions**
1. **Comprehensive GA Baseline**: First thorough comparison of repair strategies
2. **RL Integration Framework**: Novel hyper-heuristic approach for timetabling
3. **Scalable Configuration System**: Reusable framework for GA experiments
4. **Multi-Objective Analysis**: Beyond single fitness - real-world metrics

### **Publication Potential**
- **Conference Paper**: "RL-Enhanced Evolutionary Algorithms for University Timetabling"
- **Journal Paper**: "Comprehensive Comparison of Metaheuristics for Course Scheduling"
- **Workshop Paper**: "Configurable GA Framework for Constraint Satisfaction Problems"

---

## ✅ **Quick Start Checklist**

- [ ] **Validate Setup**: Run `uv run diagnose` to check system
- [ ] **Test Configs**: Run smoke tests for all methods (`--test` flag)
- [ ] **Check Disk Space**: Ensure 50GB+ available for results
- [ ] **Plan Schedule**: Block 3-5 days for continuous execution  
- [ ] **Prepare Analysis**: Set up scripts for result processing
- [ ] **Document Baselines**: Record initial system benchmarks

**Ready to start?** Run:
```bash
uv run nsga --test --name "validation-test"
```

If successful, proceed with full experimental plan! 🚀