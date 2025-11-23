# Quick Reference: Available Runtime Configurations

## 🎯 **Main Experimental Groups**

### **Group A: Baseline Methods**
```bash
# A1: Pure NSGA-II (minimal baseline)
uv run nsga --prod --name "A1-pure-nsga"
uv run nsga --test --name "A1-test"  # Smoke test

# A2: NSGA-II + Repairs after every generation  
uv run nsga --prod --repair-after-every-generation --name "A2-repairs"
uv run nsga --test --repair-after-every-generation --name "A2-test"
```

### **Group B: GA Enhancement Methods**
```bash
# B1: NSGA-II + Strategic IGLS repairs
uv run repairs --prod --name "B1-igls"
uv run repairs --test --name "B1-test"

# B2: NSGA-II + IGLS + 19 Heuristics  
uv run heuristics --prod --name "B2-heuristics"
uv run heuristics --test --name "B2-test"

# B3: Full GA (IGLS + Heuristics + LNS local search)
uv run full --prod --name "B3-full"
uv run full --test --name "B3-test"
```

### **Group C: Hyper-Heuristic Methods**
```bash
# C1: Round-robin heuristic selection
uv run roundrobin --prod --name "C1-roundrobin"
uv run roundrobin --test --name "C1-test"

# C2: RL-guided heuristic selection (requires trained model)
uv run rl --prod --name "C2-rl"
uv run rl --test --name "C2-test"
```

---

## ⚙️ **Configuration Details**

| Method | Config File | Repair | Heuristics | LNS | RL | Runtime |
|--------|-------------|--------|-------------|-----|----|---------| 
| **A1** | `configs/baseline/pure-nsga-prod.yaml` | ❌ | ❌ | ❌ | ❌ | 3-4h |
| **A2** | `configs/baseline/nsga-with-repairs-prod.yaml` | ✅ Every gen | ❌ | ❌ | ❌ | 4-5h |
| **B1** | `configs/nsga/2-nsga-repairs.yaml` | ✅ Strategic | ❌ | ❌ | ❌ | 5-6h |
| **B2** | `configs/nsga/3-nsga-heuristics.yaml` | ✅ Strategic | ✅ 19 ops | ❌ | ❌ | 6-7h |
| **B3** | `configs/nsga/4-nsga-full.yaml` | ✅ Strategic | ✅ 19 ops | ✅ | ❌ | 8-10h |
| **C1** | `configs/hybrid/6-roundrobin.yaml` | ✅ Strategic | ✅ Round-robin | ✅ | ❌ | 7-8h |
| **C2** | `configs/rl/5-rl-guided.yaml` | ✅ Strategic | ✅ RL-guided | ✅ | ✅ | 9-12h |

---

## 🚀 **Quick Commands**

### **Smoke Tests (All Methods)**
```bash
# Run all smoke tests (~30-45 min total)
scripts/run_smoke_tests.ps1

# Individual smoke tests (~2-5 min each)
uv run nsga --test
uv run repairs --test  
uv run heuristics --test
uv run full --test
uv run roundrobin --test
```

### **Full Scale Experiments**
```bash
# Run all experiments (~40-50 hours total)
scripts/run_thesis_experiments.ps1

# Run specific groups
uv run nsga --prod --name "baseline"           # A1: 3-4h
uv run repairs --prod --name "igls"            # B1: 5-6h  
uv run full --prod --name "best-ga"            # B3: 8-10h
```

### **Custom Experiments**
```bash
# Use specific config file
uv run nsga --prod --config configs/baseline/pure-nsga-prod.yaml --name "custom"

# Override runtime mode
uv run nsga --prod --mode baseline --name "mode-override"

# Different environment scales  
uv run full --test    # Small: 30 gens, 100 pop (~5 min)
uv run full --med     # Medium: 200 gens, 500 pop (~30 min)  
uv run full --prod    # Full: 2000 gens, 1000 pop (~8-10h)
```

---

## 📊 **Expected Performance Characteristics**

### **Solution Quality (Lower = Better)**
```
A1 (Pure):        Hard=2500-3500,  Soft=800-1200   # Baseline
A2 (+ Repairs):   Hard=1800-2500,  Soft=600-900    # 30% improvement  
B1 (+ IGLS):      Hard=1200-1800,  Soft=400-700    # 50% improvement
B2 (+ Heuristics): Hard=800-1200,  Soft=300-500    # 70% improvement
B3 (+ LNS):       Hard=600-1000,   Soft=200-400    # 80% improvement  
C1 (Round-Robin): Hard=500-800,    Soft=150-300    # 85% improvement
C2 (RL-Guided):   Hard=400-700,    Soft=100-250    # 90% improvement
```

### **Runtime Scaling**
```
--test:  30 gens,   100 pop  →  2-5 minutes    (smoke test)
--med:   200 gens,  500 pop  →  20-60 minutes  (medium validation)  
--prod:  2000 gens, 1000 pop →  3-12 hours     (full experiment)
```

### **Computational Overhead**
```
Pure NSGA-II (A1):     1.0x baseline
+ Repairs (A2):        1.3x baseline  
+ IGLS (B1):          1.8x baseline
+ Heuristics (B2):    2.2x baseline
+ LNS (B3):           3.0x baseline
+ Round-robin (C1):   2.8x baseline  
+ RL-guided (C2):     3.5x baseline
```

---

## 🛠️ **System Requirements**

### **Hardware Recommendations**
- **CPU**: 8+ cores (16+ recommended for production runs)
- **RAM**: 16GB+ (32GB+ for large populations)  
- **Disk**: 50GB+ free space for results
- **GPU**: Optional (only for RL training, not GA execution)

### **Software Dependencies**
```bash
# Check system readiness
uv run diagnose

# Install dependencies (if needed)
uv sync --frozen

# Verify configuration
uv run nsga --test --name "validation"
```

---

## 📈 **Analysis Commands** *(Future)*

### **Comparison Analysis**
```bash
# Generate comparison tables
python scripts/analysis/compare_experiments.py

# Statistical significance tests
python scripts/analysis/statistical_analysis.py

# Performance plots  
python scripts/analysis/generate_thesis_plots.py
```

### **Individual Experiment Analysis**
```bash
# View specific experiment results
uv run list-experiments

# Extract metrics from experiment
python scripts/analysis/extract_metrics.py --experiment A1-pure-nsga-baseline
```

---

## 🎯 **Research Questions Mapping**

| Research Question | Methods to Compare | Key Metrics |
|------------------|-------------------|-------------|
| **RQ1**: Pure NSGA-II effectiveness | A1 vs Problem complexity | Hard violations, Feasibility rate |
| **RQ2**: Repair mechanism contribution | A1 vs A2 vs B1 | Violation reduction, Convergence speed |
| **RQ3**: Heuristic impact | B1 vs B2 vs B3 | Solution quality, Hypervolume |
| **RQ4**: RL vs Fixed strategies | B3 vs C1 vs C2 | Quality/Runtime ratio, Adaptability |
| **RQ5**: Cost-benefit trade-offs | All methods | Runtime vs Quality plots |

---

**Ready to start?** Run: `scripts/run_smoke_tests.ps1` to validate all configurations! 🚀