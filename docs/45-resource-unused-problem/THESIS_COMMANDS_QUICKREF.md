#  THESIS EXPERIMENTS - QUICK COMMAND REFERENCE

**Last Updated:** November 19, 2025

---

##  LOGICAL THESIS COMMANDS (NEW!)

### **Run Experiments (Short & Clear)**

```bash
# Experiment 1: Pure NSGA-II Baseline
uv run exp1 --env prod

# Experiment 2: NSGA-II + Repairs
uv run exp2 --env prod

# Experiment 3: Repairs + Heuristics (no local search)
uv run exp3 --env prod

# Experiment 4: + Local Search (LNS)
uv run exp4 --env prod

# Experiment 5: RL-Guided Adaptive Selection
uv run exp5 --env prod
```

### **Or Use Full Descriptive Names**

```bash
uv run thesis-exp1-baseline --env prod
uv run thesis-exp2-repairs --env prod
uv run thesis-exp3-heuristics --env prod
uv run thesis-exp4-local-search --env prod
uv run thesis-exp5-rl --env prod
```

---

##  ABLATION STUDIES

```bash
# Isolate specific components
uv run ablation-no-repairs --env prod         # Baseline (no repairs)
uv run ablation-repairs-only --env prod       # Only repairs
uv run ablation-heuristics-no-ls --env prod   # Heuristics without LNS
uv run ablation-local-search --env prod       # Full with LNS
```

---

##  RL TRAINING (For Experiment 5)

```bash
# Train RL agent (100K timesteps recommended)
uv run train-rl --env prod --timesteps 100000

# Quick training for testing (10K timesteps)
uv run train-rl-quick --env prod --timesteps 10000

# Promote best model to production
uv run promote-model --model models/rl_agents/rl_agent_prod_ppo_100000_*.zip

# Validate RL model
uv run validate-rl
```

---

##  RESULTS ANALYSIS

```bash
# Compare all experiments
uv run compare-experiments

# Generate thesis-ready plots
uv run generate-thesis-plots

# Export data to CSV/LaTeX
uv run export-thesis-data

# Convergence analysis
uv run analyze-convergence

# Diversity metrics
uv run analyze-diversity
```

---

##  DIAGNOSTICS

```bash
# GPU status check
uv run diagnose-gpu

# Full system diagnostics
uv run diagnose-system

# List all available experiments
uv run list-experiments

# Check input data
uv run check-data

# Verify configuration
uv run verify-config
```

---

## ⚙️ CONFIGURATION UTILITIES

```bash
# Show all configuration
uv run show-config

# Show repair system config
uv run show-repair

# Show soft constraints only
uv run show-soft

# Show time system config
uv run show-time
```

---

##  BENCHMARKING

```bash
# GPU benchmark
uv run benchmark-gpu

# LNS benchmark
uv run benchmark-lns

# Constraints benchmark
uv run benchmark-constraints

# Run all benchmarks
uv run benchmark-all
```

---

## 🛠️ UTILITIES

```bash
# Start TensorBoard (for RL training visualization)
uv run tensorboard

# Clean old output files
uv run clean-output

# Git commit squashing
uv run git-squash
```

---

##  RECOMMENDED WORKFLOW

### **Phase 1: Quick Testing (30 minutes)**
```bash
# Test all experiments with test mode (30 gens each)
uv run exp1 --env test  # ~5 min
uv run exp2 --env test  # ~5 min
uv run exp3 --env test  # ~5 min
uv run exp4 --env test  # ~5 min
uv run exp5 --env test  # ~5 min (if RL model exists)
```

### **Phase 2: RL Training (2-3 hours)**
```bash
# Train RL agent for Experiment 5
uv run train-rl --env prod --timesteps 100000

# Promote best model
uv run promote-model --model models/rl_agents/rl_agent_prod_ppo_100000_*.zip

# Validate
uv run validate-rl
```

### **Phase 3: Production Experiments (7.5-12.5 hours)**
```bash
# Run all 5 experiments
uv run exp1 --env prod  # 1.5-2.5h
uv run exp2 --env prod  # 1.5-2.5h
uv run exp3 --env prod  # 1.5-2.5h
uv run exp4 --env prod  # 1.5-2.5h
uv run exp5 --env prod  # 1.5-2.5h
```

### **Phase 4: Analysis & Results**
```bash
# Compare results
uv run compare-experiments

# Generate plots
uv run generate-thesis-plots

# Export data
uv run export-thesis-data

# Analyze convergence
uv run analyze-convergence
```

---

##  EXPERIMENT COMPARISON TABLE

| Command | Experiment | Features | Expected Hard Violations | Quality |
|---------|-----------|----------|-------------------------|---------|
| `exp1` | Baseline | NSGA-II only | 20-50 |  |
| `exp2` | Repairs | + IGLS repairs | 5-15 |  |
| `exp3` | Heuristics | + 19 heuristics (no LNS) | 8-15 |  |
| `exp4` | Local Search | + LNS local search | 0-5 |  |
| `exp5` | RL-Guided | + RL adaptive selection | 0-3 |  |

---

##  ONE-LINE FULL RUN

```bash
# Test everything first
uv run exp1 --env test && uv run exp2 --env test && uv run exp3 --env test && uv run exp4 --env test

# Then production (will take 7.5-12.5 hours)
uv run exp1 --env prod && uv run exp2 --env prod && uv run exp3 --env prod && uv run exp4 --env prod && uv run exp5 --env prod
```

---

##  WHY THESE COMMANDS ARE BETTER

### **Before (Confusing):**
```bash
uv run prod           # What does this even run?
uv run baseline       # Which baseline?
uv run roundrobin     # Round-robin of what?
```

### **After (Clear):**
```bash
uv run exp1 --env prod     # Experiment 1, production mode
uv run exp2 --env prod     # Experiment 2, production mode
uv run exp3 --env prod     # Experiment 3, production mode
```

**Benefits:**
-  Clear experiment numbering (matches thesis)
-  Logical progression (each builds on previous)
-  Short commands (exp1, exp2, etc.)
-  Descriptive full names available (thesis-exp1-baseline)
-  Ablation study commands (isolate components)
-  Analysis commands (compare, plot, export)

---

##  FOR YOUR THESIS

**Use these commands in your thesis:**

```
Experiments were conducted using the following commands:
- Experiment 1 (Baseline): uv run exp1 --env prod
- Experiment 2 (Repairs): uv run exp2 --env prod
- Experiment 3 (Heuristics): uv run exp3 --env prod
- Experiment 4 (Local Search): uv run exp4 --env prod
- Experiment 5 (RL-Guided): uv run exp5 --env prod

All experiments used identical hardware (16-thread CPU, RTX GPU)
and dataset (4,886 courses) for fair comparison.
```

---

##  READY TO START!

```bash
# Step 1: Verify system
uv run diagnose-system

# Step 2: Test experiments
uv run exp1 --env test

# Step 3: Run production
uv run exp1 --env prod

# Step 4: Compare results
uv run compare-experiments
```

**GO!** 
