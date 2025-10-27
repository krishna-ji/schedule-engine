# Pre-Run Checklist & Analysis Guide

## ✅ Changes Made Before Your Dev Run

### 1. Fixed `configs/dev.yaml` Configuration
**Problem**: Missing GA parameters (ngen, pop_size) which would cause fallback to undefined defaults.

**Fix**: Added complete GA section:
```yaml
ga:
  ngen: 100
  pop_size: 100
  cxpb: 0.85
  mutpb: 0.25
```

**Also Changed**: Set `fail_on_infeasibility: false` so the run continues even if feasibility warnings are detected. This allows us to see how the GA handles difficult problems.

---

## 🎯 What to Provide After Your Dev Run

Please share these files from your `output/evaluation_<timestamp>/` directory:

1. **`logger.txt`** or **`logger_all.txt`** - Complete generation-by-generation log
2. **`schedule.json`** - Final best schedule
3. **`feasibility_report.txt`** - Pre-GA feasibility analysis
4. **`violation_report.txt`** - Constraint violation details
5. **Plots** (if available):
   - `plots/hard_violations.png`
   - `plots/soft_penalties.png`
   - `plots/diversity.png`
   - `plots/pareto_front.png`

---

## 🔍 What I'll Analyze

### 1. **Convergence Patterns**
- Are hard violations decreasing over time?
- When does the GA plateau/stagnate?
- Is diversity being maintained or collapsing?

### 2. **Constraint Bottlenecks**
- Which specific hard constraints are never reaching zero?
- Are violations concentrated in certain courses/groups/instructors?
- Are there structural issues (e.g., impossible requirements)?

### 3. **Operator Effectiveness**
- Is mutation helping or causing chaos?
- Is crossover preserving good genes?
- Are repair heuristics working or just masking problems?

### 4. **Configuration Issues**
- Are parameters (cxpb, mutpb) appropriate?
- Is population size sufficient?
- Should we adjust constraint weights?

### 5. **Data Quality Issues**
- Instructor availability conflicts
- Room capacity mismatches
- Over-subscribed time slots
- Unrealistic course requirements

---

## 📊 Expected Runtime

With your dev configuration:
- **Population**: 100 individuals
- **Generations**: 100
- **Multiprocessing**: Enabled
- **Estimated Time**: 5-15 minutes (depends on CPU cores and data size)

---

## 🚨 Watch For These Warning Signs

During execution, watch console output for:

1. **Feasibility warnings** - "Critical bottleneck detected"
2. **Repair failures** - "Repair heuristic X applied but made no progress"
3. **Stagnation** - "No improvement in last N generations"
4. **Diversity collapse** - "Average pairwise diversity < 0.1"

Copy any concerning warnings you see!

---

## 💡 Quick Sanity Checks Before Running

Run these commands to verify your setup:

```powershell
# Check Python environment
python --version

# Verify required packages
pip list | Select-String "deap|pydantic|rich|matplotlib|pyyaml"

# Check data files exist
Get-ChildItem data/*.json

# Verify config loads correctly
python main.py --env dev --help
```

---

## 🎬 Run Command

```powershell
python main.py --env dev
```

The engine will:
1. Load merged config (common.yaml + dev.yaml)
2. Load input data from `data/*.json`
3. Run feasibility checks
4. Execute 100 generations of NSGA-II
5. Export results to `output/evaluation_<timestamp>/`

---

## 📝 After the Run

**Share the complete output directory** or at minimum the files listed above. I'll provide:

1. **Root cause analysis** of what's preventing convergence
2. **Specific constraint violations** that need fixing
3. **Data quality issues** in your input JSONs
4. **Configuration recommendations** for better convergence
5. **Prioritized action items** to fix the most impactful issues

---

## ⚡ Ready to Go!

Your configuration is now fixed and ready for the VM run. Good luck! 🚀
