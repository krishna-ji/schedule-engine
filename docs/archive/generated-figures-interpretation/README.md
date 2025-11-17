# Generated Figures Interpretation

Complete documentation for all NSGA-II schedule engine visualizations.

## 📚 Documentation Files

### [00_INDEX.md](./00_INDEX.md) — **START HERE**
Master index with:
- Quick reference table of all plots
- Use case guides (health check, tuning, research, presentations)
- Troubleshooting guide
- FAQ and best practices

### [01_PARETO_FRONT_VISUALIZATIONS.md](./01_PARETO_FRONT_VISUALIZATIONS.md)
Detailed explanation of Pareto front plots:
- `pareto_front.pdf` - Basic population view
- `pareto_front_detail.pdf` - Zoomed front with labels
- `pareto_front_comprehensive.pdf` - 4-subplot comprehensive analysis
  - All population with jitter
  - Population with Pareto front
  - Population density heatmap
  - Frequency-coded population

### [02_QUALITY_METRICS.md](./02_QUALITY_METRICS.md)
Multi-objective performance indicators:
- `hypervolume_trend.pdf` - Gold standard quality metric
- `spacing_trend.pdf` - Solution distribution uniformity
- `spacing_distribution.pdf` - Histogram of spacing
- `spacing_pareto_combined.pdf` - Dual view (front + metric)
- `feasibility_evolution.pdf` - Constraint satisfaction rate

### [03_CONVERGENCE_ANALYSIS.md](./03_CONVERGENCE_ANALYSIS.md)
Algorithm dynamics and optimization progress:
- `convergence_dashboard.pdf` - 2×3 comprehensive dashboard
- `convergence_multi_metric.pdf` - Normalized metric comparison
- `convergence_rate_hard_violations.pdf` - Improvement rate & stagnation
- `hard_constraint_trend.pdf` - Hard violations over time
- `soft_constraint_trend.pdf` - Soft penalties over time
- `diversity.pdf` - Population genetic diversity

---

## 🚀 Quick Start

### I need to...

**Quickly check if my run was successful**
→ Read: [00_INDEX.md#1-quick-health-check](./00_INDEX.md#1-quick-health-check-2-minutes)

**Select the best schedule for deployment**
→ Read: [00_INDEX.md#2-solution-selection](./00_INDEX.md#2-solution-selection-5-minutes)

**Improve algorithm performance**
→ Read: [00_INDEX.md#3-algorithm-tuning](./00_INDEX.md#3-algorithm-tuning-15-minutes)

**Understand what each axis means**
→ Read: [00_INDEX.md#understanding-the-axes](./00_INDEX.md#understanding-the-axes)

**Troubleshoot weird plot patterns**
→ Read: [00_INDEX.md#troubleshooting-guide](./00_INDEX.md#troubleshooting-guide)

**Learn the mathematics behind metrics**
→ Read each category file's "Mathematical Definition" sections

---

## 📊 Plot Categories

### By Purpose
- **Trade-Off Analysis**: pareto_front*.pdf
- **Quality Assessment**: hypervolume, spacing, feasibility
- **Convergence Monitoring**: convergence_dashboard, rate, trends
- **Distribution Analysis**: spacing_distribution, density plots

### By Complexity
- **Beginner**: Basic Pareto, hard/soft trends, feasibility
- **Intermediate**: Dashboard, hypervolume, spacing
- **Advanced**: Multi-metric, convergence rate, density plots

---

## 📁 Where Plots are Saved

```
output/evaluation_<timestamp>/
├── plots/                           # All PDF visualizations
│   ├── pareto_front.pdf
│   ├── pareto_front_detail.pdf
│   ├── pareto_front_comprehensive.pdf
│   ├── convergence_dashboard.pdf
│   ├── convergence_multi_metric.pdf
│   ├── convergence_rate_hard_violations.pdf
│   ├── hypervolume_trend.pdf
│   ├── spacing_trend.pdf
│   ├── spacing_distribution.pdf
│   ├── spacing_pareto_combined.pdf
│   ├── feasibility_evolution.pdf
│   ├── hard_constraint_trend.pdf
│   ├── soft_constraint_trend.pdf
│   ├── diversity.pdf
│   └── ... (constraint-specific plots)
├── data/
│   └── metrics.csv                  # Raw metric data
└── CSVs/
    ├── population_fitness.csv       # All individuals
    ├── pareto_front.csv             # Non-dominated solutions
    └── hypervolume_statistics.csv   # Multi-run stats
```

---

## 🔍 Key Concepts Explained

### What is the Pareto Front?
The set of **non-dominated solutions** where improving one objective (e.g., reducing hard violations) worsens another (e.g., increasing soft penalties). Represents optimal trade-offs.

### What is Hypervolume?
The **area/volume of objective space** dominated by the Pareto front. Higher values = better convergence + diversity. Gold standard metric for multi-objective optimization.

### What is Spacing?
Standard deviation of nearest-neighbor distances on Pareto front. Lower values = more uniform distribution of solutions, providing smoother trade-off options.

### What is Feasibility Rate?
Percentage of population with **zero hard constraint violations**. Higher = more feasible solutions found.

---

## 📖 Reading Order

### For First-Time Users
1. **00_INDEX.md** - Overview & quick reference
2. **01_PARETO_FRONT_VISUALIZATIONS.md** - Basic concepts
3. Generate your first test run → check plots against docs
4. Return to **02_QUALITY_METRICS.md** and **03_CONVERGENCE_ANALYSIS.md** for deeper understanding

### For Algorithm Developers
1. **03_CONVERGENCE_ANALYSIS.md** - Understand dynamics
2. **00_INDEX.md#algorithm-tuning** - Tuning guidelines
3. **02_QUALITY_METRICS.md** - Performance indicators
4. **01_PARETO_FRONT_VISUALIZATIONS.md** - Trade-off analysis

### For Researchers
1. All mathematical definition sections
2. **00_INDEX.md#mathematical-foundations**
3. CSV data structure and statistical analysis
4. Multi-run comparison methodologies

---

## ⚠️ Common Pitfalls

❌ **Judging success by single metric** (e.g., only hypervolume)  
✅ Use convergence_dashboard.pdf for holistic view

❌ **Ignoring diversity metric** → premature convergence undetected  
✅ Always check diversity.pdf trend

❌ **Comparing absolute hypervolume across different problems**  
✅ Only compare runs with same reference point

❌ **Assuming high spacing is always bad**  
✅ Early generations naturally have high spacing

❌ **Selecting first Pareto solution without checking**  
✅ Use pareto_front_detail.pdf and CSV to choose wisely

---

## 🛠️ Code References

**Plot generation**: `src/exporter/plot*.py`  
**Orchestration**: `src/workflows/reporting.py`  
**Metrics computation**: `src/core/ga_scheduler.py`  
**Styling**: `src/exporter/thesis_style.py`

---

## 📞 Getting Help

**Interpretation questions**: Read relevant section in category files  
**Unexpected patterns**: Check [Troubleshooting Guide](./00_INDEX.md#troubleshooting-guide)  
**Technical issues**: See `CONTRIBUTING.md` in project root  
**Research questions**: Review mathematical definitions and references

---

## 🔗 Related Documentation

- **Algorithm**: `docs/QUICKREF.md`
- **Configuration**: `docs/CONFIG_QUICKSTART.md`
- **Theory**: `docs/for_report/` (thesis chapters)
- **Code**: `.github/copilot-instructions.md`

---

**Version**: 1.0 (2025-11-15)  
**Maintainer**: Schedule Engine Team  
**License**: See project LICENSE
