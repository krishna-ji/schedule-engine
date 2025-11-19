# Quick Start: Using the New Evaluation Metrics

##  Immediate Use (No Setup Required)

### Run the Engine
```bash
python main.py --env dev
```

That's it! All metrics are automatically calculated and plotted.

---

##  Where to Find Results

After the run completes, check:
```
output/evaluation_<timestamp>/
├── plots/
│   ├── hypervolume_trend.pdf              ← Phase 1
│   ├── spacing_trend.pdf                  ← Phase 1
│   ├── spacing_distribution.pdf           ← Phase 1
│   ├── spacing_pareto_combined.pdf        ← Phase 1
│   ├── feasibility_evolution.pdf          ← Phase 1
│   ├── convergence_rate_hard_violations.pdf ← Phase 2
│   ├── convergence_multi_metric.pdf       ← Phase 2
│   └── convergence_dashboard.pdf          ← Phase 2 (comprehensive)
└── CSVs/
    ├── hypervolume_trend.csv
    ├── spacing_trend.csv
    ├── convergence_metrics.csv
    └── metrics_statistics.csv
```

---

##  Understanding the Metrics

### Hypervolume (Higher = Better)
- **What it measures:** Quality of Pareto front (convergence + diversity combined)
- **How to read:** Increasing line = algorithm improving
- **Good result:** Final value >> initial value (e.g., +50% to +100%)

### Spacing (Lower = Better)
- **What it measures:** Uniformity of solution distribution
- **How to read:** Decreasing line = more evenly spaced solutions
- **Good result:** Final < 0.02 (excellent), < 0.05 (good)

### Feasibility Rate (Higher = Better)
- **What it measures:** % of population with zero hard violations
- **How to read:** Should approach 100%
- **Good result:** Final > 90% (excellent), > 75% (good)

### Pareto Front Size (More = Better Diversity)
- **What it measures:** Number of non-dominated solutions
- **How to read:** More solutions = more trade-off options
- **Good result:** 15-30 solutions typical for pop size 50-100

---

##  The Dashboard View

**`convergence_dashboard.pdf`** shows 6 key metrics in one view:
1. **Top-Left:** Hard constraints (should decrease)
2. **Top-Middle:** Soft constraints (should decrease)
3. **Top-Right:** Population diversity (should stabilize)
4. **Bottom-Left:** Hypervolume (should increase)
5. **Bottom-Middle:** Spacing (should decrease)
6. **Bottom-Right:** Feasibility rate (should increase to 100%)

---

##  For Your Thesis

### Which Metrics to Report?

**Minimum (Phase 1):**
- Hypervolume: Shows overall quality improvement
- Spacing: Shows solution distribution quality
- Feasibility Rate: Shows constraint satisfaction success

**Recommended (Phase 1 + 2):**
- All above PLUS
- Convergence dashboard plot
- Final Pareto front size

### How to Report?

**In Text:**
```
The algorithm achieved a hypervolume of 3842.91, representing a 107% 
improvement over the initial population (1250.34). Final spacing of 
0.012 indicates excellent solution uniformity (< 0.02 threshold), with 
94% of the final population satisfying all hard constraints.
```

**In Tables:**
```
| Metric            | Initial | Final  | Improvement |
|-------------------|---------|--------|-------------|
| Hypervolume       | 1250.34 | 3842.91| +107%       |
| Spacing           | 0.089   | 0.012  | -86%        |
| Feasibility Rate  | 12%     | 94%    | +683%       |
| Pareto Front Size | 8       | 23     | +188%       |
```

**In Figures:**
- Include: `convergence_dashboard.pdf` (comprehensive overview)
- Include: `hypervolume_trend.pdf` (key quality indicator)
- Optional: `spacing_pareto_combined.pdf` (shows solution distribution)

---

##  For Research/Comparison

### Comparing Two Algorithms

**Step 1:** Run both configurations
```bash
# Run baseline
python main.py --env dev --config configs/baseline.yaml

# Run enhanced
python main.py --env dev --config configs/enhanced.yaml
```

**Step 2:** Extract metrics from both runs
```python
# In your analysis script
import pandas as pd

# Load hypervolume data
baseline_hv = pd.read_csv("output/baseline/CSVs/hypervolume_trend.csv")
enhanced_hv = pd.read_csv("output/enhanced/CSVs/hypervolume_trend.csv")

# Compare final values
print(f"Baseline final HV: {baseline_hv['Hypervolume'].iloc[-1]:.2f}")
print(f"Enhanced final HV: {enhanced_hv['Hypervolume'].iloc[-1]:.2f}")
```

### Multi-Run Statistical Analysis

**Step 1:** Run 10-30 independent runs
```bash
for i in {1..10}; do
    python main.py --env dev
done
```

**Step 2:** Use statistical comparison tools
```python
from src.exporter.plot_metrics_comparison import *

# Collect data from all runs
runs_hv = [run1.hypervolume, run2.hypervolume, ..., run10.hypervolume]
runs_spacing = [run1.spacing, run2.spacing, ..., run10.spacing]

runs_data = {
    "hypervolume": runs_hv,
    "spacing": runs_spacing
}

# Generate statistical plots
plot_metrics_boxplot(runs_data, "multi_run_output/", generation=-1)
generate_statistical_summary_table(runs_data, "multi_run_output/")
```

**Step 3:** Report statistics
```
Mean Hypervolume: 3842.91 ± 123.45 (95% CI: [3765.12, 3920.70])
Success Rate (HC=0): 90% (9/10 runs achieved feasibility)
```

---

##  Troubleshooting

### "I don't see the new plots"
Check:
1. Did the run complete successfully?
2. Look in: `output/evaluation_<timestamp>/plots/`
3. Check for errors in terminal output

### "Hypervolume is 0.0"
This happens if:
- Population is empty (shouldn't occur)
- All solutions dominate reference point (very rare)
- Usually indicates a bug - check terminal output

### "Spacing is very high"
High spacing (> 0.1) means:
- Solutions are clustered (not uniformly distributed)
- May need more generations
- Or increase population diversity

### "Feasibility rate is low"
Low feasibility (< 50%) suggests:
- Problem is highly constrained
- May need more generations
- Consider enabling repair heuristics
- Check feasibility report for bottlenecks

---

##  Academic References

When citing these metrics in your thesis:

**Hypervolume:**
> While, L., Hingston, P., Barone, L., & Huband, S. (2006). A faster algorithm for calculating hypervolume. IEEE Transactions on Evolutionary Computation, 10(1), 29-38.

**Spacing:**
> Schott, J. R. (1995). Fault tolerant design using single and multicriteria genetic algorithm optimization. Doctoral dissertation, Massachusetts Institute of Technology.

**NSGA-II (General):**
> Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197.

**IGD:**
> Coello Coello, C. A., & Sierra, M. R. (2004). A study of the parallelization of a coevolutionary multi-objective evolutionary algorithm. In Mexican International Conference on Artificial Intelligence (pp. 688-697).

---

##  Pro Tips

1. **Always check the dashboard first** - `convergence_dashboard.pdf` gives you the full picture at a glance

2. **CSV data is your friend** - All metrics exportable for custom analysis in Excel/Python

3. **Compare against baseline** - Run simple random/greedy algorithm first to show improvement

4. **Multi-run is worth it** - 10 runs gives you statistical confidence, 30 is excellent

5. **Look for patterns** - HV should increase, Spacing should decrease, Feasibility should increase

---

##  Need Help?

- **Documentation:** See `docs/for_report/evaluation_metrics_comprehensive.md` for detailed explanations
- **Code Examples:** Check docstrings in `src/metrics/` and `src/exporter/` modules
- **Changelog:** See `docs/code/ENHANCE.md` for implementation details

---

##  You're Ready!

Just run `python main.py --env dev` and explore the new metrics in the output directory. No configuration needed!
