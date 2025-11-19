# NSGA-II Schedule Engine: Figure Interpretation Guide

> **Complete reference for all generated visualizations**  
> **Location**: `output/evaluation_<timestamp>/plots/`  
> **Purpose**: Understanding algorithm performance and solution quality

---

## Quick Navigation

| Category | Documentation File | Key Plots |
|----------|-------------------|-----------|
| **Pareto Front** | [01_PARETO_FRONT_VISUALIZATIONS.md](./01_PARETO_FRONT_VISUALIZATIONS.md) | Solution trade-offs, population distribution |
| **Quality Metrics** | [02_QUALITY_METRICS.md](./02_QUALITY_METRICS.md) | Hypervolume, spacing, feasibility |
| **Convergence** | [03_CONVERGENCE_ANALYSIS.md](./03_CONVERGENCE_ANALYSIS.md) | Algorithm dynamics, improvement rates |

---

## All Generated Figures (Alphabetical)

### Core Visualizations (Always Generated)

| Figure | Description | Key Insight | Doc Reference |
|--------|-------------|-------------|---------------|
| **convergence_dashboard.pdf** | 2×3 grid of all major metrics | Overall algorithm health | [03](./03_CONVERGENCE_ANALYSIS.md#1-convergence_dashboardpdf) |
| **diversity.pdf** | Population genetic diversity | Premature convergence detection | [03](./03_CONVERGENCE_ANALYSIS.md#6-diversitypdf) |
| **hard_constraint_trend.pdf** | Hard violations over time | Feasibility achievement | [03](./03_CONVERGENCE_ANALYSIS.md#4-hard_constraint_trendpdf) |
| **pareto_front.pdf** | Basic Pareto front view | Solution trade-offs | [01](./01_PARETO_FRONT_VISUALIZATIONS.md#1-pareto_frontpdf) |
| **pareto_front_comprehensive.pdf** | 4-subplot Pareto analysis | Population structure & convergence | [01](./01_PARETO_FRONT_VISUALIZATIONS.md#3-pareto_front_comprehensivepdf) |
| **pareto_front_detail.pdf** | Zoomed Pareto front with labels | Solution selection | [01](./01_PARETO_FRONT_VISUALIZATIONS.md#2-pareto_front_detailpdf) |
| **soft_constraint_trend.pdf** | Soft penalties over time | Optimization quality | [03](./03_CONVERGENCE_ANALYSIS.md#5-soft_constraint_trendpdf) |

### Advanced Metrics (Conditionally Generated)

| Figure | Description | Key Insight | Doc Reference |
|--------|-------------|-------------|---------------|
| **convergence_multi_metric.pdf** | Normalized metrics comparison | Metric interactions | [03](./03_CONVERGENCE_ANALYSIS.md#2-convergence_multi_metricpdf) |
| **convergence_rate_hard_violations.pdf** | Improvement rate analysis | Stagnation detection | [03](./03_CONVERGENCE_ANALYSIS.md#3-convergence_rate_hard_violationspdf) |
| **feasibility_evolution.pdf** | Feasible solutions percentage | Constraint satisfaction | [02](./02_QUALITY_METRICS.md#7-feasibility_evolutionpdf) |
| **hypervolume_trend.pdf** | Hypervolume over generations | Overall quality (gold standard) | [02](./02_QUALITY_METRICS.md#1-hypervolume_trendpdf) |
| **spacing_distribution.pdf** | Histogram of solution spacing | Distribution uniformity | [02](./02_QUALITY_METRICS.md#3-spacing_distributionpdf) |
| **spacing_pareto_combined.pdf** | Pareto + spacing dual view | Geometric distribution | [02](./02_QUALITY_METRICS.md#4-spacing_pareto_combinedpdf) |
| **spacing_trend.pdf** | Spacing metric over time | Solution spread quality | [02](./02_QUALITY_METRICS.md#2-spacing_trendpdf) |

---

## Quick Reference by Use Case

### 1. **Quick Health Check** (2 minutes)
**Goal**: Is the run successful?

**Check these plots**:
1. **pareto_front.pdf** → Are there solutions with low hard violations?
2. **convergence_dashboard.pdf** → Are trends improving (hard ↓, hypervolume ↑)?
3. **hard_constraint_trend.pdf** → Did hard violations reach 0?

**Good run indicators**:
-  Pareto front has solutions at (0, low_soft)
-  Hard violations trend reaches 0
-  Hypervolume increasing
-  Feasibility rate > 60%

**Problem indicators**:
-  All solutions have hard > 5
-  Flat lines on all metrics after gen 100
-  Feasibility rate < 20%

---

### 2. **Solution Selection** (5 minutes)
**Goal**: Pick best schedule for deployment

**Check these plots**:
1. **pareto_front_detail.pdf** → See all non-dominated options with labels
2. **CSVs/pareto_front.csv** → Cross-reference solution indices
3. **pareto_front_comprehensive.pdf** (Subplot 4) → Check solution frequency

**Selection criteria**:
- **Zero hard violations**: Mandatory for deployment
- **Low soft penalty**: Better quality
- **High frequency** (large dots): Algorithm confident in this solution

**Typical choice**: Solution with hard = 0 and minimum soft penalty

---

### 3. **Algorithm Tuning** (15 minutes)
**Goal**: Improve algorithm performance

**Check these plots**:
1. **convergence_rate_hard_violations.pdf** → Identify stagnation percentage
2. **diversity.pdf** → Check premature convergence
3. **spacing_trend.pdf** → Assess distribution quality
4. **convergence_dashboard.pdf** → Overall dynamics

**Tuning actions**:

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Premature convergence** | Diversity < 5%, early plateau | Increase mutation rate, population size |
| **Stagnation** | Rate plot shows > 60% yellow bars | Add local search, more generations |
| **Poor spacing** | Spacing > 2.0, gaps in front | Adjust selection pressure, use crowding |
| **Infeasibility** | Hard never reaches 0 | Check constraints, improve repair operators |

---

### 4. **Research Analysis** (30+ minutes)
**Goal**: Deep understanding of algorithm behavior

**Check all plots, focusing on**:
1. **convergence_multi_metric.pdf** → Metric trade-offs and interactions
2. **pareto_front_comprehensive.pdf** → Population structure analysis
3. **spacing_distribution.pdf** → Statistical distribution properties
4. **CSV files** → Raw data for statistical tests

**Research questions addressed**:
- How do different metrics interact during evolution?
- What is the convergence speed (generations to optimality)?
- How uniform is the final Pareto front distribution?
- What percentage of generations show improvement?

---

### 5. **Stakeholder Presentation** (Report to non-technical audience)
**Goal**: Communicate results clearly

**Use these plots**:
1. **pareto_front.pdf** → "Here are the trade-off options"
2. **hard_constraint_trend.pdf** → "We achieved feasibility"
3. **soft_constraint_trend.pdf** → "Quality improved over time"
4. **convergence_dashboard.pdf** (simplified explanation) → "Overall performance"

**Avoid**: Mathematical details, spacing distribution, convergence rates

**Emphasize**: 
- Number of feasible solutions found
- Trade-off between must-have and nice-to-have constraints
- Confidence in results (frequency, convergence)

---

## Understanding the Axes

### Common X-Axes

| X-Axis Label | Meaning | Range |
|--------------|---------|-------|
| **Generation** | Evolutionary iteration (0 = initial random population) | 0 to max_generations (typically 30-2000) |
| **Hard Constraint Violations** | Count of must-satisfy constraint failures | 0 (perfect) to 100+ (poor) |

### Common Y-Axes

| Y-Axis Label | Meaning | Better Direction | Typical Range |
|--------------|---------|------------------|---------------|
| **Hard Violations** | Must-satisfy constraint failures | ↓ Lower | 0-50 |
| **Soft Constraint Penalty** | Prefer-satisfy penalty score | ↓ Lower | 0-500 |
| **Hypervolume Indicator** | Quality measure (area dominated) | ↑ Higher | Problem-dependent |
| **Spacing Metric** | Distribution uniformity | ↓ Lower | 0-10 |
| **Diversity** | Population genetic variation | Moderate | 0-100 |
| **Feasibility Rate (%)** | Percentage of feasible solutions | ↑ Higher | 0-100 |

---

## Plot Categories

### By Purpose

#### **Trade-Off Analysis**
Understand solution space and options
- pareto_front.pdf
- pareto_front_detail.pdf
- pareto_front_comprehensive.pdf

#### **Quality Assessment**
Evaluate algorithm performance
- hypervolume_trend.pdf
- spacing_trend.pdf
- spacing_distribution.pdf
- feasibility_evolution.pdf

#### **Convergence Monitoring**
Track optimization dynamics
- convergence_dashboard.pdf
- convergence_multi_metric.pdf
- convergence_rate_hard_violations.pdf
- hard_constraint_trend.pdf
- soft_constraint_trend.pdf
- diversity.pdf

#### **Distribution Analysis**
Assess solution spread and uniformity
- spacing_distribution.pdf
- spacing_pareto_combined.pdf
- pareto_front_comprehensive.pdf (subplots 3, 4)

---

### By Complexity Level

#### **Beginner** (Easy to interpret)
- pareto_front.pdf
- hard_constraint_trend.pdf
- soft_constraint_trend.pdf
- feasibility_evolution.pdf

#### **Intermediate** (Requires MOEA knowledge)
- pareto_front_comprehensive.pdf
- convergence_dashboard.pdf
- hypervolume_trend.pdf
- spacing_trend.pdf

#### **Advanced** (Requires deep MOEA understanding)
- convergence_multi_metric.pdf
- convergence_rate_hard_violations.pdf
- spacing_distribution.pdf
- pareto_front_comprehensive.pdf (density subplot)

---

## Mathematical Foundations

### Core Concepts

#### **Pareto Dominance**
Solution A dominates solution B if A is no worse in all objectives and strictly better in at least one:
$$
A \succ B \iff \forall i: f_i(A) \leq f_i(B) \text{ and } \exists j: f_j(A) < f_j(B)
$$

#### **Pareto Front**
Set of all non-dominated solutions in the population:
$$
\mathcal{P} = \{x \in P \mid \nexists y \in P: y \succ x\}
$$

#### **Hypervolume Indicator**
Volume of objective space dominated by Pareto front:
$$
HV(\mathcal{P}, \vec{r}) = \lambda\left(\bigcup_{x \in \mathcal{P}} \{y \mid x \preceq y \preceq \vec{r}\}\right)
$$

#### **Spacing Metric**
Standard deviation of nearest-neighbor distances:
$$
S = \sqrt{\frac{1}{|\mathcal{P}|-1} \sum_{i=1}^{|\mathcal{P}|} (d_i - \bar{d})^2}
$$

#### **Diversity Metric**
Average pairwise Hamming distance:
$$
D = \frac{2}{n(n-1)} \sum_{i<j} d_H(x_i, x_j)
$$

---

## Data Export Structure

### CSV Files Location
All metrics saved to: `output/evaluation_<timestamp>/data/metrics.csv`

**Columns**:
```csv
generation,hard_total,soft_total,diversity,hypervolume,spacing,feasibility_rate
0,45,890.5,68.3,1234.5,2.45,12.0
1,38,876.2,65.1,1345.6,2.38,18.5
...
```

### Pareto-Specific CSVs

**Population fitness**: `CSVs/population_fitness.csv`
```csv
Individual_Index,Hard_Constraint_Violations,Soft_Constraint_Penalties
```

**Pareto front**: `CSVs/pareto_front.csv`
```csv
Pareto_Index,Hard_Constraint_Violations,Soft_Constraint_Penalties
```

**Hypervolume statistics** (multi-run): `CSVs/hypervolume_statistics.csv`
```csv
Generation,Mean,Std,Min,Max,CI_Lower,CI_Upper
```

---

## Troubleshooting Guide

### Issue: All plots show flat lines

**Possible causes**:
1. Algorithm stalled (check diversity - if near 0, premature convergence)
2. Insufficient generations (compare to typical run length)
3. Local optimum reached (check if metrics are at reasonable values)

**Actions**:
- Increase generations
- Increase mutation rate
- Enable local search operators

---

### Issue: Pareto front has only 1-2 solutions

**Possible causes**:
1. Premature convergence (check diversity trend)
2. Problem has limited trade-off space
3. Selection pressure too high

**Actions**:
- Increase population size
- Reduce selection pressure (increase tournament size)
- Use diversity preservation mechanisms

---

### Issue: Hard violations never reach 0

**Possible causes**:
1. Problem is over-constrained (infeasible)
2. Repair operators insufficient
3. Search space too large

**Actions**:
- Validate input data (check `validation/` output)
- Review constraint definitions
- Enable/improve IGLS repair system
- Check feasibility_evolution.pdf (if < 20%, likely infeasible)

---

### Issue: Soft penalties not decreasing

**Possible causes**:
1. All solutions infeasible (hard > 0 dominates selection)
2. Soft constraints conflicting with hard constraints
3. Fitness weights need adjustment

**Actions**:
- First ensure hard constraints satisfied
- Check constraint compatibility
- Adjust fitness weights if hard constraints are satisfied

---

### Issue: Spacing increasing over time

**Possible causes**:
1. Losing diversity (solutions converging to few points)
2. Pareto front shrinking
3. Normal early-generation behavior

**Actions**:
- Check diversity metric (confirm losing variation)
- Use crowding distance in selection
- If only early generations, this is normal

---

## Configuration Impact on Plots

### Key Parameters

| Parameter | Affects Plots | Tuning Impact |
|-----------|---------------|---------------|
| `max_generations` | X-axis range on all time series | Longer runs → more refined solutions |
| `population_size` | Density in Pareto plots | Larger → more diverse front |
| `mutation_rate` | Diversity, spacing | Higher → better exploration |
| `crossover_rate` | Convergence speed | Higher → faster convergence |
| `tournament_size` | Selection pressure | Larger → faster convergence, less diversity |
| `use_local_search` | Convergence speed, quality | Enabled → faster, better optimization |

### Typical Configurations

**Fast Test** (`configs/test.yaml`):
- 30 generations, 10 population
- Plots show early-stage behavior only
- Use for: Debugging, quick feasibility checks

**Production** (`configs/prod.yaml`):
- 2000 generations, 200 population
- Plots show full convergence
- Use for: Final schedules, research analysis

---

## Best Practices

### 1. Always Check Multiple Plots
Don't rely on single metric:
- High hypervolume alone insufficient (could be poor spacing)
- Low hard violations alone insufficient (could have poor soft optimization)
- Use convergence_dashboard.pdf for holistic view

### 2. Compare Across Runs
Single run can be misleading:
- Run same config 3-5 times
- Use multi-run plots (hypervolume_multi_run.pdf, spacing_multi_run.pdf)
- Report mean ± std deviation

### 3. Use CSV Data for Statistics
Plots are for visualization, CSVs for analysis:
- Statistical tests (t-tests, Mann-Whitney)
- Confidence intervals
- Detailed numerical comparisons

### 4. Match Plot Choice to Audience
- **Developers**: All plots, focus on convergence
- **Researchers**: Metrics, spacing, multi-run comparisons
- **Stakeholders**: Pareto front, basic trends only
- **Documentation**: Representative subset with clear captions

---

## Plot Generation Technical Details

### Code Locations
- **Pareto**: `src/exporter/plotpareto.py`
- **Quality metrics**: `src/exporter/plot_hypervolume.py`, `plot_spacing.py`
- **Convergence**: `src/exporter/plot_convergence.py`
- **Basic trends**: `src/exporter/plothard.py`, `plotsoft.py`, `plotdiversity.py`

### Orchestration
- **Entry point**: `src/workflows/reporting.py::generate_visualizations()`
- **Parallel execution**: Uses ThreadPoolExecutor with 8 workers
- **Error handling**: Failed plots logged, don't stop overall generation

### Styling
- **Theme**: Thesis-quality style via `src/exporter/thesis_style.py`
- **Colors**: Consistent palette (blue, red, green, orange, purple, cyan)
- **Fonts**: Sans-serif, appropriate sizes for publication
- **Resolution**: High-quality PDF output (vector graphics)

---

## Citation & References

### Related Documentation
- **Algorithm overview**: `docs/QUICKREF.md`
- **Configuration**: `docs/CONFIG_QUICKSTART.md`
- **NSGA-II details**: `docs/for_report/` (thesis chapters)

### Key Papers

**NSGA-II Algorithm**:
- Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"

**Hypervolume Indicator**:
- Zitzler, E., & Thiele, L. (1999). "Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach"

**Spacing Metric**:
- Schott, J. R. (1995). "Fault tolerant design using single and multicriteria genetic algorithm optimization"

---

## FAQ

**Q: Why are some plots missing?**  
A: Advanced plots (spacing, hypervolume) only generated if metrics computed during run. Check `configs/base.yaml` for `metrics.compute_*` settings.

**Q: What's a "good" hypervolume value?**  
A: Absolute values are problem-dependent. Look for:
- Increasing trend (improvement)
- Plateau after sufficient generations (convergence)
- Higher than baseline/previous runs (comparison)

**Q: Can I generate plots for old runs?**  
A: Yes, if CSV data exists in `data/metrics.csv`. Use plotting functions directly from code with archived data.

**Q: How to export plots for papers?**  
A: PDFs are vector graphics, suitable for publication. For raster, use PDF→PNG conversion at high DPI (300+).

**Q: Why do some dots overlap in Pareto plots?**  
A: Multiple individuals have identical fitness. See count labels or use pareto_front_comprehensive.pdf subplot 1 (jittered view).

---

## Version Information

**Current implementation**: Schedule Engine v2.0 (NSGA-II with IGLS repair)  
**DEAP version**: 1.4.1  
**Python version**: 3.11+  

**Last updated**: 2025-11-15  
**Documentation version**: 1.0

---

## Getting Help

**If plots show unexpected patterns**:
1. Check this guide for interpretation
2. Review CSV data for raw numbers
3. Consult algorithm documentation in `docs/`
4. Verify configuration in `configs/*.yaml`

**For technical issues**:
- Check `output/evaluation_<timestamp>/run.log`
- Review error messages in terminal output
- Consult `CONTRIBUTING.md` for reporting bugs

---

**End of Guide** — For detailed explanations, see individual category documentation files.
