<!-- Suggested thesis placement: Chapter 4 - Results and Evaluation, Section 4.2 - Performance Metrics -->

## Multi-Objective Optimization Evaluation Metrics

### Overview

Evaluating the quality of solutions in multi-objective optimization presents unique challenges compared to single-objective problems. While traditional optimization can be assessed using simple improvement metrics, multi-objective algorithms must balance multiple conflicting objectives simultaneously. This necessitates specialized quality indicators that capture both **convergence** (proximity to optimal solutions) and **diversity** (spread of solutions across the Pareto front).

This chapter describes the comprehensive suite of evaluation metrics implemented for assessing the NSGA-II-based university course scheduling algorithm. These metrics are organized into three phases based on their importance and computational complexity.

---

### 4.2.1 Phase 1: Essential Quality Indicators

#### Hypervolume Indicator (HV)

**Definition**: The hypervolume indicator measures the volume of objective space dominated by a Pareto front relative to a reference point. It is widely considered the gold standard metric for multi-objective optimization quality assessment.

**Mathematical Formulation**:
$$HV(PF, r) = \lambda\left(\bigcup_{x \in PF} [f_1(x), r_1] \times [f_2(x), r_2]\right)$$

where $PF$ is the Pareto front, $r = (r_1, r_2)$ is the reference point, and $\lambda$ denotes the Lebesgue measure (volume).

**Key Properties**:
- **Monotonic**: Adding non-dominated solutions always increases hypervolume
- **Combines Convergence and Diversity**: Higher values indicate both better convergence to the optimal front and wider solution spread
- **Sensitive to Reference Point**: Requires consistent reference point for meaningful comparison across generations

**Implementation**: The system uses DEAP's built-in WFG algorithm for efficient hypervolume calculation. The reference point is computed as $(1.1 \cdot \max(HC) + 1.0, 1.1 \cdot \max(SP) + 1.0)$ where $HC$ and $SP$ are hard constraint violations and soft penalties respectively.

**Visualization**: Line graph showing hypervolume evolution over generations. Higher final values and steeper initial slopes indicate better algorithm performance.

**Interpretation**:
- Increasing HV → Algorithm improving (finding better or more diverse solutions)
- Plateauing HV → Convergence achieved
- Decreasing HV → Infeasible (should never occur with proper reference point)

---

#### Spacing Metric (S)

**Definition**: The spacing metric quantifies the uniformity of solution distribution along the Pareto front. Lower values indicate more evenly distributed solutions, providing better trade-off options for decision makers.

**Mathematical Formulation**:
$$S = \sqrt{\frac{1}{|PF|-1} \sum_{i=1}^{|PF|} (d_i - \bar{d})^2}$$

where:
- $d_i = \min_{j \neq i} ||f(x_i) - f(x_j)||$ is the Euclidean distance to the nearest neighbor
- $\bar{d} = \frac{1}{|PF|} \sum_{i=1}^{|PF|} d_i$ is the mean distance

**Key Properties**:
- **Scale-Independent**: Measures relative uniformity, not absolute distances
- **Zero-Optimal**: $S = 0$ indicates perfectly uniform spacing
- **Sensitive to Outliers**: A few poorly spaced solutions significantly affect the metric

**Visualization**: 
1. **Line Graph**: Spacing evolution over generations (decreasing trend desired)
2. **Histogram**: Distribution of nearest-neighbor distances in final Pareto front
3. **Combined View**: Pareto front scatter plot with spacing value annotated

**Interpretation**:
- $S < 0.01$ → Excellent uniformity
- $0.01 \leq S < 0.05$ → Good uniformity
- $S \geq 0.05$ → Poor uniformity (clustered solutions)

---

#### Constraint Satisfaction Rate (CSR)

**Definition**: The percentage of population individuals with zero hard constraint violations, indicating the proportion of feasible solutions.

**Mathematical Formulation**:
$$CSR = \frac{|\{x \in Pop : HC(x) = 0\}|}{|Pop|} \times 100\%$$

where $HC(x)$ represents hard constraint violations for individual $x$.

**Key Properties**:
- **Interpretability**: Direct measure of feasibility success
- **Search Progress Indicator**: Tracks algorithm's ability to find feasible regions
- **Problem Difficulty Indicator**: Consistently low CSR suggests over-constrained problem

**Visualization**: Line graph showing feasibility rate evolution with 100% reference line. Area fill emphasizes improvement.

**Interpretation**:
- $CSR = 100\%$ → All solutions feasible (ideal scenario)
- $50\% \leq CSR < 100\%$ → Partial feasibility (acceptable for complex problems)
- $CSR < 50\%$ → Difficulty finding feasible solutions

---

#### Pareto Front Size (#PF)

**Definition**: The count of non-dominated solutions in the final Pareto front, indicating the variety of trade-off options available.

**Mathematical Formulation**:
$$\#PF = |\{x \in Pop : \nexists y \in Pop, y \prec x\}|$$

where $y \prec x$ denotes Pareto dominance.

**Interpretation**:
- **More solutions** → Greater diversity, more decision-maker choices
- **Diminishing returns**: Beyond 50-100 solutions, marginal utility decreases
- **Population-dependent**: Typically 5-20% of population size for well-tuned NSGA-II

---

### 4.2.2 Phase 2: Advanced Convergence Metrics

#### Inverted Generational Distance (IGD)

**Definition**: IGD measures the average distance from a reference Pareto front to the obtained front. Unlike GD (Generational Distance), IGD penalizes both poor convergence and incomplete coverage of the Pareto front.

**Mathematical Formulation**:
$$IGD(PF, PF_{ref}) = \frac{1}{|PF_{ref}|} \sqrt{\sum_{r \in PF_{ref}} \min_{x \in PF} ||f(r) - f(x)||^2}$$

where $PF_{ref}$ is the reference (true or best-known) Pareto front.

**Key Properties**:
- **Preferred over GD**: Penalizes missing regions of Pareto front
- **Requires Reference Front**: Uses initial population or best-known approximation
- **Lower is Better**: $IGD = 0$ indicates obtained front perfectly covers reference

**Practical Challenges**: True Pareto front unknown for NP-hard scheduling problems. Implementation uses initial population Pareto front as reference, measuring improvement relative to starting point.

---

#### Spread (Δ)

**Definition**: The spread metric assesses both the extent (coverage of extreme points) and uniformity of solution distribution.

**Mathematical Formulation**:
$$\Delta = \frac{d_f + d_l + \sum_{i=1}^{N-1}|d_i - \bar{d}|}{d_f + d_l + (N-1)\bar{d}}$$

where:
- $d_f, d_l$ are distances to extreme points in each objective
- $d_i$ are consecutive distances between sorted solutions
- $\bar{d}$ is mean consecutive distance

**Key Properties**:
- **Ideal Value**: $\Delta = 0$ indicates perfect extent and uniformity
- **Complements Spacing**: Spacing measures uniformity only; spread includes extent
- **Formulated by Deb et al.**: Part of original NSGA-II quality assessment

---

#### Convergence Rate (CR)

**Definition**: The average improvement per generation over a sliding window, measuring optimization dynamics.

**Mathematical Formulation**:
$$CR(t, w) = \frac{HC(t) - HC(t+w)}{w}$$

where $HC(t)$ is hard violations at generation $t$ and $w$ is window size (typically 10).

**Visualization**: Bar chart with color-coding:
- **Green bars**: Positive rate (improving)
- **Yellow bars**: Near-zero rate (stagnating)
- **Red bars**: Negative rate (degrading - should not occur)

**Applications**:
- Detect stagnation for adaptive mechanisms (hypermutation, restart)
- Estimate remaining generations to feasibility
- Compare algorithm configurations by early-phase convergence speed

---

### 4.2.3 Phase 3: Statistical Analysis

#### Multi-Run Statistical Summary

For robust algorithm evaluation, multiple independent runs with different random seeds are essential. The system generates:

**Central Tendency**:
- **Mean**: Average performance across runs
- **Median**: Robust to outliers

**Dispersion**:
- **Standard Deviation**: Performance variability
- **Interquartile Range (Q1-Q3)**: Robust spread measure

**Confidence Intervals**:
- **95% CI**: $\bar{x} \pm t_{0.975, n-1} \cdot \frac{s}{\sqrt{n}}$
- Provides statistical significance bounds

**Box Plot Visualization**: Shows distribution, outliers, and quartiles for each metric across runs.

---

#### Algorithm Comparison Framework

**t-Test for Significance**:
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

**Effect Size (Cohen's d)**:
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$$

Interpretation:
- $d < 0.2$: Negligible difference
- $0.2 \leq d < 0.5$: Small effect
- $0.5 \leq d < 0.8$: Medium effect
- $d \geq 0.8$: Large effect

**Success Rate**:
$$SR(\theta) = \frac{|\{run : \min(HC_{run}) \leq \theta\}|}{|Runs|} \times 100\%$$

Measures percentage of runs achieving hard violations below threshold $\theta$ (e.g., $\theta = 0$ for feasibility).

---

### 4.2.4 Implementation Architecture

**Metric Calculation Modules**:
```
src/metrics/
├── hypervolume.py          # HV calculation with DEAP
├── pareto_metrics.py       # Spacing, IGD, GD, spread, epsilon
├── convergence.py          # CR, CSR, stagnation detection
└── diversity.py            # Gene-level population diversity (existing)
```

**Visualization Modules**:
```
src/exporter/
├── plot_hypervolume.py     # HV trends, multi-run comparisons
├── plot_spacing.py         # Spacing trends, distributions
├── plot_convergence.py     # Multi-metric dashboard, convergence rates
└── plot_metrics_comparison.py  # Statistical box plots, t-tests
```

**Integration Points**:
1. **GAScheduler._track_metrics()**: Computes all metrics per generation
2. **GAMetrics dataclass**: Stores metric histories as lists
3. **generate_reports()**: Generates plots after evolution completes

---

### 4.2.5 Computational Complexity

| Metric | Complexity | Notes |
|--------|------------|-------|
| Hypervolume | $O(n \log n)$ | WFG algorithm (2D case) |
| Spacing | $O(n^2)$ | All-pairs distance computation |
| IGD | $O(|PF| \cdot |PF_{ref}|)$ | Reference front size-dependent |
| Spread | $O(n \log n)$ | Dominated by sorting |
| Diversity | $O(n^2 \cdot g)$ | Gene-level distance, expensive |

For typical population sizes (50-200), metric calculation overhead is < 1% of total runtime.

---

### 4.2.6 Practical Guidelines

**When to Use Each Metric**:

| Research Question | Recommended Metrics |
|-------------------|---------------------|
| "Which algorithm is better?" | HV, IGD, Success Rate |
| "Are solutions well-distributed?" | Spacing, Spread |
| "How fast does it converge?" | Convergence Rate, Gen-to-Target |
| "Is it reliable?" | Multi-run Statistics, 95% CI |
| "Are solutions feasible?" | CSR, Success Rate (θ=0) |

**Reporting Best Practices**:
1. Always report multiple metrics (avoid metric bias)
2. Include confidence intervals for multi-run experiments
3. Use box plots to show distribution, not just means
4. Report both final values and convergence speed
5. Compare against baseline (random search, greedy heuristic)

---

### 4.2.7 Example Results Interpretation

**Sample Output** (100 generations, population=50):
```
Hypervolume:       Initial: 1250.34  →  Final: 3842.91 (+107%)
Spacing:           Initial: 0.089    →  Final: 0.012   (-86%)
CSR:               Initial: 12%      →  Final: 94%
Pareto Front Size: Initial: 8        →  Final: 23
IGD:               Initial: 0.0      →  Final: 0.034
```

**Interpretation**:
- **Hypervolume**: Significant improvement indicates good convergence and diversity gains
- **Spacing**: Excellent final uniformity (< 0.02 threshold)
- **CSR**: High feasibility rate (94% of population feasible)
- **PF Size**: Sufficient diversity (23 solutions provide adequate trade-off options)
- **IGD**: Positive value expected (moving away from poor initial reference)

**Conclusion**: Algorithm successfully converged to high-quality, well-distributed Pareto front with strong feasibility.

---

### References

1. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. *IEEE Transactions on Evolutionary Computation*, 6(2), 182-197.

2. Zitzler, E., Thiele, L., Laumanns, M., Fonseca, C. M., & Da Fonseca, V. G. (2003). Performance assessment of multiobjective optimizers: An analysis and review. *IEEE Transactions on Evolutionary Computation*, 7(2), 117-132.

3. While, L., Hingston, P., Barone, L., & Huband, S. (2006). A faster algorithm for calculating hypervolume. *IEEE Transactions on Evolutionary Computation*, 10(1), 29-38.

4. Schott, J. R. (1995). *Fault tolerant design using single and multicriteria genetic algorithm optimization*. Doctoral dissertation, Massachusetts Institute of Technology.

5. Coello Coello, C. A., & Sierra, M. R. (2004). A study of the parallelization of a coevolutionary multi-objective evolutionary algorithm. In *Mexican International Conference on Artificial Intelligence* (pp. 688-697).
