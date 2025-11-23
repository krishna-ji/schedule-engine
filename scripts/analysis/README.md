# Thesis Analysis Scripts

This directory contains scripts for analyzing and comparing experimental results from the comprehensive thesis framework.

## Scripts

### compare_experiments.py
Comprehensive analysis and visualization of experimental results.

**Features:**
- Load experiment results from consolidated output/experiments/ structure
- Generate comparison table with key metrics
- Plot convergence curves for all methods
- Runtime vs quality trade-off analysis
- Statistical analysis and correlation matrices

**Usage:**
```bash
# Run complete analysis (requires organized structure)
uv run analyze-results

# Organize experiments first if needed
uv run migrate
```

**Outputs:**
- Rich console table with performance comparison
- `output/analysis/convergence_comparison.png` - Evolution curves
- `output/analysis/runtime_vs_quality.png` - Trade-off analysis  
- `output/analysis/statistical_analysis.json` - Statistical summaries

## Required Dependencies

The analysis scripts require additional visualization libraries:
- pandas
- matplotlib
- seaborn

Add to pyproject.toml dependencies:
```toml
[project.optional-dependencies]
analysis = ["pandas>=2.0.0", "matplotlib>=3.7.0", "seaborn>=0.12.0"]
```

Then install with:
```bash
uv sync --extra analysis
```

## Analysis Workflow

1. **Migrate Structure**: Run `uv run migrate` to organize existing experiments
2. **Run Experiments**: Use experimental method commands (`uv run baseline --prod`, etc.)
3. **Analyze Results**: Run `uv run analyze-results` for comprehensive analysis
4. **Review Outputs**: Check `output/analysis/` for plots and statistical summaries
5. **Thesis Integration**: Use consolidated results in thesis report documentation

**⚠️ Note**: Analysis requires consolidated structure at `output/experiments/`. Old scattered structure not supported.

## Metrics Analyzed

- **Hard Constraint Violations**: Primary fitness objective
- **Soft Constraint Penalties**: Secondary optimization target  
- **Runtime Performance**: Computational efficiency comparison
- **Hypervolume**: Multi-objective solution quality
- **IGD (Inverted Generational Distance)**: Convergence quality
- **Convergence Curves**: Evolution over generations

## Statistical Tests

The analysis includes:
- Descriptive statistics (mean, std, min, max)
- Correlation analysis between metrics
- Best performing method identification
- Runtime vs quality trade-offs

## Customization

To add new metrics or visualizations:
1. Modify `load_experiment_results()` to extract additional data
2. Update `generate_comparison_table()` for new columns
3. Add new plotting functions following existing patterns
4. Update `generate_statistical_analysis()` for new statistics