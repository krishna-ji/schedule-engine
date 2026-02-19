"""
Convergence and Statistical Metrics

This module implements metrics for tracking optimization dynamics and statistical analysis:

1. **Convergence Rate**: Speed of improvement over generations
2. **Constraint Satisfaction Rate**: Percentage of feasible solutions
3. **Statistical Analysis**: Mean, std, confidence intervals across runs
4. **Stagnation Detection**: Identify optimization plateaus

These metrics help:
- Understand optimization dynamics
- Compare algorithm configurations
- Detect premature convergence
- Assess reliability across multiple runs
"""

import numpy as np
from scipy import stats


def calculate_convergence_rate(
    metric_history: list[float], window: int = 10
) -> list[float]:
    """
    Calculate convergence rate (improvement per generation) over sliding window.

    Convergence rate measures how quickly the algorithm improves. Positive values
    indicate improvement (for minimization problems). Declining rates suggest
    approaching convergence or stagnation.

    Args:
        metric_history: List of metric values per generation (e.g., hard violations)
        window: Sliding window size for rate calculation (default 10)

    Returns:
        List[float]: Convergence rates. Length = len(metric_history) - window.
                     Positive = improvement, negative = degradation, ~0 = stagnation.

    Example:
        >>> rates = calculate_convergence_rate(metrics.hard_violations, window=10)
        >>> avg_rate = np.mean(rates)
        >>> print(f"Average improvement: {avg_rate:.2f} violations/gen")

    Note:
        For minimization: rate = (value_old - value_new) / window
        Positive rate = good (values decreasing)
    """
    if len(metric_history) < window + 1:
        return []

    rates = []
    for i in range(len(metric_history) - window):
        # Improvement = old - new (positive = better for minimization)
        improvement = metric_history[i] - metric_history[i + window]
        rate = improvement / window  # Average improvement per generation
        rates.append(rate)

    return rates


def calculate_constraint_satisfaction_rate(population: list) -> float:
    """
    Calculate percentage of population with zero hard constraint violations.

    This metric indicates feasibility quality. 100% means entire population is
    feasible. Lower percentages suggest difficulty finding feasible solutions.

    Args:
        population: List of DEAP individuals with fitness.values

    Returns:
        float: Percentage in [0, 100]. Higher is better.
               0 = no feasible solutions, 100 = all feasible.

    Example:
        >>> feas_rate = calculate_constraint_satisfaction_rate(population)
        >>> print(f"Feasibility rate: {feas_rate:.1f}%")

    Note:
        Assumes fitness.values[0] represents hard constraint violations.
        A value of 0 indicates a feasible solution.
    """
    if not population:
        return 0.0

    feasible_count = sum(1 for ind in population if ind.fitness.values[0] == 0)
    rate = (feasible_count / len(population)) * 100.0

    return rate


def detect_stagnation(
    metric_history: list[float], window: int = 10, threshold: float = 0.01
) -> tuple[bool, int]:
    """
    Detect if optimization has stagnated (no significant improvement).

    Stagnation occurs when the best metric hasn't improved by more than threshold
    over the window period. Useful for triggering adaptive mechanisms like
    hypermutation or population restart.

    Args:
        metric_history: List of best metric values per generation
        window: Number of generations to check (default 10)
        threshold: Minimum improvement to consider non-stagnant (default 0.01)

    Returns:
        Tuple[bool, int]: (is_stagnant, stagnation_duration)
            - is_stagnant: True if stagnated
            - stagnation_duration: Number of consecutive stagnant generations

    Example:
        >>> stagnant, duration = detect_stagnation(metrics.hard_violations, window=15)
        >>> if stagnant:
        >>>     print(f"Stagnation detected! No improvement for {duration} generations")
    """
    if len(metric_history) < window:
        return False, 0

    # Check recent window
    recent_values = metric_history[-window:]
    improvement = max(recent_values) - min(recent_values)

    if improvement <= threshold:
        # Count how many generations have been stagnant
        stagnation_duration = 0
        for i in range(len(metric_history) - 1, 0, -1):
            if abs(metric_history[i] - metric_history[i - 1]) <= threshold:
                stagnation_duration += 1
            else:
                break
        return True, stagnation_duration

    return False, 0


def calculate_improvement_percentage(initial_value: float, final_value: float) -> float:
    """
    Calculate percentage improvement from initial to final value.

    Args:
        initial_value: Metric value at generation 0
        final_value: Metric value at final generation

    Returns:
        float: Improvement percentage. Positive = improvement, negative = degradation.
               For minimization problems.

    Example:
        >>> improvement = calculate_improvement_percentage(100.0, 20.0)
        >>> print(f"Improved by {improvement:.1f}%")  # Output: 80.0%
    """
    if initial_value == 0:
        return 0.0 if final_value == 0 else -100.0

    improvement = ((initial_value - final_value) / initial_value) * 100.0
    return improvement


# Statistical Analysis for Multiple Runs
def calculate_run_statistics(
    runs_data: list[list[float]], generation: int = -1
) -> dict:
    """
    Calculate statistical measures across multiple independent runs.

    Provides comprehensive statistics for comparing algorithm reliability and
    performance across multiple runs. Essential for thesis/paper reporting.

    Args:
        runs_data: List of metric histories, one per run
                  E.g., [run1.hard_violations, run2.hard_violations, ...]
        generation: Which generation to analyze (-1 = final generation)

    Returns:
        dict: Statistical measures containing:
            - mean: Average value across runs
            - std: Standard deviation
            - min: Best value
            - max: Worst value
            - median: Median value
            - q1: First quartile (25th percentile)
            - q3: Third quartile (75th percentile)
            - confidence_interval: 95% CI tuple (lower, upper)

    Example:
        >>> stats = calculate_run_statistics([run1.hv, run2.hv, run3.hv])
        >>> print(f"HV: {stats['mean']:.4f} ± {stats['std']:.4f}")
        >>> print(f"95% CI: [{stats['confidence_interval'][0]:.4f}, "
        >>>       f"{stats['confidence_interval'][1]:.4f}]")
    """
    if not runs_data:
        return {}

    # Extract values at specified generation
    values = []
    for run in runs_data:
        if len(run) > 0:
            idx = generation if generation >= 0 else len(run) - 1
            if idx < len(run):
                values.append(run[idx])

    if not values:
        return {}

    values = np.array(values)  # type: ignore[assignment]

    # Calculate statistics
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1))  # Sample std deviation
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    median_val = float(np.median(values))
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))

    # 95% confidence interval
    confidence = 0.95
    if len(values) > 1:
        ci = stats.t.interval(
            confidence, len(values) - 1, loc=mean_val, scale=stats.sem(values)
        )
    else:
        ci = (mean_val, mean_val)

    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "min": float(min_val),
        "max": float(max_val),
        "median": float(median_val),
        "q1": float(q1),
        "q3": float(q3),
        "confidence_interval": (float(ci[0]), float(ci[1])),
        "n_runs": len(values),
    }


def calculate_success_rate(
    runs_data: list[list[float]], threshold: float = 0.0
) -> float:
    """
    Calculate percentage of runs that found solutions below threshold.

    Success rate measures algorithm reliability. For hard constraints,
    threshold=0 means finding feasible solutions.

    Args:
        runs_data: List of metric histories (one per run)
        threshold: Success threshold (e.g., 0 for feasible solutions)

    Returns:
        float: Success rate in [0, 100]

    Example:
        >>> # Check how many runs found feasible solutions
        >>> success = calculate_success_rate(all_runs_hc, threshold=0)
        >>> print(f"Success rate: {success:.1f}%")
    """
    if not runs_data:
        return 0.0

    successful_runs = 0
    for run in runs_data:
        if len(run) > 0 and min(run) <= threshold:
            successful_runs += 1

    rate = (successful_runs / len(runs_data)) * 100.0
    return rate


def compare_algorithm_performance(
    algo1_data: list[list[float]], algo2_data: list[list[float]], generation: int = -1
) -> dict:
    """
    Statistically compare two algorithm configurations.

    Performs t-test and effect size calculation to determine if performance
    difference is statistically significant.

    Args:
        algo1_data: Metric histories for algorithm 1 (multiple runs)
        algo2_data: Metric histories for algorithm 2 (multiple runs)
        generation: Which generation to compare (-1 = final)

    Returns:
        dict: Comparison results containing:
            - algo1_mean: Algorithm 1 mean
            - algo2_mean: Algorithm 2 mean
            - difference: Mean difference (algo1 - algo2)
            - p_value: Statistical significance (< 0.05 = significant)
            - significant: True if p < 0.05
            - effect_size: Cohen's d (0.2=small, 0.5=medium, 0.8=large)
            - winner: "algo1", "algo2", or "tie"

    Example:
        >>> result = compare_algorithm_performance(nsga2_runs, random_runs)
        >>> if result['significant']:
        >>>     print(f"Winner: {result['winner']} (p={result['p_value']:.4f})")
        >>>     print(f"Effect size: {result['effect_size']:.2f}")
    """
    # Extract values at specified generation
    values1 = []
    for run in algo1_data:
        if len(run) > 0:
            idx = generation if generation >= 0 else len(run) - 1
            if idx < len(run):
                values1.append(run[idx])

    values2 = []
    for run in algo2_data:
        if len(run) > 0:
            idx = generation if generation >= 0 else len(run) - 1
            if idx < len(run):
                values2.append(run[idx])

    if not values1 or not values2:
        return {}

    values1 = np.array(values1)  # type: ignore[assignment]
    values2 = np.array(values2)  # type: ignore[assignment]

    # Calculate means
    mean1 = float(np.mean(values1))
    mean2 = float(np.mean(values2))
    difference = mean1 - mean2

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(values1, values2)

    # Calculate Cohen's d (effect size)
    pooled_std = np.sqrt(
        (
            (len(values1) - 1) * np.var(values1, ddof=1)
            + (len(values2) - 1) * np.var(values2, ddof=1)
        )
        / (len(values1) + len(values2) - 2)
    )
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    # Determine winner (for minimization)
    winner = ("algo1" if mean1 < mean2 else "algo2") if p_value < 0.05 else "tie"

    return {
        "algo1_mean": float(mean1),
        "algo2_mean": float(mean2),
        "difference": float(difference),
        "p_value": float(p_value),
        "t_statistic": float(t_stat),
        "significant": bool(p_value < 0.05),
        "effect_size": float(cohens_d),
        "winner": winner,
        "interpretation": _interpret_effect_size(abs(cohens_d)),
    }


def _interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size."""
    if cohens_d < 0.2:
        return "negligible"
    if cohens_d < 0.5:
        return "small"
    if cohens_d < 0.8:
        return "medium"
    return "large"


def calculate_generation_to_target(
    metric_history: list[float], target_value: float
) -> int:
    """
    Calculate how many generations needed to reach target metric value.

    Useful for comparing algorithm convergence speed.

    Args:
        metric_history: Metric values per generation
        target_value: Target to reach (e.g., 0 for feasibility)

    Returns:
        int: Generation number where target was reached, or -1 if never reached

    Example:
        >>> gen = calculate_generation_to_target(metrics.hard_violations, 0)
        >>> if gen >= 0:
        >>>     print(f"Found feasible solution at generation {gen}")
    """
    for gen, value in enumerate(metric_history):
        if value <= target_value:
            return gen
    return -1


def calculate_area_under_curve(metric_history: list[float]) -> float:
    """
    Calculate area under convergence curve.

    Lower area = faster convergence. Useful for comparing optimization speed.

    Args:
        metric_history: Metric values per generation

    Returns:
        float: Area under curve (trapezoidal integration)

    Example:
        >>> auc = calculate_area_under_curve(metrics.hard_violations)
        >>> print(f"Convergence AUC: {auc:.2f} (lower = faster)")
    """
    if len(metric_history) < 2:
        return 0.0

    # Trapezoidal integration
    auc = np.trapezoid(metric_history)
    return float(auc)
