"""
Operator portfolio for local search.

ENHANCEMENT #6: Selects which local search operator to apply.
"""

from typing import List, Dict
import numpy as np


class OperatorPortfolio:
    """
    Manages portfolio of local search operators with adaptive selection.

    Selection methods:
    - Thompson Sampling: Bayesian multi-armed bandit
    - UCB: Upper Confidence Bound
    - Random: Uniform random selection
    - Fixed: Round-robin or priority-based

    Operators:
    - Kempe chain moves
    - Ejection chains
    - Variable depth search
    - Iterated local search
    - Guided local search (with penalties)
    """

    def __init__(
        self, operators: List[str], selection_method: str = "thompson_sampling"
    ):
        """
        Initialize operator portfolio.

        Args:
            operators: List of operator names
            selection_method: 'thompson_sampling', 'ucb', 'random', 'fixed'
        """
        self.operators = operators
        self.selection_method = selection_method

        # Thompson Sampling statistics (Beta distribution parameters)
        self.alpha = {op: 1.0 for op in operators}  # Successes + 1
        self.beta = {op: 1.0 for op in operators}  # Failures + 1

        # UCB statistics
        self.ucb_attempts = {op: 0 for op in operators}
        self.ucb_rewards = {op: 0.0 for op in operators}
        self.total_attempts = 0

    def select_operator(self) -> str:
        """
        Select operator to apply.

        Returns:
            Operator name
        """
        if self.selection_method == "thompson_sampling":
            return self._select_thompson_sampling()
        elif self.selection_method == "ucb":
            return self._select_ucb()
        elif self.selection_method == "random":
            return np.random.choice(self.operators)
        else:
            # Fixed: round-robin
            idx = self.total_attempts % len(self.operators)
            self.total_attempts += 1
            return self.operators[idx]

    def _select_thompson_sampling(self) -> str:
        """
        Thompson Sampling: Sample from Beta distributions.

        For each operator, sample theta ~ Beta(alpha, beta).
        Select operator with highest sampled theta.
        """
        samples = {
            op: np.random.beta(self.alpha[op], self.beta[op]) for op in self.operators
        }
        return max(samples.items(), key=lambda x: x[1])[0]

    def _select_ucb(self, exploration_param: float = 2.0) -> str:
        """
        UCB: Select operator with highest upper confidence bound.

        UCB(op) = avg_reward(op) + c * sqrt(ln(N) / n(op))
        """
        self.total_attempts += 1

        ucb_scores = {}
        for op in self.operators:
            if self.ucb_attempts[op] == 0:
                # Unvisited: infinite UCB
                ucb_scores[op] = float("inf")
            else:
                avg_reward = self.ucb_rewards[op] / self.ucb_attempts[op]
                exploration = exploration_param * np.sqrt(
                    np.log(self.total_attempts) / self.ucb_attempts[op]
                )
                ucb_scores[op] = avg_reward + exploration

        return max(ucb_scores.items(), key=lambda x: x[1])[0]

    def update(self, operator: str, improvement: float) -> None:
        """
        Update operator statistics after execution.

        Args:
            operator: Operator that was applied
            improvement: Fitness improvement (positive = success)
        """
        if self.selection_method == "thompson_sampling":
            # Update Beta parameters
            if improvement > 0:
                self.alpha[operator] += 1  # Success
            else:
                self.beta[operator] += 1  # Failure

        elif self.selection_method == "ucb":
            # Update UCB statistics
            self.ucb_attempts[operator] += 1
            self.ucb_rewards[operator] += improvement

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all operators.

        Returns:
            Dictionary mapping operator to statistics
        """
        stats = {}
        for op in self.operators:
            if self.selection_method == "thompson_sampling":
                # Thompson Sampling: expected success rate
                expected_rate = self.alpha[op] / (self.alpha[op] + self.beta[op])
                stats[op] = {
                    "alpha": self.alpha[op],
                    "beta": self.beta[op],
                    "expected_success_rate": expected_rate,
                }
            else:
                # UCB: average reward
                attempts = self.ucb_attempts[op]
                avg_reward = self.ucb_rewards[op] / attempts if attempts > 0 else 0.0
                stats[op] = {
                    "attempts": attempts,
                    "total_reward": self.ucb_rewards[op],
                    "avg_reward": avg_reward,
                }

        return stats

    def reset(self) -> None:
        """Reset all statistics."""
        self.alpha = {op: 1.0 for op in self.operators}
        self.beta = {op: 1.0 for op in self.operators}
        self.ucb_attempts = {op: 0 for op in self.operators}
        self.ucb_rewards = {op: 0.0 for op in self.operators}
        self.total_attempts = 0
