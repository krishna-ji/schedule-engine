"""
Heuristic Selection Strategies for Notebooks.

Provides different heuristic selection strategies for experiments:
- Local search (Mode B: Memetic)
- Round-robin selection (Mode C)
- Adaptive selection (Mode D)
- RL-guided selection (Mode E)

Each strategy wraps repair heuristics with different selection logic.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.domain.gene import SessionGene
from src.notebooks.core import NotebookData

__all__ = [
    "local_search_individual",
    "RoundRobinSelector",
    "AdaptiveSelector",
    "SimpleRLSelector",
    "load_trained_agent",
]


# =============================================================================
# MODE B: LOCAL SEARCH (Memetic)
# =============================================================================


def local_search_individual(
    individual: list[SessionGene],
    data: NotebookData,
    evaluate_fn: Callable[[list[SessionGene]], tuple[float, float]],
    max_iterations: int = 10,
) -> tuple[list[SessionGene], float]:
    """
    Apply local search to improve an individual.

    Simple hill-climbing: randomly mutate and keep if fitness improves.

    Args:
        individual: Individual to improve (modified in-place)
        data: NotebookData for context
        evaluate_fn: Fitness evaluation function
        max_iterations: Maximum improvement attempts

    Returns:
        Tuple of (improved individual, total improvement)
    """
    current_fitness = evaluate_fn(individual)
    total_improvement = 0.0

    for _ in range(max_iterations):
        # Create neighbor by mutating a random gene
        gene_idx = random.randint(0, len(individual) - 1)
        gene = individual[gene_idx]

        # Store original values
        orig_start = gene.start_quanta
        orig_room = gene.room_id
        orig_instructor = gene.instructor_id

        # Random mutation
        mutation_type = random.choice(["time", "room", "instructor"])

        if mutation_type == "time":
            max_start = data.qts.total_quanta - gene.num_quanta
            if max_start > 0:
                gene.start_quanta = random.randint(0, max_start)

        elif mutation_type == "room":
            gene.room_id = random.choice(list(data.rooms.keys()))

        elif mutation_type == "instructor":
            course_key = (gene.course_id, gene.course_type)
            qualified = [
                iid
                for iid, instr in data.instructors.items()
                if course_key in instr.qualified_courses
            ]
            if qualified:
                gene.instructor_id = random.choice(qualified)

        # Evaluate new fitness
        new_fitness = evaluate_fn(individual)

        # Accept if better (lexicographic: hard first, then soft)
        if (new_fitness[0], new_fitness[1]) < (current_fitness[0], current_fitness[1]):
            improvement = current_fitness[0] - new_fitness[0]  # Hard improvement
            total_improvement += improvement
            current_fitness = new_fitness
        else:
            # Revert
            gene.start_quanta = orig_start
            gene.room_id = orig_room
            gene.instructor_id = orig_instructor

    return individual, total_improvement


# =============================================================================
# MODE C: ROUND-ROBIN HEURISTIC SELECTION
# =============================================================================


class RoundRobinSelector:
    """
    Round-robin heuristic selector.

    Cycles through available repair heuristics in sequence.
    Simple but ensures fair exploration of all strategies.
    """

    def __init__(self) -> None:
        """Initialize with available heuristics."""
        self.heuristics = [
            ("time_shift", self._time_shift),
            ("room_swap", self._room_swap),
            ("instructor_swap", self._instructor_swap),
        ]
        self.current_idx = 0

    def apply(
        self,
        individual: list[SessionGene],
        data: NotebookData,
    ) -> tuple[str, int]:
        """
        Apply next heuristic in round-robin sequence.

        Args:
            individual: Individual to repair (modified in-place)
            data: NotebookData for context

        Returns:
            Tuple of (heuristic name, number of fixes applied)
        """
        name, heuristic_fn = self.heuristics[self.current_idx]
        fixes = heuristic_fn(individual, data)

        # Advance to next heuristic
        self.current_idx = (self.current_idx + 1) % len(self.heuristics)

        return name, fixes

    def _time_shift(self, individual: list[SessionGene], data: NotebookData) -> int:
        """Shift random session to new time slot."""
        if not individual:
            return 0

        gene = random.choice(individual)
        max_start = data.qts.total_quanta - gene.num_quanta
        if max_start > 0:
            gene.start_quanta = random.randint(0, max_start)
            return 1
        return 0

    def _room_swap(self, individual: list[SessionGene], data: NotebookData) -> int:
        """Swap room for random session."""
        if not individual:
            return 0

        gene = random.choice(individual)
        gene.room_id = random.choice(list(data.rooms.keys()))
        return 1

    def _instructor_swap(
        self, individual: list[SessionGene], data: NotebookData
    ) -> int:
        """Swap instructor for random session."""
        if not individual:
            return 0

        gene = random.choice(individual)
        course_key = (gene.course_id, gene.course_type)
        qualified = [
            iid
            for iid, instr in data.instructors.items()
            if course_key in instr.qualified_courses
        ]
        if qualified:
            gene.instructor_id = random.choice(qualified)
            return 1
        return 0


# =============================================================================
# MODE D: ADAPTIVE HEURISTIC SELECTION
# =============================================================================


class AdaptiveSelector:
    """
    Adaptive heuristic selector using roulette wheel selection.

    Adjusts selection probabilities based on past performance.
    Heuristics that produce more improvements get higher probability.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        min_prob: float = 0.05,
    ) -> None:
        """
        Initialize adaptive selector.

        Args:
            learning_rate: How quickly to adjust probabilities
            min_prob: Minimum probability for any heuristic
        """
        self.learning_rate = learning_rate
        self.min_prob = min_prob

        # Heuristics with initial uniform probabilities
        self.heuristics = ["time_shift", "room_swap", "instructor_swap"]
        self.probs = {h: 1.0 / len(self.heuristics) for h in self.heuristics}

        # Performance tracking
        self.successes: dict[str, int] = defaultdict(int)
        self.attempts: dict[str, int] = defaultdict(int)

    def apply(
        self,
        individual: list[SessionGene],
        data: NotebookData,
    ) -> tuple[str, int]:
        """
        Apply heuristic selected by adaptive roulette wheel.

        Args:
            individual: Individual to repair (modified in-place)
            data: NotebookData for context

        Returns:
            Tuple of (heuristic name, number of fixes applied)
        """
        # Select heuristic using roulette wheel
        selected = self._roulette_select()
        self.attempts[selected] += 1

        # Apply selected heuristic
        if selected == "time_shift":
            fixes = self._time_shift(individual, data)
        elif selected == "room_swap":
            fixes = self._room_swap(individual, data)
        else:  # instructor_swap
            fixes = self._instructor_swap(individual, data)

        # Update probabilities based on success
        if fixes > 0:
            self.successes[selected] += 1
            self._update_probabilities(selected, success=True)
        else:
            self._update_probabilities(selected, success=False)

        return selected, fixes

    def _roulette_select(self) -> str:
        """Select heuristic using roulette wheel."""
        r = random.random()
        cumulative = 0.0
        for h in self.heuristics:
            cumulative += self.probs[h]
            if r <= cumulative:
                return h
        return self.heuristics[-1]

    def _update_probabilities(self, selected: str, success: bool) -> None:
        """Update probabilities based on outcome."""
        if success:
            # Increase probability for successful heuristic
            reward = self.learning_rate
            self.probs[selected] = min(
                1.0 - self.min_prob * (len(self.heuristics) - 1),
                self.probs[selected] + reward,
            )

            # Decrease others proportionally
            decrease_total = reward
            for h in self.heuristics:
                if h != selected:
                    decrease = decrease_total / (len(self.heuristics) - 1)
                    self.probs[h] = max(self.min_prob, self.probs[h] - decrease)
        else:
            # Small decrease for unsuccessful heuristic
            penalty = self.learning_rate * 0.5
            self.probs[selected] = max(self.min_prob, self.probs[selected] - penalty)

            # Redistribute to others
            increase_total = penalty
            for h in self.heuristics:
                if h != selected:
                    self.probs[h] += increase_total / (len(self.heuristics) - 1)

        # Normalize
        total = sum(self.probs.values())
        self.probs = {h: p / total for h, p in self.probs.items()}

    def _time_shift(self, individual: list[SessionGene], data: NotebookData) -> int:
        """Shift random session to new time slot."""
        if not individual:
            return 0
        gene = random.choice(individual)
        max_start = data.qts.total_quanta - gene.num_quanta
        if max_start > 0:
            gene.start_quanta = random.randint(0, max_start)
            return 1
        return 0

    def _room_swap(self, individual: list[SessionGene], data: NotebookData) -> int:
        """Swap room for random session."""
        if not individual:
            return 0
        gene = random.choice(individual)
        gene.room_id = random.choice(list(data.rooms.keys()))
        return 1

    def _instructor_swap(
        self, individual: list[SessionGene], data: NotebookData
    ) -> int:
        """Swap instructor for random session."""
        if not individual:
            return 0
        gene = random.choice(individual)
        course_key = (gene.course_id, gene.course_type)
        qualified = [
            iid
            for iid, instr in data.instructors.items()
            if course_key in instr.qualified_courses
        ]
        if qualified:
            gene.instructor_id = random.choice(qualified)
            return 1
        return 0

    def get_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        return {
            "probs": dict(self.probs),
            "successes": dict(self.successes),
            "attempts": dict(self.attempts),
            "success_rates": {
                h: self.successes[h] / max(1, self.attempts[h]) for h in self.heuristics
            },
        }


# =============================================================================
# MODE E: RL-GUIDED HEURISTIC SELECTION
# =============================================================================


class SimpleRLSelector:
    """
    Simple Q-learning based heuristic selector.

    Uses tabular Q-learning to learn which heuristic to apply
    based on the current state (discretized constraint violations).

    State: (hard_bucket, soft_bucket) where buckets discretize violations
    Actions: Available repair heuristics
    Reward: Improvement in fitness after applying heuristic
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon: float = 0.3,
        epsilon_decay: float = 0.99,
        min_epsilon: float = 0.05,
    ) -> None:
        """
        Initialize Q-learning selector.

        Args:
            learning_rate: Q-value learning rate (alpha)
            discount: Future reward discount (gamma)
            epsilon: Initial exploration probability
            epsilon_decay: Epsilon decay per episode
            min_epsilon: Minimum epsilon value
        """
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Actions (heuristics)
        self.actions = ["time_shift", "room_swap", "instructor_swap"]

        # Q-table: state -> action -> Q-value
        self.q_table: dict[tuple[int, int], dict[str, float]] = {}

        # Stats
        self.total_rewards = 0.0
        self.episodes = 0

    def apply(
        self,
        individual: list[SessionGene],
        data: NotebookData,
        evaluate_fn: Callable[[list[SessionGene]], tuple[float, float]],
    ) -> tuple[str, int, float]:
        """
        Apply RL-selected heuristic.

        Args:
            individual: Individual to repair (modified in-place)
            data: NotebookData for context
            evaluate_fn: Fitness evaluation function

        Returns:
            Tuple of (action name, fixes, reward)
        """
        # Get current state and fitness
        old_fitness = evaluate_fn(individual)
        state = self._get_state(old_fitness)

        # Select action (epsilon-greedy)
        action = self._select_action(state)

        # Apply action
        if action == "time_shift":
            fixes = self._time_shift(individual, data)
        elif action == "room_swap":
            fixes = self._room_swap(individual, data)
        else:
            fixes = self._instructor_swap(individual, data)

        # Get new fitness and calculate reward
        new_fitness = evaluate_fn(individual)
        reward = self._calculate_reward(old_fitness, new_fitness)

        # Get new state
        new_state = self._get_state(new_fitness)

        # Update Q-table
        self._update_q(state, action, reward, new_state)

        self.total_rewards += reward
        self.episodes += 1

        return action, fixes, reward

    def _get_state(self, fitness: tuple[float, float]) -> tuple[int, int]:
        """Discretize fitness into state buckets."""
        hard, soft = fitness

        # Bucket hard violations: 0, 1-5, 6-10, 11-20, 20+
        if hard == 0:
            hard_bucket = 0
        elif hard <= 5:
            hard_bucket = 1
        elif hard <= 10:
            hard_bucket = 2
        elif hard <= 20:
            hard_bucket = 3
        else:
            hard_bucket = 4

        # Bucket soft penalties: 0-50, 51-100, 101-200, 200+
        if soft <= 50:
            soft_bucket = 0
        elif soft <= 100:
            soft_bucket = 1
        elif soft <= 200:
            soft_bucket = 2
        else:
            soft_bucket = 3

        return (hard_bucket, soft_bucket)

    def _select_action(self, state: tuple[int, int]) -> str:
        """Epsilon-greedy action selection."""
        # Initialize Q-values for new state
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}

        # Epsilon-greedy
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # Greedy: choose best action
            q_values = self.q_table[state]
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)

    def _calculate_reward(
        self,
        old_fitness: tuple[float, float],
        new_fitness: tuple[float, float],
    ) -> float:
        """Calculate reward from fitness improvement."""
        old_hard, old_soft = old_fitness
        new_hard, new_soft = new_fitness

        # Primary reward: hard violation reduction
        hard_improvement = old_hard - new_hard

        # Secondary reward: soft penalty reduction (scaled down)
        soft_improvement = (old_soft - new_soft) * 0.01

        return hard_improvement + soft_improvement

    def _update_q(
        self,
        state: tuple[int, int],
        action: str,
        reward: float,
        new_state: tuple[int, int],
    ) -> None:
        """Q-learning update."""
        # Initialize Q-values for new state if needed
        if new_state not in self.q_table:
            self.q_table[new_state] = {a: 0.0 for a in self.actions}

        # Q-learning update rule
        old_q = self.q_table[state][action]
        max_next_q = max(self.q_table[new_state].values())
        new_q = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self) -> None:
        """Decay epsilon after each generation."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def _time_shift(self, individual: list[SessionGene], data: NotebookData) -> int:
        """Shift random session to new time slot."""
        if not individual:
            return 0
        gene = random.choice(individual)
        max_start = data.qts.total_quanta - gene.num_quanta
        if max_start > 0:
            gene.start_quanta = random.randint(0, max_start)
            return 1
        return 0

    def _room_swap(self, individual: list[SessionGene], data: NotebookData) -> int:
        """Swap room for random session."""
        if not individual:
            return 0
        gene = random.choice(individual)
        gene.room_id = random.choice(list(data.rooms.keys()))
        return 1

    def _instructor_swap(
        self, individual: list[SessionGene], data: NotebookData
    ) -> int:
        """Swap instructor for random session."""
        if not individual:
            return 0
        gene = random.choice(individual)
        course_key = (gene.course_id, gene.course_type)
        qualified = [
            iid
            for iid, instr in data.instructors.items()
            if course_key in instr.qualified_courses
        ]
        if qualified:
            gene.instructor_id = random.choice(qualified)
            return 1
        return 0

    def get_stats(self) -> dict[str, Any]:
        """Get learning statistics."""
        return {
            "total_rewards": self.total_rewards,
            "episodes": self.episodes,
            "avg_reward": self.total_rewards / max(1, self.episodes),
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
        }


def load_trained_agent(model_dir: Path | str) -> Any | None:
    """
    Load a pre-trained RL agent from disk.

    Args:
        model_dir: Directory containing saved model

    Returns:
        Loaded agent or None if not found
    """
    model_dir = Path(model_dir)

    if not model_dir.exists():
        return None

    # Look for saved Q-table
    q_table_path = model_dir / "q_table.json"
    if q_table_path.exists():
        import json

        with open(q_table_path) as f:
            data = json.load(f)

        # Reconstruct SimpleRLSelector
        agent = SimpleRLSelector()
        # Convert string keys back to tuples
        agent.q_table = {
            tuple(map(int, k.strip("()").split(", "))): v
            for k, v in data.get("q_table", {}).items()
        }
        agent.epsilon = data.get("epsilon", 0.05)
        return agent

    return None
