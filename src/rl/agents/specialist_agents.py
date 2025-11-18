"""
Specialist RL agents for task-specific optimization.

ENHANCEMENT #4: Train separate agents for repair vs optimization phases.

The key insight: Different phases of optimization require different strategies:

1. **Repair Agent**: Focuses on feasibility (reducing hard constraint violations)
   - Trained only on infeasible solutions (hard_violations > 0)
   - Reward: Hard constraint reduction
   - Strategy: Aggressive repair heuristics

2. **Optimizer Agent**: Focuses on quality (reducing soft constraint penalties)
   - Trained only on feasible solutions (hard_violations == 0)
   - Reward: Soft constraint reduction
   - Strategy: Fine-tuning heuristics

Mathematical Formulation:
    π_repair: S_infeasible → A    (repair policy)
    π_optimize: S_feasible → A    (optimization policy)

    Policy selection:
    π(s) = {
        π_repair(s)     if hard_violations(s) > 0
        π_optimize(s)   if hard_violations(s) == 0
    }

Training Protocol:
- **Phase 1**: Train repair agent on infeasible problems (50K steps)
- **Phase 2**: Train optimizer agent on feasible problems (50K steps)
- **Phase 3**: Fine-tune both agents jointly (20K steps)

References:
- Burke et al. (2013): Hyper-Heuristics with specialist selection
- Teng et al. (2015): Multi-agent reinforcement learning for optimization
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.base_class import BaseAlgorithm
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SpecialistAgents:
    """
    Manager for specialist RL agents (repair + optimization).

    Coordinates two specialist agents:
    - Repair agent: Focuses on hard constraint reduction
    - Optimizer agent: Focuses on soft constraint reduction

    Usage:
        >>> agents = SpecialistAgents()
        >>> agents.load_repair_agent("models/repair_agent.zip")
        >>> agents.load_optimizer_agent("models/optimizer_agent.zip")
        >>> action = agents.select_action(state, hard_violations=5)
    """

    def __init__(
        self,
        repair_agent: Optional[BaseAlgorithm] = None,
        optimizer_agent: Optional[BaseAlgorithm] = None,
        switching_threshold: float = 0.5,
        use_soft_switching: bool = True,
    ):
        """
        Initialize specialist agents.

        Args:
            repair_agent: Pre-trained repair agent (optional)
            optimizer_agent: Pre-trained optimizer agent (optional)
            switching_threshold: Hard violation threshold for agent switching
            use_soft_switching: Use probabilistic blending near threshold
        """
        self.repair_agent = repair_agent
        self.optimizer_agent = optimizer_agent
        self.switching_threshold = switching_threshold
        self.use_soft_switching = use_soft_switching

        # Statistics
        self.repair_actions = 0
        self.optimizer_actions = 0
        self.blended_actions = 0

    def load_repair_agent(self, model_path: str) -> None:
        """
        Load pre-trained repair agent.

        Args:
            model_path: Path to saved model (.zip file)
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Repair agent not found: {model_path}")

        # Auto-detect model type from path or load as PPO by default
        if "dqn" in model_path.lower():
            self.repair_agent = DQN.load(model_path)
        else:
            self.repair_agent = PPO.load(model_path)

        logger.info(f"Loaded repair agent from {model_path}")

    def load_optimizer_agent(self, model_path: str) -> None:
        """
        Load pre-trained optimizer agent.

        Args:
            model_path: Path to saved model (.zip file)
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Optimizer agent not found: {model_path}")

        # Auto-detect model type
        if "dqn" in model_path.lower():
            self.optimizer_agent = DQN.load(model_path)
        else:
            self.optimizer_agent = PPO.load(model_path)

        logger.info(f"Loaded optimizer agent from {model_path}")

    def select_action(
        self,
        state: np.ndarray,
        hard_violations: float,
        deterministic: bool = True,
    ) -> Tuple[int, str]:
        """
        Select action using appropriate specialist agent.

        Agent selection logic:
        - If hard_violations > threshold: Use repair agent
        - If hard_violations == 0: Use optimizer agent
        - If near threshold: Blend agents (if use_soft_switching=True)

        Args:
            state: Current observation state
            hard_violations: Number of hard constraint violations
            deterministic: Use deterministic policy (True for inference)

        Returns:
            (action, agent_name) where agent_name ∈ {"repair", "optimizer", "blended"}
        """
        # Determine which agent to use
        if hard_violations > self.switching_threshold:
            # Use repair agent
            if self.repair_agent is None:
                raise RuntimeError("Repair agent not loaded")

            action, _ = self.repair_agent.predict(state, deterministic=deterministic)
            self.repair_actions += 1
            return int(action), "repair"

        elif hard_violations == 0:
            # Use optimizer agent
            if self.optimizer_agent is None:
                raise RuntimeError("Optimizer agent not loaded")

            action, _ = self.optimizer_agent.predict(state, deterministic=deterministic)
            self.optimizer_actions += 1
            return int(action), "optimizer"

        else:
            # Near threshold: blend or use soft switching
            if (
                self.use_soft_switching
                and self.repair_agent is not None
                and self.optimizer_agent is not None
            ):
                # Compute blending weight
                alpha = hard_violations / self.switching_threshold  # ∈ [0, 1]

                # Get actions from both agents
                repair_action, _ = self.repair_agent.predict(state, deterministic=False)
                optimizer_action, _ = self.optimizer_agent.predict(
                    state, deterministic=False
                )

                # Probabilistic selection
                if np.random.rand() < alpha:
                    action = repair_action
                else:
                    action = optimizer_action

                self.blended_actions += 1
                return int(action), "blended"
            else:
                # Fallback to repair agent if in transition zone
                if self.repair_agent is None:
                    raise RuntimeError("Repair agent not loaded")

                action, _ = self.repair_agent.predict(
                    state, deterministic=deterministic
                )
                self.repair_actions += 1
                return int(action), "repair"

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get usage statistics for specialist agents.

        Returns:
            Dictionary with action counts and selection ratios
        """
        total_actions = (
            self.repair_actions + self.optimizer_actions + self.blended_actions
        )

        return {
            "repair_actions": self.repair_actions,
            "optimizer_actions": self.optimizer_actions,
            "blended_actions": self.blended_actions,
            "total_actions": total_actions,
            "repair_ratio": (
                self.repair_actions / total_actions if total_actions > 0 else 0.0
            ),
            "optimizer_ratio": (
                self.optimizer_actions / total_actions if total_actions > 0 else 0.0
            ),
            "blended_ratio": (
                self.blended_actions / total_actions if total_actions > 0 else 0.0
            ),
        }

    def reset_statistics(self) -> None:
        """Reset action counters."""
        self.repair_actions = 0
        self.optimizer_actions = 0
        self.blended_actions = 0


class AgentCoordinator:
    """
    Coordinator for managing multiple specialist agents.

    Extends SpecialistAgents with advanced coordination strategies:
    - Portfolio-based selection (choose best agent based on historical performance)
    - Meta-learning (learn which agent to use when)
    - Dynamic agent weight adaptation

    Usage:
        >>> coordinator = AgentCoordinator()
        >>> coordinator.add_agent("repair", repair_agent, "hard_violations")
        >>> coordinator.add_agent("optimizer", optimizer_agent, "soft_violations")
        >>> action = coordinator.select_action(state, context)
    """

    def __init__(self):
        """Initialize agent coordinator."""
        self.agents: Dict[str, BaseAlgorithm] = {}
        self.agent_specializations: Dict[str, str] = (
            {}
        )  # Agent name -> specialization type
        self.agent_performance: Dict[str, float] = {}  # Agent name -> success rate
        self.agent_usage_count: Dict[str, int] = {}  # Agent name -> usage count

    def add_agent(
        self,
        name: str,
        agent: BaseAlgorithm,
        specialization: str,
    ) -> None:
        """
        Add a specialist agent to the coordinator.

        Args:
            name: Unique agent identifier
            agent: Trained RL agent
            specialization: Type of task ("hard_violations", "soft_violations", "diversity")
        """
        self.agents[name] = agent
        self.agent_specializations[name] = specialization
        self.agent_performance[name] = 0.0
        self.agent_usage_count[name] = 0

        logger.info(f"Added specialist agent '{name}' for {specialization}")

    def select_agent(
        self,
        context: Dict[str, float],
    ) -> str:
        """
        Select best agent based on current problem context.

        Uses performance history and specialization matching.

        Args:
            context: Problem context with metrics (hard_violations, soft_violations, etc.)

        Returns:
            Name of selected agent
        """
        if not self.agents:
            raise RuntimeError("No agents available")

        # Simple heuristic: select based on primary need
        hard_violations = context.get("hard_violations", 0)
        soft_violations = context.get("soft_violations", 0)

        if hard_violations > 0:
            # Need repair agent
            for name, spec in self.agent_specializations.items():
                if spec == "hard_violations":
                    return name
        else:
            # Need optimizer agent
            for name, spec in self.agent_specializations.items():
                if spec == "soft_violations":
                    return name

        # Fallback: select most successful agent
        if self.agent_performance:
            return max(self.agent_performance, key=self.agent_performance.get)

        # Final fallback: first agent
        return list(self.agents.keys())[0]

    def predict(
        self,
        state: np.ndarray,
        context: Dict[str, float],
        deterministic: bool = True,
    ) -> Tuple[int, str]:
        """
        Predict action using best specialist agent.

        Args:
            state: Current observation
            context: Problem context
            deterministic: Use deterministic policy

        Returns:
            (action, agent_name)
        """
        agent_name = self.select_agent(context)
        agent = self.agents[agent_name]

        action, _ = agent.predict(state, deterministic=deterministic)
        self.agent_usage_count[agent_name] += 1

        return int(action), agent_name

    def update_performance(
        self,
        agent_name: str,
        success: bool,
    ) -> None:
        """
        Update agent performance statistics.

        Args:
            agent_name: Name of agent that was used
            success: Whether the action improved fitness
        """
        if agent_name not in self.agent_performance:
            return

        # Exponential moving average
        alpha = 0.1
        current_perf = self.agent_performance[agent_name]
        new_perf = alpha * (1.0 if success else 0.0) + (1 - alpha) * current_perf
        self.agent_performance[agent_name] = new_perf
