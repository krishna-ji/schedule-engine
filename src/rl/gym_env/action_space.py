"""
Action space mapper for RL environment.

Maps discrete action indices to heuristic function calls.
Action space: 20 discrete actions (19 heuristics + 1 no-op)
"""

from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass

from src.heuristics import get_enabled_heuristics
from src.core.types import Individual, SchedulingContext


@dataclass
class ActionInfo:
    """Information about an action/heuristic."""

    action_id: int
    name: str
    category: str
    function: Optional[Callable] = None
    enabled: bool = True
    modifies_individual: bool = False


class ActionMapper:
    """
    Maps discrete action indices to heuristic function calls.

    Action Space (Discrete):
    - 0: No-op (do nothing)
    - 1-19: Heuristic operators from registry

    Handles:
    - Dynamic action masking (disabled heuristics)
    - Action validation
    - Heuristic execution with context
    """

    def __init__(self, use_config: bool = True):
        """
        Initialize action mapper.

        Args:
            use_config: Whether to respect config killswitches
        """
        self.use_config = use_config
        self.actions: List[ActionInfo] = []
        self._build_action_space()

    def _build_action_space(self) -> None:
        """Build action space from heuristic registry."""
        # Action 0: No-op
        self.actions.append(
            ActionInfo(
                action_id=0, name="no-op", category="meta", function=None, enabled=True
            )
        )

        # Actions 1-19: Heuristics
        if self.use_config:
            heuristics = get_enabled_heuristics().values()
        else:
            from src.heuristics import get_all_heuristics

            heuristics = get_all_heuristics().values()

        # Sort by category then name for consistent ordering
        heuristics_sorted = sorted(heuristics, key=lambda h: (h.category.value, h.name))

        for idx, h in enumerate(heuristics_sorted, start=1):
            self.actions.append(
                ActionInfo(
                    action_id=idx,
                    name=h.name,
                    category=h.category,
                    function=h.function,
                    enabled=getattr(h, "enabled", True),
                    modifies_individual=getattr(h, "modifies_individual", False),
                )
            )

    def apply_action(
        self,
        action: int,
        individual: Individual,
        context: SchedulingContext,
        population: Optional[List[Individual]] = None,
        generation: Optional[int] = None,
    ) -> Tuple[Individual, bool]:
        """
        Apply selected action to individual.

        Args:
            action: Discrete action index [0-19]
            individual: Individual to modify
            context: Scheduling context
            population: Full population (needed for diversity heuristics)
            generation: Current generation (needed for adaptive heuristics)

        Returns:
            (modified_individual, success)
        """
        import logging
        import copy

        logger = logging.getLogger(__name__)

        if not self.is_valid_action(action):
            # Invalid action - return unchanged
            return individual, False

        action_info = self.actions[action]

        # No-op: return unchanged
        if action_info.function is None:
            return individual, True

        # Apply heuristic with appropriate parameters
        try:
            # Clone individual to avoid mutation issues
            individual_copy = copy.deepcopy(individual)

            import inspect

            sig = inspect.signature(action_info.function)
            params = list(sig.parameters.keys())

            # Handle crossover operators that need two parents (parent1, parent2, context)
            if len(params) >= 2 and "parent2" in params:
                # Crossover needs 2 parents - select random second parent from population
                if population is None:
                    logger.warning(
                        f"Crossover operator {action_info.name} requires population parameter"
                    )
                    return individual, False
                import random

                valid_parents = [ind for ind in population if ind is not individual]
                if not valid_parents:
                    # Only one individual - can't crossover
                    logger.debug(
                        f"Cannot apply crossover {action_info.name}: insufficient population"
                    )
                    return individual, False
                parent2 = random.choice(valid_parents)
                result = action_info.function(individual_copy, parent2, context)
                # Crossover returns tuple of offspring - take first
                if isinstance(result, tuple):
                    modified = result[0]
                else:
                    modified = result

            # Handle construction heuristics that build from scratch (context only)
            elif action_info.category == "construction" or len(params) == 1:
                # Construction heuristics take only context and return new individual
                modified = action_info.function(context)

            # Handle diversity heuristics with population parameter (individual, population, context, ...)
            elif "population" in params:
                if population is None:
                    logger.warning(
                        f"Heuristic {action_info.name} requires population parameter"
                    )
                    return individual, False

                # Check if generation parameter also needed
                if "generation" in params:
                    if generation is None:
                        logger.warning(
                            f"Heuristic {action_info.name} requires generation parameter"
                        )
                        return individual, False
                    result = action_info.function(
                        individual_copy, population, context, generation
                    )
                else:
                    result = action_info.function(individual_copy, population, context)

                # Handle in-place modification heuristics
                if action_info.modifies_individual and isinstance(result, int):
                    modified = individual_copy
                else:
                    modified = result

            # Standard single-individual heuristics (individual, context)
            else:
                result = action_info.function(individual_copy, context)
                # Handle in-place modification heuristics (return int improvement count)
                if action_info.modifies_individual and isinstance(result, int):
                    modified = individual_copy
                else:
                    modified = result

            # Validate result
            if not self._validate_result(modified):
                logger.warning(f"Heuristic {action_info.name} returned invalid result")
                return individual, False

            return modified, True

        except TimeoutError:
            logger.error(f"Heuristic {action_info.name} timed out")
            return individual, False

        except Exception as e:
            # Heuristic failed - return original
            logger.error(f"Action {action_info.name} failed: {e}", exc_info=True)
            return individual, False

    def _validate_result(self, result) -> bool:
        """
        Validate heuristic result.

        Args:
            result: Result from heuristic function

        Returns:
            True if result is valid, False otherwise
        """
        if result is None:
            return False
        if not isinstance(result, list):
            return False
        if len(result) == 0:
            return False
        # Check if all elements have course_id attribute (basic gene validation)
        if not all(hasattr(gene, "course_id") for gene in result):
            return False
        return True

    def is_valid_action(self, action: int) -> bool:
        """Check if action is valid and enabled."""
        if action < 0 or action >= len(self.actions):
            return False
        return self.actions[action].enabled

    def get_action_mask(self) -> List[bool]:
        """
        Get action mask for masking invalid actions.

        Returns:
            Boolean mask where True = valid action
        """
        return [action.enabled for action in self.actions]

    def get_action_info(self, action: int) -> Optional[ActionInfo]:
        """Get information about an action."""
        if 0 <= action < len(self.actions):
            return self.actions[action]
        return None

    def get_action_by_name(self, name: str) -> Optional[int]:
        """Get action ID by heuristic name."""
        for action in self.actions:
            if action.name == name:
                return action.action_id
        return None

    @property
    def n_actions(self) -> int:
        """Number of actions in action space."""
        return len(self.actions)

    @property
    def enabled_actions(self) -> List[int]:
        """List of enabled action IDs."""
        return [a.action_id for a in self.actions if a.enabled]

    def describe_actions(self) -> str:
        """Get human-readable description of action space."""
        lines = [f"Action Space: {self.n_actions} actions\n"]
        for action in self.actions:
            status = "[ON]" if action.enabled else "[OFF]"
            lines.append(
                f"  [{action.action_id:2d}] {status} {action.name:30s} ({action.category})"
            )
        return "\n".join(lines)


def create_action_mapper(use_config: bool = True) -> ActionMapper:
    """
    Factory function to create action mapper.

    Args:
        use_config: Whether to respect configuration killswitches

    Returns:
        Configured ActionMapper instance
    """
    return ActionMapper(use_config=use_config)
