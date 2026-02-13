"""
Action space mapper for RL environment.

Maps discrete action indices to heuristic function calls.
Action space: 20 discrete actions (19 heuristics + 1 no-op)
"""

import logging
import signal
from collections.abc import Callable
from dataclasses import dataclass
from types import FrameType
from typing import Any

from schedule_engine.domain.types import Individual, SchedulingContext
from schedule_engine.ga.heuristics import get_enabled_heuristics

logger = logging.getLogger(__name__)


@dataclass
class ActionInfo:
    """Information about an action/heuristic."""

    action_id: int
    name: str
    category: str
    function: Callable | None = None
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

    def __init__(
        self,
        use_config: bool = True,
        timeout_seconds: float = 30.0,
        action_id_map: dict[str, int] | None = None,
    ):
        """
        Initialize action mapper.

        Args:
            use_config: Whether to respect config killswitches
            timeout_seconds: Maximum time for any single heuristic execution (default: 30s)
        """
        self.use_config = use_config
        self.timeout_seconds = timeout_seconds
        self.actions: list[ActionInfo] = []
        if use_config and action_id_map is None:
            from schedule_engine.config import get_config

            action_id_map = get_config().rl.environment.action_id_map
        self.action_id_map = action_id_map or {}
        self._build_action_space()

    def _build_action_space(self) -> None:
        """Build action space from heuristic registry."""
        # Actions 1-19: Heuristics
        if self.use_config:
            heuristics = get_enabled_heuristics().values()
        else:
            from schedule_engine.ga.heuristics import get_all_heuristics

            heuristics = get_all_heuristics().values()

        # Sort by category then name for consistent ordering
        heuristics_sorted = sorted(heuristics, key=lambda h: (h.category, h.name))

        if self.use_config and self.action_id_map:
            self._build_action_space_with_mapping(heuristics_sorted)
            return

        # Default ordering: action 0 = no-op, then sorted heuristics
        self.actions.append(
            ActionInfo(
                action_id=0, name="no-op", category="meta", function=None, enabled=True
            )
        )

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

    def _build_action_space_with_mapping(self, heuristics: list[Any]) -> None:
        """Build action space using a stable action ID map."""
        actions_by_id: dict[int, ActionInfo] = {
            0: ActionInfo(
                action_id=0, name="no-op", category="meta", function=None, enabled=True
            )
        }

        used_ids = set()
        for name, action_id in self.action_id_map.items():
            if action_id <= 0:
                raise ValueError(
                    f"Invalid action_id {action_id} for heuristic '{name}'. "
                    "Action IDs must be >= 1."
                )
            if action_id in used_ids:
                raise ValueError(
                    f"Duplicate action_id {action_id} in action_id_map."
                )
            used_ids.add(action_id)

        heuristic_by_name = {h.name: h for h in heuristics}
        for name, action_id in sorted(self.action_id_map.items(), key=lambda item: item[1]):
            heuristic = heuristic_by_name.get(name)
            if heuristic is None:
                logger.warning(
                    "Action map entry for '%s' not found in enabled heuristics; skipping.",
                    name,
                )
                continue
            actions_by_id[action_id] = ActionInfo(
                action_id=action_id,
                name=heuristic.name,
                category=heuristic.category,
                function=heuristic.function,
                enabled=getattr(heuristic, "enabled", True),
                modifies_individual=getattr(heuristic, "modifies_individual", False),
            )

        next_id = max(actions_by_id.keys(), default=0) + 1
        unmapped = [h for h in heuristics if h.name not in self.action_id_map]
        for heuristic in unmapped:
            actions_by_id[next_id] = ActionInfo(
                action_id=next_id,
                name=heuristic.name,
                category=heuristic.category,
                function=heuristic.function,
                enabled=getattr(heuristic, "enabled", True),
                modifies_individual=getattr(heuristic, "modifies_individual", False),
            )
            next_id += 1

        max_id = max(actions_by_id.keys(), default=0)
        for action_id in range(max_id + 1):
            if action_id not in actions_by_id:
                actions_by_id[action_id] = ActionInfo(
                    action_id=action_id,
                    name=f"unused_{action_id}",
                    category="meta",
                    function=None,
                    enabled=False,
                )

        self.actions = [actions_by_id[action_id] for action_id in range(max_id + 1)]

    def apply_action(
        self,
        action: int,
        individual: Individual,
        context: SchedulingContext,
        population: list[Individual] | None = None,
        generation: int | None = None,
    ) -> tuple[Individual, bool]:
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
        import copy
        import logging

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
            # Clone individual to avoid mutation issues (optimized shallow copy)
            # Use shallow copy + list copy instead of deepcopy for 10-50x speedup
            individual_copy = copy.copy(individual)
            individual_copy[:] = individual[:]
            if hasattr(individual, "fitness") and hasattr(individual.fitness, "values"):  # type: ignore[attr-defined]
                individual_copy.fitness.values = individual.fitness.values  # type: ignore[attr-defined]

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

                # Get kwargs excluding already-provided positional params
                heuristic_kwargs = self._get_heuristic_kwargs(
                    action_info, provided_params=["parent1", "parent2", "context"]
                )
                result = action_info.function(
                    individual_copy, parent2, context, **heuristic_kwargs
                )
                # Crossover returns tuple of offspring - take first
                modified = result[0] if isinstance(result, tuple) else result

            # Handle construction heuristics that build from scratch (context only)
            elif action_info.category == "construction" or len(params) == 1:
                # Construction heuristics take only context and return new individual
                heuristic_kwargs = self._get_heuristic_kwargs(
                    action_info, provided_params=["context"]
                )
                modified = action_info.function(context, **heuristic_kwargs)

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
                    heuristic_kwargs = self._get_heuristic_kwargs(
                        action_info,
                        provided_params=[
                            "individual",
                            "population",
                            "context",
                            "generation",
                        ],
                    )
                    result = action_info.function(
                        individual_copy,
                        population,
                        context,
                        generation,
                        **heuristic_kwargs,
                    )
                else:
                    heuristic_kwargs = self._get_heuristic_kwargs(
                        action_info,
                        provided_params=["individual", "population", "context"],
                    )
                    result = action_info.function(
                        individual_copy, population, context, **heuristic_kwargs
                    )

                # Handle in-place modification heuristics
                if action_info.modifies_individual and isinstance(result, int):
                    self._invalidate_fitness(individual_copy)
                    modified = individual_copy
                else:
                    modified = result

            # Standard single-individual heuristics (individual, context)
            else:
                # Get kwargs excluding already-provided positional params
                heuristic_kwargs = self._get_heuristic_kwargs(
                    action_info, provided_params=["individual", "context"]
                )

                # Apply timeout protection with kwargs
                result = self._execute_with_timeout(
                    action_info.function,
                    individual_copy,
                    context,
                    action_name=action_info.name,
                    **heuristic_kwargs,
                )
                if result is None:  # Timeout occurred
                    return individual, False

                # Handle in-place modification heuristics (return int improvement count)
                if action_info.modifies_individual and isinstance(result, int):
                    self._invalidate_fitness(individual_copy)
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

    def _get_heuristic_kwargs(
        self,
        action_info: ActionInfo,
        provided_params: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Extract heuristic configuration parameters from config.

        Args:
            action_info: Action information with heuristic metadata
            provided_params: List of parameter names already provided positionally

        Returns:
            Dict of keyword arguments to pass to heuristic function
        """
        import inspect

        from schedule_engine.config import get_config

        config = get_config()
        heuristics_config = getattr(config, "heuristics", None)

        if not heuristics_config:
            return {}

        # Get category config (e.g., improvement, perturbation, etc.)
        category_config = getattr(heuristics_config, action_info.category, None)
        if not category_config:
            return {}

        # Get heuristic-specific config
        heuristic_config = category_config.get(action_info.name, {})
        if not heuristic_config:
            return {}

        # Get function signature to understand which params it accepts
        if provided_params is None:
            provided_params = []

        # Get all parameter names from function signature
        try:
            assert action_info.function is not None
            sig = inspect.signature(action_info.function)
            func_params = set(sig.parameters.keys())
        except Exception:  # Catch all inspection errors
            func_params = set()

        # Build exclusion set: metadata fields + already provided positional params
        excluded = set(provided_params + ["enabled", "priority"])

        # Extract kwargs: only params that are in function signature, not provided positionally, and not metadata
        kwargs = {}
        for key, value in heuristic_config.items():
            if key not in excluded and key in func_params:
                kwargs[key] = value

        return kwargs

    def _timeout_handler(self, signum: int, frame: FrameType | None) -> None:
        """Signal handler for timeout."""
        raise TimeoutError("Heuristic execution timed out")

    @staticmethod
    def _invalidate_fitness(individual: Individual) -> None:
        """Mark a DEAP individual fitness as invalid after in-place mutation."""
        if hasattr(individual, "fitness") and hasattr(individual.fitness, "values"):  # type: ignore[attr-defined]
            individual.fitness.values = ()  # type: ignore[attr-defined]

    def _execute_with_timeout(
        self,
        func: Callable[..., Any],
        *args: Any,
        action_name: str = "unknown",
        **kwargs: Any,
    ) -> Any | None:
        """
        Execute function with timeout protection.

        Args:
            func: Function to execute
            *args: Positional arguments to pass to function
            action_name: Name of action for logging
            **kwargs: Keyword arguments to pass to function

        Returns:
            Function result or None if timeout
        """
        # Note: signal.alarm only works in main thread on Unix
        # For Windows/threads, we rely on the TimeoutError catch in apply_action
        import platform

        if platform.system() != "Windows" and self.timeout_seconds > 0:
            try:
                # Set up timeout alarm (Unix only)
                old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)  # type: ignore[attr-defined]
                signal.alarm(int(self.timeout_seconds))  # type: ignore[attr-defined]

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Cancel alarm and restore handler
                    signal.alarm(0)  # type: ignore[attr-defined]
                    signal.signal(signal.SIGALRM, old_handler)  # type: ignore[attr-defined]

            except TimeoutError:
                logger.warning(
                    f"Action {action_name} timed out after {self.timeout_seconds}s"
                )
                return None
        else:
            # Windows or no timeout - execute directly
            # Long-running operations will still block but user can Ctrl+C
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Action {action_name} failed: {e}")
                return None

    def _validate_result(self, result: Any) -> bool:
        """
        Validate heuristic result.

        Args:
            result: Result from heuristic function

        Returns:
            True if result is valid, False otherwise
        """
        if result is None:
            return False
        if not isinstance(result, list | tuple):
            return False
        if len(result) == 0:
            return False
        # Check if all elements have course_id attribute (basic gene validation)
        return all(hasattr(gene, "course_id") for gene in result)

    def is_valid_action(self, action: int) -> bool:
        """Check if action is valid and enabled."""
        if action < 0 or action >= len(self.actions):
            return False
        return self.actions[action].enabled

    def get_action_mask(self) -> list[bool]:
        """
        Get action mask for masking invalid actions.

        Returns:
            Boolean mask where True = valid action
        """
        return [action.enabled for action in self.actions]

    def get_action_info(self, action: int) -> ActionInfo | None:
        """Get information about an action."""
        if 0 <= action < len(self.actions):
            return self.actions[action]
        return None

    def get_action_by_name(self, name: str) -> int | None:
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
    def enabled_actions(self) -> list[int]:
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


def create_action_mapper(
    use_config: bool = True,
    timeout_seconds: float = 30.0,
    action_id_map: dict[str, int] | None = None,
) -> ActionMapper:
    """
    Factory function to create action mapper.

    Args:
        use_config: Whether to respect configuration killswitches
        timeout_seconds: Maximum time for any single heuristic execution
        action_id_map: Optional stable mapping of heuristic names to action IDs

    Returns:
        Configured ActionMapper instance
    """
    return ActionMapper(
        use_config=use_config,
        timeout_seconds=timeout_seconds,
        action_id_map=action_id_map,
    )
