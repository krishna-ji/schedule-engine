"""
Curriculum Learning Manager for RL agent training.

Implements progressive difficulty training:
- Stage 1 (Easy): Small problems (10 courses)
- Stage 2 (Medium): Medium problems (20 courses)
- Stage 3 (Hard): Large problems (40+ courses)

Features:
- Automatic stage progression based on validation performance
- Checkpoint management per stage
- Validation set evaluation
- Adaptive advancement logic
"""

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import get_config
from src.encoder import SchedulingContext
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CurriculumStage:
    """
    Represents one stage in curriculum learning.

    Attributes:
        name: Stage name (e.g., "easy", "medium", "hard")
        num_courses: Target number of courses for this difficulty
        num_episodes: Number of training episodes for this stage
        max_generations: Max GA generations per episode
        checkpoint_every: Checkpoint frequency (in episodes)
        validation_episodes: Number of validation episodes per checkpoint
        threshold: Performance threshold for advancement (mean reward)
        advancement_patience: Consecutive checkpoints above threshold to advance
    """

    name: str
    num_courses: int
    num_episodes: int
    max_generations: int
    checkpoint_every: int
    validation_episodes: int
    threshold: float
    advancement_patience: int


class CurriculumManager:
    """
    Manages multi-stage curriculum learning for RL training.

    Coordinates:
    - Stage progression
    - Problem filtering by difficulty
    - Validation set creation
    - Checkpoint and advancement logic
    """

    def __init__(
        self,
        context: SchedulingContext,
        stages: list[dict[str, Any]] | None = None,
        validation_ratio: float = 0.2,
        random_seed: int | None = None,
    ):
        """
        Initialize curriculum manager.

        Args:
            context: Full scheduling context
            stages: List of stage configurations (uses config if None)
            validation_ratio: Fraction of problems reserved for validation
            random_seed: Random seed for reproducibility
        """
        self.context = context
        self.random_seed = random_seed
        self.validation_ratio = validation_ratio

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # Load stages from config or use provided
        if stages is None:
            config = get_config()
            stages = (
                config.rl.training.curriculum
                if hasattr(config.rl.training, "curriculum")
                else []
            )

        # Parse stages
        self.stages = self._parse_stages(stages)
        self.current_stage_idx = 0

        # Validation tracking
        self.validation_scores: dict[str, list[float]] = {
            s.name: [] for s in self.stages
        }
        self.advancement_counter = 0

        logger.info(f"Initialized CurriculumManager with {len(self.stages)} stages")
        for stage in self.stages:
            logger.info(
                f"  Stage '{stage.name}': {stage.num_courses} courses, "
                f"{stage.num_episodes} episodes, threshold={stage.threshold}"
            )

    def _parse_stages(
        self, stage_configs: list[dict[str, Any]]
    ) -> list[CurriculumStage]:
        """Parse stage configurations into CurriculumStage objects."""
        stages = []

        for config in stage_configs:
            # Handle nested sample_config
            sample_config = config.get("sample_config", {})

            stage = CurriculumStage(
                name=config["name"],
                num_courses=sample_config.get(
                    "num_courses", config.get("num_courses", 10)
                ),
                num_episodes=config.get("num_episodes", 200),
                max_generations=config.get("max_generations", 100),
                checkpoint_every=config.get("checkpoint_every", 25),
                validation_episodes=config.get("validation_episodes", 5),
                threshold=config.get("threshold", 0.0),
                advancement_patience=config.get("advancement_patience", 3),
            )
            stages.append(stage)

        return stages

    def get_current_stage(self) -> CurriculumStage | None:
        """Get current training stage."""
        if self.current_stage_idx >= len(self.stages):
            return None
        return self.stages[self.current_stage_idx]

    def advance_stage(self) -> bool:
        """
        Advance to next curriculum stage.

        Returns:
            True if advanced, False if already at final stage
        """
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.advancement_counter = 0

            new_stage = self.get_current_stage()
            logger.info(
                f"Advanced to stage '{new_stage.name}' ({self.current_stage_idx + 1}/{len(self.stages)})"
            )
            return True

        logger.info("Already at final curriculum stage")
        return False

    def should_advance(self, validation_score: float) -> bool:
        """
        Check if should advance to next stage based on validation performance.

        Args:
            validation_score: Current validation mean reward

        Returns:
            True if advancement criteria met
        """
        current_stage = self.get_current_stage()
        if current_stage is None:
            return False

        # Record validation score
        self.validation_scores[current_stage.name].append(validation_score)

        # Check if above threshold
        if validation_score >= current_stage.threshold:
            self.advancement_counter += 1

            logger.info(
                f"Validation score {validation_score:.4f} >= threshold {current_stage.threshold:.4f} "
                f"({self.advancement_counter}/{current_stage.advancement_patience})"
            )

            # Check advancement patience
            if self.advancement_counter >= current_stage.advancement_patience:
                logger.info(
                    f"Advancement criteria met for stage '{current_stage.name}'"
                )
                return True
        else:
            self.advancement_counter = 0
            logger.debug(
                f"Validation score {validation_score:.4f} < threshold {current_stage.threshold:.4f}"
            )

        return False

    def filter_courses_by_difficulty(
        self,
        target_num_courses: int,
        strategy: str = "random",
    ) -> list:
        """
        Filter courses to match target difficulty.

        Args:
            target_num_courses: Target number of courses
            strategy: Selection strategy ("random", "smallest", "largest")

        Returns:
            Filtered list of courses
        """
        all_courses = self.context.courses

        if len(all_courses) <= target_num_courses:
            return all_courses

        if strategy == "random":
            return random.sample(all_courses, target_num_courses)
        elif strategy == "smallest":
            # Sort by total sessions (theory + practical)
            sorted_courses = sorted(
                all_courses, key=lambda c: c.theory_sessions + c.practical_sessions
            )
            return sorted_courses[:target_num_courses]
        elif strategy == "largest":
            sorted_courses = sorted(
                all_courses,
                key=lambda c: c.theory_sessions + c.practical_sessions,
                reverse=True,
            )
            return sorted_courses[:target_num_courses]
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def create_validation_set(
        self,
        num_problems: int = 5,
        seed: int | None = None,
    ) -> list[SchedulingContext]:
        """
        Create validation set with different problem instances.

        Args:
            num_problems: Number of validation problems
            seed: Random seed for validation set

        Returns:
            List of validation contexts
        """
        if seed is not None:
            random.seed(seed)

        current_stage = self.get_current_stage()
        if current_stage is None:
            return []

        validation_set = []

        for i in range(num_problems):
            # Filter courses to match difficulty
            filtered_courses = self.filter_courses_by_difficulty(
                current_stage.num_courses, strategy="random"
            )

            # Create new context with filtered courses
            validation_context = SchedulingContext(
                courses=filtered_courses,
                instructors=self.context.instructors,
                rooms=self.context.rooms,
                groups=self.context.groups,
                time_system=self.context.time_system,
            )

            validation_set.append(validation_context)

        logger.debug(f"Created validation set with {num_problems} problems")
        return validation_set

    def get_training_context(self) -> SchedulingContext:
        """
        Get training context for current stage.

        Returns:
            SchedulingContext filtered for current stage difficulty
        """
        current_stage = self.get_current_stage()
        if current_stage is None:
            return self.context

        # Filter courses
        filtered_courses = self.filter_courses_by_difficulty(
            current_stage.num_courses, strategy="random"
        )

        # Create training context
        training_context = SchedulingContext(
            courses=filtered_courses,
            instructors=self.context.instructors,
            rooms=self.context.rooms,
            groups=self.context.groups,
            time_system=self.context.time_system,
        )

        return training_context

    def save_progress(self, save_path: str):
        """Save curriculum progress to file."""
        progress = {
            "current_stage_idx": self.current_stage_idx,
            "current_stage": (
                self.get_current_stage().name if self.get_current_stage() else None
            ),
            "advancement_counter": self.advancement_counter,
            "validation_scores": self.validation_scores,
            "timestamp": datetime.now().isoformat(),
        }

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(progress, f, indent=2)

        logger.info(f"Saved curriculum progress to {save_path}")

    def load_progress(self, load_path: str) -> bool:
        """
        Load curriculum progress from file.

        Returns:
            True if loaded successfully
        """
        load_path = Path(load_path)

        if not load_path.exists():
            logger.warning(f"Progress file not found: {load_path}")
            return False

        try:
            with open(load_path) as f:
                progress = json.load(f)

            self.current_stage_idx = progress["current_stage_idx"]
            self.advancement_counter = progress["advancement_counter"]
            self.validation_scores = progress["validation_scores"]

            logger.info(f"Loaded curriculum progress from {load_path}")
            logger.info(
                f"Resumed at stage '{self.get_current_stage().name}' ({self.current_stage_idx + 1}/{len(self.stages)})"
            )

            return True
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return False

    def get_statistics(self) -> dict[str, Any]:
        """Get curriculum training statistics."""
        current_stage = self.get_current_stage()

        stats = {
            "total_stages": len(self.stages),
            "current_stage_idx": self.current_stage_idx,
            "current_stage_name": current_stage.name if current_stage else "completed",
            "advancement_counter": self.advancement_counter,
            "validation_history": self.validation_scores,
        }

        # Add per-stage statistics
        for stage in self.stages:
            scores = self.validation_scores.get(stage.name, [])
            if scores:
                stats[f"stage_{stage.name}_mean"] = float(np.mean(scores))
                stats[f"stage_{stage.name}_best"] = float(np.max(scores))
                stats[f"stage_{stage.name}_num_evals"] = len(scores)

        return stats


def create_default_curriculum() -> list[dict[str, Any]]:
    """Create default 3-stage curriculum configuration."""
    return [
        {
            "name": "easy",
            "enabled": True,
            "num_episodes": 200,
            "max_generations": 100,
            "checkpoint_every": 25,
            "validation_episodes": 5,
            "sample_config": {"num_courses": 10},
            "threshold": -5.0,  # Mean reward threshold
            "advancement_patience": 3,
        },
        {
            "name": "medium",
            "enabled": True,
            "num_episodes": 300,
            "max_generations": 200,
            "checkpoint_every": 25,
            "validation_episodes": 5,
            "sample_config": {"num_courses": 20},
            "threshold": -3.0,
            "advancement_patience": 3,
        },
        {
            "name": "hard",
            "enabled": True,
            "num_episodes": 500,
            "max_generations": 400,
            "checkpoint_every": 50,
            "validation_episodes": 10,
            "sample_config": {"num_courses": 40},
            "threshold": -2.0,
            "advancement_patience": 3,
        },
    ]
