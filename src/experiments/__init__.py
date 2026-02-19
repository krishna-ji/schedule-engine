"""Experiment runners — clean OOP wrappers for GA and RL experiments.

Each experiment is configured via constructor kwargs, then executed
with ``experiment.run()``.  Logging, output directories, timing, and
JSON result export are handled by the base class.

GA modes (pymoo-based):
    BaselineExperiment, MemeticExperiment, AggressiveExperiment,
    AdaptiveExperiment, CPHybridExperiment

RL experiments:
    RLTrainExperiment, RLCurriculumExperiment, RLSpecialistExperiment,
    RLRewardCompareExperiment, RLAdaptiveParamsExperiment,
    RLAblationExperiment, RLHyperparamSweepExperiment,
    RLMultiAgentExperiment, RLVerifyExperiment
"""

from .ga_experiment import (
    AdaptiveExperiment,
    AggressiveExperiment,
    BaselineExperiment,
    CPHybridExperiment,
    GAExperiment,
    MemeticExperiment,
)
from .rl_experiment import (
    RLAblationExperiment,
    RLAdaptiveParamsExperiment,
    RLCurriculumExperiment,
    RLHyperparamSweepExperiment,
    RLMultiAgentExperiment,
    RLRewardCompareExperiment,
    RLSpecialistExperiment,
    RLTrainExperiment,
    RLVerifyExperiment,
)

__all__ = [
    "AdaptiveExperiment",
    "AggressiveExperiment",
    "BaselineExperiment",
    "CPHybridExperiment",
    "GAExperiment",
    "MemeticExperiment",
    "RLAblationExperiment",
    "RLAdaptiveParamsExperiment",
    "RLCurriculumExperiment",
    "RLHyperparamSweepExperiment",
    "RLMultiAgentExperiment",
    "RLRewardCompareExperiment",
    "RLSpecialistExperiment",
    "RLTrainExperiment",
    "RLVerifyExperiment",
]
