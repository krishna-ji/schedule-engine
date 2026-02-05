"""Centralized constants for Schedule Engine.

Defines magic numbers, default values, and configuration constants used
throughout the application for better maintainability and discoverability.
"""

from __future__ import annotations

from typing import Final

# ================
# GENETIC ALGORITHM DEFAULTS
# ================

# Fitness weights for multi-objective optimization
# Both objectives are minimized: (hard_violations, soft_penalties)
FITNESS_WEIGHTS: Final[tuple[float, float]] = (-1.0, -1.0)

# Default random seed for reproducibility
DEFAULT_SEED: Final[int] = 69

# Population strategy options
POPULATION_STRATEGIES: Final[tuple[str, ...]] = ("hybrid", "smart", "random")

# Hybrid population composition (must sum to 1.0)
HYBRID_GREEDY_RATIO: Final[float] = 0.25
HYBRID_SMART_RATIO: Final[float] = 0.50
HYBRID_RANDOM_RATIO: Final[float] = 0.25


# ================
# TIME SYSTEM CONSTANTS
# ================

# Quantum duration in minutes
DEFAULT_QUANTUM_MINUTES: Final[int] = 60

# Operating hours defaults
DEFAULT_EARLIEST_TIME: Final[str] = "07:00"
DEFAULT_LATEST_TIME: Final[str] = "21:00"
DEFAULT_MIDDAY_START: Final[str] = "12:00"
DEFAULT_MIDDAY_END: Final[str] = "13:00"

# Session preferences
MAX_SESSION_COALESCENCE_DEFAULT: Final[int] = 3  # Max quanta per continuous session
MAX_SESSIONS_PER_DAY_DEFAULT: Final[int] = 4


# ================
# CONSTRAINT WEIGHTS (defaults if not in config)
# ================

# Hard constraint default weights
HARD_CONSTRAINT_WEIGHT_HIGH: Final[float] = 3.0
HARD_CONSTRAINT_WEIGHT_MEDIUM: Final[float] = 2.5
HARD_CONSTRAINT_WEIGHT_LOW: Final[float] = 2.0

# Soft constraint default weights
SOFT_CONSTRAINT_WEIGHT_HIGH: Final[float] = 2.0
SOFT_CONSTRAINT_WEIGHT_MEDIUM: Final[float] = 1.5
SOFT_CONSTRAINT_WEIGHT_LOW: Final[float] = 1.0

# Soft weight factor (scales soft penalties relative to hard)
SOFT_WEIGHT_FACTOR: Final[float] = 1.0


# ================
# REPAIR SYSTEM DEFAULTS
# ================

# Exhaustive search
EXHAUSTIVE_SEARCH_GENERATIONS: Final[tuple[int, int]] = (3, 25)
EXHAUSTIVE_SEARCH_COVERAGE: Final[float] = 0.3  # Top 30% of population
EXHAUSTIVE_SEARCH_MAX_NEIGHBORS: Final[int] = 100
EXHAUSTIVE_SEARCH_TIMEOUT: Final[int] = 180  # seconds

# Stagnation repair
STAGNATION_PATIENCE: Final[int] = 5  # Generations without improvement
STAGNATION_MIN_GEN: Final[int] = 8  # Don't trigger before gen 8
STAGNATION_COVERAGE: Final[float] = 0.5  # Top 50% of population
STAGNATION_MAX_ITERATIONS: Final[int] = 10
STAGNATION_TIMEOUT: Final[int] = 60  # seconds
STAGNATION_COOLDOWN: Final[int] = 3  # Generations between triggers

# Selective repair
SELECTIVE_REPAIR_PROBABILITY: Final[float] = 0.3  # 30% of offspring


# ================
# RL TRAINING DEFAULTS
# ================

# TensorBoard
TENSORBOARD_PORT: Final[int] = 6006
TENSORBOARD_LOG_DIR: Final[str] = "logs/tensorboard/train"

# Training
RL_TIMESTEPS_QUICK: Final[int] = 50_000
RL_TIMESTEPS_CURRICULUM: Final[int] = 100_000
RL_TIMESTEPS_FULL: Final[int] = 300_000

# Agent types
RL_AGENT_TYPES: Final[tuple[str, ...]] = ("ppo", "a2c", "dqn")

# Checkpoint
CHECKPOINT_DIR: Final[str] = "models/rl_agents/checkpoints"
CHECKPOINT_INTERVAL: Final[int] = 10_000


# ================
# OUTPUT & REPORTING
# ================

# Directory names
OUTPUT_BASE_DIR: Final[str] = "output"
LOGS_DIR: Final[str] = "logs"
MODELS_DIR: Final[str] = "models"

# File formats
EXPORT_FORMATS: Final[tuple[str, ...]] = ("json", "pdf", "csv", "png")

# Plot settings
PLOT_DPI: Final[int] = 300
PLOT_FIGSIZE: Final[tuple[int, int]] = (12, 8)


# ================
# VALIDATION & FEASIBILITY
# ================

# Tolerance margins
FEASIBILITY_TOLERANCE: Final[float] = 0.02  # 2% margin for capacity checks

# Severity levels
SEVERITY_CRITICAL: Final[str] = "critical"
SEVERITY_WARNING: Final[str] = "warning"
SEVERITY_INFO: Final[str] = "info"


# ================
# MULTIPROCESSING
# ================

# Worker defaults
DEFAULT_WORKER_COUNT: Final[int | None] = None  # None = CPU count
MIN_WORKERS: Final[int] = 1
MAX_WORKERS: Final[int] = 64


# ================
# DISPLAY & UI
# ================

# Console formatting
CONSOLE_WIDTH: Final[int] = 80
SEPARATOR_CHAR: Final[str] = "="
SUBSEPARATOR_CHAR: Final[str] = "-"

# Progress bar update interval
PROGRESS_UPDATE_INTERVAL: Final[float] = 1.0  # seconds

# Rich markup colors
COLOR_SUCCESS: Final[str] = "green"
COLOR_WARNING: Final[str] = "yellow"
COLOR_ERROR: Final[str] = "red"
COLOR_INFO: Final[str] = "cyan"
COLOR_DIM: Final[str] = "dim"


# ================
# RUNTIME MODES
# ================

# Mode categories
MODE_CATEGORY_BASELINE: Final[str] = "baseline"
MODE_CATEGORY_NSGA: Final[str] = "nsga"
MODE_CATEGORY_RL: Final[str] = "rl"
MODE_CATEGORY_HYBRID: Final[str] = "hybrid"

# Total mode count
TOTAL_RUNTIME_MODES: Final[int] = 10


# ================
# FILE PATHS (relative to project root)
# ================

# Input data
DATA_DIR: Final[str] = "data"
DATA_COURSES: Final[str] = "data/Course.json"
DATA_GROUPS: Final[str] = "data/Groups.json"
DATA_INSTRUCTORS: Final[str] = "data/Instructors.json"
DATA_ROOMS: Final[str] = "data/Rooms.json"

# Documentation
DOCS_DIR: Final[str] = "docs"
SCRIPTS_DIR: Final[str] = "scripts"
