"""
Centralized constants for Schedule Engine.

Defines magic numbers, default values, and configuration constants
used throughout the codebase for better maintainability.
"""

# ============================================================================
# GENETIC ALGORITHM DEFAULTS
# ============================================================================

# Fitness weights for multi-objective optimization
# Both objectives are minimized: (hard_violations, soft_penalties)
FITNESS_WEIGHTS = (-1.0, -1.0)

# Default random seed for reproducibility
DEFAULT_SEED = 69

# Population strategy options
POPULATION_STRATEGIES = ["hybrid", "smart", "random"]

# Hybrid population composition (must sum to 1.0)
HYBRID_GREEDY_RATIO = 0.25
HYBRID_SMART_RATIO = 0.50
HYBRID_RANDOM_RATIO = 0.25


# ============================================================================
# TIME SYSTEM CONSTANTS
# ============================================================================

# Quantum duration in minutes
DEFAULT_QUANTUM_MINUTES = 60

# Operating hours defaults
DEFAULT_EARLIEST_TIME = "07:00"
DEFAULT_LATEST_TIME = "21:00"
DEFAULT_MIDDAY_START = "12:00"
DEFAULT_MIDDAY_END = "13:00"

# Session preferences
MAX_SESSION_COALESCENCE_DEFAULT = 3  # Max quanta per continuous session
MAX_SESSIONS_PER_DAY_DEFAULT = 4


# ============================================================================
# CONSTRAINT WEIGHTS (defaults if not in config)
# ============================================================================

# Hard constraint default weights
HARD_CONSTRAINT_WEIGHT_HIGH = 3.0
HARD_CONSTRAINT_WEIGHT_MEDIUM = 2.5
HARD_CONSTRAINT_WEIGHT_LOW = 2.0

# Soft constraint default weights
SOFT_CONSTRAINT_WEIGHT_HIGH = 2.0
SOFT_CONSTRAINT_WEIGHT_MEDIUM = 1.5
SOFT_CONSTRAINT_WEIGHT_LOW = 1.0

# Soft weight factor (scales soft penalties relative to hard)
SOFT_WEIGHT_FACTOR = 0.01


# ============================================================================
# REPAIR SYSTEM DEFAULTS
# ============================================================================

# Exhaustive search
EXHAUSTIVE_SEARCH_GENERATIONS = [3, 25]
EXHAUSTIVE_SEARCH_COVERAGE = 0.3  # Top 30% of population
EXHAUSTIVE_SEARCH_MAX_NEIGHBORS = 100
EXHAUSTIVE_SEARCH_TIMEOUT = 180  # seconds

# Stagnation repair
STAGNATION_PATIENCE = 5  # Generations without improvement
STAGNATION_MIN_GEN = 8  # Don't trigger before gen 8
STAGNATION_COVERAGE = 0.5  # Top 50% of population
STAGNATION_MAX_ITERATIONS = 10
STAGNATION_TIMEOUT = 60  # seconds
STAGNATION_COOLDOWN = 3  # Generations between triggers

# Selective repair
SELECTIVE_REPAIR_PROBABILITY = 0.3  # 30% of offspring


# ============================================================================
# RL TRAINING DEFAULTS
# ============================================================================

# TensorBoard
TENSORBOARD_PORT = 6006
TENSORBOARD_LOG_DIR = "logs/tensorboard/train"

# Training
RL_TIMESTEPS_QUICK = 50_000
RL_TIMESTEPS_CURRICULUM = 100_000
RL_TIMESTEPS_FULL = 300_000

# Agent types
RL_AGENT_TYPES = ["ppo", "a2c", "dqn"]

# Checkpoint
CHECKPOINT_DIR = "models/rl_agents/checkpoints"
CHECKPOINT_INTERVAL = 10_000


# ============================================================================
# OUTPUT & REPORTING
# ============================================================================

# Directory names
OUTPUT_BASE_DIR = "output"
LOGS_DIR = "logs"
MODELS_DIR = "models"

# File formats
EXPORT_FORMATS = ["json", "pdf", "csv", "png"]

# Plot settings
PLOT_DPI = 300
PLOT_FIGSIZE = (12, 8)


# ============================================================================
# VALIDATION & FEASIBILITY
# ============================================================================

# Tolerance margins
FEASIBILITY_TOLERANCE = 0.02  # 2% margin for capacity checks

# Severity levels
SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"


# ============================================================================
# MULTIPROCESSING
# ============================================================================

# Worker defaults
DEFAULT_WORKER_COUNT = None  # None = CPU count
MIN_WORKERS = 1
MAX_WORKERS = 64


# ============================================================================
# DISPLAY & UI
# ============================================================================

# Console formatting
CONSOLE_WIDTH = 80
SEPARATOR_CHAR = "="
SUBSEPARATOR_CHAR = "-"

# Progress bar update interval
PROGRESS_UPDATE_INTERVAL = 1.0  # seconds

# Rich markup colors
COLOR_SUCCESS = "green"
COLOR_WARNING = "yellow"
COLOR_ERROR = "red"
COLOR_INFO = "cyan"
COLOR_DIM = "dim"


# ============================================================================
# RUNTIME MODES
# ============================================================================

# Mode categories
MODE_CATEGORY_BASELINE = "baseline"
MODE_CATEGORY_NSGA = "nsga"
MODE_CATEGORY_RL = "rl"
MODE_CATEGORY_HYBRID = "hybrid"

# Total mode count
TOTAL_RUNTIME_MODES = 10


# ============================================================================
# FILE PATHS (relative to project root)
# ============================================================================

# Configuration
CONFIGS_DIR = "configs"
CONFIG_BASE = "configs/base.yaml"
CONFIG_TEST = "configs/test.yaml"
CONFIG_PROD = "configs/prod.yaml"

# Input data
DATA_DIR = "data"
DATA_COURSES = "data/Course.json"
DATA_GROUPS = "data/Groups.json"
DATA_INSTRUCTORS = "data/Instructors.json"
DATA_ROOMS = "data/Rooms.json"

# Documentation
DOCS_DIR = "docs"
SCRIPTS_DIR = "scripts"
