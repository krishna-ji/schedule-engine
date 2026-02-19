"""
Instance Configuration for Schedule Engine

Institution/project-specific settings that rarely change.
These are shared across all experimental runs.

For experiment-specific parameters (ngen, pop_size, repair settings),
define them directly in your run files (runs/*.py).
"""

from __future__ import annotations

from pathlib import Path

# DATA PATHS
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
# TIME SYSTEM (Operating Hours)
# Quantum duration (in minutes)
QUANTUM_MINUTES = 60

# Operating hours (HH:MM format)
OPENING_TIME = "10:00"
CLOSING_TIME = "17:00"

# Midday break period
MIDDAY_BREAK_START = "12:00"
MIDDAY_BREAK_END = "13:00"

# Days when institution is closed
CLOSED_DAYS = ["Saturday"]

# Day-specific operating hours (if needed)
# Example: {"Friday": {"opening": "10:00", "closing": "14:00"}}
DAY_OVERRIDES = {}
# CONSTRAINT PENALTIES (Baseline weights - can override in run files)
# These are starting points. Tune in individual run files if needed.

# Hard constraint penalties (should be high to ensure feasibility)
PENALTY_INSTRUCTOR_CONFLICT = 100
PENALTY_GROUP_CONFLICT = 100
PENALTY_ROOM_CONFLICT = 100
PENALTY_INSTRUCTOR_AVAILABILITY = 50
PENALTY_GROUP_AVAILABILITY = 50
PENALTY_ROOM_AVAILABILITY = 50
PENALTY_INSTRUCTOR_QUALIFICATION = 75
PENALTY_ROOM_TYPE_MISMATCH = 75
PENALTY_ROOM_CAPACITY = 50
PENALTY_PAIRED_COHORT_PRACTICALS = 80

# Soft constraint penalties (schedule quality)
PENALTY_SCHEDULE_GAP = 5
PENALTY_SESSION_CLUSTERING = 3
PENALTY_LUNCH_BREAK = 4
PENALTY_LONG_DAY = 4
PENALTY_UNBALANCED_SCHEDULE = 3
PENALTY_ROOM_PREFERENCE = 1

# SOLVER SELECTION
# Default solver backend: "pymoo" (recommended) or "deap" (deprecated fallback)
# Can also be overridden via:
#   - CLI:  python solve.py --solver deap
#   - Env:  SCHED_SOLVER=deap
DEFAULT_SOLVER = "pymoo"

# PARALLEL PROCESSING
# Enable multiprocessing for fitness evaluation
USE_MULTIPROCESSING = True

# Number of worker processes (None = auto-detect CPU count)
NUM_WORKERS = None
# EXPORT SETTINGS
# Calendar export settings (for PDF generation)
CALENDAR_QUANTUM_MINUTES = 15  # Finer resolution for display
CALENDAR_START_HOUR = 7
CALENDAR_END_HOUR = 20
CALENDAR_DEFAULT_OUTPUT_PDF = "calendar.pdf"

# Enable/disable export formats
EXPORT_PDF = True
EXPORT_JSON = True
EXPORT_PLOTS = True
EXPORT_STATISTICS = True
# RANDOM SEEDS (for reproducibility)
# Default seed for GA experiments
DEFAULT_SEED = 42

# Default seed for RL experiments
DEFAULT_RL_SEED = 69
# LOGGING
# Log level for file output
LOG_LEVEL_FILE = "DEBUG"

# Log level for console output
LOG_LEVEL_CONSOLE = "INFO"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# VALIDATION & FEASIBILITY CHECKS
# Run feasibility checks before evolution
RUN_FEASIBILITY_CHECKS = True

# Tolerance margin for capacity checks (e.g., 1.02 = allow 2% oversubscription)
FEASIBILITY_TOLERANCE = 1.02

# Show feasibility warnings in console
SHOW_FEASIBILITY_WARNINGS = True
