# Code Structure

## Directory Layout

```
schedule-engine/
├── main.py                      # CLI entry point (500 lines)
├── pyproject.toml               # Project metadata + UV config
├── configs/                     # YAML configurations
│   ├── base.yaml               # Base settings (468 lines)
│   ├── test.yaml               # Test environment overrides
│   ├── prod.yaml               # Production environment overrides
│   ├── baseline/               # Mode 1: Pure NSGA-II
│   │   └── 1-pure-nsga.yaml
│   ├── nsga/                   # Modes 2-4: NSGA-II variants
│   │   ├── 2-nsga-repairs.yaml
│   │   ├── 3-nsga-heuristics.yaml
│   │   └── 4-nsga-full.yaml
│   ├── rl/                     # Modes 5,7-10: RL-enhanced
│   │   ├── 5-rl-guided.yaml
│   │   ├── 7-rl-specialists.yaml
│   │   ├── 8-archive-diversity.yaml
│   │   ├── 9-rl-hierarchical.yaml
│   │   └── 10-rl-multiagent.yaml
│   └── hybrid/                 # Mode 6: Hybrid approach
│       └── 6-roundrobin.yaml
├── data/                        # Input JSON files
│   ├── Course.json             # 150 courses
│   ├── Groups.json             # 30 student groups
│   ├── Instructors.json        # 50 instructors
│   └── Rooms.json              # 40 rooms
├── src/                        # Source code (~15,000 lines)
│   ├── __init__.py
│   ├── constants.py            # Global constants
│   ├── exceptions.py           # Custom exceptions
│   ├── cli/                    # Command-line interface
│   │   └── commands.py         # UV command implementations
│   ├── config/                 # Configuration system (1,200 lines)
│   │   ├── __init__.py
│   │   ├── loader.py           # YAML loading + deep merge
│   │   ├── models.py           # Pydantic config models
│   │   └── runtime_mode.py     # RuntimeMode enum (10 modes)
│   ├── core/                   # GA core (2,500 lines)
│   │   ├── ga_scheduler.py     # Main evolution loop (2,152 lines)
│   │   └── types.py            # Core type definitions
│   ├── entities/               # Domain models (800 lines)
│   │   ├── course.py           # Course entity
│   │   ├── group.py            # Student group entity
│   │   ├── instructor.py       # Instructor entity
│   │   └── room.py             # Room entity
│   ├── encoder/                # Data encoding (1,000 lines)
│   │   ├── input_encoder.py    # JSON → Python objects
│   │   └── quantum_time_system.py  # Time discretization
│   ├── decoder/                # Result decoding (600 lines)
│   │   └── schedule_decoder.py # Individual → CourseSession
│   ├── validation/             # Input validation (1,500 lines)
│   │   ├── input_validator.py  # Data integrity checks
│   │   └── feasibility_checker.py  # Constraint feasibility
│   ├── constraints/            # Constraint functions (2,000 lines)
│   │   ├── hard_*.py           # 14 hard constraint functions
│   │   └── soft_*.py           # 4 soft constraint functions
│   ├── ga/                     # GA components (4,000 lines)
│   │   ├── sessiongene.py      # Gene definition
│   │   ├── population.py       # Population initialization
│   │   ├── operators/          # Genetic operators
│   │   │   ├── crossover.py    # Course-group-aware crossover
│   │   │   ├── mutation.py     # Smart mutation
│   │   │   └── repair_igls.py  # IGLS repair system (800 lines)
│   │   └── evaluator/          # Fitness evaluation
│   │       ├── fitness.py      # CPU evaluation
│   │       ├── detailed_fitness.py  # Detailed reporting
│   │       └── gpu_batch_evaluator.py  # GPU acceleration (500 lines)
│   ├── heuristics/             # Heuristic toolbox (3,000 lines)
│   │   ├── registry.py         # Heuristic registry
│   │   ├── parallel_executor.py  # Parallel execution
│   │   ├── construction/       # Construction operators (2 ops)
│   │   ├── perturbation/       # Perturbation operators (9 ops)
│   │   ├── repair/             # Repair operators (3 ops)
│   │   ├── optimization/       # Optimization operators (3 ops)
│   │   └── diversity/          # Diversity operators (2 ops)
│   ├── rl/                     # RL integration (5,200 lines)
│   │   ├── __init__.py
│   │   ├── gym_env/            # Gymnasium environment (1,500 lines)
│   │   │   ├── schedule_env.py # Main environment class
│   │   │   ├── state_encoder.py  # 25D state encoding
│   │   │   ├── action_mapper.py  # 20 discrete actions
│   │   │   └── reward_calculator.py  # Multi-component rewards
│   │   ├── agents/             # RL agent wrappers (800 lines)
│   │   │   ├── ppo_agent.py    # PPO wrapper
│   │   │   ├── dqn_agent.py    # DQN wrapper
│   │   │   └── random_agent.py # Baseline agent
│   │   ├── training/           # Training infrastructure (1,500 lines)
│   │   │   ├── train_script.py # Training entry point
│   │   │   ├── curriculum.py   # Curriculum learning
│   │   │   └── callbacks.py    # Training callbacks
│   │   ├── deployment/         # Production deployment (800 lines)
│   │   │   ├── model_loader.py # Fast model loading
│   │   │   └── inference.py    # Inference engine
│   │   └── evaluation/         # Evaluation & benchmarking (600 lines)
│   │       ├── baselines.py    # Baseline strategies
│   │       └── evaluator.py    # Performance evaluation
│   ├── diversity/              # Diversity mechanisms (400 lines)
│   │   └── archive.py          # Behavioral archive
│   ├── metrics/                # Performance metrics (300 lines)
│   │   └── diversity.py        # Diversity calculations
│   ├── exporter/               # Result export (1,000 lines)
│   │   ├── json_exporter.py    # JSON serialization
│   │   ├── pdf_exporter.py     # PDF calendar generation
│   │   └── plot_generator.py   # Evolution plots
│   ├── utils/                  # Utilities (800 lines)
│   │   ├── console_service.py  # Rich console wrapper
│   │   └── logging_setup.py    # Logging configuration
│   └── workflows/              # Orchestration (1,200 lines)
│       ├── standard_run.py     # Standard workflow
│       └── experiment_manager.py  # Experiment tracking
├── scripts/                    # Utility scripts
│   ├── cli.py                  # CLI command implementations
│   ├── interactive_launcher.py # Interactive menu
│   ├── diagnostics/            # System diagnostics
│   ├── training/               # RL training scripts
│   ├── benchmarking/           # Performance benchmarks
│   └── validation/             # Validation utilities
├── test/                       # Test files (3,000 lines)
│   ├── unit/                   # Unit tests (>80% coverage target)
│   └── rl/                     # RL-specific tests
├── docs/                       # Documentation
│   ├── get-started/            # Installation & setup
│   ├── architecture/           # System design
│   ├── code/                   # Code documentation
│   ├── how-to/                 # Developer guides
│   ├── references/             # API & library docs
│   ├── troubleshooting/        # Common issues
│   ├── research-papers/        # Academic references
│   ├── development/            # Developer notes
│   └── ai/                     # AI suggestions
├── output/                     # Generated results
│   ├── experiment_manifest.json  # Experiment tracking
│   └── evaluation_*/           # Timestamped runs
└── models/                     # Trained RL models
    └── rl_agents/              # Saved agent checkpoints
```

## Important Modules

### 1. Core GA Engine

**`src/core/ga_scheduler.py`** (2,152 lines)
- Main evolution loop
- NSGA-II selection implementation
- Operator application logic
- Stagnation detection
- RL integration hooks
- GPU batch evaluation

**Key Functions:**
- `GAScheduler.__init__()` - Initialize GA components
- `GAScheduler._init_toolbox()` - Set up DEAP toolbox
- `GAScheduler.evolve()` - Main evolution loop
- `GAScheduler._evaluate_population()` - Fitness evaluation
- `GAScheduler._apply_operators()` - Crossover/mutation/heuristics

### 2. Configuration System

**`src/config/loader.py`** (~300 lines)
- Hierarchical YAML loading
- Deep merge strategy
- Environment variable support
- Runtime mode resolution

**`src/config/models.py`** (~700 lines)
- Pydantic configuration models
- Type validation
- Nested configuration structure
- Default value handling

**`src/config/runtime_mode.py`** (~200 lines)
- RuntimeMode enum (10 modes)
- Mode metadata and descriptions
- Killswitch validation
- Mode comparison utilities

**Key Functions:**
- `load_config(runtime_mode, env)` - Load merged config
- `get_config()` - Get cached config instance
- `init_config(path)` - Initialize config from path

### 3. Data Encoding/Decoding

**`src/encoder/input_encoder.py`** (~600 lines)
- JSON file loading
- Entity construction (Course, Group, Instructor, Room)
- Cross-reference linking
- Data validation

**`src/encoder/quantum_time_system.py`** (~400 lines)
- Time discretization (continuous → discrete)
- Quantum mapping (wall clock ↔ quantum index)
- Time constraint checking
- Break period handling

**`src/decoder/schedule_decoder.py`** (~600 lines)
- Individual → CourseSession conversion
- Session consolidation
- Conflict resolution
- Schedule validation

### 4. Constraint System

**Hard Constraints** (`src/constraints/hard_*.py`)
- `hard_student_group_exclusivity.py` - Groups can't overlap
- `hard_instructor_exclusivity.py` - Instructors can't overlap
- `hard_instructor_qualifications.py` - Instructor must be qualified
- `hard_instructor_time_availability.py` - Instructor must be available
- `hard_room_suitability.py` - Room must be suitable
- `hard_room_exclusivity.py` - Rooms can't overlap
- `hard_room_time_availability.py` - Room must be available
- `hard_course_completeness.py` - All sessions must be scheduled

**Soft Constraints** (`src/constraints/soft_*.py`)
- `soft_avoid_early_sessions.py` - Prefer later start times
- `soft_avoid_late_sessions.py` - Prefer earlier end times
- `soft_instructor_preferences.py` - Honor instructor preferences
- `soft_minimize_gaps.py` - Minimize schedule gaps

**Key Functions:**
- Each constraint: `evaluate(individual, context, config)` → int (violations)

### 5. Heuristic Toolbox

**`src/heuristics/registry.py`** (~300 lines)
- Global heuristic registry
- Operator metadata (name, category, enabled, performance)
- Lookup by ID or category

**Operator Categories:**
1. **Construction** (`src/heuristics/construction/`)
   - `greedy_construction.py` - Greedy schedule building
   - `smart_construction.py` - Constraint-aware building

2. **Perturbation** (`src/heuristics/perturbation/`)
   - `swap_*.py` - Swap rooms/times/instructors
   - `shift_*.py` - Shift sessions in time/room
   - `relocate_*.py` - Move sessions to new slots

3. **Repair** (`src/heuristics/repair/`)
   - `repair_violated_sessions.py` - Fix specific violations
   - `repair_group_conflicts.py` - Resolve group overlaps
   - `repair_room_conflicts.py` - Resolve room conflicts

4. **Optimization** (`src/heuristics/optimization/`)
   - `consolidate_blocks.py` - Merge adjacent sessions
   - `reduce_fragmentation.py` - Minimize gaps
   - `balance_workload.py` - Even instructor distribution

5. **Diversity** (`src/heuristics/diversity/`)
   - `diversity_injection.py` - Add novel solutions
   - `novelty_search.py` - Explore behavioral space

### 6. RL Integration

**`src/rl/gym_env/schedule_env.py`** (~500 lines)
- OpenAI Gym environment
- Standard interface: `reset()`, `step()`, `render()`
- Episode management (generation-based)
- Reward calculation integration

**`src/rl/gym_env/state_encoder.py`** (~300 lines)
- 25-dimensional state vector
- Population statistics (fitness, diversity)
- Constraint breakdown (per-type violations)
- Historical context (last 5 generations)
- Normalization (all features in [0, 1])

**`src/rl/gym_env/action_mapper.py`** (~400 lines)
- 20 discrete actions (19 heuristics + 1 no-op)
- Action → heuristic mapping
- Heuristic application logic
- Performance tracking

**`src/rl/gym_env/reward_calculator.py`** (~300 lines)
- Multi-component reward:
  - Fitness improvement (+)
  - Diversity bonus (+)
  - Time penalty (-)
- Configurable weights
- Normalization strategies

**`src/rl/training/train_script.py`** (~400 lines)
- Training entry point
- Curriculum learning support
- TensorBoard logging
- Checkpoint management

**`src/rl/deployment/inference.py`** (~300 lines)
- Fast model loading (<100ms target)
- Inference optimization (<10ms target)
- Timeout protection
- Fallback strategies

### 7. GPU Acceleration

**`src/ga/evaluator/gpu_batch_evaluator.py`** (~500 lines)
- PyTorch CUDA implementation
- Batch constraint evaluation
- 10-50x speedup vs CPU
- Automatic CPU fallback
- Memory management

**Key Functions:**
- `GPUConstraintEvaluator.evaluate_batch()` - Batch evaluation
- `_prepare_batch_tensors()` - Convert to GPU tensors
- `_evaluate_constraints_gpu()` - GPU constraint checks
- `_aggregate_violations()` - Sum violations

### 8. Repair System

**`src/ga/operators/repair_igls.py`** (~800 lines)
- Iterative Greedy Local Search (IGLS)
- Exhaustive initial repair
- Stagnation-triggered repair
- Selective repair (violated sessions only)
- Neighborhood exploration

**Key Functions:**
- `repair_igls()` - Main IGLS entry point
- `_exhaustive_repair()` - Try all moves exhaustively
- `_iterative_repair()` - Iterative improvement
- `_validate_repair()` - Check repair success

### 9. Workflow Orchestration

**`src/workflows/standard_run.py`** (~600 lines)
- End-to-end pipeline orchestration
- Load → Validate → Evolve → Decode → Export
- Error handling
- Progress tracking

**`src/workflows/experiment_manager.py`** (~600 lines)
- Experiment tracking (manifest.json)
- Output directory management
- Run metadata storage
- Comparison utilities

## Key Directories

### `/configs`
**Purpose:** YAML configuration files

**Structure:**
- `base.yaml` - Common settings (468 lines)
- `{env}.yaml` - Environment overrides (test/prod)
- `{category}/{mode}.yaml` - Runtime modes (10 modes)

**Best Practices:**
- Never modify base.yaml directly (use overrides)
- Document custom configs with comments
- Test configs with `uv run verify-config`

### `/data`
**Purpose:** Input JSON files

**Files:**
- `Course.json` - Course definitions
- `Groups.json` - Student group definitions
- `Instructors.json` - Instructor profiles
- `Rooms.json` - Room specifications

**Best Practices:**
- Validate with `uv run check-data`
- Backup before modifications
- Use `data/archive/` for old versions

### `/src`
**Purpose:** Source code (all production code here)

**Organization:**
- 1 module = 1 subdirectory
- `__init__.py` in every package
- Flat hierarchy (max 2-3 levels deep)

**Best Practices:**
- Follow PEP 8 (88 line length)
- Add docstrings to all public functions
- Use type hints throughout

### `/test`
**Purpose:** ALL test files (unit + integration)

**Organization:**
- `test/unit/` - Unit tests
- `test/rl/` - RL-specific tests
- Mirror `src/` structure

**Best Practices:**
- >80% coverage target
- Test-driven development (TDD)
- Use pytest fixtures for reusable setup

### `/docs`
**Purpose:** Comprehensive documentation

**Organization:**
- `get-started/` - Installation & setup
- `architecture/` - System design
- `code/` - Code documentation
- `how-to/` - Developer guides
- `references/` - API docs
- `troubleshooting/` - Common issues
- `research-papers/` - Academic references
- `development/` - Developer notes
- `ai/` - AI suggestions

### `/output`
**Purpose:** Generated experiment results

**Structure:**
```
output/
├── experiment_manifest.json      # All experiments
├── evaluation_20251120_123456/  # Timestamped run
│   ├── schedule.json            # Detailed schedule
│   ├── calendar.pdf             # Visual timetable
│   ├── report.txt               # Summary report
│   └── plots/                   # Evolution curves
│       ├── fitness_evolution.png
│       ├── diversity_evolution.png
│       └── pareto_front.png
```

### `/models`
**Purpose:** Trained RL agent checkpoints

**Structure:**
```
models/
└── rl_agents/
    ├── ppo_best.zip             # Production model
    ├── ppo_checkpoint_50000.zip # Intermediate checkpoints
    └── manifest.json            # Model metadata
```

## Coding Conventions

### Naming Conventions

**Files:**
- `snake_case.py` for modules
- `test_*.py` for test files

**Classes:**
- `PascalCase` for class names
- Example: `GAScheduler`, `SchedulingContext`, `CourseSession`

**Functions/Variables:**
- `snake_case` for functions and variables
- Example: `evaluate_individual()`, `best_fitness`

**Constants:**
- `UPPER_SNAKE_CASE` for constants
- Example: `MAX_GENERATIONS`, `DEFAULT_POPULATION_SIZE`

**Private:**
- `_prefix` for private functions/attributes
- Example: `_init_toolbox()`, `_worker_context`

### Import Organization

```python
# 1. Standard library imports (alphabetical)
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional

# 2. Third-party imports (alphabetical)
import numpy as np
import torch
from deap import base, tools
from rich.console import Console

# 3. Local imports (alphabetical)
from src.config import get_config
from src.entities.course import Course
from src.ga.sessiongene import SessionGene
```

### Docstring Format

```python
def evaluate_individual(
    individual: Individual,
    context: SchedulingContext,
    config: Config
) -> tuple[int, float]:
    """
    Evaluate fitness of an individual schedule.
    
    Calculates hard constraint violations and soft penalty
    using configured constraint weights.
    
    Args:
        individual: Chromosome (list of SessionGene)
        context: Scheduling problem data
        config: Configuration object
    
    Returns:
        Tuple of (hard_violations, soft_penalty)
        Both values are negative (minimization objectives).
    
    Example:
        >>> fitness = evaluate_individual(ind, context, config)
        >>> print(fitness)
        (-50, -28.7)  # 50 hard violations, 28.7 soft penalty
    """
    # Implementation...
```

### Error Handling

```python
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    console.print(f"[red]Error:[/red] {e}")
    raise  # Re-raise for proper handling
```

### Logging

```python
import logging
from src.utils.console_service import get_console

logger = logging.getLogger(__name__)
console = get_console()

# For debugging (logs/schedule_engine.log)
logger.debug("Detailed debug info")

# For user-facing messages (terminal)
console.print("[cyan]Loading data...[/cyan]")
```

## File Size Guidelines

**Recommended:**
- **Small modules:** <300 lines (single responsibility)
- **Medium modules:** 300-800 lines (well-organized)
- **Large modules:** 800-1500 lines (complex logic, well-documented)
- **Very large:** >1500 lines (rare, only for core components like ga_scheduler.py)

**When to split:**
- Module >1000 lines → Consider splitting by functionality
- Function >100 lines → Consider refactoring
- Class >500 lines → Consider extracting helpers

## Navigation Tips

### Find Entry Point
```powershell
# Main entry
main.py → workflows/standard_run.py → core/ga_scheduler.py
```

### Find Configuration
```powershell
# Config path
configs/base.yaml → config/loader.py → config/models.py
```

### Find Constraint
```powershell
# Constraint implementation
constraints/ → hard_*.py or soft_*.py → specific function
```

### Find Heuristic
```powershell
# Heuristic implementation
heuristics/registry.py → heuristics/{category}/{operator}.py
```

### Find RL Component
```powershell
# RL integration
rl/gym_env/schedule_env.py → state_encoder.py, action_mapper.py, reward_calculator.py
```

## See Also

- [High-Level Architecture](../architecture/01-high-level-architecture.md) - System overview
- [Data Flow](../architecture/04-data-flow.md) - Component interactions
- [How-To Guides](../how-to/) - Common developer tasks
