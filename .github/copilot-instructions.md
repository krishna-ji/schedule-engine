# Schedule Engine Agent Guide

## Project Summary

University course scheduling engine using **Google OR-Tools CP-SAT** constraint programming solver. Pure hard constraint satisfaction - no genetic algorithms, no soft constraints. Written in Python with clean logging and JSON export.

## Tech Stack

- **Language**: Python 3.8+
- **Solver**: Google OR-Tools CP-SAT (constraint programming)
- **Logging**: Python standard `logging` module
- **Output**: JSON only (no PDF, no plots)
- **Config**: Hardcoded in `main.py` (no YAML files)

## Repository Structure

```
schedule-engine/
├── main.py              # Entry point - pure CP-SAT runner
├── data/                # Input JSON files
├── src/
│   ├── ortools/         # CP-SAT scheduler implementation
│   │   └── cp_scheduler_clean.py  # Main solver
│   ├── encoder/         # JSON → entities + time system
│   ├── decoder/         # Solution → CourseSession
│   ├── entities/        # Domain models (Course, Group, etc.)
│   ├── validation/      # Input validation
│   └── workflows/       # Data loading utilities
├── test/                # Test files
└── output/              # Results (schedule_<timestamp>/)
```

## What Was Removed (Pure CP-SAT Refactor)

### Deleted Components
- ✂️ All GA code (`src/ga/`, `src/core/ga_scheduler.py`)
- ✂️ DEAP library dependency
- ✂️ Soft constraints (`src/constraints/soft.py`)
- ✂️ Multi-objective optimization (NSGA-II)
- ✂️ YAML configuration system (`config/`, `configs/`)
- ✂️ All plotting/reporting (`src/exporter/`, `src/workflows/reporting.py`)
- ✂️ PDF calendar generation
- ✂️ Repair heuristics
- ✂️ Feasibility checking
- ✂️ Rich console UI

### What Remains
- ✅ CP-SAT solver (`src/ortools/cp_scheduler_clean.py`)
- ✅ Input data loading (`src/workflows/standard_run.py::load_input_data()`)
- ✅ Entity models (`src/entities/`)
- ✅ Input validation (`src/validation/input_validator.py`)
- ✅ Time quantum system (`src/encoder/quantum_time_system.py`)
- ✅ Python standard logging
- ✅ JSON export only

## Running the Engine

```bash
python main.py
```

That's it! No arguments, no configuration. Settings hardcoded in `main.py`:
- **Time limit**: 0 (unlimited - runs until solution found)
- **Workers**: 4 (memory-safe for large problems)
- **Data**: Reads from `data/` directory
- **Output**: Saves to `output/schedule_<timestamp>/`

### Expected Output
```
2025-11-13 12:22:16 | INFO | CP-SAT Schedule Engine - Pure Constraint Programming
2025-11-13 12:22:16 | INFO | STEP 1: Loading input data
2025-11-13 12:22:16 | INFO |   Courses: 239
2025-11-13 12:22:16 | INFO | STEP 2: Input validation
2025-11-13 12:22:16 | INFO |   ✓ Input validation passed
2025-11-13 12:22:16 | INFO | STEP 3: Running CP-SAT solver
2025-11-13 12:22:16 | INFO |   Time limit: UNLIMITED
2025-11-13 12:22:16 | INFO |   Parallel workers: 4
```

### Runtime Expectations
- **Model building**: 2-5 minutes (creating 28M variables, 13M constraints)
- **Search**: Hours to days depending on problem feasibility
- **Memory**: ~4-6 GB (4 workers × 1-1.5 GB each)

## Key Concepts

### Hard Constraints Only
CP-SAT enforces these declaratively:
- No group overlaps (students can't be in two places)
- No instructor conflicts (instructor teaches one session at a time)
- No room conflicts (room hosts one session at a time)
- Availability respected (groups, instructors, rooms)
- Room type matching (lab courses → lab rooms, theory → theory rooms)
- Qualification requirements (instructors qualified for their courses)

### No Soft Constraints
Unlike GA version, there's no optimization of preferences:
- ❌ No gap minimization
- ❌ No midday break preferences  
- ❌ No session clustering
- ❌ No "better" vs "worse" feasible solutions

CP-SAT finds **any** schedule that satisfies all hard constraints. That's it.

### Time System
- **Quantum**: 60-minute discrete time unit
- **Week**: Sunday-Saturday (Sunday = day 0)
- **Hours**: 08:00-18:00 (10 quanta per day)
- **Total**: 72 quanta per week (7 days × 10 hours/day)

Always use `QuantumTimeSystem` for time conversions.

## Data Flow

```
JSON files (data/)
    ↓
load_input_data() 
    ↓
SchedulingContext (entities + quantum system)
    ↓
CPScheduler.generate_single_solution()
    ↓
List[CourseSession] (decoded schedule)
    ↓
export_schedule_json()
    ↓
output/schedule_<timestamp>/schedule.json
```

## Documentation Policy

### Code Documentation: Docstrings Only
- All code must be documented using Python docstrings
- Docstrings are the single source of truth
- Never create separate .md files to document code

### Change Documentation
- **Minor changes**: Add timestamped entry to `docs/code/{BUGFIX,ENHANCE,REFACTOR}.md`
- **Major changes**: Create thesis-ready document in `docs/for_report/`

## Commit Message Format

Format: `<type>(<scope>): <summary>`

- **Types**: `feat`, `fix`, `refactor`, `test`, `doc`, `data`
- Use imperative mood, keep under 72 characters
- Example: `feat(cpsat): add unlimited time limit support`

## General Coding Standards

- **Python Style**: PEP 8 compliant
- **Imports**: Standard lib → third-party → local (sorted alphabetically)
- **Type Hints**: Use where beneficial
- **Error Handling**: Informative error messages with context
- **Logging**: Use Python standard `logging` module
- **Docstrings**: Required for all modules, classes, functions

## Path-Specific Instructions

Detailed module instructions in `.github/instructions/`:

- `config.instructions.md` - Configuration (DEPRECATED)
- `ga-core.instructions.md` - GA operators (DEPRECATED - removed)
- `constraints.instructions.md` - Constraints (CP-SAT internal)
- `data-flow.instructions.md` - Encoder/decoder/entities
- `validation.instructions.md` - Input validation
- `export.instructions.md` - Reports (JSON only)
- `workflows.instructions.md` - Orchestration (simplified)
- `tests.instructions.md` - Testing guidelines
