# Constraint Mapping Reference - Single Source of Truth

**Last Updated**: November 22, 2025  
**Purpose**: Definitive mapping between constraint codes (hc1-hc8, sc1-sc4) and constraint functions

## Hard Constraints (hc1-hc8)

The constraint codes are **deterministically assigned** based on the order in `src/workflows/standard_run.py` lines 276-309.

| Code | Constraint Name | Description | Weight | Registry |
|------|----------------|-------------|---------|----------|
| **hc1** | `student_group_exclusivity` | Student group can only be in one session at a time | 3.0 | `src/constraints/hard.py` |
| **hc2** | `instructor_exclusivity` | Instructor can only teach one session at a time | 3.0 | `src/constraints/hard.py` |
| **hc3** | `instructor_qualifications` | Instructor must be qualified for assigned course | 3.0 | `src/constraints/hard.py` |
| **hc4** | `room_suitability` | Room must have required features for course | 2.5 | `src/constraints/hard.py` |
| **hc5** | `instructor_time_availability` | Instructor must be available at scheduled time | 3.0 | `src/constraints/hard.py` |
| **hc6** | `room_time_availability` | Room must be available at scheduled time | 2.5 | `src/constraints/hard.py` |
| **hc7** | `course_completeness` | Course must be scheduled for required quanta per group | 2.0 | `src/constraints/hard.py` |
| **hc8** | `room_exclusivity` | Room can only host one session at a time | 3.0 | `src/constraints/hard.py` |

## Soft Constraints (sc1-sc4)

| Code | Constraint Name | Description | Weight | Registry |
|------|----------------|-------------|---------|----------|
| **sc1** | `student_schedule_compactness` | Minimize gaps in student schedules | 3.0 | `src/constraints/soft.py` |
| **sc2** | `instructor_schedule_compactness` | Minimize gaps in instructor schedules | 2.0 | `src/constraints/soft.py` |
| **sc3** | `student_lunch_break` | Prefer lunch breaks around noon | 0.8 | `src/constraints/soft.py` |
| **sc4** | `session_continuity` | Prefer multi-quanta sessions to be contiguous | 2.0 | `src/constraints/soft.py` |

## Code Generation Logic

**Location**: `src/core/ga_scheduler.py` lines 315-322

```python
self.hard_constraint_codes = {
    name: f"hc{i+1}" for i, name in enumerate(self.hard_constraint_names)
}
self.soft_constraint_codes = {
    name: f"sc{i+1}" for i, name in enumerate(self.soft_constraint_names)
}
```

The `hard_constraint_names` list is populated in `src/workflows/standard_run.py` lines 276-291 using the **constraint registry**:

```python
from src.constraints.registry import get_all_hard_constraints, get_all_soft_constraints

# Build constraint lists dynamically from registry + config
all_hard_constraints = get_all_hard_constraints()
all_soft_constraints = get_all_soft_constraints()

# Get enabled constraints by checking config for each registered constraint
hard_names = []
for name in all_hard_constraints.keys():
    constraint_cfg = getattr(config.hard_constraints, name, None)
    if constraint_cfg and constraint_cfg.enabled:
        hard_names.append(name)
```

The order is **deterministic** because:
1. Registry dict preserves insertion order (Python 3.7+)
2. Constraints registered via decorators in `src/constraints/hard.py` in source order
3. `.keys()` iterator returns items in insertion order

**This is now a SINGLE SOURCE OF TRUTH** - constraint names come from the registry, not hardcoded duplicates!

## Constraint Registry (Decorator-Based)

**Location**: `src/constraints/registry.py`

All constraints are registered via decorators:
```python
@hard_constraint(
    name="student_group_exclusivity",
    description="Ensures each student group can only be in one session at a time",
    default_weight=3.0,
    needs_courses=False
)
def student_group_exclusivity(sessions: List[CourseSession]) -> int:
    # ... implementation ...
```

### Registry Access Functions

```python
from src.constraints.registry import (
    get_all_hard_constraints,      # Get all hard constraint metadata
    get_all_soft_constraints,      # Get all soft constraint metadata
    get_constraint_metadata,       # Get metadata by name
    get_hard_constraint_function,  # Get function by name
    get_enabled_hard_constraints,  # Get enabled constraints from config
)
```

## Usage in Code

### Console Output (ga_scheduler.py)

```python
# Display constraint breakdown
for name in self.hard_constraint_names:
    short_name = self.hard_constraint_codes.get(name, name[:4])
    hc_parts.append(f"{short_name}={int(hard_details.get(name, 0))}")

# Output: hc1=483, hc2=78, hc3=0, hc4=753, hc5=817, hc6=357, hc7=0, hc8=296
```

### Documentation References

When documenting constraint violations, **always use**:
- Full name: `course_completeness`
- Code: `hc8` (with mapping reference)
- Example: "Course completeness (`hc8`) violations..."

**Never assume** code numbers - always verify against this reference.

## GPU Evaluator Comments

The GPU batch evaluator (`src/ga/evaluator/gpu_batch_evaluator.py`) uses these codes in comments:

```python
# HC1: Student group exclusivity (line 511)
# HC2: Instructor exclusivity (line 512)  
# HC3: Instructor qualifications (line 557)
# HC4: Instructor time availability (line 562) ← NOT course completeness!
# HC5: Room suitability (line 544)
# HC6: Room time availability (line 567)
# HC7: Course completeness (line 572) ← Wait, this is WRONG in GPU code!
# HC8: Room exclusivity (line 513) ← This is also WRONG!
```

**CRITICAL**: The GPU evaluator comments are **INCORRECT** and don't match the actual constraint order. They need to be updated to match this reference.

## Related Files

- `src/workflows/standard_run.py` - Constraint dict definition (lines 276-291, uses registry)
- `src/core/ga_scheduler.py` - Code generation (lines 315-322), metrics initialization (lines 327-328)
- `src/constraints/registry.py` - Decorator-based registry (single source of truth)
- `src/constraints/hard.py` - Hard constraint implementations with decorators
- `src/constraints/soft.py` - Soft constraint implementations with decorators
- `src/ga/evaluator/fitness.py` - Uses `get_enabled_hard_constraints()` from registry
- `src/ga/evaluator/detailed_fitness.py` - Uses `get_enabled_hard_constraints()` from registry
- `src/ga/evaluator/gpu_batch_evaluator.py` - GPU evaluation (comments need fixing!)
- `src/exporter/plot_detailed_constraints.py` - **Plot generation (uses constraint names from registry)**
- `src/workflows/reporting.py` - Orchestrates plotting with `metrics.detailed_hard` and `metrics.detailed_soft`

## Plot System Integration

**Status**: ✅ **YES, plots dynamically use registry as single source of truth**

### Data Flow to Plots

1. **Metrics Initialization** (`src/core/ga_scheduler.py` lines 327-328):
   ```python
   self.metrics = GAMetrics(
       detailed_hard={name: [] for name in hard_constraint_names},  # From registry!
       detailed_soft={name: [] for name in soft_constraint_names},  # From registry!
   )
   ```

2. **Data Collection** (lines 1930-1934):
   ```python
   for name in self.hard_constraint_names:  # Iterates over registry names
       self.metrics.detailed_hard[name].append(hard_details[name])
   ```

3. **Plot Generation** (`src/workflows/reporting.py` lines 178-195):
   ```python
   plot_individual_hard_constraints(metrics.detailed_hard, output_dir)
   plot_individual_soft_constraints(metrics.detailed_soft, output_dir)
   ```
   - Receives `Dict[str, List[int]]` where keys are constraint names from registry
   - Plots automatically adapt to any number of constraints
   - Legend labels derived from constraint names

### Current Plot Behavior

**Plots show full constraint names** (e.g., "Student Group Exclusivity"), NOT codes (hc1-hc8).

**Files that generate plots**:
- `src/exporter/plothard.py` - Total hard violations over generations
- `src/exporter/plotsoft.py` - Total soft penalties over generations
- `src/exporter/plot_detailed_constraints.py` - Individual constraint trends + combined plots

**Dynamic adaptation**:
- ✅ Adding/removing constraints in `hard.py` automatically updates plots
- ✅ Legend entries generated from constraint names
- ✅ Number of subplots/colors adapts to constraint count
- ✅ No hardcoded constraint lists in plotting code

### Enhancement Opportunity

Currently plots show full names but not codes (hc1-hc8). To add codes to plot labels:

```python
# In plot_detailed_constraints.py, pass constraint_codes from scheduler:
def plot_individual_hard_constraints(
    hard_trends: Dict[str, List[int]], 
    output_dir: str,
    constraint_codes: Dict[str, str] = None  # NEW: hc1-hc8 mapping
):
    for constraint_name, trend in hard_trends.items():
        # Add code to label
        code = constraint_codes.get(constraint_name, "") if constraint_codes else ""
        label = f"{code} - {constraint_name.replace('_', ' ').title()}" if code else constraint_name.replace('_', ' ').title()
```

This would require passing `scheduler.hard_constraint_codes` from `ga_scheduler` → `reporting` → plot functions.

## Maintenance Notes

**When adding/removing constraints:**
1. Update `src/constraints/hard.py` or `soft.py` with `@hard_constraint` or `@soft_constraint` decorator
2. Update `src/workflows/standard_run.py` `hard_constraints_dict` (lines 276-309)
3. Update config schema in `src/config/model.py`
4. Update this reference document
5. Update GPU evaluator comments if applicable

**DO NOT** hardcode constraint codes (hc1, hc2, etc.) in logic - always use the registry and dynamic code generation system.
