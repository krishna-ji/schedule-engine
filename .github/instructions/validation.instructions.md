---
applyTo: "src/validation/**/*.py"
---

# Validation & Feasibility Instructions

## Overview
Two-stage validation: input validation (data consistency) and feasibility checking (solvability analysis). Input validation in `src/validation/input_validator.py`, feasibility in `src/validation/feasibility_checker.py`.

## Input Validation

### Purpose
Check for data consistency issues before running GA:
- Missing course/instructor/room references
- Invalid time formats
- Duplicate IDs
- Negative capacities/hours

### Implementation Pattern
```python
def validate_input(context: SchedulingContext) -> Tuple[bool, List[str]]:
    """
    Returns:
        (is_valid, warnings): True if passable, warnings list
    """
    warnings = []
    
    # Check 1: References
    for group in context.groups.values():
        for course_code in group.enrolled_courses:
            if course_code not in context.courses:
                warnings.append(f"Group {group.id} enrolled in missing course {course_code}")
    
    # Check 2: Availability
    for instructor in context.instructors.values():
        if not instructor.available_quanta:
            warnings.append(f"Instructor {instructor.id} has no availability")
    
    # More checks...
    
    is_valid = len(warnings) == 0
    return is_valid, warnings
```

### When to Fail vs Warn
- **Fail (raise ValueError)**: Corrupted data, invalid JSON structure
- **Warn (return False, warnings)**: Missing optional data, suspicious values
- **Pass (return True, [])**: All checks passed

## Feasibility Checking

### Purpose
Detect **unsolvable** problems before GA runs:
- Instructor workload exceeds availability
- Not enough qualified instructors for courses
- Room capacity insufficient
- Pigeonhole violations (more required hours than time slots)

### Checks Implemented

#### 1. Instructor Workload vs Availability
```python
total_demand = sum(course.total_hours for course in courses)
total_supply = sum(len(instructor.available_quanta) for instructor in instructors)
if total_demand > total_supply:
    # CRITICAL: Not enough instructor time
```

#### 2. Instructor Qualification Bottleneck
```python
for course in courses:
    qualified_instructors = [i for i in instructors if course.code in i.qualifications]
    qualified_capacity = sum(len(i.available_quanta) for i in qualified_instructors)
    if qualified_capacity < course.total_hours:
        # CRITICAL: Not enough qualified instructor hours for this course
```

#### 3. Room Capacity Bottleneck
```python
total_seat_hours = sum(room.capacity * len(room.available_quanta) for room in rooms)
required_seat_hours = sum(course.total_hours * group.size for enrollments)
if total_seat_hours < required_seat_hours:
    # CRITICAL: Not enough room capacity
```

#### 4. Group Pigeonhole Problem
```python
for group in groups:
    required_hours = sum(course.total_hours for course in group.enrolled_courses)
    available_hours = len(group.available_quanta)
    if required_hours > available_hours:
        # CRITICAL: Group needs more hours than available time slots
```

### Configuration
```yaml
feasibility:
  enable_checks: true
  fail_on_infeasibility: true  # Stop execution if infeasible
  tolerance_margin: 0.02       # Allow 2% over-subscription
  generate_report: true        # Save to output directory
```

### Reporting Format
```
──────────────── FEASIBILITY ANALYSIS ────────────────

✓ Instructor Workload vs Availability
  Demand: 759 quanta, Supply: 2872 quanta ✓ (26.4%)

✗ Instructor Qualification Bottleneck
  1/239 courses lack qualified instructor capacity ✗
  Most critical bottlenecks:
    • Course CS101: needs 2h more from qualified instructors

──────────────────────── Summary ─────────────────────
Total Checks: 5 | Passed: 4 | Failed: 1
```

## Rules

### Input Validation
- Run before feasibility checks
- Log all warnings to console and file
- Never silently skip validation
- Return boolean + warnings list

### Feasibility Checks
- Run after input validation passes
- Each check returns `(passed: bool, severity: str, details: dict)`
- Severity levels: "critical", "warning", "info"
- Fail fast if `fail_on_infeasibility=True` and critical check fails

### Tolerance Margins
- Use `tolerance_margin` for approximate checks (e.g., 2% over-subscription allowed)
- Document why tolerance is needed (e.g., partial session overlap acceptable)
- Default to 0% tolerance for new checks

### Report Generation
- Save report to `output/evaluation_<timestamp>/feasibility_report.txt`
- Include actionable suggestions (e.g., "Qualify more instructors for Course X")
- Use Rich formatting for console output

## Adding New Checks

### Step 1: Implement Check Function
```python
# In src/validation/feasibility_checker.py
def check_lab_equipment(context: SchedulingContext, config) -> Tuple[bool, str, dict]:
    """Check if lab courses have sufficient equipment availability."""
    lab_courses = [c for c in context.courses.values() if c.type == "lab"]
    equipment_hours = sum(room.equipment_slots for room in context.rooms)
    required_hours = sum(c.total_hours for c in lab_courses)
    
    if equipment_hours < required_hours:
        return False, "critical", {
            "required": required_hours,
            "available": equipment_hours,
            "deficit": required_hours - equipment_hours
        }
    return True, "info", {}
```

### Step 2: Add to Config
```python
# In config/models.py
class FeasibilityConfig(BaseModel):
    # ...
    checks: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        # ... existing ...
        "lab_equipment": {"enabled": True, "severity": "critical"}
    })
```

### Step 3: Register in Checker
```python
# In src/validation/feasibility_checker.py
def run_all_checks(context, config):
    checks = [
        # ... existing ...
        ("lab_equipment", check_lab_equipment),
    ]
    # ...
```

## Never Do
- ❌ Skip validation if config says `validate=False` (always validate inputs)
- ❌ Modify context during validation (read-only)
- ❌ Use hardcoded thresholds (get from config)
- ❌ Raise exceptions in check functions (return status tuples)
- ❌ Proceed with GA if critical checks fail (unless explicitly allowed)
