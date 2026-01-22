---
applyTo: "src/constraints/**/*.py"
---

# Constraints Instructions

## Overview
Constraint predicates for CSP formulation: hard constraints (feasibility requirements, must evaluate to 0) and soft constraints (preference penalties, minimize total). Hard constraints in `src/constraints/hard.py`, soft in `src/constraints/soft.py`.

## Constraint Predicate Signature
```python
def constraint_name(
    decoded_schedule: List[CourseSession],  # Phenotype (decoded chromosome)
    context: SchedulingContext               # Static problem data (entities, time system)
) -> int:
    """
    Evaluate constraint violations over decoded phenotype.

    Returns:
        int: Violation count (hard) or penalty score (soft)
             - Hard: 0 = feasible, >0 = infeasible (count of conflicts)
             - Soft: 0 = ideal, >0 = suboptimal (weighted penalty sum)
    """
    penalty = 0
    # Constraint evaluation logic: iterate sessions, check predicates, accumulate violations
    return penalty
```

## Rules

### Hard Constraints
- Return integer count of violations
- Each violation = +1 (or +weight from config)
- Examples: instructor conflicts, room double-booking, unqualified instructors
- Goal: Reduce to 0 (feasible schedule)

### Soft Constraints
- Return integer penalty (higher = worse)
- Scaled by `soft_weight_factor` (default 0.01) in fitness evaluation
- Examples: schedule gaps, midday breaks, session clustering
- Goal: Minimize penalty

### Configuration Integration
```python
from config import get_config

def my_constraint(decoded_schedule, context):
    config = get_config()
    if not config.hard_constraints.my_constraint.enabled:
        return 0

    weight = config.hard_constraints.my_constraint.weight
    violations = count_violations(decoded_schedule)
    return violations * weight
```

### Time System Integration
- Use `QuantumTimeSystem` to convert between wall-clock and quanta
- Access via `context.qts`
- Get operating quanta: `context.available_quanta`
- Day names: `QuantumTimeSystem.DAY_NAMES` (Sunday-first)

### Performance Optimization
- Cache expensive computations within function
- Use dictionaries for O(1) lookups
- Avoid nested loops when possible
- Consider numpy/pandas for bulk operations

## Adding New Constraints

### Step 1: Implement Function
```python
# In src/constraints/hard.py or soft.py
def no_triple_booking(decoded_schedule: List[CourseSession], context: SchedulingContext) -> int:
    """Penalize if any group has 3+ sessions at same time."""
    penalty = 0
    # Implementation
    return penalty
```

### Step 2: Add to Config Models
```python
# In config/models.py
class HardConstraintsConfig(BaseModel):
    # ... existing ...
    no_triple_booking: ConstraintConfig = ConstraintConfig(enabled=True, weight=2.0)
```

### Step 3: Update YAML Files
```yaml
# In configs/{test,dev,prod}.yaml
hard_constraints:
  # ... existing ...
  no_triple_booking:
    enabled: true
    weight: 2.0
```

### Step 4: Register in Evaluator
```python
# In src/ga/evaluator/detailed_fitness.py or fitness.py
from schedule_engine.constraints.hard import no_triple_booking

# Add to constraint dict
hard_constraint_funcs = {
    "no_triple_booking": no_triple_booking,
    # ... existing ...
}
```

## Common Patterns

### Group Overlap Detection
```python
# Build time → groups mapping
time_to_groups = defaultdict(set)
for session in decoded_schedule:
    for group_id in session.group_ids:
        for quantum in session.quanta:
            time_to_groups[quantum].add(group_id)

# Count overlaps
for groups in time_to_groups.values():
    if len(groups) > 1:
        penalty += len(groups) - 1
```

### Instructor Availability Check
```python
for session in decoded_schedule:
    instructor = context.instructors[session.instructor_id]
    for quantum in session.quanta:
        if quantum not in instructor.available_quanta:
            penalty += 1
```

### Gap Penalty Calculation
```python
# Get sessions sorted by time for each group
for group_id in context.groups:
    group_sessions = sorted(
        [s for s in decoded_schedule if group_id in s.group_ids],
        key=lambda s: min(s.quanta)
    )
    # Calculate gaps between consecutive sessions
    for i in range(len(group_sessions) - 1):
        gap = min(group_sessions[i+1].quanta) - max(group_sessions[i].quanta) - 1
        if gap > 0:
            penalty += gap
```

## Never Do
-  Modify `decoded_schedule` or `context` (read-only)
-  Use global variables or cached state across calls
-  Raise exceptions for violations (return penalty instead)
-  Return negative penalties
-  Access files or network resources
-  Use `print()` (use return values for reporting)
