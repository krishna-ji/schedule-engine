---
applyTo: "{src/encoder/**/*.py,src/decoder/**/*.py,src/entities/**/*.py}"
---

# Data Flow Instructions

## Overview
Data transformation pipeline: JSON → entities → chromosomes → decoded sessions. Encoders parse JSON, entities represent domain models, decoders convert GA solutions to schedules.

## Components

### Entities (`src/entities/`)
- `Course` - Course definition (code, name, L/T/P hours, type)
- `Group` - Student group (ID, enrolled courses, availability)
- `Instructor` - Faculty (ID, name, qualifications, availability)
- `Room` - Classroom (ID, capacity, type, availability)
- `CourseSession` - Decoded schedule entry (course, groups, instructor, room, time)

### Encoder (`src/encoder/input_encoder.py`)
- Loads JSON files from `data/` directory
- Converts to entity objects
- Links courses ↔ groups, courses ↔ instructors
- Creates `SchedulingContext` with all entities + time system
- **Key function**: `load_groups()`, `load_courses()`, `load_instructors()`, `load_rooms()`

### Quantum Time System (`src/encoder/quantum_time_system.py`)
- Converts wall-clock time to discrete quanta
- Default: 60-minute quanta, 7 days/week, 8:00-18:00
- **Key methods**: `time_to_quantum()`, `quantum_to_time()`, `get_all_operating_quanta()`
- Sunday-first ordering: `DAY_NAMES = ["Sunday", "Monday", ..., "Saturday"]`

### Decoder (`src/decoder/individual_decoder.py`)
- Converts GA individual (list[SessionGene]) → list[CourseSession]
- Resolves IDs to entity objects
- **Key function**: `decode_individual(individual, context) -> List[CourseSession]`

## Rules

### Entity Creation
- All entities must be immutable after creation (dataclass with frozen=True or similar)
- Use type hints for all fields
- Include docstrings explaining each field
- Validate data during parsing (raise ValueError for invalid inputs)

### JSON Schema Expectations
```json
// Course.json
{"courseCode": "CS101", "courseName": "...", "L": 3, "T": 1, "P": 0, "courseType": "theory"}

// Groups.json
{"groupID": "BCE1", "groupCourses": ["CS101", "MATH201"], "availability": {...}}

// Instructors.json
{"instructorID": "INS001", "instructorName": "...", "qualification": ["CS101"], "availability": {...}}

// Rooms.json
{"roomID": "R101", "roomCapacity": 50, "roomType": "theory", "availability": {...}}
```

### Availability Format
```json
"availability": {
  "Sunday": ["08:00-10:00", "14:00-16:00"],
  "Monday": ["09:00-17:00"]
}
```

### Time Quantum Rules
- Quanta are 0-indexed integers
- Sunday = day 0, Monday = day 1, ..., Saturday = day 6
- Each day divided into quanta (e.g., 10 quanta for 10 hours)
- Quantum uniquely identifies a time slot: `quantum = day * quanta_per_day + hour_offset`

### Decoding Rules
- One SessionGene → One CourseSession
- Preserve all gene information (course, groups, instructor, room, quanta)
- Convert quantum integers to readable time strings
- Handle missing entities gracefully (log warnings, don't crash)

## Adding New Entity Fields

### Step 1: Update JSON Schema
Add field to JSON files in `data/`

### Step 2: Update Entity Class
```python
# In src/entities/course.py
@dataclass
class Course:
    # ... existing fields ...
    prerequisites: List[str] = field(default_factory=list)  # NEW
```

### Step 3: Update Encoder
```python
# In src/encoder/input_encoder.py
def load_courses(json_path, qts):
    # ... existing parsing ...
    prerequisites = course_data.get("prerequisites", [])
    course = Course(..., prerequisites=prerequisites)
```

### Step 4: Update Decoder (if needed)
```python
# In src/decoder/individual_decoder.py
# Access new field in CourseSession creation
session = CourseSession(
    # ... existing ...
    prerequisites=course.prerequisites  # if CourseSession needs it
)
```

## Common Patterns

### Safe Quantum Conversion
```python
def wall_time_to_quantum(day_name: str, time_str: str, qts: QuantumTimeSystem) -> int:
    try:
        return qts.time_to_quantum(day_name, time_str)
    except ValueError as e:
        logger.warning(f"Invalid time {day_name} {time_str}: {e}")
        return None
```

### Default Availability
```python
# If availability missing, default to all operating quanta
if "availability" not in group_data:
    available_quanta = qts.get_all_operating_quanta()
else:
    available_quanta = parse_availability(group_data["availability"], qts)
```

### Linking Courses and Groups
```python
# After loading both
for group in groups.values():
    for course_code in group.enrolled_courses:
        if course_code in courses:
            courses[course_code].enrolled_by_groups.append(group.id)
```

## Never Do
- ❌ Mutate entities after creation
- ❌ Use 1-indexed quanta (always 0-indexed)
- ❌ Hardcode time ranges (use config.time settings)
- ❌ Skip validation of JSON data
- ❌ Use Monday-first ordering (always Sunday-first)
- ❌ Access database or API (data comes from JSON only)
