---
applyTo: "src/exporter/**/*.py"
---

# Export & Reporting Instructions

## Overview
Generate outputs: JSON schedules, PDF calendars, evolution plots, violation reports. Orchestrated by `src/workflows/reporting.py`, individual exporters in `src/exporter/`.

## Components

### Schedule Exports
- `exporter.py` - Main export function `export_everything()`
- JSON: Machine-readable schedule (`schedule.json`)
- PDF: Visual calendar (`ScheduleCalendar.pdf`)
- TXT: Violation details (`violation_report.txt`)

### Plot Modules
- `plothard.py` - Hard constraint trend over generations
- `plotsoft.py` - Soft constraint trend over generations
- `plotdiversity.py` - Population diversity metrics
- `plotpareto.py` - Pareto front visualization
- `plot_detailed_constraints.py` - Individual constraint breakdowns

### Reporting Workflow (`src/workflows/reporting.py`)
```python
def generate_reports(decoded_schedule, metrics, output_dir, context):
    # 1. Export schedule (JSON + PDF)
    # 2. Generate violation report
    # 3. Plot evolution trends
    # 4. Plot Pareto front
    # 5. Plot detailed constraints
```

## Rules

### File Organization
```
output/evaluation_<timestamp>/
├── schedule.json           # Machine-readable schedule
├── ScheduleCalendar.pdf    # Visual calendar
├── logger.txt              # GA execution log
├── feasibility_report.txt  # Pre-run feasibility analysis
├── violation_report.txt    # Constraint violation details
└── plots/
    ├── hard_constraint_trend.pdf
    ├── soft_constraint_trend.pdf
    ├── diversity_trend.pdf
    ├── pareto_front.pdf
    └── constraints/
        ├── hard/individual_constraints.pdf
        ├── soft/individual_constraints.pdf
        └── constraint_summary.pdf
```

### PDF Calendar Styling
- Use thesis-quality formatting (from `thesis_style.py`)
- Sunday-first week ordering
- Color code by course (see `config/color_palette.py`)
- Show instructor, room, group info (configurable)
- Settings in `config/calendar_config.py`

### Plot Standards
- Use `thesis_style.apply_thesis_style()` for consistent formatting
- DPI=300 for publication quality
- Include axis labels and legends
- Save as PDF (vector format preferred)
- Use colorblind-friendly palettes

### JSON Schema
```json
{
  "metadata": {
    "timestamp": "2025-10-27T10:00:00",
    "hard_violations": 0,
    "soft_penalty": 123.45
  },
  "sessions": [
    {
      "course": {"code": "CS101", "name": "..."},
      "groups": ["BCE1", "BCE2"],
      "instructor": {"id": "INS001", "name": "..."},
      "room": {"id": "R101", "capacity": 50},
      "time": {
        "day": "Monday",
        "start": "09:00",
        "end": "10:00",
        "quanta": [10, 11]
      }
    }
  ]
}
```

## Exporting Functions

### Schedule JSON
```python
def export_schedule_json(decoded_schedule, output_path):
    schedule_data = {
        "metadata": {...},
        "sessions": [session.to_dict() for session in decoded_schedule]
    }
    with open(output_path, 'w') as f:
        json.dump(schedule_data, f, indent=2)
```

### Calendar PDF
```python
from src.exporter.exporter import generate_calendar_pdf

generate_calendar_pdf(
    decoded_schedule,
    output_path="output/calendar.pdf",
    config=get_config().calendar
)
```

### Violation Report
```python
from src.exporter.violation_reporter import generate_violation_report

generate_violation_report(
    decoded_schedule,
    context,
    output_path="output/violations.txt"
)
```

### Evolution Plots
```python
from src.exporter.plothard import plot_hard_constraints
from src.exporter.plotsoft import plot_soft_constraints

plot_hard_constraints(metrics.hard_violations_per_gen, output_dir)
plot_soft_constraints(metrics.soft_penalties_per_gen, output_dir)
```

## Adding New Exports

### Step 1: Create Export Function
```python
# In src/exporter/my_export.py
def export_my_format(decoded_schedule, output_path, context):
    """Export schedule in custom format."""
    # Implementation
    with open(output_path, 'w') as f:
        # Write output
    console.print(f"[!ok] {output_path}")
```

### Step 2: Register in Reporting Workflow
```python
# In src/workflows/reporting.py
def generate_reports(...):
    # ... existing exports ...
    
    # Add new export
    console.print("  [+] Generating my format...")
    export_my_format(decoded_schedule, output_dir, context)
    console.print("      [!ok] my_format.txt")
```

### Step 3: Update Documentation
Add to `docs/code/ENHANCE.md`:
```markdown
## [2025-10-27] Added custom format exporter
- `src/exporter/my_export.py`
- `src/workflows/reporting.py`
```

## Plot Customization

### Theme Settings (thesis_style.py)
```python
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})
```

### Color Palettes
```python
# Use from config
from config.color_palette import get_course_color

color = get_course_color(course_code)  # Consistent colors across plots
```

### Multi-Panel Plots
```python
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].plot(...)  # Top-left
axes[0, 1].plot(...)  # Top-right
# ...
plt.tight_layout()
plt.savefig(output_path)
```

## Performance Considerations
- Cache decoded schedule (don't re-decode for each export)
- Generate plots in parallel if possible
- Use vector formats (PDF) to reduce file size
- Limit data points in plots (e.g., every Nth generation for long runs)

## Never Do
-  Overwrite existing output directories (always create timestamped)
-  Use raster formats (PNG/JPG) for plots unless specified
-  Hardcode file paths (use output_dir parameter)
-  Skip error handling (log and continue if one export fails)
-  Use Monday-first ordering in calendars (always Sunday-first)
-  Forget to close file handles after writing
