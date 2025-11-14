# Quick Reference: New Output File Structure

## Standard Output Directory Layout

```
output/evaluation_<timestamp>/
│
├── 📄 run.log                      # Execution summary & configuration
├── 📄 feasibility.log              # Pre-GA feasibility analysis
├── 📄 violations.log               # Final schedule constraint violations
│
├── 📊 schedule.json                # Final schedule (structured data)
├── 📈 calendar.pdf                 # Visual calendar (color-coded)
├── 📊 violation_heatmap.json       # Per-gene violation tracking
│
├── 📁 data/                        # CSV data files
│   └── 📊 metrics.csv              # Master metrics file
│                                     - All generation data
│                                     - Hard/soft constraints
│                                     - Diversity, hypervolume, spacing
│                                     - Repair statistics
│                                     - Events
│
└── 📁 plots/                       # All visualizations (PDFs)
    ├── hard_trend.pdf
    ├── soft_trend.pdf
    ├── diversity.pdf
    ├── pareto_front.pdf
    ├── hypervolume_trend.pdf
    ├── spacing_trend.pdf
    ├── convergence_dashboard.pdf
    │
    └── 📁 constraints/
        ├── 📁 hard/
        │   ├── instructor_availability_trend.pdf
        │   ├── no_overlap_trend.pdf
        │   ├── ... (one per constraint)
        │   └── hard_summary.pdf
        │
        └── 📁 soft/
            ├── group_gaps_penalty_trend.pdf
            ├── instructor_workload_trend.pdf
            ├── ... (one per constraint)
            └── soft_summary.pdf
```

## File Descriptions

### Log Files (`.log`)

| File | Content | When Created |
|------|---------|--------------|
| `run.log` | High-level execution summary with config, per-generation best fitness, runtime stats | Every run |
| `feasibility.log` | Pre-GA feasibility analysis with bottleneck detection | If `feasibility.generate_report=true` |
| `violations.log` | Detailed constraint violation breakdown for final schedule | If `course_map` provided |

### Data Files

| File | Format | Content |
|------|--------|---------|
| `schedule.json` | JSON | Final schedule with sessions, times, rooms, instructors |
| `violation_heatmap.json` | JSON | Per-gene violation tracking across generations |
| `data/metrics.csv` | CSV | Complete generation-by-generation metrics |

### Visual Files

| File | Content |
|------|---------|
| `calendar.pdf` | Multi-page visual calendar (one page per group) |
| `plots/*.pdf` | Evolution visualizations, Pareto fronts, convergence plots |
| `plots/constraints/**/*.pdf` | Individual constraint trend plots |

## `data/metrics.csv` Columns

### Core Metrics
- `generation` - Generation number (INIT for initial population)
- `hard_total` - Sum of all hard constraint violations
- `soft_total` - Sum of all soft constraint penalties
- `diversity` - Population diversity (0-1)
- `time_seconds` - Time taken for generation

### Hard Constraints (individual)
- `hard_no_group_overlap`
- `hard_no_instructor_conflict`
- `hard_instructor_not_qualified`
- `hard_room_type_mismatch`
- `hard_availability_violations`
- `hard_incomplete_or_extra_sessions`
- `hard_session_block_clustering_penalty`

### Soft Constraints (individual)
- `soft_group_gaps_penalty`
- `soft_instructor_gaps_penalty`
- `soft_group_midday_break_violation`

### Advanced Metrics
- `hypervolume` - NSGA-II quality indicator
- `spacing` - Solution distribution uniformity
- `igd` - Inverted Generational Distance
- `spread` - Solution spread metric

### Repair Statistics
- `repairs_total` - Total repairs performed
- `repairs_individuals_count` - Number of individuals repaired
- `repairs_crossover_count` - Repairs after crossover
- `repairs_mutation_count` - Repairs after mutation
- `repairs_memetic_count` - Repairs from memetic search
- `repairs_instructor_availability` - Availability fixes
- `repairs_overlap` - Overlap fixes
- `repairs_room` - Room fixes
- `repairs_instructor_conflict` - Instructor conflict fixes
- `repairs_qualification` - Qualification fixes
- `repairs_room_type` - Room type fixes
- `repairs_clustering` - Clustering fixes
- `repairs_session_count` - Session count fixes

### Events & Notes
- `events` - Semicolon-separated list of events (repair, stagnation, hypermutation, etc.)
- `notes` - Optional notes (e.g., "Perfect solution found")

## Usage Examples

### Python Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load master metrics
df = pd.read_csv("output/evaluation_20251114_120000/data/metrics.csv")

# Plot hard violations over time
plt.plot(df['generation'], df['hard_total'])
plt.xlabel('Generation')
plt.ylabel('Hard Violations')
plt.title('Hard Constraint Violations Over Time')
plt.show()

# Analyze repair effectiveness
print(f"Total repairs: {df['repairs_total'].sum()}")
print(f"Memetic repairs: {df['repairs_memetic_count'].sum()}")

# Find perfect solutions
perfect = df[df['hard_total'] == 0]
if not perfect.empty:
    print(f"First perfect solution at generation: {perfect.iloc[0]['generation']}")

# Compare metrics
correlation = df[['hard_total', 'soft_total', 'diversity']].corr()
print(correlation)
```

### Excel/LibreOffice

1. Open `data/metrics.csv`
2. Create pivot tables for analysis
3. Use built-in charts for visualization
4. Filter by generation for specific analysis

### Command Line (CSV analysis)

```bash
# Count total rows (generations + 1 header)
wc -l output/evaluation_*/data/metrics.csv

# Extract specific columns
cut -d',' -f1,2,3 output/evaluation_*/data/metrics.csv > summary.csv

# Find best generation
csvcut -c generation,hard_total,soft_total output/evaluation_*/data/metrics.csv | csvlook
```

## File Size Estimates

Typical file sizes for a 500-generation run with 8 hard + 3 soft constraints:

| File/Directory | Size | Notes |
|----------------|------|-------|
| `run.log` | ~10 KB | Text summary |
| `feasibility.log` | ~5-20 KB | If enabled |
| `violations.log` | ~5-50 KB | Depends on violations |
| `schedule.json` | ~50-500 KB | Depends on schedule size |
| `data/metrics.csv` | ~200-500 KB | Main metrics file |
| `plots/` | ~5-10 MB | All PDF visualizations |
| **Total** | **~10-20 MB** | Complete output |

**Comparison to old structure:** ~30-40% smaller (eliminated CSV redundancy)

## Quick Checks

### Verify Run Completed Successfully

```bash
# Check if run.log exists and contains summary
grep -i "complete" output/evaluation_*/run.log

# Check final hard violations
tail -20 output/evaluation_*/run.log | grep "Final Hard"

# Check if feasible solution found
grep -i "feasible solution" output/evaluation_*/run.log
```

### Validate Data Integrity

```python
import pandas as pd

df = pd.read_csv("output/evaluation_*/data/metrics.csv")

# Check completeness
expected_rows = 501  # 500 gens + 1 INIT
actual_rows = len(df)
print(f"Expected: {expected_rows}, Actual: {actual_rows}")

# Check for NaN values
print(df.isnull().sum())

# Verify generation sequence
assert df['generation'].iloc[0] == 'INIT' or df['generation'].iloc[0] == -1
assert all(df['generation'][1:] == range(len(df)-1))
```

---

**Last Updated:** November 14, 2025  
**Version:** 1.0.0
