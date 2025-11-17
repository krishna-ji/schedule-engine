# Constraint Logger - Quick Reference

## Overview
The **Constraint Logger** creates a detailed CSV file (`logger_constraints.csv`) with per-generation constraint breakdowns, events, and metrics. This provides much more granular analysis than `logger.txt`.

## Output Location
```
output/evaluation_<timestamp>/logger_constraints.csv
```

## CSV Structure

### Columns (Left to Right)

1. **Generation Info**
   - `generation`: -1 (INIT) or 0+ for evolved generations

2. **Totals**
   - `hard_total`: Sum of all hard constraint violations
   - `soft_total`: Sum of all soft constraint penalties

3. **Hard Constraints (Individual)**
   - `hard_no_group_overlap`: Group scheduling conflicts
   - `hard_no_instructor_conflict`: Instructor double-booking
   - `hard_instructor_not_qualified`: Unqualified instructor assignments
   - `hard_room_type_mismatch`: Wrong room type assignments
   - `hard_availability_violations`: Scheduling during unavailable times
   - `hard_incomplete_or_extra_sessions`: Incorrect session counts

4. **Soft Constraints (Individual)**
   - `soft_group_gaps_penalty`: Gaps in group schedules
   - `soft_instructor_gaps_penalty`: Gaps in instructor schedules
   - `soft_group_midday_break_violation`: Missing lunch breaks
   - `soft_session_block_clustering_penalty`: Non-ideal session block sizes

5. **Metrics**
   - `diversity`: Population diversity (0-1, higher = more diverse)
   - `time_seconds`: Time taken for this generation

6. **Repair Statistics**
   - `repairs_total`: Total repairs performed
   - `repairs_instructor_availability`: Availability fixes
   - `repairs_overlap`: Overlap/conflict fixes
   - `repairs_room`: Room assignment fixes
   - `repairs_instructor_conflict`: Instructor conflict fixes
   - `repairs_qualification`: Qualification fixes
   - `repairs_room_type`: Room type fixes
   - `repairs_clustering`: Clustering fixes
   - `repairs_session_count`: Session count fixes

7. **Events**
   - `events`: Semicolon-separated list of events (see below)
   - `notes`: Additional notes (e.g., "Perfect solution")

## Events Tracked

| Event | Description |
|-------|-------------|
| `stagnation_detected` | No improvement for N generations |
| `stagnation_repair` | Repair triggered by stagnation |
| `periodic_repair` | Regular periodic repair |
| `intensive_repair` | Intensive repair (longer interval) |
| `hypermutation_activated` | Hypermutation started |
| `hypermutation_active` | Hypermutation ongoing |
| `hypermutation_ended` | Hypermutation finished |
| `perfect_solution` | Zero hard violations achieved |

## Usage Examples

### Excel/Google Sheets
1. Open the CSV file
2. Use filters to analyze specific constraints
3. Create pivot tables for trend analysis
4. Chart constraint evolution over generations

### Python/Pandas
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('output/evaluation_20251027_010830/logger_constraints.csv')

# Plot hard constraint evolution
plt.figure(figsize=(12, 6))
for col in df.columns:
    if col.startswith('hard_') and col != 'hard_total':
        plt.plot(df['generation'], df[col], label=col.replace('hard_', ''))
plt.xlabel('Generation')
plt.ylabel('Violations')
plt.title('Hard Constraint Evolution')
plt.legend()
plt.grid(True)
plt.show()

# Find when repairs were triggered
repair_gens = df[df['repairs_total'] > 0]
print(f"Repairs triggered in {len(repair_gens)} generations")
print(repair_gens[['generation', 'repairs_total', 'events']])

# Analyze event correlation
stagnation_gens = df[df['events'].str.contains('stagnation', na=False)]
print(f"\nStagnation detected in generations: {stagnation_gens['generation'].tolist()}")
```

### Identify Problematic Constraints
```python
# Find constraints that aren't improving
initial = df.iloc[0]
final = df.iloc[-1]

for col in df.columns:
    if col.startswith('hard_'):
        improvement = initial[col] - final[col]
        if improvement <= 0:
            print(f"{col}: No improvement (still {final[col]:.0f})")
```

## Crash Safety
- File is flushed to disk after **every generation**
- No data loss if program crashes mid-run
- Timing updates are best-effort (may be 0.0 if crash occurs before update)

## Differences from logger.txt

| Feature | logger_constraints.csv | logger.txt |
|---------|----------------------|------------|
| Format | CSV (Excel-ready) | Plain text |
| Constraint breakdown | ✓ Individual values | ✗ Only totals |
| Event tracking | ✓ Detailed events | ✗ Manual notes only |
| Repair breakdown | ✓ Per-heuristic | ✓ Only total |
| Crash-safe | ✓ Per-generation flush | ✓ Per-generation flush |
| Analysis tools | Excel, Python, R | Text editors |
| Best for | Data analysis, research | Quick viewing, summaries |

## Example Analysis Scenarios

### 1. Find Which Constraint is Blocking Feasibility
```python
# Look at final generation
final = df.iloc[-1]
hard_constraints = {col: final[col] for col in df.columns if col.startswith('hard_')}
print("Final hard constraint violations:")
for name, value in sorted(hard_constraints.items(), key=lambda x: x[1], reverse=True):
    if value > 0:
        print(f"  {name}: {value:.0f}")
```

### 2. Measure Repair Effectiveness
```python
# Compare before/after repair triggers
repair_gens = df[df['events'].str.contains('repair', na=False)].index
for idx in repair_gens:
    if idx + 1 < len(df):
        before = df.loc[idx, 'hard_total']
        after = df.loc[idx + 1, 'hard_total']
        improvement = before - after
        print(f"Gen {idx} repair: {before:.0f} → {after:.0f} (Δ{improvement:.0f})")
```

### 3. Track Diversity vs Quality Trade-off
```python
plt.figure(figsize=(10, 6))
plt.scatter(df['diversity'], df['hard_total'], c=df.index, cmap='viridis')
plt.xlabel('Diversity')
plt.ylabel('Hard Violations')
plt.title('Diversity vs Quality Trade-off')
plt.colorbar(label='Generation')
plt.show()
```

## Tips
1. **Open in Excel**: Double-click to open, use filters on column headers
2. **Conditional Formatting**: Highlight cells > threshold (e.g., violations > 100)
3. **Pivot Tables**: Group by events to see impact of repairs/hypermutation
4. **Time Series**: Plot any column vs generation for trend analysis
5. **Compare Runs**: Load multiple CSVs and plot side-by-side for A/B testing

## Troubleshooting

**Q: Why are some time_seconds values 0.000?**  
A: Timing is updated after generation completes. If crash occurred before update, time defaults to 0.

**Q: Why are events/notes empty (NaN)?**  
A: No events occurred in that generation (normal for most generations).

**Q: Why is INIT logged twice?**  
A: First entry logs initial data, second updates timing after initial evaluation.

**Q: Can I disable constraint logging?**  
A: Not currently configurable, but it has minimal overhead (single CSV write per generation).

---

**Created:** 2025-10-27  
**Related Files:**
- `src/utils/constraint_logger.py` - Logger implementation
- `docs/code/ENHANCE.md` - Feature documentation
- `logger.txt` - Companion text log
