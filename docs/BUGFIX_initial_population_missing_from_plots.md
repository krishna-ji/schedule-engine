# Bug Fix: Initial Population Missing from Evolution Plots

## Problem
GA console showed initial evaluation (e.g., Hard=4584, Soft=5019), but plots/CSVs started at Gen 1 (Hard=1998), omitting the pre-evolution baseline.

## Root Cause
`_track_metrics()` only called inside evolution loop (gen 0..N-1), never for initial population before evolution starts.

## Solution
1. **ga_scheduler.py**: Added `_track_metrics(gen=-1)` after initial population evaluation in `initialize_population()`
2. **ga_scheduler.py**: Updated `_track_metrics()` to skip detailed logging for gen=-1 (only record data)
3. **Plot files**: Updated xlabel to clarify "Generation (0 = Initial Population)"
   - `plothard.py`
   - `plotsoft.py`
   - `plotdiversity.py`
   - `plot_detailed_constraints.py` (6 occurrences)

## Result
- Plots now show complete evolution: Gen 0 (initial) → Gen N (final)
- CSV exports include initial population baseline
- Users can see actual GA improvement from random start
- Generation 0 = unevolved random population
- Generation 1+ = evolved generations

## Files Changed
- `src/core/ga_scheduler.py` (2 edits)
- `src/exporter/plothard.py` (2 edits)
- `src/exporter/plotsoft.py` (2 edits)
- `src/exporter/plotdiversity.py` (1 edit)
- `src/exporter/plot_detailed_constraints.py` (6 edits)
