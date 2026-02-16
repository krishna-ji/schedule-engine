# Schedule Export Fix - Summary

## Problem
The GA experiment runs were **not generating** `schedule.json` and `calendar.pdf` files in the output directories. Only basic fitness information was being saved.

## Root Cause
The `BaseExporter._export_schedule()` method in `/src/schedule_engine/experiments/output/base.py` was only saving basic fitness information (number of genes, fitness values) but **not decoding the individual** into a full schedule and calling the export functions.

## Solution Implemented

### File Modified
`/home/krishna/Desktop/schedule-engine/src/schedule_engine/experiments/output/base.py`

### Changes Made
Updated the `_export_schedule()` method to:

1. **Decode the individual** using `decode_individual()` to convert genes into `CourseSession` objects
2. **Call `export_everything()`** which generates:
   - `schedule.json` - Full schedule with course names, instructors, rooms, and times
   - `calendar.pdf` - Visual calendar/timetable with color-coded sessions

### Code Changes
```python
def _export_schedule(self, best_individual: list) -> None:
    """Save best individual's fitness and decode schedule to JSON + PDF."""
    try:
        # 1. Save basic fitness info
        path = self.output_dir / "best_individual.json"
        info: dict[str, Any] = {
            "num_genes": len(best_individual),
        }
        if hasattr(best_individual, "fitness") and best_individual.fitness.valid:
            info["fitness"] = list(best_individual.fitness.values)
        self._write_json(path, info)
        self.logger.debug("Saved best individual info → %s", path)

        # 2. Decode and export full schedule (schedule.json + calendar.pdf)
        from schedule_engine.io.decoder import decode_individual
        from schedule_engine.io.export import export_everything

        # Decode individual to CourseSession list
        decoded_schedule = decode_individual(
            best_individual,
            self.data.courses,
            self.data.instructors,
            self.data.groups,
            self.data.rooms,
        )

        # Build course lookup for human-readable names
        course_lookup = self.data.courses

        # Export schedule.json and calendar.pdf
        export_everything(
            schedule=decoded_schedule,
            output_path=str(self.output_dir),
            qts=self.data.qts,  # ← Fixed: was quantum_time_system, correct is qts
            course_lookup=course_lookup,
            parallel=True,
        )
        self.logger.info("Exported schedule.json and calendar.pdf")
    except Exception as e:
        self.logger.warning("Could not export schedule: %s", e)
```

## Files Now Generated

After running any GA experiment (ga_01, ga_02, ga_03, ga_04, etc.), the output directory now contains:

### ✅ New Files
- **`schedule.json`** (~297KB) - Complete schedule with:
  - Course names, codes, and types
  - Instructor IDs and names
  - Group IDs (including multi-group sessions)
  - Room assignments
  - Time schedules (day/time slots)

- **`calendar.pdf`** (~198KB) - Visual timetable:
  - Color-coded by course type (blue=theory, red=practical)
  - One page per student group
  - Days across columns, hours down rows
  - Consecutive sessions automatically merged

### Existing Files (unchanged)
- `best_individual.json` - Fitness and gene count
- `metadata.json` - Experiment configuration
- `stats.json` - Per-generation statistics
- `plots/` - Diagnostic plots
- `csv/` - CSV data files

## Testing

### Test Scripts Created
1. `/home/krishna/Desktop/schedule-engine/runs/test_pdf_export.py` - Quick 10-gen test
2. `/home/krishna/Desktop/schedule-engine/runs/test_baseline_pdf.py` - Baseline test

### Verified Working
✅ `ga_01_baseline` - Output: `output/ga_01_baseline/20260212_123621/`
✅ `ga_04_repair_bandit` - Output: `output/ga_04_repair_bandit/20260212_123439/`

### Sample Output
```
 PDF saved as 'output/ga_01_baseline/20260212_123621/calendar.pdf'
[OK-KRISHNA] Schedule exported successfully!
[...]JSON: output/ga_01_baseline/20260212_123621/schedule.json
[...]PDF:  output/ga_01_baseline/20260212_123621/calendar.pdf
Exported schedule.json and calendar.pdf
```

## Impact

### ✅ What Now Works
- **All GA experiments** (baseline, memetic, repair, bandit, Q-learning) automatically export full schedules
- **No code changes needed** in individual run scripts - the fix is in the base exporter
- **Backward compatible** - old experiments can be re-run to generate missing files

### 📁 Applies To These Run Scripts
- `runs/ga_01_baseline.py`
- `runs/ga_02_memetic.py`
- `runs/ga_03_repair_sequential.py`
- `runs/ga_04_repair_bandit.py`
- `runs/ga_05_repair_qlearning.py`
- Any future GA-based experiments using `BaseExporter`

## Next Steps

### Re-running Old Experiments (Optional)
To generate schedule files for previous runs, simply re-run the experiments:
```bash
python runs/ga_01_baseline.py
python runs/ga_02_memetic.py
python runs/ga_04_repair_bandit.py
# etc.
```

### Viewing the Calendar
Open the generated PDF files:
```bash
xdg-open output/ga_01_baseline/20260212_123621/calendar.pdf
```

## Technical Details

### Dependencies Used
- `decode_individual()` - Converts gene list to CourseSession objects
- `export_everything()` - Orchestrates JSON and PDF generation
- `QuantumTimeSystem` (qts) - Converts quanta to human-readable times
- `matplotlib` + `PdfPages` - PDF calendar rendering

### Error Handling
- Graceful failure with warning message if export fails
- Basic fitness info still saved even if full export fails
- Exception logged for debugging

---

**Status:** ✅ FIXED - All experiments now generate complete schedule outputs
**Date:** February 12, 2026
**Files Modified:** 1 file (`src/schedule_engine/experiments/output/base.py`)
