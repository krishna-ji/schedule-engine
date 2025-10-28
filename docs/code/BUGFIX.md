# Bugfix Changelog

This file tracks bug fixes in the schedule engine codebase.

## [2025-10-28] Enhanced memory monitoring with percentage display

**Enhancement:** Added memory percentage to the monitoring display for better visibility of system memory usage. Now shows: `2.45GiB (16.0%) Peak: 2.50GiB` instead of just `2.45GiB Peak: 2.50GiB`.

**Rationale:** Percentage provides instant context for memory usage relative to total system RAM, making it easier to see if the process is approaching memory limits.

**Files Modified:**
- `src/core/ga_scheduler.py` - Added mem_percent field to memory monitoring display

## [2025-10-28] Fixed memory monitoring missing child process memory

**Issue:** Memory usage display was only showing the main Python process memory, not including child processes spawned by multiprocessing workers. This made the displayed memory much lower than actual usage and appeared "stuck" because the main process memory stays relatively constant while workers do the heavy lifting.

**Root Cause:** The monitoring used `process.memory_info().rss` which only tracks the main process (PID). When multiprocessing is enabled, worker processes' memory was completely missing from the count.

**Fix:** Enhanced memory monitoring to include all child processes recursively. This now shows the **complete memory footprint** of the schedule engine including all worker processes.

**Files Modified:**
- `src/core/ga_scheduler.py` - Modified `update_resource_monitors()` to include child process memory

## [2025-10-27] Fixed intermittent tkinter RuntimeError during export phase

**Issue:** Intermittent `RuntimeError: main thread is not in main loop` errors during plot generation when program exits. Error occurred in tkinter's `Image.__del__` and `Variable.__del__` destructors.

**Root Cause:** Matplotlib was defaulting to TkAgg backend (GUI-based) in CLI environment. When matplotlib objects were garbage collected at program exit, tkinter destructors tried to access the main event loop which didn't exist.

**Fix:** Set matplotlib backend to 'Agg' (non-interactive, file-only) before any other matplotlib imports in `src/exporter/thesis_style.py`. The Agg backend is perfect for generating PDF/PNG files without GUI requirements.

**Files Modified:**
- `src/exporter/thesis_style.py` - Added `matplotlib.use('Agg')` before imports
