# Bugfix Changelog

This file tracks bug fixes in the schedule engine codebase.

## [2025-10-27] Fixed intermittent tkinter RuntimeError during export phase

**Issue:** Intermittent `RuntimeError: main thread is not in main loop` errors during plot generation when program exits. Error occurred in tkinter's `Image.__del__` and `Variable.__del__` destructors.

**Root Cause:** Matplotlib was defaulting to TkAgg backend (GUI-based) in CLI environment. When matplotlib objects were garbage collected at program exit, tkinter destructors tried to access the main event loop which didn't exist.

**Fix:** Set matplotlib backend to 'Agg' (non-interactive, file-only) before any other matplotlib imports in `src/exporter/thesis_style.py`. The Agg backend is perfect for generating PDF/PNG files without GUI requirements.

**Files Modified:**
- `src/exporter/thesis_style.py` - Added `matplotlib.use('Agg')` before imports
