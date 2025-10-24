# Fix: Always Show Remaining Time in Progress Bar

**Date:** October 24, 2025  
**Status:** ✅ Fixed

---

## Problem

During GA evolution, the remaining time sometimes showed as `-:--:--` (blank) instead of an estimate:

```
⠙ Evolution Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╺━━━━━━━━━━ 73/100
Elapsed: 0:55:54 • Remaining: -:--:-- • 45.7s/gen
                              ^^^^^^^^ BLANK!
```

This happened when:
- Rich's built-in `TimeRemainingColumn()` didn't have enough data
- Early generations (first 1-2)
- Highly variable generation times
- Edge cases in progress calculation

**User frustration:** Can't estimate when GA will complete!

---

## Root Cause

Rich's `TimeRemainingColumn()` returns `None` for `task.time_remaining` when:
1. Not enough samples for moving average calculation
2. Insufficient elapsed time to establish reliable rate
3. Edge cases near start/end of progress

When `None` is returned, Rich displays `-:--:--` instead of a time estimate.

---

## Solution: Custom AlwaysShowTimeRemainingColumn

Created a **custom progress column** that ALWAYS shows a time estimate using fallback strategies:

### Strategy Cascade

```
1. Try Rich's built-in calculation
   └─ If available, use it (most accurate)
   
2. Fallback: Simple linear extrapolation
   └─ avg_time = elapsed / completed
   └─ remaining = avg_time × (total - completed)
   
3. Fallback: Rough estimate (first generation)
   └─ Assume ~1 min/generation
   └─ Show as "~H:MM:SS" (dimmed)
   
4. Last resort: Show "~calculating~"
   └─ Should almost never happen
```

### Implementation

**Custom Column Class:**
```python
class AlwaysShowTimeRemainingColumn(ProgressColumn):
    """Never shows blank, always provides estimate."""
    
    def render(self, task: Task) -> Text:
        # Try Rich's calculation
        if task.time_remaining is not None:
            return format_time(task.time_remaining)
        
        # Fallback: Linear extrapolation
        if task.completed > 0 and task.elapsed > 0:
            avg = task.elapsed / task.completed
            remaining = avg * (task.total - task.completed)
            return format_time(remaining, prefix="~")
        
        # Very rough estimate for first gen
        if task.total > 0:
            estimated = task.total * 60  # 1 min/gen
            return format_time(estimated, prefix="~", dim=True)
        
        # Absolute fallback
        return Text("~calculating~", style="dim")
```

**Display Indicators:**
- **No prefix:** Rich's accurate calculation
- **`~` prefix:** Extrapolated estimate (less accurate)
- **Dimmed:** Very rough estimate (first generation)

---

## Results

### Before
```
Gen 1:  Remaining: -:--:--  ❌ (blank)
Gen 5:  Remaining: -:--:--  ❌ (still blank!)
Gen 10: Remaining: 1:23:45  ✓ (finally shows)
```

### After
```
Gen 1:  Remaining: ~1:40:00  ✅ (rough estimate, dimmed)
Gen 5:  Remaining: ~1:25:30  ✅ (extrapolated)
Gen 10: Remaining: 1:23:45   ✅ (accurate)
```

**Impact:**
- ✅ Users can ALWAYS see progress estimate
- ✅ Better UX during long GA runs
- ✅ No more frustrating blank times
- ✅ Rough estimates better than nothing

---

## Testing

**Test File:** `test/test_always_show_remaining_time.py`

**Verified:**
- ✅ Shows estimate from first iteration
- ✅ Updates smoothly as more data available
- ✅ Shows accurate time when Rich has data
- ✅ Handles edge cases (near completion, just started)
- ✅ Never shows `-:--:--` blank

---

## Files Modified

| File | Changes |
|------|---------|
| `src/core/ga_scheduler.py` | Added `AlwaysShowTimeRemainingColumn` class |
| `src/core/ga_scheduler.py` | Replaced `TimeRemainingColumn()` with custom column |
| `src/core/ga_scheduler.py` | Added imports: `ProgressColumn`, `Task`, `Text` |
| `test/test_always_show_remaining_time.py` | Comprehensive tests |
| `docs/FIX_ALWAYS_SHOW_REMAINING_TIME.md` | This documentation |

---

## Technical Details

### Time Formatting
```python
# Format: H:MM:SS or M:SS
hours, remainder = divmod(seconds, 3600)
minutes, seconds = divmod(remainder, 60)

if hours > 0:
    return f"{hours}:{minutes:02d}:{seconds:02d}"
else:
    return f"{minutes}:{seconds:02d}"
```

### Extrapolation Formula
```python
avg_time_per_gen = total_elapsed / completed_gens
remaining_gens = total_gens - completed_gens
estimated_remaining = avg_time_per_gen × remaining_gens
```

**Example:**
- 10 generations completed in 500 seconds
- Average: 50 sec/gen
- 90 gens remaining
- Estimate: 90 × 50 = 4500 sec = 1:15:00

---

## Edge Cases Handled

1. **First generation (no history)**
   - Shows rough estimate: `~1:40:00` (100 gens × 60 sec)
   - Dimmed style to indicate uncertainty

2. **Variable generation times**
   - Uses average of all completed generations
   - More stable than single-generation estimate

3. **Near completion**
   - Switches to Rich's accurate calculation
   - Shows precise remaining time

4. **Task completed**
   - Shows `0:00:00` immediately

---

## Benefits

**User Experience:**
- ✅ Always know approximate completion time
- ✅ No confusing blank displays
- ✅ Better planning (can estimate when to check back)
- ✅ Smoother visual experience

**Technical:**
- ✅ Minimal overhead (simple arithmetic)
- ✅ Compatible with existing Rich infrastructure
- ✅ Falls back gracefully through strategies
- ✅ No breaking changes to other code

---

## Future Enhancements

1. **ETA smoothing** - Use exponential moving average for more stable estimates
2. **Confidence indicator** - Show ± range when extrapolating
3. **Adaptive update rate** - Update faster when estimate changes rapidly
4. **Historical data** - Use previous runs to improve first-generation estimates

---

## Summary

✅ **Fixed blank remaining time display**  
✅ **Always shows estimate using fallback strategies**  
✅ **Better UX during long GA runs**  
✅ **Tested and working**  
✅ **No breaking changes**

**No more `-:--:--` blanks during evolution!** 🎯
