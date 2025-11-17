# LNS Force Trigger Validation Guide

## Quick Test (5 minutes)

Verify LNS works before running 12-hour production:

```powershell
# Run forced trigger test
python main.py --config configs/test_lns_force.yaml --experiment lns-validation

# Or with UV
uv run python main.py --config configs/test_lns_force.yaml --experiment lns-validation
```

## What Happens

Generation 6: LNS repair FORCED to trigger with full diagnostics:

```
[!info] LNS repair triggered on gen 6 (strategy=hybrid)
   Repairing individual 1/1...
   [LNS] Detected 12 conflicted sessions with 18 violations
   [LNS] Violation breakdown: {'instructor_exclusivity': 5, 'room_exclusivity': 8, ...}
   [LNS] Expanded neighborhood: 12 → 15 sessions (1 hop)
   [LNS] Subproblem: partial=85, repair=15
   [LNS] Domain sizes: instructors=8, rooms=12, times=240
   [LNS] Pre-check: PASSED ✓
   [LNS-Hybrid] Step 1: Attempting heuristic repair (max_iter=500, timeout=5.0s)...
   [LNS-Hybrid] Heuristic repair: SUCCESS ✓ (3.2s, no escalation needed)
   [LNS] ✓ Repair SUCCESSFUL: 15 sessions repaired (strategy=hybrid, total_time=3.4s)
```

## Success Indicators

✅ **LNS triggered on gen 6** (forced)
✅ **Conflicts detected** (>0 sessions)
✅ **Pre-check passed** or gave clear reason
✅ **Heuristic attempted** with iteration count/time
✅ **CP escalation** if heuristic failed (hybrid mode)
✅ **Repair outcome** (SUCCESS/FAILED) with timing

## Failure Scenarios & Fixes

### No conflicts detected
**Symptom**: `[LNS] No conflicts detected, skipping repair`

**Fix**: Population too clean, increase randomness:
```yaml
ga:
  population_strategy: random # Force random init
```

### Pre-check always fails
**Symptom**: `[LNS] Pre-check: FAILED ✗ - domain_instructor=0`

**Fix**: Temporarily disable pre-check for test:
```yaml
lns:
  pre_check_feasibility: false
```

### Both strategies fail
**Symptom**: `[LNS] ✗ Repair FAILED: All strategies exhausted`

**Diagnosis**: Check console for:
- IGLS timeout (increase `igls_time_limit: 8.0`)
- CP-SAT INFEASIBLE (check domain sizes in diagnostics)
- Subproblem too constrained (reduce `min_subproblem_size: 1`)

**Action**: Share full console output for analysis

## Validation Checklist

After test run completes (2-5 minutes):

- [ ] Console shows "LNS: FORCED trigger on gen 6"
- [ ] Conflict detection logged with violation breakdown
- [ ] Neighborhood expansion shown (if expand_hops > 0)
- [ ] Pre-check result displayed (PASSED/FAILED + reason)
- [ ] Heuristic attempt shown with timing
- [ ] CP escalation triggered (if hybrid + heuristic failed)
- [ ] Final outcome clear (SUCCESS ✓ or FAILED ✗)
- [ ] Check `output/evaluation_*/lns_repair_stats.json`:
  ```json
  {
    "total_attempts": 1,
    "successful_repairs": 1,  // Should be 1 if worked
    "heuristic_attempts": 1,
    "cp_attempts": 0 or 1     // Depends on strategy
  }
  ```

## Production Config

Once validated, update prod config for VM run:

```yaml
lns:
  enabled: true
  force_trigger_generations: [6, 100, 200, 500] # Validate early + periodic
  trigger_interval: 100
  stagnation_threshold: 20
  # ... rest stays same
```

**Rationale**: 
- Gen 6: Early validation (confirms LNS works)
- Gen 100+: Normal interval triggers
- Stagnation: Backup trigger if stuck

## Remote VM Usage

```powershell
# Run full production with early validation
python main.py --env prod --experiment prod-hybrid-validated-r01

# Monitor for forced trigger
# (watch console around gen 6)
```

Expected log snippet:
```
Generation 6/1000
[!info] LNS: FORCED trigger on gen 6 (validation/testing mode)
[!info] LNS repair triggered on gen 6 (strategy=hybrid)
   Repairing individual 1/3...
   [LNS] Detected 8 conflicted sessions with 12 violations
   ...
   [LNS] ✓ Repair SUCCESSFUL: 8 sessions repaired (strategy=hybrid, total_time=4.1s)
```

If gen 6 trigger succeeds → LNS system works, continue run with confidence.
If gen 6 trigger fails → Stop, share logs, debug before wasting 12 hours.

## Quick Commands

```powershell
# Run validation test
python main.py --config configs/test_lns_force.yaml

# Extract LNS logs only
Select-String -Path "output/evaluation_*/run.log" -Pattern "\[LNS\]" | Out-File lns_debug.txt

# Check stats
Get-Content output/evaluation_*/lns_repair_stats.json | ConvertFrom-Json | Format-List

# Verify trigger fired
Select-String -Path "output/evaluation_*/run.log" -Pattern "FORCED trigger"
```

## Troubleshooting

**Q: Force trigger not firing?**
A: Check `lns.enabled: true` and `force_trigger_generations: [6]` in loaded config. Run with `--config configs/test_lns_force.yaml` explicitly.

**Q: Diagnostics not showing?**
A: Verify `lns.enable_diagnostics: true`. Rich console output only shows if enabled.

**Q: Want to force CP-SAT only?**
A: Change `repair_strategy: cp` and `pre_check_feasibility: false` for pure CP test.

**Q: Need multiple test points?**
A: Set `force_trigger_generations: [6, 10, 15]` to test 3 times in 20-gen run.

---

**Ready to validate!** Run the test config, confirm console output looks good, then proceed to production with confidence. 🚀
