# LNS Hybrid Repair - Production Run Guide

## Quick Start

```bash
# Run on VM with maximum resources
uv run prod
```

**Expected Runtime:** 24-48 hours on 16+ cores, 32GB+ RAM

---

## What's New: LNS Hybrid Repair System

The production config now uses **hybrid repair strategy** combining fast heuristics with constraint programming:

1. **Heuristic First**: Greedy assignment + local search (5-10x faster than CP)
2. **CP Escalation**: Falls back to CP-SAT if heuristic fails
3. **Pre-Feasibility Checks**: Skips CP-SAT for obviously infeasible subproblems
4. **Conflict Graph Expansion**: Builds connected subproblems (1-hop neighborhood)

---

## Production Config Settings

```yaml
lns:
  enabled: true
  repair_strategy: hybrid         # Try heuristic → escalate to CP
  trigger_interval: 100           # Every 100 gens
  stagnation_threshold: 20        # After 20 gens stagnation
  max_subproblem_size: 30         # Repair up to 30 sessions
  min_subproblem_size: 6          # At least 6 sessions
  expand_neighborhood_hops: 1     # 1-hop conflict graph expansion
  cp_time_limit: 20.0             # 20s CP-SAT timeout
  heuristic_max_iterations: 1000  # 1000 heuristic iterations
  heuristic_time_limit: 8.0       # 8s heuristic timeout
  apply_to_best_n: 3              # Repair top 3 individuals
  enable_diagnostics: true        # Log subproblem details
  pre_check_feasibility: true     # Pre-check before CP-SAT
```

**Key Changes from Old System:**
- **Old**: CP-SAT only, often INFEASIBLE, wasted time
- **New**: Heuristic first (fast), CP escalation (best-effort), pre-checks (skip impossible)

---

## Expected Behavior During Run

### Console Output

You'll see LNS repair triggers with strategy info:

```
[!info] LNS repair triggered on gen 100 (strategy=hybrid)
   Repairing individual 1/3...
   [LNS] Detected 15 conflict sessions
   [LNS] Expanding neighborhood: 15 → 18 sessions (1 hop)
   [LNS] Pre-check: PASSED (domain_size=240)
   [LNS] Heuristic repair: SUCCESS (580 iterations, 3.2s)
   [!ok] Repaired individual (conflicts: 15 → 2)
```

### What to Watch For

**Good Signs:**
- ✅ Heuristic success messages (fast repairs)
- ✅ Low conflict counts after repair
- ✅ Occasional CP escalation when heuristic fails
- ✅ Pre-check skips for infeasible subproblems

**Warning Signs:**
- ⚠️ All repairs failing (check input data quality)
- ⚠️ CP-SAT always INFEASIBLE (subproblems too constrained)
- ⚠️ No improvement after repair (stagnation)

---

## Monitoring Performance

### During Run

Watch terminal output for:
- LNS trigger frequency (every 100 gens + stagnation)
- Repair success rates (heuristic vs CP)
- Time per repair (should be 3-8s for heuristic, 10-20s for CP)

### After Run

Check `output/evaluation_<timestamp>/lns_repair_stats.json`:

```json
{
  "total_attempts": 60,
  "successful_repairs": 52,
  "failed_repairs": 8,
  "success_rate_percent": 86.7,
  "heuristic_attempts": 45,
  "heuristic_success": 40,
  "heuristic_success_rate_percent": 88.9,
  "cp_attempts": 15,
  "cp_success": 12,
  "cp_success_rate_percent": 80.0,
  "pre_check_skips": 3,
  "avg_subproblem_size": 22.4,
  "total_repair_time_seconds": 187.3
}
```

**Ideal Stats:**
- Success rate: 80%+ overall
- Heuristic success: 85%+ (most repairs fast)
- CP success: 70%+ (when escalated)
- Pre-check skips: 5-10% (avoiding impossible)

---

## Troubleshooting

### Problem: All Repairs Failing

**Symptom**: `success_rate_percent < 20%`

**Possible Causes:**
- Input data over-constrained (check feasibility report)
- IGLS locks too many sessions (reduces LNS flexibility)
- Subproblems too small/isolated

**Solutions:**
1. Review feasibility report for bottlenecks
2. Reduce `expand_neighborhood_hops` to 0
3. Increase `max_subproblem_size` to 40
4. Check `violations.log` for systematic issues

### Problem: CP-SAT Always INFEASIBLE

**Symptom**: `cp_success = 0`, many pre-check skips

**Possible Causes:**
- Subproblems over-constrained after IGLS
- Domain sizes too small (few feasible assignments)

**Solutions:**
1. Check `enable_diagnostics: true` logs for domain sizes
2. Set `trigger_before_igls: true` (repair before IGLS locks)
3. Increase `min_subproblem_size` to 8 or 10
4. Consider disabling IGLS exhaustive search triggers

### Problem: Heuristic Rarely Succeeds

**Symptom**: `heuristic_success_rate_percent < 50%`

**Possible Causes:**
- Heuristic time limit too short
- Local search stuck in local optima

**Solutions:**
1. Increase `heuristic_time_limit` to 12.0
2. Increase `heuristic_max_iterations` to 2000
3. Check if CP escalation works (fallback strategy)

### Problem: Repairs Too Slow

**Symptom**: Each repair takes 30s+ (blocking evolution)

**Possible Causes:**
- Subproblems too large
- CP-SAT timeout too long

**Solutions:**
1. Reduce `max_subproblem_size` to 20
2. Reduce `cp_time_limit` to 15.0
3. Reduce `heuristic_time_limit` to 5.0
4. Increase `trigger_interval` to 150 (less frequent)

---

## Comparing Results

After production run, compare with baseline (no LNS):

### Final Fitness

```bash
# Check best individual fitness
grep "Best Individual" output/evaluation_*/output.txt
```

**Expect:**
- Hard violations: 0-2 (near-feasible)
- Soft penalty: 50-150 (high quality)

### LNS Contribution

Check if LNS repairs improved population:

1. Look for "Repaired individual (conflicts: X → Y)" with Y < X
2. Compare fitness before/after LNS triggers
3. Check if final best came from repaired lineage

---

## Data to Provide After Run

Please share:

1. **Console Output**: Full terminal log (`tee` to file)
2. **LNS Stats**: `output/evaluation_*/lns_repair_stats.json`
3. **Final Metrics**: `output/evaluation_*/schedule.json` (best fitness)
4. **Violation Report**: `output/evaluation_*/violations.log` (constraint breakdown)
5. **Evolution Plots**: `hard_constraint_trend.pdf`, `soft_constraint_trend.pdf`
6. **Runtime Info**: Total time, CPU/RAM usage, number of cores

**Optional but helpful:**
- `grep "LNS" output.log > lns_only.txt` (extract all LNS messages)
- Screenshot of convergence dashboard
- Any warning/error messages

---

## Quick Diagnostics Commands

```bash
# Extract LNS events from output
grep -E "\[LNS\]|\[!info\] LNS" output.txt > lns_events.txt

# Count repair outcomes
grep "Heuristic repair: SUCCESS" output.txt | wc -l
grep "CP-SAT repair: SUCCESS" output.txt | wc -l
grep "All repair strategies failed" output.txt | wc -l

# Check subproblem sizes
grep "subproblem_size=" output.txt | sed 's/.*subproblem_size=\([0-9]*\).*/\1/' | sort -n

# Find best fitness evolution
grep "Best Individual" output.txt | tail -20
```

---

## Recommended VM Specs

**Minimum:**
- CPU: 16 cores (32 threads)
- RAM: 32 GB
- Disk: 10 GB free
- OS: Linux/Windows with Python 3.11+

**Optimal:**
- CPU: 32 cores (64 threads)
- RAM: 64 GB
- Disk: 20 GB free (for logs/plots)
- SSD recommended for faster I/O

**Expected Speedup:**
- 16 cores: ~24-36 hours
- 32 cores: ~12-24 hours
- 64 cores: ~8-16 hours

---

## Post-Run Analysis Checklist

After production run completes:

- [ ] Check `lns_repair_stats.json` for success rates
- [ ] Review `violations.log` for remaining issues
- [ ] Compare hard/soft constraint trends with baseline
- [ ] Analyze which strategy (heuristic/CP) performed better
- [ ] Check if repairs improved diversity (diversity_trend.pdf)
- [ ] Verify final schedule is practical (calendar.pdf)
- [ ] Document any unexpected behaviors
- [ ] Share results for further tuning

---

## Next Steps

Based on results, we can:

1. **Tune Parameters**: Adjust time limits, subproblem sizes
2. **Strategy Switch**: If heuristic dominates, try `repair_strategy: heuristic` only
3. **Expand Hops**: If repairs too local, try `expand_neighborhood_hops: 2`
4. **Trigger Timing**: If conflicts with IGLS, try `trigger_before_igls: true`
5. **Add Diagnostics**: Enable more detailed logging for deep analysis

**Ready to run!** 🚀
