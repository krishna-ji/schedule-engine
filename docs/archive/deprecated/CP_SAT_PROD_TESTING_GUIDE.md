# CP-SAT Production Testing Guide

## Overview

Three production configurations to test CP-SAT effectiveness:

1. **`prod_cp_only.yaml`** - Pure CP-SAT strategy (no heuristic fallback)
2. **`prod_no_local_search.yaml`** - CP-SAT + GA only (IGLS disabled)
3. **`prod.yaml`** - Hybrid strategy (heuristic → CP escalation) with IGLS

---

## Configuration 1: Pure CP-SAT Strategy

**File**: `configs/prod_cp_only.yaml`

**What it tests**: Can CP-SAT alone handle LNS repairs effectively?

**Key Settings**:
- `lns.repair_strategy: cp` (pure CP-SAT, no heuristic)
- `lns.cp_time_limit: 30.0` (30s per subproblem)
- `lns.expand_neighborhood_hops: 2` (larger connected subproblems)
- `lns.trigger_interval: 100` (every 100 gens)
- `lns.apply_to_best_n: 3` (repair top 3 individuals)
- **CP-SAT multiprocessing**: Enabled (auto-detect cores)
- **IGLS**: Enabled (standard triggers)

**Run Command**:
```powershell
uv run python main.py --config configs/prod_cp_only.yaml --experiment prod-cp-only-r01
```

**Expected Runtime**: 24-48 hours on 16+ cores

**Watch For**:
- Gen 6: Early CP-SAT validation trigger
- CP success rate (check `lns_repair_stats.json`)
- Pre-check skip rate (infeasible subproblems avoided)
- Comparison with hybrid strategy

---

## Configuration 2: No Local Search (GA + CP-SAT Only)

**File**: `configs/prod_no_local_search.yaml`

**What it tests**: Can CP-SAT replace IGLS local search entirely?

**Key Settings**:
- `lns.repair_strategy: cp` (pure CP-SAT)
- `lns.trigger_interval: 50` (more frequent, every 50 gens)
- `lns.stagnation_threshold: 15` (earlier trigger)
- `lns.apply_to_best_n: 5` (more aggressive, repair top 5)
- `lns.max_subproblem_size: 30` (larger subproblems)
- **CP-SAT multiprocessing**: Enabled
- **IGLS**: COMPLETELY DISABLED (all local search off)

**Run Command**:
```powershell
uv run python main.py --config configs/prod_no_local_search.yaml --experiment prod-no-ls-r01
```

**Expected Runtime**: 20-30 hours on 16+ cores (faster without IGLS)

**Watch For**:
- Gen 6 & 50: Forced triggers for validation
- How well pure GA evolves without local search
- CP-SAT trigger frequency and success
- Final solution quality vs IGLS-enabled runs

---

## Configuration 3: Hybrid Strategy (Baseline)

**File**: `configs/prod.yaml`

**What it tests**: Heuristic-first with CP escalation (recommended baseline)

**Key Settings**:
- `lns.repair_strategy: hybrid` (heuristic → CP)
- `lns.igls_time_limit: 8.0` (try IGLS first)
- `lns.cp_time_limit: 20.0` (escalate to CP if needed)
- **CP-SAT multiprocessing**: Enabled
- **IGLS**: Enabled (full local search)

**Run Command**:
```powershell
uv run prod --experiment prod-hybrid-r01
```

**Expected Runtime**: 24-48 hours on 16+ cores

---

## CP-SAT Multiprocessing Details

**What's Enabled**:
- `solver.parameters.num_search_workers = 0` (auto-detect cores)
- CP-SAT will run parallel search strategies internally
- Each LNS repair call uses multiple cores for the CP solve

**Performance Impact**:
- **Speedup**: 2-4x faster CP solves on multi-core systems
- **Trade-off**: Higher memory usage during repair
- **Recommended**: 16+ cores, 32+ GB RAM

**Monitoring**:
```powershell
# Watch CPU usage during gen 6 (CP trigger)
# CP-SAT should spike to 100% across multiple cores
```

---

## Validation Before Long Run

**Quick Test** (5 minutes):
```powershell
# Test CP-SAT strategy
uv run python main.py --config configs/prod_cp_only.yaml --experiment cp-test --ga.ngen 10

# Test no-local-search strategy
uv run python main.py --config configs/prod_no_local_search.yaml --experiment no-ls-test --ga.ngen 10
```

**Check for**:
- Gen 6 LNS trigger fires
- CP-SAT solver runs (check console output)
- Multiprocessing active (CPU usage spikes)
- No crashes or errors

---

## What to Share After Run

**Essential Files**:
1. `output/evaluation_*/lns_repair_stats.json` - Repair success rates
2. `output/evaluation_*/run.log` - Full execution log
3. Console output around gen 6 (LNS validation)
4. Final solution metrics (hard/soft violations)

**Key Metrics**:
```json
{
  "cp_attempts": X,
  "cp_success": Y,
  "cp_success_rate_percent": Z,
  "pre_check_skips": N,
  "avg_subproblem_size": M
}
```

---

## Troubleshooting

### Problem: CP-SAT Always Times Out

**Symptom**: `cp_success = 0`, all attempts fail with TIMEOUT

**Solutions**:
1. Increase `cp_time_limit: 45.0`
2. Reduce `max_subproblem_size: 20`
3. Check multiprocessing is active (CPU usage)

### Problem: CP-SAT Always Infeasible

**Symptom**: `cp_success = 0`, status=INFEASIBLE

**Solutions**:
1. Check pre-check is working (`pre_check_skips > 0`)
2. Increase `expand_neighborhood_hops: 3` (larger neighborhoods)
3. Review diagnostic logs for domain sizes

### Problem: No LNS Triggers

**Symptom**: `total_attempts = 0`

**Solutions**:
1. Verify `lns.enabled: true`
2. Check gen 6 output (forced trigger)
3. Ensure conflicts exist (hard violations > 0)

---

## Expected Outcomes

**Best Case** (CP-SAT excels):
- CP success rate: 70%+
- Pre-check skips: 10-20% (avoiding obviously infeasible)
- Final hard violations: 0-50
- Competitive with hybrid strategy

**Realistic Case**:
- CP success rate: 40-60%
- Pre-check skips: 20-30%
- Final hard violations: 50-200
- Hybrid strategy still better

**Worst Case** (CP-SAT struggles):
- CP success rate: <20%
- Pre-check skips: 40%+ (many infeasible subproblems)
- Final hard violations: 300+
- Need to fall back to heuristic or hybrid

---

## Recommended Test Sequence

1. **Run hybrid baseline** (prod.yaml) - establish quality target
2. **Run pure CP-SAT** (prod_cp_only.yaml) - test CP alone
3. **Run no local search** (prod_no_local_search.yaml) - test CP as IGLS replacement
4. **Compare results** - identify best strategy

**Total Time**: ~72-96 hours (run in parallel on 3 VMs if possible)

---

## Quick Commands

```powershell
# Pure CP-SAT
uv run python main.py --config configs/prod_cp_only.yaml --experiment prod-cp-only-r01

# No Local Search
uv run python main.py --config configs/prod_no_local_search.yaml --experiment prod-no-ls-r01

# Hybrid Baseline
uv run prod --experiment prod-hybrid-r01

# Extract LNS stats
Get-Content output/evaluation_*/lns_repair_stats.json | ConvertFrom-Json | Format-List

# Compare final fitness
Get-ChildItem output/evaluation_*/run.log | ForEach-Object { 
    Write-Host "`n=== $($_.Directory.Name) ==="; 
    Get-Content $_ | Select-String "hard violations:" | Select-Object -Last 1 
}
```

---

**Ready to run!**  Start with the validation test, then launch production runs on VM.
