# 🚀 QUICK START - Hybrid CP-SAT → NSGA-II

## 📦 Installation
```bash
uv pip install ortools  # Already done in your env
```

## ⚡ Run Commands

### Test Mode (Fast - 1-3 minutes)
```bash
uv run python main.py --config configs/hybrid_test.yaml --mode hybrid
```

### CP-SAT Only (Fastest - 30s-2min)
```bash
uv run python main.py --config configs/hybrid_test.yaml --mode cpsat
```

### Full Production (10-30 minutes)
```bash
uv run python main.py --config configs/hybrid_prod.yaml --mode hybrid
```

## 📊 What to Expect

### Success Output:
```
✓ Phase 1 Complete (XXs)
Generated 10 feasible solutions

✓ Phase 2 Complete (XXs)
Pareto front size: X

Best Solution:
  Strict penalty: XX.XX
  Loose penalty: XX.XX
  Sessions: XXXX
```

### If Slow:
- **Model building:** <5s (should be fast)
- **CP-SAT solving:** 30s-5min (can be slow on first solution)
- **If >5min:** Check CPU usage, wait for time_limit

## ✅ Verification

```python
# Check hard constraints (should all be 0)
result["best_individual"].fitness.values[0]  # Hard violations
result["best_individual"].fitness.values[1]  # Soft penalty
```

## 🐛 If It Fails

1. **Import Error:** `uv run` ensures correct env
2. **Config Error:** Check YAML syntax
3. **Memory Error:** Reduce `num_solutions` 
4. **Timeout:** Increase `time_limit`
5. **Infeasible:** Check data quality

## 📝 Key Files

- **Config:** `configs/hybrid_test.yaml`
- **Main:** `main.py`
- **Workflow:** `src/workflows/hybrid_workflow_v2.py`
- **CP-SAT:** `src/ortools/cp_scheduler.py`

**Ready to test! 🎯**
