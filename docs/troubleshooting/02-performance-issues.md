# Performance Issues

Use this checklist when runs are slower than expected or GPU utilization plummets.

## 1. Baseline Diagnostics

```powershell
uv run diagnose-system
uv run benchmark-gpu
python scripts/diagnostics/track_memory.py --interval 5
```

Note population size, generations, and whether GPU is enabled.

## 2. Common Bottlenecks

| Symptom | Cause | Fix |
| --- | --- | --- |
| GPU utilization <40% | Batch size too small | Increase `evaluator.gpu.batch_size` (if VRAM allows) |
| CPU pegged at 100% | Parallel heuristics using too many threads | Lower `ga.parallel.max_workers` |
| Memory climbing steadily | Individuals accumulating in history buffers | Set `metrics.keep_generations` to lower value |
| Repair step >30s | Too many sessions destroyed | Reduce `repair.igls.max_sessions` or raise trigger threshold |
| RL inference >50ms | Large policy network | Shrink `rl.policy.hidden_sizes` |

## 3. Profiling Workflow

1. Enable `performance.profile.enabled = true` to capture per-stage timings.
2. Run short workload (`--generations 50`).
3. Inspect `output/<run>/performance_report.json` for hotspots.
4. Drill down using `scripts/diagnostics/profile_ga.py --section evaluator`.

## 4. GPU Troubleshooting

- Verify CUDA clocks: `nvidia-smi --query-gpu=clocks.sm --format=csv`.
- Lock max clocks for profiling: `nvidia-smi -lgc 1500,1800` (requires admin privileges).
- Set `torch.backends.cudnn.benchmark = True` for faster convolution-like kernels (safe here).
- If GPU throttles due to thermals, ensure laptop is plugged in and well ventilated.

## 5. Configuration Tweaks

```yaml
ga:
  pop_size: 150        # reduce by 25%
  ngen: 1500           # shorten prod runs when iterating
  parallel:
    enabled: true
    max_workers: 8     # match physical cores

heuristics:
  parallel:
    enabled: false     # disable if heuristics <5ms each
```

Use `configs/test.yaml` as template for lighter workloads during debugging.

## 6. When to Use CPU Instead

- Population <50 and GPU warm-up > evaluation time.
- Running inside containers without GPU pass-through.
- CI environments where deterministic floating-point behavior matters (use CPU for consistent results).

Document final resolution in `docs/development/bugfixes/` if the bottleneck required code changes.
