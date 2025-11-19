# Experiment Log Template

Use this template whenever you run a notable experiment (thesis runs, RL benchmarks, GPU profiling, etc.).

## Metadata

| Field | Example |
| --- | --- |
| Experiment ID | `2025-11-20_rl_curriculum_v4` |
| Git Commit | `abc1234` |
| Config Hash | `5f3c9d2` (from manifest) |
| Runtime Mode | `rl` |
| Environment | `prod` |
| Seed | `42` |
| Hardware | `RTX 4090 + Ryzen 9 7950X` |

## Configuration Snapshot

```yaml
# Paste diff vs base.yaml or attach custom config path
```

## Results Summary

| Metric | Value |
| --- | --- |
| Best Hard Violations | `-8` |
| Best Soft Penalty | `-2.4` |
| Time to Feasible | `3m 12s` |
| Total Wall Time | `1h 45m` |
| GPU Utilization | `92% avg` |

## RL Telemetry (if applicable)

- Mean reward: `+0.35 ± 0.04`
- Action entropy: `0.62`
- Dominant actions: `swap_rooms`, `balance_workload`
- Repair success rate: `78%`

## Observations

- Example: "Repair trigger threshold of 12 gens prevented stagnation spikes." 
- Example: "RL agent overused diversity operators after gen 150; consider entropy bonus tuning."

## Follow-up Tasks

- [ ] Re-run with new reward weights
- [ ] Update documentation section XYZ
- [ ] Promote checkpoint `models/rl_agents/ppo_stage3.zip`

Copy this template into `docs/development/experiment-log.md` for each run so future readers can trace historical performance improvements.
