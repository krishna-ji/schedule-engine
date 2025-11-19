# Debugging & Validation Playbook

Use this guide when schedules misbehave or experiments stall.

## 1. Validate Inputs First

```powershell
uv run verify-config --config configs/custom/my.yaml
uv run check-data --data-dir data/
```

- **Config issues:** Look for YAML merge mistakes, invalid enum values, or missing killswitches.
- **Data issues:** `data_editor.ipynb` can quickly visualize JSON anomalies (duplicate course IDs, unassigned instructors).

## 2. Reproduce Quickly

| Scenario | Command |
| --- | --- |
| GA crash | `uv run test --mode baseline --seed 7 --generations 20` |
| RL-specific bug | `uv run rl --env test --seed 7 --generations 30 --rl-deterministic` |
| GPU evaluation issue | `uv run test --mode full --use-gpu --generations 5` |

Keep runs short (<2 min) until root cause found.

## 3. Inspect Constraint Violations

```powershell
python scripts/diagnostics/print_constraint_breakdown.py --latest
```

Outputs table with per-constraint counts plus top violators. If a single constraint explodes:
- Trace to relevant module (e.g., `hard_room_exclusivity.py`).
- Re-run targeted unit tests `pytest test/unit/test_constraints.py -k room_exclusivity`.

## 4. Debugging the GA

### Enable Verbose Logs

```python
# src/core/ga_scheduler.py
logger.setLevel(logging.DEBUG)
```

Watch for:
- `Applying heuristic ...` – ensure RL/round-robin chooses expected operators.
- `GPU evaluator fallback` – indicates CUDA errors.
- `Repair applied` – confirm stagnation detection firing.

### Snapshot Individuals

```python
from src.ga.debug_tools import dump_individual

dump_individual(best_individual, path="debug/best_individual.json")
```

Open JSON in IDE to verify gene assignments.

## 5. RL Debugging

- Set `rl.logging.level = DEBUG` to print actions/rewards.
- Use `uv run inspect-rl --checkpoint models/rl_agents/ppo_prod.zip --episodes 5` for dry-run inference.
- Plot action distribution:
  ```powershell
  python scripts/diagnostics/plot_rl_actions.py --log output/rl_action_histogram.csv
  ```

## 6. GPU/Performance Issues

| Symptom | Command |
| --- | --- |
| CUDA missing | `python -c "import torch; print(torch.version.cuda)"` |
| Low utilization | `uv run diagnose-gpu --watch` |
| Memory leak suspicion | `python scripts/diagnostics/track_memory.py --interval 5` |

Disable GPU temporarily via config to isolate whether bug is CUDA-specific.

## 7. Validation Hooks During Runs

Set `validation.runtime.enabled = true` to rerun feasibility checks mid-experiment. GA pauses if violations exceed threshold, helping catch runaway heuristics early.

## 8. Using Breakpoints

- VS Code: add breakpoint inside constraint or heuristic, run `Python: Debug current file`.
- CLI: `python -m pdb main.py --env test --mode baseline` to step through CLI paths.

## 9. Postmortem Checklist

After resolving an issue:
1. Add regression test.
2. Document fix under `docs/development/bugfixes/<issue>.md`.
3. Update troubleshooting entry if user-facing.
4. Link commit hash in manifest if rerunning experiments.

With this playbook you can move from "something feels off" to validated fix in a structured way.
