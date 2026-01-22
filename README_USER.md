# User README (Schedule Engine)

This short guide summarizes how to run experiments, configure RL/GA options,
and interpret outputs.

## Quick start

```bash
uv sync --frozen
uv run run
```

Open a notebook from `notebooks/` (e.g., `mode_e_rl_guided.ipynb`) and run the
cells in order.

## Key configuration knobs

### RL environment stability & runtime

These live under `config.rl.environment`:

- `action_id_map`  
  Provide a stable mapping of heuristic names to action IDs to keep RL policies
  reproducible across runs.
- `diversity_update_interval`  
  Compute diversity metrics every **N** generations (higher = faster).
- `diversity_sample_size`  
  Optional subsample size to speed up diversity calculations.

### RL rewards

Rewards now prefer population-level improvement when a population is provided,
and fall back to per-individual deltas only if population data is unavailable.

## Suggested defaults for faster runs

For quicker experiments:

```yaml
rl:
  environment:
    diversity_update_interval: 5
    diversity_sample_size: 20
```

You can also reduce `pop_size` and `max_steps_per_episode` in your notebook
config.

## Outputs

Outputs are written to the `output/` directory, including:

- `schedule.json`
- plots and diagnostics
- RL training logs and analysis (if enabled)

## Tips

- Keep `action_id_map` stable once you start training an RL agent.
- If training feels slow, increase `diversity_update_interval`, decrease
  population size, or reduce max generations/steps in your experiment.
