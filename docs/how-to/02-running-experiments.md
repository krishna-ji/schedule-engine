# Running Experiments

This guide explains how to design, execute, and analyze experiments across all runtime modes.

## 1. Decide What to Measure

| Goal | Recommended Mode | Metrics |
| --- | --- | --- |
| Baseline GA (no helpers) | `baseline` (mode 1) | Hard/soft violations per generation |
| IGLS repair impact | `repairs` (mode 2) | Time-to-feasible, repair success rate |
| Heuristic toolbox quality | `heuristics` (mode 3) | Operator hit rate, diversity |
| Full GA without RL | `full` (mode 4) | Pareto front coverage |
| RL-guided heuristics | `rl` (mode 5) | Reward trends, action entropy |
| Specialist agents | `specialists` (mode 7) | Sub-agent win rates |
| Archive diversity | `archive` (mode 8) | Archive size, novelty |
| Hierarchical RL | `hierarchical` (mode 9) | High/low-level agreement |
| Multi-agent RL | `multiagent` (mode 10) | Agent cooperation score |

## 2. Prepare Configurations

1. Start from `configs/base.yaml` + env override (test/prod).
2. Copy target runtime mode config to `configs/custom/my_experiment.yaml`.
3. Adjust knobs (population, ngen, heuristics, rl reward weights) with comments.
4. Validate:
   ```powershell
   uv run verify-config --config configs/custom/my_experiment.yaml
   ```

## 3. Execute a Single Experiment

```powershell
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tag = "exp5_rl_${timestamp}"
uv run prod --mode rl --experiment $tag --config configs/custom/my_experiment.yaml
```

Outputs saved under `output/evaluation_<timestamp>_<tag>/` with manifest entry.

### CLI Flags

| Flag | Description |
| --- | --- |
| `--env {test,prod}` | Inherit environment overrides |
| `--mode <runtime-mode>` | One of 10 runtime modes |
| `--config path` | Custom merged config |
| `--seed int` | Override RNG seed |
| `--experiment str` | Human-friendly tag stored in manifest |
| `--no-export` | Skip heavy exporters during dry runs |

## 4. Batch Experiments (PowerShell)

```powershell
$experiments = @(
    @{ Mode = "baseline"; Tag = "exp1" },
    @{ Mode = "repairs";  Tag = "exp2" },
    @{ Mode = "heuristics"; Tag = "exp3" },
    @{ Mode = "full"; Tag = "exp4" },
    @{ Mode = "rl"; Tag = "exp5" }
)

foreach ($exp in $experiments) {
    $tag = "${($exp.Tag)}_$(Get-Date -Format 'yyyyMMdd_HHmm')"
    Write-Host "Running" $exp.Mode "as" $tag
    uv run prod --mode $exp.Mode --experiment $tag --env prod
}
```

## 5. Automate via `ExperimentManager`

```python
# scripts/run_experiment_suite.py
from src.config import init_config
from src.workflows.standard_run import run_standard_workflow
from src.workflows.experiment_manager import ExperimentManager

manager = ExperimentManager()
scenarios = [
    ("baseline", "configs/nsga/1-pure-nsga.yaml"),
    ("full", "configs/nsga/4-nsga-full.yaml"),
    ("rl", "configs/rl/5-rl-guided.yaml"),
]

for mode, path in scenarios:
    config = init_config(path)
    result = run_standard_workflow(config=config, runtime_mode=mode)
    manager.record_run(result, config=config, notes=f"automated-{mode}")
```

## 6. Monitoring During Runs

- **Console:** Rich progress bar shows generation, best fitness, diversity.
- **Logs:** `logs/run.log` captures debug statements (enable by setting `LOG_LEVEL=DEBUG`).
- **TensorBoard:** If RL enabled, run `uv run tensorboard` to watch reward curves.
- **System monitors:** `uv run diagnose-system --watch` prints GPU/CPU utilization every minute.

## 7. Analyzing Results

1. **Manifest diff:**
   ```powershell
   uv run compare-experiments --left 2025-11-20_abc --right 2025-11-20_def
   ```
2. **Plots:** generated automatically under `plots/` (fitness, diversity, Pareto front).
3. **Violation heatmap:** `output/violation_heatmap.json` visualizes constraint hotspots.
4. **RL action stats:** `output/rl_action_histogram.csv` (when RL on) showing chosen heuristics.
5. **Custom notebooks:** open `notebooks/experiment_analysis.ipynb` (if available) to combine metrics.

## 8. Reproducing Experiments

- Always capture `git rev-parse HEAD`, config hash, `uv pip freeze` output inside manifest.
- Set explicit `--seed` and store in README for the experiment.
- Archive data snapshots under `data/archive/<experiment>/` to avoid silent JSON drift.

## 9. Troubleshooting Failed Runs

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| Run aborts before GA | Validation failure | Inspect `output/.../validation_report.txt` |
| GPU fallback message | CUDA unavailable | Install drivers or disable GPU in config |
| RL agent stuck | Reward weights mis-scaled | Tune `rl.reward.*` to keep values in [-1, 1] |
| Export takes >5 min | Plotting huge graphs | Use `--no-export` then run `uv run export-latest` later |

## 10. Best Practices

- Keep prod runs under 3 hours by monitoring GPU memory and population size.
- Use descriptive experiment tags (`rl-curriculum-v2-ls-on`) to simplify searching the manifest.
- Store summary tables in `docs/development/experiment-log.md` for posterity.
- Clean up `output/` after archiving to keep repository size reasonable.

Refer back to this playbook whenever you plan thesis batches or benchmark new heuristics.
