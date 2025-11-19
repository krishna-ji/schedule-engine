# Training RL Agents

This guide covers the full lifecycle of training, validating, and promoting RL agents that guide heuristic selection.

## 1. Overview

| Stage | Description | Output |
| --- | --- | --- |
| Dataset Prep | Generate curated scheduling contexts of varying difficulty | `data/training_sets/*.json` |
| Curriculum Training | Train PPO/DQN agents with progressive difficulty | `models/rl_agents/ppo_stage*.zip` |
| Evaluation | Compare checkpoints on validation suite | `logs/training/eval_metrics.json` |
| Promotion | Copy best checkpoint to production location | `models/rl_agents/ppo_prod.zip` |

## 2. Prerequisites

```powershell
uv sync --frozen --group rl
uv run diagnose-system  # verify CUDA + PyTorch
```

Ensure at least 12GB VRAM for prod-grade training. CPU-based training is supported but 10x slower.

## 3. Curriculum Configuration

`configs/training/rl_curriculum.yaml` (example):

```yaml
curriculum:
  stages:
    - name: easy
      dataset: data/training_sets/easy.json
      generations: 30
      population: 60
    - name: medium
      dataset: data/training_sets/medium.json
      generations: 60
      population: 100
    - name: hard
      dataset: data/training_sets/hard.json
      generations: 120
      population: 150
  promotion_threshold: 0.15  # min reward improvement
```

## 4. Launch Training

```powershell
# Full curriculum (takes 2-3 hours on RTX 3080)
uv run train-curriculum --config configs/training/rl_curriculum.yaml --agent ppo --timesteps 5_000_000 --logdir logs/tensorboard/training/
```

Common flags:

| Flag | Description |
| --- | --- |
| `--agent {ppo,dqn}` | RL algorithm |
| `--timesteps` | Total environment steps per stage |
| `--eval-freq` | Evaluate every N steps |
| `--checkpoint-freq` | Save checkpoint interval |
| `--resume-from` | Continue training from checkpoint |
| `--no-gpu` | Force CPU training |

## 5. Monitor Training

```powershell
uv run tensorboard --logdir logs/tensorboard/training/
```

Watch:
- `charts/episode_reward` – should trend upwards per stage.
- `charts/action_entropy` – indicates exploration; avoid collapse to zero too early.
- `charts/value_loss` – spikes may indicate reward scale mismatch.

## 6. Evaluating Checkpoints

```powershell
uv run evaluate-rl \
    --checkpoints models/rl_agents/ppo_stage1.zip,models/rl_agents/ppo_stage2.zip \
    --dataset data/training_sets/hard_validation.json \
    --episodes 25
```

Metrics collected:
- Mean reward ± std.
- Avg improvement over GA-only baseline (% reduction in hard/soft violations).
- Action histogram divergence (ensures policy variety).
- Wall-clock inference latency.

## 7. Promotion Workflow

```powershell
$best = "models/rl_agents/ppo_stage3_reward+0.32.zip"
uv run promote-model `
    --checkpoint $best `
    --alias ppo_prod `
    --notes "curriculum-v4, reward +0.32 vs baseline"
```

This command copies checkpoint to `models/rl_agents/ppo_prod.zip`, updates `models/rl_agents/manifest.json`, and optionally tags git.

## 8. Integrating with GA Runs

Update config:
```yaml
rl:
  enabled: true
  agent_path: models/rl_agents/ppo_prod.zip
  inference:
    device: cuda
    timeout_ms: 20
```

Smoke-test:
```powershell
uv run rl --env test --generations 50 --seed 21
```

## 9. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Reward plateaus quickly | Curriculum jump too aggressive | Insert intermediate stage or lower difficulty gap |
| PPO diverges (NaNs) | Learning rate too high | Reduce `rl.training.lr` (default 3e-4) |
| GPU OOM | Large population × generations × parallel envs | Lower `vec_envs`, batch size, or enable gradient accumulation |
| Agent hurts GA performance | Reward weights mis-scaled | Normalize fitness deltas (~[-5, 5]) and re-train |
| Inference >20ms | Model too large | Reduce policy network width (config: `rl.policy.hidden_sizes`) |

## 10. Checklist Before Merging New Agent

- [ ] Training logs archived (`logs/training/run_<timestamp>.zip`).
- [ ] Evaluation report stored in `docs/development/rl-eval-<timestamp>.md`.
- [ ] `models/rl_agents/manifest.json` updated with metadata (commit hash, reward delta, dataset set IDs).
- [ ] `docs/how-to/03-training-rl-agents.md` (this file) updated if workflow changed.
- [ ] `docs/00-INDEX.md` link to latest training notes.

Following this process ensures every promoted agent is reproducible and demonstrably better than the previous baseline.
