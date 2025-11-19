# RL Training Issues

Troubleshooting guide for PPO/DQN training sessions.

## 1. Training Script Fails Immediately

| Error | Cause | Fix |
| --- | --- | --- |
| `ModuleNotFoundError: torch` | RL extra deps missing | `uv sync --group rl` |
| `CUDA error: device-side assert triggered` | Invalid tensor indices | Run with `CUDA_LAUNCH_BLOCKING=1` to capture stack trace |
| `ValueError: observation size mismatch` | State encoder schema changed | Retrain after updating `state_encoder.py` + tests |

## 2. Rewards Plateau at Zero

- Ensure reward weights sum to ~1 and no component dwarfs others.
- Check if GA baseline already solved easy datasets; increase curriculum difficulty.
- Inspect action histogram for collapse; add entropy bonus or epsilon-greedy noise.

## 3. PPO Instability (NaNs)

- Lower learning rate (3e-4 → 1e-4).
- Clip gradient norms: `rl.training.max_grad_norm = 0.5`.
- Disable mixed precision if using older GPUs.

## 4. Checkpoint Quality Drops

- Validate dataset shuffle; stale contexts lead to overfitting.
- Run `uv run evaluate-rl --checkpoints <path>` weekly to catch regressions.
- Keep at least two previous "good" checkpoints for rollback.

## 5. Inference Slowdowns

- Ensure `rl.inference.device` matches hardware (cpu/cuda).
- Benchmark with `uv run inspect-rl --checkpoint ... --profile`.
- Convert models to TorchScript if latency-critical (experimental feature flag `rl.inference.torchscript`).

## 6. Logging & Debugging

```powershell
uv run train-rl --debug --timesteps 100000
```

Prints per-episode rewards, action entropy, and loss stats. Combine with TensorBoard for long runs.

Document tricky incidents in `docs/development/bugfixes/` so future contributors can learn from prior mistakes.
