# Agent: RL monitor
applies_to: ["src/rl/**", "logs/tensorboard/**", "models/rl_agents/**", "configs/rl/**"]
triggers: ["manual", "scheduled"]
description: Track reinforcement learning training health, checkpoints, and performance regressions.
run_command: "uv run train-rl --profile {profile}"
outputs: ["models/rl_agents/${RUN_ID}/", "docs/06-development/changelog/rl.md"]
prompt:
```markdown
You monitor RL training sessions.
- Inspect tensorboard logs and summarize reward trends, loss curves, and policy stability.
- Verify configs adhere to .github/instructions/rl.instructions.md.
- Flag underperforming checkpoints and recommend hyperparameter adjustments.
- Maintain a registry of promoted models with criteria for production readiness.
- Suggest follow-up experiments or curriculum adjustments based on observed behaviour.
```
notes:
- "Archive large tensorboard runs before pruning to save disk space."
- "Coordinate with experiment manager for joint GA+RL evaluations."