# Technical Q&A

Curated knowledge base for recurring architectural questions asked by contributors.

## Q1. Why use NSGA-II instead of single-objective GA?

**A:** Timetabling balances mutually competing goals (hard vs soft constraint minimization). NSGA-II preserves a Pareto front so researchers can choose different trade-offs without rerunning experiments. Single-objective GA would require manually tuning a composite weight and risks losing feasible-yet-diverse solutions.

## Q2. Why mix RL with heuristics instead of training an end-to-end policy?

**A:** Full end-to-end policies struggle with the large combinatorial search space and strict feasibility requirements. Treating heuristics as actions leverages domain knowledge (19 curated operators) while still enabling adaptive sequencing. This hybrid approach converges faster and respects feasibility constraints baked into the operators.

## Q3. How do we guarantee deterministic runs?

- CLI accepts `--seed`; stored in config and passed into Python `random`, NumPy, PyTorch.
- GA toolbox uses seeded `random.Random` instances for crossover/mutation.
- RL inference optionally set to deterministic mode.
- Experiment manifest records the seed + git commit for reproducibility.

## Q4. Why is the GPU evaluator optional?

Some contributors run on laptops/CI without CUDA. The evaluator auto-detects GPU availability, but we keep an explicit killswitch (`evaluator.gpu.enabled`) to compare CPU/GPU behavior and to simplify debugging without GPU-specific complexity.

## Q5. What motivated the 25D RL state vector?

It balances expressiveness (captures constraint mix, diversity, operator success) with inference latency (<0.1ms). More features increased overfitting and slowed inference; fewer features reduced policy quality. 25D was empirically the sweet spot.

## Q6. How are runtime modes validated?

`RuntimeMode.validate_config()` ensures each mode loads mandatory killswitches and probabilities. Example: Mode 5 (RL) asserts `rl.enabled = true`, `heuristics.enabled = true`, and `repair.igls.enabled = true`. CI can run `uv run list-modes --validate` to confirm all modes remain consistent.

## Q7. Why keep local search (IGLS) separate from heuristics?

IGLS operates on a subset of the schedule and is triggered only during stagnation, whereas heuristics run every generation on individuals chosen by the GA. Keeping them separate avoids coupling their life cycles and simplifies telemetry (we can independently track repair success rate vs heuristic success rate).

## Q8. How is experiment metadata structured?

See `output/experiment_manifest.json`. Each entry records UUID, git commit, config hash, runtime mode, environment, seed, wall-clock time, and best fitness. When comparing experiments we rely on config hashes to ensure apples-to-apples comparisons.

## Q9. When should we introduce a new runtime mode?

Only when the feature changes workflow semantics (e.g., enabling specialists, hierarchical RL). Smaller toggles should remain as config flags within existing modes to avoid combinatorial explosion. New modes require documentation in configs, docs, CLI scripts, and runtime validators.

## Q10. Can RL and GPU features be combined safely?

Yes. RL runs on CPU (or GPU for training) while GPU evaluator only handles constraint batches. Synchronization happens at GA scheduler; RL receives metrics after GPU evaluation completes. Keep GPU batch sizes moderate to prevent increased latency feeding RL state.

Use this Q&A as the living FAQ for onboarding discussions and design reviews.
