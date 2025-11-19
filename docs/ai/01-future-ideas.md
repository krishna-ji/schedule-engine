# AI Enhancement Ideas

Brainstormed directions for future AI-assisted scheduling research.

## 1. Multi-Agent RL

- **Concept:** Assign specialist agents per constraint cluster (rooms, instructors, groups) that negotiate actions.
- **Benefit:** Better handling of conflicting objectives; each agent optimizes its own slice before consensus.
- **Implementation Sketch:** Extend `rl/multiagent/` (placeholder) with PettingZoo-style API; share population state but provide constraint-specific observation heads.

## 2. Transfer Learning Across Campuses

- Train on multiple institutions with domain randomization (course counts, room capacities) to build generalized policies.
- Store embeddings of context statistics to warm-start RL on new campuses, reducing training time.

## 3. Surrogate Fitness Models

- Fit gradient-boosted trees or lightweight neural nets to approximate constraint violations, enabling faster GA iterations when GPU unavailable.
- Periodically recalibrate surrogate with real evaluations to avoid drift.

## 4. LLM-Assisted Constraint Authoring

- Use prompt templates to convert natural-language policy statements into constraint stubs.
- Leverage unit tests generated from examples to validate the generated constraints before integration.

## 5. Adaptive Curriculum Scheduler

- Automate curriculum stage progression based on reward plateau detection rather than fixed timesteps.
- Could be managed by a smaller "coach" agent deciding when to escalate difficulty.

## 6. RL Explainability Dashboard

- Combine SHAP values with RL action logs to show which features drive heuristic choices.
- Helps trust-building with academic stakeholders.

Track which ideas graduate from this list in `docs/development/implementation-notes/` once implemented.
