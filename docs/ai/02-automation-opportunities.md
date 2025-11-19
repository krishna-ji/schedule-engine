# Automation Opportunities

Catalog of tasks ripe for AI or scripting assistance.

## 1. Constraint Regression Bot

- **Goal:** Automatically generate failing individuals when new constraints are introduced.
- **Approach:** Search-based tester uses evolutionary strategies to synthesize edge cases, then opens GitHub issues with repro steps.

## 2. RL Drift Monitor

- **Goal:** Detect when deployed RL agent performance drifts below baseline.
- **Approach:** Scheduled job replays recent populations with both RL-enabled and RL-disabled runs, comparing violation deltas and alerting via Slack/Teams.

## 3. Config Linter

- **Goal:** Prevent invalid YAML combinations before runtime.
- **Approach:** Static analyzer (maybe powered by an LLM) that reviews PR diffs to ensure killswitch dependencies remain satisfied.

## 4. Experiment Summarizer

- **Goal:** Auto-generate markdown summaries for runs recorded in `experiment_manifest.json`.
- **Approach:** Script reads manifest + plots and drafts report sections (fit for thesis appendices) for human review.

## 5. Heuristic Auto-Tuner

- **Goal:** Suggest probability schedules for heuristics based on historical success metrics.
- **Approach:** Bayesian optimization over `ga.heuristics.probabilities` using offline logs, optionally guided by AI-generated priors.

## 6. Knowledge Graph Sync

- **Goal:** Keep docs/index cross-links accurate.
- **Approach:** CI job parses markdown headings and warns when referenced files disappear or rename without index update.

These automation hooks reduce toil and ensure the documentation + experimentation workflow scales with the project.
