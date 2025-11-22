# Agent: Thesis generator
applies_to: ["prompts/thesis_generator.md", "docs/thesis/**", "docs/for_report/**", "src/**"]
triggers: ["manual", "workflow:generate-thesis"]
description: Produce multi-chapter thesis drafts by synthesizing code and report sources.
run_command: "uv run generate-thesis --date ${DATE}"
outputs: ["docs/thesis/${DATE}/"]
prompt_file: "prompts/thesis_generator.md"
notes:
- "Ensure docs/for_report/ is current before running."
- "Uses project-wide context; run in clean git state when possible."