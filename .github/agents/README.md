# Thesis Generation Agents

This directory contains autonomous AI agents for synthesizing graduate-level thesis chapters from the Schedule Engine codebase and experimental results.

## Agent Overview

### Master Coordinator
- **File**: `thesis-generator.md`
- **Purpose**: Orchestrates specialized agents to produce complete thesis
- **Chapters**: Chapter 8 (Conclusion)
- **Runtime**: 1 hour (conclusion only)
- **Total Runtime (all phases)**: 13-17 hours

### Specialized Agents

#### 1. Chapter 4-5 Synthesizer
- **File**: `ch4-5-synthesizer.md`
- **Chapters**:
  - Chapter 4: System Architecture and Methodology
  - Chapter 5: Implementation Details
- **Focus**: Mathematical rigor, technical depth, algorithm analysis
- **Runtime**: 4-5 hours
- **Key Deliverables**:
  - Formal problem formulation with LaTeX equations
  - NSGA-II mathematical framework
  - Quantum Time System complexity analysis
  - SessionGene architecture evolution (Nov 2025)
  - Pseudo-code for key algorithms

#### 2. Result Analyzer
- **File**: `result-analyzer.md`
- **Output**: `result_analysis_report.md` (intermediate, not thesis chapter)
- **Focus**: Statistical analysis, empirical evaluation, comparative study
- **Runtime**: 3-4 hours
- **Key Deliverables**:
  - Quantitative performance tables
  - Statistical significance testing (Mann-Whitney U, Cohen's d)
  - Convergence analysis and phase identification
  - Constraint-specific bottleneck analysis
  - Computational performance metrics

#### 3. Chapter 6-7 Synthesizer
- **File**: `ch6-7-synthesizer.md`
- **Chapters**:
  - Chapter 6: Result Analysis and Discussion
  - Chapter 7: Future Work
- **Focus**: Results interpretation, critical analysis, research planning
- **Runtime**: 5-7 hours
- **Key Deliverables**:
  - Synthesis of empirical findings
  - Threats to validity analysis
  - Research questions answered with evidence
  - Comprehensive future work roadmap (near/medium/long-term)

## Execution Workflow

### Phase 1: Architecture & Implementation (4-5 hours)
```
Run: ch4-5-synthesizer.md
Input: src/**, configs/**
Output: chapter_4_system_architecture_methodology.md
        chapter_5_implementation_details.md
```

**What it does**:
1. Systematically inspects all source modules
2. Extracts mathematical formulations and algorithms
3. Analyzes design patterns and complexity
4. Generates graduate-level technical chapters with equations and pseudo-code

### Phase 2: Result Analysis (3-4 hours)
```
Run: result-analyzer.md
Input: output/**, experiment_manifest.json
Output: result_analysis_report.md
```

**What it does**:
1. Parses experiment manifest for run metadata
2. Extracts convergence data, constraint violations, runtime metrics
3. Performs statistical analysis (hypothesis tests, effect sizes)
4. Generates comparative tables and figure placeholders

### Phase 3: Results & Future Work (5-7 hours)
```
Run: ch6-7-synthesizer.md
Input: result_analysis_report.md, chapter_4*.md, chapter_5*.md, Todo.md
Output: chapter_6_result_analysis_discussion.md
        chapter_7_future_work.md
```

**What it does**:
1. Synthesizes empirical findings into thesis narrative
2. Connects results to methodological choices (cross-references Chapters 4-5)
3. Extracts future work from Todo.md and identified limitations
4. Provides roadmap with effort estimates and prioritization

### Phase 4: Conclusion (1 hour)
```
Run: thesis-generator.md (master agent)
Input: chapter_4*.md, chapter_5*.md, chapter_6*.md, chapter_7*.md
Output: chapter_8_conclusion.md
```

**What it does**:
1. Summarizes contributions from all chapters
2. Restates research questions and answers
3. Articulates impact and significance
4. Provides final forward-looking statement

## Prerequisites

### Required Files
-  Full codebase (`src/**/*.py`)
-  Configuration files (`configs/**/*.yaml`)
-  Experimental results (`output/*/`)
-  Experiment manifest (`output/experiment_manifest.json`)
-  Project TODO list (`Todo.md`)

### Optional but Recommended
- Recent experimental runs (within last week)
- Multiple runs per runtime mode (for statistical analysis)
- Complete convergence data (`ga_stats.json` per run)

## Output Structure

All generated files are placed in `docs/thesis/[YYYY-MM-DD]/`:

```
docs/thesis/2025-11-29/
├── chapter_4_system_architecture_methodology.md    (10-15 pages)
├── chapter_5_implementation_details.md             (12-18 pages)
├── result_analysis_report.md                       (8-12 pages, intermediate)
├── chapter_6_result_analysis_discussion.md         (12-15 pages)
├── chapter_7_future_work.md                        (6-8 pages)
└── chapter_8_conclusion.md                         (2-3 pages)
```

**Total**: ~50-70 pages of thesis-ready content

## Agent Characteristics

### Technical Depth
- **Graduate-level mathematics**: Formal derivations, LaTeX equations
- **Algorithm analysis**: Complexity bounds (Big-O), performance proofs
- **Statistical rigor**: Hypothesis tests, effect sizes, confidence intervals

### Writing Style
- **Formal academic tone**: First-person plural ("we") for our work
- **Precision**: Discipline-specific terminology (NP-hard, Pareto front, etc.)
- **Conciseness**: High entropy, no "AI-filler" phrases
- **Integration**: Cross-references between chapters, consistent notation

### Quality Assurance
- **Source verification**: All claims validated against codebase
- **Mathematical correctness**: Equations numbered and referenced
- **Statistical validity**: Proper interpretation of p-values and effect sizes
- **Reproducibility**: Clear descriptions enable replication

## Usage Tips

### Before Running Agents

1. **Run recent experiments**:
   ```bash
   uv run baseline --test
   uv run memetic --test
   uv run roundrobin --test
   ```

2. **Check experiment manifest**:
   ```bash
   cat output/experiment_manifest.json | jq '.'
   ```
   Ensure runs are complete (no null fields in critical metrics)

3. **Update Todo.md**:
   - Add any pending tasks or future work ideas
   - Categorize by priority (high/medium/low)

4. **Clean git state** (recommended):
   ```bash
   git status  # Ensure no uncommitted changes
   ```

### Running Agents

**Sequential Execution** (recommended):
```bash
# Phase 1: Architecture & Implementation
uv run generate-chapters-4-5 --date 2025-11-29

# Phase 2: Result Analysis
uv run analyze-results --date 2025-11-29

# Phase 3: Results & Future Work
uv run generate-chapters-6-7 --date 2025-11-29

# Phase 4: Conclusion
uv run generate-thesis --date 2025-11-29
```

**Parallel Execution** (if independent):
- Phase 1 and Phase 2 can run in parallel (different inputs)
- Phase 3 depends on Phase 1 + Phase 2 outputs
- Phase 4 depends on all previous phases

### After Completion

1. **Review generated chapters**:
   - Check for `[TBD]` placeholders (missing data)
   - Verify equation numbering consistency
   - Validate cross-references (Section X.Y exists?)

2. **Integrate figures**:
   - Replace placeholders with actual plots from `output/*/plots/`
   - Ensure all figures referenced in text

3. **Proofread**:
   - Run spell checker
   - Verify citation placeholders (`[citation: Author, YYYY]`)
   - Check notation consistency across chapters

4. **Export to LaTeX** (optional):
   - Convert Markdown to LaTeX via Pandoc
   - Adjust formatting for thesis template

## Customization

### Modifying Agent Behavior

Each agent is configured via its `.md` file metadata:

```markdown
# Agent: Name
applies_to: ["path/patterns"]
triggers: ["manual", "workflow:name"]
description: What the agent does
run_command: "uv run command --date ${DATE}"
outputs: ["output/paths"]
notes:
- "Configuration notes"
```

### Extending Agents

To add new chapters or modify synthesis:

1. **Copy existing agent** (e.g., `ch4-5-synthesizer.md`)
2. **Modify section structure** in the agent file
3. **Update input/output paths**
4. **Add to master coordinator** (`thesis-generator.md`)
5. **Register command** in `pyproject.toml` (if needed)

### Adjusting Writing Style

Edit the "Writing Standards" section in each agent file:
- Change persona description
- Modify tone guidelines (formal  conversational)
- Add/remove prohibited phrases
- Adjust equation/notation standards

## Troubleshooting

### Agent Produces `[TBD]` Placeholders
**Cause**: Missing experimental data or incomplete runs  
**Fix**:
- Run experiments: `uv run baseline --test`
- Check manifest: `output/experiment_manifest.json`
- Ensure `ga_stats.json` exists in run directories

### Inconsistent Notation Across Chapters
**Cause**: Agents running independently without coordination  
**Fix**:
- Run Phase 1 first (establishes notation)
- Verify Chapter 4 notation tables
- Phase 3 agent reads Chapters 4-5 for consistency

### Statistical Tests Fail (N/A or undefined)
**Cause**: Insufficient runs per mode (need $n \geq 3$ for Mann-Whitney U)  
**Fix**:
- Run multiple replicates with different seeds
- Use descriptive statistics only if $n < 3$
- Document as limitation in Chapter 6

### LaTeX Equations Not Rendering
**Cause**: Markdown viewer doesn't support KaTeX/MathJax  
**Fix**:
- Use VS Code with Markdown+Math extension
- Or convert to LaTeX: `pandoc input.md -o output.tex`
- Or use online viewer (e.g., HackMD, Typora)

## Agent Development

### Adding a New Agent

1. **Create agent file**: `.github/agents/my-agent.md`
2. **Define metadata**: applies_to, triggers, outputs
3. **Write persona**: Academic writer, researcher, etc.
4. **Specify inputs**: What files/data agent needs
5. **Structure output**: Section-by-section outline
6. **Add execution protocol**: Step-by-step workflow
7. **Define quality checklist**: Validation criteria
8. **Test on sample data**: Verify output quality

### Agent Best Practices

- **Modularity**: Each agent has clear, non-overlapping scope
- **Reproducibility**: Same inputs → same outputs (deterministic)
- **Traceability**: Log all files read and insights extracted
- **Validation**: Cross-verify claims against source code
- **Documentation**: Clear instructions for users

## References

### Related Documentation
- `.github/copilot-instructions.md` - Project coding standards
- `docs/02-user-guides/runtime-modes.md` - Experiment framework
- `docs/45-resource-unused-problem/THESIS_EXPERIMENTS_GUIDE.md` - Experimental setup

### Inspiration
- Academic thesis templates (LaTeX)
- Scientific paper structures (IMRaD format)
- Software engineering best practices (clean code, DRY)

---

## Quick Start Example

**Generate complete thesis in one session** (13-17 hours):

```bash
# Set date variable
export THESIS_DATE=$(date +%Y-%m-%d)

# Phase 1: Architecture & Implementation (4-5 hours)
uv run generate-chapters-4-5 --date $THESIS_DATE

# Phase 2: Result Analysis (3-4 hours)
uv run analyze-results --date $THESIS_DATE

# Phase 3: Results & Future Work (5-7 hours)
uv run generate-chapters-6-7 --date $THESIS_DATE

# Phase 4: Conclusion (1 hour)
uv run generate-thesis --date $THESIS_DATE

# Verify outputs
ls -lh docs/thesis/$THESIS_DATE/
```

**Expected output**:
```
docs/thesis/2025-11-29/
├── chapter_4_system_architecture_methodology.md    (~10-15 pages)
├── chapter_5_implementation_details.md             (~12-18 pages)
├── result_analysis_report.md                       (~8-12 pages)
├── chapter_6_result_analysis_discussion.md         (~12-15 pages)
├── chapter_7_future_work.md                        (~6-8 pages)
└── chapter_8_conclusion.md                         (~2-3 pages)

Total: ~50-70 pages of thesis-ready content
```

---

**Status**: Agents ready for deployment (November 2025)  
**Version**: 1.0  
**Maintainer**: Schedule Engine Team  
**License**: Same as parent project
