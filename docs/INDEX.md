# Schedule Engine Documentation Index

**Project:** University Course Timetabling Problem (UCTP) Solver  
**Last Updated:** November 17, 2025  
**Organization:** Restructured into 10 categorized sections

---

##  Documentation Structure

**Note**: Section 11 (Advanced Techniques) added November 17, 2025 - Comprehensive guide for enhancing the Memetic NSGA-II architecture.

### [01 - Getting Started](./01-getting-started/)
Quick onboarding for new users
- Installation and setup (coming soon)
- First run quickstart (coming soon)
- Configuration basics (coming soon)
- Common troubleshooting (coming soon)

### [02 - User Guides](./02-user-guides/)
How-to guides for running and using the engine
- **Running the Engine:** [PROD_RUN_GUIDE.md](./02-user-guides/PROD_RUN_GUIDE.md), [PROD_RUNTIME_BREAKDOWN.md](./02-user-guides/PROD_RUNTIME_BREAKDOWN.md)
- **Configuration:** [CONFIG_QUICKSTART.md](./02-user-guides/CONFIG_QUICKSTART.md), [CONFIG_VISUAL_GUIDE.md](./02-user-guides/CONFIG_VISUAL_GUIDE.md)
- **Understanding Output:** [OUTPUT_STRUCTURE_GUIDE.md](./02-user-guides/OUTPUT_STRUCTURE_GUIDE.md)
- **Metrics & Evaluation:** [METRICS_QUICKSTART.md](./02-user-guides/METRICS_QUICKSTART.md)
- **UV Package Manager:** [UV_QUICKSTART.md](./02-user-guides/UV_QUICKSTART.md), [UV_MIGRATION.md](./02-user-guides/UV_MIGRATION.md)

### [03 - Architecture](./03-architecture/)
System design, data flow, and component interactions
- **RL-GA Integration:** [rl-ga-integ-framework.md](./03-architecture/rl-ga-integ-framework.md)
- Overall architecture (coming soon)
- GA core components (coming soon)
- Constraint system design (coming soon)
- Repair system architecture (coming soon)

### [04 - Algorithms & Performance](./04-algorithms/)
Technical deep dives into algorithms, methods, and performance optimization
- **Heuristics:** [HEURISTICS_QUICKREF.md](./04-algorithms/HEURISTICS_QUICKREF.md)
- **Large Neighborhood Search:** [LNS_PROD_RUN_GUIDE.md](./04-algorithms/LNS_PROD_RUN_GUIDE.md), [LNS_FORCE_TRIGGER_VALIDATION.md](./04-algorithms/LNS_FORCE_TRIGGER_VALIDATION.md)
- **Block Clustering:** [BLOCK_CLUSTERING_CONFIG.md](./04-algorithms/BLOCK_CLUSTERING_CONFIG.md)
- **Parallel Processing:** [PARALLEL_QUICKSTART.md](./04-algorithms/PARALLEL_QUICKSTART.md)
- **Time Complexity Analysis:** [time-complexity-algorithmic-analysis/](./04-algorithms/time-complexity-algorithmic-analysis/)
  - [Executive Summary](./04-algorithms/time-complexity-algorithmic-analysis/00_EXECUTIVE_SUMMARY.md)
  - [Full Analysis](./04-algorithms/time-complexity-algorithmic-analysis/01_COMPLEXITY_ANALYSIS.md)
  - [Optimization Strategies](./04-algorithms/time-complexity-algorithmic-analysis/02_OPTIMIZATION_STRATEGIES.md)
  - [Benchmark Guide](./04-algorithms/time-complexity-algorithmic-analysis/03_BENCHMARK_GUIDE.md)
- **GPU Acceleration:** [nvidia-gpu/](./04-algorithms/nvidia-gpu/)
  - [Quick Start (5 min)](./04-algorithms/nvidia-gpu/QUICKSTART.md)
  - [Comprehensive Guide](./04-algorithms/nvidia-gpu/GPU_ACCELERATION_GUIDE.md)
  - [Deployment Guide](./04-algorithms/nvidia-gpu/DEPLOYMENT.md)
- **Parallelization Analysis:** [parallelism/](./04-algorithms/parallelism/)
- **Optimization Notes:** [PRODUCTION_OPTIMIZATION_SUMMARY.txt](./04-algorithms/PRODUCTION_OPTIMIZATION_SUMMARY.txt)
- **Performance Analysis Reports:** [ANALYSIS_SUMMARY.md](./04-algorithms/ANALYSIS_SUMMARY.md), [CONSTRAINT_WEIGHT_ANALYSIS.md](./04-algorithms/CONSTRAINT_WEIGHT_ANALYSIS.md)
- NSGA-II implementation (coming soon)

### [05 - Development](./06-development/)
For developers: changelogs, implementation notes, code quality
- **Changelogs:**
  - [Bug Fixes](./06-development/changelog/bugfixes.md)
  - [Enhancements](./06-development/changelog/enhancements.md)
- **Implementation Notes:**
  - [Phase 1.5: Heuristics](./06-development/implementation-notes/PHASE_1.5_SUMMARY.md)
  - [Phase 2.1: RL Environment](./06-development/implementation-notes/PHASE_2.1_SUMMARY.md)
  - [Phase 2: Complete RL](./06-development/implementation-notes/PHASE_2_RL_COMPLETE.md)
  - [Config Refactoring](./06-development/implementation-notes/CONFIG_MIGRATION_COMPLETE.md)
  - [Repair Registry Refactor](./06-development/implementation-notes/REPAIR_REGISTRY_REFACTOR.md)
- **Code Quality:** [CODE_QUALITY_ENHANCEMENTS.md](./06-development/code-quality/CODE_QUALITY_ENHANCEMENTS.md)

### [06 - Thesis & Report](./07-thesis-report/)
Thesis-ready documentation and academic content
- Academic-style documentation
- Thesis chapter materials
- Publication-ready content
- Evaluation metrics documentation

### [07 - Q&A](./08-qna/)
Questions, discussions, and AI-assisted problem solving
- Technical questions (your active workspace)
- Algorithmic discussions
- Architecture decisions
- Troubleshooting sessions

### [08 - Future Plans](./09-future-plans/)
Roadmap, enhancement ideas, and research directions
- Project roadmap (coming soon)
- Phase 3 planning (coming soon)
- Enhancement proposals (coming soon)
- Research directions (coming soon)

### [09 - AI Suggestions](./10-ai-suggestions/)
AI-generated recommendations and guides
- [RL Phase 2.2-2.4 Guide](./10-ai-suggestions/rlphase2.2-2.4_guide_manual.md)
- Historical AI suggestions: [olds/](./10-ai-suggestions/olds/)

### [10 - Advanced Techniques](./11-advanced-techniques-suggest/)
Comprehensive technical documentation for enhancing Memetic NSGA-II with advanced RL techniques
- **[README & Navigation](./11-advanced-techniques-suggest/README.md)**  Start here
- **Current State:** [01 - Current Architecture](./11-advanced-techniques-suggest/01-current-architecture.md)
- **Practical Enhancements (Better):**
  - [02 - Multi-Objective Reward](./11-advanced-techniques-suggest/02-multi-objective-reward.md) - Pareto-aware RL
  - [03 - Specialist Agents](./11-advanced-techniques-suggest/03-specialist-agents.md) - Task-specific policies
  - [04 - Constraint-Specific State](./11-advanced-techniques-suggest/04-constraint-specific-state.md) - Per-constraint breakdown
  - [05 - Archive-Based Diversity](./11-advanced-techniques-suggest/05-archive-based-diversity.md) - Novelty search
  - [06 - Memetic RL](./11-advanced-techniques-suggest/06-memetic-rl.md) - RL-guided local search
  - [07 - Adaptive Probabilities](./11-advanced-techniques-suggest/07-adaptive-probabilities.md) - Dynamic operator tuning
- **Advanced Ideas (Best):**
  - [08 - Multi-Agent RL](./11-advanced-techniques-suggest/08-multi-agent-rl.md) - Ensemble specialists
  - [09 - Hierarchical RL](./11-advanced-techniques-suggest/09-hierarchical-rl.md) - Two-level selection
  - [10 - Transfer Learning](./11-advanced-techniques-suggest/10-transfer-learning.md) - Pre-training strategies
  - [11 - Online Learning](./11-advanced-techniques-suggest/11-online-learning.md) - Production adaptation
- **[12 - Implementation Roadmap](./11-advanced-techniques-suggest/12-implementation-roadmap.md)** - Prioritization guide

### [Archive](./archive/)
Historical documentation and obsolete content
- **Deprecated Docs:** [deprecated/](./archive/deprecated/)
- **Failed Experiments:** [cp-sat-experiments/](./archive/cp-sat-experiments/)
- **Old AI Suggestions:** [old-suggestions/](./archive/old-suggestions/)
- **Comparisons:** [comparisons/](./archive/comparisons/)

---

##  Quick Access by Task

### "I want to get started"
1. [Quick Reference](./QUICKREF.md)  Start here
2. [Getting Started](./01-getting-started/)
3. [UV Quick Start](./02-user-guides/UV_QUICKSTART.md)
4. [Configuration Basics](./02-user-guides/CONFIG_QUICKSTART.md)

### "I want to run the engine"
1. [Production Run Guide](./02-user-guides/PROD_RUN_GUIDE.md)
2. [Runtime Breakdown](./02-user-guides/PROD_RUNTIME_BREAKDOWN.md)
3. [Output Structure](./02-user-guides/OUTPUT_STRUCTURE_GUIDE.md)
4. [Metrics Guide](./02-user-guides/METRICS_QUICKSTART.md)

### "I want to optimize performance"
1.  [GPU Quick Start](./04-algorithms/nvidia-gpu/QUICKSTART.md) - 5 min, 3-5× speedup
2. [Time Complexity Analysis](./04-algorithms/time-complexity-algorithmic-analysis/)
3. [Parallel Processing](./04-algorithms/PARALLEL_QUICKSTART.md)
4. [Optimization Strategies](./04-algorithms/time-complexity-algorithmic-analysis/02_OPTIMIZATION_STRATEGIES.md)

### "I want to understand the architecture"
1. [RL-GA Integration](./03-architecture/rl-ga-integ-framework.md)
2. [Architecture Overview](./03-architecture/)
3. [Algorithm Details](./04-algorithms/)
4. [Implementation Notes](./06-development/implementation-notes/)

### "I want to develop/contribute"
1. [Development Docs](./06-development/)
2. [Changelogs](./06-development/changelog/)
3. [Implementation Notes](./06-development/implementation-notes/)
4. [Code Quality Guide](./06-development/code-quality/)

### "I have questions"
1. [Q&A Section](./08-qna/) - Ask technical, algorithmic, or architecture questions
2. [Troubleshooting](./01-getting-started/) (coming soon)
3. [Archive](./archive/) - Historical context

---

##  Documentation Standards

### File Organization
- **Numbered folders** (01-10) indicate logical reading order
- **README.md** in each folder for navigation
- **Categorized by purpose** not by file type
- **Archive** for historical reference (nothing deleted)

### Naming Conventions
- **UPPERCASE.md** - Important standalone docs
- **lowercase-with-dashes.md** - Detailed technical docs
- **XX_CATEGORY.md** - Numbered sections in analysis docs

### Status Indicators
-  Complete - Stable, current documentation
-  In Progress - Active development
-  Draft - Initial version
- 🗄️ Archive - Historical reference only
-  Recommended - Start here for this topic

---

##  External References

### Project Files
- Main README: [../README.md](../README.md)
- Contributing: [../CONTRIBUTING.md](../CONTRIBUTING.md)
- Todo List: [../Todo.md](../Todo.md)
- Agent Instructions: [../.github/copilot-instructions.md](../.github/copilot-instructions.md)

### Code Structure
- Source code: [../src/](../src/)
- Tests: [../test/](../test/)
- Scripts: [../scripts/](../scripts/)
- Configuration: [../configs/](../configs/)

### Scripts (in root)
- GPU Diagnostics: `scripts/diagnose_gpu.py`
- GPU Benchmark: `scripts/benchmark_gpu_training.py`
- Constraint Benchmark: `scripts/bench_constraint_check.py`

---

##  Most Important Documents

### For New Users 
1. [QUICKREF.md](./QUICKREF.md) - Essential commands (start here!)
2. [UV Quick Start](./02-user-guides/UV_QUICKSTART.md) - Package manager
3. [Production Run Guide](./02-user-guides/PROD_RUN_GUIDE.md) - How to run

### For Performance 
1. [GPU Quick Start](./04-algorithms/nvidia-gpu/QUICKSTART.md) - 5 min, 3-5× faster
2. [Time Complexity Analysis](./04-algorithms/time-complexity-algorithmic-analysis/00_EXECUTIVE_SUMMARY.md)
3. [Parallel Processing](./04-algorithms/PARALLEL_QUICKSTART.md)

### For Development 
1. [Phase 2 RL Complete](./06-development/implementation-notes/PHASE_2_RL_COMPLETE.md)
2. [Enhancement Changelog](./06-development/changelog/enhancements.md)
3. [Architecture Framework](./03-architecture/rl-ga-integ-framework.md)

### For Research 
1. [Thesis Content](./07-thesis-report/)
2. [Time Complexity Analysis](./04-algorithms/time-complexity-algorithmic-analysis/)
3. [GPU Acceleration Guide](./04-algorithms/nvidia-gpu/GPU_ACCELERATION_GUIDE.md)

---

##  How to Use This Documentation

### As a User
- Start with [QUICKREF.md](./QUICKREF.md)
- Follow [02-user-guides/](./02-user-guides/) for daily usage
- Check [04-algorithms/](./04-algorithms/) for optimization

### As a Developer
- Review [06-development/](./06-development/) for current state
- Check [03-architecture/](./03-architecture/) for design
- Read [04-algorithms/](./04-algorithms/) for implementation details

### As a Researcher
- Mine [07-thesis-report/](./07-thesis-report/) for academic content
- Study [04-algorithms/](./04-algorithms/) for analysis and performance
- Reference [archive/](./archive/) for historical context

### For Active Development
- Use [08-qna/](./08-qna/) for questions and discussions
- Track plans in [09-future-plans/](./09-future-plans/)
- Review [10-ai-suggestions/](./10-ai-suggestions/) for AI recommendations

---

##  Recent Changes

**November 18, 2025** - Documentation structure refinement:
-  Merged `05-performance/` into `04-algorithms/` (algorithms and performance are inseparable)
-  Renumbered sections 06-11 → 05-10 for consistency
-  Updated all internal links and cross-references

**November 17, 2025** - Major documentation reorganization:
-  Restructured from 30+ loose files to 10 categorized folders
-  Moved `docs/code/` → `06-development/`
-  Moved `for_report/` → `07-thesis-report/`
-  Moved `suggest/` → `10-ai-suggestions/`
-  Created dedicated Q&A and Future Plans sections
-  Archived obsolete content (nothing deleted)
-  Added navigation READMEs to all folders

---

** Documentation is now organized, navigable, and maintainable!**

**Questions?** Use [08-qna/](./08-qna/) to ask anything!
