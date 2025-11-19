# Schedule Engine Documentation Index

**Project:** University Course Timetabling Problem (UCTP) Solver  
**Last Updated:** November 17, 2025

---

##  Quick Navigation

### Getting Started
- [Quick Reference](./QUICKREF.md) - Essential commands and daily usage
- [UV Migration Guide](./UV_QUICKSTART.md) - Modern package management
- [Virtual Environment Setup](./VENV_SETUP.md) - Traditional venv setup
- [Configuration Guide](./CONFIG_QUICKSTART.md) - YAML config basics

### Running the Engine
- [Production Run Guide](./PROD_RUN_GUIDE.md) - Best practices for production runs
- [Production Runtime Breakdown](./PROD_RUNTIME_BREAKDOWN.md) - Performance expectations
- [Configuration Visual Guide](./CONFIG_VISUAL_GUIDE.md) - Config structure explained
- [Output Structure Guide](./OUTPUT_STRUCTURE_GUIDE.md) - Understanding results

### Core Features
- [Heuristics Quick Reference](./HEURISTICS_QUICKREF.md) - Available GA operators
- [Metrics Quick Start](./METRICS_QUICKSTART.md) - Understanding fitness metrics
- [Parallel Processing](./PARALLEL_QUICKSTART.md) - Multiprocessing setup

### Advanced Topics
- [LNS Production Run Guide](./LNS_PROD_RUN_GUIDE.md) - Large Neighborhood Search
- [CP-SAT Production Testing](./CP_SAT_PROD_TESTING_GUIDE.md) - Constraint programming integration
- [Block Clustering Config](./BLOCK_CLUSTERING_CONFIG.md) - Time block optimization

### Performance & Optimization
- **[Time Complexity Analysis](./time-complexity-algorithmic-analysis/)**  **NEW**
  - [Executive Summary](./time-complexity-algorithmic-analysis/00_EXECUTIVE_SUMMARY.md)
  - [Full Complexity Analysis](./time-complexity-algorithmic-analysis/01_COMPLEXITY_ANALYSIS.md)
  - [Optimization Strategies](./time-complexity-algorithmic-analysis/02_OPTIMIZATION_STRATEGIES.md)
  - [Benchmark Guide](./time-complexity-algorithmic-analysis/03_BENCHMARK_GUIDE.md)
- **[GPU Acceleration Guide](./nvidia-gpu/)**  **NEW**
  - [Quick Start (5 min)](./nvidia-gpu/QUICKSTART.md)
  - [Comprehensive Guide](./nvidia-gpu/GPU_ACCELERATION_GUIDE.md)
  - Diagnostics: `scripts/diagnose_gpu.py`
  - Benchmark: `scripts/benchmark_gpu_training.py`

---

##  By Category

### Configuration & Setup

| Document | Description | Audience |
|----------|-------------|----------|
| [UV Migration](./UV_MIGRATION.md) | UV package manager migration details | Developers |
| [UV Quick Start](./UV_QUICKSTART.md) | Quick UV commands | All users |
| [Config Quick Start](./CONFIG_QUICKSTART.md) | Basic configuration | New users |
| [Config Visual Guide](./CONFIG_VISUAL_GUIDE.md) | Visual config explanation | Visual learners |
| [Config Standalone](./CONFIG_STANDALONE.md) | Standalone config system | Advanced users |
| [Config Refactoring](./CONFIG_REFACTORING.md) | Config system changes | Developers |

### Algorithms & Implementation

| Document | Description | Status |
|----------|-------------|--------|
| [Phase 1.5 Summary](./PHASE_1.5_SUMMARY.md) | Heuristic toolbox implementation |  Complete |
| [Phase 2.1 Summary](./PHASE_2.1_SUMMARY.md) | RL environment implementation |  Complete |
| [Implementation Complete](./IMPLEMENTATION_COMPLETE.md) | Feature completion status |  Complete |
| [Implementation Summary](./IMPLEMENTATION_SUMMARY.md) | Development summary | Archive |

### Performance & Benchmarking

| Document | Description | Focus |
|----------|-------------|-------|
| [Time Complexity Analysis](./time-complexity-algorithmic-analysis/) | Full algorithmic analysis | Constraint checking |
| [GPU Acceleration Guide](./nvidia-gpu/) | CUDA/NVIDIA GPU enablement | RL training speedup |
| [Production Optimization](./PRODUCTION_OPTIMIZATION_SUMMARY.txt) | Optimization notes | GA performance |
| [Prod Runtime Breakdown](./PROD_RUNTIME_BREAKDOWN.md) | Time estimates | Planning |
| [Before/After Comparison](./BEFORE_AFTER_COMPARISON.md) | Config refactoring impact | Historical |

### Validation & Testing

| Document | Description | Purpose |
|----------|-------------|---------|
| [LNS Validation](./LNS_FORCE_TRIGGER_VALIDATION.md) | LNS trigger testing | Validation |
| [CP-SAT Testing](./CP_SAT_PROD_TESTING_GUIDE.md) | Constraint programming tests | Testing |
| [Pre-Phase 2 Checklist](./PRE_PHASE2_CHECKLIST.md) | RL readiness checklist | Validation |

### Architecture & Design

| Document | Description | Audience |
|----------|-------------|----------|
| [RL-GA Integration Framework](./rl-ga-integ-framework.md) | RL and GA integration design | Architects |
| [Output Structure Guide](./OUTPUT_STRUCTURE_GUIDE.md) | Result organization | All users |
| [Logging Refactor](./LOGGING_REFACTOR_SUMMARY.md) | Logging system changes | Developers |

### Historical / Archive

| Document | Status |
|----------|--------|
| [CP Removal Summary](./CP_REMOVAL_SUMMARY.md) | Archive |
| [Config Refactoring Summary](./CONFIG_REFACTORING_SUMMARY.md) | Archive |
| [CP-SAT Failures](./cp-sat-badly-failed-and-infeasible/) | Archive |

---

##  By Use Case

### "I want to run the engine"
1. Start with [UV Quick Start](./UV_QUICKSTART.md) or [Quick Reference](./QUICKREF.md)
2. Configure using [Config Quick Start](./CONFIG_QUICKSTART.md)
3. Run with [Production Run Guide](./PROD_RUN_GUIDE.md)
4. Understand output with [Output Structure Guide](./OUTPUT_STRUCTURE_GUIDE.md)

### "I want to understand performance"
1. Read [Time Complexity Analysis](./time-complexity-algorithmic-analysis/00_EXECUTIVE_SUMMARY.md)
2. Check [Production Runtime Breakdown](./PROD_RUNTIME_BREAKDOWN.md)
3. Enable [GPU Acceleration](./nvidia-gpu/QUICKSTART.md) for 3-5× training speedup
4. Profile with [Benchmark Guide](./time-complexity-algorithmic-analysis/03_BENCHMARK_GUIDE.md)

### "I want to optimize the engine"
1. **Quick Win:** Enable [GPU Acceleration](./nvidia-gpu/QUICKSTART.md) (5 min, 3-5× speedup)
2. Read [Complexity Analysis](./time-complexity-algorithmic-analysis/01_COMPLEXITY_ANALYSIS.md)
3. Review [Optimization Strategies](./time-complexity-algorithmic-analysis/02_OPTIMIZATION_STRATEGIES.md)
4. Run [Benchmarks](./time-complexity-algorithmic-analysis/03_BENCHMARK_GUIDE.md)

### "I want to develop new features"
1. Review [Implementation Summaries](./PHASE_1.5_SUMMARY.md)
2. Check [RL-GA Integration Framework](./rl-ga-integ-framework.md)
3. Understand [Configuration System](./CONFIG_REFACTORING.md)

### "I want to understand constraints"
1. Read [Constraint Complexity Analysis](./time-complexity-algorithmic-analysis/01_COMPLEXITY_ANALYSIS.md)
2. Check source: `src/constraints/hard.py`, `src/constraints/soft.py`
3. See [Metrics Quick Start](./METRICS_QUICKSTART.md)

---

##  Subdirectories

### `/code/` - Code Documentation
Detailed implementation notes and technical documentation for specific features.

**Recent Additions:**
- `PHASE_2_RL_COMPLETE.md` - RL implementation completion summary

### `/for_report/` - Thesis/Report Documentation
Academic-style documentation suitable for thesis chapters or technical reports.

### `/generated-figures-interpretation/` - Figure Analysis
Interpretation guides for generated visualization outputs.

### `/time-complexity-algorithmic-analysis/`  **NEW**
Complete algorithmic complexity analysis and optimization guide.

**Contents:**
- Executive summary
- Full Big-O analysis per constraint
- Optimization strategies with code examples
- Benchmarking and profiling guide

### `/nvidia-gpu/`  **NEW**
NVIDIA GPU acceleration guide for RL training.

**Contents:**
- 5-minute quick start guide
- Comprehensive technical guide (9 parts)
- GPU diagnostics script
- GPU training benchmark
- Configuration templates

### `/_ai__suggestions_11_17/` - AI Suggestions Archive
Historical AI-generated suggestions and analysis (November 2025).

---

##  Most Important Documents

### For New Users
1. [UV Quick Start](./UV_QUICKSTART.md) - Get started in 5 minutes
2. [Config Quick Start](./CONFIG_QUICKSTART.md) - Essential configuration
3. [Production Run Guide](./PROD_RUN_GUIDE.md) - Run your first schedule

### For Developers
1. [Time Complexity Analysis](./time-complexity-algorithmic-analysis/) - Performance deep dive
2. [Phase 1.5 Summary](./PHASE_1.5_SUMMARY.md) - Current implementation state
3. [RL-GA Integration](./rl-ga-integ-framework.md) - Architecture overview

### For Optimization
1. [GPU Quick Start](./nvidia-gpu/QUICKSTART.md) - 5 min, 3-5× speedup 
2. [Executive Summary](./time-complexity-algorithmic-analysis/00_EXECUTIVE_SUMMARY.md) - Performance analysis
3. [Optimization Strategies](./time-complexity-algorithmic-analysis/02_OPTIMIZATION_STRATEGIES.md) - Actionable proposals
4. [Benchmark Guide](./time-complexity-algorithmic-analysis/03_BENCHMARK_GUIDE.md) - Measure improvements

---

##  Documentation Standards

### File Naming
- `UPPERCASE.md` - Major documentation, quick references
- `lowercase-with-dashes.md` - Detailed technical docs
- `XX_CATEGORY.md` - Numbered sections in analysis docs

### Document Structure
- **Executive Summary** at top
- **Table of Contents** for long docs
- **Examples** with code blocks
- **References** to related docs
- **Last Updated** date

### Status Indicators
-  Complete - Stable, current
-  In Progress - Active development
-  Draft - Initial version
- 🗄️ Archive - Historical reference

---

##  External References

### Project Structure
- Main README: `../README.md`
- Contributing Guide: `../CONTRIBUTING.md`
- Todo List: `../Todo.md`
- Agent Instructions: `../.github/copilot-instructions.md`

### Code Documentation
- Source code: `../src/`
- Tests: `../test/`
- Scripts: `../scripts/`
- Configuration: `../configs/`

---

**Navigation Tip:** Use your editor's search (Ctrl+F / Cmd+F) to find specific topics in this index!
