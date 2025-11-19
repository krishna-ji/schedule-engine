# Schedule Engine Documentation Index

**Last Updated:** November 20, 2025  
**Version:** 2.0.0  
**Status:** Complete

---

## Quick Start

New to Schedule Engine? Start here:

1. [**Installation Guide**](get-started/01-installation.md) - Install dependencies and set up environment
2. [**Setup Guide**](get-started/02-setup.md) - Configure data and settings  
3. [**First Run**](get-started/03-first-run.md) - Run your first experiment (5 minutes)
4. [**UV Commands Reference**](get-started/04-uv-commands.md) - All available commands

**Estimated time to first run:** 15-20 minutes

---

## Get Started

### Installation & Setup
- [**01. Installation**](get-started/01-installation.md) - Prerequisites, dependencies, GPU setup
- [**02. Setup**](get-started/02-setup.md) - Configuration system, data files, validation
- [**03. First Run**](get-started/03-first-run.md) - Smoke test, understanding output, common issues
- [**04. UV Commands**](get-started/04-uv-commands.md) - Complete command reference (50+ commands)

---

## Architecture

### System Design
- [**01. High-Level Architecture**](architecture/01-high-level-architecture.md) - System overview, components, tech stack
- [**02. Design Principles**](architecture/02-design-principles.md) - Modularity, extensibility, performance, reliability
- [**03. Component Breakdown**](architecture/03-component-breakdown.md) - Responsibilities, dependencies, extension hooks
- [**04. Data Flow**](architecture/04-data-flow.md) - Complete data flow, message passing, sequence diagrams

---

## Code Documentation

### Understanding the Codebase
- [**01. Code Structure**](code/01-code-structure.md) - Directory layout, file organization, navigation tips
- [**02. Module Deep Dives**](code/02-module-deep-dives.md) - Core modules, call graphs, extension points
- [**03. Coding Standards**](code/03-coding-standards.md) - Style guide, typing, logging, testing

---

## How-To Guides

### Common Developer Tasks
- [**01. Common Developer Tasks**](how-to/01-common-developer-tasks.md) - Development workflow, adding constraints/heuristics, training RL, experiments
- [**02. Running Experiments**](how-to/02-running-experiments.md) - Thesis experiments, batch runs, comparison tooling
- [**03. Training RL Agents**](how-to/03-training-rl-agents.md) - Curriculum learning, checkpoints, promotion workflow
- [**04. Debugging & Validation**](how-to/04-debugging-and-validation.md) - Validation gates, diagnostics, RL/GPU debugging

---

## References

### Algorithms & Libraries

#### Core Algorithms
- [**01. NSGA-II Algorithm**](references/01-nsga-ii-algorithm.md) - Multi-objective GA, Pareto optimization, DEAP implementation
- [**02. Reinforcement Learning**](references/02-reinforcement-learning.md) - PPO/DQN architecture, reward shaping, safety rails
- [**03. IGLS Repair System**](references/03-igls-repair.md) - Iterative greedy local search, subproblem solving
- [**04. GPU Acceleration**](references/04-gpu-acceleration.md) - CUDA batching, config knobs, profiling tips

#### Libraries & APIs
- [**05. Library Cheat Sheet**](references/05-library-cheatsheet.md) - DEAP, SB3, PyTorch, Pydantic, Rich usage notes
- *(More detailed library guides coming soon)*

---

## Troubleshooting

### Common Issues & Solutions
- [**01. Common Issues**](troubleshooting/01-common-issues.md) - Installation, configuration, data, GPU, runtime problems
- [**02. Performance Issues**](troubleshooting/02-performance-issues.md) - Slow runs, GPU utilization, repair overhead
- [**03. Configuration Errors**](troubleshooting/03-configuration-errors.md) - YAML mistakes, schema validation, runtime modes
- [**04. RL Training Issues**](troubleshooting/04-rl-training-issues.md) - Training stability, reward plateaus, inference latency

---

## Research Papers

### Academic References
- [**Paper Index**](research-papers/00-paper-index.md) - Complete list with DOIs, methodology mappings, citation template
- *(Individual paper summaries will be added as experiments are published.)*

---

## Q&A Questions

### Technical Discussions
- [**01. Technical Questions**](qna-questions/01-technical-questions.md) - Architecture decisions, algorithm choices, design rationale
- *(Implementation and research Q&A sections planned.)*

---

## Development

### Developer-Specific Documentation
- [**01. Changelog Process**](development/01-changelog-process.md) - Where and how to document changes
- [**02. Experiment Log Template**](development/02-experiment-log-template.md) - Template for recording experimental results
- Existing folders: `docs/06-development/changelog/`, `implementation-notes/`, `bugfixes/` (see repo tree)

---

## AI Suggestions

### AI-Generated Recommendations
- [**01. Future Ideas**](ai/01-future-ideas.md) - Multi-agent RL, transfer learning, surrogate models
- [**02. Automation Opportunities**](ai/02-automation-opportunities.md) - Bots, monitors, config linters, summarizers
- *(Additional AI suggestion tracks under discussion.)*

---

## Additional Resources

### External Links
- [**GitHub Repository**](https://github.com/krishna-ji/schedule-engine)
- [**DEAP Documentation**](https://deap.readthedocs.io/)
- [**Stable-Baselines3 Docs**](https://stable-baselines3.readthedocs.io/)
- [**PyTorch Documentation**](https://pytorch.org/docs/)
- [**Pydantic Documentation**](https://docs.pydantic.dev/)

---

## Documentation Organization

### By Audience

**For Beginners:**
1. Get Started → Installation → Setup → First Run → UV Commands
2. How-To Guides → Common Developer Tasks
3. Troubleshooting → Common Issues

**For Developers:**
1. Code Documentation → Code Structure → Important Modules
2. How-To Guides → All guides
3. References → Algorithms & Libraries

**For Researchers:**
1. Architecture → High-Level Architecture → Data Flow
2. Research Papers → All papers
3. References → Algorithms (NSGA-II, RL)
4. Development → Implementation Notes

**For Contributors:**
1. How-To Guides → Adding Features → Debugging
2. Code Documentation → Coding Conventions
3. Development → Changelog → Bugfixes

---

## Quick Links

### Most Accessed
- [Installation Guide](get-started/01-installation.md)
- [First Run Guide](get-started/03-first-run.md)
- [UV Commands Reference](get-started/04-uv-commands.md)
- [High-Level Architecture](architecture/01-high-level-architecture.md)
- [Code Structure](code/01-code-structure.md)
- [Common Developer Tasks](how-to/01-common-developer-tasks.md)
- [NSGA-II Algorithm](references/01-nsga-ii-algorithm.md)

### Thesis-Relevant
- [High-Level Architecture](architecture/01-high-level-architecture.md)
- [Data Flow](architecture/04-data-flow.md)
- [NSGA-II Algorithm](references/01-nsga-ii-algorithm.md)
- Research Papers section (when complete)

---

## Status Legend

- ✅ **Complete** - Comprehensive, ready for use
- 🚧 **In Progress** - Partially complete, being updated
- 📝 **Coming Soon** - Planned, not yet started
- *(Existing)* - Previously created, may need updates

---

## Contributing to Documentation

### Adding New Documentation

1. **Choose appropriate section:**
   - Get Started: Installation, setup, first run
   - Architecture: System design, components
   - Code: Source code documentation
   - How-To: Developer guides
   - References: Algorithms, libraries
   - Troubleshooting: Common issues
   - Research Papers: Academic references
   - Q&A: Technical discussions
   - Development: Developer notes
   - AI: AI suggestions

2. **Follow naming conventions:**
   - `01-topic-name.md` (numbered for ordering)
   - `topic-name.md` (if no specific order)
   - Use kebab-case for filenames

3. **Use standard templates:**
   - Include table of contents for long docs
   - Add "See Also" section at end
   - Use consistent heading levels
   - Include code examples where relevant

4. **Update index:**
   - Add new document to this index
   - Update relevant quick links
   - Update status legend if needed

---

## Document Maintenance

**Review Schedule:**
- Get Started: Review quarterly
- Architecture: Review on major changes
- Code: Review on refactoring
- How-To: Review on feature additions
- References: Review annually
- Troubleshooting: Review monthly
- Research Papers: Review annually

**Last Full Review:** November 20, 2025

---

## Feedback

Found an issue or have a suggestion? 
- Create GitHub issue: [schedule-engine/issues](https://github.com/krishna-ji/schedule-engine/issues)
- Contact: krishna-ji@example.com

---

**Happy Coding! 🚀**
