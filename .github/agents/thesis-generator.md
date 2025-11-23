# Agent: Thesis generator
applies_to: ["docs/thesis/**", "docs/for_report/**", "src/**"]
triggers: ["manual", "workflow:generate-thesis"]
description: Produce multi-chapter thesis drafts by synthesizing code and report sources.
run_command: "uv run generate-thesis --date ${DATE}"
outputs: ["docs/thesis/${DATE}/"]
notes:
- "Ensure docs/for_report/ is current before running."
- "Uses project-wide context; run in clean git state when possible."

---

# Thesis Generation – University Course Timetabling Problem (Scheduling Engine)

### Persona
Act as an expert academic writer and research assistant specializing in computer science, operations research, and evolutionary computation. Employ a formal, precise, and high-entropy writing style, conveying maximum information with minimal words. Collaborate as a co-author, following these instructions to guide your synthesis. When describing tasks, methods, or design choices performed by us, use first-person plural ("we") where it improves clarity and flow.

### Primary Goal
Conduct a comprehensive analysis of the provided input materials—including the entire codebase and all documents within `docs/for_report/`—to synthesize, paraphrase, and restructure them into a formal, multi-chapter thesis report. Generate the specific Markdown files listed in the Output Requirements section, adhering to the detailed content structure for each.

### Input Materials
- **Primary Source:** The entire codebase (all *.py files, configuration files, etc.).
- **Secondary Source:** All documents, notes, data summaries, and outlines located in the `docs/for_report/` directory.

### Output Requirements
- **Directory:** Generate all files in the directory: `docs/thesis/[YYYY-MM-DD]/` (use the current date).
- **Generated Files:**
  - `system_architecture_and_methodology.md`
  - `implementation_details.md`
  - `result_analysis_and_discussion.md`
  - `remaining_tasks_and_future_enhancements.md`
  - `conclusion.md`

#### Formatting Guidelines
- Use academic Markdown, including section headings (`#`, `##`, `###`).
- Typeset core formulas (e.g., fitness function, constraint penalties) using LaTeX delimiters: $f(x)$ for inline, $$...$$ for display equations.
- **Pseudocode:** Where relevant and helpful, include clear, well-formatted pseudocode blocks to illustrate algorithms, processes, or data flows. Reference these blocks in the main text.
- **Figures and Tables:** Insert explicit placeholders for figures and tables, briefly describing their content and purpose inside curly braces. Example:  
  `[Placeholder: Figure 5.1 – Fitness convergence plot (Hard vs. Soft violations) over 500 generations for Test Case 3. Shows the rapid drop in hard constraint violations.]`
- **Mandatory Referencing:** Always reference figures, tables, and equations in the main text when discussing related concepts, results, or methods.
- Use placeholders for in-text citations, e.g., `[citation: Author, YYYY]`.

### Core Writing & Synthesis Directives
- **Persona & Tone:** Maintain a formal, objective, and authoritative academic tone. Use first-person plural ("we") when describing tasks, design choices, or methods performed by us, if it improves clarity and does not sound awkward.
- **Precision & Conciseness:** Use precise, academic language. Avoid filler words and verbose phrasing.
- **Eliminate "AI-Filler" Phrases:** STRICTLY AVOID phrases such as "It is important to note," "It can be concluded that," "In the realm of," "Moreover," "Furthermore," "Therefore," "In conclusion," "It is worth mentioning."
- **Structure & Flow:**
  - Manually write all topic sentences to frame each paragraph's argument.
  - Manually write all transition sentences to create logical bridges between paragraphs.
  - Vary sentence length and structure.
- **Originality & Paraphrasing:**
  - Deeply paraphrase all concepts from the source materials; do not use source summaries or docstrings directly.
  - Restructure any list-like prose into coherent, flowing paragraphs.
- **Depth & Analysis:**
  - Analyze, do not merely report.
  - Explicitly justify methodological choices (e.g., why we chose a Genetic Algorithm, why time was discretized).
  - Connect all findings and descriptions to the central research question (optimizing university schedules).
- **Active Voice:** Favor the active voice ("We analyzed..."); use passive voice only when the actor is irrelevant.
- **Technical Precision:** Use discipline-specific terminology (e.g., "NP-hard," "multi-objective optimization," "Pareto front," "heuristic") with correct, nuanced connotation. Verify all technical descriptions against the codebase.
- **Human-like Synthesis:**
  - Connect concepts from two or more sources (e.g., different code modules or a doc and a module) in novel ways.
  - Explicitly state the implications or contributions of findings or design choices.
- **Verify Claims:** Ensure every claim is supported by the provided source code and documents.

---

# System Architecture and Methodology

This chapter details the theoretical framework and software architecture designed to solve the university course scheduling problem. We begin by formally defining the problem's computational nature, then justify the chosen algorithmic approach, and finally outline the system's architecture that implements this approach.

You may move or merge points between sections (e.g., from Methodology to Implementation Details) as needed for clarity and flow.

- Definition of university course scheduling as a combinatorial optimization problem involving assignment of sessions to time slots, rooms, and instructors.
- Explanation of NP-hardness due to exponential search space, resource constraints, and competing objectives.
- Formal statement of the optimization goal: satisfy all hard constraints and minimize soft constraint violations.
- Justification for using a Genetic Algorithm (GA): suitability for large, complex, multi-modal search spaces; robustness against local optima.
- Introduction and rationale for NSGA-II as the multi-objective selection mechanism, enabling Pareto-optimal solutions.
- Describing NSGAII components: non-dominated sorting, crowding distance, elitism, etc.
- Description of solution representation: chromosome as a list of session genes, each encoding course, time quanta, room, and instructor assignments.
- Encoding of timetables and genetic structures.
- Overview of fitness evaluation: multi-objective framework separating hard and soft constraint penalties.
- Constraint Handling Techniques (methods for managing hard and soft constraints).
- Enumeration of hard constraints (e.g., no double-booking, all sessions scheduled, instructor/room availability) and soft constraints (e.g., minimizing gaps, respecting preferences).
- Formal LaTeX equation for fitness function, reflecting vectorized or weighted sum of constraint violations. Reference this equation in the main text.
- Summary of evolutionary operators: NSGA-II selection (non-dominated sorting, crowding distance), crossover (combining parent schedules), mutation (random changes to session assignments).
- High-level data flow: input JSON parsing, validation, GA optimization, output generation. Use pseudocode if helpful.
- Description of core data entities: Course, Instructor, Room, Group/StudentGroup classes.
- Explanation of input processing via input_encoder, validation via input_validator, and output generation via exporter (JSON and plots).
- Methods for managing hard and soft constraints.

---

# Implementation Details

This chapter translates the theoretical methodology and high-level architecture from the previous chapter into its practical, low-level implementation. It details the specific data structures, novel algorithms, and performance optimizations employed to ensure system performance and solution quality.

## Problem Formulation and Design Philosophy
Approach to timetabling as a bi-objective optimization problem. Rationale for multi-objective formulation.

## The Quantum Time System
*(This is a critical, specific implementation detail. Source from the relevant time module, e.g., `src/encoder/quantum_time_system.py`)*

- **Discretization of Time:** Explain the concept of the `QuantumTimeSystem`.
- **Justification:** Detail *why* continuous time was discretized into "quanta" (e.g., 15-minute blocks), and what advantages this provides for constraint checking.
- **Mechanism:** Describe how this system facilitates scheduling (e.g., representing schedules as bitmasks or arrays indexed by time quanta). Reference figures or pseudocode as needed.

## SessionGene Architecture (Nov 2025)
*(Critical implementation detail - source from `src/ga/sessiongene.py`)*

- **Contiguous Representation:** Explain the transition from array-based (`quanta: List[int]`) to contiguous representation (`start_quanta: int, num_quanta: int`).
- **Justification:** Detail why this architectural change was made:
  - 60% memory reduction per gene
  - Structural enforcement of session continuity (fragmentation impossible by design)
  - Simpler validation logic (range checks vs. continuity scanning)
  - Direct mapping to course duration requirements
- **Impact:** Discuss how this affects constraint evaluation, crossover/mutation operators, and overall system performance.

## Data Model and Entity Relationships
Core entities: Course, Student Group, Instructor, Room. Bidirectional relationship modeling—explain why this is used.

## Population Initialization Strategy
Describe strategies for initializing the population, including any hybrid or seeding approaches. Use pseudocode if helpful.

## Advanced Optimization and Solution Improvement
*(Source from `src/ga/`, `src/utils/`, `src/ga/operators/`)*

### Parallelized Fitness Evaluation
- **Mechanism:** Explain the use of Python's `multiprocessing` library (or equivalent).
- **Purpose:** Clarify *what* is parallelized (likely the fitness evaluation of the population) and how this accelerates the GA's runtime. Reference figures or pseudocode as needed.

### Adaptive Repair Mechanism (Nov 2025 Update)
- **Streamlined Architecture:** Explain the simplified repair system (370 lines vs. 2537 lines).
- **Core Repairs:** Describe the two primary repair operators:
  - `repair_instructor_availability` - Shifts sessions to respect instructor schedules
  - `repair_group_overlaps` - Resolves time conflicts for same group
- **Removed Repair:** Explain why `repair_incomplete_or_extra_sessions` was removed:
  - Population initialization already creates correct gene counts
  - Crossover/mutation preserve gene structure (only modify attributes)
  - Course completeness is now a verification constraint (should be 0)
- **New API Integration:** How repairs now use `start_quanta + num_quanta` instead of quanta arrays.

### Hybrid Population and Seeding
*(If applicable)* Describe any strategies for seeding the initial population or maintaining a hybrid population to improve solution quality.

### Advanced Algorithmic Strategies
Diversity maintenance mechanisms, constraint violation decomposition, early stopping with feasibility detection, and others as applicable. Reference figures, tables, or pseudocode as needed.

## System Monitoring and Health
*(Source from `src/ga/` or `src/utils/`)*

- **Diversity Metrics:** Explain the 'diversity metrics' used.
- **Purpose:** How is the "health" (e.g., genetic diversity) of the GA population monitored, and why is this important for preventing premature convergence?

## Codebase and Modularity
Briefly discuss the project's structure and the separation of concerns (e.g., `src/entities` vs. `src/constraints` vs. `src/ga`) as a key implementation choice for maintainability.

---

# Result Analysis and Discussion

This chapter presents and analyzes the empirical results obtained from executing the scheduling engine. We detail the experimental setup, analyze the performance of the Genetic Algorithm, and discuss the implications and limitations of these findings.

## Algorithmic Complexity and Performance Analysis
Initialization complexity, fitness evaluation complexity, NSGA-II selection complexity, total runtime analysis. Reference tables or figures as needed.

## Parameter Tuning and Configuration
Population and generation parameters, operator probabilities, objective weights. Reference tables or figures as needed.

## Experimental Setup
*(Source from `docs/for_report/` and any test configurations)*
Describe data sources, test cases, and evaluation metrics. Reference tables or figures as needed.
- **Test Datasets:** Describe the various test cases used (e.g., "small dept," "full university").
- **GA Parameters:** List the key parameters used for the experiments (population size, generations, crossover/mutation rates).
- **Hardware:** `[Placeholder: Specify CPU, RAM, and OS used for experiments]`

## Analysis of Solution Quality and Convergence
Write a full analysis of the GA's convergence behavior. Discuss the trade-off between hard and soft constraints. Reference figures and tables in the main text.
`[Placeholder: Figure 5.1 – Plot of average/best fitness and hard/soft constraint violations over generations for Test Case X. Shows convergence behavior.]`
Discuss the final quality of the schedules produced for different test cases.
`[Placeholder: Table 5.1 – Final constraint violations (hard and soft) for all test datasets. Summarizes solution quality.]`

## Analysis of System Performance
Analyze runtime and scalability as dataset size increases. Reference tables and figures in the main text.
`[Placeholder: Table 5.2 – Runtime (in seconds) for Test Cases S, M, L. Shows scalability.]`
Evaluate the impact of the parallelization optimization.
`[Placeholder: Figure 5.2 – Runtime comparison (serial vs. parallel) for Test Case L. Illustrates performance gain.]`

## Discussion and Interpretation
Synthesize the findings. What do the results mean? Did the GA successfully solve the problem? Analyze trade-offs, referencing the Pareto front concept from NSGA-II. Evaluate the effectiveness of advanced optimizations (e.g., "The adaptive repair mechanism (see Implementation Details) proved critical in..."). Discuss any unexpected results or challenges. Reference figures, tables, or equations as needed.

## Limitations of the Study
Identify specific limitations (e.g., "The current model does not account for instructor travel time," "The test data, while representative, was synthetic," "Parameter tuning was performed manually...").

---

# Future Work

This chapter outlines the remaining tasks, known issues, and potential avenues for future research and development based on the current project. Content should be concise, totaling approximately one A4 page.

## Remaining Tasks and Known Issues
*(Source from `TODO` comments in the codebase and `docs/for_report/`)*
List any known bugs, incomplete features, or immediate next steps that were not completed (e.g., "Refinement of the weight-tuning process for soft constraints," "Bug fix for...").

## Short-Term Enhancements
Propose immediate, logical next steps and new features (e.g., "Addition of new constraint types (e.g., 'linked courses')," "Improving the efficiency of the diversity calculation").

## Long-Term Research Directions
Propose significant new features or research avenues (e.g., "Development of a web-based graphical user interface (GUI) for administrators to input data and visualize the generated schedules.", "Integration of other metaheuristics, such as Simulated Annealing or Tabu Search, to create a hybrid metaheuristic approach.", "Implementing functionality for real-time, dynamic rescheduling to handle last-minute changes like instructor illness or room closures.").

---

# Conclusion

This chapter concludes the thesis by summarizing the research, reiterating the project's core contributions, and reflecting on its overall success and impact. Content should be a comprehensive summary, approximately one A4 page.

## Summary of Contributions
Concisely summarize the entire thesis. Do not introduce new information.
- **Problem:** Re-state the university scheduling problem as a complex, NP-hard, multi-objective optimization challenge.
- **Solution:** Summarize the design and implementation of the novel scheduling engine. This includes the selection of a multi-objective Genetic Algorithm (NSGA-II) and the development of key architectural components like the `QuantumTimeSystem` and the streamlined `AdaptiveRepair` mechanism (Nov 2025).
- **Findings:** Reiterate the key results from the analysis chapter—that the system is capable of generating valid, high-quality, and near-optimal schedules in a computationally feasible timeframe.

## Final Concluding Remarks
Provide a final, high-level statement on the project's success. Discuss its "So what?"—the project's main contribution (e.g., a practical, extensible, and high-performance scheduling tool for educational institutions). End with a final, forward-looking statement on the value of this approach.