# Future Directions: A Learning-Based Hyper-Heuristic Framework for University Timetabling

**Document Version: 2.0**
**Date: 2025-11-14**

## 1. Executive Summary: From Metaheuristic to Hyper-Heuristic

The current NSGA-II based system provides a robust and effective solution to the University Course Timetabling Problem (UCTP). However, analysis—particularly the conclusive failure of a pure Constraint Programming (CP-SAT) approach on the global problem—demonstrates that no single, monolithic algorithm is optimal. The scale and complexity of the UCTP demand a more adaptive and intelligent approach.

This document outlines a strategic evolution from the current metaheuristic solver to a **Reinforcement Learning (RL) based Hyper-Heuristic framework**. This advanced architecture will leverage the existing Genetic Algorithm (GA) and integrate a CP-SAT solver as a specialized tool, all orchestrated by an RL agent that learns an optimal, problem-aware optimization strategy. This hybrid approach represents the state-of-the-art in solving complex combinatorial optimization problems.

The goal is to create a system that dynamically selects the best heuristic (e.g., a GA mutation, a crossover, or a computationally expensive CP-based repair) at each stage of the search process, maximizing solution quality while managing computational resources.

---

## 2. Proposed System Architecture: The Conductor and the Orchestra

The proposed architecture reframes the problem. Instead of a single solver, we will have a toolbox of operators (the "orchestra") and an intelligent agent that decides which tool to use and when (the "conductor").

![System Architecture Diagram](https://i.imgur.com/your-diagram-url.png)  <!-- Placeholder for a real diagram -->

**Core Components:**

1.  **Timetabling Environment:** An abstraction layer that encapsulates the current state of the schedule (the `individual` in GA terms). It exposes methods to apply heuristics and evaluate the resulting state.
2.  **Heuristic Toolbox (Action Space):** A collection of diverse operators that can modify a schedule. This is the "action space" for the RL agent.
    *   **Global Search Operators (GA-based):** `crossover_course_group_aware`, `mutate_individual`.
    *   **Local Search Operators:** Simple swaps, moves, or room changes.
    *   **Large Neighborhood Search (LNS) Operators:**
        *   *Destroy Operators:* `destroy_conflicted_sessions`, `destroy_random_sessions`, `destroy_instructor_sessions`.
        *   *Repair Operators:* `repair_greedy`, `repair_with_ga`.
    *   **High-Intensity Operator:** `LNS-CP_Repair` (the surgical tool).
3.  **Evaluation Engine:** The existing fitness function, which calculates hard and soft constraint violations. This provides the feedback (reward) needed for learning.
4.  **RL-based Hyper-Heuristic (The Agent):** The core of the new system. This agent observes the state of the environment and selects the next heuristic to apply from the toolbox.
    *   **State Representation (S):** A feature vector describing the current solution, e.g., `[num_hard_violations, num_soft_violations, fitness_improvement_delta, iterations_since_improvement]`.
    *   **Policy (π(a|s)):** The learned strategy. Given the state `s`, the policy outputs a probability distribution over the available actions (heuristics) `a`.

---

## 3. Algorithmic Deep Dive

### Algorithm 1: The Main RL-Driven Hyper-Heuristic Loop

This algorithm replaces the static `eaSimple` or `eaMuPlusLambda` loop from DEAP.

```pseudocode
function RL_HyperHeuristic_Solve(initial_schedule, max_iterations):
    // Initialization
    env = TimetablingEnvironment(initial_schedule)
    rl_agent = initialize_RL_Agent(action_space_size=len(HeuristicToolbox))

    best_solution = initial_schedule

    // Main Optimization Loop
    for i in 1..max_iterations:
        // 1. Observe State
        current_state_vector = env.get_state_representation()

        // 2. Select Action (Heuristic) using RL Policy
        action_index = rl_agent.choose_action(current_state_vector)
        selected_heuristic = HeuristicToolbox[action_index]

        // 3. Apply Action and Get New State
        old_fitness = env.get_fitness()
        new_schedule = env.apply_heuristic(selected_heuristic)
        new_fitness = env.get_fitness()

        // 4. Calculate Reward
        reward = calculate_reward(old_fitness, new_fitness)

        // 5. Update RL Agent (Learn from the experience)
        next_state_vector = env.get_state_representation()
        rl_agent.learn(current_state_vector, action_index, reward, next_state_vector)

        // 6. Update Best Solution
        if new_fitness is better than best_solution.fitness:
            best_solution = new_schedule

    return best_solution
```

### Algorithm 2: The LNS-CP Operator (The "Surgical Repair" Action)

This is the most powerful heuristic in the toolbox. It leverages the key insight from your CP-SAT failure analysis: CP is effective on small, well-defined subproblems.

```pseudocode
function LNS_CP_Repair(schedule):
    // 1. Destroy Phase
    // Identify a subset of "problematic" sessions to remove.
    // Example: all sessions involved in hard constraint violations.
    sessions_to_remove = find_conflicting_sessions(schedule)

    // Create a partial schedule by removing these sessions.
    partial_schedule = schedule.remove(sessions_to_remove)

    // Define the search space for the repair.
    // This includes the time slots and rooms freed up by the removed sessions.
    available_slots, available_rooms = get_available_resources(partial_schedule, sessions_to_remove)

    // 2. Repair Phase (with CP-SAT)
    // Create a new, small CP-SAT model.
    cp_model = new CpModel()

    // Create variables ONLY for the removed sessions.
    // Variables: start_time, room_assignment for each session in sessions_to_remove.
    // Domains: The available_slots and available_rooms.
    subproblem_vars = create_cp_variables(cp_model, sessions_to_remove, available_slots, available_rooms)

    // Add constraints for the subproblem.
    // These constraints must ensure the repaired sessions do not conflict with each other
    // OR with the fixed sessions in the partial_schedule.
    add_subproblem_constraints(cp_model, subproblem_vars, partial_schedule)

    // Add soft constraints as optimization objectives for the subproblem.
    add_subproblem_objectives(cp_model, subproblem_vars)

    // 3. Solve the Subproblem
    solver = CpSolver()
    status = solver.Solve(cp_model)

    // 4. Re-integrate Solution
    if status is OPTIMAL or FEASIBLE:
        repaired_schedule = reintegrate_solution(partial_schedule, solver.get_solution(subproblem_vars))
        return repaired_schedule
    else:
        // If CP solver fails on the subproblem, it's truly difficult.
        // Return the original schedule, the action had no effect.
        return schedule
```

---

## 4. Comparative Analysis of Methodologies

| Methodology | Strengths | Weaknesses | Role in Final System |
| :--- | :--- | :--- | :--- |
| **Pure GA (Baseline)** | Good global search; robust; effective at soft constraint optimization. | Can get stuck in local optima; performance is sensitive to operator choice and parameters. | **Foundation.** Provides the core population management and global search operators. |
| **Pure CP-SAT** | Guarantees optimality/feasibility (on small problems); excellent for highly constrained problems. | **Intractable** on the global problem due to constraint explosion. Fails completely. | **Specialized Tool.** Used only within the LNS-CP operator to solve small, localized subproblems with mathematical precision. |
| **GA + LNS/CP Hybrid** | **State-of-the-art.** Combines GA's global search with CP's exact local search. Powerful at escaping local optima and fixing hard constraints. | More complex to implement than a pure GA. Performance depends on the quality of the destroy/repair heuristics. | **The "Power-Play" Heuristic.** This becomes the most potent action in the RL agent's toolbox. |
| **RL Hyper-Heuristic** | **Adaptive.** Learns the best optimization strategy for the problem at hand. Can outperform any single, fixed strategy. Dynamically balances exploration and exploitation. | Highest implementation complexity. Requires careful design of state, action, and reward. Training can be time-consuming. | **The "Conductor".** The high-level intelligence that orchestrates all other heuristics to achieve the best possible result. |

---

## 5. Phased Implementation Roadmap

This project should be developed and benchmarked in phases to isolate the contribution of each new component.

### Phase 1: Implement and Benchmark the LNS-CP Operator

1.  **Goal:** Prove that a hybrid GA+LNS/CP model outperforms the baseline GA.
2.  **Steps:**
    a.  Create the `LNS_CP_Repair` function (Algorithm 2). This requires integrating `ortools`.
    b.  Modify the main GA loop to call this operator periodically (e.g., every 10 generations, or on individuals that have stagnated).
    c.  **Benchmark:** Run the baseline GA vs. the new GA+LNS/CP hybrid on the standard dataset.
    d.  **Expected Outcome:** The hybrid model should find solutions with fewer hard constraint violations and/or find them faster. This is a significant research contribution on its own.

### Phase 2: Develop the RL Environment and a Simple Agent

1.  **Goal:** Build the hyper-heuristic framework and train a basic agent.
2.  **Steps:**
    a.  Create the `TimetablingEnvironment` class, wrapping your scheduling logic.
    b.  Define the `state` vector, `action` space (the toolbox of heuristics, including the new LNS-CP one), and `reward` function.
    c.  Implement a simple, table-based RL agent (e.g., Q-Learning). This will require discretizing the state space.
    d.  Implement the main hyper-heuristic loop (Algorithm 1).
    e.  **Benchmark:** Compare the Q-Learning agent's performance against the baseline GA and a "random choice" hyper-heuristic.
    f.  **Expected Outcome:** The Q-Learning agent should learn a non-trivial policy that outperforms random selection, demonstrating the value of a learning-based approach.

### Phase 3: Scale to a Deep Reinforcement Learning Agent

1.  **Goal:** Overcome the limitations of a discrete state space by using a neural network.
2.  **Steps:**
    a.  Replace the Q-table with a Deep Q-Network (DQN) using a library like PyTorch or TensorFlow.
    b.  The DQN will take the continuous state vector as input and output Q-values for each action (heuristic).
    c.  Implement an experience replay buffer to stabilize training.
    d.  **Benchmark:** Compare the DQN agent against all previous models.
    e.  **Expected Outcome:** The DQN agent should develop the most sophisticated and effective policy, achieving the best overall results, especially on diverse and challenging problem instances.

By following this roadmap, you will systematically build upon your existing work to create a truly novel and powerful optimization system, with clear, publishable results at each stage of development.
