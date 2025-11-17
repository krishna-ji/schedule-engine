# Thesis Report: CP-SAT Solver Failure Analysis

**Date:** 2025-11-14

## Executive Summary

The Google OR-Tools CP-SAT solver fails to find a solution for the university course scheduling problem due to a **quadratic explosion in constraints** originating from an inefficient modeling approach. The model generates approximately **19.7 million constraints** for a 239-course problem, a number **115 times greater** than a baseline `O(n^2)` model would predict. This is computationally intractable. The low resource utilization (15.7%) indicates the problem is not infeasible due to over-constriction, but that the search space is too vast for the CP-SAT solver to explore effectively with this model. The recommendation is to abandon the pure CP-SAT approach in favor of heuristic methods like Genetic Algorithms, which are better suited for this problem's scale and characteristics.

---

## 1. Problem Scale & Data Profile

The analysis is based on the production dataset, which has the following dimensions:

- **Courses to Schedule:** 239
- **Student Groups:** 74
- **Instructors:** 181
- **Rooms:** 67
- **Time Horizon:** 72 quanta (discrete 60-minute slots per week)

## 2. Constraint Complexity Analysis

The primary source of computational complexity in this scheduling problem is the enforcement of conflict-free assignments for shared resources (instructors and rooms).

### 2.1. Theoretical Constraint Calculation

For `N` sessions, every session must be checked against every other session for potential resource conflicts. This results in a quadratic `O(n^2)` relationship.

- **Number of Sessions (N):** 239
- **Session Pairs to Check:** `N * (N - 1) / 2`
- **Calculation:** `239 * 238 / 2 = 28,441`

This calculation applies independently to both instructor and room conflicts:
- **Instructor Conflict Pairs:** 28,441
- **Room Conflict Pairs:** 28,441
- **Total Conflict Pairs:** `28,441 + 28,441 = 56,882`

Each high-level "no conflict" rule is translated into multiple low-level constraints in the CP-SAT model, particularly when using reified boolean variables (`OnlyEnforceIf`). A conservative estimate is **3 low-level constraints per pair**.

- **Estimated Baseline Constraints:** `56,882 pairs * 3 constraints/pair ≈ 170,646`

### 2.2. Observed vs. Expected Constraints

There is a critical discrepancy between the theoretical baseline and the solver's actual model.

- **Expected Constraints (Baseline):** ~170,000
- **Observed Constraints (from Solver Log):** **19,670,746**

The model implemented in the codebase generates approximately **115 times more constraints** than a standard `O(n^2)` approach would suggest.

## 3. Root Cause: Inefficient Model Implementation

The constraint explosion is caused by the implementation within `src/ortools/constraint_factory.py`.

The methods `add_no_instructor_conflict_constraints` and `add_no_room_conflict_constraints` iterate through all `28,441` pairs for each resource type. For each pair, they create intermediate boolean variables and multiple conditional constraints to check if the resources are the same and, if so, enforce that their start times are different.

This approach is fundamentally inefficient for CP-SAT. A more idiomatic and scalable method is to group sessions by resource (e.g., a list of all sessions taught by Instructor A) and apply a single, powerful global constraint like `AddAllDifferent` to the start times of those sessions. The current implementation fails to leverage these global constraints, resulting in a massively redundant and computationally expensive model.

## 4. Resource Capacity vs. Demand

Analysis of the data shows that the problem is not infeasible due to a lack of resources.

- **Total Course Hours Required:** 759 quanta
- **Total Available Room Capacity:** `67 rooms * 72 quanta/week = 4,824` room-quanta
- **Resource Utilization:** `(759 / 4,824) * 100 ≈ 15.7%`

A utilization rate of only 15.7% indicates that the problem is **loosely constrained**. While this confirms feasibility from a capacity standpoint, it creates an enormous search space with many possible (but hard to find) solutions. This type of problem structure is better suited to heuristic and metaheuristic search algorithms than to exhaustive constraint solvers.

## 5. Conclusion and Recommendation

**The CP-SAT solver is the wrong tool for this problem at this scale, primarily due to an inefficient modeling strategy.**

1.  **Constraint Explosion:** The generation of ~19.7M constraints makes the model intractable.
2.  **Inefficient Modeling:** The `O(n^2)` pairwise checking in `constraint_factory.py` is the direct cause, failing to use CP-SAT's strengths (global constraints).
3.  **Problem Nature:** The low resource utilization (15.7%) creates a vast, sparsely populated search space where heuristic methods (like Genetic Algorithms) are more effective at finding good solutions in a reasonable timeframe.

**Final Verdict:** The solver run should be terminated. Future efforts should focus on using the existing Genetic Algorithm implementation, which is designed to handle the scale and nature of this specific scheduling problem.
