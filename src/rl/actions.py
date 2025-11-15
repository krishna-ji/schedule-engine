"""
Heuristic action space for RL agent.

Defines the toolbox of optimization operators that the RL agent can select
and apply to modify the current schedule.

Action Space (6 core actions):
0. mutate_session_time: Low intensity, fast (~1ms)
1. mutate_session_room: Low intensity, fast (~1ms)
2. crossover_one_point: Medium intensity (~5ms)
3. LNS_destroy_random_10pct: Medium intensity (~50ms)
4. LNS_destroy_conflicted: High intensity (~100ms)
5. LNS_IGLS_Repair: Very high intensity, surgical (~500ms-10s)

TODO (Phase 2):
- Implement Heuristic base class
- Implement each action class
- Create HEURISTIC_TOOLBOX registry
- Add cost tracking for reward function
"""

# Placeholder for Phase 2 implementation
