# Bugfix Changelog

This file tracks bug fixes in the schedule engine codebase.

## [2025-11-19] Fixed UV dependency resolution for PyTorch CUDA builds

**Issue:** `uv sync` failed with "No solution found when resolving dependencies" for `torch==2.4.1+cu121`. The `+cu121` suffix indicates a CUDA-specific build from PyTorch's custom wheel repository, which UV couldn't locate using the default PyPI index.

**Root Cause:** PyTorch CUDA builds (e.g., `2.4.1+cu121`) are hosted on a separate index at `https://download.pytorch.org/whl/cu121`, not on the main PyPI repository. UV requires explicit index configuration to find these wheels.

**Fix:** Added PyTorch CUDA index configuration to `pyproject.toml`:
- Added `[[tool.uv.index]]` for `pytorch-cu121` pointing to PyTorch's CUDA 12.1 repository
- Set `index-strategy = "unsafe-best-match"` to allow fallback to PyPI for non-PyTorch packages
- Configured `[tool.uv.sources]` to map `torch` and `torchvision` to the custom index

**Files Modified:**
- `pyproject.toml` - Added UV index configuration for PyTorch CUDA wheels

**Result:** `uv sync` now resolves successfully and installs `torch==2.4.1+cu121` with CUDA support.

## [2025-11-19] Fixed Unicode encoding errors in Windows logging

**Issue:** Training failed with `UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'` on Windows. Unicode checkmarks (✓), warnings (⚠), and other symbols in log messages couldn't be encoded using Windows' default `cp1252` encoding.

**Root Cause:** Python logging on Windows uses the system's default encoding (`cp1252`), which doesn't support Unicode characters like ✓ (U+2713), ✗ (U+2717), ⚠ (U+26A0). These characters were used throughout the codebase for "pretty" console output, but they cause crashes when logging to stdout/stderr on Windows.

**Fix:** Replaced all Unicode symbols in logging output with ASCII-safe alternatives:
- `✓` → `[OK]`
- `✗` → `[OFF]`
- `⚠` → `[WARNING]`

**Files Modified:**
- `src/rl/training/train_script.py` - 3 checkmarks replaced
- `src/rl/deployment/model_loader.py` - Checkmark and warning replaced
- `src/rl/deployment/inference.py` - Checkmark and warning replaced
- `src/rl/gym_env/action_space.py` - Checkmark/X in action descriptions replaced

**Result:** Training now runs on Windows without encoding errors. Unicode symbols in docstrings/comments (→, etc.) are fine since they're not sent to loggers.

## [2025-11-19] Fixed duplicate 'device' parameter in RL agent creation

**Issue:** RL training failed with `TypeError: PPO() got multiple values for keyword argument 'device'` when creating agents.

**Root Cause:** The `RLTrainer.create_agent()` method in `trainer.py` adds `device` to `agent_kwargs` dictionary (line 123), which then gets passed as `**agent_kwargs` to `create_ppo_agent()`. However, `create_ppo_agent()` explicitly set `device=config.rl.agent.device`, causing the same parameter to be passed twice - once in the `**kwargs` unpacking and once explicitly.

**Fix:** Modified both `create_ppo_agent()` and `create_dqn_agent()` to extract `device` from `kwargs` using `kwargs.pop("device", config.rl.agent.device)` before passing `**kwargs` to the agent constructor. This ensures device is only set once, prioritizing the value from kwargs if provided, otherwise falling back to config.

**Files Modified:**
- `src/rl/agents/ppo_agent.py` - Extract device from kwargs before PPO constructor
- `src/rl/agents/dqn_agent.py` - Extract device from kwargs before DQN constructor

**Result:** Agent creation now handles device parameter correctly without duplication errors.

## [2025-11-19] Fixed double-wrapping of vectorized environments in RL training

**Issue:** RL training failed with `ValueError: The environment is of type SubprocVecEnv, not a Gymnasium environment` when using parallel environments (`n_envs > 1`).

**Root Cause:** The `train_script.py` creates vectorized environments (`SubprocVecEnv` or `DummyVecEnv`) for parallel training, then passes them to `create_ppo_agent()` and `create_dqn_agent()`. However, these agent creation functions unconditionally wrapped the environment in `DummyVecEnv([lambda: env])`, causing double-wrapping. Stable-Baselines3 requires environments to be vectorized, but wrapping a `VecEnv` in another `VecEnv` is invalid.

**Fix:** Added `isinstance(env, VecEnv)` checks in both `create_ppo_agent()` and `create_dqn_agent()` functions to detect already-vectorized environments and skip the wrapping step. This allows both single environments (wrapped automatically) and pre-vectorized environments (passed through) to work correctly.

**Files Modified:**
- `src/rl/agents/ppo_agent.py` - Added VecEnv import and isinstance check before wrapping (both in `create_ppo_agent()` and `load_ppo_agent()`)
- `src/rl/agents/dqn_agent.py` - Added VecEnv import and isinstance check before wrapping (both in `create_dqn_agent()` and `load_dqn_agent()`)

**Result:** Training now works with both single and parallel environments (`--n-envs 1` or `--n-envs 4+`).

## [2025-01-18] Fixed hypervolume calculation returning 0 - DEAP function misuse

**Issue:** Hypervolume indicator always returned 0 for all GA runs, making Pareto front quality assessment impossible.

**Root Cause:** DEAP's `tools.hypervolume()` is a **selector function** that returns the index of the worst contributor, not a hypervolume calculator. The function was being called with `(pareto_front, ref_point)` and treated as a metric calculator, when it actually identifies which individual to remove for hypervolume-based selection.

**Fix:** Implemented custom 2D hypervolume calculator using sweep-line algorithm with O(n log n) complexity. Algorithm sorts by first objective, calculates rectangular contributions (width × height), and sums areas.

**Files Modified:**
- `src/metrics/hypervolume.py` - Complete rewrite of `calculate_hypervolume()` function

**Verification:** Tested with 4 scenarios - single point (250.0), 3-point front (350.0), dominated solutions (correctly filtered), feasible/infeasible mix (handles both). See `docs/06-development/bugfixes/hypervolume-calculation-fix.md` for detailed technical documentation.

## [2025-01-16] Fixed "Quantum out of valid range" errors - Hardcoded constant mismatch

**Issue:** Training showed persistent warnings like "Quantum 42/43 out of valid range" when evaluating heuristic candidates during RL training. Errors occurred in `crowding_mutation`, `adaptive_large_neighborhood`, and `distance_preserving_crossover` heuristics.

**Root Cause:** **Hardcoded constant mismatch with actual configuration.**
- Code had `MAX_VALID_QUANTUM = 44` hardcoded throughout (expecting range 0-43)
- Actual `QuantumTimeSystem().total_quanta = 42` (valid range is 0-41)
- Quanta 42 and 43 were being generated by heuristics but were **out of range**
- This caused ValueError in `quanta_to_time()` during fitness evaluation

**Why This Happened:**
1. `SessionGene.quanta` contains ALL time slots occupied by session (e.g., `[38, 39, 40, 41]`)
2. Heuristics would assign valid start positions but sessions could extend to quanta 42/43
3. Hardcoded validation allowed quanta up to 43, but actual system only supports 0-41
4. During fitness evaluation, decoder tried to convert invalid quanta 42/43 to day/time format
5. `QuantumTimeSystem.quanta_to_time()` raised ValueError for out-of-range quanta

**Fix:** Replaced all hardcoded `MAX_VALID_QUANTUM = 44` constants with dynamic `QuantumTimeSystem().total_quanta` calls to match actual configuration. Now validation uses the **actual valid range** from the time system instead of assuming a fixed value.

**Files Modified:**
- `src/ga/sessiongene.py` - Fixed `__setattr__`, `_validate_and_fix_quanta()`, and `time_quantum` setter to use `QuantumTimeSystem().total_quanta`
- `src/decoder/individual_decoder.py` - Fixed pre-decoding validation to use actual total_quanta

**Result:** Training now runs with **zero quantum validation errors**. All heuristics successfully generate valid individuals.

## [2025-10-28] Enhanced memory monitoring with percentage display

**Enhancement:** Added memory percentage to the monitoring display for better visibility of system memory usage. Now shows: `2.45GiB (16.0%) Peak: 2.50GiB` instead of just `2.45GiB Peak: 2.50GiB`.

**Rationale:** Percentage provides instant context for memory usage relative to total system RAM, making it easier to see if the process is approaching memory limits.

**Files Modified:**
- `src/core/ga_scheduler.py` - Added mem_percent field to memory monitoring display

## [2025-10-28] Fixed memory monitoring missing child process memory

**Issue:** Memory usage display was only showing the main Python process memory, not including child processes spawned by multiprocessing workers. This made the displayed memory much lower than actual usage and appeared "stuck" because the main process memory stays relatively constant while workers do the heavy lifting.

**Root Cause:** The monitoring used `process.memory_info().rss` which only tracks the main process (PID). When multiprocessing is enabled, worker processes' memory was completely missing from the count.

**Fix:** Enhanced memory monitoring to include all child processes recursively. This now shows the **complete memory footprint** of the schedule engine including all worker processes.

**Files Modified:**
- `src/core/ga_scheduler.py` - Modified `update_resource_monitors()` to include child process memory

## [2025-10-27] Fixed intermittent tkinter RuntimeError during export phase

**Issue:** Intermittent `RuntimeError: main thread is not in main loop` errors during plot generation when program exits. Error occurred in tkinter's `Image.__del__` and `Variable.__del__` destructors.

**Root Cause:** Matplotlib was defaulting to TkAgg backend (GUI-based) in CLI environment. When matplotlib objects were garbage collected at program exit, tkinter destructors tried to access the main event loop which didn't exist.

**Fix:** Set matplotlib backend to 'Agg' (non-interactive, file-only) before any other matplotlib imports in `src/exporter/thesis_style.py`. The Agg backend is perfect for generating PDF/PNG files without GUI requirements.

**Files Modified:**
- `src/exporter/thesis_style.py` - Added `matplotlib.use('Agg')` before imports
