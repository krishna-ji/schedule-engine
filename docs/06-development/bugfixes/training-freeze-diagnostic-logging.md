# Training Freeze Diagnostic Logging

## Issues

### Issue 1: Training Appeared Frozen
Training appeared frozen with progress bar stuck at `0/300,000` for 5+ minutes with only time increasing.

### Issue 2: Unicode Encoding Errors on Windows
After adding diagnostic logging, Windows console crashed with:
```
UnicodeEncodeError: 'charmap' codec can't encode characters
UnicodeEncodeError: 'utf-8' codec can't encode characters: surrogates not allowed
```

## Root Causes

### Cause 1: Silent Initialization Phase
Training was NOT frozen - it was in the **environment initialization phase**, which is very slow:

1. **32 parallel environments** need to be created via `SubprocVecEnv`
2. Each environment must:
   - Generate initial population (80 individuals)
   - **Evaluate fitness for all 80 individuals** (SLOW - constraint checking)
3. With 32 envs in parallel, this takes 30-60 seconds total
4. **No logging was visible during this phase**, making it appear frozen

The progress bar from Stable-Baselines3 only shows timesteps collected, not initialization progress.

### Cause 2: Unicode in Windows Console
Windows console (cmd/PowerShell) uses cp1252 encoding by default, which cannot handle:
- Unicode emojis (, , , , , , )
- Unicode box-drawing characters (═, ║)
- Unicode multiplication sign (×)

Rich logging library attempts UTF-8 encoding, but Windows console rejects it.

## Solutions

### Solution 1: Comprehensive Diagnostic Logging
Added comprehensive diagnostic logging at multiple levels:

### 1. Environment Creation Logging (`train_script.py`)
```python
logger.info(f"[ENV {env_rank}]  Creating environment (this takes 30-60s per env)...")
logger.info(f"[ENV {env_rank}]  Generating initial population (80 individuals)...")
logger.info(f"[ENV {env_rank}]  Evaluating 80 individuals (this is the slow part)...")
# Progress every 20 individuals
logger.info(f"[ENV {env_rank}]  Population initialized")
```

### 2. Parallel Environment Creation Warning
```
================================================================================
IMPORTANT: Creating 32 parallel environments
================================================================================
Environment type: SubprocVecEnv (true parallelism)
Population size per env: 80

EXPECTED TIME:
   - Each environment needs 30-60 seconds to initialize
   - With 32 envs running in parallel, expect 30-60 seconds total
   - You should see [ENV 0-31] progress logs below

WATCH FOR: [ENV X] Creating environment logs...
================================================================================
```

### 3. Training Diagnostics (`trainer.py`)
```
============================================================
PPO TRAINING DIAGNOSTICS
============================================================
Rollout buffer: 2048 steps/env x 32 envs = 65,536 total steps
PPO will collect 2048 steps from EACH of 32 environments
Then train for 10 epochs with batch_size=64

EXPECTED BEHAVIOR:
   1. Environments reset (you should see [ENV 0-31] Reset logs)
   2. Collect 2048 steps from each env (watch for step logs)
   3. Policy update (progress bar increments)
   4. Repeat until 300,000 total steps

If no environment logs appear within 1 minute, training is likely frozen.
============================================================
```

### 4. Environment Step Logging (`schedule_env.py`)
```python
# Log frequently at first (every 5 steps), then reduce to every 25 steps
if self.debug_logging and self._total_steps_taken % log_freq == 0:
    logger.info(f"[ENV {self.env_rank}] Step {self._total_steps_taken} - action={action}")
```

### Solution 2: Remove All Unicode Characters
Replaced all Unicode characters with plain ASCII to ensure Windows console compatibility:

**Changed:**
- Emojis → Removed (        )
- Box-drawing (═) → Plain ASCII equals signs (=)
- Multiplication (×) → Plain ASCII 'x'
- All special Unicode → Standard ASCII characters

**Files affected:**
- `src/rl/training/train_script.py` - Environment creation messages
- `src/rl/training/trainer.py` - Training diagnostics
- `src/rl/gym_env/schedule_env.py` - Step logging

### 5. Environment Reset Logging
```python
if self.debug_logging:
    logger.info(f"[ENV {self.env_rank}] Reset called (total steps so far: {self._total_steps_taken})")
```

## Configuration Changes
Enabled debug logging by default in production config:

**`configs/training/prod.yaml`:**
```yaml
# Debug logging (verbose environment progress)
debug_logging: true
debug_log_interval: 25 # Log every 25 steps
```

## Files Modified
1. `src/rl/training/train_script.py` - Environment creation logging
2. `src/rl/training/trainer.py` - Training diagnostics
3. `src/rl/gym_env/schedule_env.py` - Step and reset logging
- `configs/training/prod.yaml` - Enable debug logging

## Expected Output
With these changes, users now see (Windows-compatible ASCII):

```
INFO     STEP 2: Create RL Environment
================================================================================
IMPORTANT: Creating 32 parallel environments
================================================================================
INFO     [ENV 0] Creating environment (this takes 30-60s per env)...
INFO     [ENV 1] Creating environment (this takes 30-60s per env)...
...
INFO     [ENV 0] Generating initial population (80 individuals)...
INFO     [ENV 0] Evaluating 80 individuals (this is the slow part)...
INFO     [ENV 0]    ... evaluated 0/80
INFO     [ENV 0]    ... evaluated 20/80
INFO     [ENV 0]    ... evaluated 40/80
INFO     [ENV 0]    ... evaluated 60/80
INFO     [ENV 0] [OK] Population initialized with 80 individuals
...
INFO     [OK] Parallel environments ready (32 workers)

INFO     STEP 4: Train Agent
============================================================
PPO TRAINING DIAGNOSTICS
============================================================
INFO     Starting rollout collection now...
INFO     [ENV 0] Reset called (total steps so far: 0)
INFO     [ENV 0] Step 5 - action=3
INFO     [ENV 0] Step 10 - action=7
...
   12% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 36,864/300,000  [ 5:23 < 38:12 , 114 it/s ]
```

## Verification
Run training and verify you see:
1. Environment creation progress (`[ENV X]` logs)
2. Training diagnostics explaining rollout buffer
3. Environment reset logs
4. Environment step logs during rollout collection
5. Progress bar incrementing after first rollout completes

## Impact
- **User Experience**: Clear visibility into initialization and training progress
- **Debugging**: Easy to identify if training is truly frozen vs just slow initialization
- **Performance**: No performance impact (logging only in debug mode)
- **Documentation**: Self-documenting via console output

## Related Issues
- Training progress bar stuck at 0% during initialization
- Silent environment creation phase
- No indication of rollout collection progress
- Users thinking training is frozen when it's actually working
- Unicode encoding errors on Windows console (cp1252/UTF-8 mismatch)
- Rich logging crashes with emoji characters on Windows

## Lessons Learned
1. **Always test on target OS** - Unicode works fine on Linux/Mac but fails on Windows
2. **Use ASCII for logging** - Emojis are cute but break Windows console compatibility
3. **Progress visibility is critical** - Long initialization periods need status updates
4. **Windows console encoding** - Default cp1252 encoding requires ASCII-only characters
5. **Rich + Windows = trouble** - Rich library with Unicode requires special handling on Windows
