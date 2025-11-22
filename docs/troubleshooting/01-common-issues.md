# Common Issues & Solutions

## Quick Diagnostics

```powershell
# Run comprehensive system check
uv run diagnose-system

# Check GPU availability
uv run diagnose-gpu

# Verify configuration
uv run verify-config

# Check data integrity
uv run check-data
```

---

## Installation Issues

### Issue: UV Command Not Found

**Symptoms:**
```
'uv' is not recognized as an internal or external command
```

**Solutions:**

1. **Verify UV installation:**
```powershell
# Reinstall UV
irm https://astral.sh/uv/install.ps1 | iex

# Check installation
uv --version
```

2. **Add to PATH:**
```powershell
# Add UV to PATH (Windows)
$env:PATH += ";$env:USERPROFILE\.cargo\bin"

# Verify
uv --version
```

3. **Use absolute path:**
```powershell
C:\Users\<YourUsername>\.cargo\bin\uv.exe --version
```

---

### Issue: Python Version Mismatch

**Symptoms:**
```
Python 3.11.0 is not supported. Requires Python ==3.12.*
```

**Solution:**

1. **Install Python 3.12:**
   - Download from [python.org](https://www.python.org/downloads/)
   - Verify: `python --version` should show 3.12.x

2. **Update UV to use Python 3.12:**
```powershell
# Create new venv with Python 3.12
uv venv --python 3.12

# Reinstall dependencies
uv sync --frozen
```

---

### Issue: Permission Errors (Windows)

**Symptoms:**
```
Access is denied
```

**Solution:**

1. **Run PowerShell as Administrator**

2. **Set execution policy:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

3. **Fix file permissions:**
```powershell
# Take ownership
takeown /f "C:\Users\<YourUsername>\.cargo" /r /d y

# Grant full control
icacls "C:\Users\<YourUsername>\.cargo" /grant:r "$env:USERNAME:(OI)(CI)F" /T
```

---

## Configuration Issues

### Issue: YAML Syntax Errors

**Symptoms:**
```
YAMLError: mapping values are not allowed here
```

**Solution:**

1. **Validate YAML syntax:**
```powershell
# Check syntax
uv run verify-config

# Or use Python
python -c "import yaml; yaml.safe_load(open('configs/base.yaml'))"
```

2. **Common YAML mistakes:**
```yaml
#  Wrong: Missing space after colon
ga:
  ngen:2000

#  Correct: Space after colon
ga:
  ngen: 2000

#  Wrong: Inconsistent indentation
ga:
  ngen: 2000
   pop_size: 200

#  Correct: Consistent 2-space indentation
ga:
  ngen: 2000
  pop_size: 200
```

---

### Issue: Killswitch Validation Errors

**Symptoms:**
```
Error: rl.enabled must be true for rl-guided mode
```

**Solution:**

1. **Enable required killswitch:**
```yaml
# Edit configs/prod.yaml
rl:
  enabled: true  # Set to true for RL modes
```

2. **Use compatible mode:**
```powershell
# If RL not trained, use non-RL mode
uv run full --env prod  # Mode 4: Best non-RL
```

---

### Issue: Config Not Loading

**Symptoms:**
```
FileNotFoundError: configs/base.yaml not found
```

**Solution:**

1. **Verify working directory:**
```powershell
# Check current directory
pwd  # Should be .../schedule-engine

# If not, navigate to project root
cd path\to\schedule-engine
```

2. **Verify file exists:**
```powershell
ls configs/base.yaml  # Should exist
```

---

## Data Issues

### Issue: JSON File Not Found

**Symptoms:**
```
FileNotFoundError: data/Course.json not found
```

**Solution:**

1. **Verify data files:**
```powershell
ls data/  # Should show Course.json, Groups.json, etc.
```

2. **Restore from archive:**
```powershell
# If missing, restore from backup
cp data/archive/Course.json data/
cp data/archive/Groups.json data/
cp data/archive/Instructors.json data/
cp data/archive/Rooms.json data/
```

3. **Check data integrity:**
```powershell
uv run check-data
```

---

### Issue: JSON Syntax Errors

**Symptoms:**
```
JSONDecodeError: Expecting ',' delimiter
```

**Solution:**

1. **Validate JSON:**
```powershell
# Use Python to find error
python -c "import json; json.load(open('data/Course.json'))"
```

2. **Common JSON mistakes:**
```json
//  Wrong: Trailing comma
{
  "courses": [
    {"id": "CS101"},
    {"id": "CS102"},  // ← Trailing comma
  ]
}

//  Correct: No trailing comma
{
  "courses": [
    {"id": "CS101"},
    {"id": "CS102"}
  ]
}
```

---

## GPU Issues

### Issue: CUDA Not Available

**Symptoms:**
```
[WARN] CUDA not available, using CPU fallback
```

**Solutions:**

1. **Verify NVIDIA driver:**
```powershell
nvidia-smi  # Should show GPU info
```

2. **Check PyTorch CUDA:**
```powershell
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

3. **Reinstall PyTorch with CUDA:**
```powershell
uv sync --frozen --reinstall-package torch
```

4. **Verify CUDA version:**
```powershell
python -c "import torch; print(torch.version.cuda)"
# Should print: 12.1
```

---

### Issue: GPU Out of Memory

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
```yaml
# Edit configs/prod.yaml
gpu:
  batch_size: 50  # Reduce from 100
```

2. **Reduce population size:**
```yaml
ga:
  pop_size: 100  # Reduce from 200
```

3. **Clear GPU cache:**
```python
import torch
torch.cuda.empty_cache()
```

---

## Runtime Issues

### Issue: Import Errors

**Symptoms:**
```
ModuleNotFoundError: No module named 'deap'
```

**Solution:**

1. **Reinstall dependencies:**
```powershell
uv sync --frozen
```

2. **Verify virtual environment:**
```powershell
# Check if venv active
$env:VIRTUAL_ENV  # Should show .venv path

# If not, activate
.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate  # macOS/Linux
```

---

### Issue: Slow Execution (No GPU Used)

**Symptoms:**
- Run takes 24+ hours instead of 1-2 hours
- CPU at 100%, GPU at 0%

**Solution:**

1. **Verify GPU enabled in config:**
```yaml
# Check configs/base.yaml
gpu:
  enabled: true  # Should be true
```

2. **Check GPU detection:**
```powershell
python -c "import torch; print(torch.cuda.is_available())"
# Must be True
```

3. **Monitor GPU usage:**
```powershell
# In separate terminal
watch -n 1 nvidia-smi  # Linux/macOS
# Windows: Run nvidia-smi repeatedly
```

---

### Issue: High Memory Usage

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce population size:**
```yaml
ga:
  pop_size: 100  # Reduce from 200
```

2. **Disable multiprocessing:**
```yaml
parallel:
  use_multiprocessing: false
```

3. **Close other applications**

4. **Increase system RAM** (if possible)

---

## RL Training Issues

### Issue: Model Not Loading

**Symptoms:**
```
FileNotFoundError: models/rl_agents/ppo_best.zip not found
```

**Solution:**

1. **Train RL model first:**
```powershell
uv run train-rl --timesteps 100000 --env prod
```

2. **Promote checkpoint:**
```powershell
uv run promote-model
```

3. **Verify model path in config:**
```yaml
# Check configs/prod.yaml
rl:
  model_path: "models/rl_agents/ppo_best.zip"  # Correct path
```

---

### Issue: RL Training Not Converging

**Symptoms:**
- Reward not improving after many timesteps
- Loss values exploding or staying constant

**Solutions:**

1. **Reduce learning rate:**
```yaml
# Edit configs/rl_training.yaml
training:
  learning_rate: 0.0001  # Reduce from 0.0003
```

2. **Use curriculum learning:**
```powershell
uv run train-curriculum  # Instead of train-rl
```

3. **Increase training timesteps:**
```powershell
uv run train-rl --timesteps 300000  # Increase from 100K
```

---

### Issue: Inference Too Slow

**Symptoms:**
- RL agent predictions take >10ms
- GA evolution very slow with RL enabled

**Solutions:**

1. **Check model loading:**
```python
# Model should be cached after first load
# Verify with: uv run validate-rl
```

2. **Disable RL temporarily:**
```yaml
# Edit configs/prod.yaml
rl:
  enabled: false
```

3. **Use faster fallback:**
```yaml
rl:
  inference_timeout_ms: 5  # Reduce timeout
  fallback_strategy: "random"  # Fast fallback
```

---

## Experiment Issues

### Issue: No Output Generated

**Symptoms:**
- Run completes but no output/ directory
- Missing schedule.json or calendar.pdf

**Solution:**

1. **Check for errors:**
```powershell
# Review terminal output for errors
# Check logs/schedule_engine.log
```

2. **Verify output permissions:**
```powershell
# Ensure output/ directory writable
mkdir output  # Create if missing
```

3. **Run with explicit output:**
```powershell
python main.py --env test --output output/test_run
```

---

### Issue: High Hard Violations

**Symptoms:**
- Best fitness: (-150, -42.1) after 2000 generations
- Schedule infeasible (hard violations > 0)

**Solutions:**

1. **Run longer:**
```powershell
# Increase generations
# Edit configs/prod.yaml: ngen: 5000
```

2. **Enable repair system:**
```yaml
repair:
  enabled: true
  stagnation_trigger: 50
```

3. **Use better population initialization:**
```yaml
ga:
  population_strategy: hybrid  # Best strategy
```

4. **Check data feasibility:**
```powershell
uv run check-data  # Verify problem is solvable
```

---

## Test Issues

### Issue: Tests Failing

**Symptoms:**
```
FAILED test/unit/test_module.py::test_function
```

**Solution:**

1. **Run specific test with verbose output:**
```powershell
pytest test/unit/test_module.py::test_function -v -s
```

2. **Check test dependencies:**
```powershell
# Ensure test data available
ls test/fixtures/  # Verify test data exists
```

3. **Update test fixtures:**
```python
# If data structure changed, update fixtures
# test/fixtures/sample_course.json
```

---

### Issue: Coverage Too Low

**Symptoms:**
```
Coverage: 65%  (target: 80%)
```

**Solution:**

1. **Identify uncovered code:**
```powershell
pytest --cov=src --cov-report=html test/
start htmlcov/index.html  # View coverage report
```

2. **Add missing tests:**
```python
# Focus on red areas in coverage report
# Add tests for uncovered functions
```

---

## Performance Issues

### Issue: Slow Constraint Evaluation

**Symptoms:**
- Fitness evaluation takes >1s per individual
- CPU at 100% during evaluation

**Solutions:**

1. **Enable GPU acceleration:**
```yaml
gpu:
  enabled: true
  batch_size: 100
```

2. **Enable multiprocessing:**
```yaml
parallel:
  use_multiprocessing: true
  num_workers: null  # Auto-detect cores
```

3. **Profile constraint evaluation:**
```python
# Use cProfile to identify bottlenecks
python -m cProfile -o profile.stats main.py --env test
python -m pstats profile.stats
```

---

## Getting Help

If issue persists after trying solutions:

1. **Check logs:**
```powershell
# View detailed logs
cat logs/schedule_engine.log
```

2. **Run diagnostics:**
```powershell
uv run diagnose-system > diagnostics.txt
```

3. **Create GitHub issue:**
   - Go to: https://github.com/krishna-ji/schedule-engine/issues
   - Include:
     - Error message (full stack trace)
     - Config file (if relevant)
     - Diagnostics output
     - Steps to reproduce

4. **Contact maintainer:**
   - Email: krishna-ji@example.com
   - Include diagnostics.txt and error logs

---

## See Also

- [Installation Guide](../get-started/01-installation.md) - Initial setup
- [Setup Guide](../get-started/02-setup.md) - Configuration
- [UV Commands Reference](../get-started/04-uv-commands.md) - Command help
- [Code Structure](../code/01-code-structure.md) - Understanding codebase
