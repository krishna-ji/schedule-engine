# Adding a New GA Mode (e.g. `ga_06_xxx`)

> Checklist for adding a new GA experiment mode while keeping the architecture clean.

---

## Files to create

### 1. `runs/ga_06_xxx.py` — Thin entry point

```python
from src.experiments import XxxExperiment

SEED = 42
POP_SIZE = ...
NGEN = ...
# ... mode-specific constants ...

def main() -> None:
    exp = XxxExperiment(seed=SEED, pop_size=POP_SIZE, ngen=NGEN, ...)
    exp.run()

if __name__ == "__main__":
    main()
```

**Rules**:

- [ ] Only stdlib + one `src.experiments` import
- [ ] No pymoo, no pipeline, no plotting, no data loading
- [ ] Constants + instantiate + `.run()` only

---

## Files to modify

### 2. `src/experiments/ga_experiment.py` — New subclass

- [ ] Create `class XxxExperiment(GAExperiment)` at end of file
- [ ] Set mode defaults via `kwargs.setdefault(...)` in `__init__`
- [ ] Add mode-specific `__init__` params (keyword-only)
- [ ] Override `_build_callback(self, pkl_path)` if mode needs custom per-gen logic
  - [ ] Subclass `GACallbackBase` from `callback_core.py`
  - [ ] Override `_on_generation(self, algorithm, F, G, cv, best_idx)` only
  - [ ] Do NOT duplicate init/notify metric tracking

### 3. `src/experiments/__init__.py` — Export

- [ ] Add `XxxExperiment` to import from `.ga_experiment`
- [ ] Add `XxxExperiment` to `__all__`

---

## Tests to add

- [ ] Smoke test: `import runs.ga_06_xxx; assert hasattr(runs.ga_06_xxx, 'main')`
- [ ] Default alignment: instantiate `XxxExperiment()` with no overrides, assert expected defaults
- [ ] Callback tracking: fake algorithm/pop → verify `best_hards`, `best_softs`, `gen_times` populated
- [ ] Mode-specific hook: verify custom behavior (repair/escalation/etc.) triggers correctly

---

## Validation

- [ ] `python runs/ga_06_xxx.py` runs without error
- [ ] Output directory created with expected artefacts
- [ ] No changes needed in any other `runs/ga_0*.py` file
- [ ] No changes needed in `_execute()` or `_generate_outputs()`
- [ ] Forbidden-pattern check: `grep -E 'minimize\(|NSGA2\(|Callback|src\.pipeline|plot_' runs/ga_06_xxx.py` returns nothing

---

## Architecture health check

After adding the mode, verify:

- Adding the mode required touching **at most 3 files** (run file + subclass + `__init__.py`)
- No existing mode behavior changed
- Callback base class was reused, not copied
