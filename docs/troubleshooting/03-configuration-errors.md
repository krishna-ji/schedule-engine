# Configuration Errors

Diagnose YAML and runtime-mode issues before they derail long experiments.

## 1. Validation Pipeline

```powershell
uv run show-config --mode rl --env prod
uv run verify-config --config configs/custom/my.yaml
```

Outputs highlight missing fields, invalid enums, and type mismatches.

## 2. Frequent Mistakes

| Error | Explanation | Fix |
| --- | --- | --- |
| `KeyError: rl` | Mode requires RL but config disabled | Ensure `rl.enabled = true` for modes 5,7-10 |
| `ValidationError: pop_size not divisible by batch_size` | GPU evaluator requirement | Adjust `ga.pop_size` or `evaluator.gpu.batch_size` |
| `RuntimeModeValidationError` | Mode-specific killswitch combo invalid | Run `uv run list-modes --validate` after editing configs |
| `yaml.scanner.ScannerError` | Tabs or invalid indentation | Convert tabs to spaces, use YAML linting |
| `ValueError: probability out of range` | Heuristic probability not in [0,1] | Normalize values in config |

## 3. Debugging Strategy

1. Validate base + env config before layering runtime mode.
2. Use `uv run show-config --diff` to compare against base.
3. Add comments inline but keep them on separate lines (YAML can't mix comment + value easily).
4. Leverage anchors/aliases sparingly to avoid merge surprises.

## 4. Schema Extensions

When adding new config fields:
- Update corresponding Pydantic model (`src/config/models/*.py`).
- Provide defaults to avoid breaking existing modes.
- Add to `docs/get-started/02-setup.md` if user-facing.

## 5. Advanced Tools

- `python scripts/diagnostics/dump_config.py --mode full --env prod --output tmp/config.json`
- `uv run validate-mode --mode rl` to ensure mandatory toggles.

Keeping configs validated upfront saves hours of wasted compute.
