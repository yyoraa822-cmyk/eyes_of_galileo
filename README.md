# galileo_x_full

Runnable snapshot for the final GCD experiments.

## Included

- `galileo/run_gcd.py` (final GCD runner)
- `galileo/run_dsl_unified.py` (DSL pipeline used by GCD)
- `galileo/providers.py`
- `galileo/dsl/`
- `galileo/goals/`
- `galileo/scenarios/`
- `galileo/visual_mujoco/`
- `run_all_gcd.sh` (one-click serial batch runner)

## Quick start

```bash
cd /path/to/galileo_x_full
export GALILEO_API_BASE="https://api.example.com/v1"
export GALILEO_API_KEY="<YOUR_API_KEY>"
export MODEL="claude-opus-4-7"   # or gpt-5.5
export MODE="with"                # with | no
bash run_all_gcd.sh
```

## Useful options

- `MODE=no` to run no-instrument baseline.
- `SLEEP_SECS=15` to control interval between scenes.
- `OUT_DIR=/path/to/output` to override result directory.
- `PYTHON_BIN=/path/to/python` to pin interpreter.
