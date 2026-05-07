#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Required externally:
#   export GALILEO_API_KEY=...
# Optional:
#   export GALILEO_API_BASE="https://api.example.com/v1"
#   export PYTHON_BIN=/path/to/python
#   export MODEL=claude-opus-4-7   # or gpt-5.5
#   export MODE=with               # with | no
#   export OUT_DIR=/custom/output
#   export SLEEP_SECS=15

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-claude-opus-4-7}"
MODE="${MODE:-with}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/outputs/gcd_runs}"
SLEEP_SECS="${SLEEP_SECS:-15}"

if [[ -z "${GALILEO_API_KEY:-}" ]]; then
  echo "ERROR: GALILEO_API_KEY is not set."
  exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

SCENES=(
  s1_freefall
  s2_mass
  s3_pendulum
  s4_ramp
  s6_launch
  s8_heat
  s9_spring
  s10_circular
  s16_decay
  s17_refraction
  s18_boyle
  s19_coulomb
  s20_hooke
  s21_weber
  s22_cooling
)

EXTRA_ARGS=()
if [[ "$MODE" == "no" ]]; then
  EXTRA_ARGS+=(--no-instruments)
elif [[ "$MODE" != "with" ]]; then
  echo "ERROR: MODE must be 'with' or 'no'."
  exit 1
fi

echo "=== GCD batch start ==="
echo "model=$MODEL mode=$MODE out=$OUT_DIR sleep=${SLEEP_SECS}s"

for scene in "${SCENES[@]}"; do
  echo "[$(date +%H:%M:%S)] scene=$scene"
  "$PYTHON_BIN" -m galileo.run_gcd \
    --scene "$scene" \
    --goal-idx 0 \
    --model "$MODEL" \
    --max-turns 10 \
    --out "$OUT_DIR" \
    "${EXTRA_ARGS[@]}"
  echo "sleep ${SLEEP_SECS}s"
  sleep "$SLEEP_SECS"
done

echo "=== GCD batch done ==="
