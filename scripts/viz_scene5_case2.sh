#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/wmh/miniconda3/envs/icode_mujoco/bin/python}"
CKPT="${CKPT:-runs/icode_e1_passable35_uniform_4h/icode_best.pt}"

"${PYTHON_BIN}" run_icode_mppi_e1_viewer.py \
  --device cuda \
  --planner-model icode \
  --checkpoint "${CKPT}" \
  --pose-source gt \
  --target-x 2.8 --target-y 0.2 \
  --u-init 0 0 \
  --random-obstacles \
  --scene-seed 8502 \
  --seed 8502 \
  --max-steps 1200 \
  --print-interval 40 \
  --stop-on-goal \
  --goal-tol 0.22 \
  --goal-slowdown-radius 0.70 \
  --goal-slowdown-min-scale 0.24 \
  --w-terminal-stop 45 \
  --terminal-stop-radius 0.22 \
  --dock-brake-radius 0.55 \
  --dock-enter-radius 0.20 \
  --dock-exit-radius 0.42 \
  --dock-k-brake 0.20 \
  --dock-k-yaw 0.75 \
  --dock-hold-steps 8 \
  --auto-waypoint \
  --global-guide \
  --obs-x-range 0.8 2.4 \
  --obs-y-range -1.3 1.3 \
  --min-obs-obs-dist 0.92 \
  --min-start-obs-dist 1.05 \
  --min-goal-obs-dist 1.05 \
  --constrain-obs-radius \
  --randomize-obs-radius \
  --obs-radius-min 0.14 \
  --obs-radius-max 0.22 \
  --scene-sample-attempts 2200 \
  --min-line-blockers 1 \
  --blocker-t-range 0.20 0.90 \
  --require-mixed-sides
