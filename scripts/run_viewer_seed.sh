#!/usr/bin/env bash
set -euo pipefail

SEED="${1:-6100}"
PYTHON_BIN="${PYTHON_BIN:-/home/wmh/miniconda3/envs/icode_mujoco/bin/python}"

"${PYTHON_BIN}" run_icode_mppi_e1_viewer.py \
  --device cuda \
  --planner-model kinematic \
  --pose-source gt \
  --global-guide \
  --global-planner astar \
  --reference-sampling \
  --random-obstacles \
  --auto-waypoint \
  --seed "${SEED}" \
  --scene-seed "${SEED}" \
  --target-x 2.6 \
  --target-y 0.0 \
  --goal-tol 0.15 \
  --max-steps 1000 \
  --print-interval 20 \
  --obs-x-range 0.8 2.2 \
  --obs-y-range -1.2 1.2 \
  --min-line-blockers 1 \
  --blocker-t-range 0.20 0.90 \
  --require-mixed-sides \
  --scene-sample-attempts 1600
