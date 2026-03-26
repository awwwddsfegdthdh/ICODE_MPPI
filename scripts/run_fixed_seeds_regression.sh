#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/wmh/miniconda3/envs/icode_mujoco/bin/python}"
OUT_ROOT="${OUT_ROOT:-runs/fixed_seed_regression}"
SEEDS=(6100 6101 6102 6103)

mkdir -p "${OUT_ROOT}"

for SEED in "${SEEDS[@]}"; do
  OUT_DIR="${OUT_ROOT}/seed_${SEED}"
  mkdir -p "${OUT_DIR}"
  echo "[run] seed=${SEED} -> ${OUT_DIR}"

  "${PYTHON_BIN}" run_icode_mppi_e1_test.py \
    --device cuda \
    --planner-model kinematic \
    --pose-source gt \
    --seed "${SEED}" \
    --scene-seed "${SEED}" \
    --target-x 2.6 \
    --target-y 0.0 \
    --u-init 0 0 \
    --random-obstacles \
    --auto-waypoint \
    --global-guide \
    --global-planner astar \
    --reference-sampling \
    --no-goal-direct-on-clear \
    --goal-tol 0.15 \
    --max-steps 1000 \
    --obs-x-range 0.8 2.2 \
    --obs-y-range -1.2 1.2 \
    --min-line-blockers 1 \
    --blocker-t-range 0.20 0.90 \
    --require-mixed-sides \
    --scene-sample-attempts 1600 \
    --output-dir "${OUT_DIR}" >/tmp/mppi_reg_seed_${SEED}.log 2>&1

  "${PYTHON_BIN}" - <<PY
import json
m = json.load(open("${OUT_DIR}/metrics.json", "r", encoding="utf-8"))
print({
  "seed": ${SEED},
  "reached_goal": m.get("reached_goal"),
  "final_dist": round(float(m.get("final_dist_to_goal", 0.0)), 3),
  "min_dist": round(float(m.get("min_dist_to_goal", 0.0)), 3),
  "collision_steps": int(m.get("collision_steps", 0)),
  "recover_events": int(m.get("recover_events", 0)),
  "guide_fail": int(m.get("global_guide_fail_steps", 0)),
  "guide_fail_cause": m.get("global_guide_fail_cause_counts", {}),
  "post_delta_ratio_mppi": round(float(m.get("action_post_delta_ratio_mppi", 0.0)), 3),
})
PY
done

echo "[done] results saved in ${OUT_ROOT}"
