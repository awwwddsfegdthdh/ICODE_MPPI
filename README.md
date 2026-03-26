# ICODE MPPI (MuJoCo E1)

This repository contains the E1 MuJoCo navigation stack based on MPPI, with optional ICODE dynamics prediction.

## Quick Start

1. Activate environment:

```bash
conda activate icode_mujoco
```

2. Run viewer (default collaboration profile):

```bash
/home/wmh/miniconda3/envs/icode_mujoco/bin/python run_icode_mppi_e1_viewer.py \
  --device cuda \
  --planner-model kinematic \
  --global-guide --global-planner astar --reference-sampling \
  --random-obstacles --auto-waypoint \
  --seed 6100 --scene-seed 6100 \
  --target-x 2.6 --target-y 0.0 \
  --goal-tol 0.15 --max-steps 1000
```

3. Run headless test/eval:

```bash
/home/wmh/miniconda3/envs/icode_mujoco/bin/python run_icode_mppi_e1_test.py \
  --device cuda \
  --planner-model kinematic \
  --seed 6100 --scene-seed 6100 \
  --global-guide --global-planner astar --reference-sampling \
  --random-obstacles --auto-waypoint
```

## Repository Layout

- `run_icode_mppi_e1_viewer.py`: interactive MuJoCo viewer entry.
- `run_icode_mppi_e1_test.py`: headless regression/evaluation entry.
- `mppi.py`: MPPI controller and cost definitions.
- `mppi_nav_utils.py`: global guide/path, waypoint, target projection utilities.
- `safety_supervisor.py`: supervisor state machine and recovery logic.
- `env_mujoco.py`: MuJoCo E1 environment wrapper and sensor interfaces.
- `local_occupancy_map.py`: local occupancy map fusion and obstacle extraction.
- `obs_geometry.py`: sensor ray geometry and fusion helpers.
- `tests/`: unit tests.
- `scripts/`: helper scripts for common collaborative workflows.
- `docs/`: design docs, manuals, and stage artifacts.
- `docs/stage_artifacts/`: historical stage reports/summaries (`mppi_*.md/json`).

## Collaboration Conventions

- Keep runtime outputs in `runs/` (already ignored by `.gitignore`).
- Keep temporary datasets/checkpoints out of git.
- Put new design/report docs under `docs/` (prefer `docs/stage_artifacts/` for staged experiments).
- Use fixed-seed commands when reporting algorithm changes.

## Useful Scripts

- `scripts/run_viewer_seed.sh <seed>`: start viewer with collaborative default args.
- `scripts/run_fixed_seeds_regression.sh`: run fixed-seed headless regression and print summary.
