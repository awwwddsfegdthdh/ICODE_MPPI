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
- `scripts/batch_collect_weighted_dataset.py`: weighted batch dataset collection (`collect -> convert -> prepare`).

## Weighted Data Collection (60/25/15)

Default plan file:

- `configs/e1_weighted_sampling_60_25_15.json`

Dry-run first (print commands only):

```bash
/home/wmh/miniconda3/envs/icode_mujoco/bin/python scripts/batch_collect_weighted_dataset.py \
  --config configs/e1_weighted_sampling_60_25_15.json \
  --total-episodes 600 \
  --base-seed 7000 \
  --dry-run
```

Run actual collection:

```bash
/home/wmh/miniconda3/envs/icode_mujoco/bin/python scripts/batch_collect_weighted_dataset.py \
  --config configs/e1_weighted_sampling_60_25_15.json \
  --total-episodes 600 \
  --base-seed 7000
```

Outputs are written under:

- `datasets/<plan_name>_<timestamp>/raw/*.npz`
- `datasets/<plan_name>_<timestamp>/converted/*.npz`
- `datasets/<plan_name>_<timestamp>/bundle/*_bundle.npz`
- `datasets/<plan_name>_<timestamp>/manifest.json`

## Multi-XML Rotation (Per-Episode Box/Cylinder Mix)

Default multi-XML plan file:

- `configs/e1_weighted_sampling_60_25_15_multixml.json`

This plan enables `xml_rotation` and auto-generates obstacle-shape variants from:

- `E1_Robot/simulation/models/mjcf/E1_SimpleSensor.xml`

Per-episode, one XML variant is sampled uniformly at random, then collection runs for 1 episode per shard (to guarantee episode-level XML switching).

Dry-run:

```bash
/home/wmh/miniconda3/envs/icode_mujoco/bin/python scripts/batch_collect_weighted_dataset.py \
  --config configs/e1_weighted_sampling_60_25_15_multixml.json \
  --total-episodes 120 \
  --base-seed 7100 \
  --dry-run
```

Run:

```bash
/home/wmh/miniconda3/envs/icode_mujoco/bin/python scripts/batch_collect_weighted_dataset.py \
  --config configs/e1_weighted_sampling_60_25_15_multixml.json \
  --total-episodes 120 \
  --base-seed 7100
```
