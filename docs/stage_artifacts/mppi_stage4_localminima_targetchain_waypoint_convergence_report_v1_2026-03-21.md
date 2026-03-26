# MPPI 局部最小值专项收敛（target-chain + waypoint触发语义）报告

## 1. 口径
- baseline：`/home/wmh/ICODE/domo/ICODE_MPPI/mppi_stage4_phaseB_convergence_round_v2_2026-03-21.json#defaults_after_phaseB_round_v2`
- new：`/tmp/mppi_localminima_targetchain_waypoint_20260321/full_regression/summary.json`
- fixed seeds：`[6101, 6102, 6103, 6104, 6105]`
- 统一命令主干：`planner-model=kinematic`, `max-steps=1000`, `random-obstacles + global-guide + auto-waypoint`

## 2. 聚合对比

| 指标 | baseline | new | Δ |
|---|---:|---:|---:|
| reach_rate | 0.600000 | 1.000000 | +0.400000 |
| collision_rate | 0.000000 | 0.000000 | +0.000000 |
| stuck_rate | 0.400000 | 0.000000 | -0.400000 |
| final_dist_mean | 0.747947 | 0.349949 | -0.397998 |
| mppi_steps_mean | 675.600000 | 518.400000 | -157.200000 |
| override_steps_mean | 13.400000 | 0.000000 | -13.400000 |
| goal_blocked_ratio_mean | 0.000000 | 0.000000 | +0.000000 |
| post_delta_ratio_mppi_mean | 0.988263 | 0.984386 | -0.003877 |

## 3. 关键种子（6102/6103）

| seed | baseline reached | new reached | baseline final_dist | new final_dist | baseline mppi/override | new mppi/override | baseline trigger | new trigger |
|---:|:---:|:---:|---:|---:|---:|---:|---|---|
| 6102 | False | True | 1.161 | 0.350 | 933/67 | 501/0 | `{'normal': 933, 'progress_stall': 1, 'recover_sequence': 66}` | `{'normal': 501}` |
| 6103 | False | True | 1.529 | 0.350 | 1000/0 | 939/0 | `{'normal': 1000}` | `{'normal': 939}` |

## 4. 结论
- 本轮 target-chain + waypoint 触发语义收敛后，原剩余未达 `6102/6103` 均转为到达。
- Supervisor override 在本口径下收敛为 `0`，未再触发 recover 链条。
- 新增 `waypoint_stuck_force_events` 可用于后续诊断“stuck后强制waypoint接管”是否生效。
