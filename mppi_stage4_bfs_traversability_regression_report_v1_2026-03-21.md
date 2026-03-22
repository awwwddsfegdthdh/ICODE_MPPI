# MPPI 阶段4：BFS 可通过性改造回归结果 v1

- 日期：2026-03-21
- 旧基线：`/tmp/mppi_stage4_phaseB_round_20260321/defaults_after_phaseB_round_v2`
- 新回归：`/tmp/mppi_bfs_traversability_regression_20260321`
- fixed seeds：`6101,6102,6103,6104,6105`

## 聚合对比（旧 -> 新）

| 指标 | 旧 | 新 | 变化(新-旧) |
|---|---:|---:|---:|
| reach_rate | 0.600 | 0.600 | +0.000 |
| collision_rate | 0.000 | 0.000 | +0.000 |
| stuck_rate | 0.400 | 0.400 | +0.000 |
| final_dist_mean | 0.7479 | 0.7481 | +0.0001 |
| mppi_steps_mean | 675.6 | 698.8 | +23.2 |
| override_steps_mean | 13.4 | 0.0 | -13.4 |
| progress_stall_events_mean | 0.2 | 0.0 | -0.2 |
| mppi_reverse_raw_steps_mean | 37.6 | 22.6 | -15.0 |

## 逐 seed 变化（核心）

1. `6102`：
   - 旧：`progress_stall + recover_sequence`，`override=67`
   - 新：全程 `normal`，`override=0`
   - 最终距离保持同量级（`1.1612 -> 1.1617`）
2. `6104`：
   - 新版本步数略增（`368 -> 417`），但仍到达。

## 结论

1. BFS 可通过性改造没有引入碰撞回退，reach_rate 保持不降。
2. 触发恢复链的误触发被压低（特别是 seed6102），控制链更稳定。
3. 仍有 2 个未达 seed（6102/6103），后续应做 target-chain / local minima 专项收敛。

