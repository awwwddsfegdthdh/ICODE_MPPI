# MPPI 阶段A执行报告（progress/spin 语义收敛）

- 日期：2026-03-21
- 执行内容：按 `mppi_stage4_semantic_convergence_plan_v1_2026-03-21.md` 完成阶段A代码改造，并对固定 seeds 做回归验证。
- 固定 seeds：`6101,6102,6103,6104,6105`
- 对比基线：`stage4 tuned_d`（来自阶段4 round2 结果）

## 1. 本次代码改造

1. `progress_stall` 从单阈值改为合取判定 + 连续确认：
   - 目标推进、路径推进、位移、控制努力/速度、blocked/front-clearance 共同判定。
2. `spin_stall` 从单阈值改为合取判定 + 连续确认：
   - 旋转量、位移、角速度、速度、blocked/front-clearance、目标切换稳定性共同判定。
3. `run_icode_mppi_e1_test.py` 与 `run_icode_mppi_e1_viewer.py` 增加并透传阶段A语义输入：
   - `path_progress_recent`
   - `target_switched_recent`
4. supervisor 测试配置同步到新参数面（保持可编译）。

关键文件：
- [safety_supervisor.py](/home/wmh/ICODE/domo/ICODE_MPPI/safety_supervisor.py)
- [run_icode_mppi_e1_test.py](/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_test.py)
- [run_icode_mppi_e1_viewer.py](/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_viewer.py)
- [test_safety_supervisor_transitions.py](/home/wmh/ICODE/domo/ICODE_MPPI/tests/test_safety_supervisor_transitions.py)
- [test_safety_supervisor_priority.py](/home/wmh/ICODE/domo/ICODE_MPPI/tests/test_safety_supervisor_priority.py)
- [test_safety_supervisor_cooldown.py](/home/wmh/ICODE/domo/ICODE_MPPI/tests/test_safety_supervisor_cooldown.py)
- [test_safety_supervisor_trace_contract.py](/home/wmh/ICODE/domo/ICODE_MPPI/tests/test_safety_supervisor_trace_contract.py)

## 2. 回归命令配置（与 tuned_d 对齐）

1. `planner-model=kinematic`, `pose-source=gt`, `oracle-mode`
2. `max-steps=240`, `random-obstacles`, `global-guide`, `auto-waypoint`
3. tuned_d 关键参数保持：
   - `noise-sigma 0.8 0.8`
   - `cost-ctrl-effort 0.04`
   - `cost-ctrl-smooth 0.18`
   - `max-delta-u 2.5`
   - `sup-progress-window 45`
   - `sup-progress-min-delta 0.03`
   - `sup-spin-min-progress 0.02`
   - `sup-spin-min-yaw-travel 1.8`
   - `sup-jam-min-u 3.0`
   - `sup-recover-*` 与 round2 tuned_d 一致

输出目录：`/tmp/mppi_stageA_eval`

## 3. 聚合结果（stageA vs baseline tuned_d）

| 指标 | baseline tuned_d | stageA | 变化(stageA-baseline) |
|---|---:|---:|---:|
| reach_rate | 0.000 | 0.000 | 0.000 |
| collision_rate | 0.000 | 0.000 | 0.000 |
| final_dist_mean | 1.537 | 1.803 | +0.266 |
| mppi_action_steps_mean | 186.0 | 166.4 | -19.6 |
| supervisor_override_steps_mean | 54.0 | 73.6 | +19.6 |
| action_post_delta_ratio_mppi_mean | 0.0545 | 0.0424 | -0.0122 |
| goal_blocked_ratio_mean | 0.6908 | 0.8133 | +0.1225 |

触发总数对比：

1. baseline tuned_d：`near=6, progress=2, spin=2, jam=0`
2. stageA：`near=14, progress=0, spin=0, jam=0`

## 4. 阶段A目标达成度

## 4.1 已达成

1. `progress_stall/spin_stall` 误触发被显著抑制：
   - stageA 全部 seeds 中 `progress/spin` 触发事件数为 0。
2. MPPI 后处理一致性继续改善：
   - `action_post_delta_ratio_mppi` 进一步降低。

## 4.2 未达成

1. 全局任务性能未改善，且退化：
   - `final_dist_mean` 增大；`mppi_steps` 下降，`override` 上升。
2. 根因不是新 bug，而是失败主链迁移：
   - B层几乎不触发后，A层 `near_collision` 成为主导，恢复链路占比上升。

## 5. 结论

1. 阶段A“误触发治理”在 B 层目标上是成功的。
2. 但系统级目标（到达/卡死）没有提升，原因是 A 层 `near_collision` 重入窗口尚未收敛。
3. 这与阶段规划一致：阶段A后必须进入阶段B（near 重入语义与窗口收敛），否则会出现“误触发减少但 near 接管上升”的替代失效。

## 6. 产物

1. 汇总 JSON：
   - [mppi_stageA_vs_tuned_d_summary_v1_2026-03-21.json](/home/wmh/ICODE/domo/ICODE_MPPI/mppi_stageA_vs_tuned_d_summary_v1_2026-03-21.json)
2. 原始回归输出：
   - `/tmp/mppi_stageA_eval/seed_*/metrics.json`
   - `/tmp/mppi_stageA_eval/seed_*/rollout_log.npz`

## 7. 附注

1. 已在 `icode_mujoco` 环境执行 supervisor 相关测试：
   - `tests/test_safety_supervisor_transitions.py`
   - `tests/test_safety_supervisor_priority.py`
   - `tests/test_safety_supervisor_cooldown.py`
   - `tests/test_safety_supervisor_trace_contract.py`
2. 结果：`5 passed in 0.06s`。
3. 已完成语法编译校验（`py_compile`）并通过。
