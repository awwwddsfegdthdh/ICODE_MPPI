# MPPI 阶段B执行报告（near_collision 重入窗口语义收敛）

- 日期：2026-03-21
- 执行内容：在阶段A基础上，完成 near 语义重构并进行固定 seeds 回归。
- 固定 seeds：`6101,6102,6103,6104,6105`
- 对比对象：`baseline tuned_d`、`stageA`

## 1. 阶段B改造内容

1. `near_collision` 触发由“全局 min_clearance”主导改为“方向相关风险”主导：
   - `front_clearance < near_front_enter`
   - 转向时 `side_clearance < near_side_enter`
   - `touch_force` 直接触发
2. `min_clearance` 降级为 `emergency` 判据（`< sup_emergency_clearance`）。
3. 新增 near 滞回退出：
   - 连续 `near_clear_steps` 满足 `front_clearance > near_front_exit` 且 `side_clearance > near_side_exit` 才解除 near 风险锁存。
4. 新增 near 重入控制：
   - `near_reentry_guard_steps`
   - `near_rearm_progress`（goal/path 任一达到再武装）
5. `emergency` 保持可抢占，不受重入门控限制。

## 2. 代码与测试

改造文件：
- [safety_supervisor.py](/home/wmh/ICODE/domo/ICODE_MPPI/safety_supervisor.py)
- [run_icode_mppi_e1_test.py](/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_test.py)

测试文件同步：
- [test_safety_supervisor_transitions.py](/home/wmh/ICODE/domo/ICODE_MPPI/tests/test_safety_supervisor_transitions.py)
- [test_safety_supervisor_priority.py](/home/wmh/ICODE/domo/ICODE_MPPI/tests/test_safety_supervisor_priority.py)
- [test_safety_supervisor_cooldown.py](/home/wmh/ICODE/domo/ICODE_MPPI/tests/test_safety_supervisor_cooldown.py)
- [test_safety_supervisor_trace_contract.py](/home/wmh/ICODE/domo/ICODE_MPPI/tests/test_safety_supervisor_trace_contract.py)

`icode_mujoco` 环境测试结果：`5 passed in 0.07s`

## 3. 固定 seeds 回归配置

沿用 tuned_d 主参数，并显式启用阶段B near 参数：

1. `--sup-near-front-enter 0.14`
2. `--sup-near-front-exit 0.22`
3. `--sup-near-side-enter 0.12`
4. `--sup-near-side-exit 0.18`
5. `--sup-near-clear-steps 6`
6. `--sup-near-reentry-guard-steps 24`
7. `--sup-near-rearm-progress 0.08`
8. `--sup-near-turn-wz-gate 0.35`
9. `--sup-emergency-clearance 0.08`

输出目录：`/tmp/mppi_stageB_eval`

## 4. 聚合结果（5 seeds）

| 配置 | reach_rate | collision_rate | final_dist_mean | mppi_steps_mean | override_steps_mean | post_delta_ratio_mppi | goal_blocked_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline tuned_d | 0.000 | 0.000 | 1.537 | 186.0 | 54.0 | 0.0545 | 0.6908 |
| stageA | 0.000 | 0.000 | 1.803 | 166.4 | 73.6 | 0.0424 | 0.8133 |
| **stageB** | **0.000** | **0.000** | **1.717** | **197.2** | **42.8** | **0.0400** | **0.8575** |

阶段B相对阶段A：

1. `final_dist_mean`：`-0.086`（改善）
2. `mppi_steps_mean`：`+30.8`（改善）
3. `override_steps_mean`：`-30.8`（改善）
4. `post_delta_ratio_mppi_mean`：`-0.0024`（改善）

阶段B相对 baseline tuned_d：

1. `final_dist_mean`：`+0.180`（仍劣）
2. `mppi_steps_mean`：`+11.2`
3. `override_steps_mean`：`-11.2`
4. `post_delta_ratio_mppi_mean`：`-0.0146`

## 5. 语义目标达成度（阶段B）

1. `near` 重入有所缓解：
   - stageA `near_events_mean = 2.8`
   - stageB `near_events_mean = 2.0`
2. 重入间隔拉长：
   - stageA `near_reentry_interval_median = 36`
   - stageB `near_reentry_interval_median = 52`
3. `recover_sequence` 平均步数下降：
   - stageA `70.8 -> stageB 40.8`

但仍未达到系统目标（到达率仍 0，且较 tuned_d 仍有距离劣化）。

## 6. 结论

1. 阶段B near 重入语义改造有效，明显降低了 supervisor 过度覆盖并恢复了 MPPI 连续控制窗口。
2. 但当前版本仍未恢复到 tuned_d 的 `final_dist` 水平；主瓶颈表现为 `goal_blocked_ratio` 仍偏高。
3. 下一步应进入“阶段B第二轮参数收敛”，聚焦 near enter/exit 阈值和 rearm 进展门槛，使 near 触发更贴近真实前向风险而不过度保守。

## 7. 产物

1. 汇总 JSON：
   - [mppi_stageB_vs_stageA_tuned_d_summary_v1_2026-03-21.json](/home/wmh/ICODE/domo/ICODE_MPPI/mppi_stageB_vs_stageA_tuned_d_summary_v1_2026-03-21.json)
2. 回归原始输出：
   - `/tmp/mppi_stageB_eval/seed_*/metrics.json`
   - `/tmp/mppi_stageB_eval/seed_*/rollout_log.npz`

