# MPPI 阶段4第二轮参数收敛执行报告（v1）

- 日期：2026-03-21
- 目标：按固定 seeds 对比 baseline 与阶段4，重点评估：
  - `action_post_delta_ratio_mppi`
  - 到达率
  - 碰撞率
  - 卡死率

---

## 1. 基线定义与说明

1. 当前工作区“阶段3完整代码”未形成可直接回滚提交。
2. 为保证固定 seeds 可复现实验，本次 baseline 采用 **pre-stage4 快照**（`cd8154d`）在临时 worktree 复跑：
   - `/tmp/icode_mppi_pre_stage4`
3. 为满足本次对比指标，baseline 脚本仅注入了观测字段（`planner_action_raw` 与 `action_post_delta_*`），未改控制决策逻辑。

> 说明：该 baseline 是“阶段4改造前算法快照”，不是最终归档的“阶段3 round2 提交版”；结论仅用于本轮阶段4收敛对比。

---

## 2. 固定 Seeds 与实验设置

1. 固定 seeds：`6101, 6102, 6103, 6104, 6105`
2. 统一场景参数：
   - `--random-obstacles --scene-seed=<seed> --seed=<seed>`
   - `--target-x 2.6 --target-y 0.0`
   - `--global-guide --auto-waypoint`
   - `--min-line-blockers 1 --require-mixed-sides --scene-sample-attempts 1200`
   - `--max-steps 240`
3. 对齐执行环境：
   - `--planner-model kinematic --device cpu`
   - 阶段4侧为可比性开启：`--pose-source gt --oracle-mode`

实验输出目录：

- baseline：`/tmp/mppi_stage4_round2_eval/baseline_pre_stage4`
- stage4：`/tmp/mppi_stage4_round2_eval/stage4/*`
- 汇总：`/tmp/mppi_stage4_round2_eval/stage4_round2_summary.json`

---

## 3. 参数收敛候选

本轮测试了 4 组阶段4参数：

1. `stage4_default`：默认参数
2. `stage4_tuned_b`：控制代价增强 + 放宽 `max_delta_u`
3. `stage4_tuned_c`：更强控制抑制（最小 post-delta）
4. `stage4_tuned_d`：在 `tuned_c` 思路上放宽 supervisor B 层触发和恢复时长，提升 MPPI 可执行步占比

`stage4_tuned_d` 关键参数（相对默认）：

1. `--noise-sigma 0.8 0.8`
2. `--cost-ctrl-effort 0.04`
3. `--cost-ctrl-smooth 0.18`
4. `--max-delta-u 2.5`
5. `--sup-progress-window 45`
6. `--sup-progress-min-delta 0.03`
7. `--sup-spin-min-progress 0.02`
8. `--sup-spin-min-yaw-travel 1.8`
9. `--sup-jam-min-u 3.0`
10. `--sup-recover-stop-steps 4`
11. `--sup-recover-backup-steps 6`
12. `--sup-recover-rotate-steps 10`
13. `--sup-recover-forward-steps 8`
14. `--sup-recover-refractory-steps 8`

---

## 4. 聚合对比（5 seeds）

| 配置 | 到达率 | 碰撞率 | 卡死率 | `action_post_delta_ratio_mppi` | `final_dist_mean` | `mppi_action_steps_mean` | `supervisor_override_steps_mean` |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_pre_stage4 | 0.000 | 0.400 | 1.000 | 1.0000 | 0.944 | 240.0 | 0.0 |
| stage4_default | 0.000 | 0.000 | 1.000 | 0.9984 | 2.433 | 99.0 | 141.0 |
| stage4_tuned_b | 0.000 | 0.000 | 1.000 | 0.2377 | 2.660 | 85.4 | 154.6 |
| stage4_tuned_c | 0.000 | 0.000 | 1.000 | 0.0540 | 2.593 | 70.8 | 169.2 |
| **stage4_tuned_d** | **0.000** | **0.000** | **1.000** | **0.0545** | **1.537** | **186.0** | **54.0** |

结论（本轮最优）：

1. `stage4_tuned_d` 在保持 `post_delta` 极低（约 `0.055`）的同时，明显恢复 MPPI 主导步数并降低 supervisor 过度覆盖。
2. 与 baseline 相比，碰撞率从 `0.4` 降为 `0.0`，但到达率/卡死率未改善（仍全卡死，受场景难度与恢复触发耦合影响）。

---

## 5. 固定 Seeds 明细（baseline vs tuned_d）

| seed | baseline: coll | baseline: final_dist | baseline: post_ratio | tuned_d: coll | tuned_d: final_dist | tuned_d: post_ratio | tuned_d: mppi/override |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6101 | 1 | 1.404 | 1.0000 | 0 | 1.417 | 0.0521 | 211 / 29 |
| 6102 | 0 | 1.072 | 1.0000 | 0 | 1.367 | 0.0332 | 211 / 29 |
| 6103 | 0 | 0.535 | 1.0000 | 0 | 1.803 | 0.0523 | 153 / 87 |
| 6104 | 0 | 0.486 | 1.0000 | 0 | 1.655 | 0.0972 | 144 / 96 |
| 6105 | 1 | 1.225 | 1.0000 | 0 | 1.443 | 0.0379 | 211 / 29 |

---

## 6. 典型场景链路日志

## 6.1 Baseline 典型（seed=6101，碰撞发生）

日志源：
- `/tmp/mppi_stage4_round2_eval/baseline_seed_6101.log`

关键字段摘录：

```json
{
  "steps_executed": 240,
  "collision_steps": 86,
  "reached_goal": false,
  "final_dist_to_goal": 1.4036,
  "action_post_delta_ratio_mppi": 1.0
}
```

解读：
1. baseline 后处理改写强，`post_ratio=1.0`。
2. 冲突场景中碰撞步数高。

## 6.2 阶段4 tuned_d 典型（seed=6103，安全恢复主导）

日志源：
- `/tmp/mppi_stage4_round2_eval/stage4_tuned_d_seed_6103.log`

关键字段摘录：

```json
{
  "steps_executed": 240,
  "collision_steps": 0,
  "reached_goal": false,
  "final_dist_to_goal": 1.8034,
  "supervisor_override_steps": 87,
  "mppi_action_steps": 153,
  "trigger_reason_steps": {
    "normal": 153,
    "near_collision": 3,
    "recover_sequence": 84
  },
  "action_post_delta_ratio_mppi": 0.0523
}
```

解读：
1. 执行动作一致性显著改善（`post_ratio` 由 1.0 降至约 0.05）。
2. 安全性提升（无碰撞），但仍存在恢复链路占用过多导致无法收敛到目标。

---

## 7. 本轮结论与下一步

## 7.1 本轮结论

1. 阶段4“执行一致性”目标已显著达成：`action_post_delta_ratio_mppi` 大幅下降。
2. 安全性不劣于 baseline（碰撞率下降）。
3. 到达率/卡死率未达目标，需要进入下一轮收敛（重点在 supervisor-B 与任务推进耦合）。

## 7.2 下一轮建议（阶段4 round3）

1. 在 `tuned_d` 基础上进一步收敛 supervisor-B 触发条件（减少不必要 recover 序列占比）。
2. 维持 `post_ratio` 低位的前提下，提升任务推进权重并降低“恢复链路长期占用”。
3. 对“goal_blocked 持续高”的 seeds 引入分场景参数组（仅算法参数，不引入补丁式触发器）。

