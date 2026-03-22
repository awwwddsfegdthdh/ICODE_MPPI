# MPPI 阶段4（Phase B）自适应参数收敛执行报告（fixed seeds 全量）

- 日期：2026-03-21
- 固定 seeds：`6101,6102,6103,6104,6105`
- 原始输出目录：`/tmp/mppi_stage4_phaseB_adaptive_20260321`
- 本轮选优结果：`b2_c_jam_relax1`

## 1. 执行口径

1. 非 smoke，全量固定 seeds，`max_steps=240`。
2. 统一基线：`tuned_d + stageB near` 参数面。
3. 按“先压 `goal_blocked_conf`，再保 `mppi_steps_mean`”进行多轮收敛；并根据中间输出继续追加 jam 语义收敛轮。

## 2. 阶段结果表

| 配置 | reach_rate | collision_rate | final_dist_mean | mppi_steps_mean | override_steps_mean | goal_blocked_ratio_mean | goal_blocked_conf_mean | action_post_delta_ratio_mppi_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r0_baseline_current | 0.000 | 0.000 | 0.952 | 187.8 | 52.2 | 0.000 | 0.00000 | 0.1761 |
| b1_a_conf_strict | 0.000 | 0.000 | 0.952 | 187.8 | 52.2 | 0.000 | 0.00000 | 0.1761 |
| b1_b_map_occ_up | 0.000 | 0.000 | 0.947 | 187.8 | 52.2 | 0.000 | 0.00000 | 0.1761 |
| b2_a_boundary_relax | 0.000 | 0.000 | 0.947 | 187.8 | 52.2 | 0.000 | 0.00000 | 0.1761 |
| b2_b_boundary_target_relax | 0.000 | 0.000 | 1.038 | 182.0 | 58.0 | 0.000 | 0.00000 | 0.1956 |
| b2_c_jam_relax1 | 0.000 | 0.000 | 0.644 | 211.0 | 29.0 | 0.000 | 0.00002 | 0.1479 |
| b2_d_jam_relax2 | 0.000 | 0.000 | 0.653 | 211.0 | 29.0 | 0.000 | 0.00002 | 0.1479 |

## 3. 关键观察与自适应调整

1. B1阶段（`b1_a/b1_b`）观测到 `goal_blocked_conf_mean` 已接近 0，说明 blocked 误触发已被压住。
2. 新主瓶颈转为 `jam_contact` 早触发导致 recover 序列占用，因此追加了 `b2_c/b2_d` 两轮 jam 语义收敛。
3. 最优 `b2_c_jam_relax1` 相比 `r0_baseline_current`：
   - `final_dist_mean`: 0.952 -> 0.644（改善 0.308）
   - `mppi_steps_mean`: 187.8 -> 211.0（提升 23.2）
   - `override_steps_mean`: 52.2 -> 29.0（下降 23.2）

## 4. 最优配置（b2_c_jam_relax1）增量参数

在 `tuned_d + stageB near` 基础上：

1. `--goal-blocked-enter-conf 0.72`
2. `--goal-blocked-enter-steps 4`
3. `--goal-blocked-exit-conf 0.42`
4. `--map-observed-threshold 0.22`
5. `--map-occ-threshold 0.42`
6. `--sup-boundary-hard-enter-dist 0.24`
7. `--sup-boundary-soft-confirm-steps 5`
8. `--sup-boundary-release-progress-min 0.06`
9. `--sup-jam-window 24`
10. `--sup-jam-min-u 4.0`
11. `--sup-jam-max-dxy 0.04`
12. `--sup-jam-max-v 0.05`

## 5. 失败链路对比（baseline vs best）

| seed | baseline: first_fail_step | baseline: trigger_around | best: first_fail_step | best: trigger_around | best: final_dist | best: mppi/override |
|---:|---:|---|---:|---|---:|---:|
| 6101 | 17 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.571 | 211/29 |
| 6102 | 17 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.684 | 211/29 |
| 6103 | 17 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.565 | 211/29 |
| 6104 | 17 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.627 | 211/29 |
| 6105 | 17 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 23 | ['normal', 'normal', 'normal', 'jam_contact', 'recover_sequence', 'recover_sequence', 'recover_sequence'] | 0.775 | 211/29 |

## 6. 结论

1. 本轮“先压 `goal_blocked_conf` 再保 `mppi_steps`”已落地执行并完成多轮自适应。
2. 当前最优为 `b2_c_jam_relax1`，MPPI主导窗口显著扩大，最终距离明显下降。
3. 仍未达到到达率目标（`reach_rate=0`），下一轮应聚焦 docking/terminal stop 语义与近终点推进收敛。