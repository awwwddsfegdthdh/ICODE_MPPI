# MPPI 阶段4 Phase B 收敛轮执行报告 v2

- 日期：2026-03-21
- fixed seeds：`6101,6102,6103,6104,6105`
- 输出目录：`/tmp/mppi_stage4_phaseB_round_20260321`
- 本轮选优：`phaseB_c6_push_passage_blayer_relax`（并已同步为默认参数，验证配置：`defaults_after_phaseB_round_v2`）

## 1. 聚合对比表

| 配置 | reach_rate | collision_rate | stuck_rate | final_dist_mean | mppi_steps_mean | override_steps_mean | terminal_mppi_steps_mean | speed_cap_applied_steps_mean | mppi_reverse_raw_steps_mean | goal_blocked_conf_mean | action_post_delta_ratio_mppi_mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_default | 0.200 | 0.000 | 0.800 | 1.203 | 903.4 | 13.4 | 48.0 | 658.6 | 28.6 | 0.00659 | 1.000 |
| phaseB_c1_corridor_loosen | 0.200 | 0.000 | 0.800 | 1.212 | 919.8 | 7.6 | 59.0 | 668.6 | 26.8 | 0.00665 | 1.000 |
| phaseB_c2_near_corridor | 0.200 | 0.000 | 0.800 | 1.211 | 921.6 | 0.0 | 58.2 | 691.0 | 26.8 | 0.00705 | 1.000 |
| phaseB_c3_semantic_action | 0.400 | 0.000 | 0.600 | 0.937 | 792.2 | 5.4 | 120.4 | 625.4 | 38.0 | 0.03259 | 0.999 |
| phaseB_c4_push_passage | 0.600 | 0.000 | 0.400 | 0.769 | 662.2 | 26.8 | 195.4 | 394.0 | 38.6 | 0.00834 | 0.985 |
| phaseB_c5_push_explore_jamrelax | 0.400 | 0.000 | 0.600 | 0.754 | 711.4 | 26.8 | 109.8 | 427.0 | 47.2 | 0.01587 | 0.993 |
| phaseB_c6_push_passage_blayer_relax | 0.600 | 0.000 | 0.400 | 0.748 | 675.6 | 13.4 | 195.4 | 444.6 | 37.6 | 0.00417 | 0.988 |
| defaults_after_phaseB_round_v2 | 0.600 | 0.000 | 0.400 | 0.748 | 675.6 | 13.4 | 195.4 | 444.6 | 37.6 | 0.00417 | 0.988 |

## 2. 结论

1. 相比 baseline，`phaseB_c6_push_passage_blayer_relax` 达到率从 `0.200` 提升到 `0.600`，`final_dist_mean` 从 `1.203` 降到 `0.748`。
2. `collision_rate` 全配置均为 `0.000`；`phaseB_c6_push_passage_blayer_relax` 同时把 `goal_blocked_conf_mean` 保持在低位（`0.00417`）。
3. `defaults_after_phaseB_round_v2` 与 `phaseB_c6_push_passage_blayer_relax` 的 fixed-seeds 指标一致，说明默认参数面已完成落地。

## 3. 最优配置失败链路（未到达 seeds）

- seed `6102`: final_dist=`1.161`, mppi/override=`933/67`, trigger=`{'normal': 933, 'progress_stall': 1, 'recover_sequence': 66}`
- seed `6103`: final_dist=`1.529`, mppi/override=`1000/0`, trigger=`{'normal': 1000}`

## 4. Phase B 默认参数落地项

1. 走廊/目标链：`cost_safe_corridor=120`, `path_corridor_half_width=0.34`, `corridor_target_slack=0.18`, `corridor_tight_clearance=0.36`, `corridor_tight_max_dev=0.32`, `target_jump_max/min=0.62/0.18`
2. 近障代价：`cost_safe_near=0.6`, `near_penalty_mid/hard_clearance=0.34/0.15`, `near_penalty_mid/hard_scale=0.06/0.55`, `near_penalty_hard_power=3.2`
3. 控制语义：`cost_task_progress=120`, `cost_task_path_progress=120`, `cost_task_path_track=10`, `cost_task_reverse=5.0`, `nav_speed_max=2.0`, `nominal/nav_min_forward_speed=0.07`, `max_delta_u=0.80`, `speed_cap_clearance_hard/soft=0.10/0.40`, `speed_cap_heading_min_gain=0.20`, `terminal_profile_alpha=0.10`
4. Supervisor B 层：`sup_progress_window=80`, `sup_progress_min_delta=0.01`, `sup_progress_confirm_windows=5`, `sup_progress_min_u=2.5`, `sup_jam_window=30`, `sup_jam_min_u=10.0`, `sup_jam_max_dxy=0.02`, `sup_jam_max_v=0.03`, `sup_jam_front_clearance_gate=0.25`, `sup_jam_blocked_ratio_min=0.80`, `sup_spin_window=40`, `sup_spin_min_progress=0.03`
