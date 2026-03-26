# 阶段4 Rootfix4 固定 Seeds 对比报告 v2

- 旧口径（240步）：`/home/wmh/ICODE/domo/ICODE_MPPI/mppi_stage4_rootfix_quick_regression_summary_v1_2026-03-21.json`
- 新口径（240步）：`/tmp/mppi_rootfix4_defaults240_20260321`
- 新口径（1000步）：`/tmp/mppi_rootfix4_semantic_1000_C2b_20260321`

## 聚合对比（旧240 vs 新240）

| 指标 | 旧240 | 新240 | 变化(新-旧) |
|---|---:|---:|---:|
| reach_rate | 0.0000 | 0.0000 | +0.0000 |
| collision_rate | 0.0000 | 0.0000 | +0.0000 |
| final_dist_mean | 0.3785 | 0.9729 | +0.5943 |
| mppi_steps_mean | 165.6000 | 240.0000 | +74.4000 |
| override_steps_mean | 0.0000 | 0.0000 | +0.0000 |
| terminal_mppi_steps_mean | 0.0000 | 1.6000 | +1.6000 |
| speed_cap_applied_steps_mean | 0.0000 | 35.0000 | +35.0000 |
| mppi_reverse_raw_steps_mean | 23.6000 | 27.2000 | +3.6000 |
| nominal_reverse_suppressed_steps_mean | 16.2000 | 21.0000 | +4.8000 |
| goal_blocked_ratio_mean | 0.0000 | 0.0000 | +0.0000 |
| action_post_delta_ratio_mppi_mean | 0.9732 | 1.0000 | +0.0268 |
| jam_contact_events_mean | 0.0000 | 0.0000 | +0.0000 |
| progress_stall_events_mean | 0.0000 | 0.0000 | +0.0000 |
| spin_stall_events_mean | 0.0000 | 0.0000 | +0.0000 |

## 新口径 240 步逐 Seed

| seed | reached | final_dist | mppi_steps | terminal_mppi_steps | speed_cap_steps | mppi_reverse_raw | nominal_reverse_suppressed | final_nav_mode |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 6101 | 0 | 0.7309 | 240 | 8 | 46 | 28 | 27 | NAV |
| 6102 | 0 | 0.9217 | 240 | 0 | 33 | 26 | 23 | NAV |
| 6103 | 0 | 0.8857 | 240 | 0 | 36 | 28 | 23 | NAV |
| 6104 | 0 | 1.0830 | 240 | 0 | 40 | 29 | 10 | NAV |
| 6105 | 0 | 1.2430 | 240 | 0 | 20 | 25 | 22 | NAV |

## 新口径 1000 步聚合

| 指标 | 新1000 |
|---|---:|
| reach_rate | 0.6000 |
| collision_rate | 0.0000 |
| final_dist_mean | 0.3547 |
| mppi_steps_mean | 739.6000 |
| override_steps_mean | 0.0000 |
| terminal_mppi_steps_mean | 475.2000 |
| speed_cap_applied_steps_mean | 514.6000 |
| mppi_reverse_raw_steps_mean | 27.2000 |
| nominal_reverse_suppressed_steps_mean | 21.0000 |
| goal_blocked_ratio_mean | 0.0000 |
| action_post_delta_ratio_mppi_mean | 1.0000 |
| jam_contact_events_mean | 0.0000 |
| progress_stall_events_mean | 0.0000 |
| spin_stall_events_mean | 0.0000 |

## 新口径 1000 步逐 Seed

| seed | reached | final_dist | mppi_steps | terminal_mppi_steps | speed_cap_steps | final_nav_mode |
|---:|---:|---:|---:|---:|---:|---|
| 6101 | 1 | 0.3500 | 726 | 482 | 520 | BRAKE_ALIGN |
| 6102 | 1 | 0.3497 | 518 | 255 | 294 | BRAKE_ALIGN |
| 6103 | 1 | 0.3497 | 454 | 192 | 228 | BRAKE_ALIGN |
| 6104 | 0 | 0.3605 | 1000 | 731 | 778 | NAV |
| 6105 | 0 | 0.3636 | 1000 | 716 | 753 | NAV |
