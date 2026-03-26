# 阶段4 Rootfix4 固定 Seeds 对比（6101~6105）

- 旧口径：`/home/wmh/ICODE/domo/ICODE_MPPI/mppi_stage4_rootfix_quick_regression_summary_v1_2026-03-21.json`
- 新口径：`/tmp/mppi_rootfix4_semantic_new_20260321`

| 指标 | 旧 | 新 | 变化(新-旧) |
|---|---:|---:|---:|
| reach_rate | 0.0000 | 0.0000 | +0.0000 |
| collision_rate | 0.0000 | 0.0000 | +0.0000 |
| final_dist_mean | 0.3785 | 1.8489 | +1.4703 |
| mppi_steps_mean | 165.6000 | 240.0000 | +74.4000 |
| override_steps_mean | 0.0000 | 0.0000 | +0.0000 |
| terminal_mppi_steps_mean | 0.0000 | 0.0000 | +0.0000 |
| speed_cap_applied_steps_mean | 0.0000 | 72.8000 | +72.8000 |
| mppi_reverse_raw_steps_mean | 23.6000 | 27.4000 | +3.8000 |
| nominal_reverse_suppressed_steps_mean | 16.2000 | 21.4000 | +5.2000 |
| goal_blocked_ratio_mean | 0.0000 | 0.0000 | +0.0000 |
| action_post_delta_ratio_mppi_mean | 0.9732 | 1.0000 | +0.0268 |
| jam_contact_events_mean | 0.0000 | 0.0000 | +0.0000 |
| progress_stall_events_mean | 0.0000 | 0.0000 | +0.0000 |
| spin_stall_events_mean | 0.0000 | 0.0000 | +0.0000 |

## 新口径逐 Seed

| seed | reached | final_dist | mppi_steps | override_steps | terminal_mppi_steps | speed_cap_applied_steps | mppi_reverse_raw | nominal_reverse_suppressed | final_nav_mode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 6101 | 0 | 2.0276 | 240 | 0 | 0 | 61 | 28 | 28 | NAV |
| 6102 | 0 | 1.5195 | 240 | 0 | 0 | 98 | 27 | 21 | NAV |
| 6103 | 0 | 2.0479 | 240 | 0 | 0 | 66 | 28 | 22 | NAV |
| 6104 | 0 | 1.8787 | 240 | 0 | 0 | 68 | 28 | 11 | NAV |
| 6105 | 0 | 1.7706 | 240 | 0 | 0 | 71 | 26 | 25 | NAV |
