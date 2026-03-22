# MPPI 阶段回归对比报告（rootcause 全量 fixed-seeds）

- 日期：2026-03-21
- 固定 seeds：`6101,6102,6103,6104,6105`
- 本次实跑输出：`/tmp/mppi_rootcause_full_eval_20260321`
- 对比基线来源：`mppi_stageB_vs_stageA_tuned_d_summary_v1_2026-03-21.json`

## 1. 执行口径（非 smoke）

1. 每个 seed 运行 `max_steps=240`。
2. 使用 tuned_d + 阶段B near 语义参数面。
3. 非 oracle，保留感知链路（用于评估 rootcause 感知/规划/控制语义联动）。

## 2. 阶段结果表（同口径对比）

| 配置 | reach_rate | collision_rate | final_dist_mean | mppi_steps_mean | override_steps_mean | goal_blocked_ratio_mean | action_post_delta_ratio_mppi_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_tuned_d(历史) | 0.000 | 0.000 | 1.537 | 186.0 | 54.0 | 0.691 | 0.0545 |
| stageB(历史) | 0.000 | 0.000 | 1.717 | 197.2 | 42.8 | 0.857 | 0.0400 |
| rootcause_now(本次实跑) | 0.000 | 0.000 | 3.035 | 69.4 | 170.6 | 1.000 | 0.4269 |

## 3. 关键结论

1. 本次 `rootcause_now` 在 fixed-seeds 上表现显著退化：`final_dist_mean` 上升到 `3.035`，`mppi_steps_mean` 降到 `69.4`，`override_steps_mean` 升到 `170.6`。
2. 失败模式高度一致：5/5 seeds 在约第 `43~44` 步进入 `boundary_guard`，并长时间停留（165~176 步）。
3. 与历史 `tuned_d/stageB` 相比，当前瓶颈不是 near/progress/spin/jam，而是 boundary_guard 早触发且占用过长。

## 4. 失败链路（逐 seed）

| seed | first_boundary_step | normal步数 | boundary_guard步数 | dist@first_boundary | final_dist | 触发链主段 |
|---:|---:|---:|---:|---:|---:|---|
| 6101 | 43 | 68 | 172 | 3.594 | 2.971 | boundary_guard[43..214] len=172 |
| 6102 | 44 | 75 | 165 | 3.592 | 2.911 | boundary_guard[44..208] len=165 |
| 6103 | 44 | 64 | 176 | 3.602 | 3.121 | boundary_guard[44..219] len=176 |
| 6104 | 44 | 67 | 173 | 3.601 | 3.118 | boundary_guard[44..216] len=173 |
| 6105 | 43 | 73 | 167 | 3.587 | 3.052 | boundary_guard[43..209] len=167 |

### 4.1 典型链路片段（seed=6101）

- action_source around boundary: `['MPPI', 'MPPI', 'MPPI', 'SUPERVISOR', 'SUPERVISOR', 'SUPERVISOR', 'SUPERVISOR']`
- trigger_reason around boundary: `['normal', 'normal', 'normal', 'boundary_guard', 'boundary_guard', 'boundary_guard', 'boundary_guard']`
- supervisor_state around boundary: `['NORMAL', 'NORMAL', 'NORMAL', 'BOUNDARY_GUARD', 'BOUNDARY_GUARD', 'BOUNDARY_GUARD', 'BOUNDARY_GUARD']`
- pre-boundary last10 mean cost: `{'task': 2304.6151123046875, 'safety': 6348.7552734375, 'control': 593.1700073242188, 'total': 9719.843017578125}`

### 4.2 统一失败机理（本次实跑）

1. MPPI 正常段在早期即把系统推到 boundary_guard 触发条件。
2. 进入 boundary_guard 后，动作源持续为 supervisor，MPPI 有效控制窗口被切断。
3. 因 boundary_guard 为主导触发，near/progress/spin/jam 都未触发。

## 5. 产物路径

1. 本次完整运行数据：`/tmp/mppi_rootcause_full_eval_20260321/seed_*/{metrics.json,rollout_log.npz,seed_*.log}`
2. 本次对比 JSON：`/tmp/mppi_rootcause_full_eval_20260321/rootcause_vs_history_report.json`
3. 归档 JSON：`/home/wmh/ICODE/domo/ICODE_MPPI/mppi_stage4_rootcause_full_regression_summary_v1_2026-03-21.json`