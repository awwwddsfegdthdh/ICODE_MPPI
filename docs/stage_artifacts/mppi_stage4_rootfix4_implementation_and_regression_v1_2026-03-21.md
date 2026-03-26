# 阶段4 Rootfix4 实施与回归报告

- 日期：2026-03-21
- 目标：按 4 条根因方案完成代码改造，并给出可视化命令与 fixed-seeds 对比结果。

## 1. 改造范围

1. 规划链：取消 `DOCK_STOP` 对主链控制的硬旁路，改为 terminal 区域仍使用 MPPI。
2. 控制语义：新增“净空 + 朝向”限速语义 `v_max = f(clearance, heading_error, dist_goal)`。
3. 代价函数：近障代价改为双层（中距连续约束 + 近距陡增约束）。
4. 目标链：收紧 corridor / target jump 语义，并按净空动态收缩。

## 2. 逐文件改造

1. `mppi.py`
   - 新增近障双层参数：`near_penalty_mid_clearance`、`near_penalty_hard_clearance`、`near_penalty_mid_scale`、`near_penalty_hard_scale`、`near_penalty_hard_power`。
   - 近障惩罚从单段改为双段组合。

2. `run_icode_mppi_e1_test.py`
   - 删除 `DOCK_STOP -> dock_action` 主链分支，保留 `BRAKE_ALIGN` 仅作语义状态。
   - 新增动作后处理：
     - `project_nominal_forward_action(..., terminal_mode)`
     - `apply_semantic_speed_cap(...)`
   - 新增 `dynamic_target_corridor_max_dev(...)`，并在 `project_target_with_invariants` 中使用动态 corridor/jump。
   - terminal 区域对 MPPI 应用 `DOCK` 语义 profile（仍是 MPPI）。
   - 新增指标：
     - `terminal_mppi_steps`
     - `speed_cap_applied_steps`

3. `run_icode_mppi_e1_viewer.py`
   - 与 test 链路同构改造：
     - 移除 `DOCK_STOP` 控制旁路
     - 新增 terminal MPPI + 语义限速 + corridor 动态收缩
     - 输出新增 `terminal_mppi_steps` / `speed_cap_applied_steps`

## 3. 参数面（当前默认）

1. 近障双层：
   - `near_penalty_clearance=0.20`
   - `near_penalty_mid_clearance=0.45`
   - `near_penalty_hard_clearance=0.20`
   - `near_penalty_mid_scale=0.20`
   - `near_penalty_hard_scale=1.00`
2. 走廊与目标链：
   - `path_corridor_half_width=0.25`
   - `corridor_target_slack=0.08`
   - `corridor_tight_clearance=0.60`
   - `corridor_tight_max_dev=0.20`
   - `target_jump_max=0.45`
   - `target_jump_min=0.24`
3. 速度语义：
   - `nav_speed_max=1.35`
   - `speed_cap_clearance_hard=0.18`
   - `speed_cap_clearance_soft=0.70`
   - `speed_cap_heading_gate=0.95`
   - `speed_cap_heading_min_gain=0.55`
   - `terminal_mppi_enter_radius=0.65`
   - `terminal_profile_alpha=0.20`

## 4. 验证

1. 语法检查：通过。
2. `pytest -q tests`：`25 passed`。

## 5. fixed-seeds 对比（6101~6105）

对比文件：
- `/home/wmh/ICODE/domo/ICODE_MPPI/mppi_stage4_rootfix4_fixed_seeds_compare_summary_v2_2026-03-21.json`
- `/home/wmh/ICODE/domo/ICODE_MPPI/mppi_stage4_rootfix4_fixed_seeds_compare_report_v2_2026-03-21.md`

关键聚合：
1. 旧口径（240步）：
   - `reach_rate=0.000`
   - `final_dist_mean=0.3785`
2. 新口径（240步）：
   - `reach_rate=0.000`
   - `final_dist_mean=0.9729`
   - `terminal_mppi_steps_mean=1.6`
   - `speed_cap_applied_steps_mean=35.0`
3. 新口径（1000步，C2b 参数口径）：
   - `reach_rate=0.600`
   - `final_dist_mean=0.3547`
   - `collision_rate=0.000`

解释：
1. 去除 `DOCK_STOP` 旁路后，240 步口径下更保守，近目标收敛变慢。
2. 在 1000 步口径下可恢复有效到达（3/5 到达），说明当前瓶颈主要是“收敛时间”而非“链路失效”。
