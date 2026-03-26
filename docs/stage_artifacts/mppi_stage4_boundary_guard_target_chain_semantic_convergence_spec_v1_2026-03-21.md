# MPPI 阶段4：`boundary_guard + target_chain` 语义收敛实施文档（中心高位 360 lidar）

- 日期：2026-03-21
- 目标：按根因链路收敛 `boundary_guard` 与 `target_chain`，避免“倒车朝 goal + 边界抢占 + 无进展”。
- 范围：先出实施文档与参数面，不在本文件内执行代码改造。

---

## 0. 结论先行（为什么要这样改）

当前失败不是单点参数问题，而是三段语义链条耦合失真：

1. 感知语义失真：
   - `LocalOccupancyMap.logodds` 初值为 `0`，而 `occupancy(threshold=0.0)` 判定为占据，导致“未知=占据”。
   - test/viewer 默认 `--map-occ-threshold=0.0`，使早期 `goal_blocked` 偏高甚至饱和。
2. `target_chain` 缺少几何不变量：
   - `goal -> guide -> waypoint` 切换时，缺少“目标可行域/走廊/跳变上限”硬约束。
   - 在 blocked 误判下，可能生成大横跳 target，拉偏运动方向。
3. `boundary_guard` 抢占语义过强：
   - 早触发后长占用，切断 MPPI 连续优化窗口，形成“越管越走不动”。

因此必须做“语义契约重构”，不是补丁加阈值。

---

## 1. 设计原则（根因级，不打补丁）

1. 感知先定语义：
   - 未知、空闲、占据必须分离，`goal_blocked` 必须带置信度。
2. `target_chain` 先于动作：
   - 任何 active_target 都要满足可行域、走廊和连续性约束。
3. `boundary_guard` 从“全程接管动作”改为“软引导优先、硬接管兜底”：
   - 能让 MPPI 持续优化就不切断。
4. 倒车语义明确分层：
   - `NAV` 主链禁止倒车；仅 `RECOVERY` 允许倒车。
5. 全链路可验收：
   - 每个状态切换都能被 step trace 解释（`action_source/trigger_reason/active_target/cost_terms`）。

---

## 2. 感知语义改造：中心高位 360 lidar

## 2.1 传感器几何（必须项）

采用**中心高位** 360 lidar（单环），避免车体前偏与低位遮挡：

1. 安装位姿（建议）：
   - `link_lidar` 相对 `base_link`：`pos=(0.069, 0.0, 0.185)`（接近质心，抬高）。
2. 角度覆盖（建议 12 束，30° 间隔）：
   - `[-180, -150, -120, -90, -60, -30, 0, 30, 60, 90, 120, 150]`。
3. 命名约束：
   - 统一 `lidar_*` 前缀，确保 `env_mujoco.py` 自动发现与角度推断一致。
4. 自身回波控制：
   - 保持 `sensor_obs_min_range >= 0.12m`，并在地图更新时屏蔽近场自体误命中。

## 2.2 占据图语义（必须项）

1. 未知不等于占据：
   - `map_occ_threshold` 默认从 `0.0` 调整到 `0.35`（提取障碍与 LOS 同步重定义）。
2. `goal_blocked` 引入置信门控：
   - 沿 goal 射线统计 `occupied_cells / observed_cells`，不再只看布尔阻挡。
3. blocked 进入/退出滞回：
   - `enter`: 置信度高于阈值且连续若干步；
   - `exit`: 置信度低于退出阈值且连续若干步。

---

## 3. `boundary_guard` 语义收敛设计

## 3.1 状态语义重构

将 `BOUNDARY_GUARD` 分为两级：

1. `BOUNDARY_SOFT`（默认）：
   - 不直接接管动作；
   - 仅替换 `active_target` 为“边界内收目标”（`boundary_recover_target`），继续由 MPPI 输出动作。
2. `BOUNDARY_HARD`（安全兜底）：
   - 仅在硬越界或紧急风险下启用 supervisor 直接动作。

## 3.2 触发与释放（新契约）

定义：
- `d_out`：到合法边界框的有符号越界距离（内为 `<=0`，外为 `>0`）。
- `v_in`：速度在“指向边界内法向”上的分量（向内为正）。

触发：
1. `HARD_ENTER`：
   - `d_out >= sup_boundary_enter_hard`（建议 `0.18m`）立即进入 `BOUNDARY_HARD`。
2. `SOFT_ENTER`：
   - `d_out >= sup_boundary_enter_soft`（建议 `0.04m`）且 `v_in < sup_boundary_min_inward_speed`（建议 `0.02m/s`），连续 `sup_boundary_soft_confirm_steps`（建议 `4`）进入 `BOUNDARY_SOFT`。

释放：
1. `d_out <= -sup_boundary_release_margin`（建议 `0.12m`）连续 `sup_boundary_release_steps`（建议 `8`）。
2. 且最近窗口有最小推进（goal 或 path 任一）：`progress >= sup_boundary_release_progress_min`（建议 `0.10m`）。

## 3.3 行为语义

1. `BOUNDARY_SOFT`：
   - 构造 `boundary_recover_target`（边界内缩带 + 全局路径最近可行点）。
   - `action_source=MPPI`，`trigger_reason=boundary_soft_target`。
2. `BOUNDARY_HARD`：
   - 允许 supervisor 覆盖动作；
   - 仅维持到满足释放条件后回落到 `BOUNDARY_SOFT/NORMAL`。

---

## 4. `target_chain` 语义收敛设计

## 4.1 统一状态机

将目标链统一为以下离散模式（显式日志）：

1. `GOAL_DIRECT`
2. `GUIDE_PATH`
3. `WAYPOINT_BYPASS`
4. `BOUNDARY_RECOVER_TARGET`
5. `DOCK`

每步只允许一个模式生效，并写入 `active_target_mode`。

## 4.2 几何不变量（必须硬约束）

对任意 `active_target`：

1. 有界：
   - 必须位于 `bounds` 内缩区域（内缩 `target_bound_margin`，建议 `0.10m`）。
2. 走廊：
   - 相对全局路径横向偏差 `|e_lat| <= path_corridor_half_width + corridor_target_slack`（建议 `0.35 + 0.15m`）。
3. 连续性：
   - 相邻两步目标跳变 `||t_k - t_{k-1}|| <= target_jump_max`（建议 `0.45m`）。
4. 单调性：
   - `guide_progress_idx` 不允许回退（或仅允许极小回退 `<=1`）。

若候选 target 违反任一约束，直接拒绝并退回上一级模式（先 `GUIDE_PATH`，再 `GOAL_DIRECT`）。

## 4.3 `goal_blocked` 与 mode 切换

1. `GOAL_DIRECT -> GUIDE_PATH`：
   - 必须满足 `goal_blocked_conf >= enter_conf` 且连续 `enter_steps`。
2. `GUIDE_PATH/WAYPOINT -> GOAL_DIRECT`：
   - 必须满足 `goal_blocked_conf <= exit_conf` 且连续 `exit_steps`，并且 heading/lateral 对齐。
3. `WAYPOINT_BYPASS`：
   - 只在 `GUIDE_PATH` 不可用或局部可行性不足时进入；
   - waypoint 生成要先过“有界 + 走廊 + LOS 双段可行 + 跳变上限”。

## 4.4 倒车语义

1. `NAV` 模式下：
   - 采样与执行双层约束 `v_cmd >= nav_min_forward_speed`（建议 `0.03m/s`）。
2. `RECOVERY` 模式下：
   - 允许倒车，仅在 `BACKUP_RECOVER` 内有效。

---

## 5. 参数面（含推荐初值）

## 5.1 感知与 blocked 语义参数

| 参数 | 含义 | 推荐值 | 说明 |
|---|---|---:|---|
| `map_occ_threshold` | 占据判定阈值 | `0.35` | 从 `0.0` 上调，修复未知=占据 |
| `goal_blocked_enter_conf` | blocked 进入置信阈值 | `0.65` | 基于占据/观测比 |
| `goal_blocked_exit_conf` | blocked 退出置信阈值 | `0.35` | 形成滞回 |
| `goal_blocked_enter_steps` | 进入连续步数 | `3` | 抑制噪声 |
| `goal_blocked_exit_steps` | 退出连续步数 | `5` | 抑制抖动 |
| `sensor_obs_min_range` | 近场屏蔽 | `0.12` | 抑制自体回波 |

## 5.2 `boundary_guard` 参数

| 参数 | 含义 | 推荐值 |
|---|---|---:|
| `sup_boundary_enter_soft` | 软越界触发阈值 | `0.04` |
| `sup_boundary_enter_hard` | 硬越界触发阈值 | `0.18` |
| `sup_boundary_min_inward_speed` | 软触发时最小内向速度门槛 | `0.02` |
| `sup_boundary_soft_confirm_steps` | 软触发确认步数 | `4` |
| `sup_boundary_release_margin` | 释放内缩边距 | `0.12` |
| `sup_boundary_release_steps` | 释放确认步数 | `8` |
| `sup_boundary_release_progress_min` | 释放最小推进量 | `0.10` |

## 5.3 `target_chain` 参数

| 参数 | 含义 | 推荐值 |
|---|---|---:|
| `target_bound_margin` | target 边界内缩 | `0.10` |
| `corridor_target_slack` | target 走廊冗余 | `0.15` |
| `target_jump_max` | 单步 target 最大跳变 | `0.45` |
| `guide_backtrack_max` | guide index 最大回退 | `0` |
| `waypoint_max_lateral` | waypoint 最大横向偏置 | `0.60` |
| `waypoint_min_forward` | waypoint 最小前向距离 | `0.20` |
| `waypoint_max_forward` | waypoint 最大前向距离 | `1.20` |
| `nav_min_forward_speed` | 导航最小前向速度 | `0.03` |

---

## 6. 验收指标（固定 seeds 全量回归口径）

## 6.1 主指标

1. `reach_rate >= 0.60`
2. `collision_rate <= 0.05`（目标 0）
3. `stuck_rate <= 0.10`
4. `final_dist_mean <= 0.80`

## 6.2 语义健康指标（必须同时满足）

1. `boundary_guard_steps_ratio <= 0.10`
2. `supervisor_override_steps_mean / total_steps <= 0.20`
3. `goal_blocked_ratio_mean <= 0.55`
4. `active_target_jump_p95 <= 0.45m`
5. `guide_backtrack_events == 0`
6. `waypoint_out_of_bounds_events == 0`
7. `mppi_reverse_raw_steps(nav) == 0`（`RECOVERY` 例外）
8. `action_post_delta_ratio_mppi <= 0.12`

## 6.3 诊断完备性

每个失败 seed 必须输出并可回放：

1. 逐步 `action_source/trigger_reason/active_target_mode/active_target_xy`
2. 逐步 `goal_blocked_conf/goal_blocked_state`
3. 逐步 `boundary_state(d_out, v_in, enter/release counters)`
4. 逐步 `cost_terms(task/safety/control/terminal/total)`

---

## 7. 实施顺序（编码阶段按此执行）

1. Phase A：感知语义先收敛
   - 中心高位 360 lidar + `map_occ_threshold` 与 `goal_blocked_conf` 落地。
2. Phase B：`target_chain` 不变量
   - 先加“有界/走廊/跳变/单调”约束，再改 mode 切换。
3. Phase C：`boundary_guard` 两级语义
   - 先 `SOFT` 目标注入，再 `HARD` 接管兜底。
4. Phase D：固定 seeds 全量回归
   - 通过主指标 + 语义健康指标后，再做参数细收敛。

---

## 8. 逐文件改造清单（下一步编码依据）

1. `/home/wmh/ICODE/E1_Robot/simulation/models/mjcf/E1_SimpleSensor.xml`
   - 雷达安装位姿改为中心高位。
   - 360 扫描束重排为 12 束（30°间隔）。
2. `/home/wmh/ICODE/domo/ICODE_MPPI/env_mujoco.py`
   - 对齐 360 束角度发现与排序。
   - 输出 blocked 置信度所需的观测证据统计。
3. `/home/wmh/ICODE/domo/ICODE_MPPI/local_occupancy_map.py`
   - 增加“observed/unknown”统计接口。
   - 保持占据判定与 LOS 语义一致。
4. `/home/wmh/ICODE/domo/ICODE_MPPI/mppi_nav_utils.py`
   - 增加 target 不变量检查工具（bounds/corridor/jump/monotonicity）。
   - waypoint 候选裁剪与可行性筛选。
5. `/home/wmh/ICODE/domo/ICODE_MPPI/safety_supervisor.py`
   - `BOUNDARY_SOFT/HARD` 语义重构。
   - 触发/释放改为 `d_out + v_in + progress` 合取。
6. `/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_test.py`
   - 参数面接入（新增 blocked_conf / boundary_soft / target invariants）。
   - trace 与 metrics 扩展（语义健康指标）。
7. `/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_viewer.py`
   - 与 test 完全同口径，保证可视化判断与回归一致。
8. `/home/wmh/ICODE/domo/ICODE_MPPI/tests/*`
   - 新增语义契约测试：blocked_conf 滞回、target 不变量、boundary 两级状态转换。

