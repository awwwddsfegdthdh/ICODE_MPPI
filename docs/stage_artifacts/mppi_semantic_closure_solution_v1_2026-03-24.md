# MPPI 导航语义闭环改造方案（v1, 2026-03-24）

## 0. 目标与约束
- 目标：解决“贴障推进/局部极小值/全局失败后退化盲冲/后处理放大”的系统性问题。
- 约束：
  - 不走补丁路线，按主链语义重构。
  - 不降级为单纯 conservative stop。
  - 全链路保持同一套 passability 语义。

---

## 1) 从“线”升级到“范围语义”

### 1.1 问题
当前 `goal_blocked` 近似依赖 start-goal 线判定，漏检“连线可通但走廊不可通”的情况。

### 1.2 方案
引入双语义阻塞判定：
1. `corridor_blocked_conf`（目标走廊阻塞置信）
2. `sector_blocked_conf`（前向扇区阻塞置信）

最终：
`blocked_conf = max(corridor_blocked_conf, sector_blocked_conf)`

### 1.3 计算定义
- 目标走廊：以 `base->goal_dir` 为轴，长度 `L_goal`，半宽 `W_goal`。
- 前向扇区：中心为当前运动方向或机体前向，夹角 `FOV_front`，半径 `R_front`。
- 对 occupancy + obstacle sample 计算可通行余量（clearance）并映射到 [0,1] 置信值。

### 1.4 参数面（初始建议）
- `goal_blocked_corridor_len = 1.2 m`
- `goal_blocked_corridor_half_width = 0.45 m`
- `goal_blocked_front_fov_deg = 80`
- `goal_blocked_front_range = 0.9 m`
- `goal_blocked_enter_conf = 0.60`
- `goal_blocked_exit_conf = 0.35`
- `goal_blocked_enter_steps = 3`
- `goal_blocked_exit_steps = 5`

### 1.5 验收指标
- `goal_blocked_conf` 对“连线可通但走廊卡住”场景的召回率 > 90%。
- `goal_blocked_conf_p95 >= goal_blocked_conf_mean`（统计口径正确性约束）。

---

## 2) 全局-局部衔接语义闭环

### 2.1 问题
全局规划失败时，系统退化到局部目标，但局部目标语义与全局 passability 脱节。

### 2.2 方案
建立统一 target-chain 状态机：
- `GLOBAL_TRACK`：沿全局路径
- `GLOBAL_FAIL_LATCH`：全局失败锁存，维持最近可行引导
- `LOCAL_PASSABLE_BRIDGE`：局部桥接目标（必须 passability 可行）
- `RECOVER_TRANSIT`：仅在 supervisor/recover 窗口短时激活

转换原则：
- `GLOBAL_TRACK -> GLOBAL_FAIL_LATCH`：全局 replan 失败
- `GLOBAL_FAIL_LATCH -> LOCAL_PASSABLE_BRIDGE`：锁存期且可构造局部可行桥接点
- `LOCAL_PASSABLE_BRIDGE -> GLOBAL_TRACK`：重获可行全局路径
- 任意状态 -> `RECOVER_TRANSIT`：安全状态机接管

### 2.3 关键语义
- 全局失败不等于“直接 raw_goal”，而是“沿旧可行链 + 受限桥接”。
- 局部桥接点必须通过同一 passability 判定（见第3节）。

### 2.4 参数面（初始建议）
- `guide_fail_latch_steps = 20`
- `guide_fail_latch_max_steps = 120`
- `guide_fail_backoff_mult_cap = 4`
- `guide_fail_retry_min_move = 0.12 m`
- `guide_fail_retry_min_yaw = 0.25 rad`

### 2.5 验收指标
- `guide_replans` 至少下降 50%。
- `guide_fail_steps` 不升高或下降。
- `supervisor_override_steps` 下降（同场景 seeds 对比）。

---

## 3) fallback 目标继承 passability 语义

### 3.1 问题
当前 fallback 目标可能不经过严格可通行审查，导致“全局拒绝后局部盲冲”。

### 3.2 方案
统一 `target_accept()` 函数，所有目标源必须通过：
- 来源：global lookahead / waypoint / raw_goal / recover target
- 判定：
  1. 边界合法
  2. corridor 偏移合法
  3. jump 连续性合法
  4. `passability_guard` 合法（采样最小余量 >= 阈值）

若不合法：
- 回退到“上一合法目标”；
- 若无合法目标，进入短时 `SAFE_STOP + re-evaluate`，禁止盲冲 raw_goal。

### 3.3 参数面（初始建议）
- `target_guard_min_clearance = 0.06`
- `target_guard_margin = 0.04`
- `target_guard_min_progress = 0.18`
- `target_guard_backtrack_points = 16`
- `target_guard_ray_samps = 10`

### 3.4 验收指标
- `target_passability_projected_steps` 有效但不过高（避免过严）。
- `guide_policy=RAW_GOAL` 占比显著下降（失败场景中）。

---

## 4) near-obstacle 缺乏 progress-aware 抑制

### 4.1 问题
近障时，策略可能持续输出“有控制量但无有效进展”的动作簇（贴障推进）。

### 4.2 方案
在 MPPI 代价加入 `progress-aware near-obstacle` 项：

`J_near_progress = w_np * near_risk(clearance) * stall_penalty(progress_rate, yaw_rate, wheel_diff)`

其中：
- `near_risk`：分段/阶跃（例如 clearance < 0.75m 开始上升，<0.55m 快速抬升）
- `stall_penalty`：当 `progress_rate` 小且控制努力大时急剧增加

附加语义：
- 若 `near_risk` 高且 `progress_rate` 低，抑制继续沿当前碰障方向推进（而不是全局刹停）。

### 4.3 参数面（初始建议）
- `near_progress_start = 0.75`
- `near_progress_hard = 0.55`
- `near_progress_weight = 70`
- `stall_progress_eps = 0.015 m/s`
- `stall_effort_gate = 0.45`

### 4.4 验收指标
- “贴障连续同向输出”步数下降 > 40%。
- `progress_stall_events + spin_stall_events` 下降。

---

## 5) 高频限幅/后处理放大问题

### 5.1 问题
MPPI 输出与执行动作偏差长期偏高，破坏优化一致性，放大局部极小值。

### 5.2 方案
做“约束前移”，把主要执行约束前移到采样与名义控制层：
1. 在 nominal/control sampling 就施加速度/差速可行域
2. 将后处理从“硬截断”改为“软投影 + 连续饱和”
3. 将后处理代价等效并入控制项（避免 planner unaware）

### 5.3 具体改造
- 限幅语义：
  - `u_exec = project_to_feasible_set(u_raw)`（连续投影）
  - 避免 step-to-step 硬切换
- 代价对齐：
  - 在 MPPI `control cost` 加 `projection_residual` 惩罚
- 统计监控：
  - 分解 `post_delta_ratio_mppi` 来源（speed_cap / smooth / reverse_suppress）

### 5.4 参数面（初始建议）
- `post_project_residual_weight = 0.35`
- `speed_cap_soft_alpha = 0.25`
- `delta_u_soft_clip = 0.75`

### 5.5 验收指标
- `action_post_delta_ratio_mppi` 从高位持续下降。
- `post_delta_mean_mppi` 下降且不引起碰撞率上升。

---

## 6) 日志与证据补全（必须）

新增日志字段（viewer + test）：
- `guide_fail_cause_counts: {search_fail, passability_reject, other}`
- `global_guide_replan_cooldown_final`
- `post_delta_breakdown: {speed_cap, smooth, reverse_suppress, other}`
- `target_accept_reject_reason_counts`

新增单步 trace 字段：
- `active_state`（GLOBAL_TRACK/GLOBAL_FAIL_LATCH/LOCAL_PASSABLE_BRIDGE/RECOVER_TRANSIT）
- `guide_fail_cause_t`
- `target_accept_reason_t`
- `post_project_residual_t`

---

## 7) 分阶段实施顺序

### 阶段A（优先）
- 完成第1/2/3节：blocked 语义升级 + target-chain 闭环 + fallback passability 继承。

### 阶段B
- 完成第4节：near-obstacle progress-aware 成本并收敛参数。

### 阶段C
- 完成第5节：后处理约束前移与投影一致化。

### 阶段D
- 固定 seeds 全量回归（含 6920）并输出对比表。

---

## 8) 固定 seeds 验收口径
- 种子集合：`6100/6101/6102/6103/6920`
- 统一口径：`max_steps=1000, goal_tol=0.15, no_goal_direct_on_clear`
- 重点指标：
  - reach_rate
  - collision_rate
  - progress_stall_events + spin_stall_events
  - global_guide_replans / global_guide_fail_steps
  - guide_fail_cause_counts
  - action_post_delta_ratio_mppi

通过标准（建议）：
- reach_rate 不低于当前最优基线
- collision_rate 不上升
- 6920 的 `guide_replans`、`guide_fail_steps`、`supervisor_override_steps` 显著下降
- `guide_fail_cause` 可解释且稳定
