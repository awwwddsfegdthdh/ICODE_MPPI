# MPPI 卡住问题根因分析（高置信）

日期：2026-03-21  
范围：阶段4 phaseB 当前参数面，在 MuJoCo 随机场景中复现“机器人在障碍附近长时间无法推进”的问题。

## 1. 复现实验与证据

### 1.1 长时步固定 seeds 复现（`pose_source=gt`, `max_steps=1200`）
输出目录：`/tmp/mppi_diag_localmin_20260321_gt`

6 个种子结果可分两类：
- 类A（明显局部卡住）：`6401/6403/6404`
  - 终点距目标：`1.24~1.48m`
  - `trigger_reason_steps` 全部是 `normal`（`1200`步）
  - `recover_events=0`
  - `target_jump_projected_steps` 高（`117~133`）
- 类B（近目标提前停住）：`6402/6405/6406`
  - 终点距目标约 `0.385m`，略高于 `goal_tol=0.35`
  - `final_nav_mode=DOCK_STOP`，`dock_stop` 占约 `77%` 步数

### 1.2 类A卡住轨迹的单步硬证据（seed 6401/6403/6404）
以 `6401` 为例：
- 最后 600 步平均推进几乎为 0：`mean_prog_last600 ≈ 2.19e-05`
- 但状态里的速度很高：`mean_v_last600 ≈ 1.16 m/s`
- 同时位置几乎不动：`mean_disp_last600 ≈ 5.88e-04 m/step`
- 条件 `|v|>0.8 且位移<0.003m` 的步占比：
  - 全程 `79.25%`
  - 最后600步 `96%`

这说明：**控制在“高轮速输出”，但机器人几乎不产生有效位移（接触/打滑/卡边缘状态）**。

### 1.3 为什么 supervisor 没触发
当前语义里，`jam/progress/spin` 判定大量依赖 `state[3]`（由轮速换算的 v）和 `goal_blocked/front_clearance` 门控。
在上述卡住轨迹中：
- 轮速换算 `v` 很高 -> 不满足低速卡死判据
- `goal_blocked_ratio≈0` -> `progress_stall` 的 blocked gate 常不成立
- 结果：长期“有效位移接近0”但仍被判为 `normal`

### 1.4 对照实验（同 seed 6401，`--oracle-mode`）
输出目录：`/tmp/mppi_diag_localmin_20260321_gt_oracle/seed_6401_oracle`

- 非 oracle：`trigger=normal:1200`，`recover=0`
- oracle：`near_collision=37`, `recover_sequence=1010`, `collision_steps=920`

说明：环境确实在接触风险区；非 oracle 链路下该风险未被有效转化为触发语义，导致“看起来正常但不前进”的假正常状态。

## 2. 根因结论（按重要性排序）

### 根因1（主因）
**“运动能力判定语义”错误地把轮速当作有效位移能力。**
在接触/打滑/卡边缘时，轮速高≠机器人在移动；但当前 supervisor 与部分判据仍以此为核心输入，导致无法识别真实卡住。

### 根因2
**卡住触发门控过强（必须 blocked/front gate），导致“未判阻塞但已零进展”无法进入恢复。**
即使连续长窗无进展，也可能一直停在 `normal`。

### 根因3
**目标链在卡住区仍持续切换（`target_jump_projected_steps` 高），叠加高曲率动作，强化局部吸引域。**
表现为目标在“goal / clipped lookahead”之间反复，小幅振荡而无净进展。

### 根因4（独立失败模式）
**DOCK_STOP 进入半径(`0.40`)大于成功阈值(`goal_tol=0.35`)**，会产生“停在 0.38x m，永不到达”的系统性失败。

## 3. 置信度评估

- 对“主因=有效位移语义错误（轮速替代位移）”的置信度：**97%**
  - 理由：在 3 个独立 seeds 中重复出现“高 v + 低位移 + 无触发 + 长时停滞”的同构证据链。
- 对“次因=触发门控过强导致假正常”的置信度：**96%**
  - 理由：`trigger_reason=normal` 全程成立，同时长窗净进展趋零。
- 对“DOCK_STOP 阈值矛盾是独立失败模式”的置信度：**99%**
  - 理由：3 个 seeds 几乎一致停在 `~0.385m` 且 `final_nav_mode=DOCK_STOP`。

## 4. 只改算法语义的收敛方案（不打补丁）

### A. 统一“有效运动”状态量（替换轮速代理）
新增监督状态：
- `v_eff`: 基于 `SE(2)` 位姿差分的有效前向速度
- `w_eff`: 基于位姿差分的有效角速度
- `progress_eff`: 窗口内目标距离净下降
- `traction_residual = |v_kin_from_u - v_eff| / (|v_kin_from_u| + eps)`

语义替换：
- `jam/progress/spin` 判定的速度项改用 `v_eff/w_eff`，轮速仅作辅因子。
- 高 `traction_residual` 连续成立时，直接进入 `contact_lock` 分支（recover 入口）。

### B. supervisor 触发语义重构为“进展优先”
- 去掉 `progress_stall` 对 blocked gate 的硬依赖；改为：
  - 主条件：`progress_eff` 与 `disp_eff` 长窗不足
  - 次条件：`blocked/front/side` 作为加权置信增强，而非硬门
- 引入单一 `stuck_conf`（替代分散阈值语义），由：
  - `progress deficit`
  - `disp deficit`
  - `traction residual`
  - `target switch rate`
  融合得到。

### C. target_chain 在卡住态下“冻结 + 单调”
- 当 `stuck_conf` 超阈：
  - 冻结 `active_target` 若干步（禁止高频跳变）
  - 全局路径索引单调递增，不允许回退/来回跳
- 解除条件：`progress_eff` 连续恢复到阈值以上。

### D. MPPI 成本语义加入“不可行动作惩罚”
在代价中增加：
- `J_stagnation`: 对“高控制能量 + 低有效位移历史”进行惩罚
- `J_switch`: 惩罚短窗内 target 高频切换

目标：把“高轮速原地磨蹭”从局部最优中剔除。

### E. 终端语义统一（DOCK 与到达）
- 约束：`dock_enter_radius <= goal_tol`
- 或把成功判据统一为 docking 语义（二选一，不能并存冲突阈值）。

## 5. 传感器可靠性结论

当前感知链在“近身接触判定”上可靠性不足（尤其非 oracle）：
- 触碰仅前向单点，不覆盖侧后向接触；
- 局部接触可出现“轮速高但位移低”而不被及时识别。

这不会阻止我们完成语义改造，但若要把该问题彻底压低到工程可接受水平，**建议补充近身接触可观测量（侧后触碰或更密近距环形测距）**。

