# MPPI 阶段4 语义收敛方案（仅改判定语义，不打补丁）

- 日期：2026-03-21
- 输入证据：单步 trace 诊断
  - [mppi_step_trace_failure_diagnosis_v1_2026-03-21.md](/home/wmh/ICODE/domo/ICODE_MPPI/mppi_step_trace_failure_diagnosis_v1_2026-03-21.md)
  - [mppi_step_trace_diagnosis_summary_v1_2026-03-21.json](/home/wmh/ICODE/domo/ICODE_MPPI/mppi_step_trace_diagnosis_summary_v1_2026-03-21.json)
- 目标：先收敛 `progress_stall/spin_stall` 误触发，再收敛 `near_collision` 重入窗口。

---

## 1. 设计原则（保证不是补丁）

1. 触发必须由“单一阈值”升级为“语义证据合取”：
   - 风险证据（blocked/clearance/contact）
   - 动力学证据（高努力低位移/低有效推进）
   - 意图一致性证据（目标稳定、非正常绕行切换）
2. 触发要有确认与退出机制（进入/退出滞回 + 连续窗口确认），避免单窗噪声触发。
3. A/B 层语义解耦：
   - B 层（progress/spin）处理“非紧急卡滞”
   - A 层（near）仅处理“方向相关的即时碰撞风险”
4. 复核指标从“触发次数”升级到“触发有效性”：
   - 每次触发后 `post10_progress` 是否改善
   - 触发是否发生在安全净距场景（误触发）

---

## 2. 阶段A：先收敛 progress_stall / spin_stall 误触发

## 2.1 Progress Stall 新语义

当前问题（单步证据）：
1. `default_6103@t=50` 与 `tuned_d_6104@t=44` 在 `min_clearance≈0.88` 的安全区被触发。
2. 仅因 `progress < min_delta` 一条线触发，导致长恢复覆盖。

新判定（合取 + 连续确认）：

记窗口 `Wp`（建议 45）：
1. `goal_prog = dist_goal[t-Wp+1] - dist_goal[t]`
2. `path_prog = path_remain[t-Wp+1] - path_remain[t]`（有 global_guide 时启用）
3. `disp = ||xy[t] - xy[t-Wp+1]||`
4. `u_eff = mean(||u||)`，`v_eff = mean(|v|)`
5. `blocked_ratio = mean(goal_blocked)`
6. `clr_front_min = min(front_clearance)`

触发条件（全部满足，且连续 `confirm_windows=2`）：
1. `goal_prog < 0.015`
2. `path_prog < 0.03`（无 path 时跳过该项）
3. `disp < 0.12`
4. `u_eff > 1.8` 且 `v_eff < 0.10`
5. `blocked_ratio > 0.70` 或 `clr_front_min < 0.35`

语义解释：
- 只有“确实被阻挡 + 用力但不走 + 连续无推进”才判定 progress_stall。

## 2.2 Spin Stall 新语义

当前问题（单步证据）：
1. `tuned_d_6101@t=36` 在非近障场景触发；原条件对“正常转向找路”区分不足。

新判定（旋转主导 + 阻挡 + 目标稳定）：

记窗口 `Ws`（建议 28）：
1. `spin_prog = dist_goal[t-Ws+1] - dist_goal[t]`
2. `disp = ||xy[t] - xy[t-Ws+1]||`
3. `yaw_travel = Σ|wrap(yaw_i-yaw_{i-1})|`
4. `w_eff = mean(|wz|)`，`v_eff = mean(|v|)`
5. `blocked_ratio = mean(goal_blocked)`
6. `target_switch_count`（窗口内 active_target 切换次数）

触发条件（全部满足，连续 `confirm_windows=2`）：
1. `spin_prog < 0.01`
2. `disp < 0.18`
3. `yaw_travel > 1.8`
4. `w_eff > 0.35` 且 `v_eff < 0.08`
5. `blocked_ratio > 0.60` 或 `front_clearance_min < 0.40`
6. `target_switch_count <= 2`

语义解释：
- 排除“频繁切目标导致的主动转向”，只保留“原地旋转且确实受阻”的卡滞。

## 2.3 阶段A参数收敛建议（相对 tuned_d）

1. `sup_progress_min_delta: 0.03 -> 0.015`
2. `sup_spin_max_displacement: 0.35 -> 0.18`
3. `sup_spin_min_progress: 0.02 -> 0.01`
4. 新增：
   - `sup_progress_confirm_windows=2`
   - `sup_progress_min_u=1.8`
   - `sup_progress_max_v=0.10`
   - `sup_progress_max_disp=0.12`
   - `sup_progress_blocked_ratio_min=0.70`
   - `sup_progress_front_clearance_gate=0.35`
   - `sup_spin_confirm_windows=2`
   - `sup_spin_blocked_ratio_min=0.60`
   - `sup_spin_target_switch_max=2`
   - `sup_spin_min_wz=0.35`
   - `sup_spin_max_v=0.08`

## 2.4 阶段A验收标准

1. 误触发率：`progress_stall/spin_stall` 触发中，`min_clearance > 0.6` 占比 < 10%
2. 触发有效性：每次 stall 触发后 `post10_progress > 0` 的比例 > 70%
3. `tuned_d` 组别平均 `mppi_action_steps` 不低于当前（186）
4. 到达率至少不下降，且 `final_dist_mean` 比当前 `1.537` 再下降 ≥ 15%

---

## 3. 阶段B：再收敛 near_collision 重入窗口

## 3.1 Near 触发语义重构（方向相关风险）

当前问题（单步证据）：
1. `tuned_d_6103` 在 `t=139/175/211` 周期性重入 near，恢复段切碎 MPPI。
2. near 判定依赖 `min_clearance`（全局最近障碍）过重，方向语义不足。

新判定：

1. `near_enter` 采用方向相关风险：
   - 前向风险：`front_clearance < near_front_enter`（建议 0.14）
   - 转向侧风险：当 `|w_cmd| > w_turn_gate` 时，`side_clearance < near_side_enter`（建议 0.12）
   - 接触风险：`touch_force > threshold`
2. `global min_clearance` 仅用于 `emergency`：
   - `min_clearance < emergency_clearance`（建议 0.08）
3. 滞回退出：
   - 退出 near 需连续 `N_clear=6` 步满足
     - `front_clearance > near_front_exit`（建议 0.22）
     - `side_clearance > near_side_exit`（建议 0.18）

## 3.2 重入窗口（re-entry）语义

1. `reentry_guard_steps=24`：非 emergency 下，恢复结束后 24 步内不允许再次 near-trigger。
2. `rearm_progress=0.08`：若恢复后尚未形成最小推进（goal/path 任一），不允许再次进入完整恢复序列。
3. `emergency` 始终可抢占（安全优先）。

语义解释：
- 不是“硬冷却补丁”，而是“恢复闭环完成条件”：清障 + 最小推进 + 再武装。

## 3.3 阶段B参数建议

1. `sup_near_collision_clearance` 不再直接主触发（降级为诊断字段）
2. 新增/替换：
   - `sup_near_front_enter=0.14`
   - `sup_near_front_exit=0.22`
   - `sup_near_side_enter=0.12`
   - `sup_near_side_exit=0.18`
   - `sup_near_clear_steps=6`
   - `sup_near_reentry_guard_steps=24`
   - `sup_near_rearm_progress=0.08`
   - `sup_emergency_clearance=0.08`

## 3.4 阶段B验收标准

1. near 重入间隔中位数 > 30 步
2. `recover_sequence` 占比较当前 `tuned_d` 再下降 ≥ 30%
3. 近障安全不退化：碰撞率保持 0 或不高于当前
4. 到达率显著提升（目标先到 >= 0.4，再继续收敛）

---

## 4. 文件级改造清单（实现指引）

1. [safety_supervisor.py](/home/wmh/ICODE/domo/ICODE_MPPI/safety_supervisor.py)
   - 重写 B 层触发条件为“合取+连续确认”
   - 新增 spin/progress 的 effort 与 target-stability 门控
   - 重构 near 判定为方向相关 enter/exit + reentry guard + rearm

2. [run_icode_mppi_e1_test.py](/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_test.py)
   - 向 SupervisorInput 增补：`path_progress_recent`、`target_switch_recent`、`wz_abs_recent`（或等价统计）
   - 输出新增诊断计数：`stall_false_positive_count`、`near_reentry_count`、`recovery_effective_ratio`

3. [run_icode_mppi_e1_viewer.py](/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_viewer.py)
   - 对齐同样语义，保证 viewer/test 口径一致

---

## 5. 执行顺序（严格按你要求）

1. 先做阶段A（progress/spin）：
   - 目标是把误触发从根上降下来。
2. 阶段A通过后再做阶段B（near 重入窗口）：
   - 目标是恢复 MPPI 连续优化窗口，减少“恢复切片化”。
3. 两阶段都只改判定语义与参数面，不引入旁路补丁逻辑。

