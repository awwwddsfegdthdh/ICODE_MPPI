# MPPI 单步级失败诊断报告（v1）

- 日期：2026-03-21
- 目的：将“第 t 步为何失败”精确到单步证据，验证之前失败归因是否成立。

## 1. 本次新增可观测字段

已在 `run_icode_mppi_e1_test.py` 增加逐步落盘：

1. `action_source_step`（每步动作来源：`MPPI` / `SUPERVISOR`）
2. `trigger_reason_step`（每步触发原因：`normal` / `near_collision` / `progress_stall` / `spin_stall` / `recover_sequence` / `dock_stop`）
3. `supervisor_state_step`（每步 supervisor 状态）
4. `active_target_step_xy`（每步 MPPI 实际优化目标）
5. `cost_task_step / cost_safety_step / cost_control_step / cost_terminal_step / cost_total_step`
   - 非 MPPI 步写入 `NaN`，避免把 supervisor 步误读为 MPPI 代价。

代码位置：
- 时序容器定义：[run_icode_mppi_e1_test.py:748](/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_test.py:748)
- 每步写入时序：[run_icode_mppi_e1_test.py:1299](/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_test.py:1299)
- 汇总与数组化：[run_icode_mppi_e1_test.py:1418](/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_test.py:1418)
- 写入 `rollout_log.npz`：[run_icode_mppi_e1_test.py:1534](/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_test.py:1534)

## 2. 复现实验与产物

### 2.1 复跑样本

1. `default` seed=6103
2. `tuned_d` seed=6101
3. `tuned_d` seed=6103
4. `tuned_d` seed=6104

输出目录：`/tmp/mppi_step_trace_diag/*`

### 2.2 结构化汇总

- 汇总 JSON：[mppi_step_trace_diagnosis_summary_v1_2026-03-21.json](/home/wmh/ICODE/domo/ICODE_MPPI/mppi_step_trace_diagnosis_summary_v1_2026-03-21.json)

## 3. 单步证据与失败链路

## 3.1 default_6103：失败主因是“早期 progress_stall 误触发 + 长恢复覆盖”

关键事件：

1. `t=50` 触发 `progress_stall`，进入 `SAFE_STOP -> recover_sequence` 长段（`t=50..116`, 共 67 步 supervisor）。
2. 触发点条件值：
   - `dist=2.433`
   - `min_clearance=0.875`（并不近障）
   - progress-window 内进展 `0.0726 < 0.08`（仅略低阈值）
3. 这段恢复中距离反而变差：`2.433 -> 2.472`。
4. `t=195` 再触发 `near_collision`（`min_clearance=0.131 < 0.18`），直到末尾仍被恢复链占用。

结论：该回合不是“MPPI没输出”，而是 supervisor 在 `t=50` 起大段接管，导致全局推进中断。

## 3.2 tuned_d_6103：失败主因是“near_collision 周期性重入，MPPI 窗口被切碎”

关键事件：

1. `near_collision` 起点：`t=139`, `t=175`, `t=211`。
2. 每次都形成约 29 步恢复段（触发 + 28 步 recover）。
3. 两段恢复之间仅约 7 步 MPPI 正常窗口，无法形成连续绕障推进。
4. 触发时净距分别：`0.165 / 0.144 / 0.131`（均低于 `near_collision_clearance=0.18`）。
5. 末段 `t=211..239` 虽有微弱推进，但总推进不足，最终 `final_dist=1.803`。

结论：之前“near_collision 主导型循环恢复”判断被单步证据确认。

## 3.3 tuned_d_6104：早期 progress_stall + 后期 near_collision 双重失败

关键事件：

1. `t=44` 触发 `progress_stall`：
   - `min_clearance=0.878`（安全净距）
   - progress-window 进展 `0.0228 < 0.03`
   - 触发后进入恢复长段（`t=44..72`）。
2. 后续 `near_collision` 在 `t=166` 与 `t=202` 触发。
3. 最后一段 `t=202..239` 距离恶化：`1.644 -> 1.655`。

结论：该回合验证了“B 层 progress_stall 在安全净距下也会触发”的问题，以及后期 near_collision 再次抢占。

## 3.4 tuned_d_6101：spin_stall 可在非近障场景触发

关键事件：

1. `t=36` 触发 `spin_stall`：
   - `min_clearance=0.716`（远高于 near 阈值）
   - spin-window 统计满足触发：
     - `spin_progress=0.0171 < 0.02`
     - `disp=0.1014 < 0.35`
     - `yaw_travel=1.9227 > 1.8`
2. 进入 29 步恢复段。
3. 虽然后半程 MPPI 步数高（211），但目标切换高（116 次）且 goal-share 低（0.338），全局收敛不足，`final_dist=1.417`。

结论：`spin_stall` 判据在“低进展+大角变”场景会触发，即使并非真正近障卡死。

## 4. 对之前结论的校验

本次单步证据对之前结论的验证结果：

1. ✅ 已确认：supervisor 抢占是核心失败通道。
   - 触发步号与持续段可直接在 trace 中复现。
2. ✅ 已确认：`near_collision` 在硬场景可形成周期性恢复重入（如 tuned_d_6103）。
3. ✅ 已确认：`action_post_delta_ratio_mppi` 下降不等于到达率提升。
   - tuned_d 场景中该值低，但仍反复触发恢复段导致失败。
4. ✅ 已确认：`progress_stall/spin_stall` 存在“非近障也触发”的路径。
   - default_6103 与 tuned_d_6104/tuned_d_6101 均有明确步级证据。

## 5. 结论（本轮可确定）

在当前定义下，到达失败并非单一原因，而是以下链路叠加：

1. B 层触发（`progress_stall` / `spin_stall`）在部分安全净距场景触发，过早拉起恢复。
2. A 层 `near_collision` 在关键路段周期性重入，使 MPPI 连续优化窗口过短。
3. 恢复链路每次持续多步，持续打断 `active_target` 的局部收敛过程。
4. 因为失败主链在 supervisor 接管，单看 MPPI 后处理一致性指标无法解释到达失败。

以上结论来自新增逐步时序字段，已具备“按 t 步回放”的证据闭环。
