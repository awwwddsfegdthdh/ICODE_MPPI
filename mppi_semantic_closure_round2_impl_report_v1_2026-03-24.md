# MPPI 语义收敛下一轮改造报告（v1, 2026-03-24）

## 1. 本轮目标
- 补齐“near-obstacle progress-aware + 动作后处理一致性”主链。
- 让 viewer 与 test 的 action post-processing 语义一致。
- 补全可观测证据：后处理分解、guide fail 类型分解、固定 seeds 回归。

## 2. 代码改造（本轮核心）

### 2.1 `mppi.py`
- 新增近障进展抑制项（near-progress penalty）：
  - 参数：`near_progress_start/hard/weight`、`near_stall_progress_eps`、`near_stall_effort_gate`。
  - 语义：在近障区，如果对 goal 的逐步进展不足且控制努力较大，增加 safety 代价，抑制“贴障硬推”。
- 保留并使用 step-wise collision penalty（near band + hit band 分层）。

### 2.2 `run_icode_mppi_e1_test.py`
- 动作后处理改为统一投影链：
  - `max_delta_u` 通过 `project_delta_action(...)` 处理。
  - 支持 `--soft-delta-projection/--no-soft-delta-projection`。
- 新增统计：
  - `delta_projection_applied_steps`
  - `action_bound_clip_steps`
  - `action_post_breakdown_steps`（speed_cap / delta_projection / reverse_suppress / bound_clip）
- 继续保留并输出 `global_guide_fail_cause_counts` 与 replan cooldown。

### 2.3 `run_icode_mppi_e1_viewer.py`
- 与 test 对齐动作后处理：
  - 原硬 `np.clip(prev±max_delta_u)` 改为 `project_delta_action(...)`。
  - 支持 `--soft-delta-projection/--no-soft-delta-projection`。
- 新增并输出后处理计数：
  - `delta_projection_applied_steps`
  - `action_bound_clip_steps`
  - `action_post_breakdown_steps`（在 `chain_summary` 打印）。

## 3. 固定 Seeds 回归（6100/6101/6102/6103）

### 3.1 运行口径
- 设备：`cpu`（当前执行环境无可用 CUDA）
- 主链：`--planner-model kinematic --global-guide --global-planner hybrid_astar --reference-sampling --no-goal-direct-on-clear`
- 场景：`--random-obstacles --scene-seed=seed --seed=seed --obs-x-range 0.8 2.2 --obs-y-range -1.2 1.2 --min-line-blockers 1 --blocker-t-range 0.20 0.90 --require-mixed-sides --scene-sample-attempts 1600`
- 终止：`--goal-tol 0.15 --max-steps 1000`

### 3.2 结果表

| seed | reached | final_dist | collision | steps | guide_fail_steps | cooldown_final | post_delta_ratio_mppi | delta_proj | speed_cap | reverse_suppress |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6100 | 0 | 1.2146 | 0 | 1000 | 473 | 35 | 0.9988 | 850 | 473 | 156 |
| 6101 | 1 | 0.1499 | 0 | 639 | 104 | 0 | 0.9984 | 639 | 381 | 42 |
| 6102 | 0 | 1.1443 | 0 | 1000 | 533 | 22 | 1.0000 | 894 | 243 | 385 |
| 6103 | 1 | 0.1499 | 0 | 629 | 0 | 0 | 1.0000 | 629 | 445 | 30 |

### 3.3 guide fail 原因分解
- 6100: `{"search_fail":0,"passability_reject":13,"other":1}`
- 6101: `{"search_fail":0,"passability_reject":11,"other":1}`
- 6102: `{"search_fail":0,"passability_reject":14,"other":1}`
- 6103: `{"search_fail":0,"passability_reject":8,"other":0}`

## 4. 本轮结论
- 后处理证据链已打通：viewer/test 都能分解到 `speed_cap / delta_projection / reverse_suppress / bound_clip`。
- `guide_fail` 类型证据完备：当前失败主因依然集中在 `passability_reject`，不是 `search_fail`。
- 固定 seeds 当前为 2/4 到达（6101、6103），6100/6102 未达，且两者 `guide_fail_steps` 显著高。

## 5. 下一轮收敛建议（不回退旧语义）
- 先压 `passability_reject` 触发密度：
  - 放宽 fallback 目标的 passability 几何阈值上界（仅在 fallback 分支，保持 global guide 主路径判据不变）。
  - 减少 repeated reject 对短时间目标切换的放大效应（依赖已接入 cooldown 语义继续收敛）。
- 再压“后处理几乎每步触发”：
  - 以 reference nominal 为中心收紧采样噪声与 jerk，目标是降低 `delta_projection_applied_steps / mppi_action_steps`。
- 同步验收指标：
  - `reached_goal_rate@6100~6103`
  - `global_guide_fail_steps`、`passability_reject` 次数
  - `action_post_delta_ratio_mppi` 与 `action_post_breakdown_steps`

