# 阶段4逐文件改造清单（v1）

- 日期：2026-03-21
- 阶段：阶段4（MPPI 成本函数与执行一致性重构）
- 前置完成状态（已对齐）：
  - 阶段1+2 已落地：`mppi_stage1_stage2_implementation_delivery_v1_2026-03-20.md`
  - 阶段3 已落地：`mppi_stage3_execution_update_v1_2026-03-21.md`
  - 阶段3第二轮参数收敛已完成：`mppi_stage3_round2_param_convergence_v1_2026-03-21.md`
  - 阶段4实现设计：`mppi_stage4_implementation_spec_v1_2026-03-21.md`

---

## 0. 执行约束（阶段4必须遵守）

1. 不改 supervisor 架构与 `sup-*` 参数面（阶段3边界保持不变）。
2. 默认链路继续：`kinematic-lock + lidar/touch/wheel/imu`（深度主链默认关闭）。
3. MPPI 分支只保留硬约束后处理：`clip + max_delta_u`。
4. 禁止保留“同语义多惩罚”与“优化后策略强改动作”。

---

## A. 新增文件

1. `domo/ICODE_MPPI/tests/test_mppi_cost_v2_redundancy.py`
   - 断言旧重复项不再参与 Cost 计算：
     - `away_goal`
     - `goal_motion_away`
     - `reverse_away`
     - `backward_step`
     - `path_backtrack`

2. `domo/ICODE_MPPI/tests/test_mppi_cost_v2_progress_monotonic.py`
   - 合成轨迹验证“更大有效进展 => 更低 task cost”。

3. `domo/ICODE_MPPI/tests/test_action_chain_consistency.py`
   - 断言 MPPI 分支动作后处理仅含：
     - 幅值限幅（`ctrl_low/high`）
     - 增量限幅（`max_delta_u`）

4. `domo/ICODE_MPPI/tests/test_viewer_test_cost_contract.py`
   - 断言 viewer/test 在 Cost V2 参数名、metrics 字段、trace 字段同构。

---

## B. 修改文件

1. `domo/ICODE_MPPI/mppi.py`
   - `MPPIController.__init__` 切换为 `cost_*` 参数组（四组语义）：
     - 任务组：`cost_task_*`
     - 安全组：`cost_safe_*`
     - 动力学组：`cost_ctrl_*`
     - 终端组：`cost_terminal_*`
   - 重构 `compute_cost`：
     - 拆为 `task/safety/control/terminal` 子成本计算。
     - 删除重复项路径：
       - `w_away_goal`
       - `w_goal_motion_away`
       - `w_reverse_away`
       - `w_backward_step`
       - `w_path_backtrack`
   - 倒退语义最多保留单一项（`reverse` 单项）。
   - 增加成本分组审计输出（用于 viewer/test 汇总），例如：
     - `self.last_cost_terms`（每组均值或终值）

2. `domo/ICODE_MPPI/run_icode_mppi_e1_test.py`
   - `build_argparser()`：
     - 新增 Cost V2 CLI：`--cost-*`
     - 退役旧成本权重入口：`--w-away-goal`、`--w-path-backtrack`、`--w-goal-motion-away`、`--w-reverse-away`
   - `MPPIController(...)` 构造参数映射到 `cost_*`。
   - `_adaptive_profile/_apply_adaptive_profile`：
     - 删除对退役旧权重的缩放项
     - 仅缩放 Cost V2 保留项
   - MPPI 分支动作链收敛：
     - 删除策略型后处理：
       - `BRAKE_ALIGN` 比例缩放（MPPI分支内）
       - `goal_slowdown`
       - `warmup`
       - `action_ema`
       - `nav_no_reverse` 推前
       - `gap_drive` 强改
       - `startup_no_reverse`
     - 保留：
       - `max_delta_u`（若 `>0`）
       - `np.clip(action, ctrl_low, ctrl_high)`
   - rollout 审计字段新增并写入 `metrics.json` / `rollout_log.npz`：
     - `planner_action_raw`
     - `executed_action`
     - `action_post_delta_norm`
     - `action_post_delta_ratio`
   - 保持阶段3已有字段不变：
     - `supervisor_summary`
     - `trigger_reason_steps`
     - `a_layer_preempt_count/b_layer_trigger_count`

3. `domo/ICODE_MPPI/run_icode_mppi_e1_viewer.py`
   - 与 test 同步：
     - Cost V2 参数名
     - 自适应缩放项
     - MPPI 分支后处理最小化策略
   - trace 输出新增：
     - `post_delta_norm`
     - 运行中 `post_delta_ratio`
   - `chain_summary` 增加：
     - Cost 分组统计（task/safety/control/terminal）
     - 执行一致性统计（post delta）
   - 保持 supervisor trace 字段命名不变（阶段3兼容）。

4. `domo/ICODE_MPPI/test_mppi.py`
   - stage4 smoke 合约升级：
     - 必须存在 `supervisor_summary`
     - 必须存在 `action_post_delta_ratio`
     - 必须存在 Cost V2 分组统计字段（若已输出到 metrics）

---

## C. 明确退役项（viewer/test 同步）

1. 旧成本参数
   - `--w-away-goal`
   - `--w-path-backtrack`
   - `--w-goal-motion-away`
   - `--w-reverse-away`

2. 旧策略后处理参数
   - `--warmup-steps`
   - `--warmup-min-scale`
   - `--goal-slowdown-radius`
   - `--goal-slowdown-min-scale`
   - `--action-ema-alpha`
   - `--nav-no-reverse` 及其配套参数
   - `--gap-drive-*`
   - `--startup-no-reverse-steps`
   - `--startup-min-u`

3. 保留硬约束参数
   - `--max-delta-u`

---

## D. 验收门禁（阶段4）

1. 代码结构门禁
   - `mppi.py` 中不再出现重复 away/backtrack 成本路径。
   - viewer/test 的 MPPI 分支不再出现策略型动作二次改写。

2. 指标门禁
   - `action_post_delta_ratio` 显著下降（接近仅限幅/斜率约束触发）。
   - `action_post_delta_norm` 分布与 `max_delta_u` 语义一致。

3. 性能门禁（对比阶段3基线）
   - 到达率不劣化
   - 碰撞率不劣化
   - 卡死率不劣化

---

## E. 执行顺序（按依赖）

1. 先改 `mppi.py`（Cost V2 成型 + 分组审计）。
2. 再改 `run_icode_mppi_e1_test.py`（参数面 + 动作链 + metrics）。
3. 再改 `run_icode_mppi_e1_viewer.py`（完全同构）。
4. 最后补齐 4 个阶段4测试文件并做 smoke。

