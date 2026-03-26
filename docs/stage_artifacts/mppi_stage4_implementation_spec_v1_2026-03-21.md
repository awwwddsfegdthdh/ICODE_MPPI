# MPPI 阶段4实现文档（v1）

- 日期：2026-03-21
- 范围：阶段4（MPPI 成本函数与执行一致性重构）
- 前置状态（已完成）：
  - 阶段1：状态/控制语义统一（`e1_unified_v1`）
  - 阶段2：局部占据图主链 + 深度门控（默认深度关闭）
  - 阶段3：`SafetySupervisor` 已接入并完成参数收敛（`sup-*` 单一参数面）

---

## 0. 阶段4定位

阶段3已经把“安全触发与恢复”从 MPPI 中剥离；阶段4只处理两件事：

1. 重构 MPPI 成本函数，消除重复惩罚与语义重叠。
2. 收敛执行链，保证“优化输出动作”与“最终执行动作”一致（仅保留物理硬约束）。

换句话说：

- 安全由 supervisor 决策。
- MPPI 只做性能优化。
- 执行端不再二次“策略改写”。

---

## 1. 当前痛点（基于已完成代码状态）

## 1.1 成本项重复与相互对冲

`mppi.py` 当前同时存在多组“远离目标/倒退/反向运动”惩罚：

1. `w_reverse`
2. `w_away_goal`
3. `w_goal_motion_away`
4. `w_reverse_away`
5. `w_backward_step`
6. 路径项中的 `w_path_progress + w_path_backtrack`

上述项在数值上容易重复放大，导致：

1. 近障碍场景过于保守。
2. 旋转和微小振荡被多路惩罚放大。
3. 参数调优高度耦合，迁移性差。

## 1.2 执行链与优化链不一致

在 `run_icode_mppi_e1_test.py` / `run_icode_mppi_e1_viewer.py` 的 MPPI 分支中，动作会经过多级后处理：

1. `BRAKE_ALIGN` 比例缩放
2. `goal_slowdown` 比例缩放
3. `warmup` 比例缩放
4. `action_ema` 平滑
5. `max_delta_u` 斜率限制
6. `nav_no_reverse` 推前
7. `gap_drive` 限制/推前
8. `startup_no_reverse` 强制裁剪

结果是“优化器输出”并不等于“执行动作”，削弱成本函数可解释性。

---

## 2. 阶段4目标

## 2.1 主目标

1. 建立 `Cost V2`：主项少、语义清晰、参数可标定。
2. 建立“执行一致性”硬约束：MPPI 分支只允许物理级后处理。
3. 保持阶段3监督层边界不变：A/B 层仍优先于 MPPI。

## 2.2 非目标

1. 不改阶段3状态机设计。
2. 不新增补丁式触发器。
3. 不在阶段4重新启用 ICODE 预测主链。

---

## 3. 目标架构（阶段4后）

```text
SafetySupervisor (A/B 层)
  -> NORMAL 时放行 MPPI
    -> MPPI Cost V2 优化
      -> 执行器硬约束（clip + slew）
        -> env.step
```

严格约束：

1. supervisor 覆盖动作时，不进入 MPPI 后处理链。
2. MPPI 动作执行端仅保留：
   - `ctrl_low/high` 限幅
   - `max_delta_u`（斜率限制）

---

## 4. Cost V2 设计

## 4.1 成本分组

将成本统一为四组：

1. 任务组（Task）
   - 终端目标距离（`final_dist`）
   - 路径跟踪（若有 `reference_traj`）
   - 单一进展项（从“多路 away/progress”中只保留一套）

2. 安全组（Safety）
   - 碰撞 barrier（基于 `obstacles_nav`）
   - 近障碍软惩罚（平滑 barrier，避免尖峰）
   - 边界 barrier（world bounds）

3. 动力学组（Control/Dynamics）
   - 控制能量
   - 控制变化率
   - 自旋约束（必要时保留）

4. 终端稳定组（Terminal）
   - 近目标停稳（`v,w`）
   - 终端超调约束（可选）

## 4.2 需退役/合并项

阶段4明确移除（或合并到单项）以下重复项：

1. `w_away_goal`
2. `w_goal_motion_away`
3. `w_reverse_away`
4. `w_backward_step`
5. `w_path_backtrack`（并入单一 progress 语义）

保留倒退相关语义时，最多保留一项（推荐：`w_reverse`），避免双重惩罚。

## 4.3 参数命名收敛

新增 `cost-v2` 参数组，采用 `cost_*` 前缀（与阶段3 `sup_*` 一致的“单一参数面”原则）。

示例：

1. `--cost-task-final`
2. `--cost-task-track`
3. `--cost-task-progress`
4. `--cost-safe-collision`
5. `--cost-safe-near`
6. `--cost-safe-bounds`
7. `--cost-control-effort`
8. `--cost-control-smooth`
9. `--cost-terminal-stop`

旧权重参数保留一个版本兼容窗口后移除（阶段4文档先按“直接移除”执行）。

---

## 5. 执行一致性改造

## 5.1 MPPI 分支只保留硬约束

在 NORMAL + MPPI 路径中，后处理收敛为：

1. `np.clip(action, ctrl_low, ctrl_high)`
2. `max_delta_u`（可为0表示关闭）

## 5.2 需从 MPPI 分支移出的策略后处理

1. `goal_slowdown`
2. `warmup`
3. `action_ema`
4. `nav_no_reverse` 推前
5. `gap_drive` 推前/差动裁剪
6. `startup_no_reverse`

注：`DOCK_STOP` 可继续作为 supervisor 之外的任务模式分支，不属于 MPPI 优化路径。

## 5.3 一致性审计指标

在 test/viewer 增加：

1. `planner_action_raw`
2. `executed_action`
3. `action_post_delta_norm`
4. `action_post_delta_ratio`（`>eps` 的步占比）

阶段4门禁目标：

- MPPI 分支中，`action_post_delta_ratio` 显著下降（目标接近仅在限幅步触发）。

---

## 6. 具体实现方案

## 6.1 `mppi.py`

1. 引入 `Cost V2` 计算路径（可通过 `cost_version="v2"` 选择，默认直接 v2）。
2. 重写 `compute_cost`：
   - 先拆分为 `task/safety/control/terminal` 四个子函数。
   - 删除重复项，保留单一进展语义。
3. 增加 cost 分组分量输出（用于调参与审计）。

## 6.2 `run_icode_mppi_e1_test.py`

1. CLI 切换到 `cost_*` 参数组。
2. 删除 MPPI 分支中的策略后处理，仅保留硬约束。
3. rollout log 和 metrics 输出新增执行一致性指标。
4. 保持阶段3 supervisor 链路与 `sup_*` 参数不变。

## 6.3 `run_icode_mppi_e1_viewer.py`

1. 与 test 保持同构（参数、后处理、指标命名一致）。
2. trace 增加 `post_delta` 相关字段。
3. chain summary 增加 cost 分组统计（均值/分位）。

---

## 7. 测试与门禁

## 7.1 新增测试

1. `tests/test_mppi_cost_v2_redundancy.py`
   - 验证重复惩罚项已移除（接口层与数值层）。
2. `tests/test_mppi_cost_v2_progress_monotonic.py`
   - 合成轨迹下，进展更好轨迹应低 cost。
3. `tests/test_action_chain_consistency.py`
   - MPPI 分支除限幅/斜率外不应改写动作。
4. `tests/test_viewer_test_cost_contract.py`
   - viewer/test 参数与字段同构。

## 7.2 阶段4通过条件

1. `compute_cost` 不再含重复 away/backtrack 惩罚路径。
2. MPPI 分支动作后处理只剩硬约束。
3. `action_post_delta_ratio` 相比阶段3基线显著下降。
4. 在 `kinematic + lidar-only + supervisor` 默认链路下：
   - 碰撞率不劣于阶段3
   - 卡死率不劣于阶段3
   - 到达率不劣于阶段3

---

## 8. 实施顺序（阶段4）

1. 先改 `mppi.py` 成本结构（保留最小可运行）。
2. 再改 test 脚本执行链（去策略后处理，补指标）。
3. 再改 viewer 同构。
4. 增加阶段4测试。
5. 跑阶段3基线对照并形成阶段4执行报告。

---

## 9. 风险与控制

1. 风险：一次性删减成本项导致早期性能波动。
   - 控制：先做“同权重映射版本”，再做小步参数收敛。
2. 风险：移除后处理后出现短时不稳定。
   - 控制：保留 `max_delta_u` 作为唯一动态平滑硬约束。
3. 风险：viewer/test 再次分叉。
   - 控制：参数组与指标字段在两脚本同名同义。

---

## 10. 与阶段5衔接

阶段4交付后，阶段5可直接接入“训练-部署一致性门禁”：

1. 成本分组可解释性报表。
2. 动作一致性报表。
3. 多种子回归对照（阶段3 vs 阶段4）。
