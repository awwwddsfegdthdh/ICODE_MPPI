# MPPI 阶段3实现文档（v1）

- 日期：2026-03-20
- 范围：阶段3（层次化安全监督）
- 关联前置：
  - 阶段1已完成：状态/控制语义统一（`e1_unified_v1`）
  - 阶段2已完成：局部占据图主链、LOS/BFS 同图、深度门控（默认深度关闭）

---

## 0. 执行前提（继承前两阶段现状）

1. 预测模型默认锁定：`kinematic-only`。
2. 默认传感器主链：激光 + 触碰 + 轮速/IMU（深度为可选增强，门控通过后才参与）。
3. 统一状态定义：`[x, y, yaw, v, w, dqL, dqR]`。
4. 规划几何来源：`local_occupancy_map`（非 oracle）。

阶段3不再新增“补丁式触发器”，而是将现有触发逻辑收敛到单一监督器。

---

## 1. 阶段3目标

## 1.1 主目标

把当前分散在 `run_icode_mppi_e1_viewer.py / run_icode_mppi_e1_test.py` 的多触发器和动作覆盖逻辑，重构为**层次化安全监督器**，并实现：

1. 固定优先级：`Layer A (硬安全) > Layer B (可恢复性) > Layer C (MPPI性能)`。
2. 每步动作来源唯一（可追溯、可解释）。
3. 触发指标统一管理（同一时间窗、同一冷却机制）。
4. 与阶段2地图链路一致（净空/阻塞/进展判定全部来源一致）。

## 1.2 非目标

1. 不新增新一轮规则叠加。
2. 不引入第二套默认控制器。
3. 不修改任务场景定义。

---

## 2. 目标架构（阶段3后）

```text
State Layer (stage1)
  + Obs Map Layer (stage2)
    -> SafetySupervisor (stage3)
      -> MPPI (allowed only when supervisor in NORMAL)
        -> actuator (only clip/slew limit)
```

### 2.1 职责边界

1. `SafetySupervisor` 负责“是否允许 MPPI 控制、是否覆盖动作、覆盖什么动作”。
2. `run_icode_mppi_e1_viewer.py / run_icode_mppi_e1_test.py` 仅负责：
   - 组装监督输入
   - 调用监督器
   - 执行动作
3. `mppi.py` 仅做优化，不再承担硬安全触发责任。

---

## 3. 监督器状态机设计

## 3.1 状态枚举

统一枚举（必须全链路一致）：

1. `NORMAL`
2. `SAFE_STOP`
3. `BACKUP_RECOVER`
4. `ROTATE_RECOVER`
5. `FORWARD_RECOVER`

可选扩展（若保留边界专门策略）：

6. `BOUNDARY_GUARD`（仍归属 Layer A）

## 3.2 分层优先级

1. Layer A（硬安全）
   - 条件：触碰、极低净空、越界/硬越界
   - 输出：`SAFE_STOP` 或 `BOUNDARY_GUARD`
2. Layer B（可恢复性）
   - 条件：卡死、原地旋转、进展停滞
   - 输出：恢复序列 `BACKUP -> ROTATE -> FORWARD`
3. Layer C（性能）
   - 条件：A/B均未触发
   - 输出：允许 MPPI

## 3.3 迁移规则（核心）

1. `NORMAL -> SAFE_STOP`：A层触发。
2. `SAFE_STOP -> BACKUP_RECOVER`：停稳窗口满足。
3. `BACKUP_RECOVER -> ROTATE_RECOVER`：后退步数达标。
4. `ROTATE_RECOVER -> FORWARD_RECOVER`：旋转步数达标。
5. `FORWARD_RECOVER -> NORMAL`：最小恢复窗口完成且风险解除。
6. 任意状态若A层再次触发，立即抢占到 `SAFE_STOP`。

---

## 4. 统一触发指标定义

## 4.1 监督输入（每步）

从前两阶段已有链路直接提供：

1. `state_t`: 7D 统一状态
2. `base_xy_t`, `goal_xy_t`
3. `touch_force_t`
4. `min_clearance_t`, `front_clearance_t`, `side_clearance_t`（来自 stage2 map/融合）
5. `goal_blocked_t`（同图LOS判定）
6. `prev_action_t`, `action_norm_t`, `turn_effort_t`
7. `in_bounds_t`, `hard_out_of_bounds_t`

## 4.2 统一窗口指标

1. `safety_risk`
   - 基于 `touch_force / clearance / boundary`
2. `progress_rate`
   - 基于 `dist_to_goal` 与 `path_remaining` 的滑窗导数
3. `control_effort`
   - 基于动作范数 + 左右轮差
4. `mobility`
   - 基于滑窗位移 + 累积偏航变化

## 4.3 冷却与重入

1. 单一冷却计时器：`recover_cooldown_steps`。
2. 冷却期内仅允许 A 层抢占，不允许 B 层重复触发。
3. 退出恢复后重置 B 层统计窗口，避免历史污染。

---

## 5. 接口设计（新增模块）

## 5.1 新增文件

`domo/ICODE_MPPI/safety_supervisor.py`

### 建议接口

1. `SupervisorConfig`
2. `SupervisorInput`
3. `SupervisorDecision`
4. `SafetySupervisor.step(inp) -> SupervisorDecision`

`SupervisorDecision` 必含：

1. `state`（当前监督状态）
2. `action_source`（`SUPERVISOR` 或 `MPPI`）
3. `override_action`（可为空）
4. `trigger_reason`（单值，禁止多源并行）
5. `transition`（是否发生状态跃迁）

---

## 6. 运行脚本改造方案

## 6.1 `run_icode_mppi_e1_viewer.py`

1. 删除分散触发器直接改 `recover_mode` 的路径。
2. 将 `jam/spin/progress/near_collision/boundary` 判定统一迁入 `SafetySupervisor`。
3. 主循环动作生成改为：
   - 先调用 supervisor
   - 若 `override_action` 存在，直接执行
   - 否则执行 MPPI
4. trace 输出统一字段：
   - `supervisor_state`
   - `trigger_reason`
   - `action_source`

## 6.2 `run_icode_mppi_e1_test.py`

1. 同 viewer 同构实现，确保离线回归一致。
2. 指标汇总新增：
   - 状态驻留步数
   - 迁移计数
   - A/B层抢占次数
   - 冷却期间被拒绝的B层触发次数

---

## 7. 与前两阶段的耦合点

1. 与阶段1耦合：
   - 所有速度/方向判断仅使用统一 `v` 语义。
2. 与阶段2耦合：
   - `goal_blocked`、净空与可通行判定必须来自同一占据图。
3. 深度策略：
   - 阶段3默认沿用阶段2的“深度主链关闭”，避免引入新变量。

---

## 8. 测试与门禁

## 8.1 新增测试

1. `tests/test_safety_supervisor_priority.py`
   - 验证 A > B > C 抢占顺序
2. `tests/test_safety_supervisor_transitions.py`
   - 验证恢复序列迁移图
3. `tests/test_safety_supervisor_cooldown.py`
   - 验证冷却与重入策略
4. `tests/test_safety_supervisor_trace_contract.py`
   - 验证每步动作来源唯一

## 8.2 回归门禁（阶段3通过条件）

1. 不再出现同一步多动作来源冲突。
2. 不再出现“低优先级触发覆盖高优先级安全动作”。
3. 恢复模式退出后，`progress/jam` 误触发率较当前基线下降。
4. 在 `kinematic + lidar-only` 默认链路下，碰撞率/卡死率不劣于阶段2基线。

---

## 9. 实施顺序（阶段3）

1. 新建 `safety_supervisor.py` 与单元测试。
2. viewer 接入 supervisor（先替换动作仲裁，不改MPPI）。
3. test 脚本接入 supervisor（保证结构同构）。
4. 统一 trace 与 metrics。
5. 跑基线回归并出阶段3执行报告。

---

## 10. 风险与控制

1. 风险：一次性替换分散逻辑可能改变现有行为边界。
   - 控制：先实现“行为等价版本”再逐项简化阈值。
2. 风险：指标窗口参数不合适会导致恢复过度/不足。
   - 控制：先固定窗口，基于回归日志做小步标定。
3. 风险：viewer/test实现分叉。
   - 控制：监督器模块化，二者共用同一实现。

