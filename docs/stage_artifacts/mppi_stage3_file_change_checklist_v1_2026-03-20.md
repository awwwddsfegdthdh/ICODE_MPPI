# 阶段3逐文件改造清单（v1）

- 日期：2026-03-20
- 阶段：阶段3（层次化安全监督）
- 依赖前置：
  - 阶段1已完成：统一语义与数据契约
  - 阶段2已完成：占据图主链 + 深度门控（默认深度关闭）

---

## A. 新增文件

1. `domo/ICODE_MPPI/safety_supervisor.py`
   - 定义 `SupervisorState` 枚举
   - 定义 `SupervisorConfig / SupervisorInput / SupervisorDecision`
   - 实现 `SafetySupervisor.step()`
   - 内部维护统一滑窗与冷却计时器

2. `domo/ICODE_MPPI/tests/test_safety_supervisor_priority.py`
   - 覆盖 A > B > C 抢占顺序

3. `domo/ICODE_MPPI/tests/test_safety_supervisor_transitions.py`
   - 覆盖状态迁移图（`SAFE_STOP -> BACKUP -> ROTATE -> FORWARD -> NORMAL`）

4. `domo/ICODE_MPPI/tests/test_safety_supervisor_cooldown.py`
   - 覆盖冷却与重入逻辑

5. `domo/ICODE_MPPI/tests/test_safety_supervisor_trace_contract.py`
   - 覆盖每步动作来源唯一、reason 唯一

---

## B. 修改文件

1. `domo/ICODE_MPPI/run_icode_mppi_e1_viewer.py`
   - 删除分散触发器直接改写 `recover_mode` 的路径
   - 引入 `SafetySupervisor` 统一仲裁
   - 动作主链改为：`supervisor_decision -> (override or MPPI)`
   - trace 字段统一为：
     - `supervisor_state`
     - `trigger_reason`
     - `action_source`
     - `transition`

2. `domo/ICODE_MPPI/run_icode_mppi_e1_test.py`
   - 同 viewer 做同构改造（防止仿真/回归行为分叉）
   - metrics 新增：
     - 各状态驻留步数
     - 状态迁移计数
     - A层抢占计数 / B层触发计数
     - cooldown 拒绝次数

3. `domo/ICODE_MPPI/test_mppi.py`
   - 增加 supervisor 默认配置 smoke 校验入口

---

## C. 参数与配置收敛（新增CLI）

在 `viewer/test` 中新增/收敛如下参数组（由 supervisor 消费）：

1. 安全阈值组（A层）
   - 触碰阈值、最小净空阈值、越界阈值
2. 进展/机动性组（B层）
   - 进展窗口、最小进展量、最小位移、最小偏航变化
3. 冷却组
   - `recover_cooldown_steps`
4. 恢复序列组
   - stop/back/rotate/forward 的持续步数与动作参数

要求：

1. 删除同语义重复阈值（避免并行触发器多处定义）。
2. 所有恢复相关参数只保留 supervisor 一处定义。

---

## D. 需要移除或退役的旧逻辑（viewer/test）

以下逻辑将由 supervisor 吸收，原地退役：

1. `jam_contact` 分散触发
2. `near_collision` 分散触发
3. `jam_break` 分散触发
4. `spin_break` 分散触发
5. `progress_stalled` 分散触发
6. 直接在多处分支写 `recover_mode/recover_phase_left`

---

## E. 与前两阶段的衔接约束（必须满足）

1. 仅使用阶段1统一语义状态（不得重新解释方向符号）。
2. 仅使用阶段2统一地图链路产出的净空/阻塞判定。
3. 默认继续 `kinematic-lock + no-depth-main-chain`，避免阶段3引入额外变量。

---

## F. 验收门禁清单

1. 单步动作来源唯一（`SUPERVISOR` 或 `MPPI`，不能并存）。
2. 高优先级安全触发不被低优先级逻辑覆盖。
3. 状态迁移图无非法跳转。
4. cooldown 生效且可审计。
5. 在默认链路（kinematic + lidar-only）下：
   - 碰撞率不劣于阶段2基线
   - 卡死率不劣于阶段2基线

---

## G. 交付建议（阶段3执行后）

1. 阶段3执行报告（对比阶段2基线）
2. supervisor trace 样例（成功/失败各1）
3. 参数表（默认值 + 标定范围）

