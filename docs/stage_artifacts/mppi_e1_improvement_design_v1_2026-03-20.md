# MPPI E1 改进设计文档（基于审计报告第6节与第9节）

- 文档版本：v1.0
- 日期：2026-03-20
- 适用范围：`domo/ICODE_MPPI` 当前 `viewer/test/train/collect/convert` 全链路
- 目标：按“先统一语义，再统一几何，再统一决策”的顺序，完成算法主链重构，不靠补丁堆叠

---

## 1. 设计目标与边界

## 1.1 目标

1. 建立单一、可验证的状态语义与控制语义。
2. 建立激光与深度统一的局部几何表示，消除单帧伪障碍放大。
3. 将恢复策略改为层次化安全监督，与 MPPI 明确分层。
4. 重构 MPPI 成本函数，去除重复惩罚和后处理对冲。
5. 建立训练-部署一致性回放门禁，作为发布前硬条件。

## 1.2 非目标

1. 不做“继续加阈值”式补丁优化。
2. 不引入降级版控制器作为默认主链。
3. 不改变 MuJoCo 场景任务定义（起点、目标、障碍类型）本身。

---

## 2. 目标架构（To-Be）

```text
Sensors(lidar/depth/touch/wheel/imu)
  -> Obs Geometry Layer (Ray model + Local Occupancy Map)
  -> State Layer (Canonical 7D state, single convention)
  -> Safety Supervisor (hard-safety / recoverability)
  -> MPPI Planner (Cost V2 + constraints)
  -> Low-level actuator command
```

核心原则：

1. 任何模块只处理一套语义，不做双重符号解释。
2. 任何“强制动作”必须由监督层统一发出，避免多处覆盖。
3. MPPI 输出尽量直接执行，后处理只保留物理限幅。

---

## 3. 阶段一：统一状态与控制语义（对应第6.1 + 第9.1）

## 3.1 统一定义（唯一标准）

1. 机器人体坐标：`x` 前进，`y` 左侧，`yaw` 逆时针为正。
2. 控制定义：`u = [dqL, dqR]`（左右轮角速度，单位 rad/s）。
3. 运动学定义：
   - `v = drive_sign * r * 0.5 * (dqR + dqL)`
   - `w = r * (dqR - dqL) / wheel_base`
4. 7D 状态定义固定为：
   - `[x, y, yaw, v, w, dqL, dqR]`

## 3.2 参数语义解耦

1. `drive_sign`：仅用于动力学与状态构建。
2. 删除或冻结 `v_forward_sign` 的“动力学作用”，仅允许在代价里显式使用统一后的 `v`。
3. 启动时强约束：`drive_sign` 只能取 `{+1, -1}`。

## 3.3 数据契约改造

在采集、转换、训练、运行统一写入并读取以下元信息：

1. `meta__state_convention_version`
2. `meta__drive_sign`
3. `meta__pose_source`
4. `meta__heading_source`
5. `meta__yaw_source`
6. `meta__control_definition`（明确 `u=[dqL,dqR]`）

## 3.4 代码改造点

1. 新增：`state_convention.py`
   - 提供统一的状态/控制转换函数与断言。
2. 修改：`env_mujoco.py`
   - `E1RobotEnv` 默认值与 viewer/test 统一。
3. 修改：`convert_multilayer_dataset.py`
   - 显式使用 `drive_sign`，不再隐式假设。
4. 修改：`collect_mujoco_multilayer_data.py`
   - 元信息完整落盘。
5. 修改：`mppi.py` 与 `run_icode_mppi_e1_viewer.py`
   - 清理 `v_forward_sign` 语义耦合。

## 3.5 验收标准

1. 同一段 `u_t` 回放下，`convert` 重建状态与 `env.step` 状态在 `x/y/yaw/v/w` RMSE 均低于阈值。
2. 同一模型在 `pose_source=gt/odom` 下，状态语义一致且不出现符号翻转。
3. 任意运行入口不再依赖“隐式默认参数”才能跑对。

---

## 4. 阶段二：观测几何重建（对应第6.2 + 第9.2）

## 4.1 统一几何表示

采用局部占据地图（2D log-odds）作为规划几何中间层：

1. 坐标系：机器人体坐标局部地图（滚动窗口）。
2. 分辨率：`0.05m`（可配置）。
3. 范围：前向优先窗口，例如 `x:[-1.0, 4.0], y:[-2.0, 2.0]`。

## 4.2 传感器模型

1. 激光：按射线更新 free/occupied。
2. 深度：先做深度值到距离值的严格标定，再投影为射线。
3. 触碰：作为硬碰撞事件，不直接写地图占据。
4. 每条观测附置信度，进入地图融合时加权更新。

## 4.3 时序融合

1. 使用时间衰减，短时伪障碍自动衰退。
2. 引入“最小持续观测帧数”后再转高置信占据。
3. 将 `obstacles_nav` 由“散点半径化”改为“地图提取的障碍轮廓/栅格簇”。

## 4.4 LOS 与导引改造

`line_of_sight_blocked`、global path BFS、waypoint 选侧都改为基于同一地图：

1. LOS：光线穿越占据网格判定。
2. Global path：地图栅格上统一 inflation 规则。
3. Waypoint：在同一 cost map 上选取可行侧绕点。

## 4.5 代码改造点

1. 新增：`obs_geometry.py`（深度/激光统一射线模型）
2. 新增：`local_occupancy_map.py`（地图更新、衰减、提取）
3. 修改：`env_mujoco.py`（不再直接输出散点障碍为主）
4. 修改：`mppi_nav_utils.py`（LOS/BFS 统一接入地图）

## 4.6 验收标准

1. 深度开启时不再出现系统性近场伪障碍（例如长期 `<0.1m` 虚假净空）。
2. 同场景中“仅激光”与“激光+深度”规划结果方向一致率达到目标值。
3. `goal_blocked` 判定与人工可视几何一致性显著提升。

---

## 5. 阶段三：层次化安全监督（对应第6.3 + 第9.3）

## 5.1 三层控制结构

1. Layer A：硬安全（边界、接触、极低净空）  
2. Layer B：可恢复性（卡死解锁：jam/spin/progress）  
3. Layer C：性能层（MPPI）

优先级固定：A > B > C。

## 5.2 统一触发指标

将当前分散触发器收敛为四类指标：

1. `safety_risk`：碰撞/越界风险
2. `progress_rate`：任务进展导数
3. `control_effort`：控制输入与转向努力
4. `mobility`：位移与姿态变化有效性

所有触发器共享同一时间窗管理与冷却机制。

## 5.3 状态机重构

1. `SUPERVISOR_STATE` 显式枚举：
   - `NORMAL`
   - `SAFE_STOP`
   - `BACKUP_RECOVER`
   - `ROTATE_RECOVER`
   - `FORWARD_RECOVER`
2. 迁移图单点维护，禁止多处直接改 `recover_mode`。

## 5.4 代码改造点

1. 新增：`safety_supervisor.py`
2. 修改：`run_icode_mppi_e1_viewer.py`
   - 删除分散触发与分散覆盖动作逻辑，改为 supervisor 统一仲裁。

## 5.5 验收标准

1. 不再出现“jam 先触发导致 near-collision STOP 失效”。
2. 跨模式历史污染（waypoint/progress）被隔离。
3. trace 日志中每步动作来源唯一且可解释。

---

## 6. 阶段四：MPPI 成本函数与执行一致性重构（对应第6.4 + 第9.4）

## 6.1 Cost V2 结构

保留四组主项，去掉重复项：

1. 任务项：终端目标距离 + 路径跟踪主项（二选一权重主导）
2. 安全项：占据地图距离场 barrier
3. 动力学可执行项：控制平滑、速度/角速度物理约束
4. 终端稳定项：近目标停止（`v,w` 收敛）

移除或合并重复“远离目标/后退”惩罚，避免多项互相放大。

## 6.2 后处理最小化

后处理仅保留：

1. 执行器限幅
2. 最大增量限幅（可选）

`no-reverse/gap-drive` 等策略内化进优化约束，不再在优化后强改动作。

## 6.3 代码改造点

1. 修改：`mppi.py`（`compute_cost` 重构）
2. 修改：`run_icode_mppi_e1_viewer.py`
   - 删除与 Cost V2 重复的强后处理逻辑。

## 6.4 验收标准

1. MPPI 输出与执行动作差异显著下降。
2. 动作频谱振荡与“原地旋转占比”降低。
3. 不牺牲碰撞率前提下，路径效率提升。

---

## 7. 阶段五：训练-部署一致性验证门禁（对应第6.5 + 第9.5）

## 7.1 验证流程

新增 `consistency_harness.py`，每次模型候选发布前自动运行：

1. 回放一致性：`convert` 重建 vs `env.step`
2. 符号一致性：`drive_sign`/`state fields` 自动审计
3. 观测一致性：深度开启/关闭两组最小基准场景
4. 闭环一致性：`kinematic/hybrid/icode` 三模型对照

## 7.2 发布门禁（必须同时满足）

1. 状态一致性误差低于阈值
2. 无符号翻转告警
3. 安全事件率不高于基线
4. 目标到达率不低于基线

---

## 8. 实施排期（建议）

1. 第1周：阶段一（语义统一）+ 基础一致性测试脚手架
2. 第2周：阶段二（几何重建）+ LOS/BFS 接入地图
3. 第3周：阶段三（监督层）+ 阶段四（Cost V2）
4. 第4周：阶段五（门禁）+ 回归基准对比与参数收敛

---

## 9. 交付物清单

1. 设计规范文档（本文件）
2. 状态语义规范模块与断言
3. 统一观测几何模块与局部地图模块
4. 监督层模块与统一 trace
5. MPPI Cost V2 实现
6. 一致性门禁脚本与基准报告模板

---

## 10. 本阶段需要你确认的决策点

1. 最终唯一语义是否采用本文约定：`x前进, y左, yaw逆时针, u=[dqL,dqR]`。
2. 局部地图分辨率和窗口是否采用默认建议（`0.05m`, `[-1,4]x[-2,2]`）。
3. Cost V2 是否按“最小后处理”原则执行（仅保留限幅，不再策略性强改动作）。

确认后我可以直接进入“阶段一实现文档 + 文件级改造清单（逐文件）”。
