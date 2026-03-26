# MPPI 避障全链路审计报告（E1 MuJoCo）

- 审计对象：`run_icode_mppi_e1_viewer.py` 及其直接依赖（`env_mujoco.py`、`mppi.py`、`mppi_nav_utils.py`、`run_icode_mppi_e1_test.py`），并追溯了数据链路（`collect_mujoco_multilayer_data.py`、`convert_multilayer_dataset.py`、`prepare_icode_training_data.py`）。
- 审计时间：2026-03-20
- 结论性质：代码静态审计 + 小规模实机仿真验证（MuJoCo 本地）

---

## 1. 机器人在当前定义下“能拿到什么信息”

## 1.1 原始可观测信息（非 oracle 模式）
来源：`env_mujoco.py`、`E1_SimpleSensor.xml`

1. 轮速：`wheel_l_vel`, `wheel_r_vel`
2. IMU 角速度：`imu_gyro`（用 z 轴）
3. 接触：`touch_front_force`
4. 激光：`lidar_left30`, `lidar_front`, `lidar_right30`
5. 深度图：`front_depth_cam`（通过渲染 API，而不是 sensordata）
6. 目标位置：viewer 中直接用 `env.get_goal_xy()`（`goal_pos_gt`）作为全局目标

注：非 oracle 下规划障碍不是直接读 GT 障碍体，而是由 `lidar/depth` 反投影得到 `obstacles_nav`。

## 1.2 由观测处理得到的状态
`E1RobotEnv` 输出 7D 状态：

`[x_odom, y_odom, psi_odom, v_body, wz_body, dqL, dqR]`

- `v_body = drive_sign * r * 0.5*(dqR+dqL)`
- `wz_body = yaw_blend_alpha*wz_wheel + (1-yaw_blend_alpha)*imu_wz`
- `pose_source=odom` 时，`x/y/psi` 用上式积分
- `pose_source=gt` 时，`x/y/psi` 直接由 `base_pos_gt/base_quat_gt` 给定（仍可能叠加 heading offset）

## 1.3 可靠性结论（当前代码）

1. `x/y/psi` 可靠性：中等（依赖 `pose_source` 与坐标约定一致性）
2. `v_body` 可靠性：中等（强依赖 `drive_sign` 约定）
3. `wz_body` 可靠性：中高（轮速+IMU 融合）
4. `lidar_triplet` 可靠性：中低（左/右 30 度在当前场景下有效率偏低，常回退 `default_far`）
5. 深度衍生障碍 `obstacles_nav` 可靠性：低到中（见第 4 章的深度几何问题）

---

## 2. 当前避障算法全流程（viewer 实际执行链）

每个控制步（`run_icode_mppi_e1_viewer.py` 主循环）可抽象为：

1. 采样观测：状态、目标距离、触碰、净空、障碍点集
2. 更新边界保护状态机（boundary guard）
3. 触发检测链（jam/contact/collision/stuck 等）
4. 动作仲裁（硬优先级）
5. 若进入 MPPI：目标选择（goal direct / global path / waypoint）+ MPPI 采样优化
6. 后处理（warmup/EMA/delta limit/no reverse/gap drive 等）
7. 执行动作，刷新状态

## 2.1 动作仲裁优先级（高 -> 低）

1. `BOUNDARY_GUARD`
2. `DOCK_STOP`
3. `RECOVER`（STOP->BACKUP->ROTATE->FORWARD）
4. `SPIN_BREAK`
5. `MPPI`（可能被 `progress_stalled` 当步覆盖为 `RECOVER_PROGRESS`）

## 2.2 MPPI 分支内导航目标优先级

1. `GOAL_DIRECT`（LOS 清晰且角度/横向误差满足）
2. `GLOBAL_PATH`（BFS 网格全局引导 + 参考轨迹）
3. `WAYPOINT`（自动侧绕）
4. `RAW_GOAL`

## 2.3 特殊处理触发机制（摘要）

1. `jam_contact`：大控制输入 + 小位移 + 小前向速度
2. `near_collision`：最小净空阈值或触碰阈值
3. `jam_break`：低进展 + 低位移 + 高转向努力 + 近障碍
4. `spin_break`：低进展 + 小位移 + 累计大转角
5. `progress_stalled`：路径剩余/目标距离窗口内进展不足
6. `blocked_force_waypoint`：目标受阻且短窗口进展不足
7. `adaptive_scheduler`：按净空/振荡/近目标状态切换 OPEN/TIGHT/STUCK/DOCK

这些机制本质是“启发式多状态机 + MPPI”的复合系统，不是单一优化器。

---

## 3. 特殊处理判定是否合理、是否可靠

## 3.1 合理性（正面）

1. 具备多层保护（边界、碰撞、卡死、旋转发散）
2. 有恢复相位机和 refractory，避免高频抖动重入
3. 同时有全局引导和局部绕障，理论上覆盖窄通道与局部极小值

## 3.2 可靠性风险（关键）

1. 多触发器共享同一批“非稳健特征”（`pre_min_clearance`, `path_remain_hist`, `goal_progress_hist`），一处偏差会联动误触发
2. 同一时刻触发器优先级并非按安全语义排序（见第 4.4）
3. 大量固定阈值依赖场景尺度与传感器标定，迁移性弱
4. MPPI 输出后仍被强后处理改写，削弱了“优化结果即执行”的一致性

结论：当前特殊处理“可运行”，但在观测不稳、坐标不统一时，可靠性明显下降。

---

## 4. 已确认的问题（逻辑/应用层）

以下问题均来自代码审计与本地仿真证据。

## 4.1 坐标与方向约定在链路中不统一（核心问题）

### 问题 A：训练/转换链路与运行链路采用不同状态语义

- 运行时（viewer/env/mppi）默认：`heading_source=base`, `drive_sign=-1`
- 数据转换（`convert_multilayer_dataset.py`）对 E1 数据：里程计积分未显式 `drive_sign` 参数，使用 `v = +r*avg(dq)`，且初始 yaw 取 `meta__episode_robot_init_xyyaw`

结果：`icode__x_t` 的 `psi/v_body` 语义与运行时 7D 状态并非同一坐标约定。

影响路径：

`collect/convert -> 训练 ICODE -> run_icode_mppi_e1_viewer 在线滚动`

会带来模型滚动与在线状态解释错位风险，直接影响 MPPI 代价项（尤其 heading/reverse/progress）。

### 问题 B：`v_forward_sign` 与 `drive_sign` 语义耦合

`v_forward_sign` 同时用于：

1. 动力学符号（`drive_sign` 传入模型/环境）
2. 代价“前进/倒退”判别（`v_forward = v_forward_sign * v_body`）

这种双重用途在任何一个模块改约定时都可能出现“二次翻转”。

---

## 4.2 观测链与障碍构建存在高风险几何失真

### 问题 C：深度图被直接当作距离使用，且会生成近距离伪障碍

实测（EGL 渲染，初始场景）：

- `lidar_triplet = [6.0, 1.097, 6.0]`
- 深度扇区最小值约 `[0.061, 0.066, 0.036]`
- `build_sensor_obstacles` 生成了距机器人约 `0.13m` 的点（明显偏近）

这说明深度反投影和当前几何约定下会把大量近场/非目标结构投成障碍，进而污染 `obstacles_nav` 与全局路径规划。

### 问题 D：viewer 没有像数据采集脚本那样做 GL backend 回退

`collect_mujoco_multilayer_data.py` 有 EGL/OSMesa/GLFW 探测回退；viewer/test 只做 try-create。

在无显示环境下，viewer 很可能退化到“无深度 + 稀疏激光”，与训练观测分布偏离。

---

## 4.3 控制链内部冲突与误触发风险（子代理 + 本地复核）

1. `jam_contact` 在 `near_collision` 前执行，可能先置 `recover=BACKUP`，导致更高安全优先的 STOP 当步失效
2. `blocked_force_waypoint` 未显式受 boundary guard 门控
3. `waypoint_stuck` 未限制仅 `NAV` 阶段触发，可能污染恢复阶段状态
4. adaptive 的 `osc_now` 含非 MPPI 动作变化（recover/dock/boundary），会误判 stuck
5. `goal_progress_hist` 跨模式累计，切回 NAV 后可能继承“历史低进展”误触发

---

## 4.4 训练与运行目标函数存在“重复惩罚”倾向

`mppi.py` 中对“远离目标/倒退/反向沿线”有多路并行惩罚：

- `w_goal_motion_away`
- `w_away_goal`
- `w_reverse_away`
- `w_backward_step`
- `w_reverse`

再叠加后处理 `nav_no_reverse/gap_drive`，容易出现“优化器想转，后处理强推直行”的内部张力。

---

## 5. 机器人当前“能分析的数据”与“特殊处理可靠性”匹配评估

## 5.1 能力边界

机器人当前可稳定依赖的数据：

1. 轮速、IMU z、触碰
2. 目标绝对位置（GT）
3. 稀疏激光前向量测

不稳定项：

1. 深度反投影得到的障碍点（几何风险）
2. 由不一致坐标约定衍生的 `goal_body/heading/progress` 特征

## 5.2 特殊处理可靠性判断

1. 与触碰直接相关（`near_collision` 的 touch 分支）相对可靠
2. 依赖 `pre_min_clearance`、`path_remain_hist`、`goal_blocked` 的逻辑在观测失真时可靠性明显下降
3. `progress_stalled` 与 `jam_break/spin_break` 在“观测偏差 + 后处理改写”场景下容易误触发

结论：在当前定义下，系统“能跑起来”，但稳定性主要依赖大量阈值补偿；其根因在坐标统一与观测几何一致性尚未闭环。

---

## 6. 可确定的改进建议（算法级，不是补丁）

以下建议不做降级方案，直接针对算法主链重构。

## 6.1 建立单一规范状态空间（第一优先级）

目标：全链路只允许一种 7D 语义。

1. 明确唯一导航坐标系（推荐：`x 前进, y 左, yaw 按右手系`）
2. 显式定义 `u=[dqL,dqR] -> [v,w]` 的线性映射，`drive_sign` 固化为模型参数并写入数据元信息
3. `collect/convert/train/runtime` 全部共享同一变换，不再隐式推断
4. 把 `v_forward_sign` 从“动力学符号”中移除，仅保留为代价语义开关（或直接删除，使用统一后的 `v_body`）

## 6.2 观测建模改为“几何可解释”的占据/距离场

目标：避免深度伪障碍直接污染导航。

1. 对深度渲染值做严格距离标定（相机模型 + MuJoCo 深度定义）
2. 激光与深度统一到同一 ray 模型，产出概率占据栅格/TSDF，而不是直接散点半径化
3. 在局部地图层引入时序衰减与观测置信度，防止短时伪障碍长期影响
4. `goal_blocked/line_of_sight` 基于地图而非单帧点集

## 6.3 将“恢复策略”从多触发器拼接改为层次化安全监督

目标：减少冲突优先级与误触发。

1. 第一层：硬安全约束（边界/碰撞）
2. 第二层：可行性恢复（卡死解锁）
3. 第三层：性能优化（MPPI）
4. 触发信号统一成少量可解释指标：
   - 进展导数
   - 控制努力
   - 净空统计
   - 接触事件

并在状态机中做互斥与冷却时间统一管理。

## 6.4 MPPI 成本函数去冗余并约束化

目标：避免“重复惩罚 + 后处理抵消”。

1. 合并与目标进展相关的重复项，保留一组主进展 + 一组安全障碍项
2. 将“禁止倒车/窄缝抑制旋转”从后处理挪入优化约束（或 barrier）
3. 让执行动作更接近优化动作，降低策略不一致

## 6.5 训练-部署一致性验证纳入主流程

目标：每次模型更新都做全链路一致性体检。

1. 回放同一 `u_t`，比较 `convert` 状态重建 vs `env.step` 在线状态
2. 对 `x/y/psi/v/w` 分量分别设阈值
3. 不通过则禁止该模型进入 viewer/test 主流程

---

## 7. 本次实证摘录（关键）

1. 在当前 XML，`+2,+2` 轮速控制会把机器人推向 `+x`（非反向）
2. `base_quat_gt` yaw 与 `camera_quat_gt` yaw 存在稳定大偏移（约 2.8 rad）
3. 深度开启时会出现极小“净空”与近场障碍点（即使前方激光读数约 1.1m）
4. 无显示环境下 viewer 容易退化为无深度观测（因为缺少 backend 回退）

---

## 8. 最终判断

1. 当前系统不是单一 MPPI，而是“MPPI + 多状态机补偿”
2. 在目前定义下，机器人可获得足够信息完成避障，但关键信息链（坐标、深度几何）未完全自洽
3. 主要风险不在“某个阈值”，而在“状态语义统一 + 观测几何统一”未闭环
4. 若不先统一这两件事，继续加规则会提高复杂度、降低可解释性与迁移性

---

## 9. 建议执行顺序（从根因到表现）

1. 统一状态/控制坐标语义（collect-convert-train-runtime）
2. 重建观测几何链（深度/激光 -> 统一地图）
3. 重构恢复监督层（安全优先级 + 互斥）
4. 清理 MPPI 成本与后处理重复功能
5. 建立发布前一致性回放测试

