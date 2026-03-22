# MPPI 前两阶段实现文档（v1）

- 日期：2026-03-20
- 范围：阶段1（状态/控制语义统一）+ 阶段2（观测几何重建）
- 执行前提更新：**先将 ICODE 状态预测切换为运动学状态预测（kinematic-only）**，避免模型误差污染改造评估

---

## 0. 执行策略更新（先行策略）

## 0.1 目标

在阶段1、阶段2开发与验收期间，规划预测统一使用 `DiffDriveKinematicModel`，暂不使用 `ICODEDynamics` 参与在线滚动。

## 0.2 原因

1. 当前审计已确认存在状态语义不一致风险，先做语义与观测链校正更优先。
2. 先固定预测模型（kinematic）可降低变量数量，便于定位误差来源。
3. ICODE 恢复启用应以“一致性门禁+新训练结果”达标为前提。

## 0.3 生效方式（实现约束）

1. `viewer/test` 默认 `--planner-model` 改为 `kinematic`。
2. 阶段1、2验收脚本中禁止 `planner-model=icode/hybrid`。
3. 文档与日志中显式打印 `prediction_model=kinematic_locked`。

---

## 1. 阶段1实现文档：状态与控制语义统一

## 1.1 目标

全链路（collect/convert/train/runtime）使用同一状态语义与控制语义：

- 状态：`[x, y, yaw, v, w, dqL, dqR]`
- 控制：`u=[dqL, dqR]`
- 单一符号约定：`drive_sign in {+1,-1}` 且仅用于动力学/状态构造

## 1.2 设计原则

1. 禁止“隐式默认方向”。
2. 禁止同名字段多语义。
3. 禁止 `v_forward_sign` 与 `drive_sign` 双重作用。

## 1.3 实现内容

### A. 统一语义模块（新增）

新增 `state_convention.py`，包含：

1. `StateConvention` 数据结构（版本号、坐标定义、符号定义）
2. `diff_drive_forward(u, cfg)` / `diff_drive_inverse(v,w,cfg)`
3. `world_to_body` / `body_to_world` 统一函数
4. `assert_state_contract(state)` 与 `assert_meta_contract(meta)`

### B. 运行时语义收敛

1. `env_mujoco.py`
   - `drive_sign` 输入强校验（仅 ±1）
   - `heading_source` 默认改为 `base`
   - 对 `pose_source/heading_source/drive_sign` 组合做一致性告警
2. `mppi.py`
   - 代价函数中“前进/倒退”判定只基于统一后的 `v`，去掉与 `drive_sign` 重复耦合路径
3. `run_icode_mppi_e1_viewer.py` / `run_icode_mppi_e1_test.py`
   - 预测模型默认锁定 kinematic
   - 参数打印中加入语义版本与符号版本

### C. 数据链语义收敛

1. `collect_mujoco_multilayer_data.py`
   - 输出 `meta__state_convention_version`
   - 输出 `meta__drive_sign/meta__pose_source/meta__heading_source/meta__yaw_source`
2. `convert_multilayer_dataset.py`
   - 使用显式 `drive_sign` 参与 E1 里程计重建
   - 生成字段改名防混淆：
     - `derived__goal_rel_body_from_odom_yaw`
     - `derived__goal_heading_err_from_odom_yaw`
   - 原 `raw__goal_rel_body/raw__goal_heading_err` 保留但标注 `*_from_base_yaw`
3. `prepare_icode_training_data.py`
   - 读取并校验语义元信息一致性，不一致直接 fail-fast

## 1.4 测试与门禁

1. `tests/test_state_convention_roundtrip.py`
   - `u <-> (v,w)` 往返误差阈值
2. `tests/test_convert_vs_env_rollout.py`
   - 同一 `u_t` 下 `convert` 与 `env.step` 对齐误差阈值
3. `tests/test_meta_contract.py`
   - 所有训练输入 npz 必含语义元信息

阶段1通过条件：

1. 回放一致性通过；
2. 无符号翻转告警；
3. kinematic-only 闭环结果稳定复现。

---

## 2. 阶段2实现文档：观测几何重建

## 2.1 目标

将当前“激光/深度直接散点障碍”改造为“统一射线模型 + 局部占据地图”，并用同一地图支撑 LOS/BFS/waypoint。

## 2.2 关键判断

当前深度链存在明显几何不可靠现象（近场伪障碍风险高）。  
因此阶段2按“双路径实施”：

1. `Path-A`：先完成 lidar-only 地图链路（必须项）
2. `Path-B`：深度链仅在标定通过后接入（门控项）

## 2.3 实现内容

### A. 统一观测几何层（新增）

新增 `obs_geometry.py`：

1. `LidarRayModel`
2. `DepthRayModel`（含深度到距离标定接口）
3. `fuse_rays_to_hits_free()` 统一输出 free/hit 射线集合

新增 `local_occupancy_map.py`：

1. 2D log-odds 栅格
2. 时间衰减
3. 占据簇提取（供 planner 使用）

### B. 规划接口改造

1. `mppi_nav_utils.py`
   - `line_of_sight_blocked` 支持 map backend
   - `plan_global_path_xy` 接受占据栅格输入
2. `run_icode_mppi_e1_viewer.py` / `run_icode_mppi_e1_test.py`
   - `obstacles_nav/obstacles_guide` 来源切换为地图提取结果
   - `goal_blocked` 改为 map-ray 判定

### C. 深度接入门控（必须门禁）

新增 `depth_calibration_report.md` 与自动检测脚本：

1. 单帧统计：深度最小值、分区中值、与激光前向一致性
2. 时序统计：伪近场点持续率
3. 若不达标：`sensor_use_depth` 自动降为 false（并打印原因）

## 2.4 测试与门禁

1. `tests/test_lidar_map_update.py`
2. `tests/test_los_consistency_map_vs_geom.py`
3. `tests/test_bfs_on_map.py`
4. `tests/test_depth_lidar_consistency_gate.py`

阶段2通过条件：

1. lidar-only 地图链路稳定；
2. LOS/BFS/waypoint 统一使用同一地图；
3. 深度未达标时不会污染主链；
4. kinematic-only 闭环下碰撞/卡死率较基线不劣化。

---

## 3. 执行顺序（阶段1+2）

1. 锁定 kinematic-only（先行提交）
2. 阶段1：语义模块 + 数据契约 + 回放一致性测试
3. 阶段2A：lidar-only 地图主链
4. 阶段2B：深度标定门控与可选接入
5. 阶段1+2 联合回归与报告

---

## 4. 交付物（本轮）

1. 阶段1实现说明（本文件）
2. 阶段2实现说明（本文件）
3. 逐文件改造清单（单独文件）

