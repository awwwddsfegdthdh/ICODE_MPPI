# 阶段1+阶段2执行实现文档（执行版）

- 日期：2026-03-20
- 执行范围：阶段1（状态/控制语义统一）+ 阶段2（观测几何重建）
- 执行前提：已按要求将主链默认锁为 **kinematic-only**；阶段2默认 **禁用深度主链，仅保留激光+触碰+轮速/IMU**

---

## 1. 阶段0/1执行结果（语义统一）

### 1.1 已落地项

1. 新增 `state_convention.py`：
   - 统一状态/控制定义：`[x,y,yaw,v,w,dqL,dqR]`、`u=[dqL,dqR]`
   - `diff_drive_forward / diff_drive_inverse`
   - `world_to_body / body_to_world`
   - `assert_state_contract / assert_meta_contract`
   - 约定版本：`e1_unified_v1`

2. 运行链路语义统一：
   - `env_mujoco.py` 默认 `heading_source="base"`
   - `drive_sign` 强约束为 `±1`
   - 状态估计中的速度计算统一走约定函数
   - 统一坐标变换函数

3. MPPI代价方向语义去耦：
   - `mppi.py` 的前进/后退惩罚改为直接基于统一状态 `v`，去除与 `v_forward_sign` 的重复耦合

4. 入口策略锁定：
   - `run_icode_mppi_e1_test.py` / `run_icode_mppi_e1_viewer.py`
   - 默认 `--planner-model kinematic`
   - 新增 `--kinematic-lock`（默认开启）
   - 锁开启时拒绝 `icode/hybrid`
   - 启动打印 `prediction_model=kinematic_locked`

5. 数据契约落地：
   - `collect_mujoco_multilayer_data.py` 输出：
     - `meta__state_convention_version`
     - `meta__drive_sign`
     - `meta__pose_source`
     - `meta__heading_source`
     - `meta__yaw_source`
     - `meta__control_definition`
   - `convert_multilayer_dataset.py`：
     - 新增 `--drive-sign`，显式参与E1里程计
     - `derived` 重命名为 yaw 来源明确字段：
       - `derived__goal_rel_body_from_odom_yaw`
       - `derived__goal_heading_err_from_odom_yaw`
     - 保留原始字段并新增标注别名：
       - `raw__goal_rel_body_from_base_yaw`
       - `raw__goal_heading_err_from_base_yaw`
   - `prepare_icode_training_data.py`：
     - 多输入语义元信息一致性校验（不一致 fail-fast）

6. 训练侧元信息透传：
   - `train_icode_and_eval.py` 训练入口校验语义元信息
   - checkpoint `config` 写入语义版本/符号定义字段

---

## 2. 阶段2执行结果（观测几何重建，深度门控）

### 2.1 已落地项

1. 新增观测几何模块：
   - `obs_geometry.py`
   - `LidarRayModel / DepthRayModel`
   - `fuse_rays_to_hits_free`

2. 新增局部占据图模块：
   - `local_occupancy_map.py`
   - log-odds 栅格更新、时间衰减
   - 由占据簇提取障碍圆（供MPPI/BFS使用）
   - 地图LOS判定能力

3. 新增深度一致性门控：
   - `depth_consistency_gate.py`
   - valid_ratio + 扇区差异 + 前向差异 + 历史失败率门控

4. `mppi_nav_utils.py` 扩展：
   - `line_of_sight_blocked` 支持 `occ_grid` backend
   - `plan_global_path_xy` 支持 `occ_grid` backend

5. `run_icode_mppi_e1_test.py` / `run_icode_mppi_e1_viewer.py` 主链改造：
   - 非 oracle 模式下，`obstacles_nav/obstacles_guide` 改由局部占据图提取
   - `goal_blocked` 使用同一张占据图 backend
   - `global guide BFS` 使用同一张占据图 backend
   - 深度只有通过 gate 才参与射线融合，否则自动回退 lidar-only

### 2.2 默认策略（已生效）

- `sensor_use_depth=False`（默认）
- 深度仅为可选增强，不会污染默认主链
- 默认主链传感器：激光 + 触碰 + 轮速/IMU

---

## 3. 门禁与验证

### 3.1 新增测试文件

- `tests/test_state_convention_roundtrip.py`
- `tests/test_meta_contract.py`
- `tests/test_convert_vs_env_rollout.py`
- `tests/test_lidar_map_update.py`
- `tests/test_los_consistency_map_vs_geom.py`
- `tests/test_bfs_on_map.py`
- `tests/test_depth_lidar_consistency_gate.py`
- `tests/test_kinematic_lock.py`

### 3.2 验证结果

1. `py_compile` 通过（核心脚本+新模块+测试文件）。
2. 手工执行测试函数集合通过：`manual_test_suite: OK`。
3. `run_icode_mppi_e1_test.py` smoke 通过（10步，kinematic_lock + depth off）。

### 3.3 当前环境限制

- 当前 Conda 环境缺少 `pytest` 模块，未执行 `pytest` 框架命令；已以等价手工调用完成同一批测试函数验证。

---

## 4. 你确认的关键策略是否落实

1. “阶段2默认禁用深度主链” ：**已落实**。
2. “先切运动学状态预测，避免ICODE误差污染改造” ：**已落实**（默认/锁定 kinematic-only）。

