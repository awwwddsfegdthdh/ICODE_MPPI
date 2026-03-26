# 阶段1+阶段2逐文件改造清单（执行记录）

- 日期：2026-03-20
- 说明：本清单为“本次已执行变更”

## A. 阶段0/1

1. `state_convention.py`（新增）
   - 状态/控制约定、动力学正逆映射、坐标变换、契约校验

2. `env_mujoco.py`
   - `heading_source` 默认 `base`
   - `drive_sign` 强约束 `±1`
   - 速度与坐标变换走统一约定函数

3. `mppi.py`
   - 前进/后退惩罚改为基于统一 `v`（去除重复符号耦合）

4. `run_icode_mppi_e1_test.py`
   - 默认 `planner-model=kinematic`
   - `kinematic-lock` 默认开启
   - 默认 `sensor_use_depth=False`
   - 打印 `prediction_model=kinematic_locked`
   - 坐标/逆映射走统一约定函数

5. `run_icode_mppi_e1_viewer.py`
   - 同 test 脚本策略与默认值

6. `collect_mujoco_multilayer_data.py`
   - 增加语义元信息输出
   - `drive_sign` 接入专家控制逆映射

7. `convert_multilayer_dataset.py`
   - `--drive-sign` 接入里程计
   - `derived` 重命名为来源明确字段
   - 增加 `raw__*_from_base_yaw` 标注别名

8. `prepare_icode_training_data.py`
   - 训练输入语义一致性校验（fail-fast）

9. `train_icode_and_eval.py`
   - 读取 bundle 时校验元信息
   - checkpoint 记录语义版本和符号定义

10. `test_mppi.py`
    - 默认使用 kinematic lock + depth off

## B. 阶段2

1. `obs_geometry.py`（新增）
   - 激光/深度射线模型与融合

2. `local_occupancy_map.py`（新增）
   - 局部占据图更新、障碍簇提取、LOS判定

3. `depth_consistency_gate.py`（新增）
   - 深度一致性门控

4. `mppi_nav_utils.py`
   - LOS/BFS 增加 map backend

5. `run_icode_mppi_e1_test.py`
   - 主链改为占据图提取障碍
   - LOS/BFS 使用同图 backend
   - 深度门控失败时自动回退 lidar-only

6. `run_icode_mppi_e1_viewer.py`
   - 同 test 的 map/gate 主链

## C. 新增测试

1. `tests/test_state_convention_roundtrip.py`
2. `tests/test_meta_contract.py`
3. `tests/test_convert_vs_env_rollout.py`
4. `tests/test_lidar_map_update.py`
5. `tests/test_los_consistency_map_vs_geom.py`
6. `tests/test_bfs_on_map.py`
7. `tests/test_depth_lidar_consistency_gate.py`
8. `tests/test_kinematic_lock.py`

