# 阶段1+阶段2逐文件改造清单（v1）

- 日期：2026-03-20
- 对齐文档：`mppi_stage1_stage2_implementation_spec_v1_2026-03-20.md`
- 说明：本清单为“实现任务清单”，不是已完成代码变更记录

---

## A. 阶段0（先行策略：kinematic-only）

1. `domo/ICODE_MPPI/run_icode_mppi_e1_viewer.py`
   - 将 `--planner-model` 默认值改为 `kinematic`
   - 增加 `kinematic_lock` 开关（默认开启）
   - 当 lock 开启时拒绝 `icode/hybrid`
   - 启动日志输出 `prediction_model=kinematic_locked`

2. `domo/ICODE_MPPI/run_icode_mppi_e1_test.py`
   - 同步上述默认值与锁逻辑
   - 输出同样的锁定日志字段

3. `domo/ICODE_MPPI/test_mppi.py`
   - 更新用例默认预测模型为 kinematic
   - 增加“锁开启时模型选择应失败/回退”测试

---

## B. 阶段1（状态/控制语义统一）

## B1. 新增文件

1. `domo/ICODE_MPPI/state_convention.py`
   - `StateConvention` 定义
   - `diff_drive_forward/inverse`
   - `world_to_body/body_to_world`
   - `assert_*` 合约函数

2. `domo/ICODE_MPPI/tests/test_state_convention_roundtrip.py`
   - `u <-> (v,w)` 往返一致性测试

3. `domo/ICODE_MPPI/tests/test_convert_vs_env_rollout.py`
   - `convert` 与 `env.step` 状态对齐测试

4. `domo/ICODE_MPPI/tests/test_meta_contract.py`
   - npz 语义元信息完整性测试

## B2. 修改文件

1. `domo/ICODE_MPPI/env_mujoco.py`
   - `drive_sign` 约束为 ±1
   - 默认 `heading_source` 调整为 `base`
   - 统一调用 `state_convention.py` 的旋转/动力学函数
   - 输出状态语义版本字段（供调试）

2. `domo/ICODE_MPPI/mppi.py`
   - 清理 `v_forward_sign` 与 `drive_sign` 的重复耦合
   - 代价中的前进判定改为统一 `v` 语义

3. `domo/ICODE_MPPI/run_icode_mppi_e1_viewer.py`
   - 启动时打印 `state_convention_version`
   - 读取并检查元信息一致性（若有数据回放模式）
   - 替换本地 `world_to_body` 为公共函数（避免漂移）

4. `domo/ICODE_MPPI/run_icode_mppi_e1_test.py`
   - 同 viewer 的语义版本打印与一致性断言

5. `domo/ICODE_MPPI/collect_mujoco_multilayer_data.py`
   - 在输出 `npz` 写入：
     - `meta__state_convention_version`
     - `meta__drive_sign`
     - `meta__pose_source`
     - `meta__heading_source`
     - `meta__yaw_source`
     - `meta__control_definition`

6. `domo/ICODE_MPPI/convert_multilayer_dataset.py`
   - 增加 `--drive-sign` 参数并参与里程计积分
   - `derived` 字段按 yaw 来源重命名，避免同名异义
   - 传播/校验语义元信息

7. `domo/ICODE_MPPI/prepare_icode_training_data.py`
   - 训练输入数据语义一致性校验
   - 发现不一致直接 fail-fast

8. `domo/ICODE_MPPI/train_icode_and_eval.py`
   - 保存模型 checkpoint 时记录语义版本和符号约定
   - 加载评估时校验与运行时语义一致

9. `domo/ICODE_MPPI/README.md`（若存在）
   - 更新状态/控制定义与参数语义说明

---

## C. 阶段2（观测几何重建）

## C1. 新增文件

1. `domo/ICODE_MPPI/obs_geometry.py`
   - 激光/深度统一射线模型
   - 射线融合输出接口

2. `domo/ICODE_MPPI/local_occupancy_map.py`
   - 局部占据栅格（log-odds）
   - 时间衰减
   - 障碍簇提取

3. `domo/ICODE_MPPI/depth_consistency_gate.py`
   - 深度与激光一致性检测
   - 深度接入门控判定

4. `domo/ICODE_MPPI/tests/test_lidar_map_update.py`
5. `domo/ICODE_MPPI/tests/test_los_consistency_map_vs_geom.py`
6. `domo/ICODE_MPPI/tests/test_bfs_on_map.py`
7. `domo/ICODE_MPPI/tests/test_depth_lidar_consistency_gate.py`

## C2. 修改文件

1. `domo/ICODE_MPPI/env_mujoco.py`
   - `build_sensor_obstacles` 保留兼容但降级为过渡接口
   - 新增“原始观测输出”接口，交给几何层统一处理

2. `domo/ICODE_MPPI/mppi_nav_utils.py`
   - `line_of_sight_blocked` 增加 map backend
   - `plan_global_path_xy` 支持直接输入占据栅格

3. `domo/ICODE_MPPI/run_icode_mppi_e1_viewer.py`
   - 用局部地图替换 `obstacles_nav/obstacles_guide` 的构建方式
   - `goal_blocked` 改为 map-ray 判定
   - 深度接入前先过 `depth_consistency_gate`
   - gate 不通过自动切换 lidar-only，并记录日志

4. `domo/ICODE_MPPI/run_icode_mppi_e1_test.py`
   - 同 viewer 的 map/gate 逻辑，便于离线回归一致

5. `domo/ICODE_MPPI/collect_mujoco_multilayer_data.py`
   - 可选保存射线/地图监督字段（为后续训练做准备）

6. `domo/ICODE_MPPI/convert_multilayer_dataset.py`
   - 可选生成 map-friendly 观测字段（例如统一射线特征）

---

## D. 本轮必须先确认后再写代码的点

1. 是否立即把 `viewer/test` 的默认模型强制为 `kinematic`（建议：是）。
2. 阶段2是否采用“深度默认关闭，先跑通 lidar-only 地图”（建议：是）。
3. 局部地图默认配置：
   - 分辨率 `0.05m`
   - 范围 `x:[-1,4], y:[-2,2]`
   - 更新率与控制率同步

---

## E. 风险备注（需要你知晓）

当前深度观测链在现有配置下存在明显近场伪障碍风险。  
因此阶段2实现中，我会把“深度接入”作为门控特性，不会直接放进主链默认路径；默认先用 lidar-only 地图确保可解释和稳定。

