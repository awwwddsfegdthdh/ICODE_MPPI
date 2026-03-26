# 阶段执行进展（根因级改造继续落地）

- 日期：2026-03-21
- 范围：全向感知 + 运动方向风险语义 + 常规导航前进约束 + 走廊约束一致性

## 1. 本轮已完成改造

1. 全向 lidar 感知链路打通
   - 新增后向 5 束 rangefinder 传感器，形成 10 束扫描（前5+后5）。
   - 环境侧统一输出 `lidar_scan + lidar_angles_deg`，并保留 `lidar_triplet` 兼容。

2. 运动方向风险与路径走廊约束
   - MPPI 障碍近距离代价由“车头朝向”改为“运动方向（位移方向，低速回退到航向）”。
   - 新增路径走廊代价（`cost_safe_corridor` + `path_corridor_half_width`），约束在参考路径周围穿行。

3. 控制语义：常规导航禁止倒车
   - test/viewer 两条执行链统一加入 nominal forward-only 投影。
   - 增加统计：`mppi_reverse_raw_steps`、`nominal_reverse_suppressed_steps`。

4. Omni lidar 接入后的语义一致性修正（关键）
   - 侧向扇区从“包含后半平面”改为“仅前侧扇区”，避免后向束污染 near-collision 侧向风险判定。
   - 这是全向感知上线后的必要语义收敛，而非补丁。

## 2. 文件改造清单

1. `/home/wmh/ICODE/E1_Robot/simulation/models/mjcf/E1_SimpleSensor.xml`
   - 新增：`lidar_back_left60/30`, `lidar_back`, `lidar_back_right30/60` rangefinder。

2. `/home/wmh/ICODE/domo/ICODE_MPPI/env_mujoco.py`
   - 可选 lidar 自动注册、角度推断、全扫描输出。
   - `extract_depth_features/build_sensor_obstacles/get_unified_observation` 改为全扫描输入。
   - 侧向扇区语义改为前侧扇区（排除后半平面束）。

3. `/home/wmh/ICODE/domo/ICODE_MPPI/mppi.py`
   - 新增 `cost_safe_corridor/path_corridor_half_width`。
   - 障碍风险方向从 heading 改为 motion direction。

4. `/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_test.py`
   - 感知链路切换到 `lidar_scan + lidar_angles_deg`。
   - 增加 corridor/forward-only 参数面与统计。
   - 执行链加入 nominal forward-only 投影。
   - 侧向扇区语义改为前侧扇区（与 env 一致）。

5. `/home/wmh/ICODE/domo/ICODE_MPPI/run_icode_mppi_e1_viewer.py`
   - 感知链路切换到 `lidar_scan + lidar_angles_deg`。
   - 加入 nominal forward-only 投影与 reverse 统计。

## 3. 验证结果

1. 单元测试
   - 命令：`/home/wmh/miniconda3/envs/icode_mujoco/bin/python -m pytest -q tests`
   - 结果：`22 passed`。

2. 环境扫描检查
   - 通过 `E1RobotEnv.get_lidar_scan()` 验证为 10 束。
   - 角度序列：`[-180, -150, -120, -60, -30, 0, 30, 60, 120, 150]`。

3. 固定 seed smoke（CPU，短步）
   - 输出目录：`/tmp/mppi_rootfix_smoke_cpu_20260321_v2`
   - 现象：`boundary_guard` 触发占主导（非 near 重入主导），`mppi_action_steps` 仅约 50/180。
   - 结论：链路可运行，但该 smoke 配置不代表最终收敛效果，需按既定固定 seeds 基线口径跑完整回归再判收敛。

## 4. 当前结论

1. 根因级四项改造已落地到代码链路。
2. 全向 lidar 接入后的关键语义冲突（侧向扇区含后向束）已修正。
3. 代码正确性（测试层）通过；性能收敛需下一轮按基线口径做完整固定 seeds 对比。
