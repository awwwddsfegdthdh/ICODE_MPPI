# 阶段4：传感 + blocked_conf 主链改造与 6100 回归（2026-03-22）

## 1. 执行前根因分析：为什么圆柱体会“看见但判不堵”

在当前链路里，圆柱体失效的关键不是 MuJoCo rangefinder 没有返回距离，而是**语义稀释**：

1. 圆柱在中距离的角宽较小（典型仅覆盖 1~2 束激光）。
2. `goal_blocked_conf` 原先主用占据图面积比：`occ_cells / observed_cells`。
3. 当“障碍只占 LOS 管道中的少量栅格”时，面积比会偏低，即使几何上已经穿线阻挡。
4. 结果：`goal_blocked_conf`长期偏低，`goal_blocked`很难进入，target-chain 在局部最小附近反复。

结论：问题是**blocked_conf 定义与细障碍几何阻挡语义不匹配**。

## 2. 主链改造（已落地）

### 2.1 blocked_conf 语义改造（核心）
- 文件：`mppi_nav_utils.py`
- 变更：`line_of_sight_blocked_confidence(...)`
  - 新增几何证据 `geom_conf`：基于 `obstacles_xyr` 与 start-goal 线段的最小清距。
  - 最终置信：`blocked_conf = max(occ_conf, geom_conf)`。
  - 保留 occ 分支统计，同时避免“面积比把线阻挡稀释掉”。

### 2.2 传感障碍表达改造
- 文件：`local_occupancy_map.py`
- 变更：
  - `extract_obstacles_as_circles(...)` 的半径改为 `max(rad_area, rad_extent, 0.5*res)`，避免簇半径被面积公式低估。
  - `hit_spread_cells` 参数保留（默认收敛为 0，见第4节实验结论）。

### 2.3 传感融合控制面（保留可选，不做默认）
- 文件：`run_icode_mppi_e1_test.py`、`run_icode_mppi_e1_viewer.py`
- 新增：
  - `--sensor-use-ray-obstacles/--no-sensor-use-ray-obstacles`
  - `--sensor-ray-footprint-scale`
  - `--sensor-ray-radius-max-scale`
  - `--map-hit-spread-cells`
- 默认收敛后：
  - `sensor_use_ray_obstacles=False`
  - `map_hit_spread_cells=0`

## 3. 回归口径

固定口径（非 smoke）：
- seed = 6100
- scene-seed = 6100
- planner = kinematic
- pose-source = gt
- random-obstacles + auto-waypoint + global-guide
- max-steps = 1000
- 场景约束与既有口径一致

## 4. 6100 对比结果

| 配置 | reached_goal | steps | final_dist | goal_blocked_conf_mean | 备注 |
|---|---:|---:|---:|---:|---|
| A: ray_obstacles=ON, map_spread=0 | ❌ | 1000 | 1.662 | 0.192 | 过保守，卡住 |
| B: ray_obstacles=OFF, map_spread=1 | ❌ | 1000 | 1.627 | 0.192 | 命中扩张过强，卡住 |
| C: ray_obstacles=OFF, map_spread=0（当前默认） | ✅ | 664 | 0.350 | 0.274 | 可达且无碰撞 |

结论：
1. **必须保留 blocked_conf 几何融合**（根因级修正）。
2. `ray_obstacles` 与 `hit_spread=1` 在当前参数面下会把通道压窄，导致可通过缝隙被误判，不宜作为默认主链。
3. 当前默认面收敛为：`ray_obstacles OFF + hit_spread 0 + blocked_conf 几何融合`。

## 5. 产物

- 默认回归输出：
  - `/tmp/mppi_seed6100_sensor_blockedfix_default_20260322/metrics.json`
- 对照输出：
  - `/tmp/mppi_seed6100_sensor_blockedfix_20260322/metrics.json`
  - `/tmp/mppi_seed6100_sensor_blockedfix_no_ray_mapspread1_20260322/metrics.json`

