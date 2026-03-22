# MPPI 阶段4：BFS 可通过性改造报告 v1

- 日期：2026-03-21
- 目标：修复全局 BFS 在 `occ_grid` 主链中对机器人可通过性建模不足的问题。

## 1. 根因

1. `plan_global_path_xy` 在 `occ_grid` 分支直接使用二值占据图做 BFS，未按 `robot_radius + inflation_margin` 对整图膨胀。
2. BFS 使用 8 邻接但未禁止对角穿角（corner-cut），会产生几何上不可通过的捷径。
3. 在主链（非 oracle）中，global guide 常态传入 `occ_grid`，所以该问题会直接影响在线路径。

## 2. 改造内容

1. 在 `mppi_nav_utils.py` 新增 `_inflate_occ_grid(occ, inflate_cells)`，对 `occ_grid` 做圆盘膨胀。
2. `plan_global_path_xy` 的 `occ_grid` 分支加入膨胀逻辑：
   - `inflate_cells = ceil((robot_radius + inflation_margin) / res)`
   - 膨胀后再执行 BFS。
3. `_bfs_path_cells` 与 `_bfs_path_exists` 增加对角穿角约束：
   - 对角扩展时若 `occ[r, cc]` 或 `occ[rr, c]` 为占据则跳过该邻居。

## 3. 单测增强

文件：`tests/test_bfs_on_map.py`

新增 2 条测试：
1. `test_bfs_occ_grid_respects_robot_radius_clearance`
   - 验证同一 `occ_grid` 下，小半径可通过而大半径不可通过。
2. `test_bfs_no_corner_cut_through_diagonal_gap`
   - 验证 BFS 不会走对角穿角捷径（路径长度大于 3 点）。

## 4. 验证结果

1. `PYTHONPATH=. pytest -q tests/test_bfs_on_map.py` -> `3 passed`
2. `PYTHONPATH=. pytest -q tests` -> `27 passed`
3. MuJoCo 单种子运行 sanity（seed=6102）正常完成，未报错，碰撞步数为 0。

## 5. 影响评估

1. 全局路径会更保守，窄缝可通过性判断与机器人尺寸一致。
2. guide path 质量提升：减少“路径看似可走但车体碰撞”的情况。
3. 代价：某些场景下会更容易判定“无路可走”，需配合 waypoint/recovery 语义继续收敛。

