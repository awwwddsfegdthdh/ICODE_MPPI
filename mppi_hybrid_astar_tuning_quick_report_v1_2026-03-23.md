# Hybrid A* 参数收敛报告（Quick Fixed-Seeds 6100~6103）

## 1. 回归口径
- seeds: `6100/6101/6102/6103`
- 模式: `astar_baseline` + `hybrid_default` + `hybrid_a` + `hybrid_b`
- 运行参数: `--planner-model kinematic --global-guide --reference-sampling --no-goal-direct-on-clear --goal-tol 0.15 --max-steps 300`
- 结果文件: `mppi_hybrid_astar_tuning_quick_v1_2026-03-23.json`

## 2. 汇总对比（按 final_dist_mean 升序）
| mode | final_dist_mean | goal_blocked_ratio_mean | min_clearance_mean | action_post_delta_ratio_mppi_mean | mppi_steps_mean |
|---|---:|---:|---:|---:|---:|
| hybrid_a | 1.4210 | 0.1042 | 0.5254 | 0.8967 | 300.0 |
| hybrid_b | 1.4813 | 0.1458 | 0.6677 | 0.8009 | 289.0 |
| hybrid_default | 1.9293 | 0.0717 | 0.7032 | 0.9218 | 268.8 |
| astar_baseline | 2.8275 | 0.0000 | 1.0923 | 0.7458 | 300.0 |

## 3. 收敛结论
- `hybrid_a` 在该口径下综合最优（`final_dist_mean` 最低，且无碰撞）。
- 相比 `hybrid_default`：`final_dist_mean` 从 `1.9293` 降到 `1.4210`。
- `hybrid_b` 虽有部分 seed 推进较深，但稳定性与 blocked_conf 指标更差。

## 4. 默认参数落地
- 已将默认 Hybrid A* 参数改为：
  - `hybrid_n_theta = 40`
  - `hybrid_step_cells = 2.8`
  - `hybrid_turn_penalty = 0.22`
  - `hybrid_max_expansions = 14000`
- 落地文件：
  - `run_icode_mppi_e1_viewer.py`
  - `run_icode_mppi_e1_test.py`

## 5. 备注
- 本次为 quick 口径（300 steps）用于参数收敛趋势筛选；建议下一步在同参数下跑 1000 steps 全量口径确认最终收益。