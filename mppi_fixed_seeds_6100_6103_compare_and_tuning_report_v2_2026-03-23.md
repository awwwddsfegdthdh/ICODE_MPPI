# Fixed-Seeds 全量对比与新主链参数收敛（6100/6101/6102/6103）

- 日期：2026-03-23
- 对比模式：old(bfs+no-reference), new(astar+reference), tuneA, tuneB
- best_mode: `old`

## 汇总表

| mode | reach_rate | collision_rate | final_dist_mean | mppi_steps_mean | goal_blocked_ratio_mean | action_post_delta_ratio_mppi_mean |
|---|---:|---:|---:|---:|---:|---:|
| old | 0.500 | 0.000 | 0.3541 | 801.0 | 0.0488 | 0.9143 |
| new | 0.250 | 0.000 | 0.3707 | 961.8 | 0.0297 | 0.8563 |
| tuneA | 0.000 | 0.000 | 0.3752 | 1000.0 | 0.0185 | 0.8898 |
| tuneB | 0.000 | 0.000 | 3.0643 | 1000.0 | 0.0000 | 0.9100 |

## 分种子结果

### old

| seed | reached_goal | collision_steps | final_dist | min_clearance | mppi_steps | override_steps | goal_blocked_ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6100 | 1 | 0 | 0.3497 | 0.7465 | 666 | 0 | 0.0000 |
| 6101 | 1 | 0 | 0.3500 | 0.2394 | 538 | 0 | 0.0000 |
| 6102 | 0 | 0 | 0.3663 | 0.3427 | 1000 | 0 | 0.0000 |
| 6103 | 0 | 0 | 0.3504 | 0.3820 | 1000 | 0 | 0.1950 |

### new

| seed | reached_goal | collision_steps | final_dist | min_clearance | mppi_steps | override_steps | goal_blocked_ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6100 | 0 | 0 | 0.3709 | 1.0645 | 1000 | 0 | 0.0000 |
| 6101 | 0 | 0 | 0.3657 | 0.3073 | 1000 | 0 | 0.0610 |
| 6102 | 0 | 0 | 0.3961 | 1.0649 | 1000 | 0 | 0.0000 |
| 6103 | 1 | 0 | 0.3500 | 0.3970 | 847 | 0 | 0.0579 |

### tuneA

| seed | reached_goal | collision_steps | final_dist | min_clearance | mppi_steps | override_steps | goal_blocked_ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6100 | 0 | 0 | 0.3865 | 1.0666 | 1000 | 0 | 0.0000 |
| 6101 | 0 | 0 | 0.3911 | 0.3704 | 1000 | 0 | 0.0290 |
| 6102 | 0 | 0 | 0.3676 | 1.0638 | 1000 | 0 | 0.0000 |
| 6103 | 0 | 0 | 0.3558 | 0.3483 | 1000 | 0 | 0.0450 |

### tuneB

| seed | reached_goal | collision_steps | final_dist | min_clearance | mppi_steps | override_steps | goal_blocked_ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 6100 | 0 | 0 | 3.3334 | 1.0696 | 1000 | 0 | 0.0000 |
| 6101 | 0 | 0 | 3.3167 | 1.1734 | 1000 | 0 | 0.0000 |
| 6102 | 0 | 0 | 2.2594 | 1.0657 | 1000 | 0 | 0.0000 |
| 6103 | 0 | 0 | 3.3476 | 1.0741 | 1000 | 0 | 0.0000 |

## 参数收敛结论

- TuneA（更激进的 reference tracker）未提升到达率，反而下降。
- TuneB（降低 tracker 最小速度、缩小 stop_dist、轻微提升 terminal 最小前进）仍未超过 old/new 的到达率。
- 因此本轮保留 `new` 作为新主链基线，下一轮应转向“终端判定语义与路径末端 target-chain”而非继续推 tracker 增益。
