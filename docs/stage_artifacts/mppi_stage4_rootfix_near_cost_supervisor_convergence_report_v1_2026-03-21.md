# MPPI 阶段4 根因改造执行报告（近障代价 + Supervisor 收敛）

- 日期：2026-03-21
- 改造目标：
1. 障碍代价只在“非常近”时才明显惩罚，鼓励穿行可行缝隙。
2. 针对结论落地：
   - 倒车主来源是 Supervisor/DOCK，而非 MPPI 主链。
   - `jam_contact` 误触发是倒车恢复链条起点。
   - 激光改造后需要补齐全链路收敛（触发语义、转向判定、后向安全）。

## 1. 代码改造摘要

### 1.1 近障代价语义（`mppi.py`）
1. 新增参数：`near_penalty_clearance`（默认 `0.12m`）。
2. 软近障惩罚改为只在 `gap < near_penalty_clearance` 时触发：
   - `gap = obstacle_clearance_to_contact`
   - `near_penalty = relu(near_penalty_clearance - gap)^2 / near_penalty_clearance^2`
3. 障碍“前后权重”由“车头朝向”改为“运动方向优先（低速回退到车头）”：
   - 近障风险按当前位移方向评估，避免“朝向与运动方向不一致”导致语义失真。

### 1.2 Supervisor 误触发收敛（`safety_supervisor.py`）
1. `jam_contact` 增加“近障/阻塞证据门”：
   - 触发除高力低位移低速度外，还需满足之一：
     - touch 命中
     - 前向/最小净空低于门限
     - `goal_blocked` 比率超过门限
2. Recovery 后退加后向安全门：
   - 新增 `rear_clearance`
   - 在 `BACKUP_RECOVER` 状态，若后向净空不足则改为原地转向，避免盲目倒车。
3. 新增参数面：
   - `jam_contact_front_clearance_gate`
   - `jam_contact_blocked_ratio_min`
   - `recover_min_rear_clearance`

### 1.3 控制链与可观测链路补齐（`run_icode_mppi_e1_test.py` / `run_icode_mppi_e1_viewer.py`）
1. 暴露并接入新参数：
   - `--near-penalty-clearance`
   - `--sup-jam-front-clearance-gate`
   - `--sup-jam-blocked-ratio-min`
   - `--sup-recover-min-rear-clearance`
2. `choose_turn_sign` 改为优先使用全量 lidar 扫描（而非仅 triplet）。
3. 新增 `rear_clearance` 提取并传入 Supervisor。
4. DOCK 阶段显式前向投影抑制倒车漂移，并计数 `dock_reverse_suppressed_steps`。

## 2. 验证结果

### 2.1 测试
1. `pytest -q tests`：`25 passed`。
2. 关键脚本语法检查通过：
   - `mppi.py`
   - `safety_supervisor.py`
   - `run_icode_mppi_e1_test.py`
   - `run_icode_mppi_e1_viewer.py`

### 2.2 固定 seeds 快速回归（CPU，6101~6105，max_steps=240）

聚合结果：
1. `reach_rate = 0.0`
2. `collision_rate = 0.0`
3. `final_dist_mean = 0.3785`
4. `mppi_steps_mean = 165.6`
5. `override_steps_mean = 0.0`
6. `jam_contact_events_mean = 0.0`
7. `progress_stall_events_mean = 0.0`
8. `spin_stall_events_mean = 0.0`
9. `mppi_reverse_raw_steps_mean = 23.6`
10. `nominal_reverse_suppressed_steps_mean = 16.2`
11. `dock_reverse_suppressed_steps_mean = 34.6`

解释：
1. `jam_contact` 误触发链路已被明显压制（均值 0）。
2. Supervisor 强制接管基本消失（`override_steps_mean=0`），说明“误判触发恢复”主问题已缓解。
3. 未达标主问题转为“近目标收敛（停在 goal_tol 边界外）”，而非碰撞或卡死恢复链。

## 3. 结论（对应本轮目标）

1. “近障才惩罚”已经按算法语义落地，不是阈值补丁。
2. “倒车来源是 Supervisor/DOCK”已通过链路改造被直接约束：
   - jam 误触发收敛
   - recover 后退有后向净空门
   - dock 阶段倒车被前向投影抑制
3. “激光改造后未全链路收敛”已补齐到：
   - 全量 scan 转向判定
   - rear clearance 进入 Supervisor 判定
   - 触发语义与动作执行闭环可观测

## 4. 当前剩余主瓶颈

1. 到达率仍为 0，主要表现为停在 `~0.37~0.39m` 邻域，接近但未进入 `goal_tol=0.35m`。
2. 下一轮应聚焦 terminal/dock 的“近目标推进语义”与停止判定边界，而非继续扩大恢复策略。
