# MPPI 阶段3第二轮：参数收敛与冗余清理（v1）

- 日期：2026-03-21
- 目标：将 `jam_break/spin_break` 等旧语义参数彻底并入 `SafetySupervisor` 单一参数面

## 1. 参数面收敛结果

1. 新增统一监督参数组（`sup-*`）
   - A层：
     - `--sup-touch-threshold`
     - `--sup-near-collision-clearance`
     - `--sup-side-collision-clearance`
     - `--sup-emergency-clearance`
     - `--sup-near-collision-persist-steps`
   - B层：
     - `--sup-progress-window`
     - `--sup-progress-min-delta`
     - `--sup-progress-stall-min-dist`
     - `--sup-global-stuck-clearance`
     - `--sup-jam-window`
     - `--sup-jam-min-u`
     - `--sup-jam-max-dxy`
     - `--sup-jam-max-v`
     - `--sup-spin-window`
     - `--sup-spin-min-progress`
     - `--sup-spin-max-displacement`
     - `--sup-spin-min-yaw-travel`
   - 恢复序列：
     - `--sup-recover-stop-steps`
     - `--sup-recover-backup-steps`
     - `--sup-recover-backup-speed`
     - `--sup-recover-rotate-steps`
     - `--sup-recover-forward-steps`
     - `--sup-recover-turn-rate`
     - `--sup-recover-forward-speed`
     - `--sup-recover-forward-turn-rate`
     - `--sup-recover-refractory-steps`
   - 边界：
     - `--sup-bounds-hard-margin`
     - `--sup-boundary-trigger-steps`
     - `--sup-boundary-release-steps`
     - `--sup-boundary-release-margin`

2. `build_supervisor_config(args)` 已完全切换到 `sup-*` 读取

## 2. 退役参数与逻辑

1. 退役旧参数入口（viewer/test 同步）
   - `recover-*` 旧组
   - `jam-contact-*` 旧组
   - `spin-break-*` 旧组
   - `jam-break-*` 旧组
   - `progress-*` / `global-stuck-*` 旧组
   - `collision/near/side/emergency` 的旧非 `sup-*` 入口

2. 退役旧动作覆盖函数
   - `recover_action`
   - `boundary_guard_action`
   - `anti_spin_forward_action`
   - `choose_progress_turn_sign`

3. 退役旧语义残留参数
   - `passage-*`（旧近碰撞分支配套参数）

## 3. 输出与指标

1. test metrics 中旧 `spin_break/jam_break` 命名已改为监督语义：
   - `spin_stall_events`
   - `progress_stall_events`
   - `recover_rotate_steps`
   - `supervisor_enabled`

2. viewer 运行摘要已改为监督语义事件命名

## 4. 验证

1. `py_compile`：通过
2. `run_icode_mppi_e1_test.py` kinematic smoke：通过
3. `test_mppi.py` smoke：通过
4. 动态单测：环境缺少 `pytest`，暂不可执行
