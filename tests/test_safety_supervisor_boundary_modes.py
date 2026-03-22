import numpy as np

from safety_supervisor import SafetySupervisor, SupervisorConfig, SupervisorInput


def _cfg() -> SupervisorConfig:
    return SupervisorConfig(
        touch_threshold=1e-5,
        near_collision_clearance=0.18,
        side_collision_clearance=0.13,
        emergency_clearance=0.10,
        near_collision_persist_steps=2,
        near_front_enter=0.14,
        near_front_exit=0.22,
        near_side_enter=0.12,
        near_side_exit=0.18,
        near_clear_steps=2,
        near_reentry_guard_steps=2,
        near_rearm_progress=0.05,
        near_turn_wz_gate=0.2,
        progress_window=5,
        progress_min_delta=0.03,
        progress_path_min_delta=0.03,
        progress_stall_min_dist=0.8,
        global_stuck_clearance=0.9,
        progress_confirm_windows=1,
        progress_min_u=1.0,
        progress_max_v=0.10,
        progress_max_displacement=0.12,
        progress_blocked_ratio_min=0.60,
        progress_front_clearance_gate=0.35,
        jam_contact_window=4,
        jam_contact_min_u=1.2,
        jam_contact_max_dxy=0.04,
        jam_contact_max_v=0.06,
        spin_break_window=6,
        spin_break_min_progress=0.02,
        spin_break_max_displacement=0.25,
        spin_break_min_yaw_travel=1.0,
        spin_confirm_windows=1,
        spin_blocked_ratio_min=0.60,
        spin_target_switch_max=2,
        spin_min_wz=0.2,
        spin_max_v=0.10,
        spin_front_clearance_gate=0.40,
        recover_stop_steps=2,
        recover_backup_steps=2,
        recover_rotate_steps=2,
        recover_forward_steps=2,
        recover_backup_speed=0.2,
        recover_turn_rate=1.8,
        recover_forward_speed=0.22,
        recover_forward_turn_rate=0.5,
        recover_refractory_steps=4,
        bounds_x_range=(-0.8, 3.2),
        bounds_y_range=(-2.0, 2.0),
        bounds_hard_margin=0.35,
        boundary_guard_trigger_steps=2,
        boundary_guard_release_steps=1,
        boundary_guard_release_margin=0.12,
        boundary_soft_enter_dist=0.04,
        boundary_hard_enter_dist=0.18,
        boundary_soft_confirm_steps=2,
        boundary_min_inward_speed=0.02,
        boundary_release_progress_min=0.0,
    )


def _inp(step: int, x: float, v: float = 0.0, dist: float = 2.0) -> SupervisorInput:
    state = np.array([x, 0.0, 0.0, v, 0.0, 0.0, 0.0], dtype=np.float32)
    return SupervisorInput(
        step=int(step),
        state=state,
        base_xy=np.array([x, 0.0], dtype=np.float32),
        goal_xy=np.array([2.6, 0.0], dtype=np.float32),
        dist_goal=float(dist),
        touch_force=0.0,
        min_clearance=0.6,
        side_clearance=0.6,
        front_clearance=0.6,
        goal_blocked=False,
        prev_action=np.array([0.1, 0.1], dtype=np.float32),
        turn_sign_hint=1.0,
    )


def test_boundary_soft_then_hard_modes() -> None:
    sup = SafetySupervisor(_cfg())

    d0 = sup.step(_inp(step=0, x=3.23, v=0.0, dist=2.2))
    d1 = sup.step(_inp(step=1, x=3.23, v=0.0, dist=2.2))
    assert d0.action_source == "MPPI"
    assert d1.action_source == "MPPI"
    assert d1.trigger_reason == "boundary_soft_target"
    assert d1.boundary_mode == "SOFT"

    d2 = sup.step(_inp(step=2, x=3.45, v=0.0, dist=2.2))
    assert d2.action_source == "SUPERVISOR"
    assert d2.trigger_reason == "boundary_guard"
    assert d2.boundary_mode == "HARD"
