import numpy as np

from safety_supervisor import SafetySupervisor, SupervisorConfig, SupervisorInput


def _cfg() -> SupervisorConfig:
    return SupervisorConfig(
        touch_threshold=1e-5,
        near_collision_clearance=0.18,
        side_collision_clearance=0.13,
        emergency_clearance=0.10,
        near_collision_persist_steps=1,
        near_front_enter=0.14,
        near_front_exit=0.22,
        near_side_enter=0.12,
        near_side_exit=0.18,
        near_clear_steps=2,
        near_reentry_guard_steps=2,
        near_rearm_progress=0.05,
        near_turn_wz_gate=0.2,
        progress_window=4,
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
        jam_contact_min_u=1.0,
        jam_contact_max_dxy=0.03,
        jam_contact_max_v=0.05,
        spin_break_window=5,
        spin_break_min_progress=0.02,
        spin_break_max_displacement=0.20,
        spin_break_min_yaw_travel=0.9,
        spin_confirm_windows=1,
        spin_blocked_ratio_min=0.60,
        spin_target_switch_max=2,
        spin_min_wz=0.2,
        spin_max_v=0.10,
        spin_front_clearance_gate=0.40,
        recover_stop_steps=1,
        recover_backup_steps=1,
        recover_rotate_steps=1,
        recover_forward_steps=1,
        recover_backup_speed=0.2,
        recover_turn_rate=1.8,
        recover_forward_speed=0.2,
        recover_forward_turn_rate=0.5,
        recover_refractory_steps=3,
        bounds_x_range=(-0.8, 3.2),
        bounds_y_range=(-2.0, 2.0),
        bounds_hard_margin=0.35,
        boundary_guard_trigger_steps=2,
        boundary_guard_release_steps=2,
        boundary_guard_release_margin=0.12,
    )


def _inp(step: int, x: float, yaw: float, dist: float, clear: float, touch: float, u: float) -> SupervisorInput:
    return SupervisorInput(
        step=int(step),
        state=np.array([x, 0.0, yaw, 0.0, 0.0, u, u], dtype=np.float32),
        base_xy=np.array([x, 0.0], dtype=np.float32),
        goal_xy=np.array([2.6, 0.0], dtype=np.float32),
        dist_goal=float(dist),
        touch_force=float(touch),
        min_clearance=float(clear),
        side_clearance=float(clear),
        front_clearance=float(clear),
        goal_blocked=bool(dist > 1.0),
        prev_action=np.array([u, u], dtype=np.float32),
        turn_sign_hint=1.0,
    )


def test_decision_trace_contract() -> None:
    sup = SafetySupervisor(_cfg())
    cases = [
        _inp(0, x=0.0, yaw=0.0, dist=2.4, clear=0.6, touch=0.0, u=0.0),
        _inp(1, x=0.0, yaw=0.0, dist=2.4, clear=0.08, touch=2e-5, u=0.2),
        _inp(2, x=0.0, yaw=0.2, dist=2.3, clear=0.6, touch=0.0, u=0.0),
        _inp(3, x=3.6, yaw=0.0, dist=2.3, clear=0.6, touch=0.0, u=0.0),
        _inp(4, x=3.7, yaw=0.0, dist=2.3, clear=0.6, touch=0.0, u=0.0),
    ]
    for c in cases:
        dec = sup.step(c)
        assert dec.action_source in ("MPPI", "SUPERVISOR")
        assert isinstance(dec.trigger_reason, str) and len(dec.trigger_reason) > 0
        assert isinstance(dec.transitioned, bool)
        if dec.action_source == "MPPI":
            assert dec.override_vw is None
        else:
            assert dec.override_vw is not None
