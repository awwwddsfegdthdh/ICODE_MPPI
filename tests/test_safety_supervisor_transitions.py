import numpy as np

from safety_supervisor import SafetySupervisor, SupervisorConfig, SupervisorInput, SupervisorState


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
        boundary_guard_release_steps=2,
        boundary_guard_release_margin=0.12,
    )


def _inp(step: int, min_clear: float = 0.5, touch: float = 0.0) -> SupervisorInput:
    return SupervisorInput(
        step=int(step),
        state=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        base_xy=np.array([0.0, 0.0], dtype=np.float32),
        goal_xy=np.array([2.6, 0.0], dtype=np.float32),
        dist_goal=2.0,
        touch_force=float(touch),
        min_clearance=float(min_clear),
        side_clearance=float(min_clear),
        front_clearance=float(min_clear),
        goal_blocked=False,
        prev_action=np.array([0.0, 0.0], dtype=np.float32),
        turn_sign_hint=1.0,
    )


def test_recovery_state_transition_sequence() -> None:
    sup = SafetySupervisor(_cfg())
    dec0 = sup.step(_inp(step=0, min_clear=0.07, touch=2e-5))
    assert dec0.state == SupervisorState.SAFE_STOP.value

    states = [dec0.state]
    for k in range(1, 12):
        dec = sup.step(_inp(step=k, min_clear=0.6, touch=0.0))
        states.append(dec.state)

    assert SupervisorState.BACKUP_RECOVER.value in states
    assert SupervisorState.ROTATE_RECOVER.value in states
    assert SupervisorState.FORWARD_RECOVER.value in states
    assert states[-1] == SupervisorState.NORMAL.value

    summary = sup.summary()
    trans = summary["transition_counts"]
    assert trans.get("SAFE_STOP->BACKUP_RECOVER", 0) >= 1
    assert trans.get("BACKUP_RECOVER->ROTATE_RECOVER", 0) >= 1
    assert trans.get("ROTATE_RECOVER->FORWARD_RECOVER", 0) >= 1
    assert trans.get("FORWARD_RECOVER->NORMAL", 0) >= 1
