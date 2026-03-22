import numpy as np

from safety_supervisor import SafetySupervisor, SupervisorConfig, SupervisorInput, SupervisorState


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
        boundary_guard_release_steps=2,
        boundary_guard_release_margin=0.12,
    )


def _inp(
    step: int,
    x: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    v: float = 0.0,
    dist_goal: float = 2.0,
    min_clear: float = 0.4,
    side_clear: float = 0.4,
    front_clear: float = 0.4,
    touch: float = 0.0,
    goal_blocked: bool = False,
    prev_u: tuple[float, float] = (0.0, 0.0),
    turn_sign_hint: float = 1.0,
) -> SupervisorInput:
    state = np.array([x, y, yaw, v, 0.0, prev_u[0], prev_u[1]], dtype=np.float32)
    return SupervisorInput(
        step=step,
        state=state,
        base_xy=np.array([x, y], dtype=np.float32),
        goal_xy=np.array([2.6, 0.0], dtype=np.float32),
        dist_goal=float(dist_goal),
        touch_force=float(touch),
        min_clearance=float(min_clear),
        side_clearance=float(side_clear),
        front_clearance=float(front_clear),
        goal_blocked=bool(goal_blocked),
        prev_action=np.array(prev_u, dtype=np.float32),
        turn_sign_hint=float(turn_sign_hint),
    )


def test_layer_a_preempts_layer_b() -> None:
    cfg = _cfg()
    # Make B-layer window short so progress-stall candidate can coexist with A-layer near risk.
    cfg.progress_window = 2
    cfg.spin_break_window = 2
    cfg.jam_contact_window = 2
    sup = SafetySupervisor(cfg)

    dec0 = sup.step(
        _inp(
            step=0,
            dist_goal=2.0,
            x=0.0,
            y=0.0,
            v=0.01,
            min_clear=0.15,
            side_clear=0.15,
            front_clear=0.10,
            goal_blocked=True,
            prev_u=(1.2, 1.2),
        )
    )
    dec1 = sup.step(
        _inp(
            step=1,
            dist_goal=2.0,
            x=0.0,
            y=0.0,
            v=0.01,
            min_clear=0.15,
            side_clear=0.15,
            front_clear=0.10,
            goal_blocked=True,
            prev_u=(1.2, 1.2),
        )
    )
    assert dec0.action_source == "MPPI"
    assert dec1.action_source == "SUPERVISOR"
    assert dec1.trigger_reason == "near_collision"
    assert dec1.state == SupervisorState.SAFE_STOP.value


def test_boundary_guard_has_highest_priority() -> None:
    sup = SafetySupervisor(_cfg())
    dec0 = sup.step(
        _inp(
            step=0,
            x=3.6,
            y=0.0,
            dist_goal=2.2,
            min_clear=0.6,
            side_clear=0.6,
            front_clear=0.6,
            touch=0.0,
            goal_blocked=False,
            prev_u=(0.2, 0.2),
        )
    )
    dec1 = sup.step(
        _inp(
            step=1,
            x=3.7,
            y=0.0,
            dist_goal=2.2,
            min_clear=0.07,
            side_clear=0.07,
            front_clear=0.07,
            touch=1e-4,
            goal_blocked=True,
            prev_u=(1.4, 1.4),
        )
    )
    assert dec0.action_source == "SUPERVISOR"
    assert dec0.trigger_reason == "boundary_guard"
    assert dec1.action_source == "SUPERVISOR"
    assert dec1.trigger_reason == "boundary_guard"
    assert dec1.state == SupervisorState.BOUNDARY_GUARD.value
