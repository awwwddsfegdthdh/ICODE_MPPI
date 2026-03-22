from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np

from state_convention import world_to_body


class SupervisorState(str, Enum):
    NORMAL = "NORMAL"
    SAFE_STOP = "SAFE_STOP"
    BACKUP_RECOVER = "BACKUP_RECOVER"
    ROTATE_RECOVER = "ROTATE_RECOVER"
    FORWARD_RECOVER = "FORWARD_RECOVER"
    BOUNDARY_GUARD = "BOUNDARY_GUARD"


@dataclass
class SupervisorConfig:
    touch_threshold: float
    near_collision_clearance: float
    side_collision_clearance: float
    emergency_clearance: float
    near_collision_persist_steps: int
    near_front_enter: float
    near_front_exit: float
    near_side_enter: float
    near_side_exit: float
    near_clear_steps: int
    near_reentry_guard_steps: int
    near_rearm_progress: float
    near_turn_wz_gate: float

    progress_window: int
    progress_min_delta: float
    progress_path_min_delta: float
    progress_stall_min_dist: float
    global_stuck_clearance: float
    progress_confirm_windows: int
    progress_min_u: float
    progress_max_v: float
    progress_max_displacement: float
    progress_blocked_ratio_min: float
    progress_front_clearance_gate: float

    jam_contact_window: int
    jam_contact_min_u: float
    jam_contact_max_dxy: float
    jam_contact_max_v: float

    spin_break_window: int
    spin_break_min_progress: float
    spin_break_max_displacement: float
    spin_break_min_yaw_travel: float
    spin_confirm_windows: int
    spin_blocked_ratio_min: float
    spin_target_switch_max: int
    spin_min_wz: float
    spin_max_v: float
    spin_front_clearance_gate: float

    recover_stop_steps: int
    recover_backup_steps: int
    recover_rotate_steps: int
    recover_forward_steps: int
    recover_backup_speed: float
    recover_turn_rate: float
    recover_forward_speed: float
    recover_forward_turn_rate: float
    recover_refractory_steps: int

    bounds_x_range: tuple[float, float]
    bounds_y_range: tuple[float, float]
    bounds_hard_margin: float
    boundary_guard_trigger_steps: int
    boundary_guard_release_steps: int
    boundary_guard_release_margin: float
    boundary_soft_enter_dist: float = 0.04
    boundary_hard_enter_dist: float = 0.18
    boundary_soft_confirm_steps: int = 4
    boundary_min_inward_speed: float = 0.02
    boundary_release_progress_min: float = 0.10
    jam_contact_front_clearance_gate: float = 0.45
    jam_contact_blocked_ratio_min: float = 0.50
    recover_min_rear_clearance: float = 0.18


@dataclass
class SupervisorInput:
    step: int
    state: np.ndarray
    base_xy: np.ndarray
    goal_xy: np.ndarray
    dist_goal: float
    touch_force: float
    min_clearance: float
    side_clearance: float
    front_clearance: float
    goal_blocked: bool
    prev_action: np.ndarray
    turn_sign_hint: float
    path_progress_recent: float = float("nan")
    target_switched_recent: bool = False
    goal_progress_recent: float = float("nan")
    rear_clearance: float = float("inf")


@dataclass
class SupervisorDecision:
    state: str
    action_source: str
    trigger_reason: str
    transitioned: bool
    override_vw: Optional[np.ndarray]
    boundary_mode: str = "NONE"
    boundary_signed_dist: float = 0.0
    boundary_inward_speed: float = 0.0


class SafetySupervisor:
    def __init__(self, cfg: SupervisorConfig):
        self.cfg = cfg
        self.state = SupervisorState.NORMAL
        self.phase_left = 0
        self.turn_sign = 1.0

        self.cooldown_left = 0
        self.near_collision_count = 0
        self.near_clear_count = 0
        self.near_risk_latched = False
        self.near_reentry_guard_left = 0
        self.near_rearm_pending = False
        self.near_rearm_ref_dist = float("inf")
        self.last_recovery_reason = ""
        self.boundary_guard_active = False
        self.boundary_out_count = 0
        self.boundary_in_count = 0
        self.boundary_soft_count = 0
        self.boundary_release_count = 0
        self.boundary_mode = "NONE"
        self.boundary_signed_dist = 0.0
        self.boundary_inward_speed = 0.0
        self.boundary_entry_dist_goal = float("inf")
        self.progress_candidate_count = 0
        self.spin_candidate_count = 0
        self.last_rear_clearance = float("inf")

        self.dist_hist: deque[float] = deque(maxlen=max(2, int(cfg.progress_window)))
        hist_maxlen = max(cfg.jam_contact_window, cfg.spin_break_window, cfg.progress_window)
        self.xy_hist: deque[np.ndarray] = deque(maxlen=max(2, int(hist_maxlen)))
        self.yaw_hist: deque[float] = deque(maxlen=max(2, int(cfg.spin_break_window)))
        self.u_norm_hist: deque[float] = deque(maxlen=max(2, int(hist_maxlen)))
        self.v_hist: deque[float] = deque(maxlen=max(2, int(hist_maxlen)))
        self.wz_hist: deque[float] = deque(maxlen=max(2, int(cfg.spin_break_window)))
        self.goal_blocked_hist: deque[float] = deque(maxlen=max(2, int(hist_maxlen)))
        self.front_clearance_hist: deque[float] = deque(maxlen=max(2, int(hist_maxlen)))
        self.target_switch_hist: deque[float] = deque(maxlen=max(2, int(cfg.spin_break_window)))

        self.state_steps: Dict[str, int] = {s.value: 0 for s in SupervisorState}
        self.transition_counts: Dict[str, int] = {}
        self.trigger_counts: Dict[str, int] = {
            "boundary_guard": 0,
            "boundary_soft": 0,
            "near_collision": 0,
            "jam_contact": 0,
            "spin_stall": 0,
            "progress_stall": 0,
        }
        self.cooldown_reject_count = 0

    def _record_transition(self, old: SupervisorState, new: SupervisorState) -> None:
        if old == new:
            return
        k = f"{old.value}->{new.value}"
        self.transition_counts[k] = self.transition_counts.get(k, 0) + 1
        self.state = new

    def _boundary_guard_action(self, inp: SupervisorInput) -> np.ndarray:
        x_min = float(min(self.cfg.bounds_x_range[0], self.cfg.bounds_x_range[1]))
        x_max = float(max(self.cfg.bounds_x_range[0], self.cfg.bounds_x_range[1]))
        y_min = float(min(self.cfg.bounds_y_range[0], self.cfg.bounds_y_range[1]))
        y_max = float(max(self.cfg.bounds_y_range[0], self.cfg.bounds_y_range[1]))
        center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)
        rel_body = world_to_body(center - inp.base_xy, float(inp.state[2]))
        yaw_err = float(np.arctan2(rel_body[1], max(rel_body[0], 1e-6)))
        if abs(yaw_err) > 0.35:
            return np.array([0.0, float(np.sign(yaw_err)) * float(self.cfg.recover_turn_rate)], dtype=np.float32)
        return np.array([float(self.cfg.recover_forward_speed), 0.0], dtype=np.float32)

    def _recover_action(self, mode: SupervisorState) -> np.ndarray:
        if mode == SupervisorState.SAFE_STOP:
            return np.array([0.0, 0.0], dtype=np.float32)
        if mode == SupervisorState.BACKUP_RECOVER:
            # Rear space is insufficient: keep turning instead of forcing blind backup.
            if float(self.last_rear_clearance) < float(self.cfg.recover_min_rear_clearance):
                return np.array([0.0, float(self.turn_sign) * float(self.cfg.recover_turn_rate)], dtype=np.float32)
            return np.array(
                [
                    -float(self.cfg.recover_backup_speed),
                    float(self.turn_sign) * float(self.cfg.recover_forward_turn_rate) * 0.6,
                ],
                dtype=np.float32,
            )
        if mode == SupervisorState.ROTATE_RECOVER:
            return np.array([0.0, float(self.turn_sign) * float(self.cfg.recover_turn_rate)], dtype=np.float32)
        if mode == SupervisorState.FORWARD_RECOVER:
            return np.array(
                [
                    float(self.cfg.recover_forward_speed),
                    float(self.turn_sign) * float(self.cfg.recover_forward_turn_rate),
                ],
                dtype=np.float32,
            )
        return np.array([0.0, 0.0], dtype=np.float32)

    def _start_recovery(self, reason: str, turn_sign_hint: float, dist_goal: float) -> None:
        self.trigger_counts[reason] = self.trigger_counts.get(reason, 0) + 1
        s = float(np.sign(turn_sign_hint))
        self.turn_sign = 1.0 if s == 0.0 else s
        self.last_recovery_reason = str(reason)
        if str(reason) == "near_collision":
            self.near_rearm_pending = True
            self.near_rearm_ref_dist = float(dist_goal)
            self.near_risk_latched = False
            self.near_collision_count = 0
            self.near_clear_count = 0
        self._record_transition(self.state, SupervisorState.SAFE_STOP)
        self.phase_left = max(1, int(self.cfg.recover_stop_steps))

    def _step_recovery(self) -> None:
        self.phase_left -= 1
        if self.phase_left > 0:
            return
        if self.state == SupervisorState.SAFE_STOP:
            self._record_transition(self.state, SupervisorState.BACKUP_RECOVER)
            self.phase_left = max(1, int(self.cfg.recover_backup_steps))
            return
        if self.state == SupervisorState.BACKUP_RECOVER:
            self._record_transition(self.state, SupervisorState.ROTATE_RECOVER)
            self.phase_left = max(1, int(self.cfg.recover_rotate_steps))
            return
        if self.state == SupervisorState.ROTATE_RECOVER:
            self._record_transition(self.state, SupervisorState.FORWARD_RECOVER)
            self.phase_left = max(1, int(self.cfg.recover_forward_steps))
            return
        if self.state == SupervisorState.FORWARD_RECOVER:
            self._record_transition(self.state, SupervisorState.NORMAL)
            self.phase_left = 0
            self.cooldown_left = max(0, int(self.cfg.recover_refractory_steps))
            if str(self.last_recovery_reason) == "near_collision":
                self.near_reentry_guard_left = max(0, int(self.cfg.near_reentry_guard_steps))
            self.last_recovery_reason = ""
            self.dist_hist.clear()
            self.xy_hist.clear()
            self.yaw_hist.clear()
            self.u_norm_hist.clear()
            self.v_hist.clear()
            self.wz_hist.clear()
            self.goal_blocked_hist.clear()
            self.front_clearance_hist.clear()
            self.target_switch_hist.clear()
            self.progress_candidate_count = 0
            self.spin_candidate_count = 0

    def _boundary_signed_distance(self, base_xy: np.ndarray) -> float:
        x = float(base_xy[0])
        y = float(base_xy[1])
        x_min = float(min(self.cfg.bounds_x_range[0], self.cfg.bounds_x_range[1]))
        x_max = float(max(self.cfg.bounds_x_range[0], self.cfg.bounds_x_range[1]))
        y_min = float(min(self.cfg.bounds_y_range[0], self.cfg.bounds_y_range[1]))
        y_max = float(max(self.cfg.bounds_y_range[0], self.cfg.bounds_y_range[1]))
        dx_out = max(x_min - x, 0.0, x - x_max)
        dy_out = max(y_min - y, 0.0, y - y_max)
        if dx_out > 0.0 or dy_out > 0.0:
            return float(np.hypot(dx_out, dy_out))
        d_in = min(x - x_min, x_max - x, y - y_min, y_max - y)
        return -float(d_in)

    def _boundary_inward_speed(self, inp: SupervisorInput) -> float:
        x_min = float(min(self.cfg.bounds_x_range[0], self.cfg.bounds_x_range[1]))
        x_max = float(max(self.cfg.bounds_x_range[0], self.cfg.bounds_x_range[1]))
        y_min = float(min(self.cfg.bounds_y_range[0], self.cfg.bounds_y_range[1]))
        y_max = float(max(self.cfg.bounds_y_range[0], self.cfg.bounds_y_range[1]))
        center = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)
        to_center = center - np.asarray(inp.base_xy, dtype=np.float32)
        n = float(np.linalg.norm(to_center))
        if n < 1e-9:
            return 0.0
        inward_dir = to_center / n
        yaw = float(inp.state[2]) if inp.state.shape[0] > 2 else 0.0
        v = float(inp.state[3]) if inp.state.shape[0] > 3 else 0.0
        vel_world = np.array([v * np.cos(yaw), v * np.sin(yaw)], dtype=np.float32)
        return float(np.dot(vel_world, inward_dir))

    def _update_boundary_mode(self, inp: SupervisorInput) -> str:
        signed = float(self._boundary_signed_distance(inp.base_xy))
        inward_speed = float(self._boundary_inward_speed(inp))
        self.boundary_signed_dist = signed
        self.boundary_inward_speed = inward_speed

        hard_enter = float(max(0.0, self.cfg.boundary_hard_enter_dist))
        soft_enter = float(max(0.0, self.cfg.boundary_soft_enter_dist))
        rel_margin = float(max(0.0, self.cfg.boundary_guard_release_margin))
        rel_steps = max(1, int(self.cfg.boundary_guard_release_steps))
        soft_confirm = max(1, int(self.cfg.boundary_soft_confirm_steps))
        min_inward = float(self.cfg.boundary_min_inward_speed)
        rel_prog = float(max(0.0, self.cfg.boundary_release_progress_min))

        if signed >= hard_enter:
            if self.boundary_mode != "HARD":
                self.boundary_entry_dist_goal = float(inp.dist_goal)
            self.boundary_mode = "HARD"
            self.boundary_soft_count = soft_confirm
            self.boundary_release_count = 0
            self.boundary_guard_active = True
            return self.boundary_mode

        soft_candidate = (signed >= -soft_enter) and (inward_speed < min_inward)
        if soft_candidate:
            self.boundary_soft_count += 1
        else:
            self.boundary_soft_count = 0

        if self.boundary_mode == "NONE" and self.boundary_soft_count >= soft_confirm:
            self.boundary_mode = "SOFT"
            self.boundary_entry_dist_goal = float(inp.dist_goal)
            self.boundary_release_count = 0

        if self.boundary_mode in ("SOFT", "HARD"):
            progress_now = float(self.boundary_entry_dist_goal - float(inp.dist_goal))
            release_candidate = (signed <= -rel_margin) and (progress_now >= rel_prog)
            if release_candidate:
                self.boundary_release_count += 1
                if self.boundary_release_count >= rel_steps:
                    self.boundary_mode = "NONE"
                    self.boundary_release_count = 0
                    self.boundary_soft_count = 0
                    self.boundary_entry_dist_goal = float("inf")
            else:
                self.boundary_release_count = 0

        self.boundary_guard_active = self.boundary_mode == "HARD"
        return self.boundary_mode

    def step(self, inp: SupervisorInput) -> SupervisorDecision:
        self.state_steps[self.state.value] = self.state_steps.get(self.state.value, 0) + 1

        v_now = float(inp.state[3]) if inp.state.shape[0] > 3 else 0.0
        self.last_rear_clearance = float(inp.rear_clearance)
        u_norm = float(np.linalg.norm(inp.prev_action))
        self.dist_hist.append(float(inp.dist_goal))
        self.xy_hist.append(np.asarray(inp.base_xy, dtype=np.float32).copy())
        self.yaw_hist.append(float(inp.state[2]))
        self.u_norm_hist.append(u_norm)
        self.v_hist.append(abs(v_now))
        wz_now = abs(float(inp.state[4])) if inp.state.shape[0] > 4 else 0.0
        self.wz_hist.append(wz_now)
        self.goal_blocked_hist.append(1.0 if bool(inp.goal_blocked) else 0.0)
        self.front_clearance_hist.append(float(inp.front_clearance))
        self.target_switch_hist.append(1.0 if bool(inp.target_switched_recent) else 0.0)

        if self.cooldown_left > 0:
            self.cooldown_left -= 1
        if self.near_reentry_guard_left > 0:
            self.near_reentry_guard_left -= 1
        if self.near_rearm_pending:
            goal_rearm_progress = float(self.near_rearm_ref_dist - float(inp.dist_goal))
            path_prog = float(inp.path_progress_recent)
            path_rearmed = bool(np.isfinite(path_prog) and (path_prog >= float(self.cfg.near_rearm_progress)))
            if goal_rearm_progress >= float(self.cfg.near_rearm_progress) or path_rearmed:
                self.near_rearm_pending = False

        transitioned = False
        prev_state = self.state

        prev_boundary_mode = str(self.boundary_mode)
        boundary_mode = str(self._update_boundary_mode(inp))
        if prev_boundary_mode != boundary_mode and boundary_mode == "SOFT":
            self.trigger_counts["boundary_soft"] += 1

        if boundary_mode == "HARD":
            if self.state != SupervisorState.BOUNDARY_GUARD:
                self._record_transition(self.state, SupervisorState.BOUNDARY_GUARD)
                self.trigger_counts["boundary_guard"] += 1
                transitioned = True
            return SupervisorDecision(
                state=self.state.value,
                action_source="SUPERVISOR",
                trigger_reason="boundary_guard",
                transitioned=transitioned,
                override_vw=self._boundary_guard_action(inp),
                boundary_mode=boundary_mode,
                boundary_signed_dist=float(self.boundary_signed_dist),
                boundary_inward_speed=float(self.boundary_inward_speed),
            )

        if self.state == SupervisorState.BOUNDARY_GUARD and boundary_mode != "HARD":
            self._record_transition(self.state, SupervisorState.NORMAL)
            transitioned = True

        touch_hit = float(inp.touch_force) > float(self.cfg.touch_threshold)
        turning_now = wz_now > float(self.cfg.near_turn_wz_gate)
        near_enter_raw = (
            touch_hit
            or float(inp.front_clearance) < float(self.cfg.near_front_enter)
            or (turning_now and float(inp.side_clearance) < float(self.cfg.near_side_enter))
        )
        near_exit_clear = (
            (not touch_hit)
            and float(inp.front_clearance) > float(self.cfg.near_front_exit)
            and float(inp.side_clearance) > float(self.cfg.near_side_exit)
        )
        if self.near_risk_latched:
            if near_exit_clear:
                self.near_clear_count += 1
                if self.near_clear_count >= max(1, int(self.cfg.near_clear_steps)):
                    self.near_risk_latched = False
                    self.near_clear_count = 0
                    self.near_collision_count = 0
            else:
                self.near_clear_count = 0
        else:
            if near_enter_raw:
                self.near_collision_count += 1
                if self.near_collision_count >= max(1, int(self.cfg.near_collision_persist_steps)):
                    self.near_risk_latched = True
                    self.near_collision_count = 0
                    self.near_clear_count = 0
            else:
                self.near_collision_count = 0
        near_collision = bool(self.near_risk_latched)

        emergency = (
            touch_hit
            or float(inp.min_clearance) < float(self.cfg.emergency_clearance)
        )

        if self.state in (
            SupervisorState.SAFE_STOP,
            SupervisorState.BACKUP_RECOVER,
            SupervisorState.ROTATE_RECOVER,
            SupervisorState.FORWARD_RECOVER,
        ):
            out = self._recover_action(self.state)
            state_before_step = self.state
            self._step_recovery()
            transitioned = transitioned or (self.state != state_before_step)
            return SupervisorDecision(
                state=self.state.value,
                action_source="SUPERVISOR",
                trigger_reason="recover_sequence",
                transitioned=transitioned,
                override_vw=out,
                boundary_mode=boundary_mode,
                boundary_signed_dist=float(self.boundary_signed_dist),
                boundary_inward_speed=float(self.boundary_inward_speed),
            )

        near_reentry_ready = (
            self.cooldown_left <= 0
            and self.near_reentry_guard_left <= 0
            and (not self.near_rearm_pending)
        )
        if emergency or (near_collision and near_reentry_ready):
            self._start_recovery(
                reason="near_collision",
                turn_sign_hint=float(inp.turn_sign_hint),
                dist_goal=float(inp.dist_goal),
            )
            return SupervisorDecision(
                state=self.state.value,
                action_source="SUPERVISOR",
                trigger_reason="near_collision",
                transitioned=True,
                override_vw=self._recover_action(self.state),
                boundary_mode=boundary_mode,
                boundary_signed_dist=float(self.boundary_signed_dist),
                boundary_inward_speed=float(self.boundary_inward_speed),
            )

        b_trigger_reason = ""
        if len(self.xy_hist) >= max(2, int(self.cfg.jam_contact_window)) and float(inp.dist_goal) > max(float(self.cfg.progress_stall_min_dist), 0.6):
            jam_disp = float(np.linalg.norm(self.xy_hist[-1] - self.xy_hist[0]))
            jam_u = float(np.mean(np.asarray(self.u_norm_hist, dtype=np.float32))) if len(self.u_norm_hist) > 0 else 0.0
            jam_v = float(np.mean(np.asarray(self.v_hist, dtype=np.float32))) if len(self.v_hist) > 0 else 0.0
            wj = max(2, int(self.cfg.jam_contact_window))
            blocked_vals = list(self.goal_blocked_hist)[-wj:]
            front_vals = list(self.front_clearance_hist)[-wj:]
            jam_blocked_ratio = float(np.mean(np.asarray(blocked_vals, dtype=np.float32))) if len(blocked_vals) > 0 else 0.0
            jam_front_min = float(np.min(np.asarray(front_vals, dtype=np.float32))) if len(front_vals) > 0 else float(inp.front_clearance)
            jam_clear_gate = min(float(inp.min_clearance), float(inp.front_clearance), jam_front_min)
            jam_near_or_blocked = (
                (float(inp.touch_force) > float(self.cfg.touch_threshold))
                or (jam_clear_gate < float(self.cfg.jam_contact_front_clearance_gate))
                or (jam_blocked_ratio > float(self.cfg.jam_contact_blocked_ratio_min))
            )
            if (
                jam_u > float(self.cfg.jam_contact_min_u)
                and jam_disp < float(self.cfg.jam_contact_max_dxy)
                and jam_v < float(self.cfg.jam_contact_max_v)
                and jam_near_or_blocked
            ):
                b_trigger_reason = "jam_contact"

        if (not b_trigger_reason) and len(self.dist_hist) >= max(2, int(self.cfg.progress_window)):
            w = max(2, int(self.cfg.progress_window))
            dist_vals = list(self.dist_hist)[-w:]
            xy_vals = list(self.xy_hist)[-w:]
            u_vals = list(self.u_norm_hist)[-w:]
            v_vals = list(self.v_hist)[-w:]
            blocked_vals = list(self.goal_blocked_hist)[-w:]
            front_vals = list(self.front_clearance_hist)[-w:]
            goal_progress = float(dist_vals[0] - dist_vals[-1])
            path_progress = float(inp.path_progress_recent)
            path_gate = True if not np.isfinite(path_progress) else (path_progress < float(self.cfg.progress_path_min_delta))
            disp = float(np.linalg.norm(np.asarray(xy_vals[-1], dtype=np.float32) - np.asarray(xy_vals[0], dtype=np.float32)))
            u_eff = float(np.mean(np.asarray(u_vals, dtype=np.float32))) if len(u_vals) > 0 else 0.0
            v_eff = float(np.mean(np.asarray(v_vals, dtype=np.float32))) if len(v_vals) > 0 else 0.0
            blocked_ratio = float(np.mean(np.asarray(blocked_vals, dtype=np.float32))) if len(blocked_vals) > 0 else 0.0
            front_clear_min = float(np.min(np.asarray(front_vals, dtype=np.float32))) if len(front_vals) > 0 else float(inp.front_clearance)
            blocked_gate = (
                blocked_ratio > float(self.cfg.progress_blocked_ratio_min)
                or front_clear_min < float(self.cfg.progress_front_clearance_gate)
            )
            progress_candidate = (
                goal_progress < float(self.cfg.progress_min_delta)
                and path_gate
                and disp < float(self.cfg.progress_max_displacement)
                and u_eff > float(self.cfg.progress_min_u)
                and v_eff < float(self.cfg.progress_max_v)
                and blocked_gate
                and float(inp.dist_goal) > float(self.cfg.progress_stall_min_dist)
            )
            if progress_candidate:
                self.progress_candidate_count += 1
            else:
                self.progress_candidate_count = 0
            if self.progress_candidate_count >= max(1, int(self.cfg.progress_confirm_windows)):
                b_trigger_reason = "progress_stall"
        else:
            self.progress_candidate_count = 0

        if (not b_trigger_reason) and len(self.yaw_hist) >= max(2, int(self.cfg.spin_break_window)):
            w = max(2, int(self.cfg.spin_break_window))
            dist_vals = list(self.dist_hist)[-w:]
            xy_vals = list(self.xy_hist)[-w:]
            yaw_vals = list(self.yaw_hist)[-w:]
            blocked_vals = list(self.goal_blocked_hist)[-w:]
            front_vals = list(self.front_clearance_hist)[-w:]
            switch_vals = list(self.target_switch_hist)[-w:]
            wz_vals = list(self.wz_hist)[-w:]
            v_vals = list(self.v_hist)[-w:]
            progress = float(dist_vals[0] - dist_vals[-1])
            disp = float(np.linalg.norm(np.asarray(xy_vals[-1], dtype=np.float32) - np.asarray(xy_vals[0], dtype=np.float32)))
            yaw_travel = 0.0
            for i in range(1, len(yaw_vals)):
                yaw_travel += abs(float((yaw_vals[i] - yaw_vals[i - 1] + np.pi) % (2.0 * np.pi) - np.pi))
            blocked_ratio = float(np.mean(np.asarray(blocked_vals, dtype=np.float32))) if len(blocked_vals) > 0 else 0.0
            front_clear_min = float(np.min(np.asarray(front_vals, dtype=np.float32))) if len(front_vals) > 0 else float(inp.front_clearance)
            target_switch_count = int(np.sum(np.asarray(switch_vals, dtype=np.float32)))
            w_eff = float(np.mean(np.asarray(wz_vals, dtype=np.float32))) if len(wz_vals) > 0 else 0.0
            v_eff = float(np.mean(np.asarray(v_vals, dtype=np.float32))) if len(v_vals) > 0 else 0.0
            spin_candidate = (
                progress < float(self.cfg.spin_break_min_progress)
                and disp < float(self.cfg.spin_break_max_displacement)
                and yaw_travel > float(self.cfg.spin_break_min_yaw_travel)
                and w_eff > float(self.cfg.spin_min_wz)
                and v_eff < float(self.cfg.spin_max_v)
                and (
                    blocked_ratio > float(self.cfg.spin_blocked_ratio_min)
                    or front_clear_min < float(self.cfg.spin_front_clearance_gate)
                )
                and target_switch_count <= max(0, int(self.cfg.spin_target_switch_max))
                and float(inp.dist_goal) > float(self.cfg.progress_stall_min_dist)
            )
            if spin_candidate:
                self.spin_candidate_count += 1
            else:
                self.spin_candidate_count = 0
            if self.spin_candidate_count >= max(1, int(self.cfg.spin_confirm_windows)):
                b_trigger_reason = "spin_stall"
        else:
            self.spin_candidate_count = 0

        if b_trigger_reason:
            if b_trigger_reason != "progress_stall":
                self.progress_candidate_count = 0
            if b_trigger_reason != "spin_stall":
                self.spin_candidate_count = 0
            if self.cooldown_left > 0:
                self.cooldown_reject_count += 1
            else:
                self._start_recovery(
                    reason=b_trigger_reason,
                    turn_sign_hint=float(inp.turn_sign_hint),
                    dist_goal=float(inp.dist_goal),
                )
                return SupervisorDecision(
                    state=self.state.value,
                    action_source="SUPERVISOR",
                    trigger_reason=b_trigger_reason,
                    transitioned=True,
                    override_vw=self._recover_action(self.state),
                    boundary_mode=boundary_mode,
                    boundary_signed_dist=float(self.boundary_signed_dist),
                    boundary_inward_speed=float(self.boundary_inward_speed),
                )

        normal_reason = "boundary_soft_target" if boundary_mode == "SOFT" else "normal"
        return SupervisorDecision(
            state=self.state.value,
            action_source="MPPI",
            trigger_reason=normal_reason,
            transitioned=transitioned or (self.state != prev_state),
            override_vw=None,
            boundary_mode=boundary_mode,
            boundary_signed_dist=float(self.boundary_signed_dist),
            boundary_inward_speed=float(self.boundary_inward_speed),
        )

    def summary(self) -> Dict[str, object]:
        return {
            "state_steps": {k: int(v) for k, v in self.state_steps.items()},
            "transition_counts": {k: int(v) for k, v in self.transition_counts.items()},
            "trigger_counts": {k: int(v) for k, v in self.trigger_counts.items()},
            "cooldown_reject_count": int(self.cooldown_reject_count),
            "final_state": str(self.state.value),
            "boundary_mode": str(self.boundary_mode),
            "boundary_signed_dist": float(self.boundary_signed_dist),
            "boundary_inward_speed": float(self.boundary_inward_speed),
        }
