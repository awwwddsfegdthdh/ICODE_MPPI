from typing import Dict, Optional, Tuple

import numpy as np
import torch
from state_convention import canonical_drive_sign


class DiffDriveKinematicModel(torch.nn.Module):
    def __init__(
        self,
        state_dim: int = 7,
        action_dim: int = 2,
        dt: float = 0.02,
        wheel_radius: float = 0.085,
        wheel_base: float = 0.37,
        drive_sign: float = -1.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.dt = float(dt)
        self.wheel_radius = float(wheel_radius)
        self.wheel_base = float(wheel_base)
        self.drive_sign = canonical_drive_sign(float(drive_sign))

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # x: [B,7], u: [B,2] where u=[dqL,dqR]
        dq_l = u[:, 0]
        dq_r = u[:, 1]
        psi = x[:, 2]
        v = self.drive_sign * self.wheel_radius * 0.5 * (dq_r + dq_l)
        wz = self.wheel_radius * (dq_r - dq_l) / max(self.wheel_base, 1e-6)

        xn = x.clone()
        xn[:, 0] = x[:, 0] + self.dt * v * torch.cos(psi)
        xn[:, 1] = x[:, 1] + self.dt * v * torch.sin(psi)
        psi_n = psi + self.dt * wz
        xn[:, 2] = torch.atan2(torch.sin(psi_n), torch.cos(psi_n))
        if self.state_dim >= 4:
            xn[:, 3] = v
        if self.state_dim >= 5:
            xn[:, 4] = wz
        if self.state_dim >= 6:
            xn[:, 5] = dq_l
        if self.state_dim >= 7:
            xn[:, 6] = dq_r
        return xn


class MPPIController:
    def __init__(
        self,
        dynamics_model,
        num_samples: int = 1000,
        horizon: int = 30,
        dim_action: int = 2,
        lambda_: float = 1.0,
        noise_sigma: Optional[np.ndarray] = None,
        noise_rho: float = 0.0,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
        default_obstacles: Optional[np.ndarray] = None,
        device: str = "cpu",
        u_init: Optional[np.ndarray] = None,
        cost_version: str = "v2",
        cost_task_goal: float = 2.8,
        cost_task_final: float = 140.0,
        cost_task_progress: float = 90.0,
        cost_task_heading: float = 8.0,
        cost_task_reverse: float = 2.5,
        cost_task_path_track: float = 14.0,
        cost_task_path_terminal: float = 40.0,
        cost_task_path_progress: float = 90.0,
        cost_task_lateral: float = 1.2,
        cost_task_goal_visibility: float = 6.0,
        cost_safe_collision: float = 1500.0,
        cost_safe_near: float = 1.2,
        cost_safe_corridor: float = 220.0,
        cost_safe_pred_obs: float = 80.0,
        cost_safe_bounds: float = 0.0,
        cost_safe_bounds_terminal: float = 0.0,
        collision_step_clearance: float = 0.10,
        collision_step_scale: float = 0.08,
        collision_step_hit_scale: float = 2.5,
        cost_ctrl_effort: float = 0.01,
        cost_ctrl_smooth: float = 0.08,
        cost_ctrl_spin: float = 0.08,
        cost_ctrl_wheel_diff: float = 0.02,
        cost_terminal_progress_deficit: float = 120.0,
        cost_terminal_near_goal_stall: float = 55.0,
        cost_terminal_stop: float = 120.0,
        cost_terminal_overshoot: float = 140.0,
        obs_behind_scale: float = 0.25,
        robot_radius: float = 0.28,
        obs_margin: float = 0.45,
        near_penalty_clearance: float = 0.12,
        near_penalty_mid_clearance: float = 0.45,
        near_penalty_hard_clearance: float = 0.20,
        near_penalty_mid_scale: float = 0.35,
        near_penalty_hard_scale: float = 1.20,
        near_penalty_hard_power: float = 3.0,
        pred_obs_clearance: float = 0.30,
        pred_obs_hard_clearance: float = 0.18,
        pred_obs_hard_scale: float = 2.0,
        pred_obs_clip_max: float = 8.0,
        near_progress_start: float = 0.75,
        near_progress_hard: float = 0.55,
        near_progress_weight: float = 70.0,
        near_stall_progress_eps: float = 0.015,
        near_stall_effort_gate: float = 0.45,
        path_corridor_half_width: float = 0.35,
        near_goal_radius: float = 0.90,
        near_goal_progress_eps: float = 0.003,
        los_margin: float = 0.10,
        terminal_stop_radius: float = 0.70,
        overshoot_tolerance: float = 0.05,
        noise_anneal_dist: float = 1.8,
        noise_anneal_min_scale: float = 0.30,
        enforce_forward_only: bool = False,
        forward_min_speed: float = 0.0,
        diff_wheel_radius: float = 0.04,
        diff_wheel_base: float = 0.25,
        diff_drive_sign: float = -1.0,
        world_x_min: float = -1.0e9,
        world_x_max: float = 1.0e9,
        world_y_min: float = -1.0e9,
        world_y_max: float = 1.0e9,
    ):
        self.model = dynamics_model
        self.model.eval()
        self.K = int(num_samples)
        self.T = int(horizon)
        self.u_dim = int(dim_action)
        self.device = torch.device(device)
        self.model.to(self.device)

        self.lambda_ = float(lambda_)
        self.noise_rho = float(np.clip(noise_rho, 0.0, 0.999))
        if noise_sigma is None:
            noise_sigma = np.array([2.0, 2.0], dtype=np.float32)
        self.noise_sigma = torch.tensor(noise_sigma, dtype=torch.float32, device=self.device)

        if action_low is None:
            action_low = np.array([-20.0, -20.0], dtype=np.float32)
        if action_high is None:
            action_high = np.array([20.0, 20.0], dtype=np.float32)
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=self.device)

        if u_init is None:
            u_init = np.array([0.0, 0.0], dtype=np.float32)
        u_init_t = torch.tensor(u_init, dtype=torch.float32, device=self.device).view(1, -1)
        self.U = u_init_t.repeat(self.T, 1).clone()

        if default_obstacles is None:
            default_obstacles = np.zeros((0, 3), dtype=np.float32)
        self.default_obstacles = torch.tensor(default_obstacles, dtype=torch.float32, device=self.device)

        # Cost-v2 grouped weights
        self.cost_version = str(cost_version).lower()
        self.cost_task_goal = float(max(0.0, cost_task_goal))
        self.cost_task_final = float(max(0.0, cost_task_final))
        self.cost_task_progress = float(max(0.0, cost_task_progress))
        self.cost_task_heading = float(max(0.0, cost_task_heading))
        self.cost_task_reverse = float(max(0.0, cost_task_reverse))
        self.cost_task_path_track = float(max(0.0, cost_task_path_track))
        self.cost_task_path_terminal = float(max(0.0, cost_task_path_terminal))
        self.cost_task_path_progress = float(max(0.0, cost_task_path_progress))
        self.cost_task_lateral = float(max(0.0, cost_task_lateral))
        self.cost_task_goal_visibility = float(max(0.0, cost_task_goal_visibility))

        self.cost_safe_collision = float(max(0.0, cost_safe_collision))
        self.cost_safe_near = float(max(0.0, cost_safe_near))
        self.cost_safe_corridor = float(max(0.0, cost_safe_corridor))
        self.cost_safe_pred_obs = float(max(0.0, cost_safe_pred_obs))
        self.cost_safe_bounds = float(max(0.0, cost_safe_bounds))
        self.cost_safe_bounds_terminal = float(max(0.0, cost_safe_bounds_terminal))
        self.collision_step_clearance = float(max(0.0, collision_step_clearance))
        self.collision_step_scale = float(max(0.0, collision_step_scale))
        self.collision_step_hit_scale = float(max(1.0, collision_step_hit_scale))

        self.cost_ctrl_effort = float(max(0.0, cost_ctrl_effort))
        self.cost_ctrl_smooth = float(max(0.0, cost_ctrl_smooth))
        self.cost_ctrl_spin = float(max(0.0, cost_ctrl_spin))
        self.cost_ctrl_wheel_diff = float(max(0.0, cost_ctrl_wheel_diff))

        self.cost_terminal_progress_deficit = float(max(0.0, cost_terminal_progress_deficit))
        self.cost_terminal_near_goal_stall = float(max(0.0, cost_terminal_near_goal_stall))
        self.cost_terminal_stop = float(max(0.0, cost_terminal_stop))
        self.cost_terminal_overshoot = float(max(0.0, cost_terminal_overshoot))

        # Backward-compatible aliases for existing调参与trace逻辑
        self.w_goal = self.cost_task_goal
        self.w_final = self.cost_task_final
        self.w_progress = self.cost_task_progress
        self.w_heading = self.cost_task_heading
        self.w_reverse = self.cost_task_reverse
        self.w_path_track = self.cost_task_path_track
        self.w_path_terminal = self.cost_task_path_terminal
        self.w_path_progress = self.cost_task_path_progress
        self.w_goal_visibility = self.cost_task_goal_visibility
        self.w_collision = self.cost_safe_collision
        self.w_near_obs = self.cost_safe_near
        self.w_corridor = self.cost_safe_corridor
        self.w_pred_obs = self.cost_safe_pred_obs
        self.w_bounds = self.cost_safe_bounds
        self.w_bounds_terminal = self.cost_safe_bounds_terminal
        self.w_control = self.cost_ctrl_effort
        self.w_smooth = self.cost_ctrl_smooth
        self.w_spin = self.cost_ctrl_spin
        self.w_u_diff = self.cost_ctrl_wheel_diff
        self.w_terminal_progress = self.cost_terminal_progress_deficit
        self.w_near_goal_stall = self.cost_terminal_near_goal_stall
        self.w_terminal_stop = self.cost_terminal_stop
        self.w_overshoot = self.cost_terminal_overshoot

        self.obs_behind_scale = float(np.clip(obs_behind_scale, 0.0, 1.0))
        self.w_lateral = self.cost_task_lateral
        self.robot_radius = float(robot_radius)
        self.obs_margin = float(obs_margin)
        # Legacy alias: historically near penalty used a single close-distance band.
        self.near_penalty_clearance = float(max(1e-3, near_penalty_clearance))
        # New semantics: two-layer near-obstacle shaping.
        # - mid layer starts earlier to avoid hugging obstacle boundaries.
        # - hard layer increases steeply at very close range.
        self.near_penalty_hard_clearance = float(
            max(1e-3, near_penalty_hard_clearance, self.near_penalty_clearance)
        )
        self.near_penalty_mid_clearance = float(
            max(self.near_penalty_hard_clearance + 1e-3, near_penalty_mid_clearance)
        )
        self.near_penalty_mid_scale = float(max(0.0, near_penalty_mid_scale))
        self.near_penalty_hard_scale = float(max(0.0, near_penalty_hard_scale))
        self.near_penalty_hard_power = float(max(1.0, near_penalty_hard_power))
        self.pred_obs_clearance = float(max(1e-3, pred_obs_clearance))
        self.pred_obs_hard_clearance = float(max(1e-3, min(self.pred_obs_clearance, pred_obs_hard_clearance)))
        self.pred_obs_hard_scale = float(max(0.0, pred_obs_hard_scale))
        self.pred_obs_clip_max = float(max(self.pred_obs_clearance, pred_obs_clip_max))
        self.near_progress_start = float(max(1e-3, near_progress_start))
        self.near_progress_hard = float(max(1e-3, min(self.near_progress_start - 1e-3, near_progress_hard)))
        self.near_progress_weight = float(max(0.0, near_progress_weight))
        self.near_stall_progress_eps = float(max(1e-6, near_stall_progress_eps))
        self.near_stall_effort_gate = float(max(0.0, near_stall_effort_gate))
        self.path_corridor_half_width = float(max(1e-3, path_corridor_half_width))
        self.near_goal_radius = float(near_goal_radius)
        self.near_goal_progress_eps = float(near_goal_progress_eps)
        self.los_margin = float(los_margin)
        self.terminal_stop_radius = float(max(0.0, terminal_stop_radius))
        self.overshoot_tolerance = float(max(0.0, overshoot_tolerance))
        self.noise_anneal_dist = float(max(0.0, noise_anneal_dist))
        self.noise_anneal_min_scale = float(np.clip(noise_anneal_min_scale, 0.05, 1.0))
        self.enforce_forward_only = bool(enforce_forward_only)
        self.forward_min_speed = float(max(0.0, forward_min_speed))
        self.diff_wheel_radius = float(max(1e-6, diff_wheel_radius))
        self.diff_wheel_base = float(max(1e-6, diff_wheel_base))
        self.diff_drive_sign = canonical_drive_sign(float(diff_drive_sign))
        self.world_x_min = float(world_x_min)
        self.world_x_max = float(world_x_max)
        self.world_y_min = float(world_y_min)
        self.world_y_max = float(world_y_max)
        self.bounds_enabled = (
            self.cost_safe_bounds > 0.0
            and self.world_x_min < self.world_x_max
            and self.world_y_min < self.world_y_max
        )
        self.last_cost_terms: Dict[str, float] = {
            "task": 0.0,
            "safety": 0.0,
            "pred_obs": 0.0,
            "control": 0.0,
            "terminal": 0.0,
            "total": 0.0,
        }

    def _project_forward_only_controls(self, u: torch.Tensor) -> torch.Tensor:
        if (not self.enforce_forward_only) or u.shape[-1] != 2:
            return u
        dq_l = u[..., 0]
        dq_r = u[..., 1]
        s = float(self.diff_drive_sign)
        r = float(self.diff_wheel_radius)
        wb = float(self.diff_wheel_base)
        v = s * r * 0.5 * (dq_r + dq_l)
        w = r * (dq_r - dq_l) / max(wb, 1e-6)
        v = torch.clamp(v, min=float(self.forward_min_speed))
        w_lim = 2.0 * v / max(wb, 1e-6)
        w = torch.clamp(w, -w_lim, w_lim)
        v_term = (v / s) / max(r, 1e-6)
        w_term = 0.5 * wb * w / max(r, 1e-6)
        dq_rn = v_term + w_term
        dq_ln = v_term - w_term
        return torch.stack([dq_ln, dq_rn], dim=-1)

    def compute_cost(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_xy: torch.Tensor,
        obstacles: torch.Tensor,
        init_dist: torch.Tensor,
        init_pos_xy: torch.Tensor,
        reference_traj: Optional[torch.Tensor] = None,
        pred_obs_dist: Optional[torch.Tensor] = None,
        return_terms: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # states: [K, T, D], actions: [K, T, U], target_xy: [2], obstacles: [N, 3]
        n_rollouts = states.shape[0]
        pos = states[:, :, :2]
        dist_goal = torch.linalg.norm(pos - target_xy.view(1, 1, 2), dim=-1)  # [K, T]
        final_dist = dist_goal[:, -1]
        cost_task = torch.zeros(n_rollouts, dtype=states.dtype, device=states.device)
        cost_safety = torch.zeros_like(cost_task)
        cost_pred_obs = torch.zeros_like(cost_task)
        cost_control = torch.zeros_like(cost_task)
        cost_terminal = torch.zeros_like(cost_task)

        if self.cost_task_goal > 0.0:
            cost_task = cost_task + self.cost_task_goal * torch.sum(dist_goal, dim=1)
        if self.cost_task_final > 0.0:
            cost_task = cost_task + self.cost_task_final * final_dist
        if self.cost_ctrl_effort > 0.0:
            cost_control = cost_control + self.cost_ctrl_effort * torch.sum(actions * actions, dim=(1, 2))
        if actions.shape[1] > 1:
            du = actions[:, 1:, :] - actions[:, :-1, :]
            cost_control = cost_control + self.cost_ctrl_smooth * torch.sum(du * du, dim=(1, 2))

        # Penalize excessive yaw-rate to prevent spinning in place.
        if states.shape[-1] > 4:
            wz = states[:, :, 4]
            cost_control = cost_control + self.cost_ctrl_spin * torch.sum(wz * wz, dim=1)

        # Task alignment and reverse penalty.
        if states.shape[-1] > 3:
            dx = target_xy[0] - pos[:, :, 0]
            dy = target_xy[1] - pos[:, :, 1]
            goal_heading = torch.atan2(dy, dx)
            psi = states[:, :, 2]
            heading_err = torch.atan2(torch.sin(goal_heading - psi), torch.cos(goal_heading - psi))
            cost_task = cost_task + self.cost_task_heading * torch.sum(torch.abs(heading_err), dim=1)

            v_body = states[:, :, 3]
            reverse_penalty = torch.relu(-v_body)
            cost_task = cost_task + self.cost_task_reverse * torch.sum(reverse_penalty, dim=1)

        # Progress shaping from initial position to terminal goal.
        goal_vec = target_xy - init_pos_xy
        goal_len = torch.linalg.norm(goal_vec) + 1e-6
        goal_dir = goal_vec / goal_len
        goal_perp = torch.stack([-goal_dir[1], goal_dir[0]])

        rel = pos - init_pos_xy.view(1, 1, 2)
        along = torch.sum(rel * goal_dir.view(1, 1, 2), dim=-1)  # [K, T]
        lateral = torch.sum(rel * goal_perp.view(1, 1, 2), dim=-1)  # [K, T]

        along_end = along[:, -1]
        terminal_progress_deficit = torch.relu(goal_len - along_end)
        cost_terminal = cost_terminal + self.cost_terminal_progress_deficit * terminal_progress_deficit
        cost_task = cost_task + self.cost_task_lateral * torch.mean(lateral * lateral, dim=1)
        if self.cost_terminal_overshoot > 0.0:
            overshoot = torch.relu(along_end - goal_len - self.overshoot_tolerance)
            cost_terminal = cost_terminal + self.cost_terminal_overshoot * (overshoot * overshoot)

        ref_dir_t = None
        # Path-tracking shaping.
        if reference_traj is not None and reference_traj.ndim == 2 and reference_traj.shape[0] >= pos.shape[1]:
            ref_xy = reference_traj[: pos.shape[1], :2].to(pos.dtype).to(self.device)
            path_err = pos - ref_xy.view(1, pos.shape[1], 2)
            path_err2 = torch.sum(path_err * path_err, dim=-1)
            if pos.shape[1] > 1:
                ref_tangent_t = torch.zeros_like(ref_xy)
                ref_tangent_t[:-1, :] = ref_xy[1:, :] - ref_xy[:-1, :]
                ref_tangent_t[-1, :] = ref_tangent_t[-2, :]
                tan_norm_t = torch.linalg.norm(ref_tangent_t, dim=-1, keepdim=True) + 1e-6
                ref_dir_t = ref_tangent_t / tan_norm_t
                ref_perp_t = torch.stack([-ref_dir_t[:, 1], ref_dir_t[:, 0]], dim=-1)
                lateral_signed = torch.sum(path_err * ref_perp_t.view(1, pos.shape[1], 2), dim=-1)
                corridor_violation = torch.relu(torch.abs(lateral_signed) - float(self.path_corridor_half_width))
                cost_safety = cost_safety + self.cost_safe_corridor * torch.sum(corridor_violation * corridor_violation, dim=1)
            if self.cost_task_path_track > 0.0:
                cost_task = cost_task + self.cost_task_path_track * torch.mean(path_err2, dim=1)
            if self.cost_task_path_terminal > 0.0:
                cost_task = cost_task + self.cost_task_path_terminal * path_err2[:, -1]
            if pos.shape[1] > 1 and self.cost_task_path_progress > 0.0:
                disp = pos[:, 1:, :] - pos[:, :-1, :]  # [K,T-1,2]
                ref_tangent = ref_xy[1:, :] - ref_xy[:-1, :]  # [T-1,2]
                tan_norm = torch.linalg.norm(ref_tangent, dim=-1, keepdim=True) + 1e-6
                ref_dir = ref_tangent / tan_norm
                path_motion = torch.sum(disp * ref_dir.view(1, ref_dir.shape[0], 2), dim=-1)  # [K,T-1]
                # Signed progress item: backward motion naturally increases cost.
                cost_task = cost_task - self.cost_task_path_progress * torch.sum(path_motion, dim=1)

        # Prefer symmetric wheel speeds unless steering is necessary.
        u_diff = actions[:, :, 1] - actions[:, :, 0]
        cost_control = cost_control + self.cost_ctrl_wheel_diff * torch.sum(u_diff * u_diff, dim=1)

        # Keep a weak direct progress reward to stabilize optimization.
        progress = torch.clamp(init_dist - final_dist, min=0.0)
        cost_task = cost_task - self.cost_task_progress * progress

        if dist_goal.shape[1] > 1 and self.cost_terminal_near_goal_stall > 0.0:
            # Additional shaping near goal: repeatedly failing to reduce goal distance
            # in the terminal region should be penalized.
            delta_dist = dist_goal[:, :-1] - dist_goal[:, 1:]
            no_progress = torch.relu(self.near_goal_progress_eps - delta_dist)
            near_mask = (
                (dist_goal[:, :-1] < self.near_goal_radius)
                | (dist_goal[:, 1:] < self.near_goal_radius)
            ).to(no_progress.dtype)
            cost_terminal = cost_terminal + self.cost_terminal_near_goal_stall * torch.sum(no_progress * near_mask, dim=1)

        if states.shape[-1] > 4 and self.cost_terminal_stop > 0.0 and self.terminal_stop_radius > 0.0:
            v_forward_end = states[:, -1, 3]
            wz_end = states[:, -1, 4]
            near_terminal = (final_dist < self.terminal_stop_radius).to(v_forward_end.dtype)
            stop_pen = (v_forward_end * v_forward_end) + 0.5 * (wz_end * wz_end)
            cost_terminal = cost_terminal + self.cost_terminal_stop * near_terminal * stop_pen

        if obstacles.shape[0] > 0:
            obs_xy = obstacles[:, :2]  # [N, 2]
            obs_r = obstacles[:, 2]  # [N]

            rel = pos[:, :, None, :] - obs_xy[None, None, :, :]
            d = torch.linalg.norm(rel, dim=-1)  # [K, T, N]
            safe = obs_r.view(1, 1, -1) + self.robot_radius
            gap = d - safe

            if self.collision_step_clearance > 1e-6 and self.collision_step_scale > 0.0:
                # Step-wise collision semantics:
                # - near-collision band starts before physical contact (gap < clearance)
                # - true contact/penetration receives a higher discrete level
                near_step = (gap < float(self.collision_step_clearance)).to(gap.dtype)
                hit_step = (gap < 0.0).to(gap.dtype)
                collision_penalty = float(self.collision_step_scale) * (
                    near_step + (float(self.collision_step_hit_scale) - 1.0) * hit_step
                )
            else:
                collision_penalty = torch.clamp(-gap, min=0.0)
            cost_safety = cost_safety + self.cost_safe_collision * torch.sum(collision_penalty, dim=(1, 2))

            # Two-layer near-obstacle penalty:
            # 1) mid range activates earlier to avoid grazing obstacle flanks.
            # 2) hard range rises steeply to strongly repel near-collision trajectories.
            near_mid_band = max(self.near_penalty_mid_clearance, self.near_penalty_hard_clearance + 1e-6)
            near_hard_band = max(self.near_penalty_hard_clearance, 1e-6)
            near_mid = torch.relu(near_mid_band - gap) / near_mid_band
            near_mid = near_mid * near_mid
            near_hard = torch.relu(near_hard_band - gap) / near_hard_band
            near_hard = torch.pow(near_hard, self.near_penalty_hard_power)
            near_penalty = (
                self.near_penalty_mid_scale * near_mid
                + self.near_penalty_hard_scale * near_hard
            )

            if self.near_progress_weight > 0.0 and dist_goal.shape[1] > 1:
                step_clear = torch.amin(gap, dim=2)  # [K,T]
                near_band = max(self.near_progress_start - self.near_progress_hard, 1e-6)
                near_factor = torch.clamp((self.near_progress_start - step_clear) / near_band, 0.0, 1.0)
                hard_factor = torch.clamp((self.near_progress_hard - step_clear) / max(self.near_progress_hard, 1e-6), 0.0, 1.0)
                near_factor = torch.clamp(near_factor + 0.5 * hard_factor * hard_factor, 0.0, 2.0)

                delta_dist = dist_goal[:, :-1] - dist_goal[:, 1:]  # [K,T-1], positive means progress
                low_prog = torch.relu(self.near_stall_progress_eps - delta_dist) / max(self.near_stall_progress_eps, 1e-6)
                effort = torch.linalg.norm(actions[:, 1:, :], dim=-1)  # [K,T-1]
                effort_gain = 1.0 + torch.relu(effort - self.near_stall_effort_gate)
                near_prog_pen = near_factor[:, :-1] * low_prog * effort_gain
                cost_safety = cost_safety + self.near_progress_weight * torch.sum(near_prog_pen, dim=1)

            # Obstacles behind robot should be much less important than ahead.
            if states.shape[-1] > 2 or pos.shape[1] > 1:
                if states.shape[-1] > 2:
                    psi = states[:, :, 2]
                    heading = torch.stack([torch.cos(psi), torch.sin(psi)], dim=-1)  # [K,T,2]
                else:
                    heading = torch.zeros((states.shape[0], states.shape[1], 2), dtype=states.dtype, device=states.device)
                    heading[..., 0] = 1.0
                if pos.shape[1] > 1:
                    disp = pos[:, 1:, :] - pos[:, :-1, :]  # [K,T-1,2]
                    disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
                    disp_dir = disp / (disp_norm + 1e-6)
                    move_dir = torch.cat([heading[:, :1, :], disp_dir], dim=1)
                    low_motion = torch.cat(
                        [
                            torch.zeros((disp.shape[0], 1, 1), dtype=torch.bool, device=disp.device),
                            disp_norm < 1e-4,
                        ],
                        dim=1,
                    )
                    motion_dir = torch.where(low_motion, heading, move_dir)
                else:
                    motion_dir = heading
                obs_vec = obs_xy.view(1, 1, -1, 2) - pos[:, :, None, :]  # [K,T,N,2]
                obs_dist = torch.linalg.norm(obs_vec, dim=-1) + 1e-6  # [K,T,N]
                cos_to_obs = torch.sum(motion_dir[:, :, None, :] * obs_vec, dim=-1) / obs_dist
                front_weight = torch.where(
                    cos_to_obs >= 0.0,
                    torch.ones_like(cos_to_obs),
                    torch.full_like(cos_to_obs, self.obs_behind_scale),
                )
                near_penalty = near_penalty * front_weight

            cost_safety = cost_safety + self.cost_safe_near * torch.sum(near_penalty, dim=(1, 2))

            if self.cost_task_goal_visibility > 0.0:
                # Reward trajectories that keep line-of-sight to goal,
                # with higher importance near terminal region.
                p = pos[:, :, None, :]  # [K,T,1,2]
                g = target_xy.view(1, 1, 1, 2)  # [1,1,1,2]
                c = obs_xy.view(1, 1, -1, 2)  # [1,1,N,2]

                seg = g - p
                seg_len2 = torch.sum(seg * seg, dim=-1, keepdim=True) + 1e-8
                pc = c - p
                tau = torch.sum(pc * seg, dim=-1, keepdim=True) / seg_len2
                tau = torch.clamp(tau, 0.0, 1.0)
                closest = p + tau * seg
                dist_seg = torch.linalg.norm(c - closest, dim=-1)  # [K,T,N]
                los_safe = obs_r.view(1, 1, -1) + (self.robot_radius + self.los_margin)
                blocked = dist_seg < los_safe
                blocked_any = torch.any(blocked, dim=-1).to(dist_goal.dtype)  # [K,T]

                near_scale = torch.clamp((self.near_goal_radius - dist_goal) / max(self.near_goal_radius, 1e-6), 0.0, 1.0)
                vis_reward = (1.0 - blocked_any) * (0.6 + 0.4 * near_scale)
                cost_task = cost_task - self.cost_task_goal_visibility * torch.sum(vis_reward, dim=1)

        if self.bounds_enabled:
            x = pos[:, :, 0]
            y = pos[:, :, 1]
            vx_low = torch.relu(self.world_x_min - x)
            vx_high = torch.relu(x - self.world_x_max)
            vy_low = torch.relu(self.world_y_min - y)
            vy_high = torch.relu(y - self.world_y_max)
            vbound = vx_low + vx_high + vy_low + vy_high
            cost_safety = cost_safety + self.cost_safe_bounds * torch.sum(vbound * vbound, dim=1)
            if self.cost_safe_bounds_terminal > 0.0:
                cost_safety = cost_safety + self.cost_safe_bounds_terminal * (vbound[:, -1] * vbound[:, -1])

        if pred_obs_dist is not None and self.cost_safe_pred_obs > 0.0:
            d_pred = torch.clamp(pred_obs_dist, min=0.0, max=self.pred_obs_clip_max)
            clear = max(self.pred_obs_clearance, 1e-6)
            hard_clear = max(min(self.pred_obs_hard_clearance, clear), 1e-6)
            mid_pen = torch.relu(clear - d_pred) / clear
            mid_pen = mid_pen * mid_pen
            hard_pen = torch.relu(hard_clear - d_pred) / hard_clear
            hard_pen = hard_pen * hard_pen
            pred_pen = mid_pen + self.pred_obs_hard_scale * hard_pen
            cost_pred_obs = self.cost_safe_pred_obs * torch.sum(pred_pen, dim=1)
            cost_safety = cost_safety + cost_pred_obs

        total_cost = cost_task + cost_safety + cost_control + cost_terminal
        if return_terms:
            terms = {
                "task": cost_task,
                "safety": cost_safety,
                "control": cost_control,
                "terminal": cost_terminal,
                "total": total_cost,
            }
            if pred_obs_dist is not None and self.cost_safe_pred_obs > 0.0:
                terms["pred_obs"] = cost_pred_obs
            return total_cost, terms
        return total_cost

    def rollout(self, init_state: torch.Tensor, u_samples: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        states = torch.zeros(self.K, self.T, self.model.state_dim, dtype=torch.float32, device=self.device)
        can_predict_obs = hasattr(self.model, "predict_obstacle_distance")
        pred_obs: Optional[torch.Tensor]
        if can_predict_obs:
            pred_obs = torch.zeros(self.K, self.T, dtype=torch.float32, device=self.device)
        else:
            pred_obs = None
        x = init_state
        with torch.no_grad():
            for t in range(self.T):
                if pred_obs is not None:
                    try:
                        d_pred = self.model.predict_obstacle_distance(x, u_samples[:, t, :]).to(torch.float32)
                        pred_obs[:, t] = torch.clamp(d_pred, min=0.0, max=self.pred_obs_clip_max)
                    except Exception:
                        pred_obs = None
                x = self.model(x, u_samples[:, t, :])
                states[:, t, :] = x
        return states, pred_obs

    def get_action(
        self,
        initial_state,
        target,
        obstacles: Optional[np.ndarray] = None,
        reference_traj: Optional[np.ndarray] = None,
        nominal_u_seq: Optional[np.ndarray] = None,
    ):
        state = torch.tensor(initial_state, dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.K, 1)
        target_xy = torch.tensor(target[:2], dtype=torch.float32, device=self.device)
        init_pos_xy = state[0, :2]
        init_dist = torch.linalg.norm(state[:, :2] - target_xy.view(1, 2), dim=1)

        if obstacles is None:
            obs_t = self.default_obstacles
        else:
            obs_t = torch.tensor(obstacles, dtype=torch.float32, device=self.device)

        ref_t = None
        if reference_traj is not None:
            ref_np = np.asarray(reference_traj, dtype=np.float32)
            if ref_np.ndim == 2 and ref_np.shape[0] > 0 and ref_np.shape[1] >= 2:
                if ref_np.shape[0] < self.T:
                    pad = np.repeat(ref_np[-1:, :2], repeats=self.T - ref_np.shape[0], axis=0)
                    ref_np = np.concatenate([ref_np[:, :2], pad], axis=0)
                ref_t = torch.tensor(ref_np[: self.T, :2], dtype=torch.float32, device=self.device)

        u_nom = self.U
        if nominal_u_seq is not None:
            nom_np = np.asarray(nominal_u_seq, dtype=np.float32)
            if nom_np.ndim == 2 and nom_np.shape[0] > 0 and nom_np.shape[1] >= self.u_dim:
                nom_np = nom_np[:, : self.u_dim]
                if nom_np.shape[0] < self.T:
                    pad = np.repeat(nom_np[-1:, :], repeats=self.T - nom_np.shape[0], axis=0)
                    nom_np = np.concatenate([nom_np, pad], axis=0)
                u_nom = torch.tensor(nom_np[: self.T, :], dtype=torch.float32, device=self.device)
                u_nom = torch.max(torch.min(u_nom, self.action_high.view(1, -1)), self.action_low.view(1, -1))
                u_nom = self._project_forward_only_controls(u_nom)
                u_nom = torch.max(torch.min(u_nom, self.action_high.view(1, -1)), self.action_low.view(1, -1))

        sigma_now = self.noise_sigma
        if self.noise_anneal_dist > 0.0:
            dist_scalar = float(init_dist[0].item())
            anneal = np.clip(dist_scalar / self.noise_anneal_dist, self.noise_anneal_min_scale, 1.0)
            sigma_now = self.noise_sigma * float(anneal)

        epsilon = torch.randn(self.K, self.T, self.u_dim, device=self.device) * sigma_now.view(1, 1, -1)
        if self.noise_rho > 0.0 and self.T > 1:
            rho = self.noise_rho
            sigma = float(np.sqrt(max(1e-8, 1.0 - rho * rho)))
            for t in range(1, self.T):
                epsilon[:, t, :] = rho * epsilon[:, t - 1, :] + sigma * epsilon[:, t, :]
        u_samples = u_nom.unsqueeze(0) + epsilon
        u_samples = torch.max(torch.min(u_samples, self.action_high.view(1, 1, -1)), self.action_low.view(1, 1, -1))
        u_samples = self._project_forward_only_controls(u_samples)
        u_samples = torch.max(torch.min(u_samples, self.action_high.view(1, 1, -1)), self.action_low.view(1, 1, -1))

        states, pred_obs = self.rollout(state, u_samples)
        costs, terms = self.compute_cost(
            states=states,
            actions=u_samples,
            target_xy=target_xy,
            obstacles=obs_t,
            init_dist=init_dist,
            init_pos_xy=init_pos_xy,
            reference_traj=ref_t,
            pred_obs_dist=pred_obs,
            return_terms=True,
        )
        self.last_cost_terms = {
            "task": float(torch.mean(terms["task"]).item()),
            "safety": float(torch.mean(terms["safety"]).item()),
            "pred_obs": float(torch.mean(terms["pred_obs"]).item()) if "pred_obs" in terms else 0.0,
            "control": float(torch.mean(terms["control"]).item()),
            "terminal": float(torch.mean(terms["terminal"]).item()),
            "total": float(torch.mean(terms["total"]).item()),
        }

        beta = torch.min(costs)
        weights = torch.exp(-(costs - beta) / max(self.lambda_, 1e-6))
        weights = weights / (torch.sum(weights) + 1e-9)

        weighted_noise = torch.sum(weights.view(self.K, 1, 1) * epsilon, dim=0)
        self.U = u_nom + weighted_noise
        self.U = torch.max(torch.min(self.U, self.action_high.view(1, -1)), self.action_low.view(1, -1))
        self.U = self._project_forward_only_controls(self.U)
        self.U = torch.max(torch.min(self.U, self.action_high.view(1, -1)), self.action_low.view(1, -1))

        action = self.U[0].detach().cpu().numpy()
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1] = 0.0
        return action
