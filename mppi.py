from typing import Optional

import numpy as np
import torch


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
        self.drive_sign = float(drive_sign)

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
        w_goal: float = 1.6,
        w_final: float = 85.0,
        w_progress: float = 140.0,
        w_control: float = 0.01,
        w_smooth: float = 0.08,
        w_spin: float = 0.08,
        w_collision: float = 1500.0,
        w_near_obs: float = 3.0,
        w_heading: float = 8.0,
        w_reverse: float = 0.0,
        v_forward_sign: float = 1.0,
        w_away_goal: float = 45.0,
        obs_behind_scale: float = 0.25,
        robot_radius: float = 0.28,
        obs_margin: float = 0.45,
        w_goal_visibility: float = 6.0,
        w_near_goal_stall: float = 55.0,
        near_goal_radius: float = 0.90,
        near_goal_progress_eps: float = 0.003,
        los_margin: float = 0.10,
        w_terminal_stop: float = 120.0,
        terminal_stop_radius: float = 0.70,
        w_overshoot: float = 140.0,
        overshoot_tolerance: float = 0.05,
        noise_anneal_dist: float = 1.8,
        noise_anneal_min_scale: float = 0.30,
        w_path_track: float = 14.0,
        w_path_terminal: float = 40.0,
        w_path_progress: float = 90.0,
        w_path_backtrack: float = 180.0,
        w_goal_motion_away: float = 220.0,
        w_reverse_away: float = 160.0,
        w_bounds: float = 0.0,
        w_bounds_terminal: float = 0.0,
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

        # Cost weights
        self.w_goal = float(w_goal)
        self.w_final = float(w_final)
        self.w_progress = float(w_progress)
        self.w_control = float(w_control)
        self.w_smooth = float(w_smooth)
        self.w_spin = float(w_spin)
        self.w_collision = float(w_collision)
        self.w_near_obs = float(w_near_obs)
        self.w_heading = float(w_heading)
        self.w_reverse = float(w_reverse)
        self.v_forward_sign = float(v_forward_sign)
        self.w_away_goal = float(w_away_goal)
        self.obs_behind_scale = float(np.clip(obs_behind_scale, 0.0, 1.0))
        self.w_terminal_progress = 120.0
        self.w_backward_step = 90.0
        self.w_lateral = 1.2
        self.w_u_diff = 0.02
        self.robot_radius = float(robot_radius)
        self.obs_margin = float(obs_margin)
        self.w_goal_visibility = float(w_goal_visibility)
        self.w_near_goal_stall = float(w_near_goal_stall)
        self.near_goal_radius = float(near_goal_radius)
        self.near_goal_progress_eps = float(near_goal_progress_eps)
        self.los_margin = float(los_margin)
        self.w_terminal_stop = float(w_terminal_stop)
        self.terminal_stop_radius = float(max(0.0, terminal_stop_radius))
        self.w_overshoot = float(max(0.0, w_overshoot))
        self.overshoot_tolerance = float(max(0.0, overshoot_tolerance))
        self.noise_anneal_dist = float(max(0.0, noise_anneal_dist))
        self.noise_anneal_min_scale = float(np.clip(noise_anneal_min_scale, 0.05, 1.0))
        self.w_path_track = float(max(0.0, w_path_track))
        self.w_path_terminal = float(max(0.0, w_path_terminal))
        self.w_path_progress = float(max(0.0, w_path_progress))
        self.w_path_backtrack = float(max(0.0, w_path_backtrack))
        self.w_goal_motion_away = float(max(0.0, w_goal_motion_away))
        self.w_reverse_away = float(max(0.0, w_reverse_away))
        self.w_bounds = float(max(0.0, w_bounds))
        self.w_bounds_terminal = float(max(0.0, w_bounds_terminal))
        self.world_x_min = float(world_x_min)
        self.world_x_max = float(world_x_max)
        self.world_y_min = float(world_y_min)
        self.world_y_max = float(world_y_max)
        self.bounds_enabled = (
            self.w_bounds > 0.0
            and self.world_x_min < self.world_x_max
            and self.world_y_min < self.world_y_max
        )

    def compute_cost(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        target_xy: torch.Tensor,
        obstacles: torch.Tensor,
        init_dist: torch.Tensor,
        init_pos_xy: torch.Tensor,
        reference_traj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # states: [K, T, D], actions: [K, T, U], target_xy: [2], obstacles: [N, 3]
        pos = states[:, :, :2]
        dist_goal = torch.linalg.norm(pos - target_xy.view(1, 1, 2), dim=-1)  # [K, T]
        cost = self.w_goal * torch.sum(dist_goal, dim=1)

        final_dist = dist_goal[:, -1]
        cost = cost + self.w_final * final_dist
        cost = cost + self.w_control * torch.sum(actions * actions, dim=(1, 2))
        if actions.shape[1] > 1:
            du = actions[:, 1:, :] - actions[:, :-1, :]
            cost = cost + self.w_smooth * torch.sum(du * du, dim=(1, 2))

        # Penalize excessive yaw-rate to prevent spinning in place.
        if states.shape[-1] > 4:
            wz = states[:, :, 4]
            cost = cost + self.w_spin * torch.sum(wz * wz, dim=1)

        # Encourage heading alignment and forward progress along goal direction.
        if states.shape[-1] > 3:
            dx = target_xy[0] - pos[:, :, 0]
            dy = target_xy[1] - pos[:, :, 1]
            goal_heading = torch.atan2(dy, dx)
            psi = states[:, :, 2]
            heading_err = torch.atan2(torch.sin(goal_heading - psi), torch.cos(goal_heading - psi))
            cost = cost + self.w_heading * torch.sum(torch.abs(heading_err), dim=1)

            v_body = states[:, :, 3]
            # Penalize body-frame reverse motion directly, so controller prefers
            # turning toward goal then driving forward instead of backing in.
            v_forward = self.v_forward_sign * v_body
            reverse_penalty = torch.relu(-v_forward)
            cost = cost + self.w_reverse * torch.sum(reverse_penalty, dim=1)

        # Progress-to-goal-line shaping from initial position to terminal goal.
        goal_vec = target_xy - init_pos_xy
        goal_len = torch.linalg.norm(goal_vec) + 1e-6
        goal_dir = goal_vec / goal_len
        goal_perp = torch.stack([-goal_dir[1], goal_dir[0]])

        rel = pos - init_pos_xy.view(1, 1, 2)
        along = torch.sum(rel * goal_dir.view(1, 1, 2), dim=-1)  # [K, T]
        lateral = torch.sum(rel * goal_perp.view(1, 1, 2), dim=-1)  # [K, T]

        along_end = along[:, -1]
        terminal_progress_deficit = torch.relu(goal_len - along_end)
        cost = cost + self.w_terminal_progress * terminal_progress_deficit
        cost = cost + self.w_lateral * torch.mean(lateral * lateral, dim=1)
        if self.w_overshoot > 0.0:
            overshoot = torch.relu(along_end - goal_len - self.overshoot_tolerance)
            cost = cost + self.w_overshoot * (overshoot * overshoot)

        # Path-tracking shaping (similar to reference-waypoint tracking):
        # if a horizon reference trajectory is provided, prefer staying close to it
        # instead of only minimizing the final goal distance.
        if reference_traj is not None and reference_traj.ndim == 2 and reference_traj.shape[0] >= pos.shape[1]:
            ref_xy = reference_traj[: pos.shape[1], :2].to(pos.dtype).to(self.device)
            path_err = pos - ref_xy.view(1, pos.shape[1], 2)
            path_err2 = torch.sum(path_err * path_err, dim=-1)
            if self.w_path_track > 0.0:
                cost = cost + self.w_path_track * torch.mean(path_err2, dim=1)
            if self.w_path_terminal > 0.0:
                cost = cost + self.w_path_terminal * path_err2[:, -1]
            if pos.shape[1] > 1 and (self.w_path_progress > 0.0 or self.w_path_backtrack > 0.0):
                disp = pos[:, 1:, :] - pos[:, :-1, :]  # [K,T-1,2]
                ref_tangent = ref_xy[1:, :] - ref_xy[:-1, :]  # [T-1,2]
                tan_norm = torch.linalg.norm(ref_tangent, dim=-1, keepdim=True) + 1e-6
                ref_dir = ref_tangent / tan_norm
                path_motion = torch.sum(disp * ref_dir.view(1, ref_dir.shape[0], 2), dim=-1)  # [K,T-1]
                if self.w_path_progress > 0.0:
                    cost = cost - self.w_path_progress * torch.sum(torch.relu(path_motion), dim=1)
                if self.w_path_backtrack > 0.0:
                    cost = cost + self.w_path_backtrack * torch.sum(torch.relu(-path_motion), dim=1)

        if pos.shape[1] > 1 and self.w_goal_motion_away > 0.0:
            disp = pos[:, 1:, :] - pos[:, :-1, :]  # [K,T-1,2]
            goal_vec_step = target_xy.view(1, 1, 2) - pos[:, :-1, :]
            goal_dir_step = goal_vec_step / (torch.linalg.norm(goal_vec_step, dim=-1, keepdim=True) + 1e-6)
            goal_motion = torch.sum(disp * goal_dir_step, dim=-1)  # [K,T-1]
            cost = cost + self.w_goal_motion_away * torch.sum(torch.relu(-goal_motion), dim=1)

        if along.shape[1] > 1:
            delta_along = along[:, 1:] - along[:, :-1]
            cost = cost + self.w_backward_step * torch.sum(torch.relu(-delta_along), dim=1)

        # Prefer symmetric wheel speeds unless steering is necessary.
        u_diff = actions[:, :, 1] - actions[:, :, 0]
        cost = cost + self.w_u_diff * torch.sum(u_diff * u_diff, dim=1)

        # Keep a weak direct progress reward to stabilize optimization.
        progress = torch.clamp(init_dist - final_dist, min=0.0)
        cost = cost - self.w_progress * progress
        if dist_goal.shape[1] > 1 and self.w_away_goal > 0.0:
            # Explicitly punish moving away from target step-by-step.
            away = torch.relu(dist_goal[:, 1:] - dist_goal[:, :-1])
            cost = cost + self.w_away_goal * torch.sum(away, dim=1)
            if states.shape[-1] > 3 and self.w_reverse_away > 0.0:
                v_forward_step = self.v_forward_sign * states[:, :-1, 3]
                reverse_mask = torch.relu(-v_forward_step)
                cost = cost + self.w_reverse_away * torch.sum(away * reverse_mask, dim=1)

        if dist_goal.shape[1] > 1 and self.w_near_goal_stall > 0.0:
            # Additional shaping near goal: repeatedly failing to reduce goal distance
            # in the terminal region should be penalized.
            delta_dist = dist_goal[:, :-1] - dist_goal[:, 1:]
            no_progress = torch.relu(self.near_goal_progress_eps - delta_dist)
            near_mask = (
                (dist_goal[:, :-1] < self.near_goal_radius)
                | (dist_goal[:, 1:] < self.near_goal_radius)
            ).to(no_progress.dtype)
            cost = cost + self.w_near_goal_stall * torch.sum(no_progress * near_mask, dim=1)

        if states.shape[-1] > 4 and self.w_terminal_stop > 0.0 and self.terminal_stop_radius > 0.0:
            v_forward_end = self.v_forward_sign * states[:, -1, 3]
            wz_end = states[:, -1, 4]
            near_terminal = (final_dist < self.terminal_stop_radius).to(v_forward_end.dtype)
            stop_pen = (v_forward_end * v_forward_end) + 0.5 * (wz_end * wz_end)
            cost = cost + self.w_terminal_stop * near_terminal * stop_pen

        if obstacles.shape[0] > 0:
            obs_xy = obstacles[:, :2]  # [N, 2]
            obs_r = obstacles[:, 2]  # [N]

            rel = pos[:, :, None, :] - obs_xy[None, None, :, :]
            d = torch.linalg.norm(rel, dim=-1)  # [K, T, N]
            safe = obs_r.view(1, 1, -1) + self.robot_radius
            gap = d - safe

            collision_penalty = torch.clamp(-gap, min=0.0)
            cost = cost + self.w_collision * torch.sum(collision_penalty, dim=(1, 2))

            # Soft near-obstacle penalty: avoid hard reciprocal spikes that can dominate
            # goal tracking and cause backing/circling.
            near_penalty = torch.relu(self.obs_margin - gap)
            near_penalty = (near_penalty / max(self.obs_margin, 1e-6)) ** 2

            # Obstacles behind robot should be much less important than ahead.
            if states.shape[-1] > 2:
                psi = states[:, :, 2]
                heading = torch.stack([torch.cos(psi), torch.sin(psi)], dim=-1)  # [K,T,2]
                obs_vec = obs_xy.view(1, 1, -1, 2) - pos[:, :, None, :]  # [K,T,N,2]
                obs_dist = torch.linalg.norm(obs_vec, dim=-1) + 1e-6  # [K,T,N]
                cos_to_obs = torch.sum(heading[:, :, None, :] * obs_vec, dim=-1) / obs_dist
                front_weight = torch.where(
                    cos_to_obs >= 0.0,
                    torch.ones_like(cos_to_obs),
                    torch.full_like(cos_to_obs, self.obs_behind_scale),
                )
                near_penalty = near_penalty * front_weight

            cost = cost + self.w_near_obs * torch.sum(near_penalty, dim=(1, 2))

            if self.w_goal_visibility > 0.0:
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
                cost = cost - self.w_goal_visibility * torch.sum(vis_reward, dim=1)

        if self.bounds_enabled:
            x = pos[:, :, 0]
            y = pos[:, :, 1]
            vx_low = torch.relu(self.world_x_min - x)
            vx_high = torch.relu(x - self.world_x_max)
            vy_low = torch.relu(self.world_y_min - y)
            vy_high = torch.relu(y - self.world_y_max)
            vbound = vx_low + vx_high + vy_low + vy_high
            cost = cost + self.w_bounds * torch.sum(vbound * vbound, dim=1)
            if self.w_bounds_terminal > 0.0:
                cost = cost + self.w_bounds_terminal * (vbound[:, -1] * vbound[:, -1])

        return cost

    def rollout(self, init_state: torch.Tensor, u_samples: torch.Tensor) -> torch.Tensor:
        states = torch.zeros(self.K, self.T, self.model.state_dim, dtype=torch.float32, device=self.device)
        x = init_state
        with torch.no_grad():
            for t in range(self.T):
                x = self.model(x, u_samples[:, t, :])
                states[:, t, :] = x
        return states

    def get_action(
        self,
        initial_state,
        target,
        obstacles: Optional[np.ndarray] = None,
        reference_traj: Optional[np.ndarray] = None,
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
        u_samples = self.U.unsqueeze(0) + epsilon
        u_samples = torch.max(torch.min(u_samples, self.action_high.view(1, 1, -1)), self.action_low.view(1, 1, -1))

        states = self.rollout(state, u_samples)
        costs = self.compute_cost(
            states=states,
            actions=u_samples,
            target_xy=target_xy,
            obstacles=obs_t,
            init_dist=init_dist,
            init_pos_xy=init_pos_xy,
            reference_traj=ref_t,
        )

        beta = torch.min(costs)
        weights = torch.exp(-(costs - beta) / max(self.lambda_, 1e-6))
        weights = weights / (torch.sum(weights) + 1e-9)

        weighted_noise = torch.sum(weights.view(self.K, 1, 1) * epsilon, dim=0)
        self.U = self.U + weighted_noise
        self.U = torch.max(torch.min(self.U, self.action_high.view(1, -1)), self.action_low.view(1, -1))

        action = self.U[0].detach().cpu().numpy()
        self.U = torch.roll(self.U, shifts=-1, dims=0)
        self.U[-1] = 0.0
        return action
