import argparse
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
import torch

from env_mujoco import E1RobotEnv
from mppi import DiffDriveKinematicModel, MPPIController
from mppi_nav_utils import (
    build_reference_traj_from_path,
    compute_path_remaining,
    compute_auto_waypoint_with_side,
    line_signed_lateral,
    line_of_sight_blocked,
    plan_global_path_xy,
    point_segment_distance_and_t,
    sample_obstacles_adaptive,
    select_path_lookahead_target,
)
from run_icode_mppi_e1_test import DEFAULT_XML, HybridDynamics, load_icode_checkpoint


def _adaptive_profile(mode: str) -> dict:
    mode_u = str(mode).upper()
    table = {
        "OPEN": {
            "w_goal": 1.00,
            "w_collision": 1.00,
            "w_near_obs": 1.00,
            "w_path_track": 1.00,
            "w_path_progress": 1.00,
            "w_smooth": 1.00,
            "w_terminal_stop": 1.00,
            "w_path_backtrack": 1.00,
            "w_goal_motion_away": 1.00,
            "w_reverse_away": 1.00,
            "noise_sigma": 1.00,
        },
        "TIGHT": {
            "w_goal": 0.90,
            "w_collision": 1.35,
            "w_near_obs": 1.25,
            "w_path_track": 1.25,
            "w_path_progress": 1.05,
            "w_smooth": 1.10,
            "w_terminal_stop": 0.95,
            "w_path_backtrack": 1.15,
            "w_goal_motion_away": 1.10,
            "w_reverse_away": 1.05,
            "noise_sigma": 0.62,
        },
        "STUCK": {
            "w_goal": 1.35,
            "w_collision": 0.90,
            "w_near_obs": 0.90,
            "w_path_track": 1.15,
            "w_path_progress": 1.45,
            "w_smooth": 0.90,
            "w_terminal_stop": 0.90,
            "w_path_backtrack": 1.10,
            "w_goal_motion_away": 1.25,
            "w_reverse_away": 1.20,
            "noise_sigma": 1.35,
        },
        "DOCK": {
            "w_goal": 1.10,
            "w_collision": 0.70,
            "w_near_obs": 0.70,
            "w_path_track": 0.90,
            "w_path_progress": 1.10,
            "w_smooth": 1.25,
            "w_terminal_stop": 1.45,
            "w_path_backtrack": 1.00,
            "w_goal_motion_away": 1.10,
            "w_reverse_away": 1.05,
            "noise_sigma": 0.55,
        },
    }
    return table.get(mode_u, table["OPEN"])


def _apply_adaptive_profile(
    mppi: MPPIController,
    base: dict,
    profile: dict,
    alpha: float,
) -> None:
    a = float(np.clip(alpha, 0.0, 1.0))
    scalar_keys = (
        "w_goal",
        "w_collision",
        "w_near_obs",
        "w_path_track",
        "w_path_progress",
        "w_smooth",
        "w_terminal_stop",
        "w_path_backtrack",
        "w_goal_motion_away",
        "w_reverse_away",
    )
    for k in scalar_keys:
        if hasattr(mppi, k) and k in base:
            tgt = float(base[k]) * float(profile.get(k, 1.0))
            cur = float(getattr(mppi, k))
            setattr(mppi, k, (1.0 - a) * cur + a * tgt)

    if "noise_sigma" in base and hasattr(mppi, "noise_sigma"):
        noise_scale = float(max(0.25, profile.get("noise_sigma", 1.0)))
        tgt = base["noise_sigma"] * noise_scale
        cur = mppi.noise_sigma
        mppi.noise_sigma = torch.clamp((1.0 - a) * cur + a * tgt, min=0.03)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MPPI+ICODE in MuJoCo with live viewer.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/wmh/ICODE/domo/ICODE_MPPI/runs/icode_e1_mix_stageB/icode_best.pt"),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--planner-model", type=str, default="hybrid", choices=("icode", "kinematic", "hybrid"))
    parser.add_argument("--seed", type=int, default=20260313)

    parser.add_argument("--num-samples", type=int, default=900)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--mppi-lambda", type=float, default=1.0)
    parser.add_argument("--noise-sigma", type=float, nargs=2, default=(1.6, 1.6))
    parser.add_argument("--noise-rho", type=float, default=0.82)
    parser.add_argument("--u-init", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--w-goal", type=float, default=2.8)
    parser.add_argument("--w-final", type=float, default=140.0)
    parser.add_argument("--w-progress", type=float, default=90.0)
    parser.add_argument("--w-control", type=float, default=0.01)
    parser.add_argument("--w-smooth", type=float, default=0.08)
    parser.add_argument("--w-spin", type=float, default=0.08)
    parser.add_argument("--w-collision", type=float, default=1500.0)
    parser.add_argument("--w-near-obs", type=float, default=1.2)
    parser.add_argument("--w-heading", type=float, default=8.0)
    parser.add_argument("--w-reverse", type=float, default=2.5)
    parser.add_argument("--w-away-goal", type=float, default=45.0)
    parser.add_argument("--w-goal-visibility", type=float, default=6.0)
    parser.add_argument("--w-near-goal-stall", type=float, default=55.0)
    parser.add_argument("--near-goal-radius", type=float, default=0.90)
    parser.add_argument("--near-goal-progress-eps", type=float, default=0.003)
    parser.add_argument("--los-margin", type=float, default=0.10)
    parser.add_argument("--obs-behind-scale", type=float, default=0.25)
    parser.add_argument("--v-forward-sign", type=float, default=1.0)
    parser.add_argument("--icode-blend-alpha", type=float, default=0.35)
    parser.add_argument("--robot-radius", type=float, default=0.28)
    parser.add_argument("--obs-margin", type=float, default=0.25)
    parser.add_argument("--collision-touch-threshold", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=35)
    parser.add_argument("--warmup-min-scale", type=float, default=0.12)
    parser.add_argument("--action-ema-alpha", type=float, default=0.75)
    parser.add_argument("--max-delta-u", type=float, default=0.6)
    parser.add_argument("--startup-no-reverse-steps", type=int, default=0)
    parser.add_argument("--startup-min-u", type=float, default=0.0)
    parser.add_argument("--emergency-clearance", type=float, default=0.12)
    parser.add_argument("--recover-stop-steps", type=int, default=10)
    parser.add_argument("--recover-rotate-steps", type=int, default=22)
    parser.add_argument("--recover-forward-steps", type=int, default=20)
    parser.add_argument("--recover-turn-rate", type=float, default=2.2)
    parser.add_argument("--recover-forward-speed", type=float, default=0.20)
    parser.add_argument("--recover-forward-turn-rate", type=float, default=0.55)
    parser.add_argument("--progress-window", type=int, default=35)
    parser.add_argument("--progress-min-delta", type=float, default=0.08)
    parser.add_argument("--adaptive-scheduler", action="store_true")
    parser.add_argument("--adaptive-ema", type=float, default=0.18)
    parser.add_argument("--adaptive-mode-min-steps", type=int, default=14)
    parser.add_argument("--adaptive-tight-clearance", type=float, default=0.38)
    parser.add_argument("--adaptive-tight-clearance-exit", type=float, default=0.52)
    parser.add_argument("--adaptive-stuck-progress", type=float, default=0.06)
    parser.add_argument("--adaptive-osc-window", type=int, default=10)
    parser.add_argument("--adaptive-osc-threshold", type=float, default=2.6)
    parser.add_argument("--adaptive-dock-radius", type=float, default=0.75)
    parser.add_argument("--goal-slowdown-radius", type=float, default=1.8)
    parser.add_argument("--goal-slowdown-min-scale", type=float, default=0.08)
    parser.add_argument("--w-terminal-stop", type=float, default=120.0)
    parser.add_argument("--terminal-stop-radius", type=float, default=0.70)
    parser.add_argument("--w-overshoot", type=float, default=140.0)
    parser.add_argument("--overshoot-tolerance", type=float, default=0.05)
    parser.add_argument("--noise-anneal-dist", type=float, default=1.8)
    parser.add_argument("--noise-anneal-min-scale", type=float, default=0.30)
    parser.add_argument("--w-path-track", type=float, default=14.0)
    parser.add_argument("--w-path-terminal", type=float, default=40.0)
    parser.add_argument("--w-path-progress", type=float, default=90.0)
    parser.add_argument("--w-path-backtrack", type=float, default=180.0)
    parser.add_argument("--w-goal-motion-away", type=float, default=220.0)
    parser.add_argument("--w-reverse-away", type=float, default=160.0)
    parser.add_argument("--w-bounds", type=float, default=120.0)
    parser.add_argument("--w-bounds-terminal", type=float, default=260.0)
    parser.add_argument("--bounds-x-range", type=float, nargs=2, default=(-0.8, 3.2))
    parser.add_argument("--bounds-y-range", type=float, nargs=2, default=(-2.0, 2.0))
    parser.add_argument("--bounds-hard-margin", type=float, default=0.35)

    parser.add_argument("--wheel-radius", type=float, default=0.085)
    parser.add_argument("--wheel-base", type=float, default=0.37)
    parser.add_argument("--yaw-blend-alpha", type=float, default=0.35)
    parser.add_argument("--control-decimation", type=int, default=10)
    parser.add_argument("--pose-source", type=str, default="gt", choices=("gt", "odom"))

    parser.add_argument("--max-steps", type=int, default=1500)
    parser.add_argument("--goal-tol", type=float, default=0.35)
    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-forward-m", type=float, default=2.5)
    parser.add_argument("--random-obstacles", action="store_true")
    parser.add_argument("--scene-seed", type=int, default=-1)
    parser.add_argument("--scene-sample-attempts", type=int, default=800)
    parser.add_argument("--obs-x-range", type=float, nargs=2, default=(0.6, 2.8))
    parser.add_argument("--obs-y-range", type=float, nargs=2, default=(-1.8, 1.8))
    parser.add_argument("--min-obs-obs-dist", type=float, default=0.75)
    parser.add_argument("--min-start-obs-dist", type=float, default=1.05)
    parser.add_argument("--min-goal-obs-dist", type=float, default=1.05)
    parser.add_argument("--constrain-obs-radius", action="store_true", default=True)
    parser.add_argument("--no-constrain-obs-radius", action="store_false", dest="constrain_obs_radius")
    parser.add_argument("--randomize-obs-radius", action="store_true", default=False)
    parser.add_argument("--obs-radius-min", type=float, default=0.16)
    parser.add_argument("--obs-radius-max", type=float, default=0.23)
    parser.add_argument("--scene-require-global-path", action="store_true", default=True)
    parser.add_argument("--no-scene-require-global-path", action="store_false", dest="scene_require_global_path")
    parser.add_argument("--scene-path-inflate-margin", type=float, default=0.10)
    parser.add_argument("--scene-path-grid-res", type=float, default=0.10)
    parser.add_argument("--scene-path-grid-padding", type=float, default=0.60)
    parser.add_argument("--scene-path-max-cells", type=int, default=240000)
    parser.add_argument("--scene-expand-schedule", type=float, nargs="+", default=(0.0, 0.35, 0.70))
    parser.add_argument("--scene-attempt-factors", type=float, nargs="+", default=(1.0, 1.7, 2.5))
    parser.add_argument("--scene-relax-schedule", type=float, nargs="+", default=(1.0, 1.0, 0.95))
    parser.add_argument("--min-line-blockers", type=int, default=1)
    parser.add_argument("--blocker-t-range", type=float, nargs=2, default=(0.30, 0.92))
    parser.add_argument("--blocker-extra-margin", type=float, default=0.04)
    parser.add_argument("--require-mixed-sides", action="store_true", default=True)
    parser.add_argument("--no-require-mixed-sides", action="store_false", dest="require_mixed_sides")
    parser.add_argument("--auto-waypoint", action="store_true")
    parser.add_argument("--global-guide", action="store_true", default=True)
    parser.add_argument("--no-global-guide", action="store_false", dest="global_guide")
    parser.add_argument("--guide-replan-interval", type=int, default=8)
    parser.add_argument("--guide-lookahead", type=float, default=0.9)
    parser.add_argument("--guide-ref-step-m", type=float, default=0.09)
    parser.add_argument("--guide-max-grid-cells", type=int, default=240000)
    parser.add_argument("--goal-direct-on-clear", action="store_true", default=True)
    parser.add_argument("--no-goal-direct-on-clear", action="store_false", dest="goal_direct_on_clear")
    parser.add_argument("--goal-direct-clear-steps", type=int, default=10)
    parser.add_argument("--goal-direct-dist", type=float, default=2.2)
    parser.add_argument("--corridor-commit", action="store_true", default=True)
    parser.add_argument("--no-corridor-commit", action="store_false", dest="corridor_commit")
    parser.add_argument("--commit-latch-steps", type=int, default=80)
    parser.add_argument("--commit-release-progress", type=float, default=0.35)
    parser.add_argument("--commit-lateral-bias", type=float, default=0.22)
    parser.add_argument("--commit-max-path-dev", type=float, default=0.70)
    parser.add_argument("--commit-stuck-window", type=int, default=40)
    parser.add_argument("--commit-stuck-min-progress", type=float, default=0.05)
    parser.add_argument("--waypoint-use-los-gating", action="store_true", default=True)
    parser.add_argument("--no-waypoint-use-los-gating", action="store_false", dest="waypoint_use_los_gating")
    parser.add_argument("--waypoint-replan-interval", type=int, default=10)
    parser.add_argument("--waypoint-switch-radius", type=float, default=0.35)
    parser.add_argument("--waypoint-clear-hysteresis-steps", type=int, default=8)
    parser.add_argument("--waypoint-min-hold-steps", type=int, default=12)
    parser.add_argument("--waypoint-margin", type=float, default=0.12)
    parser.add_argument("--goal-los-margin", type=float, default=0.02)
    parser.add_argument("--waypoint-lateral-extra", type=float, default=0.28)
    parser.add_argument("--waypoint-stuck-window", type=int, default=40)
    parser.add_argument("--waypoint-stuck-min-progress", type=float, default=0.08)
    parser.add_argument("--waypoint-stuck-cooldown", type=int, default=20)
    parser.add_argument("--waypoint-stuck-flip-side", action="store_true", default=True)
    parser.add_argument("--no-waypoint-stuck-flip-side", action="store_false", dest="waypoint_stuck_flip_side")
    parser.add_argument("--init-x", type=float, default=None)
    parser.add_argument("--init-y", type=float, default=None)
    parser.add_argument("--init-yaw", type=float, default=None)
    parser.add_argument("--stop-on-goal", action="store_true")
    parser.add_argument("--dock-brake-radius", type=float, default=0.90)
    parser.add_argument("--dock-enter-radius", type=float, default=0.40)
    parser.add_argument("--dock-exit-radius", type=float, default=0.65)
    parser.add_argument("--dock-k-brake", type=float, default=0.35)
    parser.add_argument("--dock-k-yaw", type=float, default=0.90)
    parser.add_argument("--dock-v-eps", type=float, default=0.08)
    parser.add_argument("--dock-w-eps", type=float, default=0.25)
    parser.add_argument("--dock-hold-steps", type=int, default=20)
    parser.add_argument("--print-interval", type=int, default=20)
    parser.add_argument("--no-realtime", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    kin_model = DiffDriveKinematicModel(
        state_dim=7,
        action_dim=2,
        dt=0.02,
        wheel_radius=args.wheel_radius,
        wheel_base=args.wheel_base,
        drive_sign=args.v_forward_sign,
    ).to(device)
    kin_model.eval()
    if args.planner_model == "kinematic":
        model = kin_model
    else:
        icode_model = load_icode_checkpoint(args.checkpoint, device=device)
        if args.planner_model == "hybrid" or args.icode_blend_alpha < 0.999:
            model = HybridDynamics(icode_model=icode_model, kin_model=kin_model, alpha=args.icode_blend_alpha).to(device)
            model.eval()
        else:
            model = icode_model

    env = E1RobotEnv(
        xml_path=str(args.xml),
        wheel_radius=args.wheel_radius,
        wheel_base=args.wheel_base,
        yaw_blend_alpha=args.yaw_blend_alpha,
        control_decimation=args.control_decimation,
        pose_source=args.pose_source,
        heading_source="camera",
    )
    mppi = MPPIController(
        dynamics_model=model,
        num_samples=args.num_samples,
        horizon=args.horizon,
        dim_action=2,
        lambda_=args.mppi_lambda,
        noise_sigma=np.array(args.noise_sigma, dtype=np.float32),
        noise_rho=args.noise_rho,
        u_init=np.array(args.u_init, dtype=np.float32),
        action_low=env.ctrl_low,
        action_high=env.ctrl_high,
        device=str(device),
        w_goal=args.w_goal,
        w_final=args.w_final,
        w_progress=args.w_progress,
        w_control=args.w_control,
        w_smooth=args.w_smooth,
        w_spin=args.w_spin,
        w_collision=args.w_collision,
        w_near_obs=args.w_near_obs,
        w_heading=args.w_heading,
        w_reverse=args.w_reverse,
        w_away_goal=args.w_away_goal,
        obs_behind_scale=args.obs_behind_scale,
        v_forward_sign=args.v_forward_sign,
        robot_radius=args.robot_radius,
        obs_margin=args.obs_margin,
        w_goal_visibility=args.w_goal_visibility,
        w_near_goal_stall=args.w_near_goal_stall,
        near_goal_radius=args.near_goal_radius,
        near_goal_progress_eps=args.near_goal_progress_eps,
        los_margin=args.los_margin,
        w_terminal_stop=args.w_terminal_stop,
        terminal_stop_radius=args.terminal_stop_radius,
        w_overshoot=args.w_overshoot,
        overshoot_tolerance=args.overshoot_tolerance,
        noise_anneal_dist=args.noise_anneal_dist,
        noise_anneal_min_scale=args.noise_anneal_min_scale,
        w_path_track=args.w_path_track,
        w_path_terminal=args.w_path_terminal,
        w_path_progress=args.w_path_progress,
        w_path_backtrack=args.w_path_backtrack,
        w_goal_motion_away=args.w_goal_motion_away,
        w_reverse_away=args.w_reverse_away,
        w_bounds=args.w_bounds,
        w_bounds_terminal=args.w_bounds_terminal,
        world_x_min=min(args.bounds_x_range[0], args.bounds_x_range[1]),
        world_x_max=max(args.bounds_x_range[0], args.bounds_x_range[1]),
        world_y_min=min(args.bounds_y_range[0], args.bounds_y_range[1]),
        world_y_max=max(args.bounds_y_range[0], args.bounds_y_range[1]),
    )
    adaptive_base = {
        "w_goal": float(mppi.w_goal),
        "w_collision": float(mppi.w_collision),
        "w_near_obs": float(mppi.w_near_obs),
        "w_path_track": float(mppi.w_path_track),
        "w_path_progress": float(mppi.w_path_progress),
        "w_smooth": float(mppi.w_smooth),
        "w_terminal_stop": float(mppi.w_terminal_stop),
        "w_path_backtrack": float(mppi.w_path_backtrack),
        "w_goal_motion_away": float(mppi.w_goal_motion_away),
        "w_reverse_away": float(mppi.w_reverse_away),
        "noise_sigma": mppi.noise_sigma.detach().clone(),
    }

    if args.init_x is not None and args.init_y is not None and args.init_yaw is not None:
        state = env.reset_with_pose(x=args.init_x, y=args.init_y, yaw=args.init_yaw)
    else:
        state = env.reset()
    base_xy0 = env.get_base_xy_gt()
    yaw0 = float(state[2])
    target_xy = env.get_goal_xy()
    if args.target_x is not None and args.target_y is not None:
        target_xy = np.array([args.target_x, args.target_y], dtype=np.float32)
        env.set_goal_xy(float(target_xy[0]), float(target_xy[1]))
    elif args.target_forward_m > 0.0:
        target_xy = base_xy0 + float(args.target_forward_m) * np.array([np.cos(yaw0), np.sin(yaw0)], dtype=np.float32)
        env.set_goal_xy(float(target_xy[0]), float(target_xy[1]))
    obstacles = env.get_obstacles_xyr()
    random_scene_success = True
    scene_sampling_stage = 0
    if args.random_obstacles:
        scene_seed = int(args.scene_seed) if args.scene_seed >= 0 else int(args.seed + 17)
        rng_scene = np.random.default_rng(scene_seed)
        if args.constrain_obs_radius:
            if args.randomize_obs_radius:
                rr = rng_scene.uniform(
                    float(args.obs_radius_min),
                    float(args.obs_radius_max),
                    size=(obstacles.shape[0],),
                ).astype(np.float32)
            else:
                rr = np.clip(
                    obstacles[:, 2],
                    float(args.obs_radius_min),
                    float(args.obs_radius_max),
                ).astype(np.float32)
            env.set_obstacles_radii(rr)
            obstacles = env.get_obstacles_xyr()
        ok, sampled_obs, stage_idx = sample_obstacles_adaptive(
            rng=rng_scene,
            base_obstacles_xyr=obstacles,
            start_xy=base_xy0.astype(np.float32),
            goal_xy=target_xy.astype(np.float32),
            obs_x_range=(float(args.obs_x_range[0]), float(args.obs_x_range[1])),
            obs_y_range=(float(args.obs_y_range[0]), float(args.obs_y_range[1])),
            min_obs_obs_dist=float(args.min_obs_obs_dist),
            min_start_obs_dist=float(args.min_start_obs_dist),
            min_goal_obs_dist=float(args.min_goal_obs_dist),
            attempts=int(args.scene_sample_attempts),
            require_global_path=bool(args.scene_require_global_path),
            robot_radius=float(args.robot_radius),
            path_inflation_margin=float(args.scene_path_inflate_margin),
            path_grid_resolution=float(args.scene_path_grid_res),
            path_grid_padding=float(args.scene_path_grid_padding),
            path_max_grid_cells=int(args.scene_path_max_cells),
            expand_schedule=tuple(float(v) for v in args.scene_expand_schedule),
            attempt_factors=tuple(float(v) for v in args.scene_attempt_factors),
            relax_schedule=tuple(float(v) for v in args.scene_relax_schedule),
            min_line_blockers=int(args.min_line_blockers),
            blocker_t_range=(float(args.blocker_t_range[0]), float(args.blocker_t_range[1])),
            blocker_extra_margin=float(args.blocker_extra_margin),
            require_mixed_sides=bool(args.require_mixed_sides),
        )
        random_scene_success = bool(ok)
        scene_sampling_stage = int(stage_idx)
        if ok:
            env.set_obstacles_xy(sampled_obs[:, :2])
            obstacles = env.get_obstacles_xyr()

    tmin = float(min(args.blocker_t_range[0], args.blocker_t_range[1]))
    tmax = float(max(args.blocker_t_range[0], args.blocker_t_range[1]))
    line_blockers = 0
    for i in range(obstacles.shape[0]):
        dseg, tseg = point_segment_distance_and_t(base_xy0.astype(np.float32), target_xy.astype(np.float32), obstacles[i, :2])
        r_eff = float(obstacles[i, 2] + args.robot_radius + args.scene_path_inflate_margin + args.blocker_extra_margin)
        if tmin <= tseg <= tmax and dseg < r_eff:
            line_blockers += 1

    def world_to_body(vec_xy: np.ndarray, yaw: float) -> np.ndarray:
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([c * vec_xy[0] + s * vec_xy[1], -s * vec_xy[0] + c * vec_xy[1]], dtype=np.float32)

    def diff_drive_to_wheels(v_cmd: float, w_cmd: float) -> np.ndarray:
        dq_r = (v_cmd + 0.5 * args.wheel_base * w_cmd) / max(args.wheel_radius, 1e-6)
        dq_l = (v_cmd - 0.5 * args.wheel_base * w_cmd) / max(args.wheel_radius, 1e-6)
        return np.array([dq_l, dq_r], dtype=np.float32)

    def wrap_to_pi(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    def pick_turn_sign_from_nearest_obstacle(state_now: np.ndarray, base_xy_now: np.ndarray) -> float:
        if obstacles.shape[0] <= 0:
            return 1.0
        yaw = float(state_now[2])
        nearest_i = int(np.argmin(np.linalg.norm(obstacles[:, :2] - base_xy_now[None, :], axis=1)))
        rel_world = obstacles[nearest_i, :2] - base_xy_now
        rel_body = world_to_body(rel_world, yaw)
        return -1.0 if rel_body[1] > 0.0 else 1.0

    def recover_action(mode: str, turn_sign: float) -> np.ndarray:
        if mode == "STOP":
            v_cmd = 0.0
            w_cmd = 0.0
        elif mode == "ROTATE":
            v_cmd = 0.0
            w_cmd = float(turn_sign) * float(args.recover_turn_rate)
        else:  # FORWARD
            v_cmd = float(args.recover_forward_speed)
            w_cmd = float(turn_sign) * float(args.recover_forward_turn_rate)
        u = diff_drive_to_wheels(v_cmd=v_cmd, w_cmd=w_cmd)
        return np.clip(u, env.ctrl_low, env.ctrl_high).astype(np.float32)

    def boundary_guard_action(state_now: np.ndarray, base_xy_now: np.ndarray) -> np.ndarray:
        x_min = float(min(args.bounds_x_range[0], args.bounds_x_range[1]))
        x_max = float(max(args.bounds_x_range[0], args.bounds_x_range[1]))
        y_min = float(min(args.bounds_y_range[0], args.bounds_y_range[1]))
        y_max = float(max(args.bounds_y_range[0], args.bounds_y_range[1]))
        center_xy = np.array([(x_min + x_max) * 0.5, (y_min + y_max) * 0.5], dtype=np.float32)
        yaw = float(state_now[2])
        rel_body = world_to_body(center_xy - base_xy_now, yaw)
        yaw_err = float(np.arctan2(rel_body[1], max(rel_body[0], 1e-6)))
        if abs(yaw_err) > 0.35:
            v_cmd = 0.0
            w_cmd = float(np.sign(yaw_err)) * float(args.recover_turn_rate)
        else:
            v_cmd = float(args.recover_forward_speed)
            w_cmd = 0.0
        u = diff_drive_to_wheels(v_cmd=v_cmd, w_cmd=w_cmd)
        return np.clip(u, env.ctrl_low, env.ctrl_high).astype(np.float32)

    def dock_action(state_now: np.ndarray, base_xy_now: np.ndarray, goal_xy_now: np.ndarray) -> np.ndarray:
        yaw = float(state_now[2])
        dq_l = float(state_now[5]) if state_now.shape[0] > 5 else 0.0
        dq_r = float(state_now[6]) if state_now.shape[0] > 6 else 0.0
        goal_heading = float(np.arctan2(goal_xy_now[1] - base_xy_now[1], goal_xy_now[0] - base_xy_now[0]))
        yaw_err = wrap_to_pi(goal_heading - yaw)
        brake = -args.dock_k_brake * np.array([dq_l, dq_r], dtype=np.float32)
        turn = args.dock_k_yaw * yaw_err
        u = brake + np.array([-turn, +turn], dtype=np.float32)
        return np.clip(u, env.ctrl_low, env.ctrl_high).astype(np.float32)

    ctrl_dt = env.model.opt.timestep * env.control_decimation
    goal_xy = target_xy.copy()
    waypoint_xy = None
    waypoint_last_side = 0
    waypoint_forced_side = 0
    waypoint_replans = 0
    waypoint_active_steps = 0
    waypoint_stuck_events = 0
    goal_dist_buffer = []
    waypoint_stuck_cooldown = 0
    goal_los_clear_count = 0
    waypoint_hold_steps = 0
    goal_blocked_steps = 0
    guide_path_xy = None
    guide_progress_idx = 0
    guide_replans = 0
    guide_active_steps = 0
    guide_fail_steps = 0
    guide_commit_side = 0
    guide_commit_until_step = -1
    guide_commit_dist_ref = 0.0
    guide_commit_buf = []
    guide_commit_active_steps = 0
    guide_commit_flip_events = 0
    goal_direct_steps = 0
    nav_mode = "NAV"
    dock_hold_count = 0
    print("Launching MuJoCo viewer...")
    print(f"target={target_xy.tolist()}, goal_tol={args.goal_tol}, ctrl_dt={ctrl_dt:.4f}s")
    print(
        f"random_obstacles={args.random_obstacles}, scene_success={random_scene_success}, "
        f"scene_stage={scene_sampling_stage}, line_blockers={line_blockers}, "
        f"global_guide={args.global_guide}, auto_waypoint={args.auto_waypoint}, "
        f"adaptive={args.adaptive_scheduler}"
    )
    print(f"obstacles(x,y,r)=\n{np.array2string(obstacles, precision=3)}")

    collision_steps = 0
    recover_events = 0
    recover_collision_events = 0
    recover_stall_events = 0
    boundary_guard_steps = 0
    progress_gate_events = 0
    recover_mode = "NONE"
    recover_phase_left = 0
    recover_turn_sign = 1.0
    path_remain_hist = []
    action_delta_hist = []
    adaptive_mode = "OPEN"
    adaptive_mode_hold = 0
    adaptive_switches = 0
    adaptive_mode_steps = {"OPEN": 0, "TIGHT": 0, "STUCK": 0, "DOCK": 0}
    prev_action = np.zeros((2,), dtype=np.float32)
    min_dist = float(np.linalg.norm(env.get_base_xy_gt() - target_xy))
    base_xy_hist = [env.get_base_xy_gt().copy()]
    x_min = float(min(args.bounds_x_range[0], args.bounds_x_range[1]))
    x_max = float(max(args.bounds_x_range[0], args.bounds_x_range[1]))
    y_min = float(min(args.bounds_y_range[0], args.bounds_y_range[1]))
    y_max = float(max(args.bounds_y_range[0], args.bounds_y_range[1]))
    bounds_margin = float(max(0.0, args.bounds_hard_margin))
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(args.max_steps):
            if not viewer.is_running():
                break

            t0 = time.time()
            base_xy_now = env.get_base_xy_gt()
            dist_goal_now = float(np.linalg.norm(base_xy_now - goal_xy))
            touch_now = env.get_touch_force()
            pre_clearances = np.linalg.norm(obstacles[:, :2] - base_xy_now[None, :], axis=1) - (obstacles[:, 2] + args.robot_radius)
            pre_min_clearance = float(np.min(pre_clearances))
            near_collision = (
                pre_min_clearance < args.emergency_clearance
                or touch_now > args.collision_touch_threshold
            )
            out_of_bounds_hard = (
                base_xy_now[0] < (x_min + bounds_margin)
                or base_xy_now[0] > (x_max - bounds_margin)
                or base_xy_now[1] < (y_min + bounds_margin)
                or base_xy_now[1] > (y_max - bounds_margin)
            )
            if near_collision and recover_mode == "NONE":
                recover_mode = "STOP"
                recover_phase_left = max(1, int(args.recover_stop_steps))
                recover_turn_sign = pick_turn_sign_from_nearest_obstacle(state_now=state, base_xy_now=base_xy_now)
                recover_events += 1
                recover_collision_events += 1

            goal_blocked = line_of_sight_blocked(
                start_xy=base_xy_now,
                goal_xy=goal_xy,
                obstacles_xyr=obstacles,
                robot_radius=args.robot_radius,
                margin=args.goal_los_margin,
            )
            if goal_blocked:
                goal_blocked_steps += 1
                goal_los_clear_count = 0
            else:
                goal_los_clear_count += 1

            dpath_recent = None
            if len(path_remain_hist) >= 2:
                dpath_recent = float(path_remain_hist[0] - path_remain_hist[-1])
            osc_now = 0.0
            if len(action_delta_hist) > 0:
                osc_now = float(np.mean(action_delta_hist))
            if args.adaptive_scheduler:
                dock_gate = float(max(args.adaptive_dock_radius, args.goal_tol * 1.3))
                stuck_gate = float(max(args.adaptive_stuck_progress, 1e-4))
                desired_mode = "OPEN"
                if dist_goal_now <= dock_gate:
                    desired_mode = "DOCK"
                else:
                    in_tight = (
                        pre_min_clearance <= float(args.adaptive_tight_clearance)
                        or (
                            adaptive_mode == "TIGHT"
                            and pre_min_clearance <= float(args.adaptive_tight_clearance_exit)
                        )
                    )
                    stuck_cond = (
                        dpath_recent is not None
                        and dpath_recent < stuck_gate
                        and dist_goal_now > dock_gate
                        and (osc_now > float(args.adaptive_osc_threshold) or adaptive_mode == "STUCK")
                    )
                    if in_tight:
                        desired_mode = "TIGHT"
                    elif stuck_cond:
                        desired_mode = "STUCK"

                min_steps = max(1, int(args.adaptive_mode_min_steps))
                if desired_mode != adaptive_mode and adaptive_mode_hold >= min_steps:
                    print(
                        f"[adaptive] step={step:04d} {adaptive_mode}->{desired_mode} "
                        f"clear={pre_min_clearance:.3f} dpath={dpath_recent if dpath_recent is not None else float('nan'):.3f} "
                        f"osc={osc_now:.3f} dist={dist_goal_now:.3f}"
                    )
                    adaptive_mode = desired_mode
                    adaptive_mode_hold = 0
                    adaptive_switches += 1
                adaptive_mode_hold += 1
                adaptive_mode_steps[adaptive_mode] += 1
                _apply_adaptive_profile(
                    mppi=mppi,
                    base=adaptive_base,
                    profile=_adaptive_profile(adaptive_mode),
                    alpha=float(args.adaptive_ema),
                )
            else:
                adaptive_mode = "OPEN"
                adaptive_mode_steps["OPEN"] += 1

            if nav_mode == "NAV" and dist_goal_now <= args.dock_brake_radius:
                nav_mode = "BRAKE_ALIGN"
                dock_hold_count = 0
            elif nav_mode == "BRAKE_ALIGN":
                if dist_goal_now <= args.dock_enter_radius:
                    nav_mode = "DOCK_STOP"
                    dock_hold_count = 0
                elif dist_goal_now > args.dock_exit_radius:
                    nav_mode = "NAV"
            elif nav_mode == "DOCK_STOP":
                if dist_goal_now > args.dock_exit_radius:
                    nav_mode = "BRAKE_ALIGN"
                    dock_hold_count = 0

            if args.auto_waypoint and args.waypoint_stuck_window > 1:
                goal_dist_buffer.append(dist_goal_now)
                if len(goal_dist_buffer) > args.waypoint_stuck_window:
                    goal_dist_buffer.pop(0)
                if waypoint_stuck_cooldown > 0:
                    waypoint_stuck_cooldown -= 1
                if len(goal_dist_buffer) == args.waypoint_stuck_window and waypoint_stuck_cooldown == 0:
                    progress = goal_dist_buffer[0] - goal_dist_buffer[-1]
                    if progress < args.waypoint_stuck_min_progress:
                        waypoint_stuck_events += 1
                        waypoint_xy = None
                        waypoint_hold_steps = 0
                        goal_dist_buffer = [goal_dist_buffer[-1]]
                        if args.waypoint_stuck_flip_side:
                            if waypoint_last_side != 0:
                                waypoint_forced_side = -waypoint_last_side
                            else:
                                waypoint_forced_side = 1 if (waypoint_stuck_events % 2 == 1) else -1
                        waypoint_stuck_cooldown = args.waypoint_stuck_cooldown

            if out_of_bounds_hard:
                action = boundary_guard_action(state_now=state, base_xy_now=base_xy_now)
                boundary_guard_steps += 1
                recover_mode = "NONE"
                recover_phase_left = 0
            elif nav_mode == "DOCK_STOP":
                action = dock_action(state_now=state, base_xy_now=base_xy_now, goal_xy_now=goal_xy)
            elif recover_mode != "NONE":
                action = recover_action(mode=recover_mode, turn_sign=recover_turn_sign)
                recover_phase_left -= 1
                if recover_phase_left <= 0:
                    if recover_mode == "STOP":
                        recover_mode = "ROTATE"
                        recover_phase_left = max(1, int(args.recover_rotate_steps))
                    elif recover_mode == "ROTATE":
                        recover_mode = "FORWARD"
                        recover_phase_left = max(1, int(args.recover_forward_steps))
                    else:
                        recover_mode = "NONE"
                        recover_phase_left = 0
            else:
                target_for_mppi = goal_xy
                reference_traj = None
                used_global_guide = False
                goal_direct_mode = (
                    bool(args.goal_direct_on_clear)
                    and goal_los_clear_count >= max(1, int(args.goal_direct_clear_steps))
                    and dist_goal_now <= float(args.goal_direct_dist)
                )
                if goal_direct_mode:
                    goal_direct_steps += 1
                    guide_commit_side = 0
                    guide_commit_buf = []
                    guide_commit_until_step = -1
                    path_remain_hist = []
                    used_global_guide = True
                elif args.global_guide:
                    need_guide_replan = (step % max(1, args.guide_replan_interval) == 0) or (guide_path_xy is None)
                    if guide_path_xy is not None and guide_path_xy.shape[0] > 0:
                        dist_to_path = float(np.min(np.linalg.norm(guide_path_xy - base_xy_now[None, :], axis=1)))
                        if args.corridor_commit and guide_commit_side != 0 and step <= guide_commit_until_step:
                            if dist_to_path <= float(args.commit_max_path_dev):
                                need_guide_replan = False
                            else:
                                need_guide_replan = True
                        elif dist_to_path > max(0.1, float(args.waypoint_switch_radius)):
                            need_guide_replan = True
                    if need_guide_replan:
                        guide_replans += 1
                        guide_path_xy = plan_global_path_xy(
                            start_xy=base_xy_now.astype(np.float32),
                            goal_xy=goal_xy.astype(np.float32),
                            obstacles_xyr=obstacles.astype(np.float32),
                            robot_radius=float(args.robot_radius),
                            inflation_margin=float(args.scene_path_inflate_margin),
                            grid_resolution=float(args.scene_path_grid_res),
                            grid_padding=float(args.scene_path_grid_padding),
                            max_grid_cells=int(args.guide_max_grid_cells),
                        )
                        guide_progress_idx = 0
                    if guide_path_xy is not None and guide_path_xy.shape[0] >= 2:
                        path_remain_now, guide_idx_rem = compute_path_remaining(
                            path_xy=guide_path_xy,
                            current_xy=base_xy_now.astype(np.float32),
                            min_index=int(guide_progress_idx),
                        )
                        guide_progress_idx = max(int(guide_progress_idx), int(guide_idx_rem))
                        target_for_mppi, guide_idx = select_path_lookahead_target(
                            path_xy=guide_path_xy,
                            current_xy=base_xy_now.astype(np.float32),
                            lookahead_m=float(args.guide_lookahead),
                            min_index=int(guide_progress_idx),
                        )
                        guide_progress_idx = max(int(guide_progress_idx), int(guide_idx))
                        path_remain_hist.append(float(path_remain_now))
                        if len(path_remain_hist) > max(2, int(args.progress_window)):
                            path_remain_hist.pop(0)
                        reference_traj = build_reference_traj_from_path(
                            path_xy=guide_path_xy,
                            start_index=int(guide_progress_idx),
                            horizon=int(args.horizon),
                            step_m=float(args.guide_ref_step_m),
                        )
                        if args.corridor_commit:
                            _, lat_ref = line_signed_lateral(
                                start_xy=base_xy0.astype(np.float32),
                                goal_xy=goal_xy.astype(np.float32),
                                point_xy=np.asarray(target_for_mppi, dtype=np.float32),
                            )
                            if guide_commit_side == 0 and goal_blocked and abs(lat_ref) > 0.05:
                                guide_commit_side = 1 if lat_ref >= 0.0 else -1
                                guide_commit_until_step = step + max(1, int(args.commit_latch_steps))
                                guide_commit_dist_ref = dist_goal_now
                                guide_commit_buf = [dist_goal_now]
                            if guide_commit_side != 0:
                                goal_vec = goal_xy - base_xy_now
                                gnorm = float(np.linalg.norm(goal_vec))
                                if gnorm > 1e-6:
                                    gdir = goal_vec / gnorm
                                    gperp = np.array([-gdir[1], gdir[0]], dtype=np.float32)
                                    target_for_mppi = np.asarray(target_for_mppi, dtype=np.float32) + (
                                        float(guide_commit_side) * float(args.commit_lateral_bias) * gperp
                                    )
                                guide_commit_active_steps += 1
                                guide_commit_buf.append(dist_goal_now)
                                if len(guide_commit_buf) > max(2, int(args.commit_stuck_window)):
                                    guide_commit_buf.pop(0)
                                if len(guide_commit_buf) >= max(2, int(args.commit_stuck_window)):
                                    pbuf = guide_commit_buf[0] - guide_commit_buf[-1]
                                    if pbuf < float(args.commit_stuck_min_progress):
                                        guide_commit_side = -guide_commit_side
                                        guide_commit_until_step = step + max(1, int(args.commit_latch_steps))
                                        guide_commit_dist_ref = dist_goal_now
                                        guide_commit_buf = [dist_goal_now]
                                        guide_commit_flip_events += 1
                                if (guide_commit_dist_ref - dist_goal_now) >= float(args.commit_release_progress):
                                    guide_commit_side = 0
                                    guide_commit_buf = []
                                elif step > guide_commit_until_step:
                                    guide_commit_side = 0
                                    guide_commit_buf = []
                        used_global_guide = True
                        guide_active_steps += 1
                    else:
                        path_remain_hist = []
                        guide_fail_steps += 1

                if (not used_global_guide) and args.auto_waypoint:
                    if waypoint_xy is not None:
                        if float(np.linalg.norm(base_xy_now - waypoint_xy)) <= args.waypoint_switch_radius:
                            waypoint_xy = None
                            waypoint_hold_steps = 0
                        else:
                            waypoint_hold_steps += 1
                            if (
                                args.waypoint_use_los_gating
                                and (not goal_blocked)
                                and goal_los_clear_count >= args.waypoint_clear_hysteresis_steps
                                and waypoint_hold_steps >= args.waypoint_min_hold_steps
                            ):
                                waypoint_xy = None
                                waypoint_hold_steps = 0
                    need_replan = (step % max(1, args.waypoint_replan_interval) == 0) or (waypoint_xy is None)
                    should_replan = True
                    if args.waypoint_use_los_gating and (not goal_blocked) and (waypoint_xy is None):
                        should_replan = False
                    if need_replan and should_replan:
                        wp, side = compute_auto_waypoint_with_side(
                            start_xy=base_xy_now,
                            goal_xy=goal_xy,
                            obstacles_xyr=obstacles,
                            robot_radius=args.robot_radius,
                            margin=args.waypoint_margin,
                            lateral_extra=args.waypoint_lateral_extra,
                            preferred_side=waypoint_forced_side,
                        )
                        waypoint_replans += 1
                        waypoint_xy = wp
                        waypoint_last_side = int(side)
                        waypoint_forced_side = 0
                        waypoint_hold_steps = 0
                    if waypoint_xy is not None:
                        target_for_mppi = waypoint_xy
                        waypoint_active_steps += 1
                if not used_global_guide:
                    path_remain_hist = []

                progress_stalled = False
                if used_global_guide and len(path_remain_hist) >= max(2, int(args.progress_window)):
                    dpath = float(path_remain_hist[0] - path_remain_hist[-1])
                    if dpath < float(args.progress_min_delta) and dist_goal_now > float(args.goal_tol):
                        progress_stalled = True
                        progress_gate_events += 1
                        path_remain_hist = [path_remain_hist[-1]]

                if progress_stalled:
                    recover_mode = "STOP"
                    recover_phase_left = max(1, int(args.recover_stop_steps))
                    recover_turn_sign = pick_turn_sign_from_nearest_obstacle(state_now=state, base_xy_now=base_xy_now)
                    recover_events += 1
                    recover_stall_events += 1
                    waypoint_xy = None
                    waypoint_hold_steps = 0
                    guide_path_xy = None
                    guide_progress_idx = 0
                    guide_commit_side = 0
                    guide_commit_buf = []
                    action = recover_action(mode="STOP", turn_sign=recover_turn_sign)
                else:
                    action = mppi.get_action(
                        initial_state=state,
                        target=target_for_mppi,
                        obstacles=obstacles,
                        reference_traj=reference_traj,
                    )

            if nav_mode == "BRAKE_ALIGN":
                denom = max(args.dock_brake_radius - args.dock_enter_radius, 1e-6)
                frac = (dist_goal_now - args.dock_enter_radius) / denom
                brake_scale = float(np.clip(frac, 0.15, 1.0))
                action = action * brake_scale

            if args.goal_slowdown_radius > 0.0:
                dist_now = float(np.linalg.norm(base_xy_now - goal_xy))
                if dist_now < args.goal_slowdown_radius:
                    ratio = dist_now / max(args.goal_slowdown_radius, 1e-6)
                    scale = float(np.clip(ratio, args.goal_slowdown_min_scale, 1.0))
                    action = action * scale

            if args.warmup_steps > 0 and step < args.warmup_steps:
                frac = float(step + 1) / float(max(args.warmup_steps, 1))
                scale = args.warmup_min_scale + (1.0 - args.warmup_min_scale) * frac
                action = action * float(np.clip(scale, 0.0, 1.0))

            if 0.0 < args.action_ema_alpha < 1.0:
                action = args.action_ema_alpha * prev_action + (1.0 - args.action_ema_alpha) * action

            if args.max_delta_u > 0.0:
                action = np.clip(action, prev_action - args.max_delta_u, prev_action + args.max_delta_u)
            if (not out_of_bounds_hard) and recover_mode == "NONE":
                goal_vec = goal_xy - base_xy_now
                goal_norm = float(np.linalg.norm(goal_vec))
                if goal_norm > 1e-6 and pre_min_clearance > max(0.18, 1.5 * float(args.emergency_clearance)):
                    goal_body = world_to_body(goal_vec / goal_norm, float(state[2]))
                    mean_u = 0.5 * float(action[0] + action[1])
                    if goal_body[0] > 0.12 and mean_u < 0.0:
                        action = action - mean_u
            if args.startup_no_reverse_steps > 0 and step < args.startup_no_reverse_steps:
                action = np.maximum(action, args.startup_min_u)
            action = np.clip(action, env.ctrl_low, env.ctrl_high).astype(np.float32)
            du_now = float(np.linalg.norm(action - prev_action))
            action_delta_hist.append(du_now)
            if len(action_delta_hist) > max(1, int(args.adaptive_osc_window)):
                action_delta_hist.pop(0)
            prev_action = action.copy()
            state = env.step(action)
            viewer.sync()

            base_xy = env.get_base_xy_gt()
            base_xy_hist.append(base_xy.copy())
            dist = float(np.linalg.norm(base_xy - goal_xy))
            min_dist = min(min_dist, dist)
            touch_force = env.get_touch_force()
            obs_clearances = np.linalg.norm(obstacles[:, :2] - base_xy[None, :], axis=1) - (obstacles[:, 2] + args.robot_radius)
            min_clearance = float(np.min(obs_clearances))
            if min_clearance < 0.0 or touch_force > args.collision_touch_threshold:
                collision_steps += 1
            if step % max(1, args.print_interval) == 0:
                print(
                    f"step={step:04d} dist={dist:.3f} "
                    f"xy=({base_xy[0]:.3f},{base_xy[1]:.3f}) "
                    f"u=({action[0]:.2f},{action[1]:.2f}) "
                    f"clear={min_clearance:.3f} mode={nav_mode}/{recover_mode}/{adaptive_mode} "
                    f"gdir={goal_direct_steps}"
                )

            if nav_mode == "DOCK_STOP":
                v_abs = abs(float(args.v_forward_sign) * float(state[3])) if state.shape[0] > 3 else 0.0
                wz_abs = abs(float(state[4])) if state.shape[0] > 4 else 0.0
                if dist < args.goal_tol and v_abs < args.dock_v_eps and wz_abs < args.dock_w_eps:
                    dock_hold_count += 1
                else:
                    dock_hold_count = 0
                if dock_hold_count >= max(1, int(args.dock_hold_steps)):
                    print(f"Dock stop achieved at step={step}, dist={dist:.3f}")
                    for _ in range(50):
                        env.step(np.zeros((2,), dtype=np.float32))
                        viewer.sync()
                    break

            if dist < args.goal_tol:
                print(f"Reached goal region at step={step}, dist={dist:.3f}")
                if args.stop_on_goal:
                    for _ in range(50):
                        env.step(np.zeros((2,), dtype=np.float32))
                        viewer.sync()
                    break

            if not args.no_realtime:
                remain = ctrl_dt - (time.time() - t0)
                if remain > 0:
                    time.sleep(remain)

    print(
        f"Viewer run finished. min_dist={min_dist:.3f}, collision_steps={collision_steps}, "
        f"recover_events={recover_events}, recover_collision_events={recover_collision_events}, "
        f"recover_stall_events={recover_stall_events}, progress_gate_events={progress_gate_events}, "
        f"boundary_guard_steps={boundary_guard_steps}, waypoint_replans={waypoint_replans}, "
        f"waypoint_active_steps={waypoint_active_steps}, waypoint_stuck_events={waypoint_stuck_events}, "
        f"guide_replans={guide_replans}, guide_active_steps={guide_active_steps}, guide_fail_steps={guide_fail_steps}, "
        f"commit_active_steps={guide_commit_active_steps}, commit_flip_events={guide_commit_flip_events}, "
        f"goal_direct_steps={goal_direct_steps}, adaptive_switches={adaptive_switches}, "
        f"adaptive_steps={adaptive_mode_steps}"
    )
    print(f"goal_blocked_ratio={goal_blocked_steps / max(1, len(base_xy_hist) - 1):.3f}")


if __name__ == "__main__":
    main()
