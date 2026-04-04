import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

from depth_consistency_gate import DepthConsistencyGate
from env_mujoco import E1RobotEnv
from icode_dynamics import ICODEDynamics
from local_occupancy_map import LocalOccupancyMap
from mppi import DiffDriveKinematicModel, MPPIController
from mppi_nav_utils import (
    blocked_confidence_range_semantics,
    build_reference_traj_from_path,
    compute_boundary_recover_target,
    compute_path_remaining,
    compute_auto_waypoint_with_side,
    line_signed_lateral,
    line_of_sight_blocked,
    line_of_sight_blocked_confidence,
    plan_global_path_xy,
    point_segment_distance_and_t,
    project_target_to_passable_point,
    project_target_with_invariants,
    sample_obstacles_adaptive,
    select_path_lookahead_target,
    waypoint_is_feasible,
)
from obs_geometry import DepthRayModel, LidarRayModel, fuse_rays_to_hits_free
from safety_supervisor import (
    SafetySupervisor,
    SupervisorConfig,
    SupervisorInput,
    SupervisorState,
)
from state_convention import (
    STATE_CONVENTION_VERSION,
    canonical_drive_sign,
    diff_drive_forward,
    diff_drive_inverse,
    world_to_body as sc_world_to_body,
)


DEFAULT_XML = (
    Path(__file__).resolve().parents[2]
    / "E1_Robot"
    / "simulation"
    / "models"
    / "mjcf"
    / "E1_SimpleSensor.xml"
)


def load_icode_checkpoint(ckpt_path: Path, device: torch.device) -> ICODEDynamics:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    model = ICODEDynamics(
        state_dim=int(cfg.get("state_dim", 7)),
        action_dim=int(cfg.get("action_dim", 2)),
        context_dim=int(cfg.get("context_dim", 0)),
        hidden_dim=int(cfg.get("hidden_dim", 640)),
        num_layers=int(cfg.get("num_layers", 6)),
        dt=float(cfg.get("dt", 0.02)),
        predict_obs_distance=bool(cfg.get("predict_obs_distance", False)),
        obs_head_hidden_dim=int(cfg.get("obs_head_hidden_dim", 128)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if bool(cfg.get("input_norm_enabled", False)):
        model.set_input_normalization(
            x_mean=cfg.get("x_mean", [0.0] * int(cfg.get("state_dim", 7))),
            x_std=cfg.get("x_std", [1.0] * int(cfg.get("state_dim", 7))),
            u_mean=cfg.get("u_mean", [0.0] * int(cfg.get("action_dim", 2))),
            u_std=cfg.get("u_std", [1.0] * int(cfg.get("action_dim", 2))),
            ctx_mean=cfg.get("ctx_mean", [0.0] * int(cfg.get("context_dim", 0))),
            ctx_std=cfg.get("ctx_std", [1.0] * int(cfg.get("context_dim", 0))),
            enabled=True,
        )
    model.eval()
    return model


class HybridDynamics(torch.nn.Module):
    def __init__(self, icode_model: torch.nn.Module, kin_model: torch.nn.Module, alpha: float):
        super().__init__()
        self.icode = icode_model
        self.kin = kin_model
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.state_dim = int(getattr(icode_model, "state_dim", 7))
        self.action_dim = int(getattr(icode_model, "action_dim", 2))
        self.context_dim = int(getattr(icode_model, "context_dim", 0))

    def forward(self, x: torch.Tensor, u: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        x_i = self.icode(x, u, ctx)
        x_k = self.kin(x, u)
        return self.alpha * x_i + (1.0 - self.alpha) * x_k

    def predict_obstacle_distance(self, x: torch.Tensor, u: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        if hasattr(self.icode, "predict_obstacle_distance"):
            return self.icode.predict_obstacle_distance(x, u, ctx)
        raise AttributeError("Underlying ICODE model does not expose predict_obstacle_distance().")


def _align_ctx_dim(ctx_vec: np.ndarray, target_dim: int) -> np.ndarray:
    c = np.asarray(ctx_vec, dtype=np.float32).reshape(-1)
    if target_dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    if c.shape[0] == target_dim:
        return c.astype(np.float32)
    if c.shape[0] > target_dim:
        return c[:target_dim].astype(np.float32)
    pad = np.zeros((target_dim - c.shape[0],), dtype=np.float32)
    return np.concatenate([c, pad], axis=0).astype(np.float32)


def _build_icode_sensor_ctx(
    goal_body_now: np.ndarray,
    dist_goal_now: float,
    sector_now: np.ndarray,
    corridor_width_now: float,
    min_clearance_now: float,
    touch_now: float,
    sensor_collision_threshold: float,
    touch_threshold: float,
) -> np.ndarray:
    gb = np.asarray(goal_body_now, dtype=np.float32).reshape(-1)
    gx = float(gb[0]) if gb.shape[0] > 0 else 0.0
    gy = float(gb[1]) if gb.shape[0] > 1 else 0.0
    heading_err = float(np.arctan2(gy, max(gx, 1e-6)))
    sec = np.asarray(sector_now, dtype=np.float32).reshape(-1)
    if sec.shape[0] < 3:
        pad = np.full((3 - sec.shape[0],), float(max(min_clearance_now, 0.0)), dtype=np.float32)
        sec = np.concatenate([sec, pad], axis=0)
    depth_lf = float(sec[0])
    depth_f = float(sec[1])
    depth_rf = float(sec[2])
    depth_collision_flag = 1.0 if float(min_clearance_now) < float(sensor_collision_threshold) else 0.0
    touch_flag = 1.0 if float(touch_now) > float(touch_threshold) else 0.0
    return np.array(
        [
            gx,
            gy,
            float(dist_goal_now),
            heading_err,
            depth_lf,
            depth_f,
            depth_rf,
            float(corridor_width_now),
            float(depth_collision_flag),
            float(touch_flag),
        ],
        dtype=np.float32,
    )


def _adaptive_profile(mode: str) -> dict:
    mode_u = str(mode).upper()
    table = {
        "OPEN": {
            "cost_task_goal": 1.00,
            "cost_safe_collision": 1.00,
            "cost_safe_near": 1.00,
            "cost_task_path_track": 1.00,
            "cost_task_path_progress": 1.00,
            "cost_ctrl_smooth": 1.00,
            "cost_terminal_stop": 1.00,
            "noise_sigma": 1.00,
        },
        "TIGHT": {
            "cost_task_goal": 0.90,
            "cost_safe_collision": 1.35,
            "cost_safe_near": 1.25,
            "cost_task_path_track": 1.25,
            "cost_task_path_progress": 1.05,
            "cost_ctrl_smooth": 1.10,
            "cost_terminal_stop": 0.95,
            "noise_sigma": 0.62,
        },
        "STUCK": {
            "cost_task_goal": 1.35,
            "cost_safe_collision": 0.90,
            "cost_safe_near": 0.90,
            "cost_task_path_track": 1.15,
            "cost_task_path_progress": 1.45,
            "cost_ctrl_smooth": 0.90,
            "cost_terminal_stop": 0.90,
            "noise_sigma": 1.35,
        },
        "DOCK": {
            "cost_task_goal": 1.05,
            "cost_safe_collision": 1.15,
            "cost_safe_near": 1.35,
            "cost_task_path_track": 1.35,
            "cost_task_path_progress": 0.95,
            "cost_ctrl_smooth": 1.30,
            "cost_terminal_stop": 1.60,
            "noise_sigma": 0.42,
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
        "cost_task_goal",
        "cost_safe_collision",
        "cost_safe_near",
        "cost_safe_corridor",
        "cost_safe_pred_obs",
        "cost_task_path_track",
        "cost_task_path_progress",
        "cost_ctrl_smooth",
        "cost_terminal_stop",
    )
    for k in scalar_keys:
        if hasattr(mppi, k) and k in base:
            tgt = float(base[k]) * float(profile.get(k, 1.0))
            cur = float(getattr(mppi, k))
            setattr(mppi, k, (1.0 - a) * cur + a * tgt)

    # Keep legacy aliases in sync for trace/debug paths that still read w_* fields.
    mppi.w_goal = float(mppi.cost_task_goal)
    mppi.w_collision = float(mppi.cost_safe_collision)
    mppi.w_near_obs = float(mppi.cost_safe_near)
    if hasattr(mppi, "cost_safe_corridor"):
        mppi.w_corridor = float(mppi.cost_safe_corridor)
    if hasattr(mppi, "cost_safe_pred_obs"):
        mppi.w_pred_obs = float(mppi.cost_safe_pred_obs)
    mppi.w_path_track = float(mppi.cost_task_path_track)
    mppi.w_path_progress = float(mppi.cost_task_path_progress)
    mppi.w_smooth = float(mppi.cost_ctrl_smooth)
    mppi.w_terminal_stop = float(mppi.cost_terminal_stop)

    if "noise_sigma" in base and hasattr(mppi, "noise_sigma"):
        noise_scale = float(max(0.25, profile.get("noise_sigma", 1.0)))
        tgt = base["noise_sigma"] * noise_scale
        cur = mppi.noise_sigma
        mppi.noise_sigma = torch.clamp((1.0 - a) * cur + a * tgt, min=0.03)


def _try_create_depth_renderer(
    model: mujoco.MjModel,
    width: int,
    height: int,
) -> mujoco.Renderer | None:
    try:
        return mujoco.Renderer(model, width=int(width), height=int(height))
    except Exception as exc:
        print(f"[warn] depth renderer unavailable, fallback to lidar-only: {exc}")
        return None


def _render_depth_image(
    renderer: mujoco.Renderer | None,
    data: mujoco.MjData,
    camera_name: str,
) -> np.ndarray | None:
    if renderer is None:
        return None
    try:
        renderer.update_scene(data, camera=str(camera_name))
        renderer.enable_depth_rendering()
        depth = renderer.render().astype(np.float32)
        renderer.disable_depth_rendering()
        return depth
    except Exception:
        return None


def _merge_obstacle_frames(
    frames: deque[np.ndarray],
    cell_size: float,
    max_points: int,
) -> np.ndarray:
    if len(frames) <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    merged = np.concatenate(list(frames), axis=0)
    if merged.shape[0] <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    bins = np.round(merged[:, :2] / max(float(cell_size), 1e-3), decimals=0).astype(np.int32)
    _, uniq_idx = np.unique(bins, axis=0, return_index=True)
    merged = merged[np.sort(uniq_idx)]
    keep = int(max(1, max_points))
    if merged.shape[0] > keep:
        sel = np.linspace(0, merged.shape[0] - 1, num=keep, dtype=np.int32)
        merged = merged[sel]
    return merged.astype(np.float32)


def _rays_to_world_obstacles(
    base_xy: np.ndarray,
    base_yaw: float,
    rays: list,
    min_range: float,
    max_range: float,
    base_radius: float,
    footprint_scale: float,
    radius_max_scale: float,
) -> np.ndarray:
    if rays is None or len(rays) <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    angles_all = np.asarray([float(r.angle_rad) for r in rays], dtype=np.float32).reshape(-1)
    if angles_all.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    base = np.asarray(base_xy, dtype=np.float32).reshape(2)
    yaw = float(base_yaw)
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    rad_min = float(max(0.04, base_radius))
    rad_max = float(max(rad_min, base_radius * max(1.0, radius_max_scale)))
    fp_scale = float(max(0.0, footprint_scale))
    out: list[np.ndarray] = []
    for r in rays:
        if not bool(getattr(r, "hit", False)):
            continue
        d = float(getattr(r, "distance", 0.0))
        if (not np.isfinite(d)) or d <= float(min_range) or d >= float(max_range):
            continue
        a = float(getattr(r, "angle_rad", 0.0))
        da = np.abs(np.arctan2(np.sin(angles_all - a), np.cos(angles_all - a)))
        da = da[da > 1e-4]
        local_step = float(np.min(da)) if da.size > 0 else float(np.deg2rad(15.0))
        local_step = float(np.clip(local_step, np.deg2rad(3.0), np.deg2rad(35.0)))
        footprint = float(d * np.tan(0.5 * local_step) * fp_scale)
        rr = float(np.clip(max(rad_min, footprint), rad_min, rad_max))
        xb = d * np.cos(a)
        yb = d * np.sin(a)
        pw = np.array([base[0] + c * xb - s * yb, base[1] + s * xb + c * yb, rr], dtype=np.float32)
        out.append(pw)
    if len(out) <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(out, dtype=np.float32)
    cell = max(1e-3, 0.7 * rad_min)
    bins = np.round(arr[:, :2] / cell, decimals=0).astype(np.int32)
    uniq: dict[tuple[int, int], np.ndarray] = {}
    for i in range(arr.shape[0]):
        k = (int(bins[i, 0]), int(bins[i, 1]))
        old = uniq.get(k)
        if old is None or float(arr[i, 2]) > float(old[2]):
            uniq[k] = arr[i]
    return np.stack(list(uniq.values()), axis=0).astype(np.float32) if len(uniq) > 0 else np.zeros((0, 3), dtype=np.float32)


def _fuse_obstacle_sets(
    obs_a: np.ndarray,
    obs_b: np.ndarray,
    cell_size: float,
    max_points: int,
) -> np.ndarray:
    a = np.asarray(obs_a, dtype=np.float32)
    b = np.asarray(obs_b, dtype=np.float32)
    if (a.ndim != 2 or a.shape[0] <= 0) and (b.ndim != 2 or b.shape[0] <= 0):
        return np.zeros((0, 3), dtype=np.float32)
    frames: list[np.ndarray] = []
    if a.ndim == 2 and a.shape[0] > 0:
        frames.append(a)
    if b.ndim == 2 and b.shape[0] > 0:
        frames.append(b)
    merged = np.concatenate(frames, axis=0)
    if merged.shape[0] <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    cell = max(1e-3, float(cell_size))
    bins = np.round(merged[:, :2] / cell, decimals=0).astype(np.int32)
    uniq: dict[tuple[int, int], np.ndarray] = {}
    for i in range(merged.shape[0]):
        k = (int(bins[i, 0]), int(bins[i, 1]))
        old = uniq.get(k)
        if old is None or float(merged[i, 2]) > float(old[2]):
            uniq[k] = merged[i]
    out = np.stack(list(uniq.values()), axis=0).astype(np.float32)
    keep = int(max(1, max_points))
    if out.shape[0] > keep:
        sel = np.linspace(0, out.shape[0] - 1, num=keep, dtype=np.int32)
        out = out[sel]
    return out.astype(np.float32)


def _sanitize_lidar_scan(
    ranges: np.ndarray,
    angles_deg: np.ndarray,
    default_far: float,
) -> tuple[np.ndarray, np.ndarray]:
    rr = np.asarray(ranges, dtype=np.float32).reshape(-1)
    aa = np.asarray(angles_deg, dtype=np.float32).reshape(-1)
    if rr.shape[0] <= 0:
        rr = np.asarray([float(default_far), float(default_far), float(default_far)], dtype=np.float32)
        aa = np.asarray([30.0, 0.0, -30.0], dtype=np.float32)
    if aa.shape[0] != rr.shape[0]:
        if rr.shape[0] == 3:
            aa = np.asarray([30.0, 0.0, -30.0], dtype=np.float32)
        else:
            aa = np.linspace(-90.0, 90.0, num=rr.shape[0], dtype=np.float32)
    bad = (~np.isfinite(rr)) | (rr <= 0.0)
    rr[bad] = float(default_far)
    return rr.astype(np.float32), aa.astype(np.float32)


def _lidar_triplet_from_scan(
    ranges: np.ndarray,
    angles_deg: np.ndarray,
    default_far: float,
) -> np.ndarray:
    rr, aa = _sanitize_lidar_scan(ranges=ranges, angles_deg=angles_deg, default_far=default_far)
    out = []
    for tgt in (30.0, 0.0, -30.0):
        d = np.abs(((aa - tgt + 180.0) % 360.0) - 180.0)
        idx = int(np.argmin(d))
        if float(d[idx]) > 80.0:
            out.append(float(default_far))
        else:
            out.append(float(rr[idx]))
    arr = np.asarray(out, dtype=np.float32)
    bad = (~np.isfinite(arr)) | (arr <= 0.0)
    arr[bad] = float(default_far)
    return arr


def _lidar_sector_min(
    ranges: np.ndarray,
    angles_deg: np.ndarray,
    default_far: float,
) -> tuple[float, float, float, float]:
    rr, aa = _sanitize_lidar_scan(ranges=ranges, angles_deg=angles_deg, default_far=default_far)
    ang = ((aa + 180.0) % 360.0) - 180.0
    # Side sectors use forward-side wedges only; rear-side beams are
    # excluded from near-collision side gating.
    front_mask = np.abs(ang) <= 25.0
    left_mask = (ang >= 25.0) & (ang <= 100.0)
    right_mask = (ang <= -25.0) & (ang >= -100.0)
    rear_mask = np.abs(ang) >= 120.0

    def _masked_min(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float(default_far)
        v = rr[mask]
        v = v[np.isfinite(v) & (v > 0.0)]
        if v.size <= 0:
            return float(default_far)
        return float(np.min(v))

    left = _masked_min(left_mask)
    front = _masked_min(front_mask)
    right = _masked_min(right_mask)
    rear = _masked_min(rear_mask)
    return left, front, right, rear


def add_supervisor_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Supervisor")
    g.add_argument("--sup-touch-threshold", type=float, default=1e-5)
    g.add_argument("--sup-near-collision-clearance", type=float, default=0.18)
    g.add_argument("--sup-side-collision-clearance", type=float, default=0.13)
    g.add_argument("--sup-emergency-clearance", type=float, default=0.08)
    g.add_argument("--sup-near-collision-persist-steps", type=int, default=3)
    g.add_argument("--sup-near-front-enter", type=float, default=0.14)
    g.add_argument("--sup-near-front-exit", type=float, default=0.22)
    g.add_argument("--sup-near-side-enter", type=float, default=0.12)
    g.add_argument("--sup-near-side-exit", type=float, default=0.18)
    g.add_argument("--sup-near-clear-steps", type=int, default=6)
    g.add_argument("--sup-near-reentry-guard-steps", type=int, default=24)
    g.add_argument("--sup-near-rearm-progress", type=float, default=0.08)
    g.add_argument("--sup-near-turn-wz-gate", type=float, default=0.35)

    g.add_argument("--sup-progress-window", type=int, default=80)
    g.add_argument("--sup-progress-min-delta", type=float, default=0.01)
    g.add_argument("--sup-progress-path-min-delta", type=float, default=0.03)
    g.add_argument("--sup-progress-stall-min-dist", type=float, default=1.0)
    g.add_argument("--sup-global-stuck-clearance", type=float, default=0.90)
    g.add_argument("--sup-progress-confirm-windows", type=int, default=5)
    g.add_argument("--sup-progress-min-u", type=float, default=2.5)
    g.add_argument("--sup-progress-max-v", type=float, default=0.10)
    g.add_argument("--sup-progress-max-displacement", type=float, default=0.12)
    g.add_argument("--sup-progress-blocked-ratio-min", type=float, default=0.70)
    g.add_argument("--sup-progress-front-clearance-gate", type=float, default=0.35)

    g.add_argument("--sup-jam-window", type=int, default=30)
    g.add_argument("--sup-jam-min-u", type=float, default=10.0)
    g.add_argument("--sup-jam-max-dxy", type=float, default=0.02)
    g.add_argument("--sup-jam-max-v", type=float, default=0.03)
    g.add_argument("--sup-jam-front-clearance-gate", type=float, default=0.25)
    g.add_argument("--sup-jam-blocked-ratio-min", type=float, default=0.80)

    g.add_argument("--sup-spin-window", type=int, default=40)
    g.add_argument("--sup-spin-min-progress", type=float, default=0.03)
    g.add_argument("--sup-spin-max-displacement", type=float, default=0.35)
    g.add_argument("--sup-spin-min-yaw-travel", type=float, default=1.2)
    g.add_argument("--sup-spin-confirm-windows", type=int, default=2)
    g.add_argument("--sup-spin-blocked-ratio-min", type=float, default=0.60)
    g.add_argument("--sup-spin-target-switch-max", type=int, default=2)
    g.add_argument("--sup-spin-min-wz", type=float, default=0.35)
    g.add_argument("--sup-spin-max-v", type=float, default=0.08)
    g.add_argument("--sup-spin-front-clearance-gate", type=float, default=0.40)

    g.add_argument("--sup-recover-stop-steps", type=int, default=10)
    g.add_argument("--sup-recover-backup-steps", type=int, default=14)
    g.add_argument("--sup-recover-backup-speed", type=float, default=0.20)
    g.add_argument("--sup-recover-rotate-steps", type=int, default=22)
    g.add_argument("--sup-recover-forward-steps", type=int, default=20)
    g.add_argument("--sup-recover-turn-rate", type=float, default=2.2)
    g.add_argument("--sup-recover-forward-speed", type=float, default=0.20)
    g.add_argument("--sup-recover-forward-turn-rate", type=float, default=0.55)
    g.add_argument("--sup-recover-forward-enter-front-clearance", type=float, default=0.34)
    g.add_argument("--sup-recover-forward-enter-min-clearance", type=float, default=0.24)
    g.add_argument("--sup-recover-forward-blocked-hold", action="store_true", default=True)
    g.add_argument("--no-sup-recover-forward-blocked-hold", action="store_false", dest="sup_recover_forward_blocked_hold")
    g.add_argument("--sup-recover-refractory-steps", type=int, default=20)
    g.add_argument("--sup-recover-min-rear-clearance", type=float, default=0.18)
    g.add_argument("--sup-recover-allow-reverse", action="store_true", default=False)

    g.add_argument("--sup-bounds-hard-margin", type=float, default=0.35)
    g.add_argument("--sup-boundary-trigger-steps", type=int, default=6)
    g.add_argument("--sup-boundary-release-steps", type=int, default=10)
    g.add_argument("--sup-boundary-release-margin", type=float, default=0.12)
    g.add_argument("--sup-boundary-soft-enter-dist", type=float, default=0.04)
    g.add_argument("--sup-boundary-hard-enter-dist", type=float, default=0.18)
    g.add_argument("--sup-boundary-soft-confirm-steps", type=int, default=4)
    g.add_argument("--sup-boundary-min-inward-speed", type=float, default=0.02)
    g.add_argument("--sup-boundary-release-progress-min", type=float, default=0.10)


def build_supervisor_config(args: argparse.Namespace) -> SupervisorConfig:
    return SupervisorConfig(
        touch_threshold=float(args.sup_touch_threshold),
        near_collision_clearance=float(args.sup_near_collision_clearance),
        side_collision_clearance=float(args.sup_side_collision_clearance),
        emergency_clearance=float(args.sup_emergency_clearance),
        near_collision_persist_steps=max(1, int(args.sup_near_collision_persist_steps)),
        near_front_enter=float(args.sup_near_front_enter),
        near_front_exit=float(args.sup_near_front_exit),
        near_side_enter=float(args.sup_near_side_enter),
        near_side_exit=float(args.sup_near_side_exit),
        near_clear_steps=max(1, int(args.sup_near_clear_steps)),
        near_reentry_guard_steps=max(0, int(args.sup_near_reentry_guard_steps)),
        near_rearm_progress=float(args.sup_near_rearm_progress),
        near_turn_wz_gate=float(args.sup_near_turn_wz_gate),
        progress_window=max(2, int(args.sup_progress_window)),
        progress_min_delta=float(args.sup_progress_min_delta),
        progress_path_min_delta=float(args.sup_progress_path_min_delta),
        progress_stall_min_dist=float(args.sup_progress_stall_min_dist),
        global_stuck_clearance=float(args.sup_global_stuck_clearance),
        progress_confirm_windows=max(1, int(args.sup_progress_confirm_windows)),
        progress_min_u=float(args.sup_progress_min_u),
        progress_max_v=float(args.sup_progress_max_v),
        progress_max_displacement=float(args.sup_progress_max_displacement),
        progress_blocked_ratio_min=float(args.sup_progress_blocked_ratio_min),
        progress_front_clearance_gate=float(args.sup_progress_front_clearance_gate),
        jam_contact_window=max(2, int(args.sup_jam_window)),
        jam_contact_min_u=float(args.sup_jam_min_u),
        jam_contact_max_dxy=float(args.sup_jam_max_dxy),
        jam_contact_max_v=float(args.sup_jam_max_v),
        spin_break_window=max(2, int(args.sup_spin_window)),
        spin_break_min_progress=float(args.sup_spin_min_progress),
        spin_break_max_displacement=float(args.sup_spin_max_displacement),
        spin_break_min_yaw_travel=float(args.sup_spin_min_yaw_travel),
        spin_confirm_windows=max(1, int(args.sup_spin_confirm_windows)),
        spin_blocked_ratio_min=float(args.sup_spin_blocked_ratio_min),
        spin_target_switch_max=max(0, int(args.sup_spin_target_switch_max)),
        spin_min_wz=float(args.sup_spin_min_wz),
        spin_max_v=float(args.sup_spin_max_v),
        spin_front_clearance_gate=float(args.sup_spin_front_clearance_gate),
        recover_stop_steps=max(1, int(args.sup_recover_stop_steps)),
        recover_backup_steps=max(0, int(args.sup_recover_backup_steps)),
        recover_rotate_steps=max(1, int(args.sup_recover_rotate_steps)),
        recover_forward_steps=max(1, int(args.sup_recover_forward_steps)),
        recover_backup_speed=float(args.sup_recover_backup_speed),
        recover_turn_rate=float(args.sup_recover_turn_rate),
        recover_forward_speed=float(args.sup_recover_forward_speed),
        recover_forward_turn_rate=float(args.sup_recover_forward_turn_rate),
        recover_refractory_steps=max(0, int(args.sup_recover_refractory_steps)),
        bounds_x_range=(float(args.bounds_x_range[0]), float(args.bounds_x_range[1])),
        bounds_y_range=(float(args.bounds_y_range[0]), float(args.bounds_y_range[1])),
        bounds_hard_margin=float(args.sup_bounds_hard_margin),
        boundary_guard_trigger_steps=max(1, int(args.sup_boundary_trigger_steps)),
        boundary_guard_release_steps=max(1, int(args.sup_boundary_release_steps)),
        boundary_guard_release_margin=float(args.sup_boundary_release_margin),
        boundary_soft_enter_dist=float(args.sup_boundary_soft_enter_dist),
        boundary_hard_enter_dist=float(args.sup_boundary_hard_enter_dist),
        boundary_soft_confirm_steps=max(1, int(args.sup_boundary_soft_confirm_steps)),
        boundary_min_inward_speed=float(args.sup_boundary_min_inward_speed),
        boundary_release_progress_min=float(args.sup_boundary_release_progress_min),
        jam_contact_front_clearance_gate=float(args.sup_jam_front_clearance_gate),
        jam_contact_blocked_ratio_min=float(args.sup_jam_blocked_ratio_min),
        recover_min_rear_clearance=float(args.sup_recover_min_rear_clearance),
        recover_allow_reverse=bool(args.sup_recover_allow_reverse),
        recover_forward_enter_front_clearance=float(args.sup_recover_forward_enter_front_clearance),
        recover_forward_enter_min_clearance=float(args.sup_recover_forward_enter_min_clearance),
        recover_forward_blocked_hold=bool(args.sup_recover_forward_blocked_hold),
    )


def build_nominal_controls_from_reference(
    state_now: np.ndarray,
    reference_traj_xy: np.ndarray | None,
    horizon: int,
    dt: float,
    wheel_radius: float,
    wheel_base: float,
    drive_sign: float,
    max_v: float,
    max_w: float,
    kp_linear: float,
    kp_angular: float,
    stop_dist: float,
    forward_only: bool,
    min_forward_v: float,
) -> np.ndarray | None:
    ref = np.asarray(reference_traj_xy, dtype=np.float32) if reference_traj_xy is not None else None
    if ref is None or ref.ndim != 2 or ref.shape[0] <= 0 or ref.shape[1] < 2:
        return None
    h = int(max(1, horizon))
    dt_ctrl = float(max(1e-4, dt))
    max_v_abs = float(max(0.0, max_v))
    max_w_abs = float(max(0.0, max_w))
    kp_v = float(max(0.0, kp_linear))
    kp_w = float(max(0.0, kp_angular))
    stop_r = float(max(0.0, stop_dist))
    min_v = float(max(0.0, min_forward_v))

    x = np.asarray(state_now, dtype=np.float32).reshape(-1)
    if x.shape[0] < 3:
        return None
    sim_x = float(x[0])
    sim_y = float(x[1])
    sim_yaw = float(x[2])

    def _wrap_to_pi(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    controls = np.zeros((h, 2), dtype=np.float32)
    for t in range(h):
        tgt = ref[min(t, ref.shape[0] - 1), :2]
        dx = float(tgt[0] - sim_x)
        dy = float(tgt[1] - sim_y)
        dist = float(np.hypot(dx, dy))
        des_yaw = float(np.arctan2(dy, dx)) if dist > 1e-9 else float(sim_yaw)
        yaw_err = _wrap_to_pi(des_yaw - sim_yaw)

        if dist <= stop_r:
            v_cmd = 0.0
            w_cmd = 0.0
        else:
            if bool(forward_only):
                heading_gate = float(max(0.0, np.cos(yaw_err)))
                v_cmd = float(np.clip(kp_v * dist * heading_gate, 0.0, max_v_abs))
                if dist > (1.5 * stop_r):
                    v_cmd = float(max(v_cmd, min_v))
            else:
                v_cmd = float(np.clip(kp_v * dist * np.cos(yaw_err), -max_v_abs, max_v_abs))
            w_cmd = float(np.clip(kp_w * yaw_err, -max_w_abs, max_w_abs))

        u = diff_drive_inverse(
            v=float(v_cmd),
            w=float(w_cmd),
            wheel_radius=float(wheel_radius),
            wheel_base=float(wheel_base),
            drive_sign=float(drive_sign),
        ).astype(np.float32)
        controls[t] = u

        vw = diff_drive_forward(
            u=u,
            wheel_radius=float(wheel_radius),
            wheel_base=float(wheel_base),
            drive_sign=float(drive_sign),
        )
        sim_x += float(vw[0]) * np.cos(sim_yaw) * dt_ctrl
        sim_y += float(vw[0]) * np.sin(sim_yaw) * dt_ctrl
        sim_yaw = _wrap_to_pi(sim_yaw + float(vw[1]) * dt_ctrl)

    return controls.astype(np.float32)


def run_episode(args: argparse.Namespace) -> dict:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.v_forward_sign = canonical_drive_sign(args.v_forward_sign)
    if args.kinematic_lock and args.planner_model != "kinematic":
        raise ValueError(
            f"kinematic_lock is enabled; planner-model must be 'kinematic', got: {args.planner_model}"
        )
    prediction_model = "kinematic_locked" if args.kinematic_lock else str(args.planner_model)
    print(
        f"[config] prediction_model={prediction_model}, "
        f"state_convention={STATE_CONVENTION_VERSION}, "
        f"drive_sign={args.v_forward_sign:+.0f}, depth_main_chain={bool(args.sensor_use_depth)}"
    )

    device = torch.device(args.device)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    kin_model = DiffDriveKinematicModel(
        state_dim=7,
        action_dim=2,
        dt=0.02,
        wheel_radius=args.wheel_radius,
        wheel_base=args.wheel_base,
        drive_sign=args.v_forward_sign,
    ).to(device)
    kin_model.eval()
    icode_model = None
    hybrid_fallback_model = None
    if args.planner_model == "kinematic":
        model = kin_model
    else:
        icode_model = load_icode_checkpoint(args.checkpoint, device)
        hybrid_fallback_model = HybridDynamics(
            icode_model=icode_model,
            kin_model=kin_model,
            alpha=float(args.icode_fallback_alpha),
        ).to(device)
        hybrid_fallback_model.eval()
        if args.planner_model == "hybrid" or args.icode_blend_alpha < 0.999:
            model = HybridDynamics(icode_model=icode_model, kin_model=kin_model, alpha=args.icode_blend_alpha).to(device)
            model.eval()
        else:
            model = icode_model
    model_context_dim = int(getattr(model, "context_dim", 0))
    env = E1RobotEnv(
        xml_path=str(args.xml),
        wheel_radius=args.wheel_radius,
        wheel_base=args.wheel_base,
        drive_sign=args.v_forward_sign,
        yaw_blend_alpha=args.yaw_blend_alpha,
        control_decimation=args.control_decimation,
        pose_source=args.pose_source,
        heading_source="base",
    )
    ctrl_dt = float(env.model.opt.timestep * env.control_decimation)

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
        cost_task_goal=args.cost_task_goal,
        cost_task_final=args.cost_task_final,
        cost_task_progress=args.cost_task_progress,
        cost_task_heading=args.cost_task_heading,
        cost_task_reverse=args.cost_task_reverse,
        cost_task_path_track=args.cost_task_path_track,
        cost_task_path_terminal=args.cost_task_path_terminal,
        cost_task_path_progress=args.cost_task_path_progress,
        cost_task_lateral=args.cost_task_lateral,
        cost_task_goal_visibility=args.cost_task_goal_visibility,
        cost_safe_collision=args.cost_safe_collision,
        cost_safe_near=args.cost_safe_near,
        cost_safe_corridor=args.cost_safe_corridor,
        cost_safe_pred_obs=args.cost_safe_pred_obs,
        cost_safe_bounds=args.cost_safe_bounds,
        cost_safe_bounds_terminal=args.cost_safe_bounds_terminal,
        collision_step_clearance=args.collision_step_clearance,
        collision_step_scale=args.collision_step_scale,
        collision_step_hit_scale=args.collision_step_hit_scale,
        cost_ctrl_effort=args.cost_ctrl_effort,
        cost_ctrl_smooth=args.cost_ctrl_smooth,
        cost_ctrl_spin=args.cost_ctrl_spin,
        cost_ctrl_wheel_diff=args.cost_ctrl_wheel_diff,
        cost_terminal_progress_deficit=args.cost_terminal_progress_deficit,
        cost_terminal_near_goal_stall=args.cost_terminal_near_goal_stall,
        cost_terminal_stop=args.cost_terminal_stop,
        cost_terminal_overshoot=args.cost_terminal_overshoot,
        obs_behind_scale=args.obs_behind_scale,
        robot_radius=args.robot_radius,
        obs_margin=args.obs_margin,
        near_penalty_clearance=args.near_penalty_clearance,
        near_penalty_mid_clearance=args.near_penalty_mid_clearance,
        near_penalty_hard_clearance=args.near_penalty_hard_clearance,
        near_penalty_mid_scale=args.near_penalty_mid_scale,
        near_penalty_hard_scale=args.near_penalty_hard_scale,
        near_penalty_hard_power=args.near_penalty_hard_power,
        pred_obs_clearance=args.pred_obs_clearance,
        pred_obs_hard_clearance=args.pred_obs_hard_clearance,
        pred_obs_hard_scale=args.pred_obs_hard_scale,
        pred_obs_clip_max=args.pred_obs_clip_max,
        near_progress_start=args.near_progress_start,
        near_progress_hard=args.near_progress_hard,
        near_progress_weight=args.near_progress_weight,
        near_stall_progress_eps=args.near_stall_progress_eps,
        near_stall_effort_gate=args.near_stall_effort_gate,
        path_corridor_half_width=args.path_corridor_half_width,
        near_goal_radius=args.near_goal_radius,
        near_goal_progress_eps=args.near_goal_progress_eps,
        los_margin=args.los_margin,
        terminal_stop_radius=args.terminal_stop_radius,
        overshoot_tolerance=args.overshoot_tolerance,
        noise_anneal_dist=args.noise_anneal_dist,
        noise_anneal_min_scale=args.noise_anneal_min_scale,
        enforce_forward_only=bool(args.nominal_forward_only),
        forward_min_speed=0.0,
        diff_wheel_radius=float(args.wheel_radius),
        diff_wheel_base=float(args.wheel_base),
        diff_drive_sign=float(args.v_forward_sign),
        world_x_min=min(args.bounds_x_range[0], args.bounds_x_range[1]),
        world_x_max=max(args.bounds_x_range[0], args.bounds_x_range[1]),
        world_y_min=min(args.bounds_y_range[0], args.bounds_y_range[1]),
        world_y_max=max(args.bounds_y_range[0], args.bounds_y_range[1]),
    )
    adaptive_base = {
        "cost_task_goal": float(mppi.cost_task_goal),
        "cost_safe_collision": float(mppi.cost_safe_collision),
        "cost_safe_near": float(mppi.cost_safe_near),
        "cost_safe_corridor": float(getattr(mppi, "cost_safe_corridor", 0.0)),
        "cost_safe_pred_obs": float(getattr(mppi, "cost_safe_pred_obs", 0.0)),
        "cost_task_path_track": float(mppi.cost_task_path_track),
        "cost_task_path_progress": float(mppi.cost_task_path_progress),
        "cost_ctrl_smooth": float(mppi.cost_ctrl_smooth),
        "cost_terminal_stop": float(mppi.cost_terminal_stop),
        "noise_sigma": mppi.noise_sigma.detach().clone(),
    }

    if args.init_x is not None and args.init_y is not None and args.init_yaw is not None:
        state = env.reset_with_pose(x=args.init_x, y=args.init_y, yaw=args.init_yaw)
    else:
        state = env.reset()
    base_xy0 = state[:2].astype(np.float32)
    yaw0 = float(state[2])
    target_xy = env.get_goal_xy().astype(np.float32)
    if args.target_x is not None and args.target_y is not None:
        target_xy = np.array([args.target_x, args.target_y], dtype=np.float32)
        env.set_goal_xy(float(target_xy[0]), float(target_xy[1]))
    elif args.target_forward_m > 0.0:
        target_xy = base_xy0 + float(args.target_forward_m) * np.array([np.cos(yaw0), np.sin(yaw0)], dtype=np.float32)
        env.set_goal_xy(float(target_xy[0]), float(target_xy[1]))

    obstacles_gt = env.get_obstacles_xyr()
    obstacles_nav = obstacles_gt.copy() if args.oracle_mode else np.zeros((0, 3), dtype=np.float32)
    random_scene_success = True
    scene_sampling_stage = 0
    scene_sampling_adaptive = False
    scene_constraints = {
        "min_obs_obs_dist": float(args.min_obs_obs_dist),
        "min_start_obs_dist": float(args.min_start_obs_dist),
        "min_goal_obs_dist": float(args.min_goal_obs_dist),
    }
    if args.random_obstacles:
        scene_seed = int(args.scene_seed) if args.scene_seed >= 0 else int(args.seed + 17)
        rng_scene = np.random.default_rng(scene_seed)
        if args.constrain_obs_radius:
            if args.randomize_obs_radius:
                rr = rng_scene.uniform(
                    float(args.obs_radius_min),
                    float(args.obs_radius_max),
                    size=(obstacles_gt.shape[0],),
                ).astype(np.float32)
            else:
                rr = np.clip(
                    obstacles_gt[:, 2],
                    float(args.obs_radius_min),
                    float(args.obs_radius_max),
                ).astype(np.float32)
            env.set_obstacles_radii(rr)
            obstacles_gt = env.get_obstacles_xyr()
        ok, sampled_obs, stage_idx = sample_obstacles_adaptive(
            rng=rng_scene,
            base_obstacles_xyr=obstacles_gt,
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
        scene_sampling_adaptive = True
        if ok:
            env.set_obstacles_xy(sampled_obs[:, :2])
            obstacles_gt = env.get_obstacles_xyr()

    if args.oracle_mode:
        obstacles_nav = obstacles_gt.copy()

    a_line = base_xy0.astype(np.float32)
    b_line = target_xy.astype(np.float32)
    line_blockers = 0
    tmin = float(min(args.blocker_t_range[0], args.blocker_t_range[1]))
    tmax = float(max(args.blocker_t_range[0], args.blocker_t_range[1]))
    for i in range(obstacles_gt.shape[0]):
        dseg, tseg = point_segment_distance_and_t(a_line, b_line, obstacles_gt[i, :2])
        r_eff = float(obstacles_gt[i, 2] + args.robot_radius + args.scene_path_inflate_margin + args.blocker_extra_margin)
        if tmin <= tseg <= tmax and dseg < r_eff:
            line_blockers += 1

    def world_to_body(vec_xy: np.ndarray, yaw: float) -> np.ndarray:
        return sc_world_to_body(np.asarray(vec_xy, dtype=np.float32), float(yaw))

    def diff_drive_to_wheels(v_cmd: float, w_cmd: float) -> np.ndarray:
        return diff_drive_inverse(
            v=float(v_cmd),
            w=float(w_cmd),
            wheel_radius=float(args.wheel_radius),
            wheel_base=float(args.wheel_base),
            drive_sign=float(args.v_forward_sign),
        ).astype(np.float32)

    def project_forward_action(action_u: np.ndarray, min_v: float) -> tuple[np.ndarray, bool]:
        u = np.asarray(action_u, dtype=np.float32).reshape(2)
        vw = diff_drive_forward(
            u=u,
            wheel_radius=float(args.wheel_radius),
            wheel_base=float(args.wheel_base),
            drive_sign=float(args.v_forward_sign),
        )
        v_cmd = float(max(float(vw[0]), float(min_v)))
        # Forward-only semantics must constrain yaw-rate as well; otherwise
        # reverse proposals can be projected into in-place spins.
        w_lim = float(2.0 * max(v_cmd, 0.0) / max(float(args.wheel_base), 1e-6))
        w_cmd = float(np.clip(float(vw[1]), -w_lim, w_lim))
        u_proj = diff_drive_to_wheels(v_cmd=v_cmd, w_cmd=w_cmd)
        projected = bool(np.linalg.norm(u_proj - u) > 1e-6)
        return u_proj.astype(np.float32), projected

    def project_nominal_forward_action(action_u: np.ndarray, terminal_mode: bool = False) -> tuple[np.ndarray, bool]:
        if not bool(args.nominal_forward_only):
            return np.asarray(action_u, dtype=np.float32), False
        if terminal_mode:
            min_v = float(max(float(args.terminal_min_forward_speed), 0.0))
        else:
            min_v = float(max(float(args.nominal_min_forward_speed), float(args.nav_min_forward_speed), 0.0))
        return project_forward_action(action_u=action_u, min_v=min_v)

    def project_delta_action(action_u: np.ndarray, prev_u: np.ndarray, max_delta_u: float) -> tuple[np.ndarray, bool]:
        u = np.asarray(action_u, dtype=np.float32).reshape(2)
        prev = np.asarray(prev_u, dtype=np.float32).reshape(2)
        md = float(max(0.0, max_delta_u))
        if md <= 0.0:
            return u, False
        du = u - prev
        if bool(args.soft_delta_projection):
            du_proj = md * np.tanh(du / max(md, 1e-6))
        else:
            du_proj = np.clip(du, -md, md)
        u_proj = prev + du_proj
        projected = bool(np.linalg.norm(u_proj - u) > 1e-6)
        return u_proj.astype(np.float32), projected

    def apply_semantic_speed_cap(
        action_u: np.ndarray,
        min_clearance: float,
        heading_err_abs: float,
        dist_goal: float,
        terminal_mode: bool,
    ) -> tuple[np.ndarray, bool, float]:
        u = np.asarray(action_u, dtype=np.float32).reshape(2)
        vw = diff_drive_forward(
            u=u,
            wheel_radius=float(args.wheel_radius),
            wheel_base=float(args.wheel_base),
            drive_sign=float(args.v_forward_sign),
        )
        v_cmd = float(vw[0])
        w_cmd = float(vw[1])

        if terminal_mode:
            v_min = float(max(0.0, args.terminal_min_forward_speed))
            v_base_max = float(max(v_min, args.terminal_speed_max))
        else:
            v_min = float(max(0.0, args.nav_min_forward_speed))
            v_base_max = float(max(v_min, args.nav_speed_max))

        c_hard = float(max(0.0, args.speed_cap_clearance_hard))
        c_soft = float(max(c_hard + 1e-3, args.speed_cap_clearance_soft))
        if np.isfinite(min_clearance):
            clearance_gain = float(np.clip((min_clearance - c_hard) / max(c_soft - c_hard, 1e-6), 0.0, 1.0))
        else:
            clearance_gain = 1.0

        h_gate = float(max(1e-3, args.speed_cap_heading_gate))
        heading_gain = float(np.clip(1.0 - heading_err_abs / h_gate, args.speed_cap_heading_min_gain, 1.0))
        v_max = v_min + (v_base_max - v_min) * (clearance_gain * heading_gain)

        if terminal_mode:
            enter = float(max(args.goal_tol + 1e-3, args.terminal_mppi_enter_radius))
            dist_gain = float(np.clip((dist_goal - args.goal_tol) / max(enter - args.goal_tol, 1e-6), 0.0, 1.0))
            v_max = v_min + (v_max - v_min) * dist_gain

        if (v_cmd >= (v_min - 1e-6)) and (v_cmd <= (v_max + 1e-6)):
            return u.astype(np.float32), False, float(v_max)

        v_out = float(np.clip(v_cmd, v_min, v_max))
        return diff_drive_to_wheels(v_cmd=v_out, w_cmd=w_cmd), True, float(v_max)

    def dynamic_target_corridor_max_dev(min_clearance: float, terminal_mode: bool) -> float:
        base_dev = float(max(0.05, args.path_corridor_half_width + args.corridor_target_slack))
        tight_dev = float(np.clip(args.corridor_tight_max_dev, 0.04, base_dev))
        trigger_clearance = float(max(0.05, args.corridor_tight_clearance))
        if terminal_mode:
            trigger_clearance = min(trigger_clearance, float(max(args.goal_tol + 0.05, args.terminal_mppi_enter_radius)))
        if (not np.isfinite(min_clearance)) or min_clearance >= trigger_clearance:
            return base_dev
        k = float(np.clip(min_clearance / max(trigger_clearance, 1e-6), 0.0, 1.0))
        return float(tight_dev + (base_dev - tight_dev) * k)

    def wrap_to_pi(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    def pick_turn_sign_from_nearest_obstacle(state_now: np.ndarray, base_xy_now: np.ndarray) -> float:
        if obstacles_nav.shape[0] <= 0:
            return 1.0
        yaw = float(state_now[2])
        nearest_i = int(np.argmin(np.linalg.norm(obstacles_nav[:, :2] - base_xy_now[None, :], axis=1)))
        rel_world = obstacles_nav[nearest_i, :2] - base_xy_now
        rel_body = world_to_body(rel_world, yaw)
        return -1.0 if rel_body[1] > 0.0 else 1.0

    def choose_turn_sign(
        default_sign: float,
        lidar_triplet: np.ndarray,
        lidar_ranges: np.ndarray | None = None,
        lidar_angles_deg: np.ndarray | None = None,
    ) -> float:
        s = float(default_sign)
        if not args.oracle_mode:
            used_full_scan = False
            if lidar_ranges is not None and lidar_angles_deg is not None:
                rr = np.asarray(lidar_ranges, dtype=np.float32).reshape(-1)
                aa = np.asarray(lidar_angles_deg, dtype=np.float32).reshape(-1)
                if rr.shape[0] > 0 and rr.shape[0] == aa.shape[0]:
                    left_s, _, right_s, _ = _lidar_sector_min(
                        ranges=rr,
                        angles_deg=aa,
                        default_far=float(args.sensor_default_far),
                    )
                    if np.isfinite(left_s) and np.isfinite(right_s) and abs(left_s - right_s) > 0.03:
                        s = 1.0 if left_s > right_s else -1.0
                        used_full_scan = True
            if not used_full_scan:
                lt = np.asarray(lidar_triplet, dtype=np.float32).reshape(-1)
                if lt.shape[0] >= 3:
                    left = float(lt[0])
                    right = float(lt[2])
                    if np.isfinite(left) and np.isfinite(right) and abs(left - right) > 0.03:
                        s = 1.0 if left > right else -1.0
        if s == 0.0:
            s = 1.0
        return s

    # NOTE: we intentionally keep MPPI as the only nominal navigation chain.
    # Docking is treated as terminal MPPI semantics (weights + speed caps),
    # rather than bypassing MPPI with an external controller.

    depth_renderer = None
    if args.sensor_use_depth and (not args.oracle_mode):
        depth_renderer = _try_create_depth_renderer(
            model=env.model,
            width=args.depth_width,
            height=args.depth_height,
        )
    if args.depth_required and (not args.sensor_use_depth):
        raise ValueError("--depth-required requires --sensor-use-depth.")
    if args.depth_required and (not args.oracle_mode) and args.sensor_use_depth and depth_renderer is None:
        raise RuntimeError("Depth is required but renderer initialization failed.")

    depth_ray_model = DepthRayModel(
        hfov_deg=float(args.sensor_depth_hfov_deg),
        sample_count=max(7, int(args.sensor_depth_max_points)),
        yaw_offset_deg=float(args.sensor_depth_yaw_offset_deg),
    )
    depth_gate = DepthConsistencyGate(
        window=int(args.depth_gate_window),
        min_valid_ratio=float(args.depth_gate_min_valid_ratio),
        max_sector_delta_mean=float(args.depth_gate_max_sector_delta),
        max_front_delta=float(args.depth_gate_max_front_delta),
        max_fail_ratio=float(args.depth_gate_max_fail_ratio),
    )
    occ_map = LocalOccupancyMap(
        x_range=(float(args.map_x_range[0]), float(args.map_x_range[1])),
        y_range=(float(args.map_y_range[0]), float(args.map_y_range[1])),
        resolution=float(args.map_resolution),
        hit_spread_cells=int(args.map_hit_spread_cells),
    )

    front_obs_memory: deque[np.ndarray] = deque(maxlen=max(1, int(args.front_guide_hold_steps)))
    depth_sector_hist: deque[np.ndarray] = deque(maxlen=max(1, int(args.depth_filter_window)))
    depth_front_ema = None
    depth_gate_reason = "disabled"
    depth_gate_drop_count = 0

    def update_map_obstacles(
        state_now: np.ndarray,
        lidar_ranges: np.ndarray,
        lidar_angles_deg: np.ndarray,
        lidar_triplet: np.ndarray,
        depth_image: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, bool]:
        nonlocal depth_gate_reason, depth_gate_drop_count
        lr, la = _sanitize_lidar_scan(
            ranges=np.asarray(lidar_ranges, dtype=np.float32),
            angles_deg=np.asarray(lidar_angles_deg, dtype=np.float32),
            default_far=float(args.sensor_default_far),
        )
        lt = np.asarray(lidar_triplet, dtype=np.float32).reshape(-1)
        if lt.shape[0] != 3:
            lt = _lidar_triplet_from_scan(ranges=lr, angles_deg=la, default_far=float(args.sensor_default_far))
        lidar_model = LidarRayModel(angles_deg=tuple(float(a) for a in la.tolist()))
        lidar_rays = lidar_model.build_rays(
            ranges=lr,
            default_far=float(args.sensor_default_far),
            min_range=float(args.sensor_obs_min_range),
            max_range=float(args.sensor_obs_max_range),
        )
        depth_rays = []
        use_depth = False
        if args.sensor_use_depth and (depth_image is not None):
            rep = depth_gate.update(
                depth_image=depth_image,
                lidar_triplet=lt,
                default_far=float(args.sensor_default_far),
                min_valid_depth=float(args.depth_min_valid),
            )
            use_depth = bool(rep.allow_depth)
            depth_gate_reason = str(rep.reason)
            if not use_depth:
                depth_gate_drop_count += 1
            if use_depth:
                depth_rays = depth_ray_model.build_rays(
                    depth_image=depth_image,
                    min_range=float(args.sensor_obs_min_range),
                    max_range=float(args.sensor_obs_max_range),
                )
        else:
            depth_gate_reason = "depth_disabled"
        rays = fuse_rays_to_hits_free(lidar_rays=lidar_rays, depth_rays=depth_rays, prefer_nearer=True)
        occ_map.update_from_rays(
            base_xy=state_now[:2],
            base_yaw=float(state_now[2]),
            rays=rays,
            min_range=float(args.sensor_obs_min_range),
            max_range=float(args.sensor_obs_max_range),
        )
        obstacles_map = occ_map.extract_obstacles_as_circles(
            threshold=float(args.map_occ_threshold),
            min_cluster_cells=int(args.map_min_cluster_cells),
            max_obstacles=int(args.map_max_obstacles),
        )
        if bool(args.sensor_use_ray_obstacles):
            obstacles_ray = _rays_to_world_obstacles(
                base_xy=state_now[:2],
                base_yaw=float(state_now[2]),
                rays=rays,
                min_range=float(args.sensor_obs_min_range),
                max_range=float(args.sensor_obs_max_range),
                base_radius=float(args.sensor_obs_radius),
                footprint_scale=float(args.sensor_ray_footprint_scale),
                radius_max_scale=float(args.sensor_ray_radius_max_scale),
            )
            obstacles = _fuse_obstacle_sets(
                obs_a=obstacles_map,
                obs_b=obstacles_ray,
                cell_size=float(args.sensor_obs_radius),
                max_points=int(args.map_max_obstacles),
            )
        else:
            obstacles = obstacles_map
        occ_grid = occ_map.occupancy(threshold=float(args.map_occ_threshold))
        occ_observed = occ_map.observed_mask(threshold=float(args.map_observed_threshold))
        occ_meta = occ_map.as_meta()
        return obstacles, occ_grid, occ_observed, occ_meta.min_xy, float(occ_meta.resolution), bool(use_depth)

    def select_front_obstacles(obstacles_xyr: np.ndarray, state_now: np.ndarray) -> np.ndarray:
        obs = np.asarray(obstacles_xyr, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        base_xy_now = state_now[:2]
        yaw_now = float(state_now[2])
        rel_world = obs[:, :2] - base_xy_now[None, :]
        rel_body = np.stack([world_to_body(v, yaw_now) for v in rel_world], axis=0)
        fov = np.deg2rad(float(max(5.0, args.front_guide_fov_deg)))
        ang = np.abs(np.arctan2(rel_body[:, 1], np.maximum(rel_body[:, 0], 1e-6)))
        m = (rel_body[:, 0] >= float(args.front_guide_min_x)) & (ang <= fov)
        front = obs[m]
        if front.shape[0] <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        return front.astype(np.float32)

    def update_front_obstacle_memory(obstacles_all: np.ndarray, state_now: np.ndarray) -> np.ndarray:
        if bool(args.guide_use_all_obstacles):
            obs_all = np.asarray(obstacles_all, dtype=np.float32)
            cur = obs_all if (obs_all.ndim == 2) else np.zeros((0, 3), dtype=np.float32)
        else:
            cur = select_front_obstacles(obstacles_xyr=obstacles_all, state_now=state_now)
        front_obs_memory.append(cur)
        merged = _merge_obstacle_frames(
            frames=front_obs_memory,
            cell_size=float(args.sensor_obs_radius),
            max_points=int(args.front_guide_max_points),
        )
        if merged.shape[0] <= 0 and args.front_guide_fallback_all:
            obs = np.asarray(obstacles_all, dtype=np.float32)
            return obs if obs.ndim == 2 else np.zeros((0, 3), dtype=np.float32)
        return merged

    def build_guide_obstacles(obstacles_all: np.ndarray, state_now: np.ndarray) -> np.ndarray:
        obs = update_front_obstacle_memory(obstacles_all=obstacles_all, state_now=state_now)
        out = np.asarray(obs, dtype=np.float32)
        if out.ndim != 2 or out.shape[0] <= 0:
            return np.zeros((0, 3), dtype=np.float32)
        if not args.oracle_mode:
            out = out.copy()
            scale = float(max(0.1, args.guide_obs_radius_scale))
            out[:, 2] = np.maximum(float(args.guide_obs_radius_min), out[:, 2] * scale)
        return out

    def filter_depth_features(
        obs_now: dict,
        depth_valid: bool,
        lidar_ranges: np.ndarray,
        lidar_angles_deg: np.ndarray,
    ) -> float:
        nonlocal depth_front_ema
        left_l, front_l, right_l, _ = _lidar_sector_min(
            ranges=lidar_ranges,
            angles_deg=lidar_angles_deg,
            default_far=float(args.sensor_default_far),
        )
        lidar_sector = np.array([left_l, front_l, right_l], dtype=np.float32)
        if not depth_valid:
            sec_fused = lidar_sector.astype(np.float32)
            obs_now["depth_sector_min"] = sec_fused
            obs_now["front_clearance"] = np.array([float(sec_fused[1])], dtype=np.float32)
            min_clear = float(np.min(sec_fused))
            obs_now["min_clearance"] = np.array([min_clear], dtype=np.float32)
            return min_clear

        sec = np.asarray(obs_now["depth_sector_min"], dtype=np.float32).reshape(-1)
        if sec.shape[0] != 3:
            sec = lidar_sector.copy()
        min_valid_depth = float(max(args.depth_min_valid, 0.5 * args.sensor_obs_min_range))
        invalid = (~np.isfinite(sec)) | (sec < min_valid_depth)
        sec_depth = sec.copy()
        sec_depth[invalid] = np.inf
        sec_fused = np.minimum(sec_depth, lidar_sector)
        bad_fused = (~np.isfinite(sec_fused)) | (sec_fused <= 0.0)
        sec_fused[bad_fused] = lidar_sector[bad_fused]

        depth_sector_hist.append(sec_fused.copy())
        sec_med = np.median(np.stack(list(depth_sector_hist), axis=0), axis=0).astype(np.float32)
        front_raw = float(sec_med[1])
        if depth_front_ema is None:
            depth_front_ema = front_raw
        else:
            a = float(np.clip(args.depth_front_ema_alpha, 0.0, 0.999))
            depth_front_ema = a * float(depth_front_ema) + (1.0 - a) * front_raw
        sec_med[1] = float(depth_front_ema)
        obs_now["depth_sector_min"] = sec_med
        obs_now["front_clearance"] = np.array([float(depth_front_ema)], dtype=np.float32)
        min_clear = float(np.min(sec_med))
        obs_now["min_clearance"] = np.array([min_clear], dtype=np.float32)
        return min_clear

    def render_depth_required() -> np.ndarray | None:
        depth = _render_depth_image(
            renderer=depth_renderer,
            data=env.data,
            camera_name=args.depth_camera,
        )
        if args.depth_required and depth is None and (not args.oracle_mode) and args.sensor_use_depth:
            raise RuntimeError("Depth is required but depth frame render failed.")
        return depth

    state_est_hist = [state.copy()]
    base_xy_hist = [base_xy0.copy()]
    action_hist = []
    planner_action_hist = []
    executed_action_hist = []
    action_post_delta_norm_hist = []
    action_post_delta_mppi_hist = []
    dist_hist = [float(np.linalg.norm(base_xy0 - target_xy))]
    step_time_hist = []
    min_clearance_hist = []
    touch_force_hist = []
    collision_count = 0
    recover_events = 0
    recover_collision_events = 0
    recover_stall_events = 0
    progress_gate_events = 0
    supervisor = SafetySupervisor(build_supervisor_config(args))
    supervisor_prev_state = SupervisorState.NORMAL.value
    supervisor_override_steps = 0
    trigger_reason_steps: dict[str, int] = {}
    mppi_action_steps = 0
    terminal_mppi_steps = 0
    speed_cap_applied_steps = 0
    delta_projection_applied_steps = 0
    action_bound_clip_steps = 0
    dock_action_steps = 0
    action_source_hist: list[str] = []
    trigger_reason_hist: list[str] = []
    supervisor_state_hist: list[str] = []
    boundary_mode_hist: list[str] = []
    boundary_signed_dist_hist: list[float] = []
    boundary_inward_speed_hist: list[float] = []
    active_target_step_hist: list[np.ndarray] = []
    cost_group_step_hist = {"task": [], "safety": [], "pred_obs": [], "control": [], "terminal": [], "total": []}

    path_remain_hist = []
    goal_progress_hist: deque[float] = deque(maxlen=max(2, int(args.sup_progress_window)))
    action_delta_hist = []
    adaptive_mode = "OPEN"
    adaptive_mode_hold = 0
    adaptive_switches = 0
    adaptive_mode_steps = {"OPEN": 0, "TIGHT": 0, "STUCK": 0, "DOCK": 0}
    prev_action = np.zeros((2,), dtype=np.float32)
    boundary_recover_latch_steps = 0
    goal_xy = target_xy.copy()
    waypoint_xy = None
    waypoint_last_side = 0
    waypoint_forced_side = 0
    waypoint_replans = 0
    waypoint_active_steps = 0
    waypoint_infeasible_count = 0
    waypoint_stuck_events = 0
    waypoint_stuck_force_events = 0
    force_waypoint_events = 0
    force_waypoint_latch_steps = 0
    blocked_goal_dist_buffer: deque[float] = deque(maxlen=max(2, int(args.blocked_force_waypoint_window)))
    goal_dist_buffer = []
    waypoint_stuck_cooldown = 0
    goal_los_clear_count = 0
    waypoint_hold_steps = 0
    goal_blocked_steps = 0
    goal_blocked_state = False
    goal_blocked_enter_count = 0
    goal_blocked_exit_count = 0
    goal_blocked_conf_hist: list[float] = []
    target_bound_projected_steps = 0
    target_corridor_projected_steps = 0
    target_jump_projected_steps = 0
    target_passability_projected_steps = 0
    boundary_soft_steps = 0
    guide_path_xy = None
    guide_progress_idx = 0
    guide_replans = 0
    guide_active_steps = 0
    guide_fail_steps = 0
    guide_fail_cause_counts = {"search_fail": 0, "passability_reject": 0, "other": 0}
    guide_replan_fail_streak = 0
    guide_replan_cooldown = 0
    guide_last_fail_pose = None
    guide_last_fail_yaw = 0.0
    guide_commit_side = 0
    guide_commit_until_step = -1
    guide_commit_dist_ref = 0.0
    guide_commit_buf = []
    guide_commit_active_steps = 0
    guide_commit_flip_events = 0
    goal_direct_steps = 0
    icode_fallback_triggered = False
    diverge_dist_hist: deque[float] = deque(maxlen=max(2, int(args.icode_diverge_window)))
    nav_mode = "NAV"
    dock_hold_count = 0
    target_active_hist = [goal_xy.copy()]
    last_active_target_for_supervisor = goal_xy.copy()
    target_switched_recent = False
    cost_group_hist = {"task": [], "safety": [], "pred_obs": [], "control": [], "terminal": [], "total": []}
    nominal_reverse_suppressed_steps = 0
    dock_reverse_suppressed_steps = 0
    mppi_reverse_raw_steps = 0
    # Razor principle: keep a single traversability radius/inflation semantics end-to-end.
    guide_robot_radius = float(args.robot_radius)
    guide_inflate_margin = float(args.scene_path_inflate_margin)
    global_guide_source = str(args.global_guide_source).strip().lower()
    use_gt_global_guide = bool(global_guide_source == "gt")
    guide_los_margin = float(args.goal_los_margin if args.oracle_mode else args.sensor_goal_los_margin)
    waypoint_margin_eff = float(args.waypoint_margin if args.oracle_mode else args.sensor_waypoint_margin)

    start = time.time()
    for step in range(args.max_steps):
        t0 = time.time()
        boundary_just_released = False
        if boundary_recover_latch_steps > 0:
            boundary_recover_latch_steps -= 1
        if guide_replan_cooldown > 0:
            guide_replan_cooldown -= 1
        recover_reentry_blocked = supervisor.cooldown_left > 0
        sensor_lidar_scan_now = np.array([args.sensor_default_far, args.sensor_default_far, args.sensor_default_far], dtype=np.float32)
        sensor_lidar_angles_now = np.array([30.0, 0.0, -30.0], dtype=np.float32)
        sensor_lidar_triplet_now = np.array([args.sensor_default_far, args.sensor_default_far, args.sensor_default_far], dtype=np.float32)
        sector_now = sensor_lidar_triplet_now.copy()
        rear_clearance_now = float(args.sensor_default_far)
        occ_grid_nav = None
        occ_observed_nav = None
        occ_min_xy_nav = None
        occ_res_nav = None
        if args.oracle_mode:
            base_xy_now = env.get_base_xy_gt()
            dist_goal_now = float(np.linalg.norm(base_xy_now - goal_xy))
            touch_now = env.get_touch_force()
            obstacles_nav = obstacles_gt.copy()
            obstacles_guide = build_guide_obstacles(obstacles_all=obstacles_nav, state_now=state)
            pre_clearances = np.linalg.norm(obstacles_nav[:, :2] - base_xy_now[None, :], axis=1) - (
                obstacles_nav[:, 2] + args.robot_radius
            )
            pre_min_clearance = float(np.min(pre_clearances)) if pre_clearances.size > 0 else float(args.sensor_default_far)
            front_clearance_now = pre_min_clearance
            corridor_width_now = float(args.sensor_default_far)
            rear_clearance_now = pre_min_clearance
        else:
            depth_now = render_depth_required()
            obs_now = env.get_unified_observation(
                goal_xy_odom=goal_xy,
                depth_image=depth_now,
                default_far=float(args.sensor_default_far),
                collision_threshold=float(args.sensor_collision_threshold),
                depth_yaw_offset_deg=float(args.sensor_depth_yaw_offset_deg),
            )
            state = obs_now["state_est"].astype(np.float32)
            base_xy_now = state[:2].copy()
            dist_goal_now = float(obs_now["goal_dist"][0])
            touch_now = float(obs_now["touch_force"][0])
            sensor_lidar_scan_now = np.asarray(
                obs_now.get("lidar_scan", obs_now.get("lidar_triplet", sensor_lidar_scan_now)),
                dtype=np.float32,
            ).reshape(-1)
            sensor_lidar_angles_now = np.asarray(
                obs_now.get("lidar_angles_deg", sensor_lidar_angles_now),
                dtype=np.float32,
            ).reshape(-1)
            sensor_lidar_scan_now, sensor_lidar_angles_now = _sanitize_lidar_scan(
                ranges=sensor_lidar_scan_now,
                angles_deg=sensor_lidar_angles_now,
                default_far=float(args.sensor_default_far),
            )
            sensor_lidar_triplet_now = np.asarray(
                obs_now.get(
                    "lidar_triplet",
                    _lidar_triplet_from_scan(
                        ranges=sensor_lidar_scan_now,
                        angles_deg=sensor_lidar_angles_now,
                        default_far=float(args.sensor_default_far),
                    ),
                ),
                dtype=np.float32,
            ).reshape(-1)
            if sensor_lidar_triplet_now.shape[0] != 3:
                sensor_lidar_triplet_now = _lidar_triplet_from_scan(
                    ranges=sensor_lidar_scan_now,
                    angles_deg=sensor_lidar_angles_now,
                    default_far=float(args.sensor_default_far),
                )
            obstacles_nav, occ_grid_nav, occ_observed_nav, occ_min_xy_nav, occ_res_nav, depth_used_now = update_map_obstacles(
                state_now=state,
                lidar_ranges=sensor_lidar_scan_now,
                lidar_angles_deg=sensor_lidar_angles_now,
                lidar_triplet=sensor_lidar_triplet_now,
                depth_image=depth_now,
            )
            pre_min_clearance = filter_depth_features(
                obs_now=obs_now,
                depth_valid=depth_used_now,
                lidar_ranges=sensor_lidar_scan_now,
                lidar_angles_deg=sensor_lidar_angles_now,
            )
            front_clearance_now = float(obs_now["front_clearance"][0])
            corridor_width_now = float(obs_now["corridor_width"][0])
            left_s, front_s, right_s, rear_s = _lidar_sector_min(
                ranges=sensor_lidar_scan_now,
                angles_deg=sensor_lidar_angles_now,
                default_far=float(args.sensor_default_far),
            )
            sector_now = np.asarray(
                obs_now.get("depth_sector_min", np.array([left_s, front_s, right_s], dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)
            if sector_now.shape[0] != 3:
                sector_now = np.array([left_s, front_s, right_s], dtype=np.float32)
            rear_clearance_now = float(rear_s)
            obstacles_guide = build_guide_obstacles(obstacles_all=obstacles_nav, state_now=state)
        goal_body_now = world_to_body(goal_xy - base_xy_now, float(state[2]))
        goal_progress_hist.append(float(dist_goal_now))
        diverge_dist_hist.append(float(dist_goal_now))
        if (
            args.icode_diverge_fallback
            and (not icode_fallback_triggered)
            and args.planner_model == "icode"
            and hybrid_fallback_model is not None
            and step >= int(args.icode_diverge_min_step)
            and len(diverge_dist_hist) >= max(2, int(args.icode_diverge_window))
        ):
            dist_increase = float(diverge_dist_hist[-1] - diverge_dist_hist[0])
            if dist_increase >= float(args.icode_diverge_min_increase):
                mppi.model = hybrid_fallback_model
                icode_fallback_triggered = True
        side_clearance = float(np.min(sector_now[[0, 2]])) if sector_now.shape[0] >= 3 else float(pre_min_clearance)

        _, goal_blocked_conf, _ = blocked_confidence_range_semantics(
            start_xy=base_xy_now,
            goal_xy=goal_xy,
            heading_xy=(base_xy_now + np.array([float(np.cos(float(state[2]))), float(np.sin(float(state[2])))], dtype=np.float32)),
            obstacles_xyr=obstacles_guide,
            robot_radius=guide_robot_radius,
            margin=guide_los_margin,
            occ_grid=occ_grid_nav if (not args.oracle_mode) else None,
            occ_observed=occ_observed_nav if (not args.oracle_mode) else None,
            occ_min_xy=occ_min_xy_nav if (not args.oracle_mode) else None,
            occ_resolution=occ_res_nav if (not args.oracle_mode) else None,
            corridor_len=float(args.goal_blocked_corridor_len),
            corridor_half_width=float(args.goal_blocked_corridor_half_width),
            front_fov_deg=float(args.goal_blocked_front_fov_deg),
            front_range=float(args.goal_blocked_front_range),
            blocked_conf_threshold=float(args.goal_blocked_enter_conf),
        )
        goal_blocked_conf_hist.append(float(goal_blocked_conf))
        if goal_blocked_state:
            if float(goal_blocked_conf) <= float(args.goal_blocked_exit_conf):
                goal_blocked_exit_count += 1
                if goal_blocked_exit_count >= max(1, int(args.goal_blocked_exit_steps)):
                    goal_blocked_state = False
                    goal_blocked_enter_count = 0
                    goal_blocked_exit_count = 0
            else:
                goal_blocked_exit_count = 0
        else:
            if float(goal_blocked_conf) >= float(args.goal_blocked_enter_conf):
                goal_blocked_enter_count += 1
                if goal_blocked_enter_count >= max(1, int(args.goal_blocked_enter_steps)):
                    goal_blocked_state = True
                    goal_blocked_enter_count = 0
                    goal_blocked_exit_count = 0
            else:
                goal_blocked_enter_count = 0
        goal_blocked = bool(goal_blocked_state)
        if goal_blocked:
            goal_blocked_steps += 1
            goal_los_clear_count = 0
        else:
            goal_los_clear_count += 1
        blocked_goal_dist_buffer.append(float(dist_goal_now))
        if force_waypoint_latch_steps > 0:
            force_waypoint_latch_steps -= 1
        if (
            args.blocked_force_waypoint_enable
            and args.auto_waypoint
            and force_waypoint_latch_steps <= 0
            and supervisor.state == SupervisorState.NORMAL
            and nav_mode == "NAV"
            and len(blocked_goal_dist_buffer) >= max(2, int(args.blocked_force_waypoint_window))
        ):
            blocked_progress = float(blocked_goal_dist_buffer[0] - blocked_goal_dist_buffer[-1])
            if (
                blocked_progress < float(args.blocked_force_waypoint_min_progress)
                and pre_min_clearance < float(args.blocked_force_waypoint_clearance)
                and (goal_blocked or recover_reentry_blocked)
            ):
                force_waypoint_latch_steps = max(1, int(args.blocked_force_waypoint_latch_steps))
                force_waypoint_events += 1
                forced_sign = choose_turn_sign(
                    pick_turn_sign_from_nearest_obstacle(state_now=state, base_xy_now=base_xy_now),
                    sensor_lidar_triplet_now,
                    lidar_ranges=sensor_lidar_scan_now,
                    lidar_angles_deg=sensor_lidar_angles_now,
                )
                waypoint_forced_side = int(np.sign(forced_sign)) if forced_sign != 0.0 else 1
                waypoint_xy = None
                waypoint_hold_steps = 0
                guide_path_xy = None
                guide_progress_idx = 0
                guide_commit_side = 0
                guide_commit_buf = []

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

        turn_sign_hint = choose_turn_sign(
            pick_turn_sign_from_nearest_obstacle(state_now=state, base_xy_now=base_xy_now),
            sensor_lidar_triplet_now,
            lidar_ranges=sensor_lidar_scan_now,
            lidar_angles_deg=sensor_lidar_angles_now,
        )
        dgoal_recent = float("nan")
        if len(goal_progress_hist) >= 2:
            dgoal_recent = float(goal_progress_hist[0] - goal_progress_hist[-1])
        supervisor_prev_state = supervisor.state.value
        if len(base_xy_hist) > 0:
            measured_speed_now = float(
                np.linalg.norm(base_xy_now - np.asarray(base_xy_hist[-1], dtype=np.float32))
                / max(float(ctrl_dt), 1e-6)
            )
        else:
            measured_speed_now = 0.0
        supervisor_decision = supervisor.step(
            SupervisorInput(
                step=int(step),
                state=np.asarray(state, dtype=np.float32),
                base_xy=np.asarray(base_xy_now, dtype=np.float32),
                goal_xy=np.asarray(goal_xy, dtype=np.float32),
                dist_goal=float(dist_goal_now),
                touch_force=float(touch_now),
                min_clearance=float(pre_min_clearance),
                side_clearance=float(side_clearance),
                front_clearance=float(front_clearance_now),
                goal_blocked=bool(goal_blocked),
                prev_action=np.asarray(prev_action, dtype=np.float32),
                turn_sign_hint=float(turn_sign_hint),
                path_progress_recent=float(dpath_recent) if dpath_recent is not None else float("nan"),
                target_switched_recent=bool(target_switched_recent),
                goal_progress_recent=float(dgoal_recent),
                rear_clearance=float(rear_clearance_now),
                measured_speed=float(measured_speed_now),
            )
        )
        trigger_reason = str(supervisor_decision.trigger_reason)
        trigger_reason_steps[trigger_reason] = trigger_reason_steps.get(trigger_reason, 0) + 1
        if supervisor_decision.action_source == "SUPERVISOR":
            supervisor_override_steps += 1
        if supervisor_prev_state == SupervisorState.BOUNDARY_GUARD.value and supervisor_decision.state == SupervisorState.NORMAL.value:
            boundary_just_released = True
            boundary_recover_latch_steps = max(
                boundary_recover_latch_steps,
                max(0, int(args.boundary_recover_guide_steps)),
            )
        if supervisor_decision.transitioned:
            if trigger_reason == "near_collision":
                recover_events += 1
                recover_collision_events += 1
            elif trigger_reason in ("jam_contact", "progress_stall", "spin_stall"):
                recover_events += 1
                recover_stall_events += 1
                if trigger_reason == "progress_stall":
                    progress_gate_events += 1
            if (
                args.auto_waypoint
                and trigger_reason in ("near_collision", "jam_contact", "progress_stall", "spin_stall")
            ):
                force_waypoint_latch_steps = max(
                    force_waypoint_latch_steps,
                    max(1, int(args.blocked_force_waypoint_latch_steps)),
                )
                if waypoint_forced_side == 0:
                    waypoint_forced_side = int(np.sign(turn_sign_hint)) if turn_sign_hint != 0.0 else 1
                waypoint_xy = None
                waypoint_hold_steps = 0
                guide_path_xy = None
                guide_progress_idx = 0
                guide_commit_side = 0
                guide_commit_buf = []

        goal_heading_err_now = float(np.arctan2(goal_body_now[1], max(goal_body_now[0], 1e-6)))
        aligned_enter = (
            abs(goal_heading_err_now) <= float(args.dock_align_yaw_enter)
            and abs(float(goal_body_now[1])) <= float(args.dock_align_lateral_enter)
        )
        if nav_mode == "NAV" and dist_goal_now <= args.dock_brake_radius:
            if aligned_enter:
                nav_mode = "BRAKE_ALIGN"
                dock_hold_count = 0
        elif nav_mode == "BRAKE_ALIGN":
            if (
                dist_goal_now > args.dock_exit_radius
                or abs(goal_heading_err_now) >= float(args.dock_align_yaw_exit)
                or abs(float(goal_body_now[1])) >= float(args.dock_align_lateral_exit)
            ):
                nav_mode = "NAV"

        terminal_gate = float(max(args.goal_tol * 1.05, args.terminal_mppi_enter_radius))
        terminal_mppi_mode = bool((nav_mode == "BRAKE_ALIGN") or (dist_goal_now <= terminal_gate))

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
                    if args.auto_waypoint and supervisor.state == SupervisorState.NORMAL and nav_mode == "NAV":
                        prev_latch = int(force_waypoint_latch_steps)
                        force_waypoint_latch_steps = max(
                            force_waypoint_latch_steps,
                            max(1, int(args.waypoint_stuck_force_latch_steps)),
                        )
                        if int(force_waypoint_latch_steps) > prev_latch:
                            waypoint_stuck_force_events += 1
                        guide_path_xy = None
                        guide_progress_idx = 0
                        guide_commit_side = 0
                        guide_commit_buf = []
                    waypoint_stuck_cooldown = args.waypoint_stuck_cooldown

        guide_failed_this_step = False
        current_nav_target = np.asarray(goal_xy, dtype=np.float32)
        if supervisor_decision.action_source == "SUPERVISOR" and supervisor_decision.override_vw is not None:
            target_active_hist.append(goal_xy.copy())
            action = diff_drive_to_wheels(
                v_cmd=float(supervisor_decision.override_vw[0]),
                w_cmd=float(supervisor_decision.override_vw[1]),
            )
            action_source = "SUPERVISOR"
        else:
            target_for_mppi = goal_xy
            reference_traj = None
            nominal_u_seq = None
            used_global_guide = False
            goal_direct_dist_eff = float(args.goal_direct_dist)
            if not args.oracle_mode:
                goal_direct_dist_eff = min(goal_direct_dist_eff, float(args.sensor_goal_direct_dist))
            goal_direct_allowed = (
                (args.oracle_mode or (not args.sensor_strict_path))
                and abs(goal_heading_err_now) <= float(args.goal_direct_yaw_max)
                and abs(float(goal_body_now[1])) <= float(args.goal_direct_lateral_max)
            )
            goal_direct_mode = (
                bool(args.goal_direct_on_clear)
                and goal_direct_allowed
                and goal_los_clear_count >= max(1, int(args.goal_direct_clear_steps))
                and dist_goal_now <= goal_direct_dist_eff
            )
            if boundary_recover_latch_steps > 0:
                goal_direct_mode = False
            if goal_direct_mode:
                goal_direct_steps += 1
                guide_commit_side = 0
                guide_commit_buf = []
                guide_commit_until_step = -1
                path_remain_hist = []
                used_global_guide = True
            elif args.global_guide:
                need_guide_replan = (
                    (step % max(1, args.guide_replan_interval) == 0)
                    or (guide_path_xy is None)
                    or boundary_just_released
                    or (boundary_recover_latch_steps > 0)
                )
                if guide_path_xy is not None and guide_path_xy.shape[0] > 0:
                    dist_to_path = float(np.min(np.linalg.norm(guide_path_xy - base_xy_now[None, :], axis=1)))
                    if args.corridor_commit and guide_commit_side != 0 and step <= guide_commit_until_step:
                        if dist_to_path <= float(args.commit_max_path_dev):
                            need_guide_replan = False
                        else:
                            need_guide_replan = True
                    elif dist_to_path > max(0.1, float(args.waypoint_switch_radius)):
                        need_guide_replan = True
                can_retry_fail = False
                if guide_last_fail_pose is not None:
                    moved = float(np.linalg.norm(base_xy_now - np.asarray(guide_last_fail_pose, dtype=np.float32)))
                    yaw_delta = float(abs(((float(state[2]) - float(guide_last_fail_yaw) + np.pi) % (2.0 * np.pi)) - np.pi))
                    can_retry_fail = (
                        moved >= float(args.guide_fail_retry_min_move)
                        or yaw_delta >= float(args.guide_fail_retry_min_yaw)
                    )
                replan_gate = bool(need_guide_replan)
                if (
                    replan_gate
                    and guide_replan_cooldown > 0
                    and (not boundary_just_released)
                    and boundary_recover_latch_steps <= 0
                    and (not can_retry_fail)
                ):
                    replan_gate = False
                if replan_gate:
                    guide_replans += 1
                    plan_debug = {}
                    guide_obstacles_for_planner = (
                        obstacles_gt.astype(np.float32)
                        if use_gt_global_guide
                        else obstacles_guide.astype(np.float32)
                    )
                    guide_occ_grid = (
                        occ_grid_nav
                        if ((not args.oracle_mode) and (not use_gt_global_guide))
                        else None
                    )
                    guide_occ_min_xy = (
                        occ_min_xy_nav
                        if ((not args.oracle_mode) and (not use_gt_global_guide))
                        else None
                    )
                    guide_occ_res = (
                        occ_res_nav
                        if ((not args.oracle_mode) and (not use_gt_global_guide))
                        else None
                    )
                    guide_path_candidate = plan_global_path_xy(
                        start_xy=base_xy_now.astype(np.float32),
                        goal_xy=goal_xy.astype(np.float32),
                        obstacles_xyr=guide_obstacles_for_planner,
                        robot_radius=guide_robot_radius,
                        inflation_margin=guide_inflate_margin,
                        grid_resolution=float(args.scene_path_grid_res),
                        grid_padding=float(args.scene_path_grid_padding),
                        max_grid_cells=int(args.guide_max_grid_cells),
                        occ_grid=guide_occ_grid,
                        occ_min_xy=guide_occ_min_xy,
                        occ_resolution=guide_occ_res,
                        planner=str(args.global_planner),
                        start_yaw=float(state[2]),
                        passability_check=bool(args.planner_passability_check),
                        passability_margin=float(args.planner_passability_margin),
                        passability_min_clearance=float(args.planner_passability_min_clearance),
                        hybrid_n_theta=int(args.hybrid_n_theta),
                        hybrid_step_cells=float(args.hybrid_step_cells),
                        hybrid_turn_penalty=float(args.hybrid_turn_penalty),
                        hybrid_max_expansions=int(args.hybrid_max_expansions),
                        debug_info=plan_debug,
                    )
                    fail_cause = str(plan_debug.get("fail_cause", "other"))
                    if guide_path_candidate is not None and guide_path_candidate.shape[0] >= 2:
                        guide_path_xy = guide_path_candidate
                        guide_progress_idx = 0
                        guide_replan_fail_streak = 0
                        guide_replan_cooldown = 0
                        guide_last_fail_pose = None
                        guide_last_fail_yaw = float(state[2])
                    else:
                        if fail_cause not in guide_fail_cause_counts:
                            fail_cause = "other"
                        guide_fail_cause_counts[fail_cause] += 1
                        guide_replan_fail_streak += 1
                        fail_backoff = int(
                            max(1, int(args.guide_fail_latch_steps))
                            * min(max(1, int(args.guide_fail_backoff_mult_cap)), guide_replan_fail_streak)
                        )
                        guide_replan_cooldown = int(
                            min(max(1, int(args.guide_fail_latch_max_steps)), fail_backoff)
                        )
                        guide_last_fail_pose = base_xy_now.astype(np.float32).copy()
                        guide_last_fail_yaw = float(state[2])
                if guide_path_xy is not None and guide_path_xy.shape[0] >= 2:
                    path_remain_now, guide_idx_rem = compute_path_remaining(
                        path_xy=guide_path_xy,
                        current_xy=base_xy_now.astype(np.float32),
                        min_index=0,
                    )
                    target_for_mppi, guide_idx = select_path_lookahead_target(
                        path_xy=guide_path_xy,
                        current_xy=base_xy_now.astype(np.float32),
                        lookahead_m=float(args.guide_lookahead),
                        min_index=0,
                    )
                    guide_progress_idx = int(guide_idx)
                    path_remain_hist.append(float(path_remain_now))
                    if len(path_remain_hist) > max(2, int(args.sup_progress_window)):
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
                    guide_failed_this_step = True

            planner_clear_req = float(max(0.0, args.planner_passability_min_clearance))
            planner_margin_req = float(max(0.0, args.planner_passability_margin))
            target_guard_clear_req = float(max(0.0, args.target_guard_min_clearance, planner_clear_req))
            target_guard_margin_req = float(max(0.0, args.target_guard_margin, planner_margin_req))
            if (not used_global_guide) and args.auto_waypoint:
                wp_seg_clear_req = float(max(0.0, args.waypoint_infeasible_min_clearance, planner_clear_req))
                wp_point_clear_req = float(max(0.0, 0.5 * wp_seg_clear_req))
                if waypoint_xy is not None:
                    if float(np.linalg.norm(base_xy_now - waypoint_xy)) <= args.waypoint_switch_radius:
                        waypoint_xy = None
                        waypoint_hold_steps = 0
                        waypoint_infeasible_count = 0
                    else:
                        waypoint_hold_steps += 1
                        wp_ok = waypoint_is_feasible(
                            base_xy=base_xy_now,
                            waypoint_xy=waypoint_xy,
                            goal_xy=goal_xy,
                            obstacles_xyr=obstacles_guide,
                            robot_radius=guide_robot_radius,
                            margin=waypoint_margin_eff,
                            min_seg_clearance=wp_seg_clear_req,
                            min_point_clearance=wp_point_clear_req,
                            goal_min_dist=float(args.waypoint_goal_min_dist),
                        )
                        if not wp_ok:
                            waypoint_infeasible_count += 1
                            if waypoint_infeasible_count >= max(1, int(args.waypoint_infeasible_confirm_steps)):
                                waypoint_xy = None
                                waypoint_hold_steps = 0
                                waypoint_infeasible_count = 0
                                if waypoint_forced_side == 0:
                                    side_hint = choose_turn_sign(
                                        pick_turn_sign_from_nearest_obstacle(state_now=state, base_xy_now=base_xy_now),
                                        sensor_lidar_triplet_now,
                                        lidar_ranges=sensor_lidar_scan_now,
                                        lidar_angles_deg=sensor_lidar_angles_now,
                                    )
                                    waypoint_forced_side = int(np.sign(side_hint)) if side_hint != 0.0 else 1
                                if args.auto_waypoint and supervisor.state == SupervisorState.NORMAL and nav_mode == "NAV":
                                    prev_latch = int(force_waypoint_latch_steps)
                                    force_waypoint_latch_steps = max(
                                        force_waypoint_latch_steps,
                                        max(1, int(args.waypoint_infeasible_force_latch_steps)),
                                    )
                                    if int(force_waypoint_latch_steps) > prev_latch:
                                        waypoint_stuck_force_events += 1
                                    guide_path_xy = None
                                    guide_progress_idx = 0
                                    guide_commit_side = 0
                                    guide_commit_buf = []
                        else:
                            waypoint_infeasible_count = 0
                            if (
                                args.waypoint_use_los_gating
                                and (not goal_blocked)
                                and goal_los_clear_count >= args.waypoint_clear_hysteresis_steps
                                and waypoint_hold_steps >= args.waypoint_min_hold_steps
                            ):
                                waypoint_xy = None
                                waypoint_hold_steps = 0
                                waypoint_infeasible_count = 0
                need_replan = bool(waypoint_xy is None)
                if args.waypoint_replan_on_interval:
                    need_replan = need_replan or (step % max(1, args.waypoint_replan_interval) == 0)
                should_replan = True
                allow_forced_waypoint = bool(waypoint_forced_side != 0)
                if (
                    args.waypoint_use_los_gating
                    and (not goal_blocked)
                    and (waypoint_xy is None)
                    and (not allow_forced_waypoint)
                    and (not guide_failed_this_step)
                ):
                    should_replan = False
                if need_replan and should_replan:
                    force_level = 0
                    if allow_forced_waypoint:
                        force_level = int(
                            np.clip(
                                waypoint_stuck_events,
                                0,
                                max(0, int(args.waypoint_force_max_level)),
                            )
                        )
                    lat_boost = float(max(0.0, args.waypoint_force_lateral_boost)) * float(force_level)
                    fwd_boost = float(max(0.0, args.waypoint_force_forward_boost)) * float(force_level)
                    waypoint_lateral_extra_eff = float(max(0.0, args.waypoint_lateral_extra + lat_boost))
                    waypoint_max_lateral_eff = float(max(0.0, args.waypoint_max_lateral + lat_boost))
                    waypoint_max_forward_eff = float(max(args.waypoint_min_forward, args.waypoint_max_forward + fwd_boost))
                    wp, side = compute_auto_waypoint_with_side(
                        start_xy=base_xy_now,
                        goal_xy=goal_xy,
                        obstacles_xyr=obstacles_guide,
                        robot_radius=guide_robot_radius,
                        margin=waypoint_margin_eff,
                        lateral_extra=waypoint_lateral_extra_eff,
                        preferred_side=waypoint_forced_side,
                        waypoint_max_lateral=waypoint_max_lateral_eff,
                        waypoint_min_forward=float(args.waypoint_min_forward),
                        waypoint_max_forward=waypoint_max_forward_eff,
                        waypoint_goal_min_dist=float(args.waypoint_goal_min_dist),
                    )
                    if wp is None:
                        side_pref = int(np.sign(waypoint_forced_side))
                        if side_pref == 0:
                            side_pref = int(
                                np.sign(
                                    choose_turn_sign(
                                        pick_turn_sign_from_nearest_obstacle(state_now=state, base_xy_now=base_xy_now),
                                        sensor_lidar_triplet_now,
                                        lidar_ranges=sensor_lidar_scan_now,
                                        lidar_angles_deg=sensor_lidar_angles_now,
                                    )
                                )
                            )
                        if side_pref == 0:
                            side_pref = 1
                        goal_vec_tmp = goal_xy - base_xy_now
                        goal_norm_tmp = float(np.linalg.norm(goal_vec_tmp))
                        if goal_norm_tmp > 1e-6:
                            gdir = goal_vec_tmp / goal_norm_tmp
                        else:
                            yaw_tmp = float(state[2])
                            gdir = np.array([np.cos(yaw_tmp), np.sin(yaw_tmp)], dtype=np.float32)
                        gperp = np.array([-gdir[1], gdir[0]], dtype=np.float32)
                        fwd_len = float(np.clip(0.8 * float(args.guide_lookahead) + fwd_boost, 0.45, 1.60))
                        lat_len = float(np.clip(float(args.waypoint_lateral_extra) + lat_boost, 0.20, 1.40))
                        wp = base_xy_now + fwd_len * gdir + float(side_pref) * lat_len * gperp
                        # Keep waypoint separated from goal so escape logic is preserved.
                        if float(np.linalg.norm(wp - goal_xy)) < float(args.waypoint_goal_min_dist):
                            away = base_xy_now - goal_xy
                            away_n = float(np.linalg.norm(away))
                            if away_n < 1e-6:
                                away = -gdir
                                away_n = float(np.linalg.norm(away))
                            away = away / max(away_n, 1e-6)
                            wp = goal_xy + away * float(args.waypoint_goal_min_dist) + float(side_pref) * 0.35 * lat_len * gperp
                        side = int(side_pref)
                    if wp is not None:
                        wp_ok = waypoint_is_feasible(
                            base_xy=base_xy_now,
                            waypoint_xy=wp,
                            goal_xy=goal_xy,
                            obstacles_xyr=obstacles_guide,
                            robot_radius=guide_robot_radius,
                            margin=waypoint_margin_eff,
                            min_seg_clearance=wp_seg_clear_req,
                            min_point_clearance=wp_point_clear_req,
                            goal_min_dist=float(args.waypoint_goal_min_dist),
                        )
                        if not wp_ok:
                            wp = None
                    if wp is not None:
                        waypoint_replans += 1
                        waypoint_xy = wp
                        waypoint_last_side = int(side)
                        waypoint_forced_side = 0
                        waypoint_hold_steps = 0
                        waypoint_infeasible_count = 0
                if waypoint_xy is not None:
                    target_for_mppi = waypoint_xy
                    waypoint_active_steps += 1

            if str(supervisor_decision.boundary_mode) == "SOFT":
                boundary_soft_steps += 1
                target_for_mppi = compute_boundary_recover_target(
                    base_xy=base_xy_now.astype(np.float32),
                    goal_xy=goal_xy.astype(np.float32),
                    bounds_x_range=(float(args.bounds_x_range[0]), float(args.bounds_x_range[1])),
                    bounds_y_range=(float(args.bounds_y_range[0]), float(args.bounds_y_range[1])),
                    bound_margin=float(args.target_bound_margin),
                    path_xy=guide_path_xy if (guide_path_xy is not None and guide_path_xy.shape[0] >= 2) else None,
                    lookahead_m=float(max(0.2, args.guide_lookahead)),
                )
                used_global_guide = True

            ref_path_for_target = None
            if guide_path_xy is not None and np.asarray(guide_path_xy).ndim == 2 and guide_path_xy.shape[0] >= 2:
                ref_path_for_target = np.asarray(guide_path_xy, dtype=np.float32)
            corridor_max_dev_now = dynamic_target_corridor_max_dev(
                min_clearance=float(pre_min_clearance),
                terminal_mode=bool(terminal_mppi_mode),
            )
            jump_max_now = float(args.target_jump_max)
            if np.isfinite(pre_min_clearance):
                c0 = float(max(0.05, args.speed_cap_clearance_hard))
                c1 = float(max(c0 + 1e-3, args.corridor_tight_clearance))
                tight_gain = float(np.clip((pre_min_clearance - c0) / max(c1 - c0, 1e-6), 0.0, 1.0))
                jump_max_now = float(args.target_jump_min + (args.target_jump_max - args.target_jump_min) * tight_gain)
            target_for_mppi, target_proj_info = project_target_with_invariants(
                candidate_xy=np.asarray(target_for_mppi, dtype=np.float32),
                prev_target_xy=np.asarray(last_active_target_for_supervisor, dtype=np.float32),
                start_xy=base_xy0.astype(np.float32),
                goal_xy=goal_xy.astype(np.float32),
                bounds_x_range=(float(args.bounds_x_range[0]), float(args.bounds_x_range[1])),
                bounds_y_range=(float(args.bounds_y_range[0]), float(args.bounds_y_range[1])),
                bound_margin=float(args.target_bound_margin),
                corridor_max_dev=float(corridor_max_dev_now),
                jump_max=float(jump_max_now),
                path_xy=ref_path_for_target,
            )
            if float(target_proj_info.get("bound_projected", 0.0)) > 0.5:
                target_bound_projected_steps += 1
            if float(target_proj_info.get("corridor_projected", 0.0)) > 0.5:
                target_corridor_projected_steps += 1
            if float(target_proj_info.get("jump_projected", 0.0)) > 0.5:
                target_jump_projected_steps += 1
            fallback_passability_required = bool((not used_global_guide) and args.auto_waypoint)
            if bool(args.target_passability_guard) or fallback_passability_required:
                target_for_mppi, target_pass_info = project_target_to_passable_point(
                    candidate_xy=np.asarray(target_for_mppi, dtype=np.float32),
                    base_xy=base_xy_now.astype(np.float32),
                    obstacles_xyr=obstacles_nav,
                    robot_radius=float(args.robot_radius),
                    margin=float(target_guard_margin_req),
                    min_seg_clearance=float(target_guard_clear_req),
                    path_xy=ref_path_for_target,
                    min_progress=float(max(0.0, args.target_guard_min_progress)),
                    backtrack_points=int(max(1, args.target_guard_backtrack_points)),
                    ray_samples=int(max(2, args.target_guard_ray_samples)),
                )
                if float(target_pass_info.get("passability_projected", 0.0)) > 0.5:
                    target_passability_projected_steps += 1

            target_active_hist.append(np.asarray(target_for_mppi, dtype=np.float32))
            current_nav_target = np.asarray(target_for_mppi, dtype=np.float32)
            if not used_global_guide:
                path_remain_hist = []
            if terminal_mppi_mode:
                _apply_adaptive_profile(
                    mppi=mppi,
                    base=adaptive_base,
                    profile=_adaptive_profile("DOCK"),
                    alpha=float(args.terminal_profile_alpha),
                )
            if bool(args.reference_sampling):
                nominal_u_seq = build_nominal_controls_from_reference(
                    state_now=np.asarray(state, dtype=np.float32),
                    reference_traj_xy=reference_traj,
                    horizon=int(args.horizon),
                    dt=float(ctrl_dt),
                    wheel_radius=float(args.wheel_radius),
                    wheel_base=float(args.wheel_base),
                    drive_sign=float(args.v_forward_sign),
                    max_v=float(args.reference_nominal_v_max),
                    max_w=float(args.reference_nominal_w_max),
                    kp_linear=float(args.reference_tracker_kp_v),
                    kp_angular=float(args.reference_tracker_kp_w),
                    stop_dist=float(args.reference_tracker_stop_dist),
                    forward_only=bool(args.reference_tracker_forward_only),
                    min_forward_v=float(args.reference_tracker_min_v),
                )
            if model_context_dim > 0:
                ctx_now = _build_icode_sensor_ctx(
                    goal_body_now=np.asarray(goal_body_now, dtype=np.float32),
                    dist_goal_now=float(dist_goal_now),
                    sector_now=np.asarray(sector_now, dtype=np.float32),
                    corridor_width_now=float(corridor_width_now),
                    min_clearance_now=float(pre_min_clearance),
                    touch_now=float(touch_now),
                    sensor_collision_threshold=float(args.sensor_collision_threshold),
                    touch_threshold=float(args.sup_touch_threshold),
                )
                mppi.set_model_context(_align_ctx_dim(ctx_now, target_dim=model_context_dim))
            else:
                mppi.set_model_context(None)
            action = mppi.get_action(
                initial_state=state,
                target=target_for_mppi,
                obstacles=obstacles_nav,
                reference_traj=reference_traj,
                nominal_u_seq=nominal_u_seq,
            )
            action_source = "MPPI"

        if action_source == "MPPI":
            mppi_action_steps += 1
            if terminal_mppi_mode:
                terminal_mppi_steps += 1
            for k in cost_group_hist:
                cost_group_hist[k].append(float(mppi.last_cost_terms.get(k, 0.0)))

        step_cost_terms = {k: float("nan") for k in cost_group_step_hist}
        if action_source == "MPPI":
            for k in step_cost_terms:
                step_cost_terms[k] = float(mppi.last_cost_terms.get(k, 0.0))
        for k, v in step_cost_terms.items():
            cost_group_step_hist[k].append(float(v))
        action_source_hist.append(str(action_source))
        trigger_reason_hist.append(str(trigger_reason))
        supervisor_state_hist.append(str(supervisor_decision.state))
        boundary_mode_hist.append(str(supervisor_decision.boundary_mode))
        boundary_signed_dist_hist.append(float(supervisor_decision.boundary_signed_dist))
        boundary_inward_speed_hist.append(float(supervisor_decision.boundary_inward_speed))
        active_target_step_hist.append(np.asarray(current_nav_target, dtype=np.float32).copy())
        target_switched_recent = bool(
            np.linalg.norm(np.asarray(current_nav_target, dtype=np.float32) - np.asarray(last_active_target_for_supervisor, dtype=np.float32))
            > 1e-4
        )
        last_active_target_for_supervisor = np.asarray(current_nav_target, dtype=np.float32).copy()

        planner_action_raw = np.asarray(action, dtype=np.float32).copy()
        if action_source == "MPPI":
            vw_raw = diff_drive_forward(
                u=planner_action_raw,
                wheel_radius=float(args.wheel_radius),
                wheel_base=float(args.wheel_base),
                drive_sign=float(args.v_forward_sign),
            )
            if float(vw_raw[0]) < -1e-5:
                mppi_reverse_raw_steps += 1
            if args.max_delta_u > 0.0:
                action, delta_projected = project_delta_action(
                    action_u=action,
                    prev_u=prev_action,
                    max_delta_u=float(args.max_delta_u),
                )
                if delta_projected:
                    delta_projection_applied_steps += 1
            action, projected = project_nominal_forward_action(action_u=action, terminal_mode=terminal_mppi_mode)
            if projected:
                nominal_reverse_suppressed_steps += 1
            action, speed_capped, _ = apply_semantic_speed_cap(
                action_u=action,
                min_clearance=float(pre_min_clearance),
                heading_err_abs=float(abs(goal_heading_err_now)),
                dist_goal=float(dist_goal_now),
                terminal_mode=terminal_mppi_mode,
            )
            if speed_capped:
                speed_cap_applied_steps += 1

        action_pre_bound = np.asarray(action, dtype=np.float32).copy()
        action = np.clip(action_pre_bound, env.ctrl_low, env.ctrl_high).astype(np.float32)
        if float(np.linalg.norm(action - action_pre_bound)) > 1e-6:
            action_bound_clip_steps += 1
        post_delta_norm = float(np.linalg.norm(action - planner_action_raw))
        action_post_delta_norm_hist.append(post_delta_norm)
        if action_source == "MPPI":
            action_post_delta_mppi_hist.append(post_delta_norm)
        planner_action_hist.append(planner_action_raw)
        executed_action_hist.append(action.copy())
        du_now = float(np.linalg.norm(action - prev_action))
        action_delta_hist.append(du_now)
        if len(action_delta_hist) > max(1, int(args.adaptive_osc_window)):
            action_delta_hist.pop(0)
        prev_action = action.copy()
        state = env.step(action)
        dt_step = time.time() - t0

        if args.oracle_mode:
            base_xy = env.get_base_xy_gt()
            touch_force = env.get_touch_force()
            dist = float(np.linalg.norm(base_xy - goal_xy))
            obs_clearances = np.linalg.norm(obstacles_gt[:, :2] - base_xy[None, :], axis=1) - (
                obstacles_gt[:, 2] + args.robot_radius
            )
            min_clearance = float(np.min(obs_clearances)) if obs_clearances.size > 0 else float(args.sensor_default_far)
        else:
            depth_after = render_depth_required()
            obs_after = env.get_unified_observation(
                goal_xy_odom=goal_xy,
                depth_image=depth_after,
                default_far=float(args.sensor_default_far),
                collision_threshold=float(args.sensor_collision_threshold),
                depth_yaw_offset_deg=float(args.sensor_depth_yaw_offset_deg),
            )
            state = obs_after["state_est"].astype(np.float32)
            base_xy = state[:2].copy()
            touch_force = float(obs_after["touch_force"][0])
            dist = float(obs_after["goal_dist"][0])
            lidar_scan_after = np.asarray(
                obs_after.get("lidar_scan", obs_after.get("lidar_triplet", sensor_lidar_scan_now)),
                dtype=np.float32,
            ).reshape(-1)
            lidar_ang_after = np.asarray(
                obs_after.get("lidar_angles_deg", sensor_lidar_angles_now),
                dtype=np.float32,
            ).reshape(-1)
            lidar_scan_after, lidar_ang_after = _sanitize_lidar_scan(
                ranges=lidar_scan_after,
                angles_deg=lidar_ang_after,
                default_far=float(args.sensor_default_far),
            )
            lidar_triplet_after = np.asarray(
                obs_after.get(
                    "lidar_triplet",
                    _lidar_triplet_from_scan(
                        ranges=lidar_scan_after,
                        angles_deg=lidar_ang_after,
                        default_far=float(args.sensor_default_far),
                    ),
                ),
                dtype=np.float32,
            ).reshape(-1)
            obstacles_nav, _, _, _, _, depth_used_after = update_map_obstacles(
                state_now=state,
                lidar_ranges=lidar_scan_after,
                lidar_angles_deg=lidar_ang_after,
                lidar_triplet=lidar_triplet_after,
                depth_image=depth_after,
            )
            min_clearance = filter_depth_features(
                obs_now=obs_after,
                depth_valid=depth_used_after,
                lidar_ranges=lidar_scan_after,
                lidar_angles_deg=lidar_ang_after,
            )
            obstacles_guide = build_guide_obstacles(obstacles_all=obstacles_nav, state_now=state)

        if min_clearance < 0.0 or touch_force > args.sup_touch_threshold:
            collision_count += 1

        action_hist.append(action.astype(np.float32))
        state_est_hist.append(state.copy())
        base_xy_hist.append(base_xy.copy())
        dist_hist.append(dist)
        step_time_hist.append(dt_step)
        min_clearance_hist.append(min_clearance)
        touch_force_hist.append(touch_force)

        if dist < args.goal_tol:
            break

    elapsed = time.time() - start

    state_est_arr = np.asarray(state_est_hist, dtype=np.float32)
    base_xy_arr = np.asarray(base_xy_hist, dtype=np.float32)
    action_arr = np.asarray(action_hist, dtype=np.float32) if action_hist else np.zeros((0, 2), dtype=np.float32)
    planner_action_arr = np.asarray(planner_action_hist, dtype=np.float32) if planner_action_hist else np.zeros((0, 2), dtype=np.float32)
    executed_action_arr = np.asarray(executed_action_hist, dtype=np.float32) if executed_action_hist else np.zeros((0, 2), dtype=np.float32)
    action_post_delta_arr = (
        np.asarray(action_post_delta_norm_hist, dtype=np.float32) if action_post_delta_norm_hist else np.zeros((0,), dtype=np.float32)
    )
    action_post_delta_mppi_arr = (
        np.asarray(action_post_delta_mppi_hist, dtype=np.float32) if action_post_delta_mppi_hist else np.zeros((0,), dtype=np.float32)
    )
    dist_arr = np.asarray(dist_hist, dtype=np.float32)
    step_time_arr = np.asarray(step_time_hist, dtype=np.float32) if step_time_hist else np.zeros((0,), dtype=np.float32)
    min_clearance_arr = np.asarray(min_clearance_hist, dtype=np.float32) if min_clearance_hist else np.zeros((0,), dtype=np.float32)
    touch_force_arr = np.asarray(touch_force_hist, dtype=np.float32) if touch_force_hist else np.zeros((0,), dtype=np.float32)
    net_disp = float(np.linalg.norm(base_xy_arr[-1] - base_xy_arr[0]))
    supervisor_metrics = supervisor.summary()
    sup_state_steps = supervisor_metrics.get("state_steps", {})
    sup_trigger_counts = supervisor_metrics.get("trigger_counts", {})
    a_layer_preempt = int(sup_trigger_counts.get("boundary_guard", 0)) + int(sup_trigger_counts.get("near_collision", 0))
    b_layer_triggers = (
        int(sup_trigger_counts.get("jam_contact", 0))
        + int(sup_trigger_counts.get("progress_stall", 0))
        + int(sup_trigger_counts.get("spin_stall", 0))
    )
    post_delta_eps = float(max(0.0, args.action_post_delta_eps))
    action_post_delta_ratio = float(np.mean(action_post_delta_arr > post_delta_eps)) if action_post_delta_arr.size > 0 else 0.0
    action_post_delta_ratio_mppi = (
        float(np.mean(action_post_delta_mppi_arr > post_delta_eps)) if action_post_delta_mppi_arr.size > 0 else 0.0
    )
    cost_group_mean = {
        k: (float(np.mean(v)) if len(v) > 0 else 0.0)
        for k, v in cost_group_hist.items()
    }
    action_source_arr = np.asarray(action_source_hist, dtype="<U16") if action_source_hist else np.zeros((0,), dtype="<U1")
    trigger_reason_arr = np.asarray(trigger_reason_hist, dtype="<U32") if trigger_reason_hist else np.zeros((0,), dtype="<U1")
    supervisor_state_arr = np.asarray(supervisor_state_hist, dtype="<U32") if supervisor_state_hist else np.zeros((0,), dtype="<U1")
    boundary_mode_arr = np.asarray(boundary_mode_hist, dtype="<U32") if boundary_mode_hist else np.zeros((0,), dtype="<U1")
    boundary_signed_dist_arr = (
        np.asarray(boundary_signed_dist_hist, dtype=np.float32) if boundary_signed_dist_hist else np.zeros((0,), dtype=np.float32)
    )
    boundary_inward_speed_arr = (
        np.asarray(boundary_inward_speed_hist, dtype=np.float32) if boundary_inward_speed_hist else np.zeros((0,), dtype=np.float32)
    )
    goal_blocked_conf_arr = (
        np.asarray(goal_blocked_conf_hist, dtype=np.float32) if goal_blocked_conf_hist else np.zeros((0,), dtype=np.float32)
    )
    active_target_step_arr = (
        np.asarray(active_target_step_hist, dtype=np.float32)
        if active_target_step_hist
        else np.zeros((0, 2), dtype=np.float32)
    )
    cost_group_step_arr = {
        k: (np.asarray(v, dtype=np.float32) if len(v) > 0 else np.zeros((0,), dtype=np.float32))
        for k, v in cost_group_step_hist.items()
    }

    metrics = {
        "steps_executed": int(action_arr.shape[0]),
        "reached_goal": bool(dist_arr[-1] < args.goal_tol),
        "goal_tol": float(args.goal_tol),
        "final_dist_to_goal": float(dist_arr[-1]),
        "min_dist_to_goal": float(np.min(dist_arr)),
        "net_displacement": net_disp,
        "min_clearance": float(np.min(min_clearance_arr)) if min_clearance_arr.size > 0 else float("nan"),
        "collision_steps": int(collision_count),
        "recover_events": int(recover_events),
        "recover_collision_events": int(recover_collision_events),
        "recover_stall_events": int(recover_stall_events),
        "progress_gate_events": int(progress_gate_events),
        "boundary_guard_steps": int(sup_state_steps.get("BOUNDARY_GUARD", 0)),
        "boundary_guard_events": int(sup_trigger_counts.get("boundary_guard", 0)),
        "boundary_soft_steps": int(boundary_soft_steps),
        "boundary_soft_events": int(sup_trigger_counts.get("boundary_soft", 0)),
        "escape_events": int(recover_events),
        "supervisor_override_steps": int(supervisor_override_steps),
        "mppi_action_steps": int(mppi_action_steps),
        "terminal_mppi_steps": int(terminal_mppi_steps),
        "speed_cap_applied_steps": int(speed_cap_applied_steps),
        "delta_projection_applied_steps": int(delta_projection_applied_steps),
        "action_bound_clip_steps": int(action_bound_clip_steps),
        "dock_action_steps": int(dock_action_steps),
        "mppi_reverse_raw_steps": int(mppi_reverse_raw_steps),
        "nominal_reverse_suppressed_steps": int(nominal_reverse_suppressed_steps),
        "dock_reverse_suppressed_steps": int(dock_reverse_suppressed_steps),
        "trigger_reason_steps": {k: int(v) for k, v in trigger_reason_steps.items()},
        "a_layer_preempt_count": int(a_layer_preempt),
        "b_layer_trigger_count": int(b_layer_triggers),
        "supervisor_summary": supervisor_metrics,
        "max_touch_force": float(np.max(touch_force_arr)) if touch_force_arr.size > 0 else 0.0,
        "mean_step_time_s": float(np.mean(step_time_arr)) if step_time_arr.size > 0 else 0.0,
        "runtime_s": float(elapsed),
        "target_xy": target_xy.tolist(),
        "random_obstacles": bool(args.random_obstacles),
        "random_scene_success": bool(random_scene_success),
        "scene_sampling_adaptive": bool(scene_sampling_adaptive),
        "scene_sampling_stage": int(scene_sampling_stage),
        "scene_constraints": scene_constraints,
        "scene_min_line_blockers": int(args.min_line_blockers),
        "scene_require_mixed_sides": bool(args.require_mixed_sides),
        "scene_line_blockers_actual": int(line_blockers),
        "obstacle_radii": obstacles_gt[:, 2].astype(float).tolist(),
        "prediction_model": prediction_model,
        "pred_obs_model_enabled": bool(hasattr(model, "predict_obstacle_distance")),
        "cost_safe_pred_obs": float(args.cost_safe_pred_obs),
        "pred_obs_clearance": float(args.pred_obs_clearance),
        "pred_obs_hard_clearance": float(args.pred_obs_hard_clearance),
        "pred_obs_hard_scale": float(args.pred_obs_hard_scale),
        "pred_obs_clip_max": float(args.pred_obs_clip_max),
        "state_convention_version": STATE_CONVENTION_VERSION,
        "drive_sign": float(args.v_forward_sign),
        "oracle_mode": bool(args.oracle_mode),
        "sensor_use_depth": bool(args.sensor_use_depth),
        "depth_gate_drop_count": int(depth_gate_drop_count),
        "depth_gate_last_reason": str(depth_gate_reason),
        "depth_required": bool(args.depth_required),
        "map_resolution": float(args.map_resolution),
        "front_guide_fov_deg": float(args.front_guide_fov_deg),
        "front_guide_hold_steps": int(args.front_guide_hold_steps),
        "waypoint_enabled": bool(args.auto_waypoint),
        "waypoint_replans": int(waypoint_replans),
        "waypoint_active_steps": int(waypoint_active_steps),
        "waypoint_stuck_events": int(waypoint_stuck_events),
        "waypoint_stuck_force_events": int(waypoint_stuck_force_events),
        "force_waypoint_events": int(force_waypoint_events),
        "waypoint_last_side": int(waypoint_last_side),
        "waypoint_forced_side": int(waypoint_forced_side),
        "waypoint_last_xy": waypoint_xy.tolist() if waypoint_xy is not None else None,
        "global_guide_enabled": bool(args.global_guide),
        "global_guide_source": str(args.global_guide_source),
        "global_guide_replans": int(guide_replans),
        "global_guide_active_steps": int(guide_active_steps),
        "global_guide_fail_steps": int(guide_fail_steps),
        "global_guide_fail_cause_counts": {k: int(v) for k, v in guide_fail_cause_counts.items()},
        "global_guide_replan_cooldown_final": int(guide_replan_cooldown),
        "corridor_commit_enabled": bool(args.corridor_commit),
        "corridor_commit_active_steps": int(guide_commit_active_steps),
        "corridor_commit_flip_events": int(guide_commit_flip_events),
        "target_bound_projected_steps": int(target_bound_projected_steps),
        "target_corridor_projected_steps": int(target_corridor_projected_steps),
        "target_jump_projected_steps": int(target_jump_projected_steps),
        "target_passability_projected_steps": int(target_passability_projected_steps),
        "goal_direct_steps": int(goal_direct_steps),
        "goal_blocked_steps": int(goal_blocked_steps),
        "goal_blocked_ratio": float(goal_blocked_steps / max(1, action_arr.shape[0])),
        "goal_blocked_conf_mean": float(np.mean(goal_blocked_conf_arr)) if goal_blocked_conf_arr.size > 0 else 0.0,
        "goal_blocked_conf_p95": float(np.percentile(goal_blocked_conf_arr, 95.0)) if goal_blocked_conf_arr.size > 0 else 0.0,
        "final_nav_mode": str(nav_mode),
        "dock_hold_count": int(dock_hold_count),
        "adaptive_scheduler": bool(args.adaptive_scheduler),
        "adaptive_switches": int(adaptive_switches),
        "adaptive_mode_steps": {k: int(v) for k, v in adaptive_mode_steps.items()},
        "supervisor_enabled": True,
        "planner_action_raw_count": int(planner_action_arr.shape[0]),
        "executed_action_count": int(executed_action_arr.shape[0]),
        "action_post_delta_eps": post_delta_eps,
        "action_post_delta_norm_mean": float(np.mean(action_post_delta_arr)) if action_post_delta_arr.size > 0 else 0.0,
        "action_post_delta_norm_max": float(np.max(action_post_delta_arr)) if action_post_delta_arr.size > 0 else 0.0,
        "action_post_delta_ratio": action_post_delta_ratio,
        "action_post_delta_ratio_mppi": action_post_delta_ratio_mppi,
        "action_post_breakdown_steps": {
            "speed_cap": int(speed_cap_applied_steps),
            "delta_projection": int(delta_projection_applied_steps),
            "reverse_suppress": int(nominal_reverse_suppressed_steps + dock_reverse_suppressed_steps),
            "bound_clip": int(action_bound_clip_steps),
        },
        "cost_group_mean": cost_group_mean,
        "mppi_last_cost_terms": {k: float(v) for k, v in mppi.last_cost_terms.items()},
        "spin_stall_events": int(sup_trigger_counts.get("spin_stall", 0)),
        "jam_contact_events": int(sup_trigger_counts.get("jam_contact", 0)),
        "progress_stall_events": int(sup_trigger_counts.get("progress_stall", 0)),
        "recover_rotate_steps": int(sup_state_steps.get("ROTATE_RECOVER", 0)),
        "icode_fallback_triggered": bool(icode_fallback_triggered),
        "step_trace_saved": True,
        "step_trace_steps": int(len(action_source_hist)),
    }

    np.savez_compressed(
        out_dir / "rollout_log.npz",
        state_est=state_est_arr,
        base_xy_est=base_xy_arr,
        actions=action_arr,
        planner_action_raw=planner_action_arr,
        executed_action=executed_action_arr,
        action_post_delta_norm=action_post_delta_arr,
        dist_to_goal=dist_arr,
        min_clearance=min_clearance_arr,
        touch_force=touch_force_arr,
        obstacles_xyr=obstacles_gt.astype(np.float32),
        target_xy=target_xy.astype(np.float32),
        active_target_xy=np.asarray(target_active_hist, dtype=np.float32),
        active_target_step_xy=active_target_step_arr,
        action_source_step=action_source_arr,
        trigger_reason_step=trigger_reason_arr,
        supervisor_state_step=supervisor_state_arr,
        boundary_mode_step=boundary_mode_arr,
        boundary_signed_dist_step=boundary_signed_dist_arr,
        boundary_inward_speed_step=boundary_inward_speed_arr,
        goal_blocked_conf_step=goal_blocked_conf_arr,
        cost_task_step=cost_group_step_arr["task"],
        cost_safety_step=cost_group_step_arr["safety"],
        cost_pred_obs_step=cost_group_step_arr["pred_obs"],
        cost_control_step=cost_group_step_arr["control"],
        cost_terminal_step=cost_group_step_arr["terminal"],
        cost_total_step=cost_group_step_arr["total"],
        scene_sampling_stage=np.asarray([scene_sampling_stage], dtype=np.int32),
    )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(base_xy_arr[:, 0], base_xy_arr[:, 1], "-b", lw=1.8, label="base trajectory")
    ax1.scatter(base_xy_arr[0, 0], base_xy_arr[0, 1], c="green", s=40, label="start")
    ax1.scatter(base_xy_arr[-1, 0], base_xy_arr[-1, 1], c="blue", s=40, label="end")
    ax1.scatter(target_xy[0], target_xy[1], c="red", s=55, marker="*", label="goal")
    for i in range(obstacles_gt.shape[0]):
        circle = plt.Circle((obstacles_gt[i, 0], obstacles_gt[i, 1]), obstacles_gt[i, 2], color="orange", alpha=0.35)
        ax1.add_patch(circle)
    ax1.set_title("XY Trajectory")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.axis("equal")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(dist_arr, "-k", lw=1.8)
    ax2.axhline(args.goal_tol, color="red", ls="--", lw=1.0, label="goal tol")
    ax2.set_title("Distance To Goal")
    ax2.set_xlabel("control step")
    ax2.set_ylabel("distance [m]")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(1, 3, 3)
    if action_arr.shape[0] > 0:
        ax3.plot(action_arr[:, 0], label="u_left")
        ax3.plot(action_arr[:, 1], label="u_right")
    ax3.set_title("Control Inputs")
    ax3.set_xlabel("control step")
    ax3.set_ylabel("ctrl")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_controls.png", dpi=160)
    plt.close(fig)

    return metrics


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MPPI with trained ICODE in E1 MuJoCo environment and export visualization.")
    parser.add_argument(
        "--xml",
        type=Path,
        default=DEFAULT_XML,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/home/wmh/ICODE/domo/ICODE_MPPI/runs/icode_e1_mix_stageB/icode_best.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/wmh/ICODE/domo/ICODE_MPPI/runs/mppi_icode_e1_test"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--planner-model", type=str, default="kinematic", choices=("icode", "kinematic", "hybrid"))
    parser.add_argument("--kinematic-lock", action="store_true", default=True)
    parser.add_argument("--no-kinematic-lock", action="store_false", dest="kinematic_lock")
    parser.add_argument("--seed", type=int, default=20260313)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--goal-tol", type=float, default=0.35)

    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--mppi-lambda", type=float, default=1.0)
    parser.add_argument("--noise-sigma", type=float, nargs=2, default=(1.6, 1.6))
    parser.add_argument("--noise-rho", type=float, default=0.82)
    parser.add_argument("--u-init", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--cost-task-goal", type=float, default=2.8)
    parser.add_argument("--cost-task-final", type=float, default=140.0)
    parser.add_argument("--cost-task-progress", type=float, default=120.0)
    parser.add_argument("--cost-task-heading", type=float, default=8.0)
    parser.add_argument("--cost-task-reverse", type=float, default=5.0)
    parser.add_argument("--cost-task-path-track", type=float, default=10.0)
    parser.add_argument("--cost-task-path-terminal", type=float, default=40.0)
    parser.add_argument("--cost-task-path-progress", type=float, default=120.0)
    parser.add_argument("--cost-task-lateral", type=float, default=1.2)
    parser.add_argument("--cost-task-goal-visibility", type=float, default=6.0)
    parser.add_argument("--cost-safe-collision", type=float, default=1500.0)
    parser.add_argument("--cost-safe-near", type=float, default=0.6)
    parser.add_argument("--cost-safe-corridor", type=float, default=120.0)
    parser.add_argument("--cost-safe-pred-obs", type=float, default=80.0)
    parser.add_argument("--cost-safe-bounds", type=float, default=120.0)
    parser.add_argument("--cost-safe-bounds-terminal", type=float, default=260.0)
    parser.add_argument("--collision-step-clearance", type=float, default=0.10)
    parser.add_argument("--collision-step-scale", type=float, default=0.08)
    parser.add_argument("--collision-step-hit-scale", type=float, default=2.5)
    parser.add_argument("--cost-ctrl-effort", type=float, default=0.01)
    parser.add_argument("--cost-ctrl-smooth", type=float, default=0.08)
    parser.add_argument("--cost-ctrl-spin", type=float, default=0.08)
    parser.add_argument("--cost-ctrl-wheel-diff", type=float, default=0.02)
    parser.add_argument("--cost-terminal-progress-deficit", type=float, default=120.0)
    parser.add_argument("--cost-terminal-near-goal-stall", type=float, default=55.0)
    parser.add_argument("--cost-terminal-stop", type=float, default=120.0)
    parser.add_argument("--cost-terminal-overshoot", type=float, default=140.0)
    parser.add_argument("--near-goal-radius", type=float, default=0.90)
    parser.add_argument("--near-goal-progress-eps", type=float, default=0.003)
    parser.add_argument("--los-margin", type=float, default=0.10)
    parser.add_argument("--obs-behind-scale", type=float, default=0.25)
    parser.add_argument("--v-forward-sign", type=float, default=-1.0)
    parser.add_argument("--icode-blend-alpha", type=float, default=0.35)
    parser.add_argument("--icode-diverge-fallback", action="store_true", default=True)
    parser.add_argument("--no-icode-diverge-fallback", action="store_false", dest="icode_diverge_fallback")
    parser.add_argument("--icode-diverge-window", type=int, default=45)
    parser.add_argument("--icode-diverge-min-step", type=int, default=30)
    parser.add_argument("--icode-diverge-min-increase", type=float, default=0.35)
    parser.add_argument("--icode-fallback-alpha", type=float, default=0.15)
    parser.add_argument("--robot-radius", type=float, default=0.28)
    parser.add_argument("--obs-margin", type=float, default=0.25)
    parser.add_argument("--near-penalty-clearance", type=float, default=0.20)
    parser.add_argument("--near-penalty-mid-clearance", type=float, default=0.34)
    parser.add_argument("--near-penalty-hard-clearance", type=float, default=0.15)
    parser.add_argument("--near-penalty-mid-scale", type=float, default=0.06)
    parser.add_argument("--near-penalty-hard-scale", type=float, default=0.55)
    parser.add_argument("--near-penalty-hard-power", type=float, default=3.2)
    parser.add_argument("--pred-obs-clearance", type=float, default=0.30)
    parser.add_argument("--pred-obs-hard-clearance", type=float, default=0.18)
    parser.add_argument("--pred-obs-hard-scale", type=float, default=2.0)
    parser.add_argument("--pred-obs-clip-max", type=float, default=8.0)
    parser.add_argument("--near-progress-start", type=float, default=0.75)
    parser.add_argument("--near-progress-hard", type=float, default=0.55)
    parser.add_argument("--near-progress-weight", type=float, default=70.0)
    parser.add_argument("--near-stall-progress-eps", type=float, default=0.015)
    parser.add_argument("--near-stall-effort-gate", type=float, default=0.45)
    parser.add_argument("--path-corridor-half-width", type=float, default=0.34)
    parser.add_argument("--max-delta-u", type=float, default=0.80)
    parser.add_argument("--soft-delta-projection", action="store_true", default=True)
    parser.add_argument("--no-soft-delta-projection", action="store_false", dest="soft_delta_projection")
    parser.add_argument("--action-post-delta-eps", type=float, default=1e-4)
    parser.add_argument("--nominal-forward-only", action="store_true", default=True)
    parser.add_argument("--no-nominal-forward-only", action="store_false", dest="nominal_forward_only")
    parser.add_argument("--nominal-min-forward-speed", type=float, default=0.07)
    parser.add_argument("--nav-min-forward-speed", type=float, default=0.07)
    parser.add_argument("--nav-speed-max", type=float, default=2.0)
    parser.add_argument("--terminal-min-forward-speed", type=float, default=0.02)
    parser.add_argument("--terminal-speed-max", type=float, default=0.55)
    parser.add_argument("--speed-cap-clearance-hard", type=float, default=0.10)
    parser.add_argument("--speed-cap-clearance-soft", type=float, default=0.40)
    parser.add_argument("--speed-cap-heading-gate", type=float, default=0.95)
    parser.add_argument("--speed-cap-heading-min-gain", type=float, default=0.20)
    parser.add_argument("--terminal-mppi-enter-radius", type=float, default=0.65)
    parser.add_argument("--terminal-profile-alpha", type=float, default=0.10)
    add_supervisor_args(parser)
    parser.add_argument("--adaptive-scheduler", action="store_true")
    parser.add_argument("--adaptive-ema", type=float, default=0.18)
    parser.add_argument("--adaptive-mode-min-steps", type=int, default=14)
    parser.add_argument("--adaptive-tight-clearance", type=float, default=0.38)
    parser.add_argument("--adaptive-tight-clearance-exit", type=float, default=0.52)
    parser.add_argument("--adaptive-stuck-progress", type=float, default=0.06)
    parser.add_argument("--adaptive-osc-window", type=int, default=10)
    parser.add_argument("--adaptive-osc-threshold", type=float, default=2.6)
    parser.add_argument("--adaptive-dock-radius", type=float, default=0.75)
    parser.add_argument("--terminal-stop-radius", type=float, default=0.70)
    parser.add_argument("--overshoot-tolerance", type=float, default=0.05)
    parser.add_argument("--noise-anneal-dist", type=float, default=1.8)
    parser.add_argument("--noise-anneal-min-scale", type=float, default=0.30)
    parser.add_argument("--bounds-x-range", type=float, nargs=2, default=(-0.8, 3.2))
    parser.add_argument("--bounds-y-range", type=float, nargs=2, default=(-2.0, 2.0))
    parser.add_argument("--boundary-recover-guide-steps", type=int, default=40)

    parser.add_argument("--wheel-radius", type=float, default=0.085)
    parser.add_argument("--wheel-base", type=float, default=0.37)
    parser.add_argument("--yaw-blend-alpha", type=float, default=0.35)
    parser.add_argument("--control-decimation", type=int, default=10)
    parser.add_argument("--pose-source", type=str, default="gt", choices=("gt", "odom"))
    parser.add_argument("--oracle-mode", action="store_true", help="Use GT pose/obstacles for planning (legacy baseline).")
    parser.add_argument("--sensor-use-depth", action="store_true", default=False)
    parser.add_argument("--no-sensor-use-depth", action="store_false", dest="sensor_use_depth")
    parser.add_argument("--depth-required", action="store_true", default=False)
    parser.add_argument("--depth-camera", type=str, default="front_depth_cam")
    parser.add_argument("--depth-width", type=int, default=160)
    parser.add_argument("--depth-height", type=int, default=120)
    parser.add_argument("--sensor-default-far", type=float, default=6.0)
    parser.add_argument("--sensor-collision-threshold", type=float, default=0.30)
    parser.add_argument("--sensor-obs-radius", type=float, default=0.12)
    parser.add_argument("--sensor-use-ray-obstacles", action="store_true", default=False)
    parser.add_argument("--no-sensor-use-ray-obstacles", action="store_false", dest="sensor_use_ray_obstacles")
    parser.add_argument("--sensor-ray-footprint-scale", type=float, default=0.90)
    parser.add_argument("--sensor-ray-radius-max-scale", type=float, default=2.60)
    parser.add_argument("--sensor-obs-max-range", type=float, default=3.0)
    parser.add_argument("--sensor-obs-min-range", type=float, default=0.12)
    parser.add_argument("--sensor-depth-max-points", type=int, default=20)
    parser.add_argument("--sensor-depth-hfov-deg", type=float, default=86.0)
    parser.add_argument("--sensor-depth-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--depth-min-valid", type=float, default=0.10)
    parser.add_argument("--sensor-obs-memory-steps", type=int, default=5)
    parser.add_argument("--sensor-obs-memory-max-points", type=int, default=96)
    parser.add_argument("--depth-filter-window", type=int, default=3)
    parser.add_argument("--depth-front-ema-alpha", type=float, default=0.6)
    parser.add_argument("--front-guide-fov-deg", type=float, default=110.0)
    parser.add_argument("--front-guide-min-x", type=float, default=0.02)
    parser.add_argument("--front-guide-hold-steps", type=int, default=3)
    parser.add_argument("--front-guide-max-points", type=int, default=64)
    parser.add_argument("--guide-use-all-obstacles", action="store_true", default=True)
    parser.add_argument("--no-guide-use-all-obstacles", action="store_false", dest="guide_use_all_obstacles")
    parser.add_argument("--front-guide-fallback-all", action="store_true", default=True)
    parser.add_argument("--no-front-guide-fallback-all", action="store_false", dest="front_guide_fallback_all")
    parser.add_argument("--guide-obs-radius-scale", type=float, default=1.00)
    parser.add_argument("--guide-obs-radius-min", type=float, default=0.12)
    parser.add_argument("--sensor-guide-robot-radius", type=float, default=0.28)
    parser.add_argument("--sensor-guide-inflate-margin", type=float, default=0.10)
    parser.add_argument("--planner-passability-check", action="store_true", default=True)
    parser.add_argument("--no-planner-passability-check", action="store_false", dest="planner_passability_check")
    parser.add_argument("--planner-passability-margin", type=float, default=0.02)
    parser.add_argument("--planner-passability-min-clearance", type=float, default=0.02)
    parser.add_argument("--sensor-goal-los-margin", type=float, default=0.0)
    parser.add_argument("--sensor-waypoint-margin", type=float, default=0.06)
    parser.add_argument("--map-resolution", type=float, default=0.05)
    parser.add_argument("--map-x-range", type=float, nargs=2, default=(-1.0, 4.0))
    parser.add_argument("--map-y-range", type=float, nargs=2, default=(-2.0, 2.0))
    parser.add_argument("--map-occ-threshold", type=float, default=0.35)
    parser.add_argument("--map-hit-spread-cells", type=int, default=0)
    parser.add_argument("--map-observed-threshold", type=float, default=0.15)
    parser.add_argument("--map-min-cluster-cells", type=int, default=2)
    parser.add_argument("--map-max-obstacles", type=int, default=96)
    parser.add_argument("--goal-blocked-enter-conf", type=float, default=0.65)
    parser.add_argument("--goal-blocked-exit-conf", type=float, default=0.35)
    parser.add_argument("--goal-blocked-enter-steps", type=int, default=3)
    parser.add_argument("--goal-blocked-exit-steps", type=int, default=5)
    parser.add_argument("--goal-blocked-corridor-len", type=float, default=1.2)
    parser.add_argument("--goal-blocked-corridor-half-width", type=float, default=0.45)
    parser.add_argument("--goal-blocked-front-fov-deg", type=float, default=80.0)
    parser.add_argument("--goal-blocked-front-range", type=float, default=0.9)
    parser.add_argument("--depth-gate-window", type=int, default=20)
    parser.add_argument("--depth-gate-min-valid-ratio", type=float, default=0.015)
    parser.add_argument("--depth-gate-max-sector-delta", type=float, default=0.70)
    parser.add_argument("--depth-gate-max-front-delta", type=float, default=1.10)
    parser.add_argument("--depth-gate-max-fail-ratio", type=float, default=0.55)

    parser.add_argument("--target-x", type=float, default=None)
    parser.add_argument("--target-y", type=float, default=None)
    parser.add_argument("--target-forward-m", type=float, default=0.0)
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
    parser.add_argument("--global-guide-source", type=str, choices=("sensor", "gt"), default="gt")
    parser.add_argument("--global-planner", type=str, default="hybrid_astar", choices=("hybrid_astar", "astar", "bfs"))
    parser.add_argument("--hybrid-n-theta", type=int, default=40)
    parser.add_argument("--hybrid-step-cells", type=float, default=2.8)
    parser.add_argument("--hybrid-turn-penalty", type=float, default=0.22)
    parser.add_argument("--hybrid-max-expansions", type=int, default=14000)
    parser.add_argument("--guide-replan-interval", type=int, default=14)
    parser.add_argument("--guide-fail-latch-steps", type=int, default=20)
    parser.add_argument("--guide-fail-latch-max-steps", type=int, default=120)
    parser.add_argument("--guide-fail-backoff-mult-cap", type=int, default=4)
    parser.add_argument("--guide-fail-retry-min-move", type=float, default=0.12)
    parser.add_argument("--guide-fail-retry-min-yaw", type=float, default=0.25)
    parser.add_argument("--guide-lookahead", type=float, default=0.9)
    parser.add_argument("--guide-ref-step-m", type=float, default=0.09)
    parser.add_argument("--guide-max-grid-cells", type=int, default=240000)
    parser.add_argument("--reference-sampling", action="store_true", default=True)
    parser.add_argument("--no-reference-sampling", action="store_false", dest="reference_sampling")
    parser.add_argument("--reference-tracker-kp-v", type=float, default=1.20)
    parser.add_argument("--reference-tracker-kp-w", type=float, default=2.40)
    parser.add_argument("--reference-nominal-v-max", type=float, default=0.90)
    parser.add_argument("--reference-nominal-w-max", type=float, default=1.80)
    parser.add_argument("--reference-tracker-stop-dist", type=float, default=0.04)
    parser.add_argument("--reference-tracker-forward-only", action="store_true", default=True)
    parser.add_argument("--no-reference-tracker-forward-only", action="store_false", dest="reference_tracker_forward_only")
    parser.add_argument("--reference-tracker-min-v", type=float, default=0.03)
    parser.add_argument("--goal-direct-on-clear", action="store_true", default=True)
    parser.add_argument("--no-goal-direct-on-clear", action="store_false", dest="goal_direct_on_clear")
    parser.add_argument("--goal-direct-clear-steps", type=int, default=18)
    parser.add_argument("--goal-direct-dist", type=float, default=1.6)
    parser.add_argument("--sensor-goal-direct-dist", type=float, default=0.9)
    parser.add_argument("--sensor-strict-path", action="store_true", default=True)
    parser.add_argument("--no-sensor-strict-path", action="store_false", dest="sensor_strict_path")
    parser.add_argument("--goal-direct-yaw-max", type=float, default=0.75)
    parser.add_argument("--goal-direct-lateral-max", type=float, default=0.30)
    parser.add_argument("--corridor-commit", action="store_true", default=True)
    parser.add_argument("--no-corridor-commit", action="store_false", dest="corridor_commit")
    parser.add_argument("--target-bound-margin", type=float, default=0.10)
    parser.add_argument("--corridor-target-slack", type=float, default=0.18)
    parser.add_argument("--corridor-tight-clearance", type=float, default=0.36)
    parser.add_argument("--corridor-tight-max-dev", type=float, default=0.32)
    parser.add_argument("--target-jump-max", type=float, default=0.62)
    parser.add_argument("--target-jump-min", type=float, default=0.18)
    parser.add_argument("--target-passability-guard", action="store_true", default=True)
    parser.add_argument("--no-target-passability-guard", action="store_false", dest="target_passability_guard")
    parser.add_argument("--target-guard-min-clearance", type=float, default=0.06)
    parser.add_argument("--target-guard-margin", type=float, default=0.04)
    parser.add_argument("--target-guard-min-progress", type=float, default=0.18)
    parser.add_argument("--target-guard-backtrack-points", type=int, default=16)
    parser.add_argument("--target-guard-ray-samples", type=int, default=10)
    parser.add_argument("--guide-backtrack-max", type=int, default=0)
    parser.add_argument("--commit-latch-steps", type=int, default=80)
    parser.add_argument("--commit-release-progress", type=float, default=0.35)
    parser.add_argument("--commit-lateral-bias", type=float, default=0.22)
    parser.add_argument("--commit-max-path-dev", type=float, default=0.70)
    parser.add_argument("--commit-stuck-window", type=int, default=40)
    parser.add_argument("--commit-stuck-min-progress", type=float, default=0.05)
    parser.add_argument("--waypoint-use-los-gating", action="store_true", default=True)
    parser.add_argument("--no-waypoint-use-los-gating", action="store_false", dest="waypoint_use_los_gating")
    parser.add_argument("--waypoint-replan-interval", type=int, default=16)
    parser.add_argument("--waypoint-replan-on-interval", action="store_true", default=False)
    parser.add_argument("--waypoint-goal-min-dist", type=float, default=0.20)
    parser.add_argument("--waypoint-infeasible-min-clearance", type=float, default=0.012)
    parser.add_argument("--waypoint-infeasible-confirm-steps", type=int, default=3)
    parser.add_argument("--waypoint-infeasible-force-latch-steps", type=int, default=90)
    parser.add_argument("--waypoint-switch-radius", type=float, default=0.35)
    parser.add_argument("--waypoint-clear-hysteresis-steps", type=int, default=14)
    parser.add_argument("--waypoint-min-hold-steps", type=int, default=20)
    parser.add_argument("--waypoint-margin", type=float, default=0.12)
    parser.add_argument("--goal-los-margin", type=float, default=0.02)
    parser.add_argument("--waypoint-lateral-extra", type=float, default=0.28)
    parser.add_argument("--waypoint-max-lateral", type=float, default=0.75)
    parser.add_argument("--waypoint-min-forward", type=float, default=0.20)
    parser.add_argument("--waypoint-max-forward", type=float, default=1.20)
    parser.add_argument("--waypoint-stuck-window", type=int, default=40)
    parser.add_argument("--waypoint-stuck-min-progress", type=float, default=0.08)
    parser.add_argument("--waypoint-stuck-cooldown", type=int, default=20)
    parser.add_argument("--waypoint-stuck-force-latch-steps", type=int, default=120)
    parser.add_argument("--waypoint-force-max-level", type=int, default=3)
    parser.add_argument("--waypoint-force-lateral-boost", type=float, default=0.20)
    parser.add_argument("--waypoint-force-forward-boost", type=float, default=0.35)
    parser.add_argument("--waypoint-stuck-flip-side", action="store_true", default=True)
    parser.add_argument("--no-waypoint-stuck-flip-side", action="store_false", dest="waypoint_stuck_flip_side")
    parser.add_argument("--blocked-force-waypoint-enable", action="store_true", default=True)
    parser.add_argument("--no-blocked-force-waypoint-enable", action="store_false", dest="blocked_force_waypoint_enable")
    parser.add_argument("--blocked-force-waypoint-window", type=int, default=18)
    parser.add_argument("--blocked-force-waypoint-min-progress", type=float, default=0.03)
    parser.add_argument("--blocked-force-waypoint-clearance", type=float, default=0.40)
    parser.add_argument("--blocked-force-waypoint-latch-steps", type=int, default=80)
    parser.add_argument("--init-x", type=float, default=None)
    parser.add_argument("--init-y", type=float, default=None)
    parser.add_argument("--init-yaw", type=float, default=None)
    parser.add_argument("--dock-brake-radius", type=float, default=0.90)
    parser.add_argument("--dock-enter-radius", type=float, default=0.40)
    parser.add_argument("--dock-exit-radius", type=float, default=0.65)
    parser.add_argument("--dock-k-brake", type=float, default=0.35)
    parser.add_argument("--dock-k-yaw", type=float, default=0.90)
    parser.add_argument("--dock-v-eps", type=float, default=0.08)
    parser.add_argument("--dock-w-eps", type=float, default=0.25)
    parser.add_argument("--dock-hold-steps", type=int, default=20)
    parser.add_argument("--dock-align-yaw-enter", type=float, default=0.55)
    parser.add_argument("--dock-align-lateral-enter", type=float, default=0.30)
    parser.add_argument("--dock-align-yaw-exit", type=float, default=0.85)
    parser.add_argument("--dock-align-lateral-exit", type=float, default=0.45)
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    metrics = run_episode(args)
    print("MPPI+ICODE test done.")
    print(json.dumps(metrics, indent=2))
