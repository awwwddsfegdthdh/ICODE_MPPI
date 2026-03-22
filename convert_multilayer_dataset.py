import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from state_convention import (
    CONTROL_DEFINITION,
    STATE_CONVENTION_VERSION,
    canonical_drive_sign,
    convention_as_meta,
    wrap_to_pi,
    world_to_body,
)


def _world_to_body_batch(vec_xy: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    return world_to_body(vec_xy, yaw)


def sanitize_ranges(ranges: np.ndarray, default_far: float) -> np.ndarray:
    out = ranges.copy()
    invalid = (~np.isfinite(out)) | (out <= 0.0)
    out[invalid] = default_far
    return out


def get_array_or_default(
    src: np.lib.npyio.NpzFile,
    key: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    if key in src.files:
        return src[key].astype(dtype)
    return np.zeros(shape, dtype=dtype)


def inverse_diff_drive(v_body: np.ndarray, wz_body: np.ndarray, wheel_radius: float, wheel_base: float) -> np.ndarray:
    dq_r = (v_body + 0.5 * wheel_base * wz_body) / wheel_radius
    dq_l = (v_body - 0.5 * wheel_base * wz_body) / wheel_radius
    return np.stack([dq_l, dq_r], axis=1).astype(np.float32)


def integrate_diff_drive_odometry(
    raw_episode: np.ndarray,
    sim_time: np.ndarray,
    wheel_vel: np.ndarray,
    imu_gyro: np.ndarray,
    wheel_radius: float,
    wheel_base: float,
    yaw_blend_alpha: float,
    drive_sign: float,
    init_pose_by_episode: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = sim_time.shape[0]
    x_odom = np.zeros((n,), dtype=np.float32)
    y_odom = np.zeros((n,), dtype=np.float32)
    psi_odom = np.zeros((n,), dtype=np.float32)
    v_body = np.zeros((n,), dtype=np.float32)
    wz_body = np.zeros((n,), dtype=np.float32)

    for i in range(n):
        ep = int(raw_episode[i])
        is_new_episode = i == 0 or ep != int(raw_episode[i - 1])

        dq_l = float(wheel_vel[i, 0])
        dq_r = float(wheel_vel[i, 1])

        v_now = float(drive_sign) * wheel_radius * 0.5 * (dq_r + dq_l)
        wz_wheel = wheel_radius * (dq_r - dq_l) / wheel_base
        imu_wz = float(imu_gyro[i, 2]) if imu_gyro.shape[1] >= 3 else wz_wheel
        if not np.isfinite(imu_wz):
            imu_wz = wz_wheel
        wz_now = yaw_blend_alpha * wz_wheel + (1.0 - yaw_blend_alpha) * imu_wz

        v_body[i] = np.float32(v_now)
        wz_body[i] = np.float32(wz_now)

        if is_new_episode:
            if init_pose_by_episode is not None and ep < init_pose_by_episode.shape[0]:
                x_odom[i] = np.float32(init_pose_by_episode[ep, 0])
                y_odom[i] = np.float32(init_pose_by_episode[ep, 1])
                psi_odom[i] = np.float32(init_pose_by_episode[ep, 2])
            else:
                x_odom[i] = 0.0
                y_odom[i] = 0.0
                psi_odom[i] = 0.0
            continue

        dt = float(sim_time[i] - sim_time[i - 1])
        if dt < 0.0 or not np.isfinite(dt):
            dt = 0.0

        prev_psi = float(psi_odom[i - 1])
        x_odom[i] = np.float32(x_odom[i - 1] + v_now * np.cos(prev_psi) * dt)
        y_odom[i] = np.float32(y_odom[i - 1] + v_now * np.sin(prev_psi) * dt)
        psi_odom[i] = np.float32(wrap_to_pi(np.array([prev_psi + wz_now * dt], dtype=np.float64))[0])

    return x_odom, y_odom, psi_odom, v_body, wz_body


def extract_goal_and_obstacles(src: np.lib.npyio.NpzFile) -> Tuple[np.ndarray, np.ndarray]:
    if "gt__target_pos_gt" in src.files:
        target_pos_gt = src["gt__target_pos_gt"].astype(np.float32)
    elif "gt__goal_pos_gt" in src.files:
        target_pos_gt = src["gt__goal_pos_gt"].astype(np.float32)
    else:
        raise KeyError("Neither gt__target_pos_gt nor gt__goal_pos_gt exists in source dataset.")

    if "gt__obs_pos_gt" in src.files:
        obs_pos_gt = src["gt__obs_pos_gt"].astype(np.float32)
    else:
        needed = ["gt__obs_front_pos_gt", "gt__obs_left_pos_gt", "gt__obs_right_pos_gt"]
        if not all(key in src.files for key in needed):
            raise KeyError("Obstacle GT keys are missing; expected gt__obs_pos_gt or front/left/right keys.")
        obs_pos_gt = np.stack(
            [
                src["gt__obs_front_pos_gt"].astype(np.float32),
                src["gt__obs_left_pos_gt"].astype(np.float32),
                src["gt__obs_right_pos_gt"].astype(np.float32),
            ],
            axis=1,
        )

    return target_pos_gt, obs_pos_gt


def convert_dataset(args: argparse.Namespace) -> None:
    src = np.load(args.input, allow_pickle=True)
    src_drive_sign = None
    if "meta__drive_sign" in src.files:
        try:
            src_drive_sign = float(src["meta__drive_sign"].reshape(-1)[0])
        except Exception:
            src_drive_sign = None
    effective_drive_sign = args.drive_sign if args.drive_sign is not None else src_drive_sign
    if effective_drive_sign is None:
        effective_drive_sign = -1.0
    effective_drive_sign = canonical_drive_sign(float(effective_drive_sign))

    raw_episode = src["raw__episode"].astype(np.int32)
    raw_step = src["raw__step"].astype(np.int32)
    sim_time = src["raw__sim_time"].astype(np.float64)
    u_applied = src["raw__ctrl_applied"].astype(np.float32)

    n = raw_episode.shape[0]
    imu_acc = get_array_or_default(src, "raw__imu_acc", (n, 3), np.float32)
    imu_gyro = get_array_or_default(src, "raw__imu_gyro", (n, 3), np.float32)
    touch_force = get_array_or_default(src, "raw__touch_front_force", (n,), np.float32)
    lidar_triplet = get_array_or_default(src, "raw__lidar_triplet", (n, 3), np.float32)

    source_mode = "legacy"
    wheel_pos = None
    wheel_vel = None
    joint_pos = None
    joint_vel = None

    if "raw__wheel_pos" in src.files and "raw__wheel_vel" in src.files:
        source_mode = "e1"
        wheel_pos = src["raw__wheel_pos"].astype(np.float32)
        wheel_vel = src["raw__wheel_vel"].astype(np.float32)

        init_pose_by_episode = None
        if "meta__episode_robot_init_xyyaw" in src.files:
            init_pose_by_episode = src["meta__episode_robot_init_xyyaw"].astype(np.float32)

        x_odom, y_odom, psi_odom, v_body, wz_body = integrate_diff_drive_odometry(
            raw_episode=raw_episode,
            sim_time=sim_time,
            wheel_vel=wheel_vel,
            imu_gyro=imu_gyro,
            wheel_radius=args.wheel_radius,
            wheel_base=args.wheel_base,
            yaw_blend_alpha=args.yaw_blend_alpha,
            drive_sign=effective_drive_sign,
            init_pose_by_episode=init_pose_by_episode,
        )

        dq_l = wheel_vel[:, 0].astype(np.float32)
        dq_r = wheel_vel[:, 1].astype(np.float32)

    elif "raw__joint_pos" in src.files and "raw__joint_vel" in src.files:
        source_mode = "legacy"
        joint_pos = src["raw__joint_pos"].astype(np.float32)  # [x, y, yaw]
        joint_vel = src["raw__joint_vel"].astype(np.float32)  # [vx_world, vy_world, wz]

        x_odom = joint_pos[:, 0].astype(np.float32)
        y_odom = joint_pos[:, 1].astype(np.float32)
        psi_odom = wrap_to_pi(joint_pos[:, 2]).astype(np.float32)

        v_body_xy = _world_to_body_batch(np.stack([joint_vel[:, 0], joint_vel[:, 1]], axis=1), psi_odom)
        v_body = v_body_xy[:, 0].astype(np.float32)
        wz_body = joint_vel[:, 2].astype(np.float32)

        wheel_vel = inverse_diff_drive(
            v_body=v_body,
            wz_body=wz_body,
            wheel_radius=args.wheel_radius,
            wheel_base=args.wheel_base,
        )
        dq_l = wheel_vel[:, 0]
        dq_r = wheel_vel[:, 1]
    else:
        raise KeyError("Input dataset schema not recognized. Need either raw__wheel_* or raw__joint_* keys.")

    target_pos_gt, obs_pos_gt = extract_goal_and_obstacles(src)

    goal_delta_world = target_pos_gt[:, :2] - np.stack([x_odom, y_odom], axis=1)
    goal_rel_body = _world_to_body_batch(goal_delta_world, psi_odom)
    goal_dist = np.linalg.norm(goal_rel_body, axis=1)
    goal_heading_err = np.arctan2(goal_rel_body[:, 1], goal_rel_body[:, 0])

    depth_sector_min = sanitize_ranges(lidar_triplet, args.default_far)
    free_corridor_width = np.clip(depth_sector_min[:, 0] + depth_sector_min[:, 2], 0.0, 2.0 * args.default_far)
    depth_collision_flag = (np.min(depth_sector_min, axis=1) < args.collision_threshold).astype(np.float32)
    touch_flag = (touch_force > args.touch_force_threshold).astype(np.float32)

    state_est = np.stack([x_odom, y_odom, psi_odom, v_body, wz_body, dq_l, dq_r], axis=1).astype(np.float32)

    icode_x_t = state_est.copy()
    icode_u_t = u_applied.copy()

    mppi_state_t = state_est.copy()
    mppi_u_t = u_applied.copy()
    mppi_u_prev = np.zeros_like(mppi_u_t)
    if mppi_u_t.shape[0] > 1:
        mppi_u_prev[1:] = mppi_u_t[:-1]
    episode_starts = np.zeros((n,), dtype=bool)
    episode_starts[0] = True
    episode_starts[1:] = raw_episode[1:] != raw_episode[:-1]
    mppi_u_prev[episode_starts] = 0.0

    mppi_cost_context = np.concatenate(
        [
            goal_rel_body.astype(np.float32),
            goal_dist[:, None].astype(np.float32),
            goal_heading_err[:, None].astype(np.float32),
            depth_sector_min.astype(np.float32),
            free_corridor_width[:, None].astype(np.float32),
            depth_collision_flag[:, None],
            touch_flag[:, None],
            v_body[:, None].astype(np.float32),
            wz_body[:, None].astype(np.float32),
        ],
        axis=1,
    )

    base_pos_gt = get_array_or_default(src, "gt__base_pos_gt", (n, 3), np.float32)
    base_quat_gt = get_array_or_default(src, "gt__base_quat_gt", (n, 4), np.float32)
    base_linvel_gt = get_array_or_default(src, "gt__base_linvel_gt", (n, 3), np.float32)
    base_angvel_gt = get_array_or_default(src, "gt__base_angvel_gt", (n, 3), np.float32)
    camera_pos_gt = get_array_or_default(src, "gt__camera_pos_gt", (n, 3), np.float32)
    camera_quat_gt = get_array_or_default(src, "gt__camera_quat_gt", (n, 4), np.float32)

    output_parent = args.output.parent
    output_parent.mkdir(parents=True, exist_ok=True)

    save_dict: Dict[str, np.ndarray] = {
        "meta__source_file": np.array([str(args.input)], dtype=object),
        "meta__source_mode": np.array([source_mode], dtype=object),
        "meta__wheel_radius": np.array([args.wheel_radius], dtype=np.float32),
        "meta__wheel_base": np.array([args.wheel_base], dtype=np.float32),
        "meta__yaw_blend_alpha": np.array([args.yaw_blend_alpha], dtype=np.float32),
        "meta__state_convention_version": np.array([STATE_CONVENTION_VERSION], dtype=object),
        "meta__drive_sign": np.array([effective_drive_sign], dtype=np.float32),
        "meta__control_definition": np.array([CONTROL_DEFINITION], dtype=object),
        "meta__state_est_fields": np.array(
            ["x_odom", "y_odom", "psi_odom", "v_body", "wz_body", "dqL", "dqR"],
            dtype=object,
        ),
        "meta__mppi_context_fields": np.array(
            [
                "goal_rel_body_x",
                "goal_rel_body_y",
                "goal_dist",
                "goal_heading_err",
                "depth_lf_min",
                "depth_f_min",
                "depth_rf_min",
                "free_corridor_width",
                "depth_collision_flag",
                "touch_flag",
                "v_body",
                "wz_body",
            ],
            dtype=object,
        ),
        # raw layer
        "raw__episode": raw_episode,
        "raw__step": raw_step,
        "raw__sim_time": sim_time,
        "raw__ctrl_applied": u_applied,
        "raw__imu_acc": imu_acc,
        "raw__imu_gyro": imu_gyro,
        "raw__touch_front_force": touch_force,
        "raw__lidar_triplet": lidar_triplet,
        # derived layer
        "derived__goal_rel_body_from_odom_yaw": goal_rel_body.astype(np.float32),
        "derived__goal_dist": goal_dist.astype(np.float32),
        "derived__goal_heading_err_from_odom_yaw": goal_heading_err.astype(np.float32),
        "derived__depth_sector_min": depth_sector_min.astype(np.float32),
        "derived__free_corridor_width": free_corridor_width.astype(np.float32),
        "derived__depth_collision_flag": depth_collision_flag.astype(np.float32),
        "derived__touch_flag": touch_flag.astype(np.float32),
        # state_est layer
        "state_est__x_t": state_est,
        # supervision_gt layer
        "gt__base_pos_gt": base_pos_gt,
        "gt__base_quat_gt": base_quat_gt,
        "gt__base_linvel_gt": base_linvel_gt,
        "gt__base_angvel_gt": base_angvel_gt,
        "gt__camera_pos_gt": camera_pos_gt,
        "gt__camera_quat_gt": camera_quat_gt,
        "gt__target_pos_gt": target_pos_gt,
        "gt__obs_pos_gt": obs_pos_gt,
        # ICODE/MPPI ready tensors
        "icode__x_t": icode_x_t,
        "icode__u_t": icode_u_t,
        "mppi__state_t": mppi_state_t,
        "mppi__u_t": mppi_u_t,
        "mppi__u_prev": mppi_u_prev,
        "mppi__cost_context": mppi_cost_context.astype(np.float32),
    }

    if wheel_pos is not None:
        save_dict["raw__wheel_pos"] = wheel_pos
    if wheel_vel is not None:
        save_dict["raw__wheel_vel"] = wheel_vel
    if joint_pos is not None:
        save_dict["raw__joint_pos"] = joint_pos
    if joint_vel is not None:
        save_dict["raw__joint_vel"] = joint_vel

    if "gt__goal_pos_gt" in src.files:
        save_dict["gt__goal_pos_gt"] = src["gt__goal_pos_gt"].astype(np.float32)
    if "gt__obs_front_pos_gt" in src.files:
        save_dict["gt__obs_front_pos_gt"] = src["gt__obs_front_pos_gt"].astype(np.float32)
    if "gt__obs_left_pos_gt" in src.files:
        save_dict["gt__obs_left_pos_gt"] = src["gt__obs_left_pos_gt"].astype(np.float32)
    if "gt__obs_right_pos_gt" in src.files:
        save_dict["gt__obs_right_pos_gt"] = src["gt__obs_right_pos_gt"].astype(np.float32)

    if "raw__depth_image" in src.files:
        save_dict["raw__depth_image"] = src["raw__depth_image"]
    if "raw__distance_image" in src.files:
        save_dict["raw__distance_image"] = src["raw__distance_image"]
    if "raw__depth_valid_mask" in src.files:
        save_dict["raw__depth_valid_mask"] = src["raw__depth_valid_mask"]
    for k in (
        "raw__goal_rel_body",
        "raw__goal_dist",
        "raw__goal_heading_err",
        "raw__depth_sector_min",
        "raw__front_clearance",
        "raw__corridor_width",
        "raw__sensor_collision_flag",
        "raw__recover_trigger",
    ):
        if k in src.files:
            save_dict[k] = src[k]
    if "raw__goal_rel_body" in src.files:
        save_dict["raw__goal_rel_body_from_base_yaw"] = src["raw__goal_rel_body"].astype(np.float32)
    if "raw__goal_heading_err" in src.files:
        save_dict["raw__goal_heading_err_from_base_yaw"] = src["raw__goal_heading_err"].astype(np.float32)
    if "meta__camera_intrinsics" in src.files:
        save_dict["meta__camera_intrinsics"] = src["meta__camera_intrinsics"]
    if "meta__mujoco_gl_backend" in src.files:
        save_dict["meta__mujoco_gl_backend"] = src["meta__mujoco_gl_backend"]
    if "meta__episode_robot_init_xyyaw" in src.files:
        save_dict["meta__episode_robot_init_xyyaw"] = src["meta__episode_robot_init_xyyaw"]
    if "meta__episode_goal_pos" in src.files:
        save_dict["meta__episode_goal_pos"] = src["meta__episode_goal_pos"]
    if "meta__episode_obs_pos" in src.files:
        save_dict["meta__episode_obs_pos"] = src["meta__episode_obs_pos"]
    if "meta__episode_obs_active_mask" in src.files:
        save_dict["meta__episode_obs_active_mask"] = src["meta__episode_obs_active_mask"]
    if "meta__episode_obs_count" in src.files:
        save_dict["meta__episode_obs_count"] = src["meta__episode_obs_count"]
    if "meta__episode_obs_geom_size" in src.files:
        save_dict["meta__episode_obs_geom_size"] = src["meta__episode_obs_geom_size"]
    if "meta__episode_layout_success" in src.files:
        save_dict["meta__episode_layout_success"] = src["meta__episode_layout_success"]
    if "meta__control_mode" in src.files:
        save_dict["meta__control_mode"] = src["meta__control_mode"]
    if "meta__ctrl_hold_steps" in src.files:
        save_dict["meta__ctrl_hold_steps"] = src["meta__ctrl_hold_steps"]
    if "meta__ctrl_noise_std" in src.files:
        save_dict["meta__ctrl_noise_std"] = src["meta__ctrl_noise_std"]
    if "meta__ctrl_resample_prob" in src.files:
        save_dict["meta__ctrl_resample_prob"] = src["meta__ctrl_resample_prob"]
    if "meta__pose_source" in src.files:
        save_dict["meta__pose_source"] = src["meta__pose_source"]
    if "meta__heading_source" in src.files:
        save_dict["meta__heading_source"] = src["meta__heading_source"]
    if "meta__yaw_source" in src.files:
        save_dict["meta__yaw_source"] = src["meta__yaw_source"]
    save_dict.update(
        convention_as_meta(
            drive_sign=float(effective_drive_sign),
            pose_source=str(save_dict.get("meta__pose_source", np.array(["odom"], dtype=object)).reshape(-1)[0]),
            heading_source=str(save_dict.get("meta__heading_source", np.array(["base"], dtype=object)).reshape(-1)[0]),
            yaw_source="odom_yaw",
        )
    )

    np.savez_compressed(args.output, **save_dict)
    print(f"Saved converted multilayer dataset to: {args.output}")
    print(f"Samples: {state_est.shape[0]}")
    print(f"State dim: {state_est.shape[1]}")
    print("Layers: raw / derived / state_est / supervision_gt")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert raw+gt MuJoCo dataset to ICODE/MPPI-ready multilayer format.")
    parser.add_argument("--input", type=Path, default=Path("datasets/multilayer_raw_gt.npz"))
    parser.add_argument("--output", type=Path, default=Path("datasets/multilayer_converted.npz"))
    parser.add_argument("--default-far", type=float, default=10.0, help="Fallback range for invalid lidar/depth scalar values.")
    parser.add_argument("--collision-threshold", type=float, default=0.35)
    parser.add_argument("--touch-force-threshold", type=float, default=1e-5)
    parser.add_argument("--wheel-radius", type=float, default=0.085)
    parser.add_argument("--wheel-base", type=float, default=0.37)
    parser.add_argument("--drive-sign", type=float, default=None, help="Override drive sign (+1/-1). Default: use source meta.")
    parser.add_argument(
        "--yaw-blend-alpha",
        type=float,
        default=0.35,
        help="Wheel-based yaw-rate blending weight in [0, 1].",
    )
    return parser


if __name__ == "__main__":
    convert_dataset(build_argparser().parse_args())
