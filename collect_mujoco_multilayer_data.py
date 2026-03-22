import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np
from mppi_nav_utils import sample_obstacles_adaptive
from state_convention import (
    CONTROL_DEFINITION,
    STATE_CONVENTION_VERSION,
    canonical_drive_sign,
    convention_as_meta,
    diff_drive_inverse,
)


DEFAULT_XML = (
    Path(__file__).resolve().parents[2]
    / "E1_Robot"
    / "simulation"
    / "models"
    / "mjcf"
    / "E1_SimpleSensor_5obs.xml"
)

CAMERA_NAME = "front_depth_cam"
GOAL_BODY_NAME = "goal_marker"
OBSTACLE_BODY_NAMES = ("obs_front", "obs_left", "obs_right", "obs_extra1", "obs_extra2")
OBSTACLE_GEOM_NAMES = ("obs_front_geom", "obs_left_geom", "obs_right_geom", "obs_extra1_geom", "obs_extra2_geom")
OBSTACLE_SITE_NAMES = ("obs_front_site", "obs_left_site", "obs_right_site", "obs_extra1_site", "obs_extra2_site")
OBSTACLE_GT_SENSOR_NAMES = (
    "obs_front_pos_gt",
    "obs_left_pos_gt",
    "obs_right_pos_gt",
    "obs_extra1_pos_gt",
    "obs_extra2_pos_gt",
)

RAW_SENSOR_NAMES = [
    "wheel_l_pos",
    "wheel_r_pos",
    "wheel_l_vel",
    "wheel_r_vel",
    "imu_acc",
    "imu_gyro",
    "touch_front_force",
    "lidar_left30",
    "lidar_front",
    "lidar_right30",
]

GT_SENSOR_NAMES = [
    "base_pos_gt",
    "base_quat_gt",
    "base_linvel_gt",
    "base_angvel_gt",
    "camera_pos_gt",
    "camera_quat_gt",
    "goal_pos_gt",
]


def choose_gl_backends(gl_backend: str) -> Tuple[str, ...]:
    if gl_backend != "auto":
        return (gl_backend,)

    # Headless by default: try EGL first, then OSMesa, then GLFW.
    if os.environ.get("DISPLAY"):
        return ("glfw", "egl", "osmesa")
    return ("egl", "osmesa", "glfw")


def probe_gl_backend(backend: str, xml_path: Path, width: int, height: int) -> Tuple[bool, str]:
    env = os.environ.copy()
    env["MUJOCO_GL"] = backend
    if backend in ("egl", "osmesa"):
        env["PYOPENGL_PLATFORM"] = backend
    elif "PYOPENGL_PLATFORM" in env:
        del env["PYOPENGL_PLATFORM"]

    probe_code = (
        "import sys, mujoco; "
        "m=mujoco.MjModel.from_xml_path(sys.argv[1]); "
        "r=mujoco.Renderer(m, height=int(sys.argv[2]), width=int(sys.argv[3])); "
        "r.close(); "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe_code, str(xml_path), str(height), str(width)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode == 0:
        return True, proc.stdout.strip() or "ok"
    return False, proc.stderr.strip() or proc.stdout.strip() or f"backend {backend} probe failed"


def resolve_gl_backend(xml_path: Path, width: int, height: int, gl_backend: str) -> Tuple[str, str]:
    errors = []
    for backend in choose_gl_backends(gl_backend):
        ok, msg = probe_gl_backend(backend, xml_path=xml_path, width=width, height=height)
        if ok:
            return backend, " | ".join(errors)
        errors.append(f"{backend}: {msg}")

    joined = " | ".join(errors)
    raise RuntimeError(
        "Failed to create MuJoCo renderer for depth collection. "
        f"Tried backends: {choose_gl_backends(gl_backend)}. Errors: {joined}"
    )


def ensure_gl_backend_or_reexec(args: argparse.Namespace) -> str:
    if args.no_depth:
        return "none"

    if os.environ.get("_ICODE_GL_LOCK") == "1":
        return os.environ.get("_ICODE_GL_SELECTED", os.environ.get("MUJOCO_GL", "unknown"))

    selected_backend, _ = resolve_gl_backend(
        xml_path=args.xml,
        width=args.width,
        height=args.height,
        gl_backend=args.gl_backend,
    )

    env = os.environ.copy()
    env["MUJOCO_GL"] = selected_backend
    env["_ICODE_GL_LOCK"] = "1"
    env["_ICODE_GL_SELECTED"] = selected_backend
    if selected_backend in ("egl", "osmesa"):
        env["PYOPENGL_PLATFORM"] = selected_backend
    elif "PYOPENGL_PLATFORM" in env:
        del env["PYOPENGL_PLATFORM"]

    print(f"Depth backend probe selected: {selected_backend}. Relaunching with MUJOCO_GL={selected_backend}.")
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)
    raise RuntimeError("Unreachable after os.execvpe")


def sensor_value(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"Sensor '{name}' not found in model.")
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[adr : adr + dim].copy()


def discover_obstacles(model: mujoco.MjModel) -> Tuple[List[str], List[str], List[str], List[str], List[int], List[int], List[int]]:
    body_names: List[str] = []
    geom_names: List[str] = []
    site_names: List[str] = []
    gt_sensor_names: List[str] = []
    body_ids: List[int] = []
    geom_ids: List[int] = []
    site_ids: List[int] = []

    for body_name, geom_name, site_name, sensor_name in zip(
        OBSTACLE_BODY_NAMES,
        OBSTACLE_GEOM_NAMES,
        OBSTACLE_SITE_NAMES,
        OBSTACLE_GT_SENSOR_NAMES,
    ):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if body_id < 0 or geom_id < 0 or site_id < 0 or sensor_id < 0:
            continue
        body_names.append(body_name)
        geom_names.append(geom_name)
        site_names.append(site_name)
        gt_sensor_names.append(sensor_name)
        body_ids.append(body_id)
        geom_ids.append(geom_id)
        site_ids.append(site_id)

    return body_names, geom_names, site_names, gt_sensor_names, body_ids, geom_ids, site_ids


def camera_intrinsics(model: mujoco.MjModel, camera_id: int, width: int, height: int) -> np.ndarray:
    fovy_rad = np.deg2rad(model.cam_fovy[camera_id])
    fy = 0.5 * height / np.tan(0.5 * fovy_rad)
    fx = fy * (width / height)
    cx = 0.5 * (width - 1)
    cy = 0.5 * (height - 1)
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
    w, x, y, z = quat_wxyz
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def pair_distance_xy(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1[:2] - p2[:2]))


def sample_pose_xyyaw(
    rng: np.random.Generator,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    yaw_range: Tuple[float, float],
) -> np.ndarray:
    return np.array(
        [
            rng.uniform(x_range[0], x_range[1]),
            rng.uniform(y_range[0], y_range[1]),
            rng.uniform(yaw_range[0], yaw_range[1]),
        ],
        dtype=np.float32,
    )


def sample_layout(
    rng: np.random.Generator,
    args: argparse.Namespace,
    robot_pose_xyyaw: np.ndarray,
    default_goal_pos: np.ndarray,
    default_obs_pos: np.ndarray,
    active_mask: np.ndarray,
    obs_footprint_radius: np.ndarray,
) -> Tuple[bool, np.ndarray, np.ndarray]:
    active_indices = np.where(active_mask > 0)[0].tolist()
    inactive_indices = np.where(active_mask == 0)[0].tolist()
    if len(active_indices) == 0:
        return False, default_goal_pos.astype(np.float32), default_obs_pos.astype(np.float32)

    for _ in range(args.scene_layout_attempts):
        goal = np.array(
            [
                rng.uniform(args.goal_x_range[0], args.goal_x_range[1]),
                rng.uniform(args.goal_y_range[0], args.goal_y_range[1]),
                default_goal_pos[2],
            ],
            dtype=np.float32,
        )

        robot_xy = np.array([robot_pose_xyyaw[0], robot_pose_xyyaw[1]], dtype=np.float32)
        if pair_distance_xy(robot_xy, goal) < args.min_robot_goal_dist:
            continue

        base_active = np.zeros((len(active_indices), 3), dtype=np.float32)
        base_active[:, :2] = default_obs_pos[active_indices, :2]
        base_active[:, 2] = obs_footprint_radius[active_indices]
        ok, sampled_active_xyr, _ = sample_obstacles_adaptive(
            rng=rng,
            base_obstacles_xyr=base_active,
            start_xy=robot_xy,
            goal_xy=goal[:2],
            obs_x_range=tuple(args.obs_x_range),
            obs_y_range=tuple(args.obs_y_range),
            min_obs_obs_dist=args.min_obs_obs_dist,
            min_start_obs_dist=args.min_robot_obs_dist,
            min_goal_obs_dist=args.min_goal_obs_dist,
            attempts=args.scene_sample_attempts,
            require_global_path=bool(args.require_global_path),
            robot_radius=args.path_robot_radius,
            path_inflation_margin=args.path_inflation_margin,
            path_grid_resolution=args.path_grid_resolution,
            path_grid_padding=args.path_grid_padding,
            path_max_grid_cells=args.path_max_grid_cells,
            expand_schedule=tuple(args.sampler_expand_schedule),
            attempt_factors=tuple(args.sampler_attempt_factors),
            relax_schedule=tuple(args.sampler_relax_schedule),
        )
        if not ok:
            continue

        obs = default_obs_pos.copy()
        for local_idx, obs_idx in enumerate(active_indices):
            obs[obs_idx, :2] = sampled_active_xyr[local_idx, :2]

        for k, i in enumerate(inactive_indices):
            obs[i, :2] = np.array(
                [
                    args.inactive_obs_x + k * args.inactive_obs_stride,
                    args.inactive_obs_y + k * args.inactive_obs_stride,
                ],
                dtype=np.float32,
            )

        return True, goal, obs

    return False, default_goal_pos.astype(np.float32), default_obs_pos.astype(np.float32)


def expert_control_action(
    base_pos: np.ndarray,
    base_quat_wxyz: np.ndarray,
    goal_pos: np.ndarray,
    lidar_triplet: np.ndarray,
    touch_force: float,
    prev_action: np.ndarray,
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    yaw = quat_wxyz_to_yaw(base_quat_wxyz)
    goal_vec = goal_pos[:2] - base_pos[:2]
    c = np.cos(yaw)
    s = np.sin(yaw)
    goal_rel_x = c * goal_vec[0] + s * goal_vec[1]
    goal_rel_y = -s * goal_vec[0] + c * goal_vec[1]
    goal_dist = float(np.linalg.norm(goal_vec))
    heading_err = float(np.arctan2(goal_rel_y, goal_rel_x))

    ranges = lidar_triplet.astype(np.float32).copy()
    invalid = (~np.isfinite(ranges)) | (ranges <= 0.0)
    ranges[invalid] = args.expert_default_far
    left, front, right = float(ranges[0]), float(ranges[1]), float(ranges[2])

    v_cmd = min(args.expert_v_max, args.expert_k_v * goal_dist)
    if abs(heading_err) > args.expert_heading_slowdown:
        v_cmd *= 0.45
    if front < args.expert_front_slow_dist:
        den = max(args.expert_front_slow_dist - args.expert_front_stop_dist, 1e-6)
        scale = np.clip((front - args.expert_front_stop_dist) / den, 0.0, 1.0)
        v_cmd *= max(0.1, float(scale))

    w_goal = args.expert_k_heading * heading_err
    w_obs = args.expert_k_obs * ((1.0 / max(right, 1e-3)) - (1.0 / max(left, 1e-3)))
    w_cmd = w_goal + w_obs

    if front < args.expert_front_stop_dist:
        v_cmd = args.expert_escape_speed
        turn_sign = 1.0 if right > left else -1.0
        w_cmd = turn_sign * args.expert_escape_turn_rate

    if touch_force > args.expert_touch_threshold:
        v_cmd = args.expert_touch_backoff_speed
        turn_sign = 1.0 if right > left else -1.0
        w_cmd = turn_sign * args.expert_touch_turn_rate

    w_cmd = float(np.clip(w_cmd, -args.expert_w_max, args.expert_w_max))

    action = diff_drive_inverse(
        v=float(v_cmd),
        w=float(w_cmd),
        wheel_radius=float(args.wheel_radius),
        wheel_base=float(args.wheel_base),
        drive_sign=float(args.drive_sign),
    ).astype(np.float32)
    action = np.clip(action, ctrl_low, ctrl_high)

    alpha = float(np.clip(args.expert_action_smoothing, 0.0, 1.0))
    action = (1.0 - alpha) * prev_action + alpha * action
    action = np.clip(action, ctrl_low, ctrl_high).astype(np.float32)
    return action


def init_storage(
    total_steps: int,
    episodes: int,
    ctrl_dim: int,
    num_obstacles: int,
    width: int,
    height: int,
    with_depth: bool,
) -> Dict[str, np.ndarray]:
    storage: Dict[str, np.ndarray] = {
        "episode": np.zeros((total_steps,), dtype=np.int32),
        "step": np.zeros((total_steps,), dtype=np.int32),
        "sim_time": np.zeros((total_steps,), dtype=np.float64),
        "ctrl_applied": np.zeros((total_steps, ctrl_dim), dtype=np.float32),
        "wheel_pos": np.zeros((total_steps, 2), dtype=np.float32),
        "wheel_vel": np.zeros((total_steps, 2), dtype=np.float32),
        "imu_acc": np.zeros((total_steps, 3), dtype=np.float32),
        "imu_gyro": np.zeros((total_steps, 3), dtype=np.float32),
        "touch_front_force": np.zeros((total_steps,), dtype=np.float32),
        "lidar_triplet": np.zeros((total_steps, 3), dtype=np.float32),
        "goal_rel_body": np.zeros((total_steps, 2), dtype=np.float32),
        "goal_dist": np.zeros((total_steps,), dtype=np.float32),
        "goal_heading_err": np.zeros((total_steps,), dtype=np.float32),
        "depth_sector_min": np.zeros((total_steps, 3), dtype=np.float32),
        "front_clearance": np.zeros((total_steps,), dtype=np.float32),
        "corridor_width": np.zeros((total_steps,), dtype=np.float32),
        "sensor_collision_flag": np.zeros((total_steps,), dtype=np.float32),
        "recover_trigger": np.zeros((total_steps,), dtype=np.uint8),
        "base_pos_gt": np.zeros((total_steps, 3), dtype=np.float32),
        "base_quat_gt": np.zeros((total_steps, 4), dtype=np.float32),
        "base_linvel_gt": np.zeros((total_steps, 3), dtype=np.float32),
        "base_angvel_gt": np.zeros((total_steps, 3), dtype=np.float32),
        "camera_pos_gt": np.zeros((total_steps, 3), dtype=np.float32),
        "camera_quat_gt": np.zeros((total_steps, 4), dtype=np.float32),
        "goal_pos_gt": np.zeros((total_steps, 3), dtype=np.float32),
        "obs_pos_gt": np.zeros((total_steps, num_obstacles, 3), dtype=np.float32),
        "episode_robot_init_xyyaw": np.zeros((episodes, 3), dtype=np.float32),
        "episode_goal_pos": np.zeros((episodes, 3), dtype=np.float32),
        "episode_obs_pos": np.zeros((episodes, num_obstacles, 3), dtype=np.float32),
        "episode_obs_active_mask": np.zeros((episodes, num_obstacles), dtype=np.uint8),
        "episode_obs_count": np.zeros((episodes,), dtype=np.int32),
        "episode_obs_geom_size": np.zeros((episodes, num_obstacles, 3), dtype=np.float32),
        "episode_layout_success": np.zeros((episodes,), dtype=np.uint8),
    }
    if with_depth:
        storage["depth_image"] = np.zeros((total_steps, height, width), dtype=np.float32)
        storage["distance_image"] = np.zeros((total_steps, height, width), dtype=np.float32)
        storage["depth_valid_mask"] = np.zeros((total_steps, height, width), dtype=np.uint8)
    return storage


def collect_dataset(args: argparse.Namespace) -> None:
    renderer_backend = ensure_gl_backend_or_reexec(args)

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)

    ctrl_range = model.actuator_ctrlrange.copy()
    ctrl_low = ctrl_range[:, 0]
    ctrl_high = ctrl_range[:, 1]

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if camera_id < 0:
        raise KeyError(f"Camera '{CAMERA_NAME}' not found in model.")
    intrinsics = camera_intrinsics(model, camera_id, args.width, args.height)

    goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GOAL_BODY_NAME)
    (
        obs_body_names,
        obs_geom_names,
        obs_site_names,
        obs_sensor_names,
        obs_body_ids,
        obs_geom_ids,
        obs_site_ids,
    ) = discover_obstacles(model)
    if goal_body_id < 0:
        raise KeyError(f"Goal body '{GOAL_BODY_NAME}' not found in XML.")
    if len(obs_body_ids) < 3:
        raise KeyError(
            "Need at least 3 valid obstacle entities (body+geom+site+sensor). "
            f"Found {len(obs_body_ids)} from names {OBSTACLE_BODY_NAMES}."
        )

    default_goal_pos = model.body_pos[goal_body_id].copy()
    default_obs_pos = np.stack([model.body_pos[body_id].copy() for body_id in obs_body_ids], axis=0)
    default_obs_geom_size = np.stack([model.geom_size[g].copy() for g in obs_geom_ids], axis=0)
    default_obs_site_pos = np.stack([model.site_pos[s].copy() for s in obs_site_ids], axis=0)

    qpos_default = model.qpos0.copy()
    robot_default_xyyaw = np.array(
        [
            qpos_default[0],
            qpos_default[1],
            quat_wxyz_to_yaw(qpos_default[3:7]),
        ],
        dtype=np.float32,
    )

    renderer = None
    if not args.no_depth:
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        print(f"Depth renderer backend: {renderer_backend}")

    total_steps = args.episodes * args.horizon
    num_obstacles = len(obs_body_ids)
    print(
        f"Obstacle set ({num_obstacles}): "
        + ", ".join([f"{bn}/{sn}" for bn, sn in zip(obs_body_names, obs_sensor_names)])
    )
    ds = init_storage(
        total_steps=total_steps,
        episodes=args.episodes,
        ctrl_dim=model.nu,
        num_obstacles=num_obstacles,
        width=args.width,
        height=args.height,
        with_depth=(renderer is not None),
    )

    index = 0
    for ep in range(args.episodes):
        active_max = min(args.max_active_obstacles, len(obs_body_ids))
        active_min = min(max(1, args.min_active_obstacles), active_max)
        if args.fixed_scene:
            active_count = len(obs_body_ids)
        else:
            active_count = int(rng.integers(active_min, active_max + 1))
        active_mask = np.zeros((len(obs_body_ids),), dtype=np.uint8)
        chosen = rng.choice(len(obs_body_ids), size=active_count, replace=False)
        active_mask[chosen] = 1

        obs_geom_size = default_obs_geom_size.copy()
        obs_pose_template = default_obs_pos.copy()
        obs_footprint_radius = np.zeros((len(obs_body_ids),), dtype=np.float32)
        for i, geom_id in enumerate(obs_geom_ids):
            gtype = int(model.geom_type[geom_id])
            if active_mask[i] and not args.fixed_scene:
                xy_scale = float(rng.uniform(args.obs_size_xy_scale_range[0], args.obs_size_xy_scale_range[1]))
                z_scale = float(rng.uniform(args.obs_size_z_scale_range[0], args.obs_size_z_scale_range[1]))
            else:
                xy_scale = 1.0
                z_scale = 1.0

            size_vec = default_obs_geom_size[i].copy()
            if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                size_vec[0] *= xy_scale
                size_vec[1] *= z_scale
                size_vec[2] = default_obs_geom_size[i, 2]
                obs_footprint_radius[i] = np.float32(size_vec[0])
                obstacle_half_height = float(size_vec[1])
            elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                size_vec[0] *= xy_scale
                size_vec[1] *= xy_scale
                size_vec[2] *= z_scale
                obs_footprint_radius[i] = np.float32(np.sqrt(size_vec[0] ** 2 + size_vec[1] ** 2))
                obstacle_half_height = float(size_vec[2])
            elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                size_vec[0] *= xy_scale
                size_vec[1] = default_obs_geom_size[i, 1]
                size_vec[2] = default_obs_geom_size[i, 2]
                obs_footprint_radius[i] = np.float32(size_vec[0])
                obstacle_half_height = float(size_vec[0])
            else:
                obs_footprint_radius[i] = np.float32(max(size_vec[0], size_vec[1], size_vec[2]))
                obstacle_half_height = float(default_obs_pos[i, 2])

            model.geom_size[geom_id] = size_vec
            model.site_pos[obs_site_ids[i], :2] = default_obs_site_pos[i, :2]
            model.site_pos[obs_site_ids[i], 2] = obstacle_half_height
            obs_geom_size[i] = size_vec
            obs_pose_template[i, 2] = obstacle_half_height

        if args.fixed_scene:
            robot_pose_xyyaw = robot_default_xyyaw.copy()
            goal_pos = default_goal_pos.astype(np.float32)
            obs_pos = obs_pose_template.astype(np.float32)
            layout_success = True
        else:
            layout_success = False
            robot_pose_xyyaw = robot_default_xyyaw.copy()
            goal_pos = default_goal_pos.astype(np.float32)
            obs_pos = obs_pose_template.astype(np.float32)
            for _ in range(max(1, args.episode_layout_retries)):
                sampled_robot = sample_pose_xyyaw(
                    rng,
                    tuple(args.robot_x_range),
                    tuple(args.robot_y_range),
                    tuple(args.robot_yaw_range),
                )
                success, sampled_goal, sampled_obs = sample_layout(
                    rng=rng,
                    args=args,
                    robot_pose_xyyaw=sampled_robot,
                    default_goal_pos=default_goal_pos,
                    default_obs_pos=obs_pose_template,
                    active_mask=active_mask,
                    obs_footprint_radius=obs_footprint_radius,
                )
                if success:
                    robot_pose_xyyaw = sampled_robot
                    goal_pos = sampled_goal
                    obs_pos = sampled_obs
                    layout_success = True
                    break
            if not layout_success:
                raise RuntimeError(
                    f"Episode {ep}: failed to sample passable layout after "
                    f"{args.episode_layout_retries} retries x {args.scene_layout_attempts} scene attempts."
                )

        model.body_pos[goal_body_id] = goal_pos
        for obs_i, body_id in enumerate(obs_body_ids):
            model.body_pos[body_id] = obs_pos[obs_i]

        mujoco.mj_resetData(model, data)
        data.qpos[:] = qpos_default
        data.qvel[:] = 0.0
        data.qpos[0] = float(robot_pose_xyyaw[0])
        data.qpos[1] = float(robot_pose_xyyaw[1])
        data.qpos[3:7] = yaw_to_quat_wxyz(float(robot_pose_xyyaw[2]))
        mujoco.mj_forward(model, data)

        ds["episode_robot_init_xyyaw"][ep] = robot_pose_xyyaw
        ds["episode_goal_pos"][ep] = goal_pos
        ds["episode_obs_pos"][ep] = obs_pos
        ds["episode_obs_active_mask"][ep] = active_mask
        ds["episode_obs_count"][ep] = np.int32(np.sum(active_mask))
        ds["episode_obs_geom_size"][ep] = obs_geom_size
        ds["episode_layout_success"][ep] = np.uint8(layout_success)

        held_action = rng.uniform(ctrl_low, ctrl_high).astype(np.float32)
        for step in range(args.horizon):
            if args.control_mode == "iid":
                action = rng.uniform(ctrl_low, ctrl_high).astype(np.float32)
            elif args.control_mode == "expert":
                base_pos_now = sensor_value(model, data, "base_pos_gt").astype(np.float32)
                base_quat_now = sensor_value(model, data, "base_quat_gt").astype(np.float32)
                goal_pos_now = sensor_value(model, data, "goal_pos_gt").astype(np.float32)
                lidar_now = np.array(
                    [
                        sensor_value(model, data, "lidar_left30")[0],
                        sensor_value(model, data, "lidar_front")[0],
                        sensor_value(model, data, "lidar_right30")[0],
                    ],
                    dtype=np.float32,
                )
                touch_now = float(sensor_value(model, data, "touch_front_force")[0])
                action = expert_control_action(
                    base_pos=base_pos_now,
                    base_quat_wxyz=base_quat_now,
                    goal_pos=goal_pos_now,
                    lidar_triplet=lidar_now,
                    touch_force=touch_now,
                    prev_action=held_action,
                    ctrl_low=ctrl_low,
                    ctrl_high=ctrl_high,
                    args=args,
                )
                held_action = action
            else:
                if step == 0:
                    action = held_action
                else:
                    if step % max(1, args.ctrl_hold_steps) == 0:
                        span = ctrl_high - ctrl_low
                        noise = rng.normal(0.0, args.ctrl_noise_std, size=model.nu).astype(np.float32)
                        held_action = np.clip(held_action + noise * span, ctrl_low, ctrl_high).astype(np.float32)
                        if rng.uniform() < args.ctrl_resample_prob:
                            held_action = rng.uniform(ctrl_low, ctrl_high).astype(np.float32)
                    action = held_action
            data.ctrl[:] = action
            mujoco.mj_step(model, data)

            ds["episode"][index] = ep
            ds["step"][index] = step
            ds["sim_time"][index] = data.time
            ds["ctrl_applied"][index] = action

            ds["wheel_pos"][index] = np.array(
                [
                    sensor_value(model, data, "wheel_l_pos")[0],
                    sensor_value(model, data, "wheel_r_pos")[0],
                ],
                dtype=np.float32,
            )
            ds["wheel_vel"][index] = np.array(
                [
                    sensor_value(model, data, "wheel_l_vel")[0],
                    sensor_value(model, data, "wheel_r_vel")[0],
                ],
                dtype=np.float32,
            )
            ds["imu_acc"][index] = sensor_value(model, data, "imu_acc").astype(np.float32)
            ds["imu_gyro"][index] = sensor_value(model, data, "imu_gyro").astype(np.float32)
            ds["touch_front_force"][index] = float(sensor_value(model, data, "touch_front_force")[0])
            ds["lidar_triplet"][index] = np.array(
                [
                    sensor_value(model, data, "lidar_left30")[0],
                    sensor_value(model, data, "lidar_front")[0],
                    sensor_value(model, data, "lidar_right30")[0],
                ],
                dtype=np.float32,
            )

            ds["base_pos_gt"][index] = sensor_value(model, data, "base_pos_gt").astype(np.float32)
            ds["base_quat_gt"][index] = sensor_value(model, data, "base_quat_gt").astype(np.float32)
            ds["base_linvel_gt"][index] = sensor_value(model, data, "base_linvel_gt").astype(np.float32)
            ds["base_angvel_gt"][index] = sensor_value(model, data, "base_angvel_gt").astype(np.float32)
            ds["camera_pos_gt"][index] = sensor_value(model, data, "camera_pos_gt").astype(np.float32)
            ds["camera_quat_gt"][index] = sensor_value(model, data, "camera_quat_gt").astype(np.float32)
            ds["goal_pos_gt"][index] = sensor_value(model, data, "goal_pos_gt").astype(np.float32)
            for obs_i, sensor_name in enumerate(obs_sensor_names):
                ds["obs_pos_gt"][index, obs_i] = sensor_value(model, data, sensor_name).astype(np.float32)

            base_pos_now = ds["base_pos_gt"][index]
            base_quat_now = ds["base_quat_gt"][index]
            goal_pos_now = ds["goal_pos_gt"][index]
            goal_vec = goal_pos_now[:2] - base_pos_now[:2]
            yaw_now = quat_wxyz_to_yaw(base_quat_now)
            c = np.cos(yaw_now)
            s = np.sin(yaw_now)
            goal_rel_x = c * goal_vec[0] + s * goal_vec[1]
            goal_rel_y = -s * goal_vec[0] + c * goal_vec[1]
            goal_rel = np.array([goal_rel_x, goal_rel_y], dtype=np.float32)
            goal_dist = float(np.linalg.norm(goal_vec))
            goal_heading_err = float(np.arctan2(goal_rel_y, goal_rel_x))
            ds["goal_rel_body"][index] = goal_rel
            ds["goal_dist"][index] = goal_dist
            ds["goal_heading_err"][index] = goal_heading_err

            depth_sector_min = ds["lidar_triplet"][index].astype(np.float32).copy()
            invalid = ~np.isfinite(depth_sector_min) | (depth_sector_min <= 0.0)
            depth_sector_min[invalid] = float(args.expert_default_far)

            if renderer is not None:
                renderer.update_scene(data, camera=CAMERA_NAME)
                renderer.enable_depth_rendering()
                depth = renderer.render().astype(np.float32)
                renderer.disable_depth_rendering()
                distance = depth.copy()
                valid = np.isfinite(depth) & (depth > 0.0)
                ds["depth_image"][index] = depth
                ds["distance_image"][index] = distance
                ds["depth_valid_mask"][index] = valid.astype(np.uint8)
                if np.any(valid):
                    cols = depth.shape[1]
                    i0 = int(cols * 0.0)
                    i1 = int(cols * (1.0 / 3.0))
                    i2 = int(cols * (2.0 / 3.0))
                    i3 = cols
                    d_left = depth[:, i0:i1]
                    d_front = depth[:, i1:i2]
                    d_right = depth[:, i2:i3]
                    if np.any(np.isfinite(d_left) & (d_left > 0.0)):
                        depth_sector_min[0] = min(
                            float(depth_sector_min[0]),
                            float(np.nanmin(d_left[np.isfinite(d_left) & (d_left > 0.0)])),
                        )
                    if np.any(np.isfinite(d_front) & (d_front > 0.0)):
                        depth_sector_min[1] = min(
                            float(depth_sector_min[1]),
                            float(np.nanmin(d_front[np.isfinite(d_front) & (d_front > 0.0)])),
                        )
                    if np.any(np.isfinite(d_right) & (d_right > 0.0)):
                        depth_sector_min[2] = min(
                            float(depth_sector_min[2]),
                            float(np.nanmin(d_right[np.isfinite(d_right) & (d_right > 0.0)])),
                        )

            front_clear = float(depth_sector_min[1])
            corridor_width = float(np.clip(depth_sector_min[0] + depth_sector_min[2], 0.0, 2.0 * args.expert_default_far))
            coll_flag = float(1.0 if float(np.min(depth_sector_min)) < float(args.sensor_collision_threshold) else 0.0)
            recover_flag = (
                (float(ds["touch_front_force"][index]) > float(args.expert_touch_threshold))
                or (front_clear < float(args.recover_front_threshold))
            )
            ds["depth_sector_min"][index] = depth_sector_min
            ds["front_clearance"][index] = front_clear
            ds["corridor_width"][index] = corridor_width
            ds["sensor_collision_flag"][index] = coll_flag
            ds["recover_trigger"][index] = np.uint8(1 if recover_flag else 0)

            index += 1

    output_parent = args.output.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    gt_sensor_names = GT_SENSOR_NAMES + obs_sensor_names

    save_dict = {
        "meta__xml_path": np.array([str(args.xml)], dtype=object),
        "meta__episodes": np.array([args.episodes], dtype=np.int32),
        "meta__horizon": np.array([args.horizon], dtype=np.int32),
        "meta__timestep": np.array([model.opt.timestep], dtype=np.float32),
        "meta__camera_name": np.array([CAMERA_NAME], dtype=object),
        "meta__camera_intrinsics": intrinsics,
        "meta__control_mode": np.array([args.control_mode], dtype=object),
        "meta__ctrl_hold_steps": np.array([args.ctrl_hold_steps], dtype=np.int32),
        "meta__ctrl_noise_std": np.array([args.ctrl_noise_std], dtype=np.float32),
        "meta__ctrl_resample_prob": np.array([args.ctrl_resample_prob], dtype=np.float32),
        "meta__wheel_radius": np.array([args.wheel_radius], dtype=np.float32),
        "meta__wheel_base": np.array([args.wheel_base], dtype=np.float32),
        "meta__state_convention_version": np.array([STATE_CONVENTION_VERSION], dtype=object),
        "meta__drive_sign": np.array([args.drive_sign], dtype=np.float32),
        "meta__pose_source": np.array([args.pose_source], dtype=object),
        "meta__heading_source": np.array([args.heading_source], dtype=object),
        "meta__yaw_source": np.array([args.yaw_source], dtype=object),
        "meta__control_definition": np.array([CONTROL_DEFINITION], dtype=object),
        "meta__expert_k_v": np.array([args.expert_k_v], dtype=np.float32),
        "meta__expert_k_heading": np.array([args.expert_k_heading], dtype=np.float32),
        "meta__expert_k_obs": np.array([args.expert_k_obs], dtype=np.float32),
        "meta__expert_v_max": np.array([args.expert_v_max], dtype=np.float32),
        "meta__expert_w_max": np.array([args.expert_w_max], dtype=np.float32),
        "meta__sensor_collision_threshold": np.array([args.sensor_collision_threshold], dtype=np.float32),
        "meta__recover_front_threshold": np.array([args.recover_front_threshold], dtype=np.float32),
        "meta__min_active_obstacles": np.array([args.min_active_obstacles], dtype=np.int32),
        "meta__max_active_obstacles": np.array([args.max_active_obstacles], dtype=np.int32),
        "meta__obs_size_xy_scale_range": np.array(args.obs_size_xy_scale_range, dtype=np.float32),
        "meta__obs_size_z_scale_range": np.array(args.obs_size_z_scale_range, dtype=np.float32),
        "meta__require_global_path": np.array([int(args.require_global_path)], dtype=np.uint8),
        "meta__path_robot_radius": np.array([args.path_robot_radius], dtype=np.float32),
        "meta__path_inflation_margin": np.array([args.path_inflation_margin], dtype=np.float32),
        "meta__path_grid_resolution": np.array([args.path_grid_resolution], dtype=np.float32),
        "meta__path_grid_padding": np.array([args.path_grid_padding], dtype=np.float32),
        "meta__path_max_grid_cells": np.array([args.path_max_grid_cells], dtype=np.int32),
        "meta__sampler_expand_schedule": np.array(args.sampler_expand_schedule, dtype=np.float32),
        "meta__sampler_attempt_factors": np.array(args.sampler_attempt_factors, dtype=np.float32),
        "meta__sampler_relax_schedule": np.array(args.sampler_relax_schedule, dtype=np.float32),
        "meta__obstacle_body_names": np.array(obs_body_names, dtype=object),
        "meta__obstacle_geom_names": np.array(obs_geom_names, dtype=object),
        "meta__obstacle_site_names": np.array(obs_site_names, dtype=object),
        "meta__obstacle_sensor_names": np.array(obs_sensor_names, dtype=object),
        "meta__raw_sensor_names": np.array(RAW_SENSOR_NAMES, dtype=object),
        "meta__gt_sensor_names": np.array(gt_sensor_names, dtype=object),
        "meta__mujoco_gl_backend": np.array([renderer_backend], dtype=object),
        "meta__scene_randomized": np.array([0 if args.fixed_scene else 1], dtype=np.uint8),
        "meta__episode_robot_init_xyyaw": ds["episode_robot_init_xyyaw"],
        "meta__episode_goal_pos": ds["episode_goal_pos"],
        "meta__episode_obs_pos": ds["episode_obs_pos"],
        "meta__episode_obs_active_mask": ds["episode_obs_active_mask"],
        "meta__episode_obs_count": ds["episode_obs_count"],
        "meta__episode_obs_geom_size": ds["episode_obs_geom_size"],
        "meta__episode_layout_success": ds["episode_layout_success"],
        "raw__episode": ds["episode"],
        "raw__step": ds["step"],
        "raw__sim_time": ds["sim_time"],
        "raw__ctrl_applied": ds["ctrl_applied"],
        "raw__wheel_pos": ds["wheel_pos"],
        "raw__wheel_vel": ds["wheel_vel"],
        "raw__imu_acc": ds["imu_acc"],
        "raw__imu_gyro": ds["imu_gyro"],
        "raw__touch_front_force": ds["touch_front_force"],
        "raw__lidar_triplet": ds["lidar_triplet"],
        "raw__goal_rel_body": ds["goal_rel_body"],
        "raw__goal_dist": ds["goal_dist"],
        "raw__goal_heading_err": ds["goal_heading_err"],
        "raw__depth_sector_min": ds["depth_sector_min"],
        "raw__front_clearance": ds["front_clearance"],
        "raw__corridor_width": ds["corridor_width"],
        "raw__sensor_collision_flag": ds["sensor_collision_flag"],
        "raw__recover_trigger": ds["recover_trigger"],
        "gt__base_pos_gt": ds["base_pos_gt"],
        "gt__base_quat_gt": ds["base_quat_gt"],
        "gt__base_linvel_gt": ds["base_linvel_gt"],
        "gt__base_angvel_gt": ds["base_angvel_gt"],
        "gt__camera_pos_gt": ds["camera_pos_gt"],
        "gt__camera_quat_gt": ds["camera_quat_gt"],
        "gt__goal_pos_gt": ds["goal_pos_gt"],
        # Backward-compatible aliases for existing downstream scripts.
        "gt__target_pos_gt": ds["goal_pos_gt"],
        "gt__obs_pos_gt": ds["obs_pos_gt"],
    }
    for obs_i, sensor_name in enumerate(obs_sensor_names):
        save_dict[f"gt__{sensor_name}"] = ds["obs_pos_gt"][:, obs_i]
    save_dict.update(
        convention_as_meta(
            drive_sign=float(args.drive_sign),
            pose_source=str(args.pose_source),
            heading_source=str(args.heading_source),
            yaw_source=str(args.yaw_source),
        )
    )

    if renderer is not None:
        save_dict["raw__depth_image"] = ds["depth_image"]
        save_dict["raw__distance_image"] = ds["distance_image"]
        save_dict["raw__depth_valid_mask"] = ds["depth_valid_mask"]
        renderer.close()

    np.savez_compressed(args.output, **save_dict)
    print(f"Saved multilayer raw+gt dataset to: {args.output}")
    print(f"Total samples: {total_steps}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect MuJoCo raw sensor stream + groundtruth.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML, help="Path to MuJoCo XML.")
    parser.add_argument("--output", type=Path, default=Path("datasets/multilayer_raw_gt.npz"))
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--no-depth", action="store_true", help="Disable depth/distance image recording.")
    parser.add_argument(
        "--gl-backend",
        type=str,
        default="auto",
        choices=("auto", "glfw", "egl", "osmesa"),
        help="MuJoCo OpenGL backend for depth rendering. 'auto' tries backend fallback.",
    )
    parser.add_argument("--fixed-scene", action="store_true", help="Disable random scene generation.")
    parser.add_argument("--control-mode", type=str, default="random_walk", choices=("random_walk", "iid", "expert"))
    parser.add_argument("--ctrl-hold-steps", type=int, default=5)
    parser.add_argument("--ctrl-noise-std", type=float, default=0.08)
    parser.add_argument("--ctrl-resample-prob", type=float, default=0.12)
    parser.add_argument("--wheel-radius", type=float, default=0.085)
    parser.add_argument("--wheel-base", type=float, default=0.37)
    parser.add_argument("--drive-sign", type=float, default=-1.0)
    parser.add_argument("--pose-source", type=str, default="odom")
    parser.add_argument("--heading-source", type=str, default="base")
    parser.add_argument("--yaw-source", type=str, default="odom_yaw")
    parser.add_argument("--expert-default-far", type=float, default=10.0)
    parser.add_argument("--expert-k-v", type=float, default=0.9)
    parser.add_argument("--expert-k-heading", type=float, default=2.5)
    parser.add_argument("--expert-k-obs", type=float, default=1.2)
    parser.add_argument("--expert-v-max", type=float, default=1.0)
    parser.add_argument("--expert-w-max", type=float, default=2.8)
    parser.add_argument("--expert-heading-slowdown", type=float, default=0.55)
    parser.add_argument("--expert-front-slow-dist", type=float, default=1.2)
    parser.add_argument("--expert-front-stop-dist", type=float, default=0.55)
    parser.add_argument("--expert-escape-speed", type=float, default=-0.18)
    parser.add_argument("--expert-escape-turn-rate", type=float, default=1.8)
    parser.add_argument("--expert-touch-threshold", type=float, default=1e-5)
    parser.add_argument("--expert-touch-backoff-speed", type=float, default=-0.22)
    parser.add_argument("--expert-touch-turn-rate", type=float, default=2.2)
    parser.add_argument("--expert-action-smoothing", type=float, default=0.45)
    parser.add_argument("--sensor-collision-threshold", type=float, default=0.30)
    parser.add_argument("--recover-front-threshold", type=float, default=0.25)

    parser.add_argument("--scene-layout-attempts", type=int, default=120)
    parser.add_argument("--scene-sample-attempts", type=int, default=180)
    parser.add_argument("--episode-layout-retries", type=int, default=6)
    parser.add_argument("--min-active-obstacles", type=int, default=3)
    parser.add_argument("--max-active-obstacles", type=int, default=5)
    parser.add_argument("--obs-size-xy-scale-range", type=float, nargs=2, default=(0.7, 1.5))
    parser.add_argument("--obs-size-z-scale-range", type=float, nargs=2, default=(0.7, 1.4))
    parser.add_argument("--require-global-path", action="store_true", default=True)
    parser.add_argument("--no-require-global-path", action="store_false", dest="require_global_path")
    parser.add_argument("--path-robot-radius", type=float, default=0.28)
    parser.add_argument("--path-inflation-margin", type=float, default=0.10)
    parser.add_argument("--path-grid-resolution", type=float, default=0.10)
    parser.add_argument("--path-grid-padding", type=float, default=0.80)
    parser.add_argument("--path-max-grid-cells", type=int, default=240000)
    parser.add_argument(
        "--sampler-expand-schedule",
        type=float,
        nargs="+",
        default=(0.0,),
        help="Adaptive sampler x/y range expansion schedule. Default 0 keeps strict uniform area.",
    )
    parser.add_argument(
        "--sampler-attempt-factors",
        type=float,
        nargs="+",
        default=(1.0,),
        help="Adaptive sampler attempts multiplier per stage.",
    )
    parser.add_argument(
        "--sampler-relax-schedule",
        type=float,
        nargs="+",
        default=(1.0,),
        help="Constraint relax multiplier per stage.",
    )
    parser.add_argument("--inactive-obs-x", type=float, default=8.0)
    parser.add_argument("--inactive-obs-y", type=float, default=8.0)
    parser.add_argument("--inactive-obs-stride", type=float, default=1.5)
    parser.add_argument("--robot-x-range", type=float, nargs=2, default=(-0.8, 0.8))
    parser.add_argument("--robot-y-range", type=float, nargs=2, default=(-0.8, 0.8))
    parser.add_argument("--robot-yaw-range", type=float, nargs=2, default=(-np.pi, np.pi))
    parser.add_argument("--goal-x-range", type=float, nargs=2, default=(1.0, 3.5))
    parser.add_argument("--goal-y-range", type=float, nargs=2, default=(-2.0, 2.0))
    parser.add_argument("--obs-x-range", type=float, nargs=2, default=(0.4, 3.0))
    parser.add_argument("--obs-y-range", type=float, nargs=2, default=(-2.0, 2.0))
    parser.add_argument("--min-robot-goal-dist", type=float, default=1.0)
    parser.add_argument("--min-robot-obs-dist", type=float, default=0.8)
    parser.add_argument("--min-goal-obs-dist", type=float, default=0.8)
    parser.add_argument("--min-obs-obs-dist", type=float, default=0.6)
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    args.drive_sign = canonical_drive_sign(args.drive_sign)
    collect_dataset(args)
