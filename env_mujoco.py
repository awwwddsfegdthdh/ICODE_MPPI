import re
from typing import Dict, Optional

import mujoco
import numpy as np
from state_convention import (
    StateConvention,
    assert_state_contract,
    body_to_world,
    canonical_drive_sign,
    diff_drive_forward,
    world_to_body,
)


class CarEnv:
    def __init__(self, xml_path="car_scene.xml"):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self.get_state()
        
    def step(self, action):
        # action: [forward_force, turning_torque]
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        return self.get_state()
        
    def get_state(self):
        # 返回状态: [x, y, theta, v_x, v_y, omega]
        qpos = self.data.qpos.copy() # [x, y, theta]
        qvel = self.data.qvel.copy() # [vx, vy, omega]
        
        x, y = qpos[0], qpos[1]
        theta = qpos[2]
        
        vx, vy = qvel[0], qvel[1]
        omega = qvel[2]
        
        return np.array([x, y, theta, vx, vy, omega])


class E1RobotEnv:
    """
    E1 MuJoCo wrapper that exposes:
    - control step with decimation to match model dt (~0.02)
    - 7D state estimate used by ICODE/MPPI:
      [x_odom, y_odom, psi_odom, v_body, wz_body, dqL, dqR]
    """

    def __init__(
        self,
        xml_path: str,
        wheel_radius: float = 0.085,
        wheel_base: float = 0.37,
        drive_sign: float = -1.0,
        yaw_blend_alpha: float = 0.35,
        control_decimation: int = 10,
        pose_source: str = "odom",
        heading_source: str = "base",
    ):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.wheel_radius = float(wheel_radius)
        self.wheel_base = float(wheel_base)
        self.drive_sign = canonical_drive_sign(float(drive_sign))
        self.state_convention = StateConvention()
        self.yaw_blend_alpha = float(yaw_blend_alpha)
        self.control_decimation = max(1, int(control_decimation))
        self.pose_source = str(pose_source).lower()
        if self.pose_source not in ("odom", "gt"):
            raise ValueError(f"pose_source must be 'odom' or 'gt', got: {pose_source}")
        self.heading_source = str(heading_source).lower()
        if self.heading_source not in ("base", "camera"):
            raise ValueError(f"heading_source must be 'base' or 'camera', got: {heading_source}")
        if self.pose_source == "gt" and self.heading_source == "camera":
            print("[warn] pose_source=gt + heading_source=camera may mix coordinate semantics.")

        self.ctrl_low = self.model.actuator_ctrlrange[:, 0].copy().astype(np.float32)
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1].copy().astype(np.float32)

        obs_body_candidates = ("obs_front", "obs_left", "obs_right", "obs_mid_left", "obs_mid_right")
        self._obstacle_specs = []
        for bname in obs_body_candidates:
            gname = f"{bname}_geom"
            sname = f"{bname}_pos_gt"
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, gname)
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            if bid >= 0 and gid >= 0 and sid >= 0:
                self._obstacle_specs.append(
                    {
                        "body_name": bname,
                        "body_id": int(bid),
                        "geom_id": int(gid),
                        "sensor_name": sname,
                    }
                )
        if len(self._obstacle_specs) == 0:
            raise KeyError(
                f"No obstacle triplets found in '{xml_path}'. "
                "Expected body/geom/sensor like obs_front, obs_front_geom, obs_front_pos_gt."
            )
        self._obs_geom_ids = [s["geom_id"] for s in self._obstacle_specs]
        self._obs_body_ids = [s["body_id"] for s in self._obstacle_specs]
        self._goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
        self._sensor_names = {
            "wheel_l_vel": "wheel_l_vel",
            "wheel_r_vel": "wheel_r_vel",
            "imu_gyro": "imu_gyro",
            "base_pos_gt": "base_pos_gt",
            "base_quat_gt": "base_quat_gt",
            "camera_quat_gt": "camera_quat_gt",
            "goal_pos_gt": "goal_pos_gt",
            "touch_front_force": "touch_front_force",
        }
        for spec in self._obstacle_specs:
            sname = spec["sensor_name"]
            self._sensor_names[sname] = sname
        self._sensor_meta = {}
        for k, name in self._sensor_names.items():
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise KeyError(f"Sensor '{name}' not found in model '{xml_path}'.")
            self._sensor_meta[k] = (self.model.sensor_adr[sid], self.model.sensor_dim[sid])
        self._register_optional_lidar_sensors()
        self._lidar_specs = self._discover_lidar_specs()

        self._odom_x = 0.0
        self._odom_y = 0.0
        self._odom_psi = 0.0
        self._v_body = 0.0
        self._wz_body = 0.0
        self._dq_l = 0.0
        self._dq_r = 0.0
        self._last_time = 0.0
        self._heading_offset = 0.0
        self._state_est = np.zeros((7,), dtype=np.float32)

    def _sensor(self, key: str) -> np.ndarray:
        if key not in self._sensor_meta:
            raise KeyError(f"Sensor key not available: {key}")
        adr, dim = self._sensor_meta[key]
        return self.data.sensordata[adr : adr + dim].copy()

    def _register_optional_lidar_sensors(self) -> None:
        for sid in range(int(self.model.nsensor)):
            if int(self.model.sensor_type[sid]) != int(mujoco.mjtSensor.mjSENS_RANGEFINDER):
                continue
            sname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, sid)
            if (sname is None) or (not str(sname).startswith("lidar_")):
                continue
            key = str(sname)
            if key not in self._sensor_meta:
                self._sensor_meta[key] = (self.model.sensor_adr[sid], self.model.sensor_dim[sid])

    @staticmethod
    def _wrap_to_pi_deg(a: float) -> float:
        return float((float(a) + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _infer_lidar_angle_deg(sensor_name: str) -> Optional[float]:
        name = str(sensor_name).strip().lower()
        alias = {
            "lidar_front": 0.0,
            "lidar_back": 180.0,
            "lidar_left30": 30.0,
            "lidar_left60": 60.0,
            "lidar_right30": -30.0,
            "lidar_right60": -60.0,
            "lidar_back_left30": 150.0,
            "lidar_back_left60": 120.0,
            "lidar_back_right30": -150.0,
            "lidar_back_right60": -120.0,
        }
        if name in alias:
            return float(alias[name])

        def _try_float(x: str) -> Optional[float]:
            try:
                return float(x)
            except ValueError:
                return None

        if not name.startswith("lidar_"):
            return None
        token = name[len("lidar_") :]
        m = re.match(r"^(left|right)([-+]?\d+(\.\d+)?)$", token)
        if m is not None:
            side = m.group(1)
            v = _try_float(m.group(2))
            if v is None:
                return None
            if side == "left":
                return float(v) if v >= 0.0 else float(180.0 + v)
            return float(-v) if v >= 0.0 else float(-180.0 - v)
        m = re.match(r"^back_(left|right)([-+]?\d+(\.\d+)?)$", token)
        if m is not None:
            side = m.group(1)
            v = _try_float(m.group(2))
            if v is None:
                return None
            av = abs(float(v))
            if side == "left":
                return float(180.0 - av)
            return float(-180.0 + av)
        return None

    def _discover_lidar_specs(self) -> list[dict[str, float]]:
        specs: list[dict[str, float]] = []
        for key in sorted(self._sensor_meta.keys()):
            if not str(key).startswith("lidar_"):
                continue
            ang = self._infer_lidar_angle_deg(str(key))
            if ang is None:
                continue
            specs.append({"name": str(key), "angle_deg": self._wrap_to_pi_deg(float(ang))})
        if len(specs) <= 0:
            fallback = (
                ("lidar_left30", 30.0),
                ("lidar_front", 0.0),
                ("lidar_right30", -30.0),
            )
            for k, a in fallback:
                if k in self._sensor_meta:
                    specs.append({"name": str(k), "angle_deg": float(a)})
        specs.sort(key=lambda s: float(s["angle_deg"]))
        return specs

    @staticmethod
    def _quat_wxyz_to_yaw(quat_wxyz: np.ndarray) -> float:
        w, x, y, z = quat_wxyz
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return float(np.arctan2(siny_cosp, cosy_cosp))

    @staticmethod
    def _wrap_to_pi(a: float) -> float:
        return float((a + np.pi) % (2.0 * np.pi) - np.pi)

    def _compute_heading_offset(self) -> float:
        if self.heading_source != "camera":
            return 0.0
        try:
            base_quat = self._sensor("base_quat_gt")
            cam_quat = self._sensor("camera_quat_gt")
        except KeyError:
            return 0.0
        base_yaw = self._quat_wxyz_to_yaw(base_quat)
        cam_yaw = self._quat_wxyz_to_yaw(cam_quat)
        return self._wrap_to_pi(cam_yaw - base_yaw)

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        return self._reset_common()

    def reset_with_pose(self, x: float, y: float, yaw: float) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = float(x)
        self.data.qpos[1] = float(y)
        half = 0.5 * float(yaw)
        quat_wxyz = np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)
        self.data.qpos[3:7] = quat_wxyz
        self.data.qvel[:] = 0.0
        return self._reset_common()

    def _reset_common(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)

        base_pos = self._sensor("base_pos_gt")
        base_quat = self._sensor("base_quat_gt")
        self._heading_offset = self._compute_heading_offset()
        self._odom_x = float(base_pos[0])
        self._odom_y = float(base_pos[1])
        base_yaw = self._quat_wxyz_to_yaw(base_quat)
        self._odom_psi = self._wrap_to_pi(base_yaw + self._heading_offset)
        self._v_body = 0.0
        self._wz_body = 0.0
        self._dq_l = 0.0
        self._dq_r = 0.0
        self._last_time = float(self.data.time)
        self._state_est[:] = np.array(
            [self._odom_x, self._odom_y, self._odom_psi, self._v_body, self._wz_body, self._dq_l, self._dq_r],
            dtype=np.float32,
        )
        return self._state_est.copy()

    def _update_state_estimate(self) -> np.ndarray:
        t_now = float(self.data.time)
        dt = max(0.0, t_now - self._last_time)
        self._last_time = t_now

        dq_l = float(self._sensor("wheel_l_vel")[0])
        dq_r = float(self._sensor("wheel_r_vel")[0])
        imu_gyro = self._sensor("imu_gyro")
        imu_wz = float(imu_gyro[2]) if imu_gyro.shape[0] >= 3 else 0.0

        vw = diff_drive_forward(
            np.array([dq_l, dq_r], dtype=np.float32),
            wheel_radius=self.wheel_radius,
            wheel_base=self.wheel_base,
            drive_sign=self.drive_sign,
        )
        v = float(vw[0])
        wz_wheel = float(vw[1])
        wz = self.yaw_blend_alpha * wz_wheel + (1.0 - self.yaw_blend_alpha) * imu_wz

        if self.pose_source == "gt":
            base_pos = self._sensor("base_pos_gt")
            base_quat = self._sensor("base_quat_gt")
            self._odom_x = float(base_pos[0])
            self._odom_y = float(base_pos[1])
            base_yaw = self._quat_wxyz_to_yaw(base_quat)
            self._odom_psi = self._wrap_to_pi(base_yaw + self._heading_offset)
        else:
            self._odom_x += v * np.cos(self._odom_psi) * dt
            self._odom_y += v * np.sin(self._odom_psi) * dt
            self._odom_psi = self._wrap_to_pi(self._odom_psi + wz * dt)

        self._v_body = v
        self._wz_body = wz
        self._dq_l = dq_l
        self._dq_r = dq_r

        self._state_est[:] = np.array(
            [self._odom_x, self._odom_y, self._odom_psi, self._v_body, self._wz_body, self._dq_l, self._dq_r],
            dtype=np.float32,
        )
        return self._state_est.copy()

    def step(self, action: np.ndarray) -> np.ndarray:
        act = np.asarray(action, dtype=np.float32)
        act = np.clip(act, self.ctrl_low, self.ctrl_high)
        for _ in range(self.control_decimation):
            self.data.ctrl[:] = act
            mujoco.mj_step(self.model, self.data)
        return self._update_state_estimate()

    def get_state(self) -> np.ndarray:
        return self._state_est.copy()

    def get_goal_xy(self) -> np.ndarray:
        g = self._sensor("goal_pos_gt")
        return g[:2].astype(np.float32)

    def set_goal_xy(self, x: float, y: float) -> bool:
        if self._goal_body_id < 0:
            return False
        self.model.body_pos[self._goal_body_id, 0] = float(x)
        self.model.body_pos[self._goal_body_id, 1] = float(y)
        mujoco.mj_forward(self.model, self.data)
        return True

    def set_obstacles_xy(self, obs_xy: np.ndarray) -> bool:
        arr = np.asarray(obs_xy, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] != len(self._obs_body_ids):
            raise ValueError(
                f"obs_xy must have shape ({len(self._obs_body_ids)}, 2), got {arr.shape}"
            )
        if any(bid < 0 for bid in self._obs_body_ids):
            return False
        for i, bid in enumerate(self._obs_body_ids):
            self.model.body_pos[bid, 0] = float(arr[i, 0])
            self.model.body_pos[bid, 1] = float(arr[i, 1])
        mujoco.mj_forward(self.model, self.data)
        return True

    def set_obstacles_radii(self, obs_radii: np.ndarray) -> bool:
        arr = np.asarray(obs_radii, dtype=np.float32).reshape(-1)
        if arr.shape[0] != len(self._obs_geom_ids):
            raise ValueError(
                f"obs_radii must have shape ({len(self._obs_geom_ids)},), got {arr.shape}"
            )
        if any(gid < 0 for gid in self._obs_geom_ids):
            return False
        for i, gid in enumerate(self._obs_geom_ids):
            r = float(max(1e-3, arr[i]))
            gtype = int(self.model.geom_type[gid])
            if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                self.model.geom_size[gid, 0] = r
            elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                self.model.geom_size[gid, 0] = r
            elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                # Keep circumscribed radius approximately equal to r.
                half = r / np.sqrt(2.0)
                self.model.geom_size[gid, 0] = half
                self.model.geom_size[gid, 1] = half
            else:
                self.model.geom_size[gid, 0] = r
        mujoco.mj_forward(self.model, self.data)
        return True

    def get_obstacles_xyr(self) -> np.ndarray:
        obs = np.stack(
            [self._sensor(spec["sensor_name"]) for spec in self._obstacle_specs],
            axis=0,
        ).astype(np.float32)
        radii = []
        for gid in self._obs_geom_ids:
            gtype = int(self.model.geom_type[gid])
            gsize = self.model.geom_size[gid]
            if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                r = float(gsize[0])
            elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
                r = float(np.sqrt(gsize[0] ** 2 + gsize[1] ** 2))
            elif gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                r = float(gsize[0])
            else:
                r = float(max(gsize[0], gsize[1], gsize[2]))
            radii.append(r)
        return np.concatenate([obs[:, :2], np.asarray(radii, dtype=np.float32)[:, None]], axis=1)

    def get_base_xy_gt(self) -> np.ndarray:
        b = self._sensor("base_pos_gt")
        return b[:2].astype(np.float32)

    def get_touch_force(self) -> float:
        return float(self._sensor("touch_front_force")[0])

    @staticmethod
    def _sanitize_lidar_ranges(ranges: np.ndarray, default_far: float) -> np.ndarray:
        rr = np.asarray(ranges, dtype=np.float32).reshape(-1)
        if rr.size <= 0:
            return np.asarray([float(default_far)], dtype=np.float32)
        bad = (~np.isfinite(rr)) | (rr <= 0.0)
        rr[bad] = float(default_far)
        return rr

    def get_lidar_scan(
        self,
        default_far: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(self._lidar_specs) <= 0:
            return (
                np.asarray([float(default_far), float(default_far), float(default_far)], dtype=np.float32),
                np.asarray([30.0, 0.0, -30.0], dtype=np.float32),
            )
        vals = []
        angs = []
        for spec in self._lidar_specs:
            key = str(spec["name"])
            if key not in self._sensor_meta:
                continue
            vals.append(float(self._sensor(key)[0]))
            angs.append(float(spec["angle_deg"]))
        if len(vals) <= 0:
            return (
                np.asarray([float(default_far), float(default_far), float(default_far)], dtype=np.float32),
                np.asarray([30.0, 0.0, -30.0], dtype=np.float32),
            )
        ranges = self._sanitize_lidar_ranges(np.asarray(vals, dtype=np.float32), default_far=float(default_far))
        angles = np.asarray(angs, dtype=np.float32).reshape(-1)
        if angles.shape[0] != ranges.shape[0]:
            angles = np.linspace(-60.0, 60.0, num=ranges.shape[0], dtype=np.float32)
        return ranges.astype(np.float32), angles.astype(np.float32)

    def get_lidar_triplet(self, default_far: float = 10.0) -> np.ndarray:
        ranges, angles = self.get_lidar_scan(default_far=default_far)
        return self._triplet_from_scan(ranges=ranges, angles_deg=angles, default_far=default_far)

    @staticmethod
    def _triplet_from_scan(ranges: np.ndarray, angles_deg: np.ndarray, default_far: float) -> np.ndarray:
        rr = np.asarray(ranges, dtype=np.float32).reshape(-1)
        aa = np.asarray(angles_deg, dtype=np.float32).reshape(-1)
        if rr.shape[0] != aa.shape[0] or rr.shape[0] <= 0:
            return np.asarray([float(default_far), float(default_far), float(default_far)], dtype=np.float32)
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

    @staticmethod
    def _lidar_sector_min(
        ranges: np.ndarray,
        angles_deg: np.ndarray,
        default_far: float,
    ) -> tuple[float, float, float, float]:
        rr = np.asarray(ranges, dtype=np.float32).reshape(-1)
        aa = np.asarray(angles_deg, dtype=np.float32).reshape(-1)
        if rr.shape[0] != aa.shape[0] or rr.shape[0] <= 0:
            ff = float(default_far)
            return ff, ff, ff, ff
        ang = ((aa + 180.0) % 360.0) - 180.0
        # Keep side-risk semantics focused on the forward half-plane.
        # With omni lidar enabled, back-side beams should not directly
        # drive near-collision side gates during nominal forward motion.
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

    @staticmethod
    def _world_to_body(vec_xy: np.ndarray, yaw: float) -> np.ndarray:
        return world_to_body(np.asarray(vec_xy, dtype=np.float32), float(yaw))

    @staticmethod
    def _body_to_world(vec_xy: np.ndarray, yaw: float) -> np.ndarray:
        return body_to_world(np.asarray(vec_xy, dtype=np.float32), float(yaw))

    def extract_depth_features(
        self,
        depth_image: Optional[np.ndarray],
        lidar_ranges: np.ndarray,
        lidar_angles_deg: np.ndarray,
        default_far: float = 10.0,
        collision_threshold: float = 0.30,
        depth_yaw_offset_deg: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        lidar = self._sanitize_lidar_ranges(np.asarray(lidar_ranges, dtype=np.float32), default_far=float(default_far))
        lidar_ang = np.asarray(lidar_angles_deg, dtype=np.float32).reshape(-1)
        if lidar_ang.shape[0] != lidar.shape[0]:
            if lidar.shape[0] == 3:
                lidar_ang = np.asarray([30.0, 0.0, -30.0], dtype=np.float32)
            else:
                lidar_ang = np.linspace(-90.0, 90.0, num=lidar.shape[0], dtype=np.float32)

        left, front, right, rear = self._lidar_sector_min(
            ranges=lidar,
            angles_deg=lidar_ang,
            default_far=float(default_far),
        )
        if depth_image is not None:
            depth = np.asarray(depth_image, dtype=np.float32)
            if depth.ndim == 2 and depth.size > 0:
                valid = np.isfinite(depth) & (depth > 0.0)
                if np.any(valid):
                    h, w = depth.shape
                    row0 = int(h * 0.45)
                    row1 = int(h * 0.65)
                    band = depth[row0:row1, :]
                    if band.shape[0] > 0:
                        d_row = np.nanmedian(band, axis=0).astype(np.float32)
                        idx = np.where(np.isfinite(d_row) & (d_row > 0.0))[0]
                        if idx.size > 0:
                            hfov = np.deg2rad(86.0)
                            yaw_off = np.deg2rad(float(depth_yaw_offset_deg))
                            uu = (idx.astype(np.float32) / max(1.0, float(w - 1))) * 2.0 - 1.0
                            ang = 0.5 * hfov * uu + yaw_off
                            ang = ((ang + np.pi) % (2.0 * np.pi)) - np.pi
                            dep = d_row[idx]
                            left_mask = (ang >= np.deg2rad(25.0)) & (ang <= np.deg2rad(100.0))
                            front_mask = np.abs(ang) <= np.deg2rad(25.0)
                            right_mask = (ang <= -np.deg2rad(25.0)) & (ang >= -np.deg2rad(100.0))
                            if np.any(left_mask):
                                left = min(left, float(np.min(dep[left_mask])))
                            if np.any(front_mask):
                                front = min(front, float(np.min(dep[front_mask])))
                            if np.any(right_mask):
                                right = min(right, float(np.min(dep[right_mask])))

        sector_min = np.array([left, front, right], dtype=np.float32)
        corridor_width = float(np.clip(left + right, 0.0, 2.0 * float(default_far)))
        min_clearance = float(min(float(np.min(lidar)), float(np.min(sector_min))))
        return {
            "sector_min": sector_min,
            "front_clearance": np.array([front], dtype=np.float32),
            "corridor_width": np.array([corridor_width], dtype=np.float32),
            "collision_flag": np.array([1.0 if min_clearance < collision_threshold else 0.0], dtype=np.float32),
            "min_clearance": np.array([min_clearance], dtype=np.float32),
            "rear_clearance": np.array([float(rear)], dtype=np.float32),
        }

    def build_sensor_obstacles(
        self,
        state_est: np.ndarray,
        lidar_ranges: np.ndarray,
        lidar_angles_deg: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        obs_radius: float = 0.16,
        max_range: float = 3.0,
        min_range: float = 0.12,
        depth_max_points: int = 20,
        depth_hfov_deg: float = 86.0,
        depth_yaw_offset_deg: float = 0.0,
    ) -> np.ndarray:
        x = float(state_est[0])
        y = float(state_est[1])
        yaw = float(state_est[2])
        points_world = []

        lidar = np.asarray(lidar_ranges, dtype=np.float32).reshape(-1)
        if lidar.shape[0] > 0:
            if lidar_angles_deg is None:
                if lidar.shape[0] == 3:
                    lidar_angles = np.asarray([30.0, 0.0, -30.0], dtype=np.float32)
                else:
                    lidar_angles = np.linspace(-90.0, 90.0, num=lidar.shape[0], dtype=np.float32)
            else:
                lidar_angles = np.asarray(lidar_angles_deg, dtype=np.float32).reshape(-1)
                if lidar_angles.shape[0] != lidar.shape[0]:
                    lidar_angles = np.linspace(-90.0, 90.0, num=lidar.shape[0], dtype=np.float32)
            for r, ang_deg in zip(lidar, lidar_angles):
                rr = float(r)
                if (not np.isfinite(rr)) or rr <= min_range or rr >= max_range:
                    continue
                ang = np.deg2rad(float(ang_deg))
                p_body = np.array([rr * np.cos(ang), rr * np.sin(ang)], dtype=np.float32)
                p_world = np.array([x, y], dtype=np.float32) + self._body_to_world(p_body, yaw)
                points_world.append(p_world)

        if depth_image is not None:
            depth = np.asarray(depth_image, dtype=np.float32)
            if depth.ndim == 2 and depth.size > 0:
                h, w = depth.shape
                row0 = int(h * 0.45)
                row1 = int(h * 0.65)
                band = depth[row0:row1, :]
                if band.shape[0] > 0:
                    d_row = np.nanmedian(band, axis=0)
                    valid = np.isfinite(d_row) & (d_row > min_range) & (d_row < max_range)
                    idx = np.where(valid)[0]
                    if idx.size > 0:
                        stride = max(1, int(np.ceil(idx.size / max(1, depth_max_points))))
                        idx = idx[::stride]
                        hfov = np.deg2rad(float(depth_hfov_deg))
                        yaw_off = np.deg2rad(float(depth_yaw_offset_deg))
                        for cidx in idx:
                            rr = float(d_row[cidx])
                            if rr <= min_range or rr >= max_range:
                                continue
                            u = (float(cidx) / max(1.0, float(w - 1))) * 2.0 - 1.0
                            ang = 0.5 * hfov * u + yaw_off
                            p_body = np.array([rr * np.cos(ang), rr * np.sin(ang)], dtype=np.float32)
                            p_world = np.array([x, y], dtype=np.float32) + self._body_to_world(p_body, yaw)
                            points_world.append(p_world)

        if len(points_world) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        arr = np.asarray(points_world, dtype=np.float32)
        bins = np.round(arr / max(obs_radius, 1e-3), decimals=0).astype(np.int32)
        _, uniq_idx = np.unique(bins, axis=0, return_index=True)
        arr = arr[np.sort(uniq_idx)]
        r = np.full((arr.shape[0], 1), float(obs_radius), dtype=np.float32)
        return np.concatenate([arr, r], axis=1)

    def get_unified_observation(
        self,
        goal_xy_odom: Optional[np.ndarray] = None,
        depth_image: Optional[np.ndarray] = None,
        default_far: float = 10.0,
        collision_threshold: float = 0.30,
        depth_yaw_offset_deg: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        state = self.get_state()
        assert_state_contract(state)
        lidar_scan, lidar_angles_deg = self.get_lidar_scan(default_far=default_far)
        lidar_triplet = self._triplet_from_scan(
            ranges=lidar_scan,
            angles_deg=lidar_angles_deg,
            default_far=default_far,
        )
        touch = self.get_touch_force()
        depth_feat = self.extract_depth_features(
            depth_image=depth_image,
            lidar_ranges=lidar_scan,
            lidar_angles_deg=lidar_angles_deg,
            default_far=default_far,
            collision_threshold=collision_threshold,
            depth_yaw_offset_deg=depth_yaw_offset_deg,
        )
        if goal_xy_odom is None:
            goal_rel_body = np.zeros((2,), dtype=np.float32)
        else:
            goal_xy = np.asarray(goal_xy_odom, dtype=np.float32).reshape(2)
            delta_world = goal_xy - state[:2]
            goal_rel_body = self._world_to_body(delta_world, float(state[2]))
        obs = {
            "state_est": state.astype(np.float32),
            "lidar_triplet": lidar_triplet.astype(np.float32),
            "lidar_scan": lidar_scan.astype(np.float32),
            "lidar_angles_deg": lidar_angles_deg.astype(np.float32),
            "touch_force": np.array([float(touch)], dtype=np.float32),
            "goal_rel_body": goal_rel_body.astype(np.float32),
            "goal_dist": np.array([float(np.linalg.norm(goal_rel_body))], dtype=np.float32),
            "goal_heading_err": np.array([float(np.arctan2(goal_rel_body[1], goal_rel_body[0]))], dtype=np.float32),
            "depth_sector_min": depth_feat["sector_min"].astype(np.float32),
            "front_clearance": depth_feat["front_clearance"].astype(np.float32),
            "corridor_width": depth_feat["corridor_width"].astype(np.float32),
            "depth_collision_flag": depth_feat["collision_flag"].astype(np.float32),
            "min_clearance": depth_feat["min_clearance"].astype(np.float32),
            "rear_clearance": depth_feat["rear_clearance"].astype(np.float32),
        }
        return obs
